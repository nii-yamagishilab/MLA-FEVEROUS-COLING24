# Copyright (c) 2022, Yamagishi Laboratory, National Institute of Informatics
# Author: Shirin Dabbaghi(sdabbag@gwdg.de)
# All rights reserved.

import argparse
import io
import jsonlines
import bisect
from collections import defaultdict
from filelock import FileLock
from tqdm import tqdm
import itertools
import pickle
import base64
import re

# import pdb #for debugging
import numpy as np


PAD_SENT = ["[PAD]", -1, "[PAD]"]  # doc_id, sent_id, sent_text


def pad_to_max(sent_list, max_evidence_per_claim):
    if len(sent_list) < max_evidence_per_claim:
        sent_list += [PAD_SENT] * (max_evidence_per_claim - len(sent_list))


def get_all_sentences(
    corpus, pred_evidence, max_evidence_per_claim_tab, max_evidence_per_claim_text
):

    pred_evidence_set_tab = get_preprocessed_evidence_tab(pred_evidence, corpus)
    pred_evidence_set_text = get_preprocessed_evidence_text(pred_evidence, corpus)
    pred_evidence_set = pred_evidence_set_text.union(pred_evidence_set_tab)

    sent_doc_list = []
    tab_list = []
    sent_list = []
    for doc_id, sent_id, doc in pred_evidence_set:
        if doc_id not in corpus:
            print("doc_id not in corpus", doc_id)
            continue
        sent_doc_list.append([doc_id, sent_id, doc])

    for doc_id, id, doc in sent_doc_list:
        if id.startswith("table_"):
            tab_list.append([doc_id, id, doc])
        else:
            sent_list.append([doc_id, id, doc])

    pad_to_max(sent_list, max_evidence_per_claim_text)
    pad_to_max(tab_list, max_evidence_per_claim_tab)

    for doc_id, id, evid_context in sent_list[:max_evidence_per_claim_text]:
        yield ["text_evid", doc_id, id, evid_context]

    for doc_id, id, evid_context in tab_list[:max_evidence_per_claim_tab]:
        yield ["tab_evid", doc_id, id, [evid_context]]


def get_train_sentences(
    corpus,
    evidence,
    pred_evidence,
    label,
    max_evidence_per_claim_tab,
    max_evidence_per_claim_text,
):
    # breakpoint()
    pred_evidence_set_tab = get_preprocessed_evidence_tab(pred_evidence, corpus)
    pred_evidence_set_text = get_preprocessed_evidence_text(pred_evidence, corpus)
    pred_evidence_set = pred_evidence_set_text.union(pred_evidence_set_tab)
    train_evidence = []

    pos_sents = defaultdict(lambda: set())  # positive sentences, to keep gold evidences
    for doc_id, sent_id, doc in pred_evidence_set:
        if doc_id not in corpus:
            print("doc_id not in corpus", doc_id)
            continue
        bisect.insort(
            train_evidence, (-1, doc_id, sent_id, doc)
        )  # assign negative 1 for gold evidence ,
        # bisect is used to insert the element in the list in sorted order,
        # that means gold evidences come at the beggining
        pos_sents[doc_id].add(sent_id)

    gold_evidence_set_tab = get_preprocessed_evidence_tab(evidence, corpus)
    gold_evidence_set_text = get_preprocessed_evidence_text(evidence, corpus)
    gold_evidence_set = gold_evidence_set_text.union(gold_evidence_set_tab)

    for doc_id, sent_id, doc in sorted(gold_evidence_set):
        if doc_id not in corpus:
            print("doc_id not in corpus", doc_id)
            continue
        if doc_id in pos_sents and sent_id in pos_sents[doc_id]:
            continue
        bisect.insort(train_evidence, (1, doc_id, sent_id, doc))

    sent_list = []
    tab_list = []
    for score, doc_id, id, evid_context in train_evidence:
        if id.startswith("table_"):
            tab_list.append([doc_id, id, evid_context])
        else:
            sent_list.append([doc_id, id, evid_context])

    pad_to_max(sent_list, max_evidence_per_claim_text)
    pad_to_max(tab_list, max_evidence_per_claim_tab)

    for doc_id, id, evid_context in sent_list[:max_evidence_per_claim_text]:
        yield ["text_evid", doc_id, id, evid_context, label]

    for doc_id, id, evid_context in tab_list[:max_evidence_per_claim_tab]:
        yield ["tab_evid", doc_id, id, [evid_context], label]


def process_data(text):
    # Remove '\t' character
    text = text.replace("\t", "")

    # Remove any other characters causing new lines
    text = re.sub(r"\n", "", text)

    return text


def build_examples(args, corpus, line):
    claim_id = line["id"]
    claim_text = line["claim"]
    evidence = line.get("evidence", [])
    pred_evidence = line["predicted_evidence"]
    examples = []

    evidence = [el["content"] for el in evidence]
    flat_evidence = list(itertools.chain.from_iterable(evidence))

    claim_text = process_data(claim_text)

    if args.training:
        label = line["label"][0]
        examples.append(["claim"] + [claim_id, claim_text] + PAD_SENT + [label])

        for evidence_sent in get_train_sentences(
            corpus,
            flat_evidence,
            pred_evidence,
            label,
            args.max_evidence_per_claim_tab,
            args.max_evidence_per_claim_text,
        ):
            if "tab_evid" == evidence_sent[0]:
                evidence_sent.remove("tab_evid")
                examples.append(["tab_evid"] + [claim_id, claim_text] + evidence_sent)
            else:
                evidence_sent.remove("text_evid")
                examples.append(["text_evid"] + [claim_id, claim_text] + evidence_sent)

    else:
        examples.append(["claim"] + [claim_id, claim_text] + PAD_SENT)

        for evidence_sent in get_all_sentences(
            corpus,
            pred_evidence,
            args.max_evidence_per_claim_tab,
            args.max_evidence_per_claim_text,
        ):
            if "tab_evid" == evidence_sent[0]:
                evidence_sent.remove("tab_evid")
                examples.append(["tab_evid"] + [claim_id, claim_text] + evidence_sent)
            else:
                evidence_sent.remove("text_evid")
                examples.append(["text_evid"] + [claim_id, claim_text] + evidence_sent)
    return examples


def get_preprocessed_evidence_tab(evidence, corpus):
    def merge_tables(table, wiki_title):
        linearized_lines = []
        col_num = len(table[0])
        # print("col_num", col_num)
        # padding_str = ' | ' * col_num
        linearized_lines.append(" | ".join(["[T] " + wiki_title] * col_num))
        linearized_lines.extend(
            [
                " | ".join(
                    [
                        "[H] " + cell["value"].replace(" | ", " ")
                        if cell["is_header"]
                        else cell["value"].replace(" | ", " ")
                        for cell in row
                    ]
                )
                for row in table
            ]
        )

        linearized_table = "\n".join(linearized_lines)
        # [row.split(' | ') for row in linearized_table.split("\n")]
        # breakpoint()
        return linearized_table

    cells = [ele for ele in evidence if "_cell_" in ele]
    cells = list(set(cells))
    grouped_cells = {}
    for cell in cells:
        wiki_title = cell.split("_")[0]
        table_id = cell.split("_cell_")[1].split("_")[0]
        table_key = (wiki_title, table_id)
        cell_id = "_".join(cell.split("_")[1:])
        if table_key in grouped_cells:
            grouped_cells[table_key].append(cell_id)
        else:
            grouped_cells[table_key] = [cell_id]

    evidence_set_tab = set()
    for table_key, cell_ids in grouped_cells.items():
        wiki_title, table_id = table_key
        corpus_tab_id = "table_" + table_id
        if wiki_title not in corpus:
            print("wiki_title not in corpus", wiki_title)
            continue
        corp_tabs = {i: s for (i, s) in corpus[wiki_title]["tables"]}
        rtr_table = corp_tabs.get(corpus_tab_id)
        deserialized_table = pickle.loads(base64.b64decode(rtr_table.encode("utf-8")))

        keep_columns = []
        keep_rows = []

        for i, row in enumerate(deserialized_table.rows):
            for j, c in enumerate(row.row):
                if c.name in cell_ids:
                    keep_rows.append(i)
                    keep_columns.append(j)
                    # break

        keep_rows = sorted(list(set(keep_rows)))
        keep_columns = sorted(list(set(keep_columns)))
        trunc_table = np.array(deserialized_table.table)[keep_rows][
            :, keep_columns
        ].tolist()
        linearized_table = merge_tables(trunc_table, wiki_title)
        evidence_set_tab.add((wiki_title, corpus_tab_id, linearized_table))

    return evidence_set_tab


def get_preprocessed_evidence_text(evidence, corpus):
    gold_evidence_set_text = set()
    for evid in evidence:
        split_string = evid.split("_")
        doc_id = split_string[0]
        if doc_id is not None and doc_id in corpus:
            if "_sentence_" in evid:
                sent_id = "_".join(split_string[1:])
                doc_sent = {i: s for (i, s) in corpus[doc_id]["lines"]}
                doc = doc_sent.get(sent_id)
                gold_evidence_set_text.add((doc_id, sent_id, doc))
    return gold_evidence_set_text


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--max_evidence_per_claim_text", type=int, default=5)
    parser.add_argument("--max_evidence_per_claim_tab", type=int, default=2)
    return parser.parse_args()


def main():
    args = build_args()
    corpus = {doc["doc_id"]: doc for doc in jsonlines.open(args.corpus)}
    # lines = [line for line in jsonlines.open(args.in_file)]
    lines = []
    with open(args.in_file, "rb") as file:
        reader = jsonlines.Reader(io.TextIOWrapper(file, encoding="utf-8"))
        for line in reader:
            lines.append(line)
    out_examples = []

    for idx, line in tqdm(enumerate(lines), total=len(lines), desc="Building examples"):
        if idx == 0:
            continue  # Skip the first line

        out_examples.extend(build_examples(args, corpus, line))

    lock_path = args.out_file + ".lock"
    with FileLock(lock_path):
        print(f"Save to {args.out_file}")
        with io.open(args.out_file, "w", encoding="utf-8", errors="ignore") as out:
            for e in out_examples:
                e = list(map(str, e))
                out.write("\t".join(e) + "\n")


if __name__ == "__main__":
    main()
