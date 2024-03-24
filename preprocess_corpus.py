# Copyright (c) 2022, Yamagishi Laboratory, National Institute of Informatics
# Author: Shirin Dabbaghi(sdabbag@gwdg.de)
# All rights reserved.

import argparse
import jsonlines
import unicodedata
from functools import partial
from multiprocessing import Pool, cpu_count
from multiprocessing.util import Finalize
from tqdm import tqdm
from feverous.database.feverous_db import FeverousDB
from feverous.utils.wiki_page import WikiPage
import itertools
import pickle
import base64


def get_documents(args, line):
    global PROCESS_DB
    sorted_p = list(
        sorted(line["predicted_pages"], reverse=True, key=lambda elem: elem[1])
    )
    # sorted_p [['Pete Varney', 3.0], ['David L. Hoof', 2.5], ['Peter Blanck', 2.5],
    # ['Peter Gabel', 1.0], ['Deerfield Academy', 0]]
    pages = [p[0] for p in sorted_p[: args.max_page]]
    # pages ['Pete Varney', 'David L. Hoof', 'Peter Blanck', 'Peter Gabel', 'Deerfield Academy']

    docs = []
    for page in pages:
        page = unicodedata.normalize("NFD", page)
        lines = PROCESS_DB.get_doc_json(page)
        if lines is None:
            print("page not found", page)
            continue
        current_page = WikiPage(page, lines)
        all_sentences = current_page.get_sentences()
        sentences = [
            [sent.get_id(), str(sent)] for sent in all_sentences[: len(all_sentences)]
        ]
        all_tables = current_page.get_tables()
        tables = [
            [table.get_id(), base64.b64encode(pickle.dumps(table)).decode("utf-8")]
            for table in all_tables[: len(all_tables)]
        ]

        docs.append([page, sentences, tables])

    return docs


def get_documents_gold(args, lines_gold):
    global PROCESS_DB
    docs = []
    evidence_gold = [el["content"] for el in lines_gold["evidence"]]
    flat_evidence = list(itertools.chain.from_iterable(evidence_gold))
    flat_titles = [el.split("_")[0] for el in flat_evidence]

    for page in flat_titles:
        page = unicodedata.normalize("NFD", page)
        lines = PROCESS_DB.get_doc_json(page)
        if lines is None:
            print("page gold not found", page)
            continue
        current_page = WikiPage(page, lines)
        all_sentences = current_page.get_sentences()
        sentences = [
            [sent.get_id(), str(sent)] for sent in all_sentences[: len(all_sentences)]
        ]
        all_tables = current_page.get_tables()
        tables = [
            [table.get_id(), base64.b64encode(pickle.dumps(table)).decode("utf-8")]
            for table in all_tables[: len(all_tables)]
        ]

        docs.append([page, sentences, tables])

    return docs


def init(db_class, db_opts):
    global PROCESS_DB
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_file", type=str, required=True)
    parser.add_argument("--in_file", type=str, required=False, default=None)
    parser.add_argument("--in_file_gold", type=str, required=False, default=None)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_page", type=int, default=5)
    return parser.parse_args()


def main():
    args = build_args()
    threads = min(args.num_workers, cpu_count())
    workers = Pool(
        threads,
        initializer=init,
        initargs=(FeverousDB, {"db_path": args.db_file}),
    )
    out_docs = []
    if args.in_file is not None:
        print("args.in_file", args.in_file)
        lines = [line for line in jsonlines.open(args.in_file)]
        _get_documents = partial(get_documents, args)
        for docs in tqdm(
            workers.imap_unordered(_get_documents, lines),
            total=len(lines),
            desc="Getting documents",
        ):
            out_docs.extend(docs)
    else:
        print("args.in_file_gold", args.in_file_gold)
        lines_gold = [line for line in jsonlines.open(args.in_file_gold)]
        _get_documents_gold = partial(get_documents_gold, args)
        for docs in tqdm(
            workers.imap_unordered(_get_documents_gold, lines_gold),
            total=len(lines_gold),
            desc="Getting documents Gold",
        ):
            out_docs.extend(docs)

    workers.close()
    workers.join()
    out_docs = {docs[0]: (docs[1], docs[2]) for docs in out_docs}

    print(f"Save to {args.out_file}")
    with jsonlines.open(args.out_file, "w") as out:
        for k, (v, t) in sorted(out_docs.items(), key=lambda x: x[0]):
            out.write({"doc_id": k, "lines": v, "tables": t})


if __name__ == "__main__":
    main()
