# Copyright (c) 2022, Yamagishi Laboratory, National Institute of Informatics
# Author: Shirin Dabbaghi(sdabbag@gwdg.de)
# All rights reserved.
import ast
import numpy as np
import re
from dataclasses import dataclass

from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from transformers.data.processors.utils import DataProcessor
from typing import List, Optional, Union
import pandas as pd
import os


@dataclass
class InputExample:
    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None
    data_type: str = None
    index: Optional[int] = None


@dataclass(frozen=True)
class InputFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    data_type: Optional[str] = None
    index: Optional[int] = None


def convert_example_to_features(
    example, max_length, label_map, text_model, table_model
):
    if max_length is None:
        max_length = tokenizer_text.max_len

    def label_from_example(example: InputExample):
        if example.label is None:
            return None
        else:
            return label_map[example.label]

    label = label_from_example(example)
    eos_token_id = 2
    if example.data_type == "tab_evid":
        if example.text_b == "[PAD]":
            inputs = {
                "input_ids": [eos_token_id] + [0] * (max_length - 1),
                "attention_mask": [1] + [0] * (max_length - 1),
                "token_type_ids": [[0] * 7 for _ in range(max_length)]
                if "tapas" in table_model
                else [0] * max_length,
            }
        else:
            if max_length == 128:
                example.text_a = example.text_a[:128]
            if "tapas" in table_model or "tapex" in table_model:
                tab = read_text_as_pandas_table(example.text_b)
                inputs = tokenizer_table(
                    table=tab,
                    queries=example.text_a,
                    padding="max_length",
                    truncation=True,
                )
            else:
                table_linearized = convert_table_text_to_pandas_pasta(example.text_b)
                table_claim = example.text_a + " " + table_linearized
                inputs = tokenizer_table(
                    table_claim,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

    elif example.data_type == "text_evid":
        if example.text_b == "[PAD]":
            inputs = {
                "input_ids": [eos_token_id] + [0] * (max_length - 1),
                "attention_mask": [1] + [0] * (max_length - 1),
                "token_type_ids": [0] * max_length if "bert" in text_model else None,
            }
        else:
            if "bart" in text_model:
                claim_text = example.text_a + "  [PAD]  " + example.text_b  # for bart
            else:  # for deberta and roberta
                claim_text = example.text_a + " </s>  " + example.text_b

            inputs = tokenizer_text(
                claim_text,
                padding="max_length",
                truncation=True,
            )
    else:
        inputs = tokenizer_text(
            example.text_a,
            padding="max_length",
            truncation=True,
        )
    data_type_map = {"claim": 0, "text_evid": 1, "tab_evid": 2}
    data_type = data_type_map[example.data_type]
    return InputFeatures(
        **inputs,
        label=label,
        data_type=data_type,
        index=example.index,
    )


def convert_example_to_features_init(
    tokenizer_for_convert_text, tokenizer_for_convert_table
):
    global tokenizer_text
    global tokenizer_table
    tokenizer_text = tokenizer_for_convert_text
    tokenizer_table = tokenizer_for_convert_table


def convert_examples_to_features(
    examples,
    tokenizer_for_text,
    tokenizer_for_tab,
    max_length=None,
    task=None,
    label_list=None,
    threads=8,
    text_model="bert",
    table_model="tapas",
):
    if task is not None:
        processor = ClaimVerificationProcessor()
        if label_list is None:
            label_list = processor.get_labels()

    label_map = {label: i for i, label in enumerate(label_list)}
    print("label_map", label_map)

    if "tapas" in table_model or "tapex" in table_model:

        threads = min(threads, cpu_count())
        with Pool(
            threads,
            initializer=convert_example_to_features_init,
            initargs=(
                tokenizer_for_text,
                tokenizer_for_tab,
            ),
        ) as p:
            annotate_ = partial(
                convert_example_to_features,
                max_length=max_length,
                label_map=label_map,
                text_model=text_model,
                table_model=table_model,
            )
            features = list(
                tqdm(
                    p.imap(annotate_, examples, chunksize=32),
                    total=len(examples),
                )
            )
    else:
        # parallel processing for pasta models need to be fixed
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        convert_example_to_features_init(
            tokenizer_for_text,
            tokenizer_for_tab,
        )
        features = []
        for example in examples:
            feature = convert_example_to_features(
                example,
                max_length=max_length,
                label_map=label_map,
                text_model=text_model,
                table_model=table_model,
            )
            features.append(feature)
    return features


def compute_metrics(preds, labels):
    assert len(preds) == len(
        labels
    ), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"

    preds = np.argmax(preds, axis=1)
    return {
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="micro"),
        "precision": precision_score(labels, preds, average="micro"),
        "recall": recall_score(labels, preds, average="micro"),
    }


def remove_bracket_w_nonascii(s):
    """
    remove bracketed sub-strings with more than 2 or more than 10% non ascii charactors
    :param s:
    :return: cleaned s
    """
    p = 0
    lrb = s.find("(")
    while lrb != -1:
        rrb = s.find(")", lrb)
        if rrb == -1:
            break
        ori_r = s[lrb : rrb + 1]
        new_r = []

        for r in ori_r[1:-1].split(";"):
            innormal_chars_num = ascii(r).count("\\u") + ascii(r).count("\\x")
            if len(r) > 0 and (innormal_chars_num * 1.0 / len(r) < 0.5):
                new_r.append(r)

        new_r = ";".join(new_r)
        if new_r:
            new_r = "(" + new_r + ")"
        s = s.replace(ori_r, new_r)
        p += len(new_r)

        lrb = s.find("(", p)
    return s


def process_data(text):
    pt = re.compile(r"\[\[.*?\|(.*?)]]")
    text = re.sub(pt, r"\1", text)
    text = remove_bracket_w_nonascii(text)
    return text


def read_text_as_pandas_table(table_text: str):
    table = pd.DataFrame(
        [x.split(" | ") for x in table_text.split("\n")][:255],
        columns=[x for x in table_text.split("\n")[0].split(" | ")],
    ).fillna("")
    table = table.astype(str)

    return table


def convert_table_text_to_pandas_pasta(_table_text):
    """Runs the structured pandas table object for _table_text.
    An example _table_text can be: round#clubs remaining\nfirst round#156\n
    """
    _table_content = [_table_row for _table_row in _table_text.strip("\n").split("\n")]
    _table_str = ""
    table = []
    for row in _table_content:
        if "[T]" in row:
            parts = row.split("|")
            for i in range(len(parts)):
                if "[T]" in parts[i]:
                    parts[i] = parts[i].replace("[T]", "").strip()
            row_cleaned = " | ".join(parts)
            _table_str += "[Title]: " + row_cleaned + " "
        else:
            table.append(row)

    if len(table) < 1:
        print("table has only table name:", _table_text)
        print("table has only table name _table_str:", _table_str)
        return _table_str
    else:
        # print("table has only headers name:", table)
        headers = table[0]
        _table_str += "[Header]: " + headers + " "
        rows = table[1:]
        for row in rows:
            row_cleaned = row
            if "[H]" in row:
                parts = row.split("|")
                for i in range(len(parts)):
                    if "[H]" in parts[i]:
                        parts[i] = parts[i].replace("[H]", "").strip()
                row_cleaned = " | ".join(parts)
            _table_str += "[Row]: " + row_cleaned + " "
    return _table_str


class SentenceSelectionProcessor(DataProcessor):
    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def get_dummy_label(self):
        return "0"

    def get_length(self, file_path):
        return sum(1 for line in open(file_path, "r", encoding="utf-8-sig"))

    def get_examples(
        self,
        file_path,
        set_type,
        training=True,
        use_title=True,
    ):
        examples = []
        print("*********Reading from {}".format(file_path))

        for (i, line) in enumerate(self._read_tsv(file_path)):
            guid = f"{set_type}-{i}"
            data_type = line[0]
            index = int(line[1])
            text_a = process_data(line[2])

            if data_type == "claim":
                text_b = None
            elif data_type == "text_evid":
                title = process_data(line[3])
                sentence = process_data(line[5])
                text_b = f"{title} : {sentence}" if use_title else sentence
            else:
                title = process_data(line[3])
                tab_value = ast.literal_eval(line[5])[0]
                tab = process_data(tab_value)
                text_b = f"{title} : {tab}" if use_title else tab

            label = line[6] if training else self.get_dummy_label()
            examples.append(
                InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    label=label,
                    data_type=data_type,
                    index=index,
                )
            )
        return examples


class ClaimVerificationProcessor(SentenceSelectionProcessor):
    def get_labels(self):
        """See base class."""
        return ["S", "R", "N"]  # SUPPORTS, REFUTES, NOT ENOUGH INFO

    def get_dummy_label(self):
        return "S"
