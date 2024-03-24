# Copyright (c) 2022, Yamagishi Laboratory, National Institute of Informatics
# Author: Shirin Dabbaghi(sdabbag@gwdg.de)
# All rights reserved.

import argparse
import jsonlines
import numpy as np


def get_predictions(args, pred_sent_file, prob_file):
    lines_0 = list(line for line in jsonlines.open(pred_sent_file))
    print(" get_predictions: lines_0[0] = ", lines_0[0])
    lines_0 = lines_0[1:]  # remove the first line
    lines_1 = np.loadtxt(prob_file, dtype=np.float64)
    assert len(lines_0) == len(lines_1)

    labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    predictions = {}
    for line_0, line_1 in zip(lines_0, lines_1):
        assert line_1.shape == (3,)
        scores = line_1
        label_idx = np.argmax(scores)
        label = labels[label_idx]
        claim_id = int(line_0["id"])
        if args.testing:
            predictions[claim_id] = {
                "predicted_label": label,
                "predicted_evidence": line_0["predicted_evidence"],
            }
        else:
            predictions[claim_id] = {
                "predicted_label": label,
                "predicted_evidence": line_0["predicted_evidence"],
                "evidence": line_0["evidence"],
                "label": line_0["label"],
            }

    return predictions


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_sent_file", type=str, required=True)
    parser.add_argument("--prob_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--testing", action="store_true")
    return parser.parse_args()


def main():
    args = build_args()
    predictions = get_predictions(args, args.pred_sent_file, args.prob_file)
    print(f"Save to {args.out_file}")
    if args.testing:
        with jsonlines.open(args.out_file, "w") as out:
            for claim_id, line in predictions.items():
                out.write(
                    {
                        "predicted_label": line["predicted_label"],
                        "predicted_evidence": [
                            [
                                el.split("_")[0],
                                el.split("_")[1]
                                if "table_caption" not in el and "header_cell" not in el
                                else "_".join(el.split("_")[1:3]),
                                "_".join(el.split("_")[2:])
                                if "table_caption" not in el and "header_cell" not in el
                                else "_".join(el.split("_")[3:]),
                            ]
                            for el in line["predicted_evidence"]
                        ],
                    }
                )
    else:
        with jsonlines.open(args.out_file, "w") as out:
            for claim_id, line in predictions.items():
                out.write(
                    {
                        "id": claim_id,
                        "predicted_label": line["predicted_label"],
                        "predicted_evidence": line["predicted_evidence"],
                        "label": line["label"],
                        "evidence": line["evidence"],
                    }
                )


if __name__ == "__main__":
    main()
