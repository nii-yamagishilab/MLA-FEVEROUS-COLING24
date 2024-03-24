import argparse
import jsonlines
import os
from feverous_scorer import feverous_score
from task_metric import compute_metrics
import io


def main(input_path, use_gold_verdict=False):

    predictions = []

    preds_epoch = []
    golds_epoch = []

    label2idx = {
        "SUPPORTS": 0,
        "REFUTES": 1,
        "NOT ENOUGH INFO": 2,
    }

    with jsonlines.open(os.path.join(input_path)) as f:
        print("Reading predictions from {}".format(input_path))
        for i, line in enumerate(f.iter()):
            # if i == 2:
            #     break
            # print("line: in eval feverous ", line)
            if use_gold_verdict:
                line["predicted_label"] = line["label"]
                # if not (line["label"] == "NOT ENOUGH INFO"):
                #     continue

            preds_epoch.append(label2idx[line["predicted_label"]])
            golds_epoch.append(label2idx[line["label"]])

            line["evidence"] = [el["content"] for el in line["evidence"]]
            for j in range(len(line["evidence"])):
                line["evidence"][j] = [
                    [
                        el.split("_")[0],
                        el.split("_")[1]
                        if "table_caption" not in el and "header_cell" not in el
                        else "_".join(el.split("_")[1:3]),
                        "_".join(el.split("_")[2:])
                        if "table_caption" not in el and "header_cell" not in el
                        else "_".join(el.split("_")[3:]),
                    ]
                    for el in line["evidence"][j]
                ]
            try:
                line["predicted_evidence"] = [
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
                ]
            except Exception as e:
                print("error", e)
            predictions.append(line)

    scores = compute_metrics(preds_epoch, golds_epoch)

    print("Feverous scores...")
    strict_score, label_accuracy, precision, recall, f1 = feverous_score(predictions)
    print("feverous score:", strict_score)  # 0.5
    print("label accuracy:", label_accuracy)  # 1.0
    print(
        "evidence precision:", precision
    )  # 0.833 (first example scores 1, second example scores 2/3)
    print(
        "evidence recall:", recall
    )  # 0.5 (first example scores 0, second example scores 1)
    print("evidence f1:", f1)  # 0.625

    res = "\n".join(
        [
            "Confusion Matrix:",
            "",
            f"{scores['conf_matrix']}",
            "",
            f"Evidence precision: {precision:.2f}",
            f"Evidence recall:    {recall:.2f}",
            f"Evidence F1:        {f1:.2f}",
            f"Label accuracy:     {label_accuracy:.2f}",
            f"feverous score:        {strict_score:.2f}",
        ]
    )
    print(res)
    print(f"Save to {args.out_file}")
    with io.open(args.out_file, "w", encoding="utf-8", errors="ignore") as out:
        out.write(res + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    args = parser.parse_args()
    main(args.in_file)
