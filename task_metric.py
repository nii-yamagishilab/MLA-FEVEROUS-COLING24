# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/7/6 13:17
# Description:
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
log = logger


def compute_metrics(preds, labels):
    log.info("compute_metrics is running ...")
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="micro"
    )
    acc = accuracy_score(labels, preds)
    class_rep = classification_report(
        labels,
        preds,
        target_names=["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"],
        output_dict=False,
    )
    conf_matrix = confusion_matrix(labels, preds)
    log.info(f"compute_metrics class_rep : {class_rep}")
    log.info(
        f"compute_metrics acc : {acc} recall : {recall} precision : {precision} f1 : {f1}"
    )
    log.info(f"compute_metrics conf_matrix : {conf_matrix}")
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "class_rep": class_rep,
        "conf_matrix": conf_matrix,
    }
