# Copyright (c) 2022, Yamagishi Laboratory, National Institute of Informatics
# Author: Shirin Dabbaghi(sdabbag@gwdg.de)
# All rights reserved.

import argparse
import torch
import numpy as np
import pytorch_lightning as pl
from argparse import Namespace
from datetime import datetime
from filelock import FileLock
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_info
from lightning_base import BaseTransformer, generic_train
from modeling_verification import VerificationModel
from processors import (
    ClaimVerificationProcessor,
    compute_metrics,
    convert_examples_to_features,
)
from transformers import AutoTokenizer, AutoConfig, DebertaV2Tokenizer
from itertools import chain


from torchmetrics.classification import MulticlassAccuracy

import wandb
from pytorch_lightning.loggers import WandbLogger

# wandb.login()


class FactCheckerTransformer(BaseTransformer):
    def __init__(self, hparams, **kwargs):
        if type(hparams) == dict:
            hparams = Namespace(**hparams)
        num_labels = hparams.num_labels
        rank_zero_info(f"num_labels = {num_labels}")

        model = VerificationModel(hparams, num_labels)

        print("model config", model.config)

        super().__init__(
            hparams,
            num_labels=num_labels,
            model=model,
            config=model.config,
        )

        if "pasta" in hparams.pretrained_model_name_table:
            self.tokenizer_table = DebertaV2Tokenizer.from_pretrained(
                self.hparams.pretrained_model_name_table,
                model_max_length=hparams.max_seq_length,
            )
        else:
            self.tokenizer_table = AutoTokenizer.from_pretrained(
                self.hparams.pretrained_model_name_table,
                model_max_length=hparams.max_seq_length,
            )

        self.tokenizer_text = AutoTokenizer.from_pretrained(
            self.hparams.pretrained_model_name_text,
            model_max_length=hparams.max_seq_length,
        )

        self.config_text = AutoConfig.from_pretrained(
            hparams.pretrained_model_name_text,
            **({"num_labels": num_labels} if num_labels is not None else {}),
        )
        self.config_table = AutoConfig.from_pretrained(
            hparams.pretrained_model_name_table,
            **({"num_labels": num_labels} if num_labels is not None else {}),
        )
        self.train_acc = MulticlassAccuracy(num_classes=num_labels, average="micro")
        self.val_acc = MulticlassAccuracy(
            num_classes=self.hparams.num_labels, average="micro"
        )

    def forward(self, **inputs):
        return self.model(**inputs)

    def create_features(self, set_type, file_path):
        rank_zero_info(f"Creating features from '{file_path}'")
        hparams = self.hparams
        processor = ClaimVerificationProcessor()
        examples = processor.get_examples(
            file_path,
            set_type,
            self.training,
            hparams.use_title,
        )
        num_examples = processor.get_length(file_path)
        print(f"num_examples = {num_examples}")
        claim_evidence_num = (
            hparams.max_evidence_per_claim_text
            + hparams.max_evidence_per_claim_table
            + 1
        )  # +1 for claim
        print(f"claim_evidence_num = {claim_evidence_num}")
        assert num_examples % claim_evidence_num == 0
        num_samples = num_examples // claim_evidence_num
        print(f"num_samples = {num_samples}")
        num_examples_table = num_samples * hparams.max_evidence_per_claim_table
        num_examples_text = num_samples * hparams.max_evidence_per_claim_text

        features = convert_examples_to_features(
            examples,
            self.tokenizer_text,
            self.tokenizer_table,
            max_length=hparams.max_seq_length,
            task=hparams.task,
            threads=hparams.num_workers,
            text_model=hparams.pretrained_model_name_text,
            table_model=hparams.pretrained_model_name_table,
        )

        def empty_tensor_1():
            return torch.zeros(num_examples, dtype=torch.long)

        def empty_tensor_2_text():
            return torch.zeros(
                (num_examples_text, hparams.max_seq_length), dtype=torch.long
            )

        def empty_tensor_2_claim():
            return torch.zeros((num_samples, hparams.max_seq_length), dtype=torch.long)

        def empty_tensor_2_table():
            return torch.zeros(
                (num_examples_table, hparams.max_seq_length), dtype=torch.long
            )

        def empty_tensor_3_table():
            return torch.zeros(
                (num_examples_table, hparams.max_seq_length, 7), dtype=torch.long
            )

        input_ids_text = empty_tensor_2_text()
        attention_mask_text = empty_tensor_2_text()
        token_type_ids_text = empty_tensor_2_text()
        input_ids_claim = empty_tensor_2_claim()
        attention_mask_cliam = empty_tensor_2_claim()
        token_type_ids_claim = empty_tensor_2_claim()
        input_ids_tab = empty_tensor_2_table()
        attention_mask_tab = empty_tensor_2_table()
        token_type_ids_tab = (
            empty_tensor_3_table()
            if "tapas" in hparams.pretrained_model_name_table
            else empty_tensor_2_table()
        )
        print("token_type_ids_tab shape", token_type_ids_tab.shape)
        labels = empty_tensor_1()
        indices = empty_tensor_1()

        counter_text = 0
        counter_table = 0
        counter_claim = 0

        for i, feature in enumerate(features):

            if feature.data_type == 2:  # "tab_evid":
                input_ids_tab[counter_table] = torch.tensor(feature.input_ids)
                attention_mask_tab[counter_table] = torch.tensor(feature.attention_mask)
                if feature.token_type_ids is not None:
                    token_type_ids_tab[counter_table] = torch.tensor(
                        feature.token_type_ids
                    )
                counter_table += 1
            elif feature.data_type == 1:  # text_evid
                input_ids_text[counter_text] = torch.tensor(feature.input_ids)
                attention_mask_text[counter_text] = torch.tensor(feature.attention_mask)
                if feature.token_type_ids is not None:
                    token_type_ids_text[counter_text] = torch.tensor(
                        feature.token_type_ids
                    )
                counter_text += 1
            else:  # claim
                input_ids_claim[counter_claim] = torch.tensor(feature.input_ids)
                attention_mask_cliam[counter_claim] = torch.tensor(
                    feature.attention_mask
                )
                if feature.token_type_ids is not None:
                    token_type_ids_claim[counter_claim] = torch.tensor(
                        feature.token_type_ids
                    )
                counter_claim += 1
            labels[i] = torch.tensor(feature.label)
            indices[i] = torch.tensor(feature.index)
        feature_list = [
            input_ids_claim,
            attention_mask_cliam,
            token_type_ids_claim,
            input_ids_text,
            attention_mask_text,
            token_type_ids_text,
            input_ids_tab,
            attention_mask_tab,
            token_type_ids_tab,
            indices,
            labels,
        ]
        feature_list = reshape_features(
            feature_list,
            self.hparams.max_evidence_per_claim_text,
            self.hparams.max_evidence_per_claim_table,
            self.hparams.max_seq_length,
            table_name=hparams.pretrained_model_name_table,
        )
        return feature_list

    def cached_feature_file(self, mode):
        task = Path("mla-2_" + self.hparams.task)
        dirname = Path(self.hparams.data_dir).parts[-1]
        feat_dirpath = Path(self.hparams.cache_dir) / task / dirname
        feat_dirpath.mkdir(parents=True, exist_ok=True)
        pt = self.hparams.pretrained_model_name.replace("/", "__")
        return feat_dirpath / f"cached_{mode}_{pt}_{self.hparams.max_seq_length}"

    def prepare_data(self):
        if self.training:
            for set_type in ["train", "dev", "test"]:
                cached_feature_file = self.cached_feature_file(set_type)
                lock_path = cached_feature_file.with_suffix(".lock")
                with FileLock(lock_path):
                    if not cached_feature_file.exists() or self.hparams.overwrite_cache:
                        file_path = Path(self.hparams.data_dir) / f"{set_type}.tsv"
                        if not file_path.exists():
                            continue
                        feature_list = self.create_features(set_type, file_path)
                        rank_zero_info(f"Saving features to '{cached_feature_file}'")
                        torch.save(feature_list, cached_feature_file)

    def init_parameters_org(self):
        base_name_text = self.config_text.model_type
        base_name_table = self.config_table.model_type
        no_init_text = [base_name_text] + self.hparams.no_init_text
        no_init_table = [base_name_table] + self.hparams.no_init_table
        no_init_combined = list(chain(no_init_text, no_init_table))
        print("no_init_combined", no_init_combined)
        for n, p in self.model.named_parameters():
            if not any(ni in n for ni in no_init_combined):
                rank_zero_info(f"Initialize '{n}'")
                if "bias" not in n:
                    p.data.normal_(mean=0.0, std=self.config.initializer_range)
                else:
                    p.data.zero_()

    def init_parameters(self):
        base_name_text = self.config_text.model_type
        base_name_table = self.config_table.model_type
        no_init_text = [base_name_text] + self.hparams.no_init_text
        no_init_table = [base_name_table] + self.hparams.no_init_table
        no_init_combined = list(chain(no_init_text, no_init_table))
        print("NEW no_init_combined", no_init_combined)
        for idx, (name, param) in enumerate(self.named_parameters()):
            if not any(ni in name for ni in no_init_combined):
                if ("weight" in name) and param.requires_grad is True:
                    print("init", end=" ")
                    gain = nn.init.calculate_gain("relu")
                    nn.init.xavier_normal_(param.data, gain=gain)
                else:
                    print("not init", end=" ")
                    nn.init.constant_(param.data, 0)
                print(name, param.shape)

    def get_dataloader(self, mode, batch_size):
        cached_feature_file = self.cached_feature_file(mode)
        if not cached_feature_file.exists():
            return None

        rank_zero_info(f"Loading features from '{cached_feature_file}'")
        feature_list = torch.load(cached_feature_file)
        if self.hparams.class_weighting and mode == "train":
            labels = feature_list[4]
            assert labels.dim() == 1
            classes, samples_per_class = torch.unique(labels, return_counts=True)
            assert len(classes) == self.model.num_labels
            weights = len(labels) / (len(classes) * samples_per_class.float())
            self.class_weights = weights / weights.sum()
            rank_zero_info(f"Class weights: {self.class_weights}")
        return DataLoader(
            TensorDataset(*feature_list),
            batch_size=batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True if mode == "train" and self.training else False,
        )

    def build_inputs(self, batch):
        inputs = {
            "input_ids_claim": batch[0],
            "attention_mask_claim": batch[1],
            "token_type_ids_claim": batch[2],
            "input_ids_text": batch[3],
            "attention_mask_text": batch[4],
            "token_type_ids_text": batch[5],
            "input_ids_tab": batch[6],
            "attention_mask_tab": batch[7],
            "token_type_ids_tab": batch[8],
            "labels": batch[10],
            "text_model": self.hparams.pretrained_model_name_text,
            "table_model": self.hparams.pretrained_model_name_table,
        }

        if self.training:
            if self.hparams.label_smoothing > 0.0:
                inputs["label_smoothing"] = self.hparams.label_smoothing
            if hasattr(self, "class_weights"):
                inputs["class_weights"] = self.class_weights
        return inputs

    def training_step(self, batch, batch_idx):
        inputs = self.build_inputs(batch)
        ys = inputs["labels"]
        outputs = self(**inputs)
        loss = outputs[0]
        logits = outputs[1]
        preds = torch.softmax(logits, dim=-1)
        self.train_acc(preds, ys)
        self.log("train/loss", loss, on_epoch=True)
        self.log("train/acc", self.train_acc, on_epoch=True)
        self.log_dict({"train_loss": loss, "lr": self.lr_scheduler.get_last_lr()[-1]})
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = self.build_inputs(batch)
        outputs = self(**inputs)
        loss, logits = outputs[:2]
        preds = torch.softmax(logits, dim=-1)
        ys = inputs["labels"]
        self.val_acc(preds, ys)
        self.log("val/loss", loss, on_step=True, on_epoch=True)
        self.log("val/acc", self.val_acc, on_step=True, on_epoch=True)
        return {
            "loss": loss.detach().cpu(),
            "preds": preds.detach().cpu().numpy(),
            "labels": inputs["labels"].detach().cpu().numpy(),
        }

    def validation_epoch_end(self, outputs):
        avg_loss = (
            torch.stack([x["loss"] for x in outputs]).mean().detach().cpu().item()
        )
        labels = np.concatenate([x["labels"] for x in outputs], axis=0)
        preds = np.concatenate([x["preds"] for x in outputs], axis=0)
        results = {
            **{"loss": avg_loss},
            **compute_metrics(preds, labels),
        }
        log_dict = {f"val_{k}": torch.tensor(v) for k, v in results.items()}
        self.log_dict(log_dict)

    def predict_step(self, batch, batch_idx, **kwargs):
        inputs = self.build_inputs(batch)
        outputs = self(**inputs)
        _, logits = outputs[:2]
        probs = torch.softmax(logits, dim=-1)
        return probs.detach().cpu().numpy()

    @staticmethod
    def add_model_specific_args(parser):
        BaseTransformer.add_model_specific_args(parser)
        parser.add_argument("--task", type=str, required=True)
        parser.add_argument("--cache_dir", type=str, default="/tmp")
        parser.add_argument("--overwrite_cache", action="store_true")
        parser.add_argument("--save_all_checkpoints", action="store_true")
        parser.add_argument("--max_seq_length", type=int, default=512)
        parser.add_argument("--num_labels", type=int, default=3)
        parser.add_argument("--num_evidence", type=int, default=7)
        parser.add_argument("--use_title", action="store_true")
        parser.add_argument("--dropout", type=int, default=0.5)
        parser.add_argument(
            "--attn_bias_type",
            default="none",
            choices=["none", "key_only", "value_only", "both", "dot"],
        )
        parser.add_argument("--no_init_text", nargs="+", default=[])
        parser.add_argument("--no_init_table", nargs="+", default=[])
        parser.add_argument("--freeze_params", nargs="+", default=[])
        parser.add_argument("--class_weighting", action="store_true")
        parser.add_argument("--do_eval", action="store_true")
        parser.add_argument("--label_smoothing", type=float, default=0.0)
        parser.add_argument("--max_evidence_per_claim_table", type=int, default=2)
        parser.add_argument("--max_evidence_per_claim_text", type=int, default=5)
        parser.add_argument("--use_wandb", action="store_true")
        return parser


def reshape_features(
    feature_list,
    num_evidence_text=5,
    num_evidence_table=2,
    max_seq_length=128,
    table_name="tapas",
):
    print("reshape feature_list", len(feature_list))
    assert len(feature_list) >= 11
    assert len(feature_list[3]) % (num_evidence_text) == 0
    num_evidence_plus = num_evidence_text + num_evidence_table + 1
    num_examples_text = len(feature_list[3]) // (num_evidence_text)
    num_examples_table = len(feature_list[6]) // (num_evidence_table)
    assert num_examples_text == num_examples_table

    # input_ids, attention_mask, token_type_ids
    for i in range(0, 3):
        feature_list[i] = feature_list[i].view(-1, max_seq_length)
        assert feature_list[i].size(0) == num_examples_text

    for i in range(3, 6):
        feature_list[i] = feature_list[i].view(-1, num_evidence_text, max_seq_length)
        assert feature_list[i].size(0) == num_examples_text

    for i in range(6, 8):
        feature_list[i] = feature_list[i].view(-1, num_evidence_table, max_seq_length)
        assert feature_list[i].size(0) == num_examples_table

    if "tapas" in table_name:
        feature_list[8] = feature_list[8].view(
            -1, num_evidence_table, max_seq_length, 7
        )  # token_type_ids_tab
    else:
        feature_list[8] = feature_list[8].view(-1, num_evidence_table, max_seq_length)

    # incdices, labels
    for i in range(9, 11):
        feature_list[i] = torch.unique(
            feature_list[i].view(-1, num_evidence_plus), dim=1
        )
        assert (
            feature_list[i].size(0) == num_examples_text
            and feature_list[i].size(1) == 1
        )
        feature_list[i] = feature_list[i].view(-1)
    return feature_list


def build_args():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = FactCheckerTransformer.add_model_specific_args(parser)
    return parser.parse_args()


def main():
    t_start = datetime.now()

    args = build_args()

    if args.seed > 0:
        pl.seed_everything(args.seed)

    model = FactCheckerTransformer(args)

    ckpt_dirpath = Path(args.default_root_dir) / "checkpoints"
    ckpt_dirpath.mkdir(parents=True, exist_ok=True)

    monitor, mode, ckpt_filename = None, "min", "{epoch}"
    if args.do_eval:
        monitor, mode = "val_acc", "max"
        ckpt_filename = "{epoch}-{" + monitor + ":.4f}"

    print("monitor", monitor)
    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            dirpath=ckpt_dirpath,
            filename=ckpt_filename,
            monitor=monitor,
            mode=mode,
            save_top_k=5,
            # every_n_train_steps=10000,
            # save_top_k=-1 if args.save_all_checkpoints else 1,
        )
    )

    if monitor is not None:
        callbacks.append(
            EarlyStopping(monitor=monitor, mode=mode, patience=args.patience)
        )

    wandb_logger = None
    if args.use_wandb:
        wandb.login()
        wandb_logger = WandbLogger(project="GWDG-NHR", log_model="all")

    trainer = generic_train(model, args, callbacks, wandb_logger)

    if args.do_predict:
        trainer.test()

    t_delta = datetime.now() - t_start
    rank_zero_info(f"\nTraining took '{t_delta}'")


if __name__ == "__main__":
    main()
