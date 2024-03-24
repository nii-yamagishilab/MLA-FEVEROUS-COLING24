# Copyright (c) 2022, Yamagishi Laboratory, National Institute of Informatics
# Author: Shirin Dabbaghi(sdabbag@gwdg.de)
# All rights reserved.

import torch
from torch.nn import CrossEntropyLoss
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from attentions import (
    MultiHeadedAttention,
)
from einops import rearrange
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    BartForSequenceClassification,
)


class VerificationModel(nn.Module):
    def __init__(self, hparams, num_labels):
        super().__init__()
        config_table = AutoConfig.from_pretrained(
            hparams.pretrained_model_name_table,
            num_labels=num_labels,
            output_hidden_states=True,
        )
        config_text = AutoConfig.from_pretrained(
            hparams.pretrained_model_name_text,
            num_labels=num_labels,
            output_hidden_states=True,
        )
        self.config = config_text
        hidden_size = self.config.hidden_size
        self.linear1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, num_labels)
        self.dropout = nn.Dropout(hparams.dropout)
        self.attn_bias_type = hparams.attn_bias_type
        self.aggregate_attn = MultiHeadedAttention(self.config, self.attn_bias_type)
        self.bert_tab = AutoModelForSequenceClassification.from_pretrained(
            hparams.pretrained_model_name_table,
            config=config_table,
            ignore_mismatched_sizes=True,
        )

        if "bart" in hparams.pretrained_model_name_text:
            self.bert_text = BartForSequenceClassification.from_pretrained(
                hparams.pretrained_model_name_text, config=config_text
            )
        else:
            self.bert_text = AutoModelForSequenceClassification.from_pretrained(
                hparams.pretrained_model_name_text, config=config_text
            )

    def forward(
        self,
        input_ids_claim=None,
        attention_mask_claim=None,
        token_type_ids_claim=None,
        input_ids_text=None,
        attention_mask_text=None,
        token_type_ids_text=None,
        input_ids_tab=None,
        attention_mask_tab=None,
        token_type_ids_tab=None,
        labels=None,
        text_model=None,
        table_model=None,
        class_weights=None,
        label_smoothing=0.0,
    ):
        input_ids_einops = rearrange(
            input_ids_tab, "batch sequences sentences -> (batch sequences) sentences"
        )
        attention_mask_einops = rearrange(
            attention_mask_tab,
            "batch sequences sentences -> (batch sequences) sentences",
        )

        if "tapas" in table_model:
            token_type_ids_einops = rearrange(
                token_type_ids_tab,
                "batch sequences sentences lists-> (batch sequences) sentences lists",
            )
        else:
            token_type_ids_einops = rearrange(
                token_type_ids_tab,
                "batch sequences sentences-> (batch sequences) sentences",
            )

        if "tapas" in table_model:
            outputs_tab = self.bert_tab(
                input_ids_einops,
                attention_mask=attention_mask_einops,
                token_type_ids=token_type_ids_einops,
            )
            output_tab = outputs_tab.hidden_states
        elif "tapex" in table_model:
            outputs_tab = self.bert_tab(
                input_ids_einops,
                attention_mask=attention_mask_einops,
            )
            output_tab = outputs_tab.encoder_hidden_states
        else:  #no token_type_ids_einops
            outputs_tab = self.bert_tab(
                input_ids_einops,
                attention_mask=attention_mask_einops,
            )
            output_tab = outputs_tab.hidden_states

        output_tab = output_tab[-1][:, 0, :]
        einops_output_table = rearrange(
            output_tab,
            "(batch sequences) hidden_size -> batch sequences hidden_size",
            batch=input_ids_tab.shape[0],
        )

        input_ids_text_einops = rearrange(
            input_ids_text, "batch sequences sentences -> (batch sequences) sentences"
        )
        attention_mask_text_einops = rearrange(
            attention_mask_text,
            "batch sequences sentences -> (batch sequences) sentences",
        )
        # if token_type_ids_text is not None:
        #    token_type_ids_text_einops = rearrange(
        #        token_type_ids_text,
        #        "batch sequences sentences -> (batch sequences) sentences",
        #    )

        outputs_text = self.bert_text(
            input_ids_text_einops,
            attention_mask=attention_mask_text_einops,
        )
        # token_type_ids=token_type_ids_text_einops
        # if token_type_ids_text is not None
        # else None,

        if "bart" in text_model:
            output_text = outputs_text.encoder_hidden_states
            # output_text= outputs_text[0]# for roberta
        else:  # for deberat and roberta
            output_text = outputs_text.hidden_states
        output_text = output_text[-1][:, 0, :]

        einops_output_text = rearrange(
            output_text,
            "(batch sequences) hidden_size -> batch sequences hidden_size",
            batch=input_ids_text.shape[0],
        )
        outputs_claim = self.bert_text(
            input_ids_claim,
            attention_mask=attention_mask_claim,
        )  # token_type_ids=token_type_ids_claim,
        if "bart" in text_model:
            outputs_claim = outputs_claim.encoder_hidden_states
        else:  # for deberat and roberta
            outputs_claim = outputs_claim.hidden_states

        outputs_claim = outputs_claim[-1][:, 0, :]
        evidence = torch.cat((einops_output_table, einops_output_text), dim=1)

        # claims: batch x hidden
        # evidence: batch x evidence x hidden
        aggregate_output = self.aggregate_attn(
            outputs_claim,
            evidence,
            evidence,
        ).squeeze(1)

        hg = self.dropout(aggregate_output)
        hg = self.relu(self.linear1(hg))
        pred_logits = self.linear2(hg)

        if class_weights is not None:
            class_weights = class_weights.to(labels.device)

        loss_fct = CrossEntropyLoss(
            weight=class_weights, label_smoothing=label_smoothing
        )
        loss = loss_fct(pred_logits, labels)
        result = SequenceClassifierOutput(loss=loss, logits=pred_logits)
        return result
