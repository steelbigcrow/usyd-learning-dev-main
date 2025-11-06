from __future__ import annotations
import torch.nn as nn

from .. import AbstractNNModel, NNModelArgs, NNModel


class NNModel_DistilRoBERTaBaseSeqCls(NNModel):
    HF_NAME = "distilroberta-base"

    def __init__(self):
        super().__init__()
        self.model = None

    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)

        try:
            from transformers import AutoModelForSequenceClassification
        except Exception as e:
            raise ImportError("need to install transformers"
                              "please run: pip install transformers") from e

        num_classes = int(getattr(args, "num_classes", 2))
        hf_name = getattr(args, "hf_name", None) or self.HF_NAME
        local_only = bool(getattr(args, "hf_local_files_only", False))

        self.model = AutoModelForSequenceClassification.from_pretrained(
            hf_name, num_labels=num_classes, local_files_only=local_only
        )

        if getattr(args, "freeze_base", False):
            for n, p in self.model.named_parameters():
                if not n.startswith("classifier"):
                    p.requires_grad = False

        setattr(args, "hf_name", hf_name)
        return self

    def forward(self, batch):
        inputs = {k: v for k, v in batch.items()
                  if k in ("input_ids", "attention_mask", "token_type_ids")}
        out = self.model(**inputs)
        return out.logits
