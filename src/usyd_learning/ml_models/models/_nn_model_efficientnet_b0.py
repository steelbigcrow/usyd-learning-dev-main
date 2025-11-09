from __future__ import annotations
import torch.nn as nn
from .. import AbstractNNModel, NNModelArgs, NNModel
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class NNModel_EfficientNetB0(NNModel): 
    def __init__(self):
        super().__init__()
        self.model = None

    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)

        num_classes = int(getattr(args, "num_classes", 10))
        pretrained  = bool(getattr(args, "pretrained", False))
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None

        self.model = efficientnet_b0(weights=weights)

        in_ch = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_ch, num_classes)

        if getattr(args, "freeze_base", False):
            for n, p in self.model.named_parameters():
                if not n.startswith("classifier"):
                    p.requires_grad = False

        return self 

    def forward(self, x):
        return self.model(x)
