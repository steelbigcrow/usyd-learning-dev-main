from __future__ import annotations
import torch.nn as nn
from .. import AbstractNNModel, NNModelArgs, NNModel
from torchvision.models import resnet18, ResNet18_Weights


class NNModel_ResNet18(NNModel): 
    def __init__(self):
        super().__init__()
        self.model = None

    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)

        num_classes = int(getattr(args, "num_classes", 10))
        pretrained  = bool(getattr(args, "pretrained", False))
        weights = ResNet18_Weights.DEFAULT if pretrained else None

        self.model = resnet18(weights=weights)

        in_ch = self.model.fc.in_features
        self.model.fc = nn.Linear(in_ch, num_classes)

        if getattr(args, "freeze_base", False):
            for n, p in self.model.named_parameters():
                if n.startswith("fc"):
                    continue
                p.requires_grad = False

        return self 

    def forward(self, x):
        return self.model(x)
