from __future__ import annotations
import torch.nn as nn
from .. import AbstractNNModel, NNModelArgs, NNModel
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights


class NNModel_SqueezeNet1_1(NNModel): 
    def __init__(self):
        super().__init__()
        self.model = None

    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)

        num_classes = int(getattr(args, "num_classes", 10))
        pretrained  = bool(getattr(args, "pretrained", False))
        weights = SqueezeNet1_1_Weights.DEFAULT if pretrained else None

        self.model = squeezenet1_1(weights=weights)

        # Modify the classifier for the number of classes
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        if getattr(args, "freeze_base", False):
            for n, p in self.model.named_parameters():
                if not n.startswith("classifier"):
                    p.requires_grad = False

        return self 

    def forward(self, x):
        return self.model(x).view(x.size(0), -1)

