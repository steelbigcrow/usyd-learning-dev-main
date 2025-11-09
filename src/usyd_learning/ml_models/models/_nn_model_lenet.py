from __future__ import annotations

import torch.nn as nn

from .. import AbstractNNModel, NNModelArgs, NNModel


class NNModel_LeNet(NNModel):

    def __init__(self):
        super().__init__()

    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)

        in_channels = int(getattr(args, "in_channels", 1))
        num_classes = int(getattr(args, "num_classes", 10))

        self._conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=0)
        self._pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self._conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self._act = nn.ReLU(inplace=True)
        self._adapt = nn.AdaptiveAvgPool2d((4, 4))

        self._fc1 = nn.Linear(16 * 4 * 4, 120)
        self._fc2 = nn.Linear(120, 84)
        self._fc3 = nn.Linear(84, num_classes)
        self._softmax = nn.Softmax(dim=1)
        return self

    # override
    def forward(self, x):
        x = self._act(self._conv1(x))
        x = self._pool(x)
        x = self._act(self._conv2(x))
        x = self._pool(x)
        x = self._adapt(x)
        x = x.view(x.size(0), -1)
        x = self._act(self._fc1(x))
        x = self._act(self._fc2(x))
        x = self._fc3(x)
        return self._softmax(x)


