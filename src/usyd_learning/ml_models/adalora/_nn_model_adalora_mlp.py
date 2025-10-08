from __future__ import annotations

import torch.nn as nn

from .. import AbstractNNModel, NNModelArgs, NNModel


class NNModel_AdaLoRAMLP(NNModel):
    """
    AdaLoRA-ready MLP with standard Linear layers.

    Shape mirrors the LoRA MLP (without LoRA wrappers):
      - Flatten -> Linear(784->200) -> ReLU -> Linear(200->200) -> ReLU -> Linear(200->10)

    Returns raw logits (no Softmax) to pair correctly with CrossEntropyLoss.
    AdaLoRA adapters are injected at trainer build time onto the Linear layers.
    """

    def __init__(self):
        super().__init__()

    # override
    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)

        self._flatten = nn.Flatten()
        self._relu1 = nn.ReLU()
        self._relu2 = nn.ReLU()

        self._fc1 = nn.Linear(784, 200, bias=True)
        self._fc2 = nn.Linear(200, 200, bias=True)
        self._fc3 = nn.Linear(200, 10, bias=True)

        return self

    # override
    def forward(self, x):
        x = self._flatten(x)
        x = self._relu1(self._fc1(x))
        x = self._relu2(self._fc2(x))
        x = self._fc3(x)
        return x
