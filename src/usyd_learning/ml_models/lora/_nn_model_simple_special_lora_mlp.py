from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import AbstractNNModel, NNModelArgs, NNModel
from ...ml_algorithms.lora import MSLoRALinear

class NNModel_SimpleSpecialLoRAMLP(NNModel):
    """
    Simple MLP with LoRA-enabled Linear layers (using Microsoft MSLoRALinear).
    """

    def __init__(self):
        super().__init__()
        self.lora_mode = "standard" 

    # override
    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)

        scaling = getattr(args, "lora_scaling", 0.5)
        use_bias = getattr(args, "use_bias", True)

        self._flatten = nn.Flatten()
        self._fc1 = MSLoRALinear(784, 200, r=12, lora_alpha=int(12 * scaling),
                                 lora_dropout=0.0, fan_in_fan_out=False,
                                 merge_weights=False, bias=use_bias)
        self._relu1 = nn.ReLU()
        self._fc2 = MSLoRALinear(200, 200, r=12, lora_alpha=int(12 * scaling),
                                 lora_dropout=0.0, fan_in_fan_out=False,
                                 merge_weights=False, bias=use_bias)
        self._relu2 = nn.ReLU()
        self._fc3 = MSLoRALinear(200, 10, r=8, lora_alpha=int(8 * scaling),
                                 lora_dropout=0.0, fan_in_fan_out=False,
                                 merge_weights=False, bias=use_bias)

        return self  

    # override
    def forward(self, x):
        x = self._flatten(x)
        x = self._relu1(self._fc1(x))
        x = self._relu2(self._fc2(x))
        x = self._fc3(x)

        return x

    def set_lora_mode(self, mode: str):
        if mode not in ["standard", "lora_only", "lora_disabled", "scaling"]:
            raise ValueError(f"Unsupported lora_mode: {mode}")
        self.lora_mode = mode
        for layer in [self._fc1, self._fc2, self._fc3]:
            layer.lora_mode = mode
