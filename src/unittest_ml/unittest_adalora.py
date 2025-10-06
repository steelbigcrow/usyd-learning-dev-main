import unittest
from collections import OrderedDict

import torch
import torch.nn as nn

# Init startup path
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))

from usyd_learning.ml_algorithms import AdaLoRAOptions, wrap_with_adalora
from usyd_learning.fl_algorithms import FedAggregatorFactory, FedAggregatorArgs


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 2)
        self.linear2 = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


def make_peft_model():
    base = SimpleModel()
    opts = AdaLoRAOptions(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["linear1", "linear2"]) 
    peft_model = wrap_with_adalora(base, opts)
    return peft_model


class TestAdaLoRAIntegration(unittest.TestCase):
    def test_trainable_params(self):
        model = make_peft_model()
        # Only LoRA params should be trainable
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(any("lora_" in n for n in trainable))
        self.assertTrue(all(("lora_" in n) or ("ranknum" in n) for n in trainable))

        # One step train verifies no runtime error
        x = torch.randn(8, 4)
        y = torch.randn(8, 1)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        opt.zero_grad()
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        opt.step()

    def test_fedavg_aggregate_peft_state(self):
        model = make_peft_model()
        sd1 = OrderedDict({k: v.clone().detach() for k, v in model.state_dict().items()})

        # Create a second client state: add +1.0 to all lora_* params to simulate different update
        sd2 = OrderedDict()
        for k, v in sd1.items():
            if "lora_" in k:
                sd2[k] = v + 1.0
            else:
                sd2[k] = v.clone()

        clients = [
            {"updated_weights": sd1, "train_record": {"data_sample_num": 1}},
            {"updated_weights": sd2, "train_record": {"data_sample_num": 1}},
        ]

        agg = FedAggregatorFactory.create_aggregator(FedAggregatorArgs())
        out = agg.aggregate(clients)

        # Check lora params averaged to sd1 + 0.5
        for k in sd1.keys():
            if "lora_" in k:
                expected = sd1[k] + 0.5
                self.assertTrue(torch.allclose(out[k], expected, atol=1e-6), f"Mismatch at {k}")


if __name__ == "__main__":
    unittest.main()
