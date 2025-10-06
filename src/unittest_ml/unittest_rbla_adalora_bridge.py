import unittest
from collections import OrderedDict

import torch

# Init startup path
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))

from usyd_learning.fl_algorithms import FedAggregatorFactory, FedAggregatorArgs
from usyd_learning.fl_algorithms.aggregation.methods._fed_aggregator_rbla import FedAggregator_RBLA
from usyd_learning.ml_algorithms.adalora.adalora_rbla_bridge import (
    peft_to_plain_lora_shrunk,
    plain_lora_to_peft,
)


def make_fake_peft_sd(prefix: str, r: int, in_dim: int, out_dim: int, base_prefix: str = "base_model.model"):
    sd = OrderedDict()
    # PEFT-style keys with .default
    kA = f"{base_prefix}.{prefix}.lora_A.default"
    kB = f"{base_prefix}.{prefix}.lora_B.default"
    kW = f"{base_prefix}.{prefix}.base_layer.weight"
    kr = f"{base_prefix}.{prefix}.ranknum"
    sd[kA] = torch.randn(r, in_dim)
    sd[kB] = torch.randn(out_dim, r)
    sd[kW] = torch.randn(out_dim, in_dim)
    # Rank hint vector: 1/0 per channel
    sd[kr] = torch.tensor([1.0] * r)
    return sd


class TestAdaLoRARBLABridge(unittest.TestCase):
    def test_pipeline_aggregate_and_broadcast(self):
        # Two clients share base r=4, but rank hints choose rr=2 and rr=3 respectively
        in_dim, out_dim, r = 10, 6, 4
        sd1 = make_fake_peft_sd("layer1", r=r, in_dim=in_dim, out_dim=out_dim)
        sd2 = make_fake_peft_sd("layer1", r=r, in_dim=in_dim, out_dim=out_dim)
        # set client-specific rank hints
        sd1["base_model.model.layer1.ranknum"] = torch.tensor([1.0, 1.0, 0.0, 0.0])  # rr=2
        sd2["base_model.model.layer1.ranknum"] = torch.tensor([1.0, 1.0, 1.0, 0.0])  # rr=3

        clients = [
            {"updated_weights": sd1, "train_record": {"data_sample_num": 1}},
            {"updated_weights": sd2, "train_record": {"data_sample_num": 1}},
        ]

        # Preprocess to plain lora with shrink
        pre = []
        for c in clients:
            plain = peft_to_plain_lora_shrunk(c["updated_weights"])  # only lora_A/B
            pre.append({"updated_weights": plain, "train_record": c["train_record"]})

        # Aggregate with RBLA
        agg = FedAggregatorFactory.create_aggregator(FedAggregatorArgs({"aggregation": {"method": "rbla"}}))
        out_plain = agg.aggregate(pre)

        # out_plain must contain the layer keys and have max rr (3) in shapes
        self.assertIn("layer1.lora_A", out_plain)
        self.assertIn("layer1.lora_B", out_plain)
        self.assertEqual(out_plain["layer1.lora_A"].shape[0], 3)
        self.assertEqual(out_plain["layer1.lora_B"].shape[1], 3)

        # Map back to PEFT-style using client 1 template
        global_sd = plain_lora_to_peft(out_plain, sd1)

        # Simulate broadcast to client 1 local sd
        new_local1 = FedAggregator_RBLA.broadcast_lora_state_dict(global_sd, sd1)
        self.assertIn("base_model.model.layer1.lora_A.default", new_local1)
        self.assertEqual(new_local1["base_model.model.layer1.lora_A.default"].shape, sd1["base_model.model.layer1.lora_A.default"].shape)
        self.assertIn("base_model.model.layer1.lora_B.default", new_local1)
        self.assertEqual(new_local1["base_model.model.layer1.lora_B.default"].shape, sd1["base_model.model.layer1.lora_B.default"].shape)

        # Simulate broadcast to client 2 local sd
        new_local2 = FedAggregator_RBLA.broadcast_lora_state_dict(global_sd, sd2)
        self.assertEqual(new_local2["base_model.model.layer1.lora_A.default"].shape, sd2["base_model.model.layer1.lora_A.default"].shape)
        self.assertEqual(new_local2["base_model.model.layer1.lora_B.default"].shape, sd2["base_model.model.layer1.lora_B.default"].shape)


if __name__ == "__main__":
    unittest.main()

