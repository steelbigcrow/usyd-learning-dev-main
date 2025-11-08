import os
import unittest
from types import SimpleNamespace

import torch

from startup_init import startup_init_path

startup_init_path(os.path.dirname(os.path.abspath(__file__)))

from usyd_learning.fed_strategy.server_strategy_impl._zp_server import ZpServerStrategy
from usyd_learning.fl_algorithms.aggregation.methods._fed_aggregator_zeropad import (
    FedAggregator_ZeroPad,
)
from usyd_learning.fl_algorithms.aggregation.fed_aggregator_args import (
    FedAggregatorArgs,
)


def _peft_state(rank_mask):
    bp = "base_model.model"
    layer = "layer"
    r = len(rank_mask)
    A = torch.arange(1, r * 3 + 1, dtype=torch.float32).view(r, 3)
    B = torch.arange(1, 2 * r + 1, dtype=torch.float32).view(2, r)
    return {
        f"{bp}.{layer}.lora_A.default": A,
        f"{bp}.{layer}.lora_B.default": B,
        f"{bp}.{layer}.lora_E.default": torch.tensor(rank_mask, dtype=torch.float32).view(-1, 1),
        f"{bp}.{layer}.ranknum": torch.tensor(float(sum(rank_mask)), dtype=torch.float32),
    }


class FakeServerNode:
    def __init__(self, updates, aggregator):
        self.node_var = SimpleNamespace(
            client_updates=updates,
            aggregation_method=aggregator,
            aggregated_weight=None,
            model_weight=None,
        )
        self.client_nodes = []


class TestZpAggregationPipeline(unittest.TestCase):
    def test_peft_to_plain_roundtrip(self):
        sd0 = _peft_state([1.0, 1.0, 0.0, 0.0])
        sd1 = _peft_state([1.0, 1.0, 1.0, 0.0])
        updates = [
            {"updated_weights": sd0, "train_record": {"data_sample_num": 1}},
            {"updated_weights": sd1, "train_record": {"data_sample_num": 1}},
        ]
        args = FedAggregatorArgs({"aggregation": {"device": "cpu", "method": "zp"}})
        aggregator = FedAggregator_ZeroPad(args)
        node = FakeServerNode(updates, aggregator)

        strategy = ZpServerStrategy(args, node)
        strategy.aggregation()

        self.assertIsNotNone(node.node_var.aggregated_weight)
        agg = node.node_var.aggregated_weight
        self.assertIn("base_model.model.layer.lora_A.default", agg)
        self.assertTrue(torch.all(agg["base_model.model.layer.lora_A.default"][2] == 0))
        # aggregated weight should also be the broadcast weight
        self.assertIs(node.node_var.model_weight, agg)


if __name__ == "__main__":
    unittest.main()
