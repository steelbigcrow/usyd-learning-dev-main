import os
import unittest
from types import SimpleNamespace

import torch

from startup_init import startup_init_path

startup_init_path(os.path.dirname(os.path.abspath(__file__)))

from usyd_learning.fed_strategy.server_strategy_impl._svd_server import SvdServerStrategy
from usyd_learning.fl_algorithms.aggregation.methods._fed_aggregator_svd import (
    FedAggregator_SVD,
)
from usyd_learning.fl_algorithms.aggregation.fed_aggregator_args import (
    FedAggregatorArgs,
)


def _peft(rank_mask):
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


class FakeEvaluator:
    def __init__(self):
        self.model = SimpleNamespace(
            state_dict=lambda: {"layer.lora_A": torch.zeros(2, 3), "layer.lora_B": torch.zeros(2, 2)}
        )
        self.updated = None

    def update_model(self, sd):
        self.updated = sd

    def evaluate(self):
        return {"acc": 1.0}

    def print_results(self):  # pragma: no cover - logging only
        pass


class FakeServerNode:
    def __init__(self, updates, aggregator):
        evaluator = FakeEvaluator()
        self.node_var = SimpleNamespace(
            client_updates=updates,
            aggregation_method=aggregator,
            aggregated_weight=None,
            model_weight=None,
            model_evaluator=evaluator,
            config_dict={"trainer": {"trainer_type": "adalora", "adalora": {"lora_alpha": 16}}},
        )
        self.client_nodes = []


class TestSvdServerPipeline(unittest.TestCase):
    def test_end_to_end(self):
        sd0 = _peft([1.0, 1.0, 0.0, 0.0])
        sd1 = _peft([1.0, 1.0, 1.0, 0.0])
        updates = [
            {"updated_weights": sd0, "train_record": {"data_sample_num": 1}},
            {"updated_weights": sd1, "train_record": {"data_sample_num": 1}},
        ]
        args = FedAggregatorArgs({"aggregation": {"method": "svd", "device": "cpu"}})
        aggregator = FedAggregator_SVD(args)
        node = FakeServerNode(updates, aggregator)
        strategy = SvdServerStrategy(args, node)

        strategy.aggregation()
        self.assertIsNotNone(node.node_var.aggregated_weight)

        # apply_weight should map back to evaluator model and call compensation utility.
        with unittest.mock.patch("usyd_learning.ml_algorithms.lora.lora_utils.LoRAUtils.compensate_for_adalora_scaling") as mock_comp:
            strategy.apply_weight()
            self.assertTrue(mock_comp.called)
        self.assertIsNotNone(node.node_var.model_evaluator.updated)


if __name__ == "__main__":
    unittest.main()
