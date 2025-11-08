import os
import unittest

import torch

from startup_init import startup_init_path

startup_init_path(os.path.dirname(os.path.abspath(__file__)))

from usyd_learning.fl_algorithms.aggregation.methods._fed_aggregator_zeropad import (
    FedAggregator_ZeroPad,
)
from usyd_learning.fl_algorithms.aggregation.fed_aggregator_args import (
    FedAggregatorArgs,
)


def _make_state(rank: int, value_base: float) -> dict:
    A = torch.tensor(
        [
            [value_base + i for _ in range(3)]
            for i in range(rank)
        ],
        dtype=torch.float32,
    )
    B = torch.tensor(
        [
            [value_base + j for j in range(rank)]
            for _ in range(2)
        ],
        dtype=torch.float32,
    )
    dense = torch.full((2, 3), value_base, dtype=torch.float32)
    return {
        "layer.lora_A": A,
        "layer.lora_B": B,
        "layer.weight": dense,
    }


class TestZeroPadAggregator(unittest.TestCase):
    def setUp(self):
        args = FedAggregatorArgs({"aggregation": {"device": "cpu", "method": "zp"}})
        self.aggregator = FedAggregator_ZeroPad(args)

    def test_equal_weights_pad_logic(self):
        sd0 = _make_state(rank=2, value_base=1.0)
        sd1 = _make_state(rank=4, value_base=3.0)
        out = self.aggregator.aggregate([(sd0, 1.0), (sd1, 1.0)])
        self.assertEqual(tuple(out["layer.lora_A"].shape), (4, 3))
        self.assertEqual(tuple(out["layer.lora_B"].shape), (2, 4))
        # Rows shared by both clients should be simple averages.
        self.assertTrue(torch.allclose(out["layer.lora_A"][0], torch.tensor([2.0, 2.0, 2.0]), atol=1e-6))
        self.assertTrue(torch.allclose(out["layer.lora_A"][2], torch.tensor([2.5, 2.5, 2.5]), atol=1e-6))
        self.assertTrue(torch.allclose(out["layer.lora_A"][3], torch.tensor([3.0, 3.0, 3.0]), atol=1e-6))
        # Dense tensor aggregated with classic averaging.
        self.assertTrue(torch.allclose(out["layer.weight"], torch.tensor([[2.0, 2.0, 2.0]] * 2), atol=1e-6))

    def test_unequal_weights_and_shape_mismatch(self):
        sd0 = _make_state(rank=2, value_base=1.0)
        sd1 = _make_state(rank=4, value_base=3.0)
        # Add 3D tensor to verify generic padding path.
        sd0["conv.weight"] = torch.ones(2, 2, 2)
        sd1["conv.weight"] = torch.full((2, 2, 3), 2.0)

        vols = (1.0, 3.0)
        out = self.aggregator.aggregate([(sd0, vols[0]), (sd1, vols[1])])
        # Weighted rows follow normalized weights.
        self.assertTrue(torch.allclose(out["layer.lora_A"][0], torch.tensor([2.5, 2.5, 2.5]), atol=1e-6))
        self.assertTrue(torch.allclose(out["layer.lora_A"][3], torch.tensor([4.5, 4.5, 4.5]), atol=1e-6))
        # 3D tensor padded to common shape (2,2,3).
        self.assertEqual(tuple(out["conv.weight"].shape), (2, 2, 3))
        self.assertAlmostEqual(float(out["conv.weight"][0, 0, 2]), 1.5, places=6)


if __name__ == "__main__":
    unittest.main()
