import os
import unittest

import torch

from startup_init import startup_init_path

startup_init_path(os.path.dirname(os.path.abspath(__file__)))

from usyd_learning.fl_algorithms.aggregation.methods._fed_aggregator_svd import (
    FedAggregator_SVD,
)
from usyd_learning.fl_algorithms.aggregation.fed_aggregator_args import (
    FedAggregatorArgs,
)
from usyd_learning.ml_algorithms.lora.lora_utils import LoRAUtils


class TestFedAggregatorSVD(unittest.TestCase):
    def setUp(self):
        args = FedAggregatorArgs({"aggregation": {"method": "svd", "device": "cpu"}})
        self.aggregator = FedAggregator_SVD(args)

    def test_delta_matrix_matches_rank_split(self):
        A0 = torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
        B0 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
        A1 = torch.tensor([[2.0, -1.0, 0.0]])
        B1 = torch.tensor([[1.0], [0.0], [0.0], [1.0]])
        W0 = B0 @ A0
        W1 = B1 @ A1
        vol0, vol1 = 10.0, 6.0
        expected = (vol0 * W0 + vol1 * W1) / (vol0 + vol1)

        out = self.aggregator.aggregate([
            {"updated_weights": {"fc.lora_A": A0, "fc.lora_B": B0}, "train_record": {"data_sample_num": vol0}},
            {"updated_weights": {"fc.lora_A": A1, "fc.lora_B": B1}, "train_record": {"data_sample_num": vol1}},
        ])
        W_rec = out["fc.lora_B"] @ out["fc.lora_A"]
        A_exp, B_exp = LoRAUtils.svd_split(expected, r=2, method="sqrt")
        truncated = B_exp @ A_exp
        self.assertTrue(torch.allclose(W_rec, truncated, atol=1e-4))

    def test_rank_mask_padding_from_peft_inputs(self):
        bp = "base_model.model"
        layer = "layer"

        def peft_state(mask):
            r = len(mask)
            A = torch.arange(1, r * 3 + 1, dtype=torch.float32).view(r, 3)
            B = torch.arange(1, 2 * r + 1, dtype=torch.float32).view(2, r)
            return {
                f"{bp}.{layer}.lora_A.default": A,
                f"{bp}.{layer}.lora_B.default": B,
                f"{bp}.{layer}.lora_E.default": torch.tensor(mask, dtype=torch.float32).view(-1, 1),
                f"{bp}.{layer}.ranknum": torch.tensor(float(sum(mask)), dtype=torch.float32),
            }

        c0 = peft_state([1.0, 1.0, 0.0, 0.0])
        c1 = peft_state([1.0, 1.0, 1.0, 0.0])

        out = self.aggregator.aggregate([
            {"updated_weights": c0, "train_record": {"data_sample_num": 1}},
            {"updated_weights": c1, "train_record": {"data_sample_num": 1}},
        ])
        self.assertEqual(tuple(out[f"{layer}.lora_A"].shape), (2, 3))
        self.assertEqual(tuple(out[f"{layer}.lora_B"].shape), (2, 2))


if __name__ == "__main__":
    unittest.main()
