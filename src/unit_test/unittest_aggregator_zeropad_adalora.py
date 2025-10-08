import os
import unittest
import torch

# Ensure imports work when running under different IDEs
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))


class TestZeroPadAggregatorAdalora(unittest.TestCase):
    def _make_client_sds(self):
        # Simulate AdaLoRA-style hetero ranks across clients for the same layer
        # Client 0: r = 2
        A0 = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
            ],
            dtype=torch.float32,
        )  # [2, 3]
        B0 = torch.tensor(
            [
                [10.0, 20.0],
                [10.0, 20.0],
            ],
            dtype=torch.float32,
        )  # [2, 2]
        W0 = torch.ones((2, 3), dtype=torch.float32)  # non-LoRA parameter
        sd0 = {
            "layer.lora_A": A0.clone(),
            "layer.lora_B": B0.clone(),
            "layer.weight": W0.clone(),
        }

        # Client 1: r = 4 (higher rank)
        A1 = torch.tensor(
            [
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
                [5.0, 5.0, 5.0],
                [6.0, 6.0, 6.0],
            ],
            dtype=torch.float32,
        )  # [4, 3]
        B1 = torch.tensor(
            [
                [7.0, 8.0, 9.0, 10.0],
                [7.0, 8.0, 9.0, 10.0],
            ],
            dtype=torch.float32,
        )  # [2, 4]
        W1 = 2.0 * torch.ones((2, 3), dtype=torch.float32)
        sd1 = {
            "layer.lora_A": A1.clone(),
            "layer.lora_B": B1.clone(),
            "layer.weight": W1.clone(),
        }
        return sd0, sd1

    def test_zp_variable_rank_unequal_weights_adalora(self):
        from usyd_learning.fl_algorithms.aggregation.methods._fed_aggregator_zeropad import (
            FedAggregator_ZeroPad,
        )
        from usyd_learning.fl_algorithms.aggregation.fed_aggregator_args import (
            FedAggregatorArgs,
        )

        sd0, sd1 = self._make_client_sds()

        print("\n=== [ZP] Pre-Aggregation (Client 0) ===")
        print("A0 shape:", tuple(sd0["layer.lora_A"].shape))
        print(sd0["layer.lora_A"])
        print("B0 shape:", tuple(sd0["layer.lora_B"].shape))
        print(sd0["layer.lora_B"])
        print("W0 shape:", tuple(sd0["layer.weight"].shape))
        print(sd0["layer.weight"])

        print("\n=== [ZP] Pre-Aggregation (Client 1) ===")
        print("A1 shape:", tuple(sd1["layer.lora_A"].shape))
        print(sd1["layer.lora_A"])
        print("B1 shape:", tuple(sd1["layer.lora_B"].shape))
        print(sd1["layer.lora_B"])
        print("W1 shape:", tuple(sd1["layer.weight"].shape))
        print(sd1["layer.weight"])

        # Unequal volumes to simulate AdaLoRA round (e.g., different data sizes or client weights)
        vols = (1.0, 3.0)  # normalized internally -> w0 = 0.25, w1 = 0.75

        args = FedAggregatorArgs({"aggregation": {"device": "cpu", "method": "zp"}})
        aggregator = FedAggregator_ZeroPad(args)

        aggregated = aggregator.aggregate([(sd0, vols[0]), (sd1, vols[1])])

        A_agg = aggregated["layer.lora_A"]
        B_agg = aggregated["layer.lora_B"]
        W_agg = aggregated["layer.weight"]

        print("\n=== [ZP] Post-Aggregation (Zero-Pad, unequal weights 1:3) ===")
        print("A_agg shape:", tuple(A_agg.shape))
        print(A_agg)
        print("B_agg shape:", tuple(B_agg.shape))
        print(B_agg)
        print("W_agg shape:", tuple(W_agg.shape))
        print(W_agg)

        # Shapes should be padded to r_max = 4
        self.assertEqual(tuple(A_agg.shape), (4, 3))
        self.assertEqual(tuple(B_agg.shape), (2, 4))
        self.assertEqual(tuple(W_agg.shape), (2, 3))

        # Expected values under Zero-Pad with normalized weights w0=0.25, w1=0.75:
        # Overlap rows/cols -> 0.25*A0 + 0.75*A1
        # Missing rows/cols -> 0.25*0 + 0.75*A1 = 0.75*A1 (note: ZP divides by total_weight=1)
        A_expected = torch.tensor(
            [
                [0.25 * 1.0 + 0.75 * 3.0] * 3,  # 2.5
                [0.25 * 2.0 + 0.75 * 4.0] * 3,  # 3.5
                [0.75 * 5.0] * 3,               # 3.75
                [0.75 * 6.0] * 3,               # 4.5
            ],
            dtype=torch.float32,
        )

        B_expected = torch.tensor(
            [
                [0.25 * 10.0 + 0.75 * 7.0, 0.25 * 20.0 + 0.75 * 8.0, 0.75 * 9.0, 0.75 * 10.0],
                [0.25 * 10.0 + 0.75 * 7.0, 0.25 * 20.0 + 0.75 * 8.0, 0.75 * 9.0, 0.75 * 10.0],
            ],
            dtype=torch.float32,
        )

        W_expected = 0.25 * sd0["layer.weight"] + 0.75 * sd1["layer.weight"]

        self.assertTrue(torch.allclose(A_agg.cpu(), A_expected, atol=1e-6))
        self.assertTrue(torch.allclose(B_agg.cpu(), B_expected, atol=1e-6))
        self.assertTrue(torch.allclose(W_agg.cpu(), W_expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()

