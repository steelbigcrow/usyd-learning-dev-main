import os
import unittest
import torch

# Ensure imports work when running under different IDEs
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))


class TestZeroPadAggregator(unittest.TestCase):
    def test_zp_variable_rank_shapes_and_values(self):
        from usyd_learning.fl_algorithms.aggregation.methods._fed_aggregator_zeropad import (
            FedAggregator_ZeroPad,
        )
        from usyd_learning.fl_algorithms.aggregation.fed_aggregator_args import (
            FedAggregatorArgs,
        )

        # Two clients with different LoRA ranks for the same layer
        # Client 0: r = 2
        A0 = torch.tensor(
            [
                [1.0, 1.0, 1.0],  # row 0
                [2.0, 2.0, 2.0],  # row 1
            ],
            dtype=torch.float32,
        )  # shape [2, 3]
        B0 = torch.tensor(
            [
                [10.0, 20.0],  # col 0..1 across out=2 rows
                [10.0, 20.0],
            ],
            dtype=torch.float32,
        )  # shape [2, 2]
        W0 = torch.ones((2, 3), dtype=torch.float32)  # a non-LoRA parameter

        sd0 = {
            "layer.lora_A": A0.clone(),  # [r0=2, in=3]
            "layer.lora_B": B0.clone(),  # [out=2, r0=2]
            "layer.weight": W0.clone(),  # [out=2, in=3]
        }

        # Client 1: r = 4
        A1 = torch.tensor(
            [
                [3.0, 3.0, 3.0],  # row 0
                [4.0, 4.0, 4.0],  # row 1
                [5.0, 5.0, 5.0],  # row 2
                [6.0, 6.0, 6.0],  # row 3
            ],
            dtype=torch.float32,
        )  # shape [4, 3]
        B1 = torch.tensor(
            [
                [7.0, 8.0, 9.0, 10.0],
                [7.0, 8.0, 9.0, 10.0],
            ],
            dtype=torch.float32,
        )  # shape [2, 4]
        W1 = 2.0 * torch.ones((2, 3), dtype=torch.float32)

        sd1 = {
            "layer.lora_A": A1.clone(),  # [r1=4, in=3]
            "layer.lora_B": B1.clone(),  # [out=2, r1=4]
            "layer.weight": W1.clone(),  # [out=2, in=3]
        }

        # Print pre-aggregation tensors for inspection
        print("\n=== Pre-Aggregation (Client 0) ===")
        print("A0 shape:", tuple(sd0["layer.lora_A"].shape))
        print(sd0["layer.lora_A"])
        print("B0 shape:", tuple(sd0["layer.lora_B"].shape))
        print(sd0["layer.lora_B"])
        print("W0 shape:", tuple(sd0["layer.weight"].shape))
        print(sd0["layer.weight"])

        print("\n=== Pre-Aggregation (Client 1) ===")
        print("A1 shape:", tuple(sd1["layer.lora_A"].shape))
        print(sd1["layer.lora_A"])
        print("B1 shape:", tuple(sd1["layer.lora_B"].shape))
        print(sd1["layer.lora_B"])
        print("W1 shape:", tuple(sd1["layer.weight"].shape))
        print(sd1["layer.weight"])

        # Build aggregator (Zero-Pad mode)
        args = FedAggregatorArgs({"aggregation": {"device": "cpu", "method": "zp"}})
        aggregator = FedAggregator_ZeroPad(args)

        # Two clients with equal volumes (weights)
        aggregated = aggregator.aggregate([(sd0, 1.0), (sd1, 1.0)])

        # Extract aggregated tensors
        A_agg = aggregated["layer.lora_A"]
        B_agg = aggregated["layer.lora_B"]
        W_agg = aggregated["layer.weight"]

        # Print post-aggregation tensors for inspection
        print("\n=== Post-Aggregation (Zero-Pad) ===")
        print("A_agg shape:", tuple(A_agg.shape))
        print(A_agg)
        print("B_agg shape:", tuple(B_agg.shape))
        print(B_agg)
        print("W_agg shape:", tuple(W_agg.shape))
        print(W_agg)

        # Shape checks: r_max = 4, in=3, out=2
        self.assertEqual(tuple(A_agg.shape), (4, 3))
        self.assertEqual(tuple(B_agg.shape), (2, 4))
        self.assertEqual(tuple(W_agg.shape), (2, 3))

        # Expected values under Zero-Pad with equal weights:
        # - A rows 0..1: average of A0 and A1
        # - A rows 2..3: (0 from client 0 + A1 rows 2..3) / 2
        A_expected = torch.tensor(
            [
                [(1.0 + 3.0) / 2, (1.0 + 3.0) / 2, (1.0 + 3.0) / 2],
                [(2.0 + 4.0) / 2, (2.0 + 4.0) / 2, (2.0 + 4.0) / 2],
                [(0.0 + 5.0) / 2, (0.0 + 5.0) / 2, (0.0 + 5.0) / 2],
                [(0.0 + 6.0) / 2, (0.0 + 6.0) / 2, (0.0 + 6.0) / 2],
            ],
            dtype=torch.float32,
        )

        # - B cols 0..1: average of B0 and B1
        # - B cols 2..3: (0 from client 0 + B1 cols 2..3) / 2
        B_expected = torch.tensor(
            [
                [(10.0 + 7.0) / 2, (20.0 + 8.0) / 2, (0.0 + 9.0) / 2, (0.0 + 10.0) / 2],
                [(10.0 + 7.0) / 2, (20.0 + 8.0) / 2, (0.0 + 9.0) / 2, (0.0 + 10.0) / 2],
            ],
            dtype=torch.float32,
        )

        # - Non-LoRA param uses the same normalized weighting (equal weights -> plain average)
        W_expected = (W0 + W1) / 2.0

        self.assertTrue(torch.allclose(A_agg.cpu(), A_expected, atol=1e-6))
        self.assertTrue(torch.allclose(B_agg.cpu(), B_expected, atol=1e-6))
        self.assertTrue(torch.allclose(W_agg.cpu(), W_expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()

