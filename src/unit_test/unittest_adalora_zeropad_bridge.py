import os
import unittest
import torch

# Ensure imports work when running under different IDEs
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))


class TestAdaLoRAWithZeroPad(unittest.TestCase):
    def _make_peft_clients(self):
        # Create two PEFT/AdaLoRA-shaped state_dicts for the same logical layer
        bp = "base_model.model"
        layer = "layer"

        # Base (max) rank and dims
        in_dim = 3
        out_dim = 2

        # Client 0: effective rank via E mask = 2 (rows/cols 0,1)
        A0_full = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [30.0, 30.0, 30.0],
                [40.0, 40.0, 40.0],
            ],
            dtype=torch.float32,
        )  # [4, 3]
        B0_full = torch.tensor(
            [
                [10.0, 20.0, 30.0, 40.0],
                [10.0, 20.0, 30.0, 40.0],
            ],
            dtype=torch.float32,
        )  # [2, 4]
        E0 = torch.tensor([[1.0], [1.0], [0.0], [0.0]], dtype=torch.float32)
        ranknum0 = torch.tensor(2.0, dtype=torch.float32)

        sd0 = {
            f"{bp}.{layer}.lora_A.default": A0_full.clone(),
            f"{bp}.{layer}.lora_B.default": B0_full.clone(),
            f"{bp}.{layer}.lora_E.default": E0.clone(),
            f"{bp}.{layer}.ranknum": ranknum0.clone(),
        }

        # Client 1: effective rank via E mask = 3 (rows/cols 0,1,2)
        A1_full = torch.tensor(
            [
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
                [5.0, 5.0, 5.0],
                [6.0, 6.0, 6.0],
            ],
            dtype=torch.float32,
        )  # [4, 3]
        B1_full = torch.tensor(
            [
                [7.0, 8.0, 9.0, 10.0],
                [7.0, 8.0, 9.0, 10.0],
            ],
            dtype=torch.float32,
        )  # [2, 4]
        E1 = torch.tensor([[1.0], [1.0], [1.0], [0.0]], dtype=torch.float32)
        ranknum1 = torch.tensor(3.0, dtype=torch.float32)

        sd1 = {
            f"{bp}.{layer}.lora_A.default": A1_full.clone(),
            f"{bp}.{layer}.lora_B.default": B1_full.clone(),
            f"{bp}.{layer}.lora_E.default": E1.clone(),
            f"{bp}.{layer}.ranknum": ranknum1.clone(),
        }

        return sd0, sd1, bp, layer

    def test_adalora_zeropad_equal_weights(self):
        from usyd_learning.ml_algorithms.adalora.adalora_rbla_bridge import (
            peft_to_plain_lora_shrunk,
            plain_lora_to_peft,
            select_template_with_max_rank,
        )
        from usyd_learning.fl_algorithms.aggregation.methods._fed_aggregator_zeropad import (
            FedAggregator_ZeroPad,
        )
        from usyd_learning.fl_algorithms.aggregation.fed_aggregator_args import (
            FedAggregatorArgs,
        )

        sd0_peft, sd1_peft, bp, layer = self._make_peft_clients()

        # Convert to shrunk plain-LoRA dicts based on AdaLoRA masks
        sd0_plain = peft_to_plain_lora_shrunk(sd0_peft)
        sd1_plain = peft_to_plain_lora_shrunk(sd1_peft)

        # Aggregate via ZeroPad with equal weights (1,1)
        args = FedAggregatorArgs({"aggregation": {"device": "cpu", "method": "zp"}})
        aggregator = FedAggregator_ZeroPad(args)
        aggregated_plain = aggregator.aggregate([(sd0_plain, 1.0), (sd1_plain, 1.0)])

        A_agg = aggregated_plain[f"{layer}.lora_A"]
        B_agg = aggregated_plain[f"{layer}.lora_B"]
        print("\n=== [ZP+AdaLoRA] Equal Weights — Aggregated (plain) ===")
        print("A_agg shape:", tuple(A_agg.shape))
        print(A_agg)
        print("B_agg shape:", tuple(B_agg.shape))
        print(B_agg)

        # Expected shapes: union rank r_max = 3 after shrink
        self.assertEqual(tuple(A_agg.shape), (3, 3))
        self.assertEqual(tuple(B_agg.shape), (2, 3))

        # Under ZeroPad with equal weights (0.5, 0.5):
        # Overlap rows/cols -> (A0 + A1)/2 ; missing dims -> (0 + value)/2
        A_expected = torch.tensor(
            [
                [(1.0 + 3.0) / 2] * 3,  # 2.0
                [(2.0 + 4.0) / 2] * 3,  # 3.0
                [0.5 * 5.0] * 3,        # 2.5
            ],
            dtype=torch.float32,
        )
        B_expected = torch.tensor(
            [
                [(10.0 + 7.0) / 2, (20.0 + 8.0) / 2, 0.5 * 9.0],
                [(10.0 + 7.0) / 2, (20.0 + 8.0) / 2, 0.5 * 9.0],
            ],
            dtype=torch.float32,
        )

        self.assertTrue(torch.allclose(A_agg.cpu(), A_expected, atol=1e-6))
        self.assertTrue(torch.allclose(B_agg.cpu(), B_expected, atol=1e-6))

        # Map back to PEFT-shaped dict using a max-rank template
        template = select_template_with_max_rank([
            {"updated_weights": sd0_peft},
            {"updated_weights": sd1_peft},
        ])
        aggregated_peft = plain_lora_to_peft(aggregated_plain, template)

        kA = f"{bp}.{layer}.lora_A.default"
        kB = f"{bp}.{layer}.lora_B.default"
        kE = f"{bp}.{layer}.lora_E.default"
        kR = f"{bp}.{layer}.ranknum"

        A_peft = aggregated_peft[kA]
        B_peft = aggregated_peft[kB]
        E_peft = aggregated_peft[kE]
        R_peft = aggregated_peft[kR]

        print("\n=== [ZP+AdaLoRA] Equal Weights — Aggregated (PEFT-shaped) ===")
        print("A_peft shape:", tuple(A_peft.shape))
        print(A_peft)
        print("B_peft shape:", tuple(B_peft.shape))
        print(B_peft)
        print("E_peft:", E_peft.view(-1))
        print("ranknum:", float(R_peft.item()) if torch.is_tensor(R_peft) else R_peft)

        # Aggregated mask = 0.5*[1,1,0,0] + 0.5*[1,1,1,0] = [1,1,0.5,0] -> sum=2.5 -> K=2
        # plain_lora_to_peft zeros out rows/cols beyond K
        self.assertEqual(tuple(A_peft.shape), (3, 3))
        self.assertEqual(tuple(B_peft.shape), (2, 3))
        self.assertTrue(torch.allclose(A_peft[0:2].cpu(), A_expected[0:2], atol=1e-6))
        self.assertTrue(torch.allclose(B_peft[:, 0:2].cpu(), B_expected[:, 0:2], atol=1e-6))
        self.assertTrue(torch.allclose(A_peft[2].cpu(), torch.zeros_like(A_peft[2]), atol=1e-6))
        self.assertTrue(torch.allclose(B_peft[:, 2].cpu(), torch.zeros_like(B_peft[:, 2]), atol=1e-6))
        self.assertTrue(torch.allclose(E_peft.view(-1).cpu(), torch.tensor([1.0, 1.0, 0.0, 0.0]), atol=1e-6))
        if torch.is_tensor(R_peft):
            self.assertEqual(int(round(float(R_peft.item()))), 2)
        else:
            self.assertEqual(int(round(float(R_peft))), 2)

    def test_adalora_zeropad_unequal_weights(self):
        from usyd_learning.ml_algorithms.adalora.adalora_rbla_bridge import (
            peft_to_plain_lora_shrunk,
            plain_lora_to_peft,
            select_template_with_max_rank,
        )
        from usyd_learning.fl_algorithms.aggregation.methods._fed_aggregator_zeropad import (
            FedAggregator_ZeroPad,
        )
        from usyd_learning.fl_algorithms.aggregation.fed_aggregator_args import (
            FedAggregatorArgs,
        )

        sd0_peft, sd1_peft, bp, layer = self._make_peft_clients()
        sd0_plain = peft_to_plain_lora_shrunk(sd0_peft)
        sd1_plain = peft_to_plain_lora_shrunk(sd1_peft)

        # Unequal volumes -> normalized weights w0=0.25, w1=0.75
        vols = (1.0, 3.0)
        args = FedAggregatorArgs({"aggregation": {"device": "cpu", "method": "zp"}})
        aggregator = FedAggregator_ZeroPad(args)
        aggregated_plain = aggregator.aggregate([(sd0_plain, vols[0]), (sd1_plain, vols[1])])

        A_agg = aggregated_plain[f"{layer}.lora_A"]
        B_agg = aggregated_plain[f"{layer}.lora_B"]
        print("\n=== [ZP+AdaLoRA] Unequal Weights (1:3) — Aggregated (plain) ===")
        print("A_agg shape:", tuple(A_agg.shape))
        print(A_agg)
        print("B_agg shape:", tuple(B_agg.shape))
        print(B_agg)

        # Shapes are union rank r_max = 3
        self.assertEqual(tuple(A_agg.shape), (3, 3))
        self.assertEqual(tuple(B_agg.shape), (2, 3))

        w0, w1 = 0.25, 0.75
        A_expected = torch.tensor(
            [
                [w0 * 1.0 + w1 * 3.0] * 3,  # 2.5
                [w0 * 2.0 + w1 * 4.0] * 3,  # 3.5
                [w1 * 5.0] * 3,             # 3.75
            ],
            dtype=torch.float32,
        )
        B_expected = torch.tensor(
            [
                [w0 * 10.0 + w1 * 7.0, w0 * 20.0 + w1 * 8.0, w1 * 9.0],
                [w0 * 10.0 + w1 * 7.0, w0 * 20.0 + w1 * 8.0, w1 * 9.0],
            ],
            dtype=torch.float32,
        )

        self.assertTrue(torch.allclose(A_agg.cpu(), A_expected, atol=1e-6))
        self.assertTrue(torch.allclose(B_agg.cpu(), B_expected, atol=1e-6))

        # Map back to PEFT and validate unified K
        template = select_template_with_max_rank([
            {"updated_weights": sd0_peft},
            {"updated_weights": sd1_peft},
        ])
        aggregated_peft = plain_lora_to_peft(aggregated_plain, template)

        kA = f"{bp}.{layer}.lora_A.default"
        kB = f"{bp}.{layer}.lora_B.default"
        kE = f"{bp}.{layer}.lora_E.default"
        kR = f"{bp}.{layer}.ranknum"

        A_peft = aggregated_peft[kA]
        B_peft = aggregated_peft[kB]
        E_peft = aggregated_peft[kE]
        R_peft = aggregated_peft[kR]

        print("\n=== [ZP+AdaLoRA] Unequal Weights — Aggregated (PEFT-shaped) ===")
        print("A_peft shape:", tuple(A_peft.shape))
        print(A_peft)
        print("B_peft shape:", tuple(B_peft.shape))
        print(B_peft)
        print("E_peft:", E_peft.view(-1))
        print("ranknum:", float(R_peft.item()) if torch.is_tensor(R_peft) else R_peft)

        # Aggregated mask = 0.25*[1,1,0,0] + 0.75*[1,1,1,0] -> [1,1,0.75,0]; sum=2.75 -> K=3
        self.assertEqual(tuple(A_peft.shape), (3, 3))
        self.assertEqual(tuple(B_peft.shape), (2, 3))
        self.assertTrue(torch.allclose(A_peft.cpu(), A_expected, atol=1e-6))
        self.assertTrue(torch.allclose(B_peft.cpu(), B_expected, atol=1e-6))
        self.assertTrue(torch.allclose(E_peft.view(-1).cpu(), torch.tensor([1.0, 1.0, 1.0, 0.0]), atol=1e-6))
        if torch.is_tensor(R_peft):
            self.assertEqual(int(round(float(R_peft.item()))), 3)
        else:
            self.assertEqual(int(round(float(R_peft))), 3)


if __name__ == "__main__":
    unittest.main()

