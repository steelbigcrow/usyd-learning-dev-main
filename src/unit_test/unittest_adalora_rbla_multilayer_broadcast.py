import os
import unittest
import torch

# Ensure imports work when running under different IDEs
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))


class TestAdaLoRA_RBLA_MultiLayer_Broadcast(unittest.TestCase):
    def _make_three_client_peft(self):
        bp = "base_model.model"
        L1 = "L1"
        L2 = "L2"

        # All clients share base rank r=4, dims in=3, out=2
        # Values chosen to make expected aggregates easy to verify

        # Client 0
        A0_L1 = torch.tensor([[1, 1, 1], [2, 2, 2], [30, 30, 30], [40, 40, 40]], dtype=torch.float32)
        B0_L1 = torch.tensor([[10, 20, 30, 40], [10, 20, 30, 40]], dtype=torch.float32)
        E0_L1 = torch.tensor([[1.0], [1.0], [0.0], [0.0]], dtype=torch.float32)  # rr=2

        A0_L2 = torch.tensor([[101, 101, 101], [102, 102, 102], [130, 130, 130], [140, 140, 140]], dtype=torch.float32)
        B0_L2 = torch.tensor([[110, 120, 130, 140], [110, 120, 130, 140]], dtype=torch.float32)
        E0_L2 = torch.tensor([[1.0], [0.0], [0.0], [0.0]], dtype=torch.float32)  # rr=1

        sd0 = {
            f"{bp}.{L1}.lora_A.default": A0_L1.clone(),
            f"{bp}.{L1}.lora_B.default": B0_L1.clone(),
            f"{bp}.{L1}.lora_E.default": E0_L1.clone(),
            f"{bp}.{L1}.ranknum": torch.tensor(2.0),
            f"{bp}.{L2}.lora_A.default": A0_L2.clone(),
            f"{bp}.{L2}.lora_B.default": B0_L2.clone(),
            f"{bp}.{L2}.lora_E.default": E0_L2.clone(),
            f"{bp}.{L2}.ranknum": torch.tensor(1.0),
        }

        # Client 1
        A1_L1 = torch.tensor([[3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6]], dtype=torch.float32)
        B1_L1 = torch.tensor([[7, 8, 9, 10], [7, 8, 9, 10]], dtype=torch.float32)
        E1_L1 = torch.tensor([[1.0], [1.0], [1.0], [0.0]], dtype=torch.float32)  # rr=3

        A1_L2 = torch.tensor([[103, 103, 103], [104, 104, 104], [105, 105, 105], [106, 106, 106]], dtype=torch.float32)
        B1_L2 = torch.tensor([[107, 108, 109, 110], [107, 108, 109, 110]], dtype=torch.float32)
        E1_L2 = torch.tensor([[1.0], [1.0], [0.0], [0.0]], dtype=torch.float32)  # rr=2

        sd1 = {
            f"{bp}.{L1}.lora_A.default": A1_L1.clone(),
            f"{bp}.{L1}.lora_B.default": B1_L1.clone(),
            f"{bp}.{L1}.lora_E.default": E1_L1.clone(),
            f"{bp}.{L1}.ranknum": torch.tensor(3.0),
            f"{bp}.{L2}.lora_A.default": A1_L2.clone(),
            f"{bp}.{L2}.lora_B.default": B1_L2.clone(),
            f"{bp}.{L2}.lora_E.default": E1_L2.clone(),
            f"{bp}.{L2}.ranknum": torch.tensor(2.0),
        }

        # Client 2
        A2_L1 = torch.tensor([[11, 11, 11], [12, 12, 12], [13, 13, 13], [14, 14, 14]], dtype=torch.float32)
        B2_L1 = torch.tensor([[15, 16, 17, 18], [15, 16, 17, 18]], dtype=torch.float32)
        E2_L1 = torch.tensor([[1.0], [0.0], [0.0], [0.0]], dtype=torch.float32)  # rr=1

        A2_L2 = torch.tensor([[111, 111, 111], [112, 112, 112], [113, 113, 113], [114, 114, 114]], dtype=torch.float32)
        B2_L2 = torch.tensor([[115, 116, 117, 118], [115, 116, 117, 118]], dtype=torch.float32)
        E2_L2 = torch.tensor([[1.0], [1.0], [1.0], [0.0]], dtype=torch.float32)  # rr=3

        sd2 = {
            f"{bp}.{L1}.lora_A.default": A2_L1.clone(),
            f"{bp}.{L1}.lora_B.default": B2_L1.clone(),
            f"{bp}.{L1}.lora_E.default": E2_L1.clone(),
            f"{bp}.{L1}.ranknum": torch.tensor(1.0),
            f"{bp}.{L2}.lora_A.default": A2_L2.clone(),
            f"{bp}.{L2}.lora_B.default": B2_L2.clone(),
            f"{bp}.{L2}.lora_E.default": E2_L2.clone(),
            f"{bp}.{L2}.ranknum": torch.tensor(3.0),
        }

        return sd0, sd1, sd2, bp, L1, L2

    def test_multilayer_rbla_aggregate_and_broadcast(self):
        from usyd_learning.ml_algorithms.adalora.adalora_rbla_bridge import (
            peft_to_plain_lora_shrunk,
            plain_lora_to_peft,
            select_template_with_max_rank,
        )
        from usyd_learning.fl_algorithms.aggregation.methods._fed_aggregator_rbla import (
            FedAggregator_RBLA,
        )
        from usyd_learning.fl_algorithms.aggregation.fed_aggregator_args import (
            FedAggregatorArgs,
        )

        sd0_peft, sd1_peft, sd2_peft, bp, L1, L2 = self._make_three_client_peft()

        # Convert each client PEFT -> plain LoRA (AdaLoRA-shrunk)
        sds_plain = [
            peft_to_plain_lora_shrunk(sd0_peft),
            peft_to_plain_lora_shrunk(sd1_peft),
            peft_to_plain_lora_shrunk(sd2_peft),
        ]

        # RBLA aggregation with weights 1:2:1 -> normalized 0.25, 0.5, 0.25
        vols = [1.0, 2.0, 1.0]
        args = FedAggregatorArgs({"aggregation": {"device": "cpu", "method": "rbla"}})
        aggregator = FedAggregator_RBLA(args)
        aggregated_plain = aggregator.aggregate(list(zip(sds_plain, vols)))

        # Shapes: union effective ranks per layer -> r_max(L1)=3, r_max(L2)=3
        self.assertEqual(tuple(aggregated_plain[f"{L1}.lora_A"].shape), (3, 3))
        self.assertEqual(tuple(aggregated_plain[f"{L1}.lora_B"].shape), (2, 3))
        self.assertEqual(tuple(aggregated_plain[f"{L2}.lora_A"].shape), (3, 3))
        self.assertEqual(tuple(aggregated_plain[f"{L2}.lora_B"].shape), (2, 3))

        # Validate a few numeric expectations for L1
        A1 = aggregated_plain[f"{L1}.lora_A"]
        B1 = aggregated_plain[f"{L1}.lora_B"]
        # row0 present all: 0.25*1 + 0.5*3 + 0.25*11 = 4.5
        self.assertTrue(torch.allclose(A1[0], torch.tensor([4.5, 4.5, 4.5]), atol=1e-6))
        # row1 present clients 0 & 1: (0.25*2 + 0.5*4) / 0.75 = 10/3
        self.assertTrue(torch.allclose(A1[1], torch.tensor([10/3, 10/3, 10/3], dtype=torch.float32), atol=1e-6))
        # row2 present only client 1: equals 5
        self.assertTrue(torch.allclose(A1[2], torch.tensor([5.0, 5.0, 5.0]), atol=1e-6))
        # B col0 present all: 0.25*10 + 0.5*7 + 0.25*15 = 9.75
        self.assertTrue(torch.allclose(B1[:, 0], torch.tensor([9.75, 9.75]), atol=1e-6))
        # B col1 present clients 0 & 1: (0.25*20 + 0.5*8)/0.75 = 12.0
        self.assertTrue(torch.allclose(B1[:, 1], torch.tensor([12.0, 12.0]), atol=1e-6))
        # B col2 present only client 1: equals 9
        self.assertTrue(torch.allclose(B1[:, 2], torch.tensor([9.0, 9.0]), atol=1e-6))

        # Validate a few numeric expectations for L2
        A2 = aggregated_plain[f"{L2}.lora_A"]
        B2 = aggregated_plain[f"{L2}.lora_B"]
        # row0 present all: 0.25*101 + 0.5*103 + 0.25*111 = 104.5
        self.assertTrue(torch.allclose(A2[0], torch.tensor([104.5, 104.5, 104.5]), atol=1e-6))
        # row1 present clients 1 & 2: (0.5*104 + 0.25*112)/0.75 = 106.666...
        self.assertTrue(torch.allclose(A2[1], torch.tensor([106.6666667, 106.6666667, 106.6666667]), atol=1e-5))
        # row2 present only client 2: equals 113
        self.assertTrue(torch.allclose(A2[2], torch.tensor([113.0, 113.0, 113.0]), atol=1e-6))
        # B col0 present all: 0.25*110 + 0.5*107 + 0.25*115 = 109.75
        self.assertTrue(torch.allclose(B2[:, 0], torch.tensor([109.75, 109.75]), atol=1e-6))
        # B col1 present clients 1 & 2: (0.5*108 + 0.25*116)/0.75 = 110.666...
        self.assertTrue(torch.allclose(B2[:, 1], torch.tensor([110.6666667, 110.6666667]), atol=1e-5))
        # B col2 present only client 2: equals 117
        self.assertTrue(torch.allclose(B2[:, 2], torch.tensor([117.0, 117.0]), atol=1e-6))

        # Map back to PEFT per-layer templates (pick client with max base rank)
        template = select_template_with_max_rank([
            {"updated_weights": sd0_peft},
            {"updated_weights": sd1_peft},
            {"updated_weights": sd2_peft},
        ])
        aggregated_peft = plain_lora_to_peft(aggregated_plain, template)

        # Unified budget per layer -> expected K=2 for both L1 and L2
        for layer in (L1, L2):
            kA = f"{bp}.{layer}.lora_A.default"
            kB = f"{bp}.{layer}.lora_B.default"
            kE = f"{bp}.{layer}.lora_E.default"
            kR = f"{bp}.{layer}.ranknum"

            A_p = aggregated_peft[kA]
            B_p = aggregated_peft[kB]
            E_p = aggregated_peft[kE]
            R_p = aggregated_peft[kR]

            # Shapes remain (3,3)/(2,3), with channels beyond K=2 zeroed
            self.assertEqual(tuple(A_p.shape), (3, 3))
            self.assertEqual(tuple(B_p.shape), (2, 3))
            self.assertTrue(torch.allclose(A_p[2], torch.zeros_like(A_p[2]), atol=1e-6))
            self.assertTrue(torch.allclose(B_p[:, 2], torch.zeros_like(B_p[:, 2]), atol=1e-6))
            # E vector and ranknum reflect K=2
            self.assertEqual(tuple(E_p.shape), (4, 1))
            self.assertTrue(torch.allclose(E_p.view(-1).cpu(), torch.tensor([1.0, 1.0, 0.0, 0.0]), atol=1e-6))
            if torch.is_tensor(R_p):
                self.assertEqual(int(round(float(R_p.item()))), 2)
            else:
                self.assertEqual(int(round(float(R_p))), 2)

        # --- Broadcast (plain keys) ---
        from usyd_learning.fl_algorithms.aggregation.methods._fed_aggregator_rbla import FedAggregator_RBLA

        # Local evaluator A expects L1 r=2, L2 r=2
        localA = {
            f"{L1}.lora_A": torch.zeros((2, 3), dtype=torch.float32),
            f"{L1}.lora_B": torch.zeros((2, 2), dtype=torch.float32),
            f"{L2}.lora_A": torch.zeros((2, 3), dtype=torch.float32),
            f"{L2}.lora_B": torch.zeros((2, 2), dtype=torch.float32),
        }
        adaptedA = FedAggregator_RBLA.broadcast_lora_state_dict(aggregated_plain, localA)
        # Check shapes and that we received top-K slices
        self.assertEqual(tuple(adaptedA[f"{L1}.lora_A"].shape), (2, 3))
        self.assertEqual(tuple(adaptedA[f"{L1}.lora_B"].shape), (2, 2))
        self.assertTrue(torch.allclose(adaptedA[f"{L1}.lora_A"], aggregated_plain[f"{L1}.lora_A"][0:2, :], atol=1e-6))
        self.assertTrue(torch.allclose(adaptedA[f"{L1}.lora_B"], aggregated_plain[f"{L1}.lora_B"][:, 0:2], atol=1e-6))
        self.assertEqual(tuple(adaptedA[f"{L2}.lora_A"].shape), (2, 3))
        self.assertEqual(tuple(adaptedA[f"{L2}.lora_B"].shape), (2, 2))
        self.assertTrue(torch.allclose(adaptedA[f"{L2}.lora_A"], aggregated_plain[f"{L2}.lora_A"][0:2, :], atol=1e-6))
        self.assertTrue(torch.allclose(adaptedA[f"{L2}.lora_B"], aggregated_plain[f"{L2}.lora_B"][:, 0:2], atol=1e-6))

        # --- Broadcast (PEFT-shaped keys) ---
        # Local evaluator B: L1 r=3, L2 r=1
        localB = {
            f"{bp}.{L1}.lora_A.default": torch.zeros((3, 3), dtype=torch.float32),
            f"{bp}.{L1}.lora_B.default": torch.zeros((2, 3), dtype=torch.float32),
            f"{bp}.{L2}.lora_A.default": torch.zeros((1, 3), dtype=torch.float32),
            f"{bp}.{L2}.lora_B.default": torch.zeros((2, 1), dtype=torch.float32),
        }
        adaptedB = FedAggregator_RBLA.broadcast_lora_state_dict(aggregated_peft, localB)
        # L1 receives its first 3 rows/cols (already 3), L2 down to r=1
        self.assertEqual(tuple(adaptedB[f"{bp}.{L1}.lora_A.default"].shape), (3, 3))
        self.assertEqual(tuple(adaptedB[f"{bp}.{L1}.lora_B.default"].shape), (2, 3))
        self.assertTrue(torch.allclose(adaptedB[f"{bp}.{L1}.lora_A.default"], aggregated_peft[f"{bp}.{L1}.lora_A.default"][0:3, :], atol=1e-6))
        self.assertTrue(torch.allclose(adaptedB[f"{bp}.{L1}.lora_B.default"], aggregated_peft[f"{bp}.{L1}.lora_B.default"][:, 0:3], atol=1e-6))
        self.assertEqual(tuple(adaptedB[f"{bp}.{L2}.lora_A.default"].shape), (1, 3))
        self.assertEqual(tuple(adaptedB[f"{bp}.{L2}.lora_B.default"].shape), (2, 1))
        self.assertTrue(torch.allclose(adaptedB[f"{bp}.{L2}.lora_A.default"], aggregated_peft[f"{bp}.{L2}.lora_A.default"][0:1, :], atol=1e-6))
        self.assertTrue(torch.allclose(adaptedB[f"{bp}.{L2}.lora_B.default"], aggregated_peft[f"{bp}.{L2}.lora_B.default"][:, 0:1], atol=1e-6))


if __name__ == "__main__":
    unittest.main()

