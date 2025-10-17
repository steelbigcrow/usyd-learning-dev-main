import unittest
import torch

from usyd_learning.fl_algorithms.aggregation.methods._fed_aggregator_svd import (
    FedAggregator_SVD,
)
from usyd_learning.fl_algorithms.aggregation.fed_aggregator_args import (
    FedAggregatorArgs,
)
from usyd_learning.ml_algorithms.lora.lora_utils import LoRAUtils


class TestFedAggregatorSVD(unittest.TestCase):
    def test_svd_aggregates_delta_correctly(self):
        # Two clients, one linear layer prefix 'fc'
        # Client 0: r=2, shapes A[2,in], B[out,2]
        # Client 1: r=1
        m, n = 4, 3  # out, in
        # Construct A/B so that W = B @ A is easy to reason about
        A0 = torch.tensor([[1.0, 0.0, 2.0],
                           [0.0, 1.0, 1.0]])  # [2,3]
        B0 = torch.tensor([[1.0, 0.0],
                           [0.0, 1.0],
                           [1.0, 1.0],
                           [0.5, 0.5]])      # [4,2]
        A1 = torch.tensor([[2.0, -1.0, 0.0]])  # [1,3]
        B1 = torch.tensor([[1.0],
                           [0.0],
                           [0.0],
                           [1.0]])            # [4,1]

        sd0 = {
            "fc.lora_A": A0,
            "fc.lora_B": B0,
        }
        sd1 = {
            "fc.lora_A": A1,
            "fc.lora_B": B1,
        }

        # volumes/weights
        vol0, vol1 = 10.0, 6.0

        # Expected weighted delta W_g
        W0 = B0 @ A0  # [4,3]
        W1 = B1 @ A1  # [4,3]
        Wg = (vol0 * W0 + vol1 * W1) / (vol0 + vol1)

        args = FedAggregatorArgs({
            "aggregation": {"method": "svd", "device": "cpu"}
        })
        agg = FedAggregator_SVD(args)
        out = agg.aggregate([
            {"updated_weights": sd0, "train_record": {"data_sample_num": vol0}},
            {"updated_weights": sd1, "train_record": {"data_sample_num": vol1}},
        ])

        # Aggregator returns plain-LoRA A/B with r_max = 2
        self.assertIn("fc.lora_A", out)
        self.assertIn("fc.lora_B", out)
        Amax = out["fc.lora_A"]
        Bmax = out["fc.lora_B"]

        self.assertEqual(tuple(Amax.shape), (2, n))
        self.assertEqual(tuple(Bmax.shape), (m, 2))

        # Reconstruct approximated W and compare with rank-2 SVD truncation of Wg
        Wrec = Bmax @ Amax
        self.assertEqual(tuple(Wrec.shape), (m, n))

        # Ground-truth best rank-2 approx via the same split method
        A2, B2 = LoRAUtils.svd_split(Wg, r=2, method="sqrt")
        Wg_r2 = B2 @ A2
        self.assertTrue(torch.allclose(Wrec, Wg_r2, atol=1e-4, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
