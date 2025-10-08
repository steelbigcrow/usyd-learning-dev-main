import os
import unittest
import torch

from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))


class TestAdaLoRA_RankHints(unittest.TestCase):
    def _make_base_peft(self, bp: str = "base_model.model", layer: str = "L"):
        # Base rank r=4, dims: out=2, in=3
        A = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
            ],
            dtype=torch.float32,
        )
        B = torch.tensor(
            [
                [10.0, 20.0, 30.0, 40.0],
                [10.0, 20.0, 30.0, 40.0],
            ],
            dtype=torch.float32,
        )
        sd = {
            f"{bp}.{layer}.lora_A.default": A.clone(),
            f"{bp}.{layer}.lora_B.default": B.clone(),
        }
        return sd, bp, layer

    def test_scalar_ranknum_hint_shrinks(self):
        from usyd_learning.ml_algorithms.adalora.adalora_rbla_bridge import (
            peft_to_plain_lora_shrunk,
            plain_lora_to_peft,
        )

        sd, bp, L = self._make_base_peft()
        # Provide scalar rank hint r=2
        sd[f"{bp}.{L}.ranknum"] = torch.tensor(2.0)

        plain = peft_to_plain_lora_shrunk(sd)
        # Expect shrink to r=2
        self.assertEqual(tuple(plain[f"{L}.lora_A"].shape), (2, 3))
        self.assertEqual(tuple(plain[f"{L}.lora_B"].shape), (2, 2))
        # rank_mask should exist (built from scores top-rr)
        self.assertIn(f"{L}.rank_mask", plain)
        self.assertEqual(plain[f"{L}.rank_rr"].item(), 2.0)

        # Map back to PEFT and ensure lora_E and ranknum reflect K=2
        peft = plain_lora_to_peft(plain, sd)
        self.assertEqual(tuple(peft[f"{bp}.{L}.lora_A.default"].shape), (2, 3))
        self.assertEqual(tuple(peft[f"{bp}.{L}.lora_B.default"].shape), (2, 2))
        # 'lora_E' may not exist if not present in template; 'ranknum' should be updated when present
        self.assertIn(f"{bp}.{L}.ranknum", peft)
        self.assertEqual(int(round(float(peft[f"{bp}.{L}.ranknum"].item()))), 2)

    def test_vector_rankmask_hint_shrinks_and_carries_mask(self):
        from usyd_learning.ml_algorithms.adalora.adalora_rbla_bridge import (
            peft_to_plain_lora_shrunk,
            plain_lora_to_peft,
        )

        sd, bp, L = self._make_base_peft()
        # Provide vector hint v=[1,1,0,0] => rr=sum(v)=2 and scores=v
        v = torch.tensor([1.0, 1.0, 0.0, 0.0])
        sd[f"{bp}.{L}.ranknum"] = v.clone()

        plain = peft_to_plain_lora_shrunk(sd)
        self.assertEqual(tuple(plain[f"{L}.lora_A"].shape), (2, 3))
        self.assertEqual(tuple(plain[f"{L}.lora_B"].shape), (2, 2))
        self.assertIn(f"{L}.rank_mask", plain)
        # rank_mask built from top-rr channels of vector v should be [1,1,0,0]
        self.assertTrue(torch.allclose(plain[f"{L}.rank_mask"].view(-1), v, atol=1e-6))

        peft = plain_lora_to_peft(plain, sd)
        # ranknum should be present and reflect K=2
        self.assertIn(f"{bp}.{L}.ranknum", peft)
        self.assertEqual(int(round(float(peft[f"{bp}.{L}.ranknum"].item()))), 2)

    def test_no_hint_no_mask_full_rank(self):
        from usyd_learning.ml_algorithms.adalora.adalora_rbla_bridge import (
            peft_to_plain_lora_shrunk,
            plain_lora_to_peft,
        )

        sd, bp, L = self._make_base_peft()
        # No E, no ranknum -> use full rank r=4
        plain = peft_to_plain_lora_shrunk(sd)
        self.assertEqual(tuple(plain[f"{L}.lora_A"].shape), (4, 3))
        self.assertEqual(tuple(plain[f"{L}.lora_B"].shape), (2, 4))
        # rank_mask should be all ones of length 4
        self.assertTrue(torch.allclose(plain[f"{L}.rank_mask"].view(-1), torch.ones(4), atol=1e-6))
        self.assertEqual(int(plain[f"{L}.rank_rr"].item()), 4)

        # Mapping back without ranknum/lora_E in template will not add those keys; ensure A/B remain full-rank if later mapped
        peft = plain_lora_to_peft(plain, sd)
        self.assertEqual(tuple(peft[f"{bp}.{L}.lora_A.default"].shape), (4, 3))
        self.assertEqual(tuple(peft[f"{bp}.{L}.lora_B.default"].shape), (2, 4))


if __name__ == "__main__":
    unittest.main()
