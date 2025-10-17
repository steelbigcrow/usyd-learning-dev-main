import unittest
import torch

from usyd_learning.ml_algorithms.lora.lora_utils import LoRAUtils


class TestLoRABroadcastUtils(unittest.TestCase):
    def test_broadcast_slice_and_pad(self):
        # Global has r=3, local expects r=2
        gA = torch.randn(3, 5)
        gB = torch.randn(4, 3)
        lA = torch.zeros(2, 5)
        lB = torch.zeros(4, 2)

        global_sd = {
            "fc.lora_A": gA,
            "fc.lora_B": gB,
            "fc.weight": torch.randn(4, 5),
        }
        local_sd = {
            "fc.lora_A": lA,
            "fc.lora_B": lB,
            "fc.weight": torch.zeros(4, 5),
        }

        out = LoRAUtils.broadcast_lora_state_dict(global_sd, local_sd)

        self.assertEqual(tuple(out["fc.lora_A"].shape), (2, 5))
        self.assertEqual(tuple(out["fc.lora_B"].shape), (4, 2))
        # Slice correctness
        self.assertTrue(torch.allclose(out["fc.lora_A"], gA[:2, :]))
        self.assertTrue(torch.allclose(out["fc.lora_B"], gB[:, :2]))
        # Non-LoRA keys forwarded
        self.assertTrue(torch.allclose(out["fc.weight"], global_sd["fc.weight"]))

    def test_broadcast_pad_when_global_smaller(self):
        # Global r=1, local expects r=2 → pad
        gA = torch.randn(1, 5)
        gB = torch.randn(4, 1)
        lA = torch.zeros(2, 5)
        lB = torch.zeros(4, 2)

        global_sd = {
            "fc.lora_A": gA,
            "fc.lora_B": gB,
        }
        local_sd = {
            "fc.lora_A": lA,
            "fc.lora_B": lB,
        }

        out = LoRAUtils.broadcast_lora_state_dict(global_sd, local_sd)

        self.assertEqual(tuple(out["fc.lora_A"].shape), (2, 5))
        self.assertEqual(tuple(out["fc.lora_B"].shape), (4, 2))
        self.assertTrue(torch.allclose(out["fc.lora_A"][:1, :], gA))
        self.assertTrue(torch.allclose(out["fc.lora_B"][:, :1], gB))
        self.assertTrue(torch.all(out["fc.lora_A"][1:, :] == 0))
        self.assertTrue(torch.all(out["fc.lora_B"][:, 1:] == 0))


if __name__ == "__main__":
    unittest.main()
