import os
import unittest

import torch

from startup_init import startup_init_path

startup_init_path(os.path.dirname(os.path.abspath(__file__)))

from usyd_learning.fed_strategy.client_strategy_impl._svd_client import (
    SvdClientTrainingStrategy,
)


class TestSvdClientBroadcast(unittest.TestCase):
    def test_local_rank_slice_and_pad(self):
        global_sd = {
            "layer.lora_A": torch.arange(12, dtype=torch.float32).view(4, 3),
            "layer.lora_B": torch.arange(8, dtype=torch.float32).view(2, 4),
        }
        local_sd = {
            "layer.lora_A": torch.zeros(2, 3),
            "layer.lora_B": torch.zeros(2, 2),
        }
        result = SvdClientTrainingStrategy._broadcast_lora_state_dict(global_sd, local_sd)
        self.assertEqual(tuple(result["layer.lora_A"].shape), (2, 3))
        self.assertTrue(torch.allclose(result["layer.lora_A"], global_sd["layer.lora_A"][:2], atol=1e-6))

        # When local rank is larger than global, expect padding.
        local_big = {
            "layer.lora_A": torch.zeros(6, 3),
            "layer.lora_B": torch.zeros(2, 6),
        }
        padded = SvdClientTrainingStrategy._broadcast_lora_state_dict(global_sd, local_big)
        self.assertEqual(tuple(padded["layer.lora_A"].shape), (6, 3))
        self.assertTrue(torch.allclose(padded["layer.lora_A"][:4], global_sd["layer.lora_A"], atol=1e-6))
        self.assertTrue(torch.all(padded["layer.lora_A"][4:] == 0))


if __name__ == "__main__":
    unittest.main()
