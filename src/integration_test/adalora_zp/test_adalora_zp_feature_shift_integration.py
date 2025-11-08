import os
import sys
import unittest


# Ensure 'src' on sys.path so 'integration_test' can be imported
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from integration_test.common.utils import ensure_startup, run_scenario
from usyd_learning.ml_utils.model_utils import ModelUtils


class TestAdaLoRAZPFeatureShiftIntegration(unittest.TestCase):
    def test_adalora_zp_feature_shift_end_to_end(self):
        ensure_startup(__file__)

        cfg = os.path.abspath(os.path.join(os.path.dirname(__file__), "adalora_zp_feature_shift.yaml"))
        device = ModelUtils.accelerator_device()
        app, runner, server_var = run_scenario(cfg, rounds=1, device=device)

        # Aggregator method is Zero-Pad (zp)
        self.assertIsNotNone(server_var.aggregation_method)
        self.assertEqual(getattr(server_var.aggregation_method, "_aggregation_method", None), "zp")

        # Confirm AdaLoRA trainer in clients
        self.assertTrue(any(
            getattr(c.node_var.trainer.trainer_args, "trainer_type", "") == "adalora"
            for c in runner.client_node_list
        ))

        # Aggregated model_weight exists and is a dict
        self.assertIsNotNone(getattr(server_var, "model_weight", None))
        self.assertTrue(isinstance(server_var.model_weight, dict))
        self.assertGreater(len(server_var.model_weight.keys()), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
