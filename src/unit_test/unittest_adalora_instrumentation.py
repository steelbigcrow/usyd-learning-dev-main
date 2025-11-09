import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from startup_init import startup_init_path

startup_init_path(os.path.dirname(os.path.abspath(__file__)))

from usyd_learning.monitoring.instrumentation import adalora as adalora_instr


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)


class TestAdaLoRAInstrumentation(unittest.TestCase):
    def _fake_hub(self):
        cfg = SimpleNamespace(raw={"monitoring": {"adalora": {"enabled": "true"}}})
        hub = SimpleNamespace(
            config=cfg,
            run_id="run-test",
            current_round=SimpleNamespace(round_index=3),
            records=[],
            pad_count=0,
            slice_count=0,
        )

        def write(payload):
            hub.records.append(payload)

        def pad(delta):
            hub.pad_count += delta

        def slc(delta):
            hub.slice_count += delta

        hub.write_adalora_rank = write
        hub.incr_broadcast_pad = pad
        hub.incr_broadcast_slice = slc
        return hub

    def test_patch_train_step_emits_rank_rows(self):
        hub = self._fake_hub()
        model = DummyModel()

        class DummyTrainer:
            def __init__(self):
                self.trainer_args = SimpleNamespace(model=model, device="cpu")
                self.model = model
                self._epoch_idx = 1
                self.train_calls = 0

            def train_step(self, *args, **kwargs):
                self.train_calls += 1
                return torch.tensor(0.0)

        with mock.patch(
            "usyd_learning.model_trainer.trainer._model_trainer_standard.ModelTrainer_Standard",
            DummyTrainer,
        ), mock.patch.object(
            adalora_instr, "LoRAUtils"
        ) as mock_utils, mock.patch(
            "usyd_learning.ml_algorithms.adalora.adalora_rbla_bridge.peft_to_plain_lora_shrunk",
            return_value={"layer.rank_rr": torch.tensor(2.0)},
        ), mock.patch.object(
            adalora_instr, "get_hub", return_value=hub
        ):
            mock_utils.get_lora_ranks.return_value = {"layer": 4}
            adalora_instr.patch_train_step_rank_snapshot()
            trainer = DummyTrainer()
            trainer.train_step(None)

        self.assertEqual(len(hub.records), 1)
        rec = hub.records[0]
        self.assertEqual(rec["layer"], "layer")
        self.assertEqual(rec["r"], 4)
        self.assertEqual(rec["r_eff"], 2)

    def test_patch_broadcast_counts_pad_and_slice(self):
        hub = self._fake_hub()

        def fake_broadcast(global_sd, local_sd, lora_suffixes=None):
            return local_sd

        with mock.patch.object(adalora_instr, "get_hub", return_value=hub), mock.patch.object(
            adalora_instr.LoRAUtils, "broadcast_lora_state_dict", side_effect=fake_broadcast
        ):
            adalora_instr.patch_broadcast_pad_slice_counters()
            global_sd = {
                "layer.lora_A": torch.zeros(4, 3),
                "layer.lora_B": torch.zeros(2, 4),
            }
            local_sd = {
                "layer.lora_A": torch.zeros(2, 3),
                "layer.lora_B": torch.zeros(2, 2),
            }
            adalora_instr.LoRAUtils.broadcast_lora_state_dict(global_sd, local_sd)
            # Trigger pad path (local expects larger rank than global).
            local_big = {
                "layer.lora_A": torch.zeros(6, 3),
                "layer.lora_B": torch.zeros(2, 6),
            }
            adalora_instr.LoRAUtils.broadcast_lora_state_dict(local_sd, local_big)

        self.assertGreaterEqual(hub.slice_count, 1)
        self.assertGreaterEqual(hub.pad_count, 1)


if __name__ == "__main__":
    unittest.main()
