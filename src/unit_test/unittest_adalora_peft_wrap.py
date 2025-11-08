import os
import sys
import types
import unittest
from types import SimpleNamespace

import torch.nn as nn

from startup_init import startup_init_path

startup_init_path(os.path.dirname(os.path.abspath(__file__)))

from usyd_learning.ml_algorithms.adalora.peft_adalora import (
    AdaLoRAOptions,
    wrap_with_adalora,
)


class FakeAdaLoraConfig:
    last_kwargs = None

    def __init__(
        self,
        r=None,
        lora_alpha=None,
        lora_dropout=None,
        target_modules=None,
        total_step=None,
        **kwargs,
    ):
        stored = dict(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            total_step=total_step,
        )
        stored.update(kwargs)
        FakeAdaLoraConfig.last_kwargs = stored


def fake_get_peft_model(model, config):
    fake_get_peft_model.calls.append((model, config))
    # Emulate PEFT returning the same module (wrapped in-place).
    model._adalora_wrapped = True  # type: ignore[attr-defined]
    return model


class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):  # pragma: no cover - not invoked
        return self.fc2(self.relu(self.fc1(x)))


class TestAdaLoRAPeftWrap(unittest.TestCase):
    def setUp(self):
        # Inject a fake peft module so wrap_with_adalora works without the optional dependency.
        self._orig = sys.modules.get("peft")
        fake_mod = types.SimpleNamespace(
            AdaLoraConfig=FakeAdaLoraConfig,
            get_peft_model=fake_get_peft_model,
        )
        sys.modules["peft"] = fake_mod
        fake_get_peft_model.calls = []
        FakeAdaLoraConfig.last_kwargs = None

    def tearDown(self):
        if self._orig is None:
            sys.modules.pop("peft", None)
        else:
            sys.modules["peft"] = self._orig

    def test_auto_detects_linear_targets(self):
        model = SimpleLinear()
        opts = AdaLoRAOptions(r=4, total_step=10)
        wrapped = wrap_with_adalora(model, opts)
        self.assertIs(wrapped, model)
        self.assertTrue(getattr(model, "_adalora_wrapped", False))
        # Target modules should automatically include fc1 and fc2 (leaf Linear modules).
        target_modules = FakeAdaLoraConfig.last_kwargs["target_modules"]
        self.assertListEqual(sorted(target_modules), ["fc1", "fc2"])

    def test_extra_kwargs_are_filtered(self):
        model = SimpleLinear()
        opts = AdaLoRAOptions(
            r=8,
            total_step=100,
            extra_kwargs={"beta1": 0.9, "nonsense": 123},
        )
        wrap_with_adalora(model, opts)
        kwargs = FakeAdaLoraConfig.last_kwargs
        self.assertNotIn("beta1", kwargs)  # unsupported keys dropped
        self.assertNotIn("nonsense", kwargs)
        self.assertEqual(kwargs["r"], 8)

    def test_raise_when_no_linear_layer(self):
        class NoLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3)

            def forward(self, x):  # pragma: no cover - not invoked
                return self.conv(x)

        model = NoLinear()
        opts = AdaLoRAOptions(r=4, total_step=10)
        with self.assertRaisesRegex(ValueError, "No target modules"):
            wrap_with_adalora(model, opts)


if __name__ == "__main__":
    unittest.main()
