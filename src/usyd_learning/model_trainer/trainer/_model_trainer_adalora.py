from __future__ import annotations
from typing import Any, Dict, List

import torch.nn as nn

from ._model_trainer_standard import ModelTrainer_Standard
from ..model_trainer_args import ModelTrainerArgs
from ...ml_utils.model_utils import ModelUtils

try:
    from peft import PeftModel
except Exception:  # peft is an optional dependency at runtime
    PeftModel = None  # type: ignore


class ModelTrainer_AdaLoRA(ModelTrainer_Standard):
    """
    AdaLoRA trainer using the standard loop. Assumes the incoming model has been
    wrapped via PEFT AdaLoRA (see ml_algorithms.adalora.peft_adalora).

    This trainer intentionally reuses the base training loop semantics from
    ModelTrainer to minimize behavioral delta; the AdaLoRA behavior is driven by
    the PEFT-wrapped model and the optimizer built on top of its trainable params.
    """

    def __init__(self, trainer_args: ModelTrainerArgs):
        super().__init__(trainer_args)
        if trainer_args.model is None:
            raise ValueError("Training Model is None.")
        if trainer_args.optimizer is None:
            raise ValueError("Training optimizer is None.")

        self.device = ModelUtils.accelerator_device()
        self.model: nn.Module = trainer_args.model

    def set_model(self, model: nn.Module):
        """Ensure model is on correct device (assumes already PEFT-wrapped)."""
        self.trainer_args.model = model
        if str(next(model.parameters()).device) != self.trainer_args.device:
            self.trainer_args.model = model.to(self.trainer_args.device)
        self.model = self.trainer_args.model
        return self

    # Inherit train_step/train from ModelTrainer_Standard.
