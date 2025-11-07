from typing import Any
import torch.nn as nn
import torch
import numpy as np
import random
import math
from tqdm import tqdm
from usyd_learning.model_trainer.model_trainer_args import ModelTrainerArgs

from ..model_trainer import ModelTrainer
from ...ml_algorithms import ModelExtractor
from ...ml_utils import console
from ...ml_utils.model_utils import ModelUtils

class ModelTrainer_Standard(ModelTrainer):
    def __init__(self, trainer_args: ModelTrainerArgs):
        super().__init__(trainer_args)

        if trainer_args.model is None:
            raise ValueError("Training Model is None.")
        if trainer_args.optimizer is None:
            raise ValueError("Training optimizer is None.")
        
        self.device = ModelUtils.accelerator_device()
        self.model: nn.Module = trainer_args.model
        clip_cfg = None
        try:
            clip_cfg = trainer_args.get("max_grad_norm", 5.0)
        except Exception:
            clip_cfg = 5.0
        self.max_grad_norm = None
        try:
            if clip_cfg not in (None, False):
                clip_val = float(clip_cfg)
                if clip_val > 0:
                    self.max_grad_norm = clip_val
        except (TypeError, ValueError):
            self.max_grad_norm = None
        return

    def set_model(self, model: nn.Module):
        self.trainer_args.model = model
        if str(next(model.parameters()).device) != self.trainer_args.device:
            self.trainer_args.model = model.to(self.trainer_args.device)
        self.model = self.trainer_args.model
        return self

    def train_step(self) -> float:
        ta = self.trainer_args
        if ta.optimizer is None:
            raise ValueError("Trainer optimizer is None.")
        if ta.model is None:
            raise ValueError("Trainer model is None.")
        if ta.loss_func is None:
            raise ValueError("Trainer loss function is None.")
        if ta.train_loader is None:
            raise ValueError("Trainer train_loader is None.")

        train_dl = ta.train_loader.data_loader
        if not hasattr(train_dl, "__iter__"):
            raise TypeError(f"train_loader must be an iterable DataLoader, got {type(train_dl).__name__}")

        total_epochs = getattr(ta, "total_epochs", getattr(ta, "epochs", None))

        ta.model.to(self.device)
        ta.model.train()
        running_loss, total_batch = 0.0, 0

        from tqdm.auto import tqdm
        loop = tqdm(
            train_dl,
            desc=f"Training (epoch {self._epoch_idx}{'/' + str(total_epochs) if total_epochs else ''})",
            leave=True, ncols=120, mininterval=0.1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        for inputs, labels in loop:
            inputs = inputs.to(ta.device)
            labels = labels.to(ta.device)

            ta.optimizer.zero_grad()
            outputs = ta.model(inputs)
            loss = ta.loss_func(outputs, labels)
            if not torch.isfinite(loss).item():
                console.warn(
                    "[Trainer] Non-finite loss detected; skipping batch and scrubbing model parameters."
                )
                self._scrub_model_params(ta.model)
                continue

            loss.backward()

            if not self._grads_are_finite(ta.model):
                console.warn(
                    "[Trainer] Non-finite gradients detected; skipping optimizer step for this batch."
                )
                ta.optimizer.zero_grad()
                self._scrub_model_params(ta.model)
                continue

            if self.max_grad_norm and self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(ta.model.parameters(), self.max_grad_norm)

            ta.optimizer.step()
            cleaned = self._scrub_model_params(ta.model)
            if cleaned > 0:
                console.warn(f"[Trainer] Scrubbed {cleaned} non-finite parameter values post-update.")

            running_loss += float(loss.item())
            total_batch += 1

            avg_display = running_loss / max(total_batch, 1)
            loop.set_postfix(
                batch=total_batch,
                loss=f"{loss.item():.4f}",
                avg_loss=f"{avg_display:.4f}",
                lr=ta.optimizer.param_groups[0]["lr"]
            )

        avg_loss = running_loss / max(total_batch, 1)

        from tqdm.auto import tqdm as _tqdm
        _tqdm.write(
            f"[Epoch {self._epoch_idx}{'/' + str(total_epochs) if total_epochs else ''} Finished] "
            f"avg_loss={avg_loss:.6f} | batches={total_batch} | device={ta.device}"
        )
        return avg_loss

    def train(self, epochs) -> Any:
        self.trainer_args.total_epochs = epochs

        self._epoch_idx = 0

        train_stats = {"train_loss_sum": 0, "epoch_loss": [], "train_loss_power_two_sum": 0}
        for _ in range(epochs):
            self._epoch_idx += 1
            train_loss = self.train_step()
            train_stats["train_loss_sum"] += train_loss
            train_stats["train_loss_power_two_sum"] += train_loss ** 2
            train_stats["epoch_loss"].append(train_loss)

        self._epoch_idx = 0
        train_stats["avg_loss"] = train_stats["train_loss_sum"] / epochs
        train_stats["sqrt_train_loss_power_two_sum"] = math.sqrt(train_stats["train_loss_power_two_sum"])

        return self.trainer_args.model.state_dict(), train_stats

    def observe(self, epochs=5) -> Any:
        self.trainer_args.total_epochs = epochs
        train_stats = {"train_loss_sum": 0, "epoch_loss": [], "train_loss_power_two_sum": 0}

        for _ in range(epochs):
            train_loss = self.train_step()
            train_stats["train_loss_sum"] += train_loss
            train_stats["train_loss_power_two_sum"] += train_loss ** 2
            train_stats["epoch_loss"].append(train_loss)

        train_stats["avg_loss"] = train_stats["train_loss_sum"] / epochs
        train_stats["sqrt_train_loss_power_two_sum"] = math.sqrt(train_stats["train_loss_power_two_sum"])
        return self.trainer_args.model.state_dict(), train_stats

    # ---------- internal helpers ----------
    @staticmethod
    def _grads_are_finite(model: nn.Module) -> bool:
        for param in model.parameters():
            grad = param.grad
            if grad is None:
                continue
            if (grad.is_floating_point() or grad.is_complex()) and not torch.isfinite(grad).all().item():
                return False
        return True

    @staticmethod
    def _scrub_model_params(model: nn.Module) -> int:
        replaced = 0
        for param in model.parameters():
            if not (param.is_floating_point() or param.is_complex()):
                continue
            finite_mask = torch.isfinite(param)
            if finite_mask.all().item():
                continue
            replaced += int((~finite_mask).sum().item())
            torch.nan_to_num_(param, nan=0.0, posinf=0.0, neginf=0.0)
        return replaced
