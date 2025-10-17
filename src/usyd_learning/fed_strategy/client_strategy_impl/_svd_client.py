import copy
from typing import Any, Tuple

import torch
import torch.nn as nn

from usyd_learning.fed_strategy.strategy_args import StrategyArgs
from ..client_strategy import ClientStrategy
from ...ml_utils.model_utils import ModelUtils
from ...model_trainer import model_trainer_factory
from ...model_trainer.model_trainer_args import ModelTrainerArgs
from ...model_trainer.model_trainer_factory import ModelTrainerFactory
from ...ml_algorithms.optimizer_builder import OptimizerBuilder
from ...ml_data_loader import DatasetLoaderFactory
from ...ml_algorithms.loss_function_builder import LossFunctionBuilder
from ...ml_utils import console
from ...fed_node.fed_node_vars import FedNodeVars


class SvdClientTrainingStrategy(ClientStrategy):
    """
    Client-side training strategy for SVD aggregation.

    Mirrors RBLA client strategy structure but without importing RBLA code.
    Implements its own broadcast slice/pad for LoRA A/B to match local ranks.
    """

    def __init__(self, args: StrategyArgs, client_node) -> None:
        super().__init__()
        self._args = args
        self._strategy_type = "svd"
        self._obj = client_node

        # Stable seeds for reproducibility (same as RBLA client)
        import random
        import numpy as np
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

    def _create_inner(self, args, client_node) -> None:
        self._args = args
        self._strategy_type = "svd"
        self._obj = client_node
        return

    # ------------------- Public: Observation wrapper -------------------
    def run_observation(self) -> dict:
        print(f"\n Observation Client [{self._obj.node_id}] ...\n")
        _, train_record = self.observation_step()
        return {
            "node_id": self._obj.node_id,
            "train_record": train_record,
            "data_sample_num": self._obj.node_var.data_sample_num,
        }

    # ------------------- Observation (no state write-back) -------------------
    def observation_step(self) -> Tuple[dict, Any]:
        node_vars: FedNodeVars = self._obj.node_var
        cfg: dict = self._obj.node_var.config_dict
        device = node_vars.device if hasattr(node_vars, "device") and node_vars.device else "cpu"

        observe_model: nn.Module = copy.deepcopy(node_vars.model)
        observe_model.load_state_dict(node_vars.model_weight, strict=True)
        optimizer = self._obj.node_var.optimizer_builder.rebuild(self, observe_model.parameters())

        ModelUtils.clear_all(observe_model, optimizer)

        self._obj.node_var.trainer.set_optimizer(optimizer)

        local_epochs = int(cfg.get("training", {}).get("local_epochs", 1))
        updated_weights, train_record = self.trainer.train(local_epochs)

        return copy.deepcopy(updated_weights), train_record

    # ------------------- Public: Local training wrapper -------------------
    def run_local_training(self) -> dict:
        updated_weights, train_record = self.local_training_step()
        return updated_weights, {
            "node_id": self._obj.node_id,
            "updated_weights": updated_weights,
            "train_record": train_record,
            "data_sample_num": self._obj.node_var.data_sample_num,
        }

    # ------------------- Full local training (write-back to node_var) -------------------
    def local_training_step(self) -> Tuple[dict, Any]:
        node_vars: FedNodeVars = self._obj.node_var
        cfg: dict = node_vars.config_dict
        device = getattr(node_vars, "device", None) or "cpu"

        training_model: nn.Module = copy.deepcopy(node_vars.model).to(device)
        training_model.load_state_dict(node_vars.model_weight, strict=True)

        optimizer = node_vars.optimizer_builder.rebuild(training_model.parameters())

        ModelUtils.clear_all(training_model, optimizer)

        tr = node_vars.trainer
        tr.set_model(training_model)
        tr.set_optimizer(optimizer)
        tr.trainer_args.device = device

        local_epochs = int(cfg.get("training", {}).get("epochs", 1))
        updated_weights, train_record = tr.train(local_epochs)

        node_vars.model_weight = copy.deepcopy(updated_weights)
        return copy.deepcopy(updated_weights), train_record

    def receive_weight(self, global_weight) -> dict:
        self._obj.node_var.cache_weight = global_weight

    @staticmethod
    def _broadcast_lora_state_dict(global_sd: dict, local_sd: dict, lora_suffixes: set[str] = {"lora_A", "lora_B"}) -> dict:
        """
        Slice or pad global LoRA matrices back to each client's local rank, copy non-LoRA tensors directly.
        Implemented here to avoid importing RBLA code.
        """
        new_local_sd = {}
        for key, local_tensor in local_sd.items():
            if key not in global_sd:
                new_local_sd[key] = local_tensor.clone() if torch.is_tensor(local_tensor) else local_tensor
                continue

            global_tensor = global_sd[key]

            # Robust suffix detection: accept keys like '*.lora_A.default'
            raw_suffix = key.rsplit(".", 1)[-1]
            suffix = raw_suffix
            if raw_suffix not in lora_suffixes:
                if ".lora_A" in key:
                    suffix = "lora_A"
                elif ".lora_B" in key:
                    suffix = "lora_B"

            if suffix not in lora_suffixes:
                new_local_sd[key] = global_tensor.clone() if torch.is_tensor(global_tensor) else global_tensor
            else:
                if suffix == "lora_A":  # [r, in]
                    r_local = local_tensor.shape[0]
                    r_global = global_tensor.shape[0]
                    if r_global >= r_local:
                        new_local_sd[key] = global_tensor[:r_local, :].clone()
                    else:
                        pad = torch.zeros((r_local, global_tensor.shape[1]), dtype=global_tensor.dtype, device=global_tensor.device)
                        pad[:r_global, :] = global_tensor
                        new_local_sd[key] = pad
                elif suffix == "lora_B":  # [out, r]
                    r_local = local_tensor.shape[1]
                    r_global = global_tensor.shape[1]
                    if r_global >= r_local:
                        new_local_sd[key] = global_tensor[:, :r_local].clone()
                    else:
                        pad = torch.zeros((global_tensor.shape[0], r_local), dtype=global_tensor.dtype, device=global_tensor.device)
                        pad[:, :r_global] = global_tensor
                        new_local_sd[key] = pad
                else:
                    new_local_sd[key] = global_tensor.clone() if torch.is_tensor(global_tensor) else global_tensor
        return new_local_sd

    def set_local_weight(self) -> dict:
        # Slice/pad global LoRA A/B to local ranks using neutral utility in LoRAUtils.
        from usyd_learning.ml_algorithms.lora.lora_utils import LoRAUtils
        new_local = LoRAUtils.broadcast_lora_state_dict(
            self._obj.node_var.cache_weight,
            self._obj.node_var.model_weight,
        )

        # If AdaLoRA is used locally, compensate for alpha/r scaling so that
        # forward-time scaling restores the aggregated delta magnitude.
        try:
            trainer_cfg = self._obj.node_var.config_dict.get("trainer", {})
            if str(trainer_cfg.get("trainer_type", "")).lower() == "adalora":
                alpha = float(trainer_cfg.get("adalora", {}).get("lora_alpha", 1.0))
                new_local = LoRAUtils.compensate_for_adalora_scaling(new_local, lora_alpha=alpha)
        except Exception:
            # Best-effort compensation; ignore errors to avoid blocking broadcast
            pass

        self._obj.node_var.model_weight = new_local
        return
