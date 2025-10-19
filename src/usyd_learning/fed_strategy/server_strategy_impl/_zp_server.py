from __future__ import annotations
from typing import Dict, Any

import torch
from usyd_learning.fed_strategy.server_strategy import ServerStrategy
from usyd_learning.fl_algorithms.aggregation.fed_aggregator_facotry import FedAggregatorFactory
from usyd_learning.model_trainer.model_evaluator import ModelEvaluator
from usyd_learning.ml_utils import console
from usyd_learning.ml_algorithms.lora.lora_utils import LoRAUtils


class ZpServerStrategy(ServerStrategy):
    """
    Server strategy for Zero-Pad aggregation.

    Flow (parallels SVD/RBLA server structure but decoupled from RBLA):
      - Convert client PEFT/AdaLoRA updates to plain LoRA (shrunk) prior to aggregation.
      - Run ZP aggregator (zero-padding across clients) on plain LoRA A/B.
      - Map back onto a PEFT-shaped template for storage/broadcast.
      - Apply to evaluator by mapping PEFT->plain aligned to evaluator keys, then slice/pad
        LoRA A/B per local rank using LoRAUtils.broadcast_lora_state_dict.
    """

    def __init__(self, args, server_node) -> None:
        super().__init__()
        self._args = args
        self._strategy_type = "zp"
        self._obj = server_node

    def _create_inner(self, args, server_node) -> None:
        self._args = args
        self._strategy_type = "zp"
        self._obj = server_node
        return self

    def aggregation(self) -> dict:
        # Convert PEFT/AdaLoRA -> plain LoRA for robust aggregation
        from ...ml_algorithms.adalora.adalora_rbla_bridge import (
            peft_to_plain_lora_shrunk,
            plain_lora_to_peft,
            select_template_with_max_rank,
        )

        client_updates = self._obj.node_var.client_updates

        preprocessed = []
        for item in client_updates:
            sd_peft = item["updated_weights"]
            sd_plain = peft_to_plain_lora_shrunk(sd_peft)
            preprocessed.append({
                "updated_weights": sd_plain,
                "train_record": item["train_record"],
            })

        # Aggregate using Zero-Pad on plain-LoRA keys
        aggregator = self._obj.node_var.aggregation_method
        aggregated_plain = aggregator.aggregate(preprocessed)

        # Map aggregated plain-LoRA back to a PEFT-shaped state_dict using a template
        template_sd = select_template_with_max_rank(client_updates)
        aggregated_peft = plain_lora_to_peft(aggregated_plain, template_sd)

        # Keep the aggregated weight for broadcasting
        self._obj.node_var.aggregated_weight = aggregated_peft
        # Also make it the model_weight so broadcast() sends the latest global weight
        self._obj.node_var.model_weight = aggregated_peft
        return

    def select_clients(self, available_clients) -> list:
        selector = self._obj.node_var.client_selection
        selected_clients = selector.select(available_clients, self._obj.node_var.config_dict["client_selection"]["number"])
        return selected_clients

    def record_evaluation(self) -> None:
        self._obj.node_var.training_logger.record(self._obj.eval_results)
        return

    def receive_client_updates(self, client_updates) -> None:
        self._obj.node_var.client_updates = client_updates

    def apply_weight(self):
        # Aggregated weight is kept in PEFT shape; adapt to evaluator model (plain LoRA keys)
        try:
            evaluator: ModelEvaluator = self._obj.node_var.model_evaluator
            target_sd = evaluator.model.state_dict()

            # Map PEFT keys -> plain LoRA keys aligned to target key names
            plain_mapped = LoRAUtils.map_peft_to_lora_state_dict(
                target_state_dict=target_sd,
                peft_state_dict=self._obj.node_var.aggregated_weight,
            )

            # Ensure LoRA A/B shapes match evaluator's ranks (slice/pad as needed)
            adapted_local = LoRAUtils.broadcast_lora_state_dict(
                global_sd=plain_mapped,
                local_sd=target_sd,
            )

            # If using AdaLoRA trainer, compensate for alpha/r runtime scaling so that
            # the effective delta applied at inference equals the aggregated W_g.
            try:
                trainer_cfg = self._obj.node_var.config_dict.get("trainer", {})
                if str(trainer_cfg.get("trainer_type", "")).lower() == "adalora":
                    alpha = float(trainer_cfg.get("adalora", {}).get("lora_alpha", 1.0))
                    adapted_local = LoRAUtils.compensate_for_adalora_scaling(
                        adapted_local,
                        lora_alpha=alpha,
                    )
            except Exception:
                # Best-effort; do not block apply_weight on compensation errors
                pass

            evaluator.update_model(adapted_local)
        except Exception as e:
            raise e
        return

    def broadcast(self) -> None:
        for client in self._obj.client_nodes:
            client.receive_weight(self._obj.node_var.model_weight)
            client.set_local_weight()
        return

    def run(self) -> Dict[str, Any]:
        raise NotImplementedError

    def evaluate(self) -> None:
        self._obj.eval_results = self._obj.node_var.model_evaluator.evaluate()
        self._obj.node_var.model_evaluator.print_results()
        console.info("Server Evaluation Completed.\n")
        return

    def prepare(self, logger_header, client_nodes_in) -> None:
        self._obj.node_var.training_logger.begin(logger_header)
        self._obj.set_client_nodes(client_nodes_in)
        return

