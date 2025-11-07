from __future__ import annotations
from typing import Dict, List, Any, Optional
from collections import defaultdict

import torch
from usyd_learning.fed_strategy.server_strategy import ServerStrategy
from usyd_learning.fl_algorithms.aggregation.fed_aggregator_facotry import FedAggregatorFactory
from usyd_learning.fl_algorithms.selection.fed_client_selector_factory import FedClientSelectorFactory
from usyd_learning.model_trainer.model_evaluator import ModelEvaluator
from usyd_learning.ml_utils import console
from usyd_learning.ml_algorithms.lora.lora_utils import LoRAUtils
from usyd_learning.fl_algorithms.aggregation.methods._fed_aggregator_rbla import (
    FedAggregator_RBLA,
)

class RblaServerStrategy(ServerStrategy):
    def __init__(self, args, server_node) -> None:
        super().__init__()
        self._args = args
        self._strategy_type = "rbla"
        self._obj = server_node        
        
    def _create_inner(self, args, server_node) -> None:
        self._args = args
        self._strategy_type = "rbla"
        self._obj = server_node
        return self

    def aggregation(self) -> dict:
        # Support both LoRA/AdaLoRA clients and plain non‑LoRA clients.
        client_updates = self._obj.node_var.client_updates

        def _is_lora_state_dict(sd: dict) -> bool:
            try:
                for k in sd.keys():
                    if (".lora_A" in k) or (".lora_B" in k):
                        return True
            except Exception:
                pass
            return False

        # Decide mode by checking the first client's keys
        is_lora = False
        if isinstance(client_updates, list) and len(client_updates) > 0:
            first_sd = client_updates[0]["updated_weights"]
            is_lora = _is_lora_state_dict(first_sd)

        aggregator = self._obj.node_var.aggregation_method

        if is_lora:
            # LoRA/AdaLoRA path: shrink to plain‑LoRA then aggregate
            from ...ml_algorithms.adalora.adalora_rbla_bridge import (
                peft_to_plain_lora_shrunk,
                plain_lora_to_peft,
                select_template_with_max_rank,
            )

            preprocessed = []
            for item in client_updates:
                sd_peft = item["updated_weights"]
                sd_plain = peft_to_plain_lora_shrunk(sd_peft)
                # For RBLA aggregation, keep only LoRA A/B keys to avoid stacking
                # variable-length auxiliary vectors (e.g., rank_mask, rank_rr).
                sd_plain = {k: v for k, v in sd_plain.items() if k.endswith(".lora_A") or k.endswith(".lora_B")}
                preprocessed.append({
                    "updated_weights": sd_plain,
                    "train_record": item["train_record"],
                })

            aggregated_plain = aggregator.aggregate(preprocessed)

            # Map aggregated plain‑LoRA back to PEFT‑shaped dict using template with max rank
            template_sd = select_template_with_max_rank(client_updates)
            aggregated = plain_lora_to_peft(aggregated_plain, template_sd)

        else:
            # Non‑LoRA path: aggregate dense state_dicts directly
            preprocessed = []
            for item in client_updates:
                preprocessed.append({
                    "updated_weights": item["updated_weights"],
                    "train_record": item["train_record"],
                })
            aggregated = aggregator.aggregate(preprocessed)

        # Keep the aggregated weight for broadcasting
        self._obj.node_var.aggregated_weight = aggregated
        return

    def select_clients(self, available_clients) -> list:
        selector = self._obj.node_var.client_selection
        selected_clients = selector.select(available_clients, self._obj.node_var.config_dict["client_selection"]["number"])
        return selected_clients

    def record_evaluation(self)-> None:
        self._obj.node_var.training_logger.record(self._obj.eval_results)
        return

    def receive_client_updates(self, client_updates) -> None:
        self._obj.node_var.client_updates = client_updates #{client1: {weight:"", data_vol:""}, client2: {weight:"", data_vol:""}}
    
    def apply_weight(self):
        # Broadcast weight to clients; if LoRA, adapt to evaluator; if non‑LoRA, load directly.
        self._obj.node_var.model_weight = self._obj.node_var.aggregated_weight

        evaluator: ModelEvaluator = self._obj.node_var.model_evaluator
        target_sd = evaluator.model.state_dict()

        # Detect LoRA keys in aggregated weight
        def _is_lora_state_dict(sd: dict) -> bool:
            try:
                for k in sd.keys():
                    if (".lora_A" in k) or (".lora_B" in k):
                        return True
            except Exception:
                pass
            return False

        if _is_lora_state_dict(self._obj.node_var.aggregated_weight):
            # Map PEFT keys -> plain LoRA keys aligned to target key names
            plain_mapped = LoRAUtils.map_peft_to_lora_state_dict(
                target_state_dict=target_sd,
                peft_state_dict=self._obj.node_var.aggregated_weight,
            )
            # Ensure LoRA A/B shapes match evaluator's ranks (slice/pad as needed)
            adapted_local = FedAggregator_RBLA.broadcast_lora_state_dict(
                global_sd=plain_mapped,
                local_sd=target_sd,
            )
            evaluator.update_model(adapted_local)
        else:
            # Plain non‑LoRA model: load the aggregated weights directly
            evaluator.update_model(self._obj.node_var.aggregated_weight)
        return

    def broadcast(self) -> None:
        for client in self._obj.client_nodes:
            client.receive_weight(self._obj.node_var.model_weight)
            client.set_local_weight()
            #client.node_var.model_weight = self._obj.node_var.model_weight
        return

    def run(self) -> None:
        raise NotImplementedError

    def evaluate(self) -> None:
        self._obj.eval_results =  self._obj.node_var.model_evaluator.evaluate()
        self._obj.node_var.model_evaluator.print_results()
        console.info("Server Evaluation Completed.\n")

        return

    def prepare(self, logger_header, client_nodes_in) -> None:
        self._obj.node_var.training_logger.begin(logger_header)
        self._obj.set_client_nodes(client_nodes_in)
        return

    def run(self) -> Dict[str, Any]:
        raise NotImplementedError
