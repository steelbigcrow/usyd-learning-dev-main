from __future__ import annotations

import time
from typing import Any, List

import torch

from ...ml_utils import console
from ..hub import get_hub
from .aggregators import record_aggregation_details


def _is_tensor(x: Any) -> bool:
    return isinstance(x, torch.Tensor)


def _sd_num_bytes(sd: dict) -> int:
    total = 0
    for v in sd.values():
        if _is_tensor(v):
            try:
                total += int(v.numel()) * int(v.element_size())
            except Exception:
                pass
    return total


def _patch_method(cls, name: str, wrapper) -> None:
    orig = getattr(cls, name, None)
    if orig is None:
        return
    # Avoid double-patching
    if getattr(orig, "__monitor_patched__", False):
        return

    def _wrapped(self, *args, **kwargs):
        return wrapper(self, orig, *args, **kwargs)

    setattr(_wrapped, "__monitor_patched__", True)
    setattr(cls, name, _wrapped)


def patch_fed_server() -> None:
    """Patch FedNodeServer methods to emit monitoring events."""
    try:
        from ...fed_node.fed_node_server import FedNodeServer
    except Exception as e:
        console.warn(f"[monitor] skip server patch (import error): {e}")
        return

    # ---- select_clients: begin round ----
    def _select_wrapper(self: "FedNodeServer", orig, available_clients):
        hub = get_hub()
        hub.next_round()
        out = orig(self, available_clients)
        try:
            ids = [str(x) for x in out]
            hub.set_selected_clients(ids, total_clients=len(getattr(self, "client_nodes", []) or available_clients))
        except Exception:
            pass
        return out

    _patch_method(FedNodeServer, "select_clients", _select_wrapper)

    # ---- receive_client_updates: track payload in + samples ----
    def _recv_updates_wrapper(self: "FedNodeServer", orig, client_updates):
        t0 = time.perf_counter()
        out = orig(self, client_updates)
        try:
            total_bytes, total_samples = 0, 0
            for item in (client_updates or []):
                sd = item.get("updated_weights", {}) if isinstance(item, dict) else None
                total_bytes += _sd_num_bytes(sd) if isinstance(sd, dict) else 0
                rec = item.get("train_record", {}) if isinstance(item, dict) else {}
                total_samples += int(rec.get("data_sample_num", 0)) if isinstance(rec, dict) else 0
            hub = get_hub()
            hub.add_payload_in(total_bytes, total_samples)
        except Exception:
            pass
        return out

    _patch_method(FedNodeServer, "receive_client_updates", _recv_updates_wrapper)

    # ---- aggregation timing & aggregator name ----
    def _agg_wrapper(self: "FedNodeServer", orig, *args, **kwargs):
        hub = get_hub()
        try:
            agg_obj = getattr(self.node_var, "aggregation_method", None)
            agg_name = getattr(agg_obj, "_aggregation_method", None) or agg_obj.__class__.__name__ if agg_obj else None
            if agg_name:
                hub.set_aggregator(str(agg_name))
        except Exception:
            pass
        hub.agg_begin()
        try:
            ret = orig(self, *args, **kwargs)
            # After aggregation completes, record per-layer summaries when applicable
            try:
                method = str(getattr(getattr(self.node_var, "aggregation_method", None), "_aggregation_method", "")).lower()
                if not method:
                    method = str(getattr(getattr(self.node_var, "aggregation_method", None), "__class__", type("x", (), {})()).__name__).lower()
                record_aggregation_details(self, method)
            except Exception:
                pass
            return ret
        finally:
            hub.agg_end()

    _patch_method(FedNodeServer, "aggregation", _agg_wrapper)

    # ---- apply_weight timing ----
    def _apply_wrapper(self: "FedNodeServer", orig, *args, **kwargs):
        hub = get_hub()
        hub.apply_begin()
        try:
            return orig(self, *args, **kwargs)
        finally:
            hub.apply_end()

    _patch_method(FedNodeServer, "apply_weight", _apply_wrapper)

    # ---- broadcast timing & payload out ----
    def _broadcast_wrapper(self: "FedNodeServer", orig, *args, **kwargs):
        hub = get_hub()
        # Estimate payload bytes per client before broadcast
        try:
            sd = getattr(self.node_var, "model_weight", None)
            per_client = _sd_num_bytes(sd) if isinstance(sd, dict) else 0
            n_clients = len(getattr(self, "client_nodes", []) or [])
            hub.set_broadcast_bytes(per_client * n_clients)
        except Exception:
            pass
        hub.bcast_begin()
        try:
            return orig(self, *args, **kwargs)
        finally:
            hub.bcast_end()

    _patch_method(FedNodeServer, "broadcast", _broadcast_wrapper)

    # ---- evaluate timing and metrics capture ----
    def _eval_wrapper(self: "FedNodeServer", orig, *args, **kwargs):
        hub = get_hub()
        hub.eval_begin()
        try:
            ret = orig(self, *args, **kwargs)
            metrics = getattr(self, "eval_results", None)
            if not isinstance(metrics, dict):
                metrics = None
            hub.eval_end(metrics)
            return ret
        except Exception:
            hub.eval_end(None)
            raise

    _patch_method(FedNodeServer, "evaluate", _eval_wrapper)

    # ---- finalize round on record_evaluation ----
    def _record_eval_wrapper(self: "FedNodeServer", orig, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        try:
            get_hub().finalize_round()
        except Exception:
            pass
        return out

    _patch_method(FedNodeServer, "record_evaluation", _record_eval_wrapper)


def patch_client_training() -> None:
    """Patch client strategies' local training to emit per-epoch metrics."""
    # Attempt to import all known client strategy classes
    client_classes: List[type] = []
    try:
        from ...fed_strategy.client_strategy_impl._fedavg_client import (
            FedAvgClientTrainingStrategy,
        )

        client_classes.append(FedAvgClientTrainingStrategy)
    except Exception:
        pass
    try:
        from ...fed_strategy.client_strategy_impl._rbla_client import (
            RblaClientTrainingStrategy,
        )

        client_classes.append(RblaClientTrainingStrategy)
    except Exception:
        pass
    try:
        from ...fed_strategy.client_strategy_impl._svd_client import (
            SvdClientTrainingStrategy,
        )

        client_classes.append(SvdClientTrainingStrategy)
    except Exception:
        pass
    try:
        from ...fed_strategy.client_strategy_impl._sp_client import (
            SpClientTrainingStrategy,
        )

        client_classes.append(SpClientTrainingStrategy)
    except Exception:
        pass

    def _local_training_step_wrapper(self, orig, *args, **kwargs):
        # Resolve client id and samples
        try:
            node = getattr(self, "_obj", None)
            client_id = getattr(node, "node_id", None) or str(node)
            node_vars = getattr(node, "node_var", None)
            cfg = getattr(node_vars, "config_dict", {}) if node_vars else {}
            local_epochs = int(cfg.get("training", {}).get("epochs", 1))
            data_samples = int(getattr(node_vars, "data_sample_num", 0)) if node_vars else 0
        except Exception:
            client_id, local_epochs, data_samples = ("?", 0, 0)

        hub = get_hub()
        # Set current client context for downstream (e.g., AdaLoRA rank snapshot)
        try:
            hub._current_client_id = str(client_id)
        except Exception:
            pass
        hub.client_train_begin(str(client_id), local_epochs, data_samples)
        updated_weights, train_record = orig(self, *args, **kwargs)
        try:
            rec = train_record if isinstance(train_record, dict) else {}
            hub.client_train_end(str(client_id), rec)
        except Exception:
            pass
        finally:
            try:
                hub._current_client_id = None
            except Exception:
                pass
        return updated_weights, train_record

    for cls in client_classes:
        _patch_method(cls, "local_training_step", _local_training_step_wrapper)


def patch_prepare_events() -> None:
    """Patch FedNodeVars.raise_event to echo prepare events to console (low overhead)."""
    try:
        from ...fed_node.fed_node_vars import FedNodeVars
    except Exception as e:
        console.warn(f"[monitor] skip prepare-events patch (import error): {e}")
        return

    def _raise_event_wrapper(self: "FedNodeVars", orig, event_name: str, args):
        # print prepare stage events compactly
        try:
            if isinstance(event_name, str) and event_name.startswith("on_prepare_"):
                console.info(f"[monitor] event: {event_name}")
        except Exception:
            pass
        return orig(self, event_name, args)

    _patch_method(FedNodeVars, "raise_event", _raise_event_wrapper)
