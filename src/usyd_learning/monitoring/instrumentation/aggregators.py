from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from ...ml_utils import console
from ..hub import get_hub


def _prefix_from_key(k: str, suffixes: tuple[str, str] = ("lora_A", "lora_B")) -> str | None:
    if k.endswith(suffixes[0]) or k.endswith(suffixes[1]):
        return k.rsplit(".", 1)[0]
    # tolerant: accept keys like '*.lora_A.default'
    if ".lora_A" in k:
        return k.split(".lora_A", 1)[0]
    if ".lora_B" in k:
        return k.split(".lora_B", 1)[0]
    return None


def _collect_lora_dims(agg_sd: Dict[str, torch.Tensor]) -> Dict[str, Tuple[int, int, int]]:
    """Return mapping prefix -> (m, n, r_max) for LoRA A/B in aggregated state_dict."""
    A: Dict[str, torch.Tensor] = {}
    B: Dict[str, torch.Tensor] = {}
    for k, t in agg_sd.items():
        if not torch.is_tensor(t):
            continue
        pref = _prefix_from_key(k)
        if pref is None:
            continue
        if k.endswith("lora_A") or ".lora_A" in k:
            A[pref] = t
        elif k.endswith("lora_B") or ".lora_B" in k:
            B[pref] = t

    dims: Dict[str, Tuple[int, int, int]] = {}
    prefs = set(A.keys()) | set(B.keys())
    for p in prefs:
        a = A.get(p, None)
        b = B.get(p, None)
        try:
            if a is not None and b is not None:
                # a: [r, in], b: [out, r]
                r = int(a.shape[0])
                m = int(b.shape[0])
                n = int(a.shape[1])
            elif a is not None:
                r = int(a.shape[0])
                m = -1
                n = int(a.shape[1])
            elif b is not None:
                r = int(b.shape[1])
                m = int(b.shape[0])
                n = -1
            else:
                continue
            dims[p] = (m, n, r)
        except Exception:
            continue
    return dims


def _collect_sp_dims(agg_sd: Dict[str, torch.Tensor]) -> Dict[str, Tuple[int, int]]:
    """For SP aggregator outputs with keys '<prefix>.sp_aggregated', return prefix->(m,n)."""
    out: Dict[str, Tuple[int, int]] = {}
    for k, t in agg_sd.items():
        if not torch.is_tensor(t):
            continue
        if k.endswith(".sp_aggregated"):
            try:
                m, n = int(t.shape[0]), int(t.shape[1])
                out[k[:-len(".sp_aggregated")]] = (m, n)
            except Exception:
                pass
    return out


def _try_compute_fedavg_l2(server_node, agg_sd: Dict[str, torch.Tensor], layer: str) -> float | None:
    """Compute mean L2 norm between per-client A matrices and aggregated A for a given prefix.
    Returns None on any failure. Only considers '<prefix>.lora_A' to keep it lightweight.
    """
    try:
        keyA = f"{layer}.lora_A"
        if keyA not in agg_sd:
            return None
        Aagg = agg_sd[keyA]
        updates = getattr(server_node.node_var, "client_updates", [])
        if not isinstance(updates, list) or len(updates) == 0:
            return None
        diffs: List[float] = []
        for it in updates:
            sd = it.get("updated_weights", {}) if isinstance(it, dict) else None
            if not isinstance(sd, dict) or keyA not in sd:
                continue
            Ai = sd[keyA]
            if not (torch.is_tensor(Ai) and torch.is_tensor(Aagg)):
                continue
            if Ai.shape != Aagg.shape:
                # skip on mismatch (e.g., different ranks)
                continue
            diffs.append(float(torch.linalg.norm((Ai - Aagg).reshape(-1), ord=2).item()))
        if not diffs:
            return None
        return float(sum(diffs) / len(diffs))
    except Exception:
        return None


def record_aggregation_details(server_node, method_name: str) -> None:
    """Compute per-layer summaries after aggregation and write to sinks.

    For LoRA-based methods (fedavg | svd | rbla | zeropad): writes rows with
      m, n, r_max (when inferable) per prefix.
    For sum-product: writes m, n for '<prefix>.sp_aggregated'.
    """
    hub = get_hub()
    cfg = hub.config.raw.get("monitoring", {}).get("aggregation", {}) if hub else {}
    enabled = bool(cfg.get(method_name, {}).get("enabled", True)) if isinstance(cfg, dict) else True
    if not enabled:
        return

    try:
        # Aggregated state_dict
        agg_sd = getattr(server_node.node_var, "aggregated_weight", None)
        if not isinstance(agg_sd, dict):
            return
        # Round index
        round_idx = hub.current_round.round_index if (hub and hub.current_round) else -1
        run_id = hub.run_id if hub else ""

        if method_name in ("fedavg", "svd", "rbla", "zeropad"):
            dims = _collect_lora_dims(agg_sd)
            for layer, (m, n, r) in dims.items():
                row = {
                    "ts": int(__import__("time").time()),
                    "run_id": run_id,
                    "round": round_idx,
                    "method": method_name,
                    "layer": layer,
                    "m": m,
                    "n": n,
                    "r_max": r,
                }
                # Optional: FedAvg delta L2 sampling
                if method_name == "fedavg":
                    l2 = _try_compute_fedavg_l2(server_node, agg_sd, layer)
                    if l2 is not None:
                        row["delta_l2"] = l2
                hub.write_aggregation_layer(row)
        elif method_name in ("sp",):
            dims = _collect_sp_dims(agg_sd)
            for layer, (m, n) in dims.items():
                hub.write_aggregation_layer(
                    {
                        "ts": int(__import__("time").time()),
                        "run_id": run_id,
                        "round": round_idx,
                        "method": method_name,
                        "layer": layer,
                        "m": m,
                        "n": n,
                        "r_max": -1,
                    }
                )
        else:
            # Unknown method -> skip
            return
    except Exception as e:
        console.warn(f"[monitor] agg-layer record error: {e}")
