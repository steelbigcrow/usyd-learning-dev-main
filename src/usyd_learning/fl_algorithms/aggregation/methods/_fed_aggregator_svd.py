from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, List, Tuple, Any

import torch

from ..fed_aggregator_abc import AbstractFedAggregator
from ..fed_aggregator_args import FedAggregatorArgs
from ....ml_utils import console
from ....ml_algorithms.lora.lora_utils import LoRAUtils


class FedAggregator_SVD(AbstractFedAggregator):
    """
    SVD-based aggregation (full split) for LoRA updates.

    Algorithm per-layer (prefix):
      1) For each client i with LoRA pair (A_i in R^{r_i×in}, B_i in R^{out×r_i}),
         form delta W_i = B_i @ A_i.
      2) Weighted average: W_g = (Σ_i w_i·(B_i@A_i)) / (Σ_i w_i).
      3) Compute truncated SVD of W_g at r_max = max_i r_i: W_g ≈ U[:, :r_max] Σ[:r_max,:r_max] V[:r_max, :].
      4) Full split to LoRA A/B:
            B_max = U[:, :r_max] @ Σ[:r_max, :r_max]   # [out, r_max]
            A_max = V[:r_max, :]                       # [r_max, in]

    Output is a plain LoRA state_dict only containing keys:
        '{prefix}.lora_A' = A_max, '{prefix}.lora_B' = B_max

    Notes:
      - Supports mixed ranks across clients (AdaLoRA) by operating on delta matrices.
      - If inputs are PEFT/AdaLoRA-shaped keys, they are converted to plain-LoRA
        using adalora bridge before aggregation.
    """

    def __init__(self, args: FedAggregatorArgs | None = None) -> None:
        super().__init__(args)
        self._aggregation_method = "svd"
        self._suffix_A = "lora_A"
        self._suffix_B = "lora_B"
        self._normalize_weights: bool = True
        # Force 'full' split per requirement
        # Use balanced factorization to keep A/B magnitudes stable for training
        # "sqrt":  B = U * sqrt(S),  A = sqrt(S) * V^T
        # This preserves W=B@A while avoiding extremely imbalanced factors that can hinder convergence.
        self._svd_method: str = "sqrt"

    # ---------- FedAvg-style data building (kept for API symmetry) ----------
    def build_data_list(self, aggregation_data_dict: dict) -> None:
        self._aggregation_data_list = list(aggregation_data_dict.values())

    def build_data_dict(self, aggregation_data_dict: Any) -> None:
        self._aggregation_data_dict = aggregation_data_dict

    # ---------- Helpers ----------
    @staticmethod
    def _suffix_of(key: str) -> str:
        return key.rsplit(".", 1)[-1]

    @staticmethod
    def _prefix_of(key: str) -> str:
        return key.rsplit(".", 1)[0]

    def _collect_inputs(self) -> Tuple[List[Dict[str, torch.Tensor]], List[float]]:
        """
        Collect (state_dicts, weights). Accepts
          - self._aggregation_data_list: [(state_dict, weight), ...]
          - self._aggregation_data_dict: same as list or legacy dict {'state_dicts': [...], 'weights': [...]}
        """
        if hasattr(self, "_aggregation_data_list") and self._aggregation_data_list:
            pairs = self._aggregation_data_list
            sds = [sd for sd, vol in pairs]
            ws = [float(vol) for _, vol in pairs]
        elif hasattr(self, "_aggregation_data_dict") and isinstance(self._aggregation_data_dict, list):
            pairs = self._aggregation_data_dict
            sds = [sd for sd, vol in pairs]
            ws = [float(vol) for _, vol in pairs]
        else:
            agg = self._aggregation_data_dict
            sds = agg["state_dicts"]
            ws = agg.get("weights", [1.0] * len(sds))

        if len(sds) != len(ws):
            raise ValueError(f"Number of state_dicts and weights must match: {len(sds)} vs {len(ws)}")

        if self._normalize_weights:
            tw = float(sum(ws))
            ws = [w / tw for w in ws] if tw > 0 else [1.0 / len(ws)] * len(ws)

        return sds, ws

    @staticmethod
    def _maybe_convert_peft_to_plain(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Best-effort convert a PEFT/AdaLoRA-shaped state_dict to plain LoRA keys
        if no plain 'lora_A'/'lora_B' keys are found.
        """
        has_plain = any(k.endswith("lora_A") or k.endswith("lora_B") for k in sd.keys())
        if has_plain:
            return sd

        # Heuristic: detect presence of PEFT/AdaLoRA patterns
        peft_like = any(
            ("base_model.model" in k) or (".lora_A." in k) or (".lora_B." in k)
            for k in sd.keys()
        )
        if not peft_like:
            return sd

        try:
            from ....ml_algorithms.adalora.adalora_rbla_bridge import peft_to_plain_lora_shrunk
            return peft_to_plain_lora_shrunk(sd)
        except Exception:
            # If conversion fails, fall back to original dict
            return sd

    # ---------- Aggregation core ----------
    def _before_aggregation(self) -> None:
        return

    def _do_aggregation(self) -> None:
        state_dicts, weights = self._collect_inputs()

        dev = self._device
        # Convert AdaLoRA/PEFT-shaped keys to plain LoRA where needed
        sds_plain: List[Dict[str, torch.Tensor]] = [self._maybe_convert_peft_to_plain(sd) for sd in state_dicts]
        # Move to device
        sds_on_device: List[Dict[str, torch.Tensor]] = [
            {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in sd.items()} for sd in sds_plain
        ]

        console.debug(f"\n[SVD] Aggregating {len(sds_on_device)} clients (full split)...")
        total_w = sum(weights)
        if total_w <= 0:
            raise ValueError("Sum of client weights must be positive.")

        # Gather sample keys to define deterministic base order
        sample_keys = list(sds_on_device[0].keys())

        # 1) Collect per-client LoRA A/B pairs per prefix
        #    Map: prefix -> list[(client_index, A_i, B_i)]
        pair_groups: Dict[str, List[Tuple[int, torch.Tensor, torch.Tensor]]] = {}
        for ci, sd in enumerate(sds_on_device):
            ab_per_client: Dict[str, Dict[str, torch.Tensor]] = {}
            for k, t in sd.items():
                if not torch.is_tensor(t):
                    continue
                suf = self._suffix_of(k)
                if suf == self._suffix_A or suf == self._suffix_B:
                    pref = self._prefix_of(k)
                    ab_per_client.setdefault(pref, {})[suf] = t
            for pref, d in ab_per_client.items():
                if self._suffix_A in d and self._suffix_B in d:
                    A_i = d[self._suffix_A]  # [r_i, in]
                    B_i = d[self._suffix_B]  # [out, r_i]
                    pair_groups.setdefault(pref, []).append((ci, A_i, B_i))

        # Also capture optional AdaLoRA rank masks if present for budget unification
        mask_groups: Dict[str, List[Tuple[int, torch.Tensor]]] = {}

        # 2) For each prefix, compute W_g and SVD split to rank r_max (balanced factors)
        aggregated: Dict[str, torch.Tensor] = {}
        for pref, entries in pair_groups.items():
            if not entries:
                continue

            # Infer output/input dims and r_max, handling both naming conventions:
            #   (A: [r, in], B: [out, r])  => W = B @ A
            #   (A: [out, r], B: [r, in])  => W = A @ B
            def infer_dims(A: torch.Tensor, B: torch.Tensor) -> Tuple[int, int, int, str]:
                # returns (m, n, r_i, mode)
                if A.dim() != 2 or B.dim() != 2:
                    raise ValueError(f"LoRA A/B must be 2D, got {A.dim()}D and {B.dim()}D")
                # mode1: A[r,in], B[out,r] -> W=B@A
                if A.shape[0] == B.shape[1]:
                    r_i = int(A.shape[0])
                    m, n = int(B.shape[0]), int(A.shape[1])
                    return m, n, r_i, "B@A"
                # mode2: A[out,r], B[r,in] -> W=A@B
                if A.shape[1] == B.shape[0]:
                    r_i = int(A.shape[1])
                    m, n = int(A.shape[0]), int(B.shape[1])
                    return m, n, r_i, "A@B"
                raise ValueError(
                    f"Incompatible LoRA shapes for prefix '{pref}': A{tuple(A.shape)} vs B{tuple(B.shape)}"
                )

            # Use first entry to deduce m, n, and collect per-client r_i + mode
            m, n, r0, mode0 = infer_dims(entries[0][1], entries[0][2])
            r_list: List[int] = [r0]
            modes: List[str] = [mode0]
            for _, A_i, B_i in entries[1:]:
                m_i, n_i, r_i, mode_i = infer_dims(A_i, B_i)
                if (m_i != m) or (n_i != n):
                    raise ValueError(
                        f"Mismatched inferred dims for '{pref}': (m,n)=({m},{n}) vs ({m_i},{n_i})"
                    )
                r_list.append(r_i)
                modes.append(mode_i)
            r_max = max(r_list)

            # Weighted average of delta matrices
            W_sum = torch.zeros((m, n), dtype=entries[0][1].dtype, device=dev)
            for ci, A_i, B_i in entries:
                w = float(weights[ci])
                # Choose multiplication order based on shapes
                if A_i.shape[0] == B_i.shape[1]:
                    # mode: W = B @ A (A: [r,in], B: [out,r])
                    W_delta = B_i @ A_i
                elif A_i.shape[1] == B_i.shape[0]:
                    # mode: W = A @ B (A: [out,r], B: [r,in])
                    W_delta = A_i @ B_i
                else:
                    raise ValueError(
                        f"Incompatible LoRA shapes for prefix '{pref}': A{tuple(A_i.shape)} vs B{tuple(B_i.shape)}"
                    )
                # Replace non-finite values to keep SVD stable
                W_delta = torch.nan_to_num(W_delta, nan=0.0, posinf=0.0, neginf=0.0)
                W_sum = W_sum + (w * W_delta)
            W_g = W_sum / total_w
            W_g = torch.nan_to_num(W_g, nan=0.0, posinf=0.0, neginf=0.0)

            # SVD split to rank r_max (clamped to min(m, n))
            A_max, B_max = LoRAUtils.svd_split(W_g, r=r_max, method=self._svd_method)
            # Place into output as plain-LoRA keys
            aggregated[f"{pref}.{self._suffix_A}"] = A_max
            aggregated[f"{pref}.{self._suffix_B}"] = B_max

        # 3) Aggregate AdaLoRA-style rank masks when available to help unify budgets
        #    Expect keys like '<prefix>.rank_mask' (vector length r_i). We average with weights
        #    and right-pad with zeros to r_max used above for each layer.
        #    First, collect masks per prefix from the original (plain) dicts to ensure we cover
        #    all prefixes present in A/B groups.
        if len(sds_on_device) > 0:
            # Build a quick lookup of r_max per prefix from aggregated A tensors
            rmax_by_pref: Dict[str, int] = {
                k[:-len(f".{self._suffix_A}")]: int(v.shape[0])
                for k, v in aggregated.items() if k.endswith(f".{self._suffix_A}") and torch.is_tensor(v)
            }

            # Collect masks
            for ci, sd in enumerate(sds_on_device):
                for k, t in sd.items():
                    if not isinstance(t, torch.Tensor):
                        continue
                    if k.endswith(".rank_mask"):
                        pref = k[: -len(".rank_mask")]
                        mask_groups.setdefault(pref, []).append((ci, t.detach()))

            # Weighted average and pad to r_max, then attach to output 'aggregated'
            for pref, items in mask_groups.items():
                if pref not in rmax_by_pref:
                    # Skip masks for layers that did not participate in aggregation
                    continue
                r_max = int(rmax_by_pref[pref])
                # Prepare accumulator
                acc = torch.zeros(r_max, dtype=torch.float32, device=dev)
                wsum = 0.0
                for ci, vec in items:
                    w = float(weights[ci])
                    v = vec.detach().to(device=dev, dtype=torch.float32).view(-1)
                    # pad or trim to r_max (right side) to align indices; unused entries remain zero
                    if v.numel() < r_max:
                        vv = torch.zeros(r_max, dtype=torch.float32, device=dev)
                        vv[: v.numel()] = v
                    else:
                        vv = v[:r_max]
                    acc += w * vv
                    wsum += w
                if wsum > 0:
                    acc = acc / wsum
                # Ensure numeric sanity
                acc = torch.nan_to_num(acc, nan=0.0, posinf=0.0, neginf=0.0)
                aggregated[f"{pref}.rank_mask"] = acc.to(dtype=aggregated[f"{pref}.{self._suffix_A}"].dtype)

        # Keep a deterministic order: non-LoRA keys from sample (skipped) + LoRA keys sorted
        ordered = OrderedDict()
        for k in sorted(aggregated.keys()):
            ordered[k] = aggregated[k]

        self._aggregated_weight = ordered

    def _after_aggregation(self) -> None:
        console.debug("[SVD] Aggregation completed.")
        return
