from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import torch


def _strip_base_prefix(k: str, base_prefix: str = "base_model.model") -> str:
    if k.startswith(base_prefix + "."):
        return k[len(base_prefix) + 1 :]
    return k


def _find_peft_lora_prefixes(sd: Dict[str, torch.Tensor], base_prefix: str = "base_model.model") -> Dict[str, Dict[str, str]]:
    """
    Scan a PEFT/AdaLoRA state_dict and collect LoRA A/B parameter keys per logical prefix.

    Returns mapping: { prefix: { 'A': peft_key_for_A, 'B': peft_key_for_B } }
    Prefix is normalized by stripping base prefix and any trailing ".base_layer".
    """
    prefixes: Dict[str, Dict[str, str]] = {}

    for k in sd.keys():
        k_stripped = _strip_base_prefix(k, base_prefix)
        if ".lora_A" in k_stripped:
            p = k_stripped.split(".lora_A", 1)[0]
            if p.endswith(".base_layer"):
                p = p[: -len(".base_layer")]
            prefixes.setdefault(p, {})["A"] = k
        elif ".lora_B" in k_stripped:
            p = k_stripped.split(".lora_B", 1)[0]
            if p.endswith(".base_layer"):
                p = p[: -len(".base_layer")]
            prefixes.setdefault(p, {})["B"] = k
    return prefixes


def _find_lora_e_key(sd: Dict[str, torch.Tensor], prefix: str, base_prefix: str = "base_model.model") -> Optional[str]:
    """Best-effort locate the PEFT key for lora_E for a given logical prefix.

    Tries both '<base_prefix>.<prefix>.lora_E.default' and
    '<base_prefix>.<prefix>.base_layer.lora_E.default'.
    """
    # Common candidate paths used by PEFT AdaLoRA
    cand1 = f"{base_prefix}.{prefix}.lora_E.default"
    cand2 = f"{base_prefix}.{prefix}.base_layer.lora_E.default"
    if cand1 in sd:
        return cand1
    if cand2 in sd:
        return cand2
    # Relaxed scan as a fallback
    for k in sd.keys():
        ks = _strip_base_prefix(k, base_prefix)
        if ks.startswith(prefix + ".") and ".lora_E" in ks:
            return k
    return None


def _find_rankhint_key(sd: Dict[str, torch.Tensor], prefix: str, base_prefix: str = "base_model.model") -> Optional[str]:
    """Heuristically find a key that may encode rank hints (e.g., 'ranknum') for the given prefix."""
    candidates: List[str] = []
    match_prefix_1 = f"{base_prefix}.{prefix}.ranknum"
    match_prefix_2 = f"{base_prefix}.{prefix}.base_layer.ranknum"
    for k in sd.keys():
        if k == match_prefix_1 or k == match_prefix_2:
            return k
        # relaxed matching: any key that contains '<prefix>.*ranknum'
        ks = _strip_base_prefix(k, base_prefix)
        if ks.startswith(prefix + ".") and ks.endswith("ranknum"):
            candidates.append(k)
    return candidates[0] if candidates else None


def _choose_rr(A: torch.Tensor, B: torch.Tensor, rank_hint: Optional[torch.Tensor]) -> Tuple[int, torch.Tensor]:
    """
    Choose effective rank rr and a per-channel importance score vector for selection.
    If rank_hint is provided:
      - scalar: use round(scalar)
      - vector: use sum(vector) as rr and vector as importance scores
    Otherwise, fall back to average of per-channel norms from A rows and B columns.
    Returns (rr, scores) where scores.shape == [r].
    """
    r = A.shape[0]
    if rank_hint is not None and torch.is_tensor(rank_hint):
        v = rank_hint.detach().float().view(-1)
        if v.numel() == 1:
            rr = int(max(1, min(r, int(round(float(v.item()))))))
            # derive scores from norms if only scalar provided
            a_scores = A.detach().float().pow(2).sum(dim=1).sqrt()
            b_scores = B.detach().float().pow(2).sum(dim=0).sqrt()
            scores = 0.5 * (a_scores + b_scores)
            return rr, scores
        else:
            # use vector both for rr and scores
            rr = int(max(1, min(r, int(round(float(v.sum().item()))))))
            return rr, v

    # No hint available: use norms
    a_scores = A.detach().float().pow(2).sum(dim=1).sqrt()  # [r]
    b_scores = B.detach().float().pow(2).sum(dim=0).sqrt()  # [r]
    scores = 0.5 * (a_scores + b_scores)
    rr = r
    return rr, scores


def peft_to_plain_lora_shrunk(
    peft_sd: Dict[str, torch.Tensor],
    *,
    base_prefix: str = "base_model.model",
) -> Dict[str, torch.Tensor]:
    """
    Convert PEFT/AdaLoRA state_dict to a plain-LoRA dict containing only lora_A/lora_B per-layer,
    optionally shrunk to an effective rank inferred from rank hints.

    Output keys are of form '<prefix>.lora_A' and '<prefix>.lora_B'.
    Non-LoRA parameters are intentionally omitted, as LoRA training freezes them.
    """
    out: Dict[str, torch.Tensor] = {}
    prefixes = _find_peft_lora_prefixes(peft_sd, base_prefix=base_prefix)

    for prefix, pair in prefixes.items():
        if "A" not in pair or "B" not in pair:
            # incomplete pair; skip
            continue
        A_key = pair["A"]
        B_key = pair["B"]
        A = peft_sd[A_key]
        B = peft_sd[B_key]
        if A.dim() != 2 or B.dim() != 2:
            # Only support Linear-style LoRA matrices here
            continue

        # Prefer using lora_E as an explicit mask to infer effective rank
        E_key = _find_lora_e_key(peft_sd, prefix, base_prefix=base_prefix)
        mask_vec: Optional[torch.Tensor] = None
        if E_key is not None and torch.is_tensor(peft_sd[E_key]):
            E = peft_sd[E_key]
            # E is [r, 1]; derive a [r] boolean-ish mask (1.0 for selected, 0.0 otherwise)
            v = E.detach().float().view(-1)
            mask_vec = (v.abs() > 0.0).float()

        hint_key = _find_rankhint_key(peft_sd, prefix, base_prefix=base_prefix)
        hint = peft_sd[hint_key] if hint_key is not None else None

        # If we have an explicit mask from E, use it as the rank hint vector
        if mask_vec is not None:
            rr, scores = _choose_rr(A, B, mask_vec)
        else:
            rr, scores = _choose_rr(A, B, hint)

        r = A.shape[0]
        rr = max(1, min(rr, r))
        if rr < r:
            # select top-rr channels consistently for A(row) and B(col)
            idx = torch.topk(scores, k=rr, largest=True).indices.sort().values  # ascending order
            A_eff = A.index_select(dim=0, index=idx)
            B_eff = B.index_select(dim=1, index=idx)
        else:
            A_eff, B_eff = A, B

        out[f"{prefix}.lora_A"] = A_eff.clone().detach()
        out[f"{prefix}.lora_B"] = B_eff.clone().detach()

        # Also carry a mask signal for downstream unification (shape [r])
        # If lora_E was available, prefer its binary mask; otherwise, derive from scores top-rr
        if mask_vec is not None:
            out[f"{prefix}.rank_mask"] = mask_vec.clone().detach()
        else:
            # Build a mask by selecting the same top-rr indices used above
            m = torch.zeros(r, dtype=A.detach().dtype)
            if rr < r:
                idx = torch.topk(scores, k=rr, largest=True).indices.sort().values
                m.index_fill_(0, idx, 1.0)
            else:
                m.fill_(1.0)
            out[f"{prefix}.rank_mask"] = m

        # Carry a scalar rr for convenience (weighted-averaged during aggregation)
        out[f"{prefix}.rank_rr"] = torch.tensor(float(rr))

    return out


def plain_lora_to_peft(
    plain_sd: Dict[str, torch.Tensor],
    peft_template: Dict[str, torch.Tensor],
    *,
    base_prefix: str = "base_model.model",
) -> Dict[str, torch.Tensor]:
    """
    Map plain-LoRA '{prefix}.lora_A/B' tensors back onto a PEFT/AdaLoRA-shaped state_dict template.
    For LoRA keys, this function will overwrite template values with the provided tensors without enforcing shape match.
    For all other keys, the template values are preserved.
    """
    out: Dict[str, torch.Tensor] = {}

    # Build lookup for plain
    plain_A = {k[:-len(".lora_A")]: v for k, v in plain_sd.items() if k.endswith(".lora_A")}
    plain_B = {k[:-len(".lora_B")]: v for k, v in plain_sd.items() if k.endswith(".lora_B")}

    for k_t, v_t in peft_template.items():
        ks = _strip_base_prefix(k_t, base_prefix)
        written = False
        if ".lora_A" in ks:
            p = ks.split(".lora_A", 1)[0]
            if p.endswith(".base_layer"):
                p = p[: -len(".base_layer")]
            if p in plain_A:
                out[k_t] = plain_A[p]
                written = True
        elif ".lora_B" in ks:
            p = ks.split(".lora_B", 1)[0]
            if p.endswith(".base_layer"):
                p = p[: -len(".base_layer")]
            if p in plain_B:
                out[k_t] = plain_B[p]
                written = True

        if not written:
            out[k_t] = v_t.clone().detach() if torch.is_tensor(v_t) else v_t

    # --- Unify effective rank per layer using aggregated rank_mask (if present) ---
    # Build a prefix -> peft A/B key map from the partially filled 'out'
    prefixes = _find_peft_lora_prefixes(out, base_prefix=base_prefix)

    for prefix, pair in prefixes.items():
        # Gather unified budget from aggregated plain_sd if available
        mask_key = f"{prefix}.rank_mask"
        if mask_key not in plain_sd:
            continue  # nothing to unify for this prefix

        mask_vec = plain_sd[mask_key].detach().to(dtype=torch.float32).view(-1)
        r = int(mask_vec.numel())

        # Decide K as rounded sum of averaged mask (weights sum to 1 in aggregator)
        K = int(round(float(mask_vec.sum().item())))
        K = max(1, min(K, r))

        # Build unified mask as first-K channels selected to match broadcast slicing behavior
        uni_mask = torch.zeros(r, dtype=torch.float32, device=mask_vec.device)
        if K > 0:
            uni_mask[:K] = 1.0

        # Apply the unified mask to A/B (zero out masked-off channels)
        A_key = pair.get("A", None)
        B_key = pair.get("B", None)
        if A_key in out and torch.is_tensor(out[A_key]) and out[A_key].dim() == 2:
            A = out[A_key]
            ra = A.shape[0]
            # zero out rows beyond K in whatever aggregated shape we have
            if ra > K:
                mask_rows = torch.ones(ra, 1, dtype=A.dtype, device=A.device)
                mask_rows[K:, :] = 0.0
                out[A_key] = (A * mask_rows).clone().detach()
        if B_key in out and torch.is_tensor(out[B_key]) and out[B_key].dim() == 2:
            B = out[B_key]
            rb = B.shape[1]
            if rb > K:
                mask_cols = torch.ones(1, rb, dtype=B.dtype, device=B.device)
                mask_cols[:, K:] = 0.0
                out[B_key] = (B * mask_cols).clone().detach()

        # Update lora_E in the PEFT-shaped dict using the unified mask as a gate
        E_key = _find_lora_e_key(out, prefix, base_prefix=base_prefix)
        if E_key is not None and E_key in out and torch.is_tensor(out[E_key]):
            E = out[E_key]
            # E can be [r, 1]; broadcast mask
            m = uni_mask.view(-1, 1).to(dtype=E.dtype, device=E.device)
            out[E_key] = (E * 0.0 + 1.0) * m  # set selected entries to 1.0, others 0.0
        elif E_key is not None and E_key in peft_template and torch.is_tensor(peft_template[E_key]):
            E_t = peft_template[E_key]
            m = uni_mask.view(-1, 1).to(dtype=E_t.dtype, device=E_t.device)
            out[E_key] = (E_t * 0.0 + 1.0) * m

        # Update ranknum if available
        rk_key = _find_rankhint_key(out, prefix, base_prefix=base_prefix)
        if rk_key is not None and rk_key in out and torch.is_tensor(out[rk_key]):
            out[rk_key] = torch.tensor(float(K), dtype=out[rk_key].dtype, device=out[rk_key].device)
        elif rk_key is not None and rk_key in peft_template and torch.is_tensor(peft_template[rk_key]):
            out[rk_key] = torch.tensor(float(K), dtype=peft_template[rk_key].dtype, device=peft_template[rk_key].device)

    return out


def select_template_with_max_rank(
    client_updates: List[Dict],
    *,
    base_prefix: str = "base_model.model",
) -> Dict[str, torch.Tensor]:
    """
    Given a list of client update dicts (each having 'updated_weights'), pick the template state_dict
    from the client that appears to have the largest LoRA rank (by inspecting lora_A.* shape[0]).
    """
    best_sd = None
    best_r = -1
    for d in client_updates:
        sd = d["updated_weights"]
        prefixes = _find_peft_lora_prefixes(sd, base_prefix=base_prefix)
        local_best = -1
        for pair in prefixes.values():
            if "A" in pair:
                r = int(sd[pair["A"]].shape[0])
                local_best = max(local_best, r)
        if local_best > best_r:
            best_r = local_best
            best_sd = sd
    # fallback to first client's sd if not found
    return best_sd if best_sd is not None else client_updates[0]["updated_weights"]
