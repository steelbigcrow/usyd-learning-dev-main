from __future__ import annotations

from typing import Dict

import torch

from ...ml_utils import console
from ...ml_algorithms.lora.lora_utils import LoRAUtils
from ..hub import get_hub


def _patch_method(cls, name: str, wrapper) -> None:
    orig = getattr(cls, name, None)
    if orig is None:
        return
    if getattr(orig, "__monitor_patched__", False):
        return

    def _wrapped(self, *args, **kwargs):
        return wrapper(self, orig, *args, **kwargs)

    setattr(_wrapped, "__monitor_patched__", True)
    setattr(cls, name, _wrapped)


def patch_train_step_rank_snapshot() -> None:
    """Patch ModelTrainer_Standard.train_step to snapshot LoRA ranks at epoch end.

    This runs after each train_step call from ModelTrainer_Standard.train, where
    'self._epoch_idx' indicates the current epoch (1-based).
    """
    try:
        from ...model_trainer.trainer._model_trainer_standard import ModelTrainer_Standard
    except Exception as e:
        console.warn(f"[monitor] skip adalora train_step patch: {e}")
        return

    def _train_step_wrapper(self: "ModelTrainer_Standard", orig, *args, **kwargs):
        loss = orig(self, *args, **kwargs)
        try:
            hub = get_hub()
            cfg = hub.config.raw.get("monitoring", {}).get("adalora", {}) if hub else {}
            enabled = cfg.get("enabled", "auto")
            if str(enabled).lower() == "false":
                return loss

            # Determine role/node id context
            node_id = getattr(hub, "_current_client_id", None) or "client"
            role = "client"

            # Access current model; prefer self.trainer_args.model if exists
            model = getattr(self, "model", None) or getattr(getattr(self, "trainer_args", None), "model", None)
            if model is None:
                return loss

            # Infer LoRA ranks; if none, and enabled=="auto", do nothing
            ranks: Dict[str, int] = LoRAUtils.get_lora_ranks(model)
            if not ranks:
                return loss

            # Try to derive effective ranks (r_eff) from AdaLoRA mask/hints when available
            r_eff_map: Dict[str, int] = {}
            try:
                # Use PEFT/AdaLoRA bridge to infer per-layer rr from lora_E/ranknum
                from ...ml_algorithms.adalora.adalora_rbla_bridge import (
                    peft_to_plain_lora_shrunk,
                )

                peft_sd = model.state_dict()
                shrunk = peft_to_plain_lora_shrunk(peft_sd)

                # Extract '<prefix>.rank_rr' -> int(rr)
                for k, v in list(shrunk.items()):
                    if not isinstance(k, str):
                        continue
                    if k.endswith(".rank_rr"):
                        pref = k[: -len(".rank_rr")]
                        try:
                            r_eff_map[pref] = int(round(float(getattr(v, "item", lambda: v)())))  # type: ignore
                        except Exception:
                            try:
                                r_eff_map[pref] = int(round(float(v)))  # type: ignore[arg-type]
                            except Exception:
                                pass

                # Provide alternate keys prefixed with 'base_model.model.' to match possible layer names
                for pref, rr in list(r_eff_map.items()):
                    bm_pref = f"base_model.model.{pref}"
                    if bm_pref not in r_eff_map:
                        r_eff_map[bm_pref] = rr
            except Exception:
                # Best-effort only; if unavailable, fall back to physical ranks
                r_eff_map = {}

            # Emit rank rows per layer
            round_idx = hub.current_round.round_index if (hub and hub.current_round) else -1
            epoch_idx = int(getattr(self, "_epoch_idx", 0) or 0)
            ts_now = int(__import__("time").time())
            for layer, r in ranks.items():
                r_eff = int(r_eff_map.get(layer, r))
                hub.write_adalora_rank(
                    {
                        "ts": ts_now,
                        "run_id": hub.run_id if hub else "",
                        "round": round_idx,
                        "role": role,
                        "node_id": node_id,
                        "epoch": epoch_idx,
                        "layer": layer,
                        "r": int(r),
                        "r_eff": int(r_eff),
                    }
                )
        except Exception:
            pass
        return loss

    _patch_method(ModelTrainer_Standard, "train_step", _train_step_wrapper)


def patch_broadcast_pad_slice_counters() -> None:
    """Patch LoRAUtils.broadcast_lora_state_dict to track pad/slice operations during broadcast."""
    try:
        from ...ml_algorithms.lora.lora_utils import LoRAUtils as _LU
    except Exception as e:
        console.warn(f"[monitor] skip broadcast pad/slice patch: {e}")
        return

    orig = getattr(_LU, "broadcast_lora_state_dict", None)
    if orig is None or getattr(orig, "__monitor_patched__", False):
        return

    def _wrapped(global_sd: dict, local_sd: dict, lora_suffixes: set[str] = {"lora_A", "lora_B"}):
        # Count pad/slice decisions by comparing local/global shapes
        hub = get_hub()
        try:
            for k, lt in local_sd.items():
                if k not in global_sd:
                    continue
                gt = global_sd[k]
                if not (torch.is_tensor(lt) and torch.is_tensor(gt)):
                    continue
                # Determine suffix
                suffix = k.rsplit(".", 1)[-1]
                if suffix not in lora_suffixes:
                    if ".lora_A" in k:
                        suffix = "lora_A"
                    elif ".lora_B" in k:
                        suffix = "lora_B"
                if suffix not in lora_suffixes:
                    continue
                if suffix == "lora_A":
                    r_local, r_global = int(lt.shape[0]), int(gt.shape[0])
                else:  # lora_B
                    r_local, r_global = int(lt.shape[1]), int(gt.shape[1])
                if r_global > r_local:
                    hub.incr_broadcast_slice(1)
                elif r_global < r_local:
                    hub.incr_broadcast_pad(1)
        except Exception:
            pass
        return orig(global_sd, local_sd, lora_suffixes)

    setattr(_LU.broadcast_lora_state_dict, "__monitor_patched__", True)
    _LU.broadcast_lora_state_dict = staticmethod(_wrapped)  # type: ignore[attr-defined]
