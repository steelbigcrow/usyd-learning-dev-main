from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import torch.nn as nn


@dataclass
class AdaLoRAOptions:
    """
    Lightweight options holder for AdaLoRA wrapping.
    Only a subset of PEFT AdaLoraConfig options are exposed directly, but
    additional AdaLoRA hyperparameters can be passed via `extra_kwargs` and
    will be forwarded to `peft.AdaLoraConfig` if supported by the installed PEFT
    version.
    """
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Optional[Union[Sequence[str], str]] = None  # names or name substrings to match
    total_step: int = 1000  # required by AdaLoRA scheduler
    # Any additional AdaLoRA kwargs (e.g., init_r, tinit, tfinal, deltaT,
    # beta1, beta2, orth_reg_weight, rank_pattern, target_r, bias, etc.)
    extra_kwargs: Optional[Dict[str, Any]] = None


def _auto_linear_module_names(model: nn.Module) -> List[str]:
    """Collect leaf module names that are instances of nn.Linear."""
    names: List[str] = []
    for name, mod in model.named_modules():
        # skip root and non-leaf children (named_modules includes parents)
        if name and isinstance(mod, nn.Linear) and len(list(mod.children())) == 0:
            names.append(name)
    return names


def wrap_with_adalora(model: nn.Module, opts: AdaLoRAOptions) -> nn.Module:
    """
    Wrap a torch.nn.Module with PEFT AdaLoRA.

    This function finds target modules (nn.Linear by default) and applies
    AdaLoRA adapters with the provided configuration.
    """
    try:
        import inspect
        from peft import AdaLoraConfig, get_peft_model
    except Exception as e:  # pragma: no cover - surfaced during tests if missing
        raise ImportError("peft is required for AdaLoRA integration. pip install peft") from e

    # target modules: accept list[str] or str; default to nn.Linear leaves
    if opts.target_modules is None:
        target = _auto_linear_module_names(model)
    else:
        target = opts.target_modules
    if not target:
        raise ValueError("No target modules found for AdaLoRA. Provide target_modules or ensure the model has nn.Linear layers.")

    # Build kwargs and filter by current AdaLoraConfig signature for robustness
    cfg_kwargs: Dict[str, Any] = dict(
        r=int(opts.r),
        lora_alpha=int(opts.lora_alpha),
        lora_dropout=float(opts.lora_dropout),
        target_modules=target,
        total_step=int(opts.total_step) if opts.total_step is not None else None,
    )

    if opts.extra_kwargs:
        cfg_kwargs.update(opts.extra_kwargs)

    allowed = set(inspect.signature(AdaLoraConfig).parameters.keys())
    # Drop unknown keys and None values to avoid overriding library defaults
    filtered = {k: v for k, v in cfg_kwargs.items() if k in allowed and v is not None}

    config = AdaLoraConfig(**filtered)
    peft_model = get_peft_model(model, config)
    return peft_model
