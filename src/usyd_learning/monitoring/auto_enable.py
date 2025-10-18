from __future__ import annotations

"""
Single-line activation for monitoring.

Usage (top of entry script):
    import usyd_learning.monitoring.auto_enable  # noqa: F401

Optionally load a specific config before patching:
    import usyd_learning.monitoring.auto_enable as mon
    mon.load_config("./path/to/monitoring.yaml")
"""

from typing import Optional

from ..ml_utils import console
from .config import MonitoringConfig
from .hub import MonitorHub
from .instrumentation.federated import (
    patch_client_training,
    patch_fed_server,
    patch_prepare_events,
)
from .instrumentation.aggregators import record_aggregation_details  # noqa: F401
from .instrumentation.adalora import (
    patch_train_step_rank_snapshot,
    patch_broadcast_pad_slice_counters,
)
from .instrumentation.training_logger import patch_training_logger


_hub: MonitorHub | None = None


def _ensure_hub(cfg: MonitoringConfig) -> MonitorHub:
    global _hub
    if _hub is None:
        _hub = MonitorHub(cfg)
    return _hub


def load_config(path: Optional[str] = None) -> MonitoringConfig:
    cfg = MonitoringConfig.load(path)
    _ensure_hub(cfg)
    return cfg


def _enable_instrumentation() -> None:
    # Load default config (may already be loaded via load_config)
    cfg = MonitoringConfig.load(None)
    _ensure_hub(cfg)

    if not cfg.enabled:
        console.warn("[monitor] disabled by config")
        return

    # Instrument server lifecycle and client training
    patch_prepare_events()
    patch_fed_server()
    patch_client_training()
    patch_train_step_rank_snapshot()
    patch_broadcast_pad_slice_counters()
    patch_training_logger()
    console.ok("[monitor] instrumentation enabled")


# Auto-enable upon import
_enable_instrumentation()
