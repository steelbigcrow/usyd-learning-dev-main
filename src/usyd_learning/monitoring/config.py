from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..ml_utils import console


def _read_yaml_file(path: str) -> Dict[str, Any] | None:
    """Best-effort YAML reader. Falls back to None if PyYAML is unavailable or read fails."""
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return None


def _default_config() -> Dict[str, Any]:
    return {
        "monitoring": {
            "enabled": True,
            "sinks": ["csv", "console"],
            "dir": "./.monitor",
            "sampling": {"batch_interval": 1},
            "resources": {"gpu_mem": False, "cpu_mem": False},
            "adalora": {"enabled": "auto", "snapshot": "epoch"},
            "aggregation": {
                "common": {"enabled": True},
                "fedavg": {"enabled": True, "layer_delta_sample": 0},
                "svd": {"enabled": True, "energy_sample_layers": 0},
                "rbla": {"enabled": True, "track_pad_slice": True},
                "sp": {"enabled": True, "apply_weight_mode_metrics": False},
                "zeropad": {"enabled": True},
            },
        }
    }


@dataclass
class MonitoringConfig:
    enabled: bool = True
    sinks: List[str] = field(default_factory=lambda: ["csv", "console"])
    dir: str = "./.monitor"
    raw: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def load(config_path: str | None = None) -> "MonitoringConfig":
        """Load monitoring config by precedence order defined in Monitor Plan.

        Search paths:
          1) ./monitoring/monitoring.yaml
          2) ./monitoring.yaml
          3) ./src/test/fl_lora_sample/yamls/monitoring/standard_monitor.yaml
        """
        candidates: List[str] = []
        if config_path:
            candidates = [config_path]
        else:
            candidates = [
                os.path.join("./monitoring", "monitoring.yaml"),
                "./monitoring.yaml",
                os.path.join(
                    "./src/test/fl_lora_sample/yamls/monitoring", "standard_monitor.yaml"
                ),
            ]

        cfg_obj: Dict[str, Any] | None = None
        chosen: str | None = None
        for p in candidates:
            if os.path.isfile(p):
                cfg_obj = _read_yaml_file(p)
                if cfg_obj is not None:
                    chosen = p
                    break

        if cfg_obj is None:
            cfg_obj = _default_config()
            chosen = "<built-in-default>"

        # Extract
        mon = cfg_obj.get("monitoring", {}) if isinstance(cfg_obj, dict) else {}
        enabled = bool(mon.get("enabled", True))
        sinks = list(mon.get("sinks", ["csv", "console"]))
        out_dir = str(mon.get("dir", "./.monitor"))

        console.info(f"[monitor] config loaded from: {chosen}")

        return MonitoringConfig(
            enabled=enabled,
            sinks=sinks,
            dir=out_dir,
            raw=cfg_obj if isinstance(cfg_obj, dict) else {},
        )

