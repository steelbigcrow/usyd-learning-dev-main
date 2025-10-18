from __future__ import annotations

import os
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from ..ml_utils import console
from .config import MonitoringConfig
from .sinks.base import BaseSink
from .sinks.csv_sink import CsvSink
from .sinks.console_sink import ConsoleSink
from .sinks.tensorboard_sink import TensorBoardSink
from .types import RoundContext


_global_hub: "MonitorHub | None" = None


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


class MonitorHub:
    """Central hub for monitoring events and CSV/console output."""

    def __init__(self, cfg: MonitoringConfig) -> None:
        self.cfg = cfg
        self.enabled: bool = bool(cfg.enabled)
        self.run_id: str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.round_counter: int = -1  # starts at -1 so first next_round() -> 0
        self.current_round: Optional[RoundContext] = None
        self.sinks: List[BaseSink] = []
        # Output directories
        self._base_dir: str = cfg.dir
        # By default, write each run into a dedicated subfolder under .monitor
        self._current_out_dir: str = os.path.join(self._base_dir, self.run_id)

        # Phase timestamps
        self._t_agg: float | None = None
        self._t_bcast: float | None = None
        self._t_apply: float | None = None
        self._t_eval: float | None = None

        # Defer opening sinks until we know the final run folder
        # (training_logger instrumentation may retarget run folder).
        # Sinks will be opened on set_run_folder() or first round/write.
        # Lazy init for optional resource libs
        self._psutil = None
        self._pynvml = None

    # ---------- sinks ----------
    def _open_sinks(self) -> None:
        out_dir = self._current_out_dir
        os.makedirs(out_dir, exist_ok=True)
        for kind in self.cfg.sinks:
            k = str(kind).lower().strip()
            if k == "csv":
                s = CsvSink(out_dir)
                s.open()
                self.sinks.append(s)
            elif k == "console":
                self.sinks.append(ConsoleSink())
            elif k == "tensorboard":
                self.sinks.append(TensorBoardSink(log_dir=os.path.join(out_dir, "tb")))
            else:
                console.warn(f"[monitor] unknown sink '{kind}', skip")

    def _close_sinks(self) -> None:
        for s in self.sinks:
            try:
                s.close()
            except Exception:
                pass
        self.sinks.clear()

    def close(self) -> None:
        self._close_sinks()

    def _ensure_sinks_open(self) -> None:
        """Open sinks on-demand if not already opened."""
        if not self.enabled:
            return
        if not self.sinks:
            self._open_sinks()

    # ---------- retarget run folder ----------
    def set_run_folder(self, folder_name: str) -> None:
        """Move outputs into .monitor/<folder_name> for this run.

        Safe to call early (before any writes). If sinks already opened, they will
        be closed and reopened pointing to the new directory.
        """
        if not isinstance(folder_name, str) or not folder_name:
            return
        new_dir = os.path.join(self._base_dir, folder_name)
        # If unchanged, skip
        if os.path.normpath(new_dir) == os.path.normpath(self._current_out_dir):
            return
        self._current_out_dir = new_dir
        try:
            self._close_sinks()
        except Exception:
            pass
        try:
            # Open sinks now that final folder is known
            self._open_sinks()
            console.info(f"[monitor] run folder set to: {self._current_out_dir}")
        except Exception:
            pass

    # ---------- round lifecycle ----------
    def next_round(self) -> RoundContext | None:
        if not self.enabled:
            return None
        # Ensure sinks are opened before any writes in this round
        self._ensure_sinks_open()
        self.round_counter += 1
        self.current_round = RoundContext(
            run_id=self.run_id,
            round_index=self.round_counter,
        )
        return self.current_round

    def set_selected_clients(self, selected_ids: List[str], total_clients: int) -> None:
        if not self.enabled or self.current_round is None:
            return
        self.current_round.selected_ids = selected_ids
        self.current_round.total_clients = total_clients

    def set_aggregator(self, name: str) -> None:
        if not self.enabled or self.current_round is None:
            return
        self.current_round.aggregator = name

    def add_payload_in(self, bytes_in: int, total_samples: int) -> None:
        if not self.enabled or self.current_round is None:
            return
        self.current_round.payload_in_bytes += int(bytes_in)
        self.current_round.total_samples += int(total_samples)

    def set_broadcast_bytes(self, bytes_out_total: int) -> None:
        if not self.enabled or self.current_round is None:
            return
        self.current_round.payload_broadcast_bytes = int(bytes_out_total)

    # ---------- phase timers ----------
    def agg_begin(self) -> None:
        if not self.enabled:
            return
        self._t_agg = _now_ms()

    def agg_end(self) -> None:
        if not self.enabled or self.current_round is None:
            return
        if self._t_agg is not None:
            self.current_round.agg_ms = max(0.0, _now_ms() - self._t_agg)
        self._t_agg = None

    def apply_begin(self) -> None:
        if not self.enabled:
            return
        self._t_apply = _now_ms()

    def apply_end(self) -> None:
        if not self.enabled or self.current_round is None:
            return
        if self._t_apply is not None:
            self.current_round.apply_ms = max(0.0, _now_ms() - self._t_apply)
        self._t_apply = None

    def bcast_begin(self) -> None:
        if not self.enabled:
            return
        self._t_bcast = _now_ms()

    def bcast_end(self) -> None:
        if not self.enabled or self.current_round is None:
            return
        if self._t_bcast is not None:
            self.current_round.broadcast_ms = max(0.0, _now_ms() - self._t_bcast)
        self._t_bcast = None

    def eval_begin(self) -> None:
        if not self.enabled:
            return
        self._t_eval = _now_ms()

    def eval_end(self, metrics: Dict[str, Any] | None) -> None:
        if not self.enabled or self.current_round is None:
            return
        if self._t_eval is not None:
            self.current_round.eval_ms = max(0.0, _now_ms() - self._t_eval)
        self._t_eval = None
        self.current_round.metrics = metrics or {}

    def finalize_round(self) -> None:
        if not self.enabled or self.current_round is None:
            return
        self._ensure_sinks_open()
        c = self.current_round
        row = {
            "ts": int(time.time()),
            "run_id": c.run_id,
            "round": c.round_index,
            "aggregator": c.aggregator or "",
            "clients": c.total_clients,
            "selected_ids": ",".join(c.selected_ids or []),
            "total_samples": c.total_samples,
            "agg_ms": c.agg_ms,
            "apply_ms": c.apply_ms,
            "broadcast_ms": c.broadcast_ms,
            "eval_ms": c.eval_ms,
            "payload_bytes_in": c.payload_in_bytes,
            "payload_bytes_out": c.payload_broadcast_bytes,
            "bcast_pad_ops": getattr(self, "_broadcast_pad_count", 0),
            "bcast_slice_ops": getattr(self, "_broadcast_slice_count", 0),
        }

        # Optional resource sampling
        res_cfg = self.cfg.raw.get("monitoring", {}).get("resources", {}) if isinstance(self.cfg.raw, dict) else {}
        if isinstance(res_cfg, dict):
            cpu_on = bool(res_cfg.get("cpu_mem", False))
            gpu_on = bool(res_cfg.get("gpu_mem", False))
        else:
            cpu_on = gpu_on = False

        if cpu_on:
            try:
                if self._psutil is None:
                    import psutil  # type: ignore

                    self._psutil = psutil
                process = self._psutil.Process()
                rss = float(process.memory_info().rss) / (1024 * 1024)
                row["cpu_rss_mb"] = rss
            except Exception:
                pass
        if gpu_on:
            try:
                if self._pynvml is None:
                    import pynvml  # type: ignore

                    self._pynvml = pynvml
                    self._pynvml.nvmlInit()
                h = self._pynvml.nvmlDeviceGetHandleByIndex(0)
                mem = self._pynvml.nvmlDeviceGetMemoryInfo(h)
                row["gpu_mem_used_mb"] = float(mem.used) / (1024 * 1024)
            except Exception:
                pass

        # Attach common metrics if available
        if c.metrics:
            for k in ("accuracy", "average_loss", "precision", "recall", "f1_score"):
                if k in c.metrics and c.metrics[k] is not None:
                    # map to short names
                    key_map = {
                        "accuracy": "acc",
                        "average_loss": "val_loss",
                        "precision": "precision",
                        "recall": "recall",
                        "f1_score": "f1",
                    }
                    row[key_map[k]] = c.metrics[k]

        for s in self.sinks:
            try:
                s.write_round(row)
            except Exception:
                pass

        # Reset per-round broadcast op counters
        self._broadcast_pad_count = 0
        self._broadcast_slice_count = 0

    # ---------- client training ----------
    def client_train_begin(self, client_id: str, local_epochs: int, data_samples: int) -> None:
        if not self.enabled:
            return
        console.info(
            f"[monitor] client begin id={client_id} epochs={local_epochs} samples={data_samples}"
        )

    def client_train_end(self, client_id: str, train_record: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._ensure_sinks_open()
        # train_record typically contains: epoch_loss (list), avg_loss, ...
        epoch_losses: List[float] = list(train_record.get("epoch_loss", []))
        avg_loss = train_record.get("avg_loss", None)

        round_idx = self.current_round.round_index if self.current_round else -1

        for idx, loss in enumerate(epoch_losses, start=1):
            row = {
                "ts": int(time.time()),
                "run_id": self.run_id,
                "round": round_idx,
                "client_id": client_id,
                "epoch": idx,
                "loss": loss,
                "avg_loss": avg_loss,
            }
            for s in self.sinks:
                try:
                    s.write_client_train(row)
                except Exception:
                    pass

    # ---------- Aggregation per-layer ----------
    def write_aggregation_layer(self, row: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._ensure_sinks_open()
        for s in self.sinks:
            try:
                s.write_aggregation_layer(row)
            except Exception:
                pass

    # ---------- AdaLoRA ranks ----------
    def write_adalora_rank(self, row: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._ensure_sinks_open()
        for s in self.sinks:
            try:
                s.write_adalora_rank(row)
            except Exception:
                pass

    # ---------- Broadcast slice/pad counters ----------
    _broadcast_pad_count: int = 0
    _broadcast_slice_count: int = 0

    def incr_broadcast_pad(self, n: int = 1) -> None:
        if not self.enabled:
            return
        self._broadcast_pad_count += int(n)

    def incr_broadcast_slice(self, n: int = 1) -> None:
        if not self.enabled:
            return
        self._broadcast_slice_count += int(n)

    # Expose cfg for instrumentation
    @property
    def config(self) -> MonitoringConfig:
        return self.cfg


def get_hub() -> MonitorHub:
    global _global_hub
    if _global_hub is None:
        # Construct with defaults if not initialized by auto_enable
        from .config import MonitoringConfig

        _global_hub = MonitorHub(MonitoringConfig.load(None))
    return _global_hub
