from __future__ import annotations

from typing import Any, Dict

from .base import BaseSink


class TensorBoardSink(BaseSink):
    """Write key scalars to TensorBoard if tensorboardX or torch.utils.tensorboard is available."""

    def __init__(self, log_dir: str) -> None:
        self._writer = None
        self._step_round_offset = 0
        # Try tensorboard from torch first
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore

            self._writer = SummaryWriter(log_dir=log_dir)
        except Exception:
            try:
                from tensorboardX import SummaryWriter  # type: ignore

                self._writer = SummaryWriter(log_dir=log_dir)
            except Exception:
                self._writer = None

    def open(self) -> None:
        return

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.flush()
                self._writer.close()
            except Exception:
                pass

    def _add_scalar(self, tag: str, val: float, step: int) -> None:
        if self._writer is None:
            return
        try:
            self._writer.add_scalar(tag, float(val), global_step=step)
        except Exception:
            pass

    def write_round(self, row: Dict[str, Any]) -> None:
        if self._writer is None:
            return
        step = int(row.get("round", 0))
        for k, v in (row or {}).items():
            if isinstance(v, (int, float)):
                if k in ("acc", "val_loss", "precision", "recall", "f1", "agg_ms", "broadcast_ms", "eval_ms"):
                    self._add_scalar(f"round/{k}", float(v), step)

    def write_client_train(self, row: Dict[str, Any]) -> None:
        if self._writer is None:
            return
        step = int(row.get("round", 0)) * 1000 + int(row.get("epoch", 0))
        cid = str(row.get("client_id", "?"))
        if "loss" in row and isinstance(row["loss"], (int, float)):
            self._add_scalar(f"client/{cid}/loss", float(row["loss"]), step)
        if "avg_loss" in row and isinstance(row["avg_loss"], (int, float)):
            self._add_scalar(f"client/{cid}/avg_loss", float(row["avg_loss"]), step)

    def write_aggregation_layer(self, row: Dict[str, Any]) -> None:
        # Could add histograms in future; keep minimal now
        return

    def write_adalora_rank(self, row: Dict[str, Any]) -> None:
        if self._writer is None:
            return
        step = int(row.get("epoch", 0))
        layer = str(row.get("layer", "?"))
        r = row.get("r", None)
        if isinstance(r, (int, float)):
            self._add_scalar(f"adalora/{layer}/rank", float(r), step)

