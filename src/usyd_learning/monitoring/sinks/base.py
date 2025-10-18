from __future__ import annotations

from typing import Any, Dict


class BaseSink:
    """Interface for monitor sinks."""

    def open(self) -> None:  # pragma: no cover - simple plumbing
        pass

    def close(self) -> None:  # pragma: no cover - simple plumbing
        pass

    # ---- Round-level ----
    def write_round(self, row: Dict[str, Any]) -> None:
        pass

    # ---- Client epoch-level ----
    def write_client_train(self, row: Dict[str, Any]) -> None:
        pass

    # ---- Aggregation per-layer ----
    def write_aggregation_layer(self, row: Dict[str, Any]) -> None:
        pass

    # ---- AdaLoRA ranks ----
    def write_adalora_rank(self, row: Dict[str, Any]) -> None:
        pass
