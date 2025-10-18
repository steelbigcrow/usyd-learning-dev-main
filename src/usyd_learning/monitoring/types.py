from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class RoundContext:
    run_id: str
    round_index: int
    aggregator: str | None = None
    selected_ids: List[str] | None = None
    total_clients: int = 0
    payload_in_bytes: int = 0
    payload_broadcast_bytes: int = 0
    agg_ms: float | None = None
    broadcast_ms: float | None = None
    eval_ms: float | None = None
    apply_ms: float | None = None
    total_samples: int = 0
    metrics: Dict[str, Any] | None = None

