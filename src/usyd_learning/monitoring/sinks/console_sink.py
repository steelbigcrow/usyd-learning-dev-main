from __future__ import annotations

from typing import Any, Dict

from ...ml_utils import console
from .base import BaseSink


class ConsoleSink(BaseSink):
    def write_round(self, row: Dict[str, Any]) -> None:
        rid = row.get("run_id", "?")
        rnd = row.get("round", "?")
        agg = row.get("aggregator", "?")
        acc = row.get("acc", None)
        console.info(
            f"[monitor] run={rid} round={rnd} agg={agg} "
            f"agg_ms={row.get('agg_ms', '-')}, bcast_ms={row.get('broadcast_ms', '-')}, eval_ms={row.get('eval_ms','-')} "
            + (f"acc={acc:.4f}" if isinstance(acc, float) else "")
        )

    def write_client_train(self, row: Dict[str, Any]) -> None:
        console.info(
            f"[monitor] client={row.get('client_id','?')} round={row.get('round','?')} "
            f"epoch={row.get('epoch','?')} loss={row.get('loss','?')} avg={row.get('avg_loss','?')}"
        )

    def write_aggregation_layer(self, row: Dict[str, Any]) -> None:
        console.info(
            f"[monitor] agg-layer method={row.get('method','?')} round={row.get('round','?')} "
            f"layer={row.get('layer','?')} m={row.get('m','?')} n={row.get('n','?')} r_max={row.get('r_max','?')}"
        )

    def write_adalora_rank(self, row: Dict[str, Any]) -> None:
        r = row.get('r', '?')
        r_eff = row.get('r_eff', None)
        r_part = f"r={r}"
        if isinstance(r_eff, (int, float)):
            r_part += f" r_eff={int(r_eff)}"
        console.info(
            f"[monitor] adalora role={row.get('role','?')} node={row.get('node_id','?')} "
            f"epoch={row.get('epoch','?')} layer={row.get('layer','?')} {r_part}"
        )
