from __future__ import annotations

import os
from typing import Any, Dict

from ...ml_utils.csv_data_recorder import CsvDataRecorder
from .base import BaseSink


class CsvSink(BaseSink):
    """CSV sink that writes monitoring rows under a directory.

    Files:
      - rounds.csv
      - client_train.csv
    """

    def __init__(self, out_dir: str) -> None:
        self.out_dir = out_dir
        self._rounds: CsvDataRecorder | None = None
        self._client_train: CsvDataRecorder | None = None
        self._layers: CsvDataRecorder | None = None
        self._adalora: CsvDataRecorder | None = None

    def open(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        if self._rounds is None:
            self._rounds = CsvDataRecorder(os.path.join(self.out_dir, "rounds.csv"))
            self._rounds.begin(head_config=None)
        if self._client_train is None:
            self._client_train = CsvDataRecorder(os.path.join(self.out_dir, "client_train.csv"))
            self._client_train.begin(head_config=None)
        if self._layers is None:
            self._layers = CsvDataRecorder(os.path.join(self.out_dir, "aggregation_layers.csv"))
            self._layers.begin(head_config=None)
        if self._adalora is None:
            self._adalora = CsvDataRecorder(os.path.join(self.out_dir, "adalora_ranks.csv"))
            self._adalora.begin(head_config=None)

    def close(self) -> None:
        if self._rounds is not None:
            self._rounds.end()
            self._rounds = None
        if self._client_train is not None:
            self._client_train.end()
            self._client_train = None
        if self._layers is not None:
            self._layers.end()
            self._layers = None
        if self._adalora is not None:
            self._adalora.end()
            self._adalora = None

    def write_round(self, row: Dict[str, Any]) -> None:
        if self._rounds is None:
            return
        self._rounds.record(row)

    def write_client_train(self, row: Dict[str, Any]) -> None:
        if self._client_train is None:
            return
        self._client_train.record(row)

    def write_aggregation_layer(self, row: Dict[str, Any]) -> None:
        if self._layers is None:
            return
        self._layers.record(row)

    def write_adalora_rank(self, row: Dict[str, Any]) -> None:
        if self._adalora is None:
            return
        self._adalora.record(row)
