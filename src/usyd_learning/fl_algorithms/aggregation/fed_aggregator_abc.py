from __future__ import annotations
from abc import ABC, abstractmethod

import random
import time
from typing import Any
import torch

from .fed_aggregator_args import FedAggregatorArgs


class AbstractFedAggregator(ABC):
    """
    Fed Server Aggregator class interface declare
    """
    
    def __init__(self, args: FedAggregatorArgs|None = None):
        """
        Initialize aggregator with updated weight list and aggregation method,
        random seed is set with current time milliseconds(range in 0~999)

        Arg:
            aggregation_data_list(list): list of aggregation data, each element is a tuple of (model_weight: dict / wbab, vol)
            aggregation_method(EFedServerAggregationMethod): method to aggregate the weights, such as FedAvg, RBLA, etc.
        """

        if args is None:
            self.args = FedAggregatorArgs()
        else:
            self.args = args

        # data list
        self._aggregation_data_dict: dict = {}

        # Select method
        self._aggregation_method = ""

        # Aggregated weight
        self._aggregated_weight: Any = None  # can be wbab or torch dict or ....

        # perform aggregation on ('cpu' or 'cuda')
        self._device = torch.device(self.args.device)

        # seed range milliseconds 0~999
        self.with_random_seed(int(time.time() * 1000) % 1000)
        return

    @property       # aggregated weight property
    def aggregated_weight(self): return self._aggregated_weight

    @property       # aggregation method property
    def aggregated_method(self): return self._aggregation_method

    def with_random_seed(self, seed: int):
        """
        Manual set random seed, Return self
        """
        random.seed(seed)
        return self

    def with_clients_update(self, clients_update: Any):
        """
        When select clients according to some client data

        Args:
            client_data: any clients data, maybe dict or list

        Return:
            self
        """
        self._clients_update = clients_update
        return self

    def aggregate(self, client_data):
        """
        Run aggregation on client updates.

        Supports multiple input formats for backward compatibility:
        - List[Dict]: [{"updated_weights": sd, "train_record": {"data_sample_num": vol}}, ...]
        - List[Tuple]: [(sd, vol), ...] or [[sd, vol], ...]
        - Dict legacy: {"state_dicts": [sd1, sd2, ...], "weights": [w1, w2, ...]}
        """
        # Reset any previous data containers
        if hasattr(self, "_aggregation_data_list"):
            delattr(self, "_aggregation_data_list")
        self._aggregation_data_dict = None  # type: ignore

        # Normalize inputs into the forms understood by concrete aggregators
        if isinstance(client_data, list):
            if len(client_data) == 0:
                self._aggregation_data_dict = []
            else:
                first = client_data[0]
                # Case 1: List[Dict]
                if isinstance(first, dict) and "updated_weights" in first and "train_record" in first:
                    self._aggregation_data_dict = [
                        (d["updated_weights"], d["train_record"]["data_sample_num"]) for d in client_data  # type: ignore[index]
                    ]
                # Case 2: List[(sd, vol)]
                elif isinstance(first, (list, tuple)) and len(first) == 2:
                    self._aggregation_data_dict = client_data
                else:
                    raise TypeError(
                        "Unsupported list format for aggregation. Expected list of dicts with "
                        "'updated_weights' and 'train_record' or list of (state_dict, volume)."
                    )

        elif isinstance(client_data, dict):
            # Legacy dict input: {"state_dicts": [...], "weights": [...]} or other prebuilt forms
            self._aggregation_data_dict = client_data
        else:
            raise TypeError(
                f"Unsupported aggregation input type: {type(client_data).__name__}. "
                "Provide a list or a dict."
            )

        self._before_aggregation()
        self._do_aggregation()
        self._after_aggregation()
        return self._aggregated_weight

    ########################################
    # Abstract method

    @abstractmethod
    def _before_aggregation(self) -> None:
        """
        Call before aggregate
        """
        pass

    @abstractmethod
    def _do_aggregation(self) -> None:
        """
        do aggregate
        """
        pass

    @abstractmethod
    def _after_aggregation(self) -> None:
        """
        Call after aggregate
        """
        pass
