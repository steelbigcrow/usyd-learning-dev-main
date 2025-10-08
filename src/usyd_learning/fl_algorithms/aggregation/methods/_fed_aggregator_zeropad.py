import torch
from collections import OrderedDict

from ..fed_aggregator_abc import AbstractFedAggregator
from ..fed_aggregator_args import FedAggregatorArgs
from ._fed_aggregator_rbla import FedAggregator_RBLA
from ....ml_utils import console


class FedAggregator_ZeroPad(FedAggregator_RBLA):
    """
    Zero-Padding LoRA aggregator (ZP):
    - Compatible with variable-rank LoRA across clients.
    - For each LoRA layer, determine the maximum rank among clients.
    - Pad each client's `lora_A`/`lora_B` along the rank dimension with zeros to match
      the maximum rank for that layer, then compute a weighted average.
    - Non-LoRA parameters are aggregated via weighted sum (FedAvg-style).

    Notes
    - This class reuses RBLA's robust input handling and broadcasting helpers, but
      sets the padding mode to zeros so that padded entries contribute as 0 and the
      weighted average is simply normalized by total weights.
    - Shapes follow the internal convention used in RBLA helpers:
        lora_A: [r, in_features]
        lora_B: [out_features, r]
    """

    def __init__(self, args: FedAggregatorArgs | None = None):
        super().__init__(args)
        self._aggregation_method = "zp"
        # Force zero-padding behaviour for LoRA matrices
        self.set_pad_mode("zero")

    # The core aggregation flow is inherited from RBLA; the only change is pad_mode.
    # _before_aggregation, _do_aggregation and _after_aggregation are adequate as-is.

