import torch
from collections import OrderedDict

from ..fed_aggregator_abc import AbstractFedAggregator
from ..fed_aggregator_args import FedAggregatorArgs
from ....ml_utils import console


class FedAggregator_ZeroPad(AbstractFedAggregator):
    """
    Zero-Padding LoRA aggregator (ZP), implemented independently of RBLA.

    Behaviour:
    - For LoRA matrices, pad along the rank dimension with zeros to the layer-wise
      maximum rank across clients, then compute a weighted average.
      Convention:
        lora_A shape: [r, in_features]
        lora_B shape: [out_features, r]
    - Non-LoRA parameters are aggregated by weighted averaging (FedAvg-style).

    Input formats accepted by .aggregate() are normalized in AbstractFedAggregator:
      - list[(state_dict, volume)]
      - list[{"updated_weights": sd, "train_record": {"data_sample_num": vol}}]
      - dict{"state_dicts": [...], "weights": [...]} (legacy)
    """

    def __init__(self, args: FedAggregatorArgs | None = None):
        super().__init__(args)
        self._aggregation_method = "zp"
        self._lora_suffixes: set[str] = {"lora_A", "lora_B"}

    # ---------- Aggregation lifecycle ----------
    def _before_aggregation(self) -> None:
        # console.debug(f"[ZP] Starting aggregation...")
        return

    def _do_aggregation(self) -> None:
        """Aggregate client state_dicts with zero-padding for LoRA tensors."""
        state_dicts, weights = None, None

        # Normalize input from the container set by AbstractFedAggregator.aggregate
        if hasattr(self, "_aggregation_data_list") and self._aggregation_data_list:
            pairs = self._aggregation_data_list
            state_dicts = [sd for sd, vol in pairs]
            weights = [float(vol) for sd, vol in pairs]
        elif hasattr(self, "_aggregation_data_dict") and isinstance(self._aggregation_data_dict, list):
            pairs = self._aggregation_data_dict
            state_dicts = [sd for sd, vol in pairs]
            weights = [float(vol) for sd, vol in pairs]
        elif hasattr(self, "_aggregation_data_dict") and isinstance(self._aggregation_data_dict, dict):
            state_dicts = self._aggregation_data_dict["state_dicts"]
            weights = self._aggregation_data_dict.get("weights", None)
        else:
            raise ValueError("[ZP] No aggregation data found. Provide a list of (state_dict, data_volume) "
                             "or dict {'state_dicts': [...], 'weights': [...] }.")

        console.debug(f"\n[ZP] Aggregating {len(state_dicts)} clients...")

        # Move all tensors to the configured device
        dev = self._device
        sds_on_device = [{k: v.to(dev) for k, v in sd.items()} for sd in state_dicts]

        aggregated = self.aggregate_state_dicts(
            sds_on_device,
            weights=weights,
            lora_suffixes=self._lora_suffixes,
        )

        # Keep order of the first client's keys for stability
        sample_keys = list(state_dicts[0].keys())
        ordered = OrderedDict((k, aggregated[k]) for k in sample_keys)
        self._aggregated_weight = ordered

        first_param_name = next(iter(ordered.keys()))
        console.debug(f"[ZP] Aggregated first param mean: {ordered[first_param_name].mean():.6f}")

    def _after_aggregation(self) -> None:
        console.debug("[ZP] Aggregation completed.")

    # ---------- Core Zero-Pad ops ----------
    @staticmethod
    def get_suffix(key: str) -> str:
        """Return the suffix after the last dot."""
        return key.rsplit(".", 1)[-1]

    @staticmethod
    def pad_tensors_to_max_shape(tensors: list[torch.Tensor]) -> torch.Tensor:
        """
        Pad tensors to a common shape with zeros; return stacked tensor with leading batch dim.

        Used for LoRA matrices (2D) as well as feature-shifted dense weights/biases that may have
        different row/column lengths. All tensors must share dimensionality, otherwise the mismatch
        stems from architecture differences that the zero-pad strategy cannot reconcile.
        """
        if len(tensors) == 0:
            raise ValueError("pad_tensors_to_max_shape: empty tensor list")

        ref_dim = tensors[0].dim()
        if any(t.dim() != ref_dim for t in tensors):
            raise ValueError(
                f"pad_tensors_to_max_shape: inconsistent tensor dims "
                f"{[t.dim() for t in tensors]} (expected all {ref_dim}D)"
            )

        max_shape = tuple(max(t.shape[d] for t in tensors) for d in range(ref_dim))
        device = tensors[0].device
        dtype = tensors[0].dtype

        padded = torch.zeros((len(tensors),) + max_shape, dtype=dtype, device=device)
        for idx, tensor in enumerate(tensors):
            slices = tuple(slice(0, s) for s in tensor.shape)
            padded[(idx,) + slices] = tensor
        return padded

    @staticmethod
    def aggregate_lora_tensors(
        tensors: list[torch.Tensor],
        weights: list[float],
    ) -> torch.Tensor:
        """Weighted average with zero-padding for LoRA matrices."""
        if len(tensors) == 0:
            raise ValueError("aggregate_lora_tensors: empty tensor list")

        # Normalize weights to sum=1 for stability
        tw = float(sum(weights))
        weights = [w / tw for w in weights] if tw > 0 else [1.0 / len(tensors)] * len(tensors)

        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=tensors[0].device).view(-1, 1, 1)
        padded = FedAggregator_ZeroPad.pad_tensors_to_max_shape(tensors)

        # Simple weighted average (padded zeros contribute 0)
        weighted_sum = (padded * weights_tensor).sum(dim=0)
        return weighted_sum  # weights already normalized

    @staticmethod
    def aggregate_state_dicts(
        state_dicts: list[dict],
        weights: list[float] | None = None,
        lora_suffixes: set[str] = {"lora_A", "lora_B"},
    ) -> dict:
        """Aggregate multiple state_dicts with LoRA-aware zero-padding."""
        if len(state_dicts) == 0:
            raise ValueError("aggregate_state_dicts: empty state_dicts")

        if weights is None:
            weights = [1.0] * len(state_dicts)

        # Normalize weights to sum=1
        tw = float(sum(weights))
        weights = [w / tw for w in weights] if tw > 0 else [1.0 / len(weights)] * len(weights)

        keys = list(state_dicts[0].keys())
        aggregated: dict[str, torch.Tensor] = {}

        for key in keys:
            values = [sd[key] for sd in state_dicts]
            suffix = FedAggregator_ZeroPad.get_suffix(key)

            # Be robust to keys like '*.lora_A.default'
            if suffix not in lora_suffixes:
                if ".lora_A" in key:
                    suffix = "lora_A"
                elif ".lora_B" in key:
                    suffix = "lora_B"

            if suffix in lora_suffixes:
                aggregated[key] = FedAggregator_ZeroPad.aggregate_lora_tensors(values, weights)
            else:
                # Generic weighted average for non-LoRA params, zero-padding when shapes mismatch
                same_shape = all(v.shape == values[0].shape for v in values)
                if same_shape:
                    stacked = torch.stack(values, dim=0)  # (N, ...)
                else:
                    stacked = FedAggregator_ZeroPad.pad_tensors_to_max_shape(values)

                view_shape = (len(weights),) + (1,) * (stacked.dim() - 1)
                weight_tensor = torch.as_tensor(weights, dtype=stacked.dtype, device=stacked.device).view(*view_shape)
                weighted_sum = (stacked * weight_tensor).sum(dim=0)
                aggregated[key] = weighted_sum

        return aggregated
