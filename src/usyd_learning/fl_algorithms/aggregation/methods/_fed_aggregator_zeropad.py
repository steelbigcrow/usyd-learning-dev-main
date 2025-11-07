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
        Pad 1D/2D tensors to a common shape with zeros; return stacked tensor with leading dim N.
        - 1D inputs -> shape (N, max_len)
        - 2D inputs -> shape (N, max_rows, max_cols)
        Used primarily for LoRA matrices where row/col mismatch comes from rank differences,
        and for LoRA-related auxiliary vectors (e.g., rank masks) under heterogeneous ranks.
        """
        if len(tensors) == 0:
            raise ValueError("pad_tensors_to_max_shape: empty tensor list")

        # Ensure 2D for LoRA matrices
        dim = tensors[0].dim()
        if dim not in (1, 2):
            raise ValueError(f"pad_tensors_to_max_shape supports 1D/2D, got {dim}D")

        device = tensors[0].device
        dtype = tensors[0].dtype

        if dim == 1:
            max_len = max(int(t.shape[0]) for t in tensors)
            padded = []
            for t in tensors:
                pad = torch.zeros((max_len,), dtype=dtype, device=device)
                pad[: t.shape[0]] = t
                padded.append(pad)
            return torch.stack(padded, dim=0)
        else:
            max_rows = max(t.shape[0] for t in tensors)
            max_cols = max(t.shape[1] for t in tensors)
            padded_list = []
            for t in tensors:
                pad = torch.zeros((max_rows, max_cols), dtype=dtype, device=device)
                pad[: t.shape[0], : t.shape[1]] = t
                padded_list.append(pad)
            return torch.stack(padded_list, dim=0)

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

            # Special-case LoRA-related auxiliary vectors that vary with rank across clients
            # e.g., '<prefix>.rank_mask' (1D of length r)
            if suffix in ("rank_mask",):
                # Normalize weights to sum=1
                tw = float(sum(weights))
                w_norm = [w / tw for w in weights] if tw > 0 else [1.0 / len(weights)] * len(weights)
                # Right-pad vectors to max length
                if any(v.dim() != 1 for v in values):
                    raise ValueError(f"rank_mask expected 1D tensors, got {[tuple(v.shape) for v in values]}")
                stacked = FedAggregator_ZeroPad.pad_tensors_to_max_shape(values)  # (N, Lmax)
                wt = torch.as_tensor(w_norm, dtype=stacked.dtype, device=stacked.device).view(len(w_norm), 1)
                aggregated[key] = (stacked * wt).sum(dim=0)
                continue

            if suffix in lora_suffixes:
                aggregated[key] = FedAggregator_ZeroPad.aggregate_lora_tensors(values, weights)
            else:
                # Generic weighted average for non-LoRA params
                # If shapes mismatch (common for LoRA-related side tensors), zero-pad to max shape first.
                same_shape = all(tuple(values[0].shape) == tuple(v.shape) for v in values)
                if same_shape:
                    stacked = torch.stack(values, dim=0)  # (N, ...)
                else:
                    dims = {int(v.dim()) for v in values}
                    # Allow mix of scalars (0D) and vectors (1D): upcast scalars to 1D then pad
                    if dims.issubset({0, 1}):
                        vecs = [v.view(1) if v.dim() == 0 else v for v in values]
                        stacked = FedAggregator_ZeroPad.pad_tensors_to_max_shape(vecs)
                    # Allow uniform 1D or uniform 2D: pad to max
                    elif (len(dims) == 1) and (next(iter(dims)) in (1, 2)):
                        stacked = FedAggregator_ZeroPad.pad_tensors_to_max_shape(values)
                    else:
                        raise RuntimeError(
                            f"Cannot aggregate key '{key}' with mismatched shapes/dims: "
                            f"shapes={[tuple(v.shape) for v in values]}, dims={sorted(dims)}"
                        )
                view_shape = (len(weights),) + (1,) * (stacked.dim() - 1)
                weight_tensor = torch.as_tensor(weights, dtype=stacked.dtype, device=stacked.device).view(*view_shape)
                weighted_sum = (stacked * weight_tensor).sum(dim=0)
                aggregated[key] = weighted_sum

        return aggregated

