from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


SparseSpec = Dict[int, Dict[int, float]]


@dataclass
class SkewedLongtailSpec:
    """
    Holds and transforms a sparse skewed long-tail non-IID specification into a
    dense client-by-class matrix and normalizes it against available label counts.

    This class is independent from other non-IID modules in the repository to
    keep coupling minimal.
    """

    spec: SparseSpec

    def client_count(self) -> int:
        if not self.spec:
            return 0
        # Clients are keyed by int; allow sparse/non-contiguous keys.
        return max(self.spec.keys()) + 1

    def known_label_ids(self) -> List[int]:
        """
        Returns sorted label ids that appear in the spec.
        """
        labels = set()
        for _, d in self.spec.items():
            labels.update(d.keys())
        return sorted(int(x) for x in labels)

    def to_dense_weights(
        self, label_ids: List[int] | None = None, num_clients: int | None = None
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Builds a dense weight matrix from the sparse spec.

        Args:
            label_ids: Optional explicit label id ordering. If None, uses labels
                       that appear in the spec, sorted ascending.
            num_clients: If provided, fixes the number of rows. Otherwise uses
                         max(spec.keys())+1.

        Returns:
            (weights, label_ids):
                - weights: Float tensor of shape [num_clients, num_labels]
                - label_ids: The label ordering used for the columns
        """
        if num_clients is None:
            num_clients = self.client_count()
        if label_ids is None:
            label_ids = self.known_label_ids()

        col_index = {c: j for j, c in enumerate(label_ids)}
        mat = torch.zeros((num_clients, len(label_ids)), dtype=torch.float32)

        for ci, label_map in self.spec.items():
            if ci < 0 or ci >= num_clients:
                # Skip out-of-range clients
                continue
            for lbl, w in label_map.items():
                if lbl in col_index:
                    mat[ci, col_index[lbl]] = float(w)
                # Unknown labels (not in label_ids) are ignored

        return mat, label_ids

    @staticmethod
    def normalize_weights_to_counts(
        weights: torch.Tensor,
        available_counts: List[int] | torch.Tensor,
    ) -> torch.Tensor:
        """
        Converts per-label relative weights to integer sample counts using the
        largest-remainder method so that, for each label column j:
            sum_i counts[i, j] == available_counts[j]
        while respecting the relative proportions in `weights`.

        Args:
            weights: Float tensor [num_clients, num_labels]
            available_counts: 1D list/tensor of length num_labels

        Returns:
            counts: Int64 tensor [num_clients, num_labels]
        """
        if not isinstance(available_counts, torch.Tensor):
            available_counts = torch.tensor(available_counts, dtype=torch.int64)
        else:
            available_counts = available_counts.to(dtype=torch.int64)

        n_c, n_l = weights.shape
        counts = torch.zeros((n_c, n_l), dtype=torch.int64)

        # For each label/column independently, apply proportional allocation
        for j in range(n_l):
            avail = int(available_counts[j].item())
            col = weights[:, j]
            total_w = float(col.sum().item())

            if avail <= 0 or total_w <= 0:
                # Nothing to allocate for this label
                continue

            # Proportional quotas
            quotas = col / total_w * float(avail)
            floors = torch.floor(quotas).to(torch.int64)
            counts[:, j] = floors

            remainder = avail - int(floors.sum().item())
            if remainder <= 0:
                continue

            # Largest remainder: assign the remaining one-by-one by fractional part
            frac = (quotas - floors.to(quotas.dtype))
            # Get indices sorted descending by fractional part
            order = torch.argsort(frac, descending=True)

            # Distribute leftover to top-k fractional parts
            for k in range(min(remainder, n_c)):
                idx = int(order[k].item())
                counts[idx, j] += 1

            # If remainder > n_c, continue a second pass (rare)
            left = remainder - min(remainder, n_c)
            if left > 0:
                # Cycle through clients if extreme imbalance
                # This preserves total counts and monotonicity.
                for k in range(left):
                    idx = int(order[k % n_c].item())
                    counts[idx, j] += 1

        return counts

