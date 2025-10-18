from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from ...ml_data_loader import CustomDataset
from .skewed_longtail_spec import SkewedLongtailSpec, SparseSpec


@dataclass
class SkewedLongtailArgs:
    """Arguments controlling the final per-client dataset or loader creation."""

    batch_size: int = 64
    shuffle: bool = False
    num_workers: int = 0
    return_loaders: bool = False  # If True, returns DataLoaders; otherwise CustomDataset


class SkewedLongtailPartitioner:
    """
    Standalone skewed long-tail non-IID partitioner.

    This class decouples from existing non-IID implementations and uses only
    lightweight utilities (CustomDataset) to form outputs compatible with the
    rest of the codebase without introducing tight coupling.
    """

    def __init__(self, base_loader: DataLoader):
        self._base_loader = base_loader
        self._x_train: torch.Tensor | None = None
        self._y_train: torch.Tensor | None = None
        self._label_ids: List[int] | None = None

        self._load_data()

    @property
    def x_train(self) -> torch.Tensor:
        assert self._x_train is not None
        return self._x_train

    @property
    def y_train(self) -> torch.Tensor:
        assert self._y_train is not None
        return self._y_train

    @property
    def label_ids(self) -> List[int]:
        assert self._label_ids is not None
        return self._label_ids

    def _load_data(self) -> None:
        """
        Materialize the entire DataLoader into contiguous tensors for partitioning.
        """
        images_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        for images, labels in self._base_loader:
            if not torch.is_tensor(images):
                images = torch.as_tensor(images)
            if not torch.is_tensor(labels):
                labels = torch.as_tensor(labels)
            images_list.append(images)
            labels_list.append(labels)

        self._x_train = torch.cat(images_list, dim=0)
        self._y_train = torch.cat(labels_list, dim=0).view(-1)

        uniq = torch.unique(self._y_train).tolist()
        self._label_ids = [int(x) for x in sorted(uniq)]

    def _available_counts(self) -> List[int]:
        counts: List[int] = []
        y = self.y_train
        for lbl in self.label_ids:
            counts.append(int((y == lbl).sum().item()))
        return counts

    def _label_indices(self) -> Dict[int, torch.Tensor]:
        idx_map: Dict[int, torch.Tensor] = {}
        y = self.y_train
        for lbl in self.label_ids:
            idx = torch.nonzero(y == lbl, as_tuple=False).view(-1)
            # Shuffle within label to avoid correlation among clients
            perm = torch.randperm(idx.numel())
            idx_map[lbl] = idx[perm]
        return idx_map

    def partition(
        self,
        spec: SparseSpec,
        args: SkewedLongtailArgs | None = None,
    ) -> List[CustomDataset] | List[DataLoader]:
        """
        Partitions the base loader into client datasets according to a sparse
        skewed long-tail specification.

        Args:
            spec: Sparse client->label->weight mapping.
            args: Optional control args for output creation.

        Returns:
            List of CustomDataset or DataLoader, one per client (in order of
            client id from 0..N-1). Clients with zero samples get an empty
            dataset/loader to preserve positional alignment.
        """
        if args is None:
            args = SkewedLongtailArgs()

        # Build weights using dataset's label order, ensure client rows align
        spec_obj = SkewedLongtailSpec(spec)

        weights, label_ids = spec_obj.to_dense_weights(
            label_ids=self.label_ids,
            num_clients=spec_obj.client_count(),
        )

        # Normalize per label to available counts (use entire dataset)
        counts = SkewedLongtailSpec.normalize_weights_to_counts(
            weights, self._available_counts()
        )

        # Make per-label index pools and consume as we allocate
        lbl_to_idx = self._label_indices()
        lbl_pos: Dict[int, int] = {lbl: 0 for lbl in label_ids}

        client_datasets: List[CustomDataset] = []

        for ci in range(weights.shape[0]):
            sel_indices: List[int] = []
            for col, lbl in enumerate(label_ids):
                k = int(counts[ci, col].item())
                if k <= 0:
                    continue
                pool = lbl_to_idx[lbl]
                start = lbl_pos[lbl]
                end = start + k
                if end > pool.numel():
                    raise ValueError(
                        f"Label {lbl}: requested {k} exceeds available {pool.numel()-start} (total {pool.numel()})"
                    )
                take = pool[start:end]
                lbl_pos[lbl] = end
                sel_indices.extend(take.tolist())

            if len(sel_indices) == 0:
                # Return an empty dataset placeholder to keep alignment
                empty_x = self.x_train[:0]
                empty_y = self.y_train[:0]
                ds = CustomDataset(empty_x, empty_y)
                client_datasets.append(ds)
                continue

            sel = torch.tensor(sel_indices, dtype=torch.int64)
            data_x = self.x_train.index_select(dim=0, index=sel)
            data_y = self.y_train.index_select(dim=0, index=sel)
            ds = CustomDataset(data_x, data_y)
            client_datasets.append(ds)

        if not args.return_loaders:
            return client_datasets

        # Wrap datasets into DataLoaders
        loaders: List[DataLoader] = []
        for ds in client_datasets:
            loader = CustomDataset.create_custom_loader(
                ds,
                batch_size=args.batch_size,
                shuffle=args.shuffle,
                num_workers=args.num_workers,
            )
            loaders.append(loader)

        return loaders

