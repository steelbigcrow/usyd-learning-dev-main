from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader

from ...ml_data_loader import CustomDataset


@dataclass
class FeatureShiftedArgs:
    """Arguments controlling the final per-client dataset or loader creation."""

    batch_size: int = 64
    shuffle: bool = False
    num_workers: int = 0
    return_loaders: bool = False  # If True, returns DataLoaders; otherwise CustomDataset


class FeatureShiftedPartitioner:
    """
    Partition a dataset so that each label/class is split as evenly as possible
    across N clients ("feature-shifted" in the repo configs). Remainders are
    assigned one-by-one to the last clients, matching the YAML patterns under
    src/test/fl_lora_sample/yamls/data_distribution/*_feature_shift.yaml.

    The class can also consume an explicit client-by-class integer count matrix
    and allocate data accordingly.
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

    # --- internals -----------------------------------------------------------------
    def _load_data(self) -> None:
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

    # --- public API ----------------------------------------------------------------
    def make_balanced_counts(self, num_clients: int) -> torch.Tensor:
        """
        Build a [num_clients, num_labels] integer count matrix that splits each
        label as evenly as possible, assigning remainders to the last clients.
        """
        if num_clients <= 0:
            raise ValueError("num_clients must be > 0")

        avail = self._available_counts()
        n_labels = len(avail)
        counts = torch.zeros((num_clients, n_labels), dtype=torch.int64)

        for j, total in enumerate(avail):
            base = total // num_clients
            rem = total % num_clients
            if base > 0:
                counts[:, j] = base
            if rem > 0:
                # Assign the +1 remainders to the last `rem` clients
                counts[-rem:, j] += 1

        return counts

    def partition_from_counts(
        self,
        counts: Sequence[Sequence[int]],
        args: FeatureShiftedArgs | None = None,
    ) -> List[CustomDataset] | List[DataLoader]:
        """
        Partition according to an explicit client-by-class count matrix. The
        column order is assumed to correspond to this dataset's `label_ids`.
        """
        if args is None:
            args = FeatureShiftedArgs()

        # Validate shape and convert to tensor
        mat = torch.as_tensor(counts, dtype=torch.int64)
        if mat.dim() != 2:
            raise ValueError("counts must be a 2D matrix [num_clients, num_labels]")
        if mat.shape[1] != len(self.label_ids):
            raise ValueError(
                f"counts has {mat.shape[1]} columns but dataset exposes {len(self.label_ids)} labels"
            )
        if (mat < 0).any():
            raise ValueError("counts must be non-negative")

        # Prepare per-label index pools and consume sequentially per client
        lbl_to_idx = self._label_indices()
        lbl_pos: Dict[int, int] = {lbl: 0 for lbl in self.label_ids}

        client_datasets: List[CustomDataset] = []

        for ci in range(mat.shape[0]):
            sel_indices: List[int] = []
            for col, lbl in enumerate(self.label_ids):
                k = int(mat[ci, col].item())
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

    def partition_evenly(
        self,
        num_clients: int,
        args: FeatureShiftedArgs | None = None,
    ) -> List[CustomDataset] | List[DataLoader]:
        """
        Convenience wrapper that builds the balanced counts (with remainder
        handling) and partitions the dataset accordingly.
        """
        counts = self.make_balanced_counts(num_clients)
        return self.partition_from_counts(counts.tolist(), args=args)

