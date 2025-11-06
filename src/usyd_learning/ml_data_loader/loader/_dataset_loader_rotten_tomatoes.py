from __future__ import annotations

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs
from ..dataset_loader_util import DatasetLoaderUtil

from torch.utils.data import DataLoader, Dataset
from functools import partial

try:
    from datasets import load_dataset
except Exception as e:  # pragma: no cover
    raise ImportError("pip install datasets") from e


class _TupleDataset(Dataset):
    def __init__(self, hf_split, text_key: str, label_key: str = "label"):
        self.data = hf_split
        self.text_key = text_key
        self.label_key = label_key
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        ex = self.data[idx]
        return int(ex[self.label_key]), ex[self.text_key]


class DatasetLoader_RottenTomatoes(DatasetLoader):
    def __init__(self):
        super().__init__()

    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        ds = load_dataset("rotten_tomatoes", cache_dir=getattr(args, "root", None))
        split = args.split or "train"
        if split not in ds:
            raise ValueError(f"rotten_tomatoes split '{split}' not found. Available: {list(ds.keys())}.")

        dataset = _TupleDataset(ds[split], text_key="text", label_key="label")
        self._dataset = dataset
        self._data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            collate_fn=args.text_collate_fn,
        )
        return


