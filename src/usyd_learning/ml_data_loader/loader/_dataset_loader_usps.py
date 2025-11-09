# usyd_learning/ml_data_loader/loader/_dataset_loader_usps.py
from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs

try:
    from datasets import load_dataset  # type: ignore
    _HAS_HF = True
except Exception:
    _HAS_HF = False

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class _HF_USPS_Wrapper(Dataset):
    def __init__(self, hf_split, transform):
        self.split = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        ex = self.split[idx]
        img = ex["image"]  # PIL.Image 或 numpy.ndarray
        if self.transform is not None:
            img = self.transform(img)
        label = int(ex["label"])
        return img, label


class _NumpyUSPSWrapper(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, transform):
        self.X = X
        self.y = y.astype(int)
        self.transform = transform

        if self.X.ndim == 2 and self.X.shape[1] == 256:
            self.X = self.X.reshape(-1, 16, 16)
        if self.X.dtype != np.uint8:
            Xmax = float(self.X.max())
            if Xmax <= 1.0:
                self.X = (self.X * 255.0).round().astype(np.uint8)
            else:
                self.X = self.X.clip(0, 255).astype(np.uint8)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = Image.fromarray(self.X[idx], mode="L")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.y[idx])


class DatasetLoader_USPS(DatasetLoader):
    def __init__(self):
        super().__init__()

    # 选择 transform（若上层已给 args.transform，则尊重上层）
    def _build_transform(self, args: DatasetLoaderArgs):
        if args.transform is not None:
            return args.transform
        if getattr(args, "imagenet_mode", True):
            return transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])

    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        self.task_type = "cv"

        split = (getattr(args, "split", None) or ("train" if args.is_train else "test")).lower()
        if split not in ("train", "test"):
            raise ValueError(f"USPS split must be 'train' or 'test', got: {split}")
        is_train = (split == "train")

        transform = self._build_transform(args)
        cache_dir = getattr(args, "root", None)

        if _HAS_HF:
            try:
                ds = load_dataset("tensorflow_datasets", "usps", cache_dir=cache_dir)
                if split not in ds:
                    raise ValueError(f"HF(TFDS) 'usps' split '{split}' not found. Available: {list(ds.keys())}")

                self._dataset = _HF_USPS_Wrapper(ds[split], transform)
                self._data_loader = DataLoader(
                    self._dataset,
                    batch_size=args.batch_size,
                    shuffle=args.shuffle,
                    num_workers=args.num_workers,
                    pin_memory=getattr(args, "pin_memory", False),
                )

                test_transform  = getattr(args, "test_transform", None) or transform
                test_batch_size = getattr(args, "test_batch_size", None) or args.batch_size
                self._test_dataset = _HF_USPS_Wrapper(ds["test"], test_transform)
                self._test_data_loader = DataLoader(
                    self._test_dataset,
                    batch_size=test_batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=getattr(args, "pin_memory", False),
                )
                return
            except Exception as e_hf:
                pass  

        try:
            from sklearn.datasets import fetch_openml  # type: ignore
        except Exception as e:
            raise ImportError("USPS need：`pip install tensorflow-datasets` or `pip install scikit-learn`") from e

        try:
            ds_ = fetch_openml("usps", version=1, as_frame=False, parser="auto")
        except Exception:
            ds_ = fetch_openml("USPS", version=1, as_frame=False, parser="auto")

        X = ds_.data  # (N, 256)
        y = ds_.target.astype(int)

        n_train = 7291
        X_tr, y_tr = X[:n_train], y[:n_train]
        X_te, y_te = X[n_train:], y[n_train:]

        data_X, data_y = (X_tr, y_tr) if is_train else (X_te, y_te)
        self._dataset = _NumpyUSPSWrapper(data_X, data_y, transform)
        self._data_loader = DataLoader(
            self._dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            pin_memory=getattr(args, "pin_memory", False),
        )

        test_transform  = getattr(args, "test_transform", None) or transform
        test_batch_size = getattr(args, "test_batch_size", None) or args.batch_size
        self._test_dataset = _NumpyUSPSWrapper(X_te, y_te, test_transform)
        self._test_data_loader = DataLoader(
            self._test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=getattr(args, "pin_memory", False),
        )
        return
