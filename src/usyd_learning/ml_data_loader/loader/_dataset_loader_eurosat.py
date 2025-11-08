# usyd_learning/ml_data_loader/loader/_dataset_loader_eurosat.py
from __future__ import annotations

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs

try:
    from datasets import load_dataset
    _HAS_HF = True
except Exception:
    _HAS_HF = False

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class _HF_EuroSAT_Wrapper(Dataset):
    def __init__(self, hf_split, transform):
        self.split = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        ex = self.split[idx]
        img = ex["image"]  # PIL.Image
        if self.transform is not None:
            img = self.transform(img)
        label = int(ex["label"])
        return img, label


class DatasetLoader_EuroSAT(DatasetLoader):
    def __init__(self):
        super().__init__()

    def _build_transform(self, args: DatasetLoaderArgs):
        if args.transform is not None:
            return args.transform
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        self.task_type = "cv"

        if not _HAS_HF:
            raise ImportError("EuroSAT requires 'datasets' library. Please `pip install datasets`.")

        cache_dir = getattr(args, "root", None)
        try:
            ds = load_dataset("tensorflow_datasets", "eurosat/rgb", cache_dir=cache_dir)
        except Exception as e:
            raise ImportError(
                f"Failed to load EuroSAT dataset: {e}. "
                f"Make sure 'tensorflow-datasets' is installed: pip install tensorflow-datasets"
            ) from e

        transform = self._build_transform(args)
        bs = getattr(args, "batch_size", 128)
        nw = getattr(args, "num_workers", 4)
        shuffle = bool(getattr(args, "shuffle", True))
        test_bs = getattr(args, "test_batch_size", None) or bs
        test_transform = getattr(args, "test_transform", None) or transform

        # 训练集
        if "train" in ds:
            train_ds = _HF_EuroSAT_Wrapper(ds["train"], transform)
            self._dataset = train_ds
            self._data_loader = DataLoader(
                train_ds, batch_size=bs, shuffle=shuffle, num_workers=nw
            )

        # 测试集
        if "test" in ds:
            test_ds = _HF_EuroSAT_Wrapper(ds["test"], test_transform)
            self._test_dataset = test_ds
            self._test_data_loader = DataLoader(
                test_ds, batch_size=test_bs, shuffle=False, num_workers=nw
            )
        elif "validation" in ds:
            val_ds = _HF_EuroSAT_Wrapper(ds["validation"], test_transform)
            self._test_dataset = val_ds
            self._test_data_loader = DataLoader(
                val_ds, batch_size=test_bs, shuffle=False, num_workers=nw
            )

        return

