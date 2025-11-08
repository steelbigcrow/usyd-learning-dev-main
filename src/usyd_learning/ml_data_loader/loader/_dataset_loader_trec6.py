# ml_data_loader/loader/_dataset_loader_trec6.py
from __future__ import annotations
from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs
from torch.utils.data import DataLoader, Dataset

try:
    from datasets import load_dataset
except Exception as e:
    raise ImportError(
        "HuggingFace 'datasets' is required for TREC6. Please `pip install datasets`."
    ) from e


class _TupleDataset(Dataset):
    def __init__(self, hf_split, text_key: str, label_key: str = "label"):
        self.data = hf_split
        self.text_key = text_key
        self.label_key = label_key

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        return (int(ex[self.label_key]), ex[self.text_key])


class DatasetLoader_TREC6(DatasetLoader):
    def __init__(self):
        super().__init__()

    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        if getattr(args, "text_collate_fn", None) is None:
            raise ValueError("TREC6 requires args.text_collate_fn (bind your tokenizer & vocab).")

        # TREC dataset loading - try different methods
        try:
            # Method 1: Try with trust_remote_code
            ds = load_dataset("trec", cache_dir=getattr(args, "root", None), trust_remote_code=True)
        except Exception as e1:
            try:
                # Method 2: Try loading from a different source
                import datasets
                # Use revision parameter to get an older version that might work
                ds = datasets.load_dataset("trec", cache_dir=getattr(args, "root", None), revision="main")
            except Exception as e2:
                # Method 3: Try with a specific config
                try:
                    ds = load_dataset("trec", "coarse", cache_dir=getattr(args, "root", None), trust_remote_code=True)
                except Exception as e3:
                    raise ImportError(
                        f"Failed to load TREC dataset. Tried multiple methods. "
                        f"Last error: {e3}. "
                        f"Note: TREC dataset may require specific datasets library version or manual setup."
                    ) from e3

        load_train = bool(getattr(args, "is_load_train_set", True))
        load_test  = bool(getattr(args, "is_load_test_set", False))

        bs       = getattr(args, "batch_size", 32)
        nw       = getattr(args, "num_workers", 0)
        shuffle  = bool(getattr(args, "shuffle", True))
        collate  = args.text_collate_fn
        test_bs  = getattr(args, "test_batch_size", None) or bs

        # 训练 DataLoader
        if load_train:
            train_ds = _TupleDataset(ds["train"], text_key="text", label_key="label-coarse")
            self._dataset = train_ds
            self._data_loader = DataLoader(
                train_ds, batch_size=bs, shuffle=shuffle, num_workers=nw, collate_fn=collate
            )
            self.data_sample_num = len(train_ds)

        # 测试 DataLoader
        if load_test:
            if "test" in ds:
                test_ds = _TupleDataset(ds["test"], text_key="text", label_key="label-coarse")
            elif "validation" in ds:
                test_ds = _TupleDataset(ds["validation"], text_key="text", label_key="label-coarse")
            else:
                test_ds = None
            
            if test_ds is not None:
                self._test_dataset = test_ds
                self._test_data_loader = DataLoader(
                    test_ds, batch_size=test_bs, shuffle=False, num_workers=nw, collate_fn=collate
                )

        self.task_type = "nlp"
        return

