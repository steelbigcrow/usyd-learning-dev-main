# ml_data_loader/loader/_dataset_loader_mrpc.py
from __future__ import annotations
from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs
from torch.utils.data import DataLoader, Dataset

try:
    from datasets import load_dataset
except Exception as e:
    raise ImportError(
        "HuggingFace 'datasets' is required for MRPC. Please `pip install datasets`."
    ) from e


class _GluePairConcatDataset(Dataset):
    def __init__(self, hf_split, text_key1: str = "sentence1", text_key2: str = "sentence2",
                 label_key: str = "label", sep_token: str = " [SEP] "):
        self.data = hf_split
        self.text_key1 = text_key1
        self.text_key2 = text_key2
        self.label_key = label_key
        self.sep_token = sep_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        s1 = ex[self.text_key1]
        s2 = ex[self.text_key2]
        label = int(ex[self.label_key])
        text = f"{s1}{self.sep_token}{s2}"
        return (label, text)


class DatasetLoader_MRPC(DatasetLoader):
    def __init__(self):
        super().__init__()

    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        if getattr(args, "text_collate_fn", None) is None:
            raise ValueError("MRPC requires args.text_collate_fn (bind your tokenizer & vocab).")

        ds = load_dataset("glue", "mrpc", cache_dir=getattr(args, "root", None))

        load_train = bool(getattr(args, "is_load_train_set", True))
        load_test  = bool(getattr(args, "is_load_test_set", False))

        bs       = getattr(args, "batch_size", 32)
        nw       = getattr(args, "num_workers", 0)
        shuffle  = bool(getattr(args, "shuffle", True))
        collate  = args.text_collate_fn
        test_bs  = getattr(args, "test_batch_size", None) or bs
        sep      = getattr(args, "sep_token", " [SEP] ")

        if load_train:
            train_ds = _GluePairConcatDataset(ds["train"], sep_token=sep)
            self._dataset = train_ds
            self._data_loader = DataLoader(
                train_ds, batch_size=bs, shuffle=shuffle, num_workers=nw, collate_fn=collate
            )
            self.data_sample_num = len(train_ds)

        if load_test:
            val_ds = _GluePairConcatDataset(ds["validation"], sep_token=sep)
            self._test_dataset = val_ds
            self._test_data_loader = DataLoader(
                val_ds, batch_size=test_bs, shuffle=False, num_workers=nw, collate_fn=collate
            )

        self.task_type = "nlp"
        return



        
