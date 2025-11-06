from __future__ import annotations

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class DatasetLoader_OxfordPets(DatasetLoader):
    def __init__(self):
        super().__init__()

    # override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        if args.transform is None:
            args.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

        self.task_type = "cv"

        split = getattr(args, "split", None) or ("trainval" if args.is_train else "test")
        self._dataset = OxfordIIITPet(
            root=args.root,
            split=split,                
            target_types="category",     
            transform=args.transform,
            download=args.is_download,
        )
        self._data_loader = DataLoader(
            self._dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
        )

    
        test_transform   = getattr(args, "test_transform", None) or args.transform
        test_batch_size  = getattr(args, "test_batch_size", None) or args.batch_size
        self._test_dataset = OxfordIIITPet(
            root=args.root,
            split="test",
            target_types="category",
            transform=test_transform,
            download=args.is_download,
        )
        self._test_data_loader = DataLoader(
            self._test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        return
