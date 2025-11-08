'''
pip install -U pip
pip install torch torchvision datasets torchtext colorama certifi scipy tensorflow-datasets scikit-learn
export SSL_CERT_FILE=$(python -c 'import certifi; print(certifi.where())')
python src/unittest_ml/unittest_data_loader.py
'''
from __future__ import annotations

# Init startup path, change current path to test py file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

from usyd_learning.ml_utils import console, ConfigLoader
from usyd_learning.ml_data_loader import DatasetLoaderFactory, DatasetLoader

def after_create_fn(loader: DatasetLoader):
    console.info(f"\nAfter create fn object: {loader}")
    return

def simple_text_collate_fn(batch):
    # batch: List[(label:int, text:str)]
    try:
        import torch
        labels, texts = zip(*batch)
        return torch.tensor(labels), list(texts)
    except Exception:
        return batch
    
def test_dataset_loader(args, dataset_type):
    args.dataset_type = dataset_type
    args.root = "../../../.dataset"
    args.is_train = True
    args.is_download = True

    dataset_loader = DatasetLoaderFactory.create(args, after_create_fn)
    print(dataset_loader)
    return


def main():
    yaml_file_name = './test_data/node_config_template_client.yaml'

    # Form yaml file
    console.out(f"Test from yaml file: {yaml_file_name}")
    console.out("------------- Begin ---------------")

    yaml = ConfigLoader.load(yaml_file_name)
    # console.out(yaml)

    args = DatasetLoaderFactory.create_args(yaml)
    
    test_dataset_loader(args, "mnist")
    test_dataset_loader(args, "fmnist")

    args.split = 'balanced'
    test_dataset_loader(args, "emnist")

    test_dataset_loader(args, "kmnist")
    test_dataset_loader(args, "qmnist")

    test_dataset_loader(args, "cifar10")
    test_dataset_loader(args, "cifar100")

    args.split = 'train'
    test_dataset_loader(args, "stl10")
    args.text_collate_fn = simple_text_collate_fn
    test_dataset_loader(args, "svhn")
    test_dataset_loader(args, "rotten_tomatoes")
    test_dataset_loader(args, "emotion")
    test_dataset_loader(args, "poem_sentiment")
    test_dataset_loader(args, "clinc_oos")
    test_dataset_loader(args, "cola")
    test_dataset_loader(args, "sst2")
    test_dataset_loader(args, "mrpc")
    test_dataset_loader(args, "subj")
    # TREC6 dataset may have compatibility issues with newer datasets library versions
    try:
        test_dataset_loader(args, "trec6")
    except Exception as e:
        console.out(f"[SKIP] trec6: {e}")
    args.split = 'train'
    # Flowers102 and OxfordPets require tensorflow-datasets package
    try:
        test_dataset_loader(args, "flowers102")
    except Exception as e:
        console.out(f"[SKIP] flowers102: {e}")
    test_dataset_loader(args, "stl10")
    args.split = 'trainval' 
    try:
        test_dataset_loader(args, "oxford_pets")
    except Exception as e:
        console.out(f"[SKIP] oxford_pets: {e}")
    if hasattr(args, "split"):
        delattr(args, "split") 
    args.is_train = True
    test_dataset_loader(args, "usps")
    # EuroSAT requires tensorflow-datasets package
    try:
        test_dataset_loader(args, "eurosat")
    except Exception as e:
        console.out(f"[SKIP] eurosat: {e}")
    # args.split = 'train'
    # test_dataset_loader(args, "imagenet")

    console.out("------------- End -----------------")
    return


if __name__ == "__main__":
    main()

