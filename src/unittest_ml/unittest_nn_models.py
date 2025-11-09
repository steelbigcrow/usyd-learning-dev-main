'''
pip install torch torchvision transformers torchsummary
python src/unittest_ml/unittest_nn_models.py
'''
# Init startup path, change current path to test py file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

import torch
from usyd_learning.ml_models import NNModelFactory
from usyd_learning.ml_utils import ConfigLoader


def main():

    # Load config from yaml file
    config = ConfigLoader.load("./test_data/node_config_template_client.yaml")

    #create args
    args = NNModelFactory.create_args(config)
    print(args)

    # nn_model = NNModelFactory.create(args)

    # print(nn_model)

    # 1) capstone_mlp
    args.model_type = "capstone_mlp"
    args.num_classes = 10
    model = NNModelFactory.create(args)
    x = torch.randn(2, 1, 28, 28)
    with torch.no_grad():
        out = model(x)
    print("[OK] capstone_mlp forward ->", out.shape)

    # 2) simple_mlp
    args = NNModelFactory.create_args(config)
    args.model_type = "simple_mlp"
    args.input_dim = 28 * 28      
    args.hidden_dim = 64          
    args.output_dim = 10        
    args.softmax_dim = 1         
    model = NNModelFactory.create(args)
    x = torch.randn(2, 1, 28, 28)
    with torch.no_grad():
        out = model(x)
    print("[OK] simple_mlp forward ->", out.shape)

    # 3) resnet18
    args = NNModelFactory.create_args(config)
    args.model_type = "resnet18"
    args.num_classes = 10
    args.pretrained = False
    model = NNModelFactory.create(args)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    print("[OK] resnet18 forward ->", out.shape)

    # 4) efficientnet_b0
    args = NNModelFactory.create_args(config)
    args.model_type = "efficientnet_b0"
    args.num_classes = 10
    args.pretrained = False
    model = NNModelFactory.create(args)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    print("[OK] efficientnet_b0 forward ->", out.shape)

    # 5) cifar_convnet
    args = NNModelFactory.create_args(config)
    args.model_type = "cifar_convnet"
    args.num_classes = 10
    model = NNModelFactory.create(args)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    print("[OK] cifar_convnet forward ->", out.shape)

    # 6) mnist_nn_brenden
    args = NNModelFactory.create_args(config)
    args.model_type = "mnist_nn_brenden"
    args.num_classes = 10
    model = NNModelFactory.create(args)
    x = torch.randn(2, 28 * 28) 
    with torch.no_grad():
        out = model(x)
    print("[OK] mnist_nn_brenden forward ->", out.shape)

    # 7) roberta_base_seqcls
    try:
        from transformers import AutoTokenizer, logging
        logging.set_verbosity_error()
        tok = AutoTokenizer.from_pretrained("roberta-base")
        batch = tok(
            ["dummy 1", "dummy 2"],
            padding="max_length",
            truncation=True,
            max_length=16,
            return_tensors="pt",
        )
        args = NNModelFactory.create_args(config)
        args.model_type = "roberta_base_seqcls"
        args.num_classes = 2
        model = NNModelFactory.create(args)
        with torch.no_grad():
            out = model(batch)
        print("[OK] roberta_base_seqcls forward ->", out.shape)
    except ImportError as e:
        print("[SKIP] roberta_base_seqcls: transformers not installed")
    except Exception as e:
        print("[FAIL] roberta_base_seqcls:", e)

    # 8) lenet
    args = NNModelFactory.create_args(config)
    args.model_type = "lenet"        
    args.in_channels = 1              
    args.num_classes = 10
    model = NNModelFactory.create(args)
    x = torch.randn(2, 1, 28, 28)     
    with torch.no_grad():
        out = model(x)
    print("[OK] lenet forward ->", out.shape)


if __name__ == "__main__":
    main()