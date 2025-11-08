from __future__ import annotations
from .nn_model import AbstractNNModel, NNModelArgs


class NNModelFactory:
    """
    NN Model factory static class
    """
    @staticmethod
    def create_args(config_dict: dict|None = None, is_clone_dict=False) -> NNModelArgs:
        return NNModelArgs(config_dict, is_clone_dict)

    @staticmethod
    def create(args: NNModelArgs) -> AbstractNNModel:
        match args.model_type:
            case "mnist_nn_brenden":
                from .models._nn_model_mnist_nn_brenden import NNModel_MnistNNBrenden
                return NNModel_MnistNNBrenden().create_model(args)
            case "adalora_simple_mlp":
                from .adalora._nn_model_adalora_mlp import NNModel_AdaLoRAMLP
                return NNModel_AdaLoRAMLP().create_model(args)
            case "simple_lora_mlp":
                from .lora._nn_model_simple_lora_mlp import NNModel_SimpleLoRAMLP
                return NNModel_SimpleLoRAMLP().create_model(args)
            case "capstone_mlp":
                from .models._nn_model_capstone_mlp import NNModel_CapstoneMLP
                return NNModel_CapstoneMLP().create_model(args)
            case "simple_mlp":
                from .models._nn_model_simple_mlp import NNModel_SimpleMLP
                return NNModel_SimpleMLP().create_model(args)
            case "cifar_convnet":
                from .models._nn_model_cifar_convnet import NNModel_CifarConvnet
                return NNModel_CifarConvnet().create_model(args)
            case "simple_lora_mlp":
                from .lora._nn_model_simple_lora_mlp import NNModel_SimpleLoRAMLP
                return NNModel_SimpleLoRAMLP().create_model(args)
            case "simple_lora_cnn":
                from .lora._nn_model_simple_lora_cnn import NNModel_SimpleLoRACNN
                return NNModel_SimpleLoRACNN().create_model(args)
            case "cifar_lora_cnn":
                from .lora._nn_model_cifar_lora_cnn import NNModel_CifarLoRACNN
                return NNModel_CifarLoRACNN().create_model(args)
            case "cifar10_lora_vit":
                from .vit._nn_model_cifar10_lora_vit import NNModel_ViTMSLoRACIFAR10
                return NNModel_ViTMSLoRACIFAR10().create_model(args)
            case "imdb_lora_transformer":
                from .lora._nn_model_imdb_lora_transformer import NNModel_ImdbMSLoRATransformer
                return NNModel_ImdbMSLoRATransformer().create_model(args)
            case "distilroberta_base_seqcls":
                from .models._nn_model_distilroberta_base_seqcls import NNModel_DistilRoBERTaBaseSeqCls
                return NNModel_DistilRoBERTaBaseSeqCls().create_model(args)
            case "efficientnet_b0":
                from .models._nn_model_efficientnet_b0 import NNModel_EfficientNetB0
                return NNModel_EfficientNetB0().create_model(args)
            case "roberta_base_seqcls":
                from .models._nn_model_roberta_base_seqcls import NNModel_RoBERTaBaseSeqCls
                return NNModel_RoBERTaBaseSeqCls().create_model(args)
            case "resnet18":
                from .models._nn_model_resnet18 import NNModel_ResNet18
                return NNModel_ResNet18().create_model(args)
            case "lenet":
                from .models._nn_model_lenet import NNModel_LeNet
                return NNModel_LeNet().create_model(args)
            case "lstm":
                from .models._nn_model_lstm import NNModel_LSTM
                return NNModel_LSTM().create_model(args)
            case "squeezenet1_1":
                from .models._nn_model_squeezenet1_1 import NNModel_SqueezeNet1_1
                return NNModel_SqueezeNet1_1().create_model(args)
            case "distilbert":
                from .models._nn_model_distilbert import NNModel_DistilBERT
                return NNModel_DistilBERT().create_model(args)
            case "textcnn":
                from .models._nn_model_textcnn import NNModel_TextCNN
                return NNModel_TextCNN().create_model(args)
            case "resnet9":
                from .models._nn_model_resnet9 import NNModel_ResNet9
                return NNModel_ResNet9().create_model(args)

        raise ValueError(f"Unknown mode type '{args.model_type}'")
