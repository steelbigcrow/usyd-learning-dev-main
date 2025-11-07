from __future__ import annotations

from .fed_aggregator_abc import AbstractFedAggregator
from .fed_aggregator_args import FedAggregatorArgs


class FedAggregatorFactory:
    '''
    ' Fed aggregator factory
    '''

    @staticmethod
    def create_args(config_dict: dict, is_clone_dict: bool = False) -> FedAggregatorArgs:
        """
        " Static method to create fed aggregator args
        """
        return FedAggregatorArgs(config_dict, is_clone_dict)

    @staticmethod
    def create_aggregator(args: FedAggregatorArgs) -> AbstractFedAggregator:
        match args.method:
            case "fedavg":
                from .methods._fed_aggregator_fedavg import FedAggregator_FedAvg
                print("Using FedAvg aggregator")
                return FedAggregator_FedAvg(args)
            case "svd":
                from .methods._fed_aggregator_svd import FedAggregator_SVD
                print("Using SVD (balanced sqrt) aggregator")
                return FedAggregator_SVD(args)
            case "rbla":
                from .methods._fed_aggregator_rbla import FedAggregator_RBLA
                print("Using RBLA aggregator")
                return FedAggregator_RBLA(args)
            case "rbla_hetero":
                from .methods._fed_aggregator_rbla_hetero import FedAggregator_RBLA_HET
                print("Using RBLA-HET (heterogeneous LoRA) aggregator")
                return FedAggregator_RBLA_HET(args)
            case "sp":
                from .methods._fed_aggregator_sp import FedAggregator_SP
                print("Using Sum-Product aggregator")
                return FedAggregator_SP(args)
            case "zp" | "zeropad" | "zero_pad" | "zero-padding" | "zero_padding":
                from .methods._fed_aggregator_zeropad import FedAggregator_ZeroPad
                print("Using Zero-Pad (LoRA) aggregator")
                return FedAggregator_ZeroPad(args)
            case _:
                raise ValueError(f"Unsupported aggregation method: {args.method}")
        return

