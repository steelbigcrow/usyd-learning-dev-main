"""
Entry class of LoRA sample (feature-shifted variant).

This mirrors the skewed_longtail_noniid entry but uses a feature-shifted
partitioner. It consumes the server YAML's data_distribution (a client-by-class
count matrix) when provided, and falls back to an even per-label split across
clients otherwise.
"""

from __future__ import annotations

from usyd_learning.fed_node import FedNodeVars, FedNodeEventArgs
from usyd_learning.fed_runner import FedRunner
from usyd_learning.ml_utils import AppEntry, console
from usyd_learning.fl_algorithms.feature_shifted import (
    FeatureShiftedPartitioner,
    FeatureShiftedArgs,
)
from usyd_learning.ml_data_loader.dataset_loader_factory import DatasetLoaderFactory
from usyd_learning.ml_data_loader.dataset_loader_args import DatasetLoaderArgs


class SampleAppEntry(AppEntry):
    def __init__(self):
        super().__init__()

        # Define runner, client, server, edge yaml variables, can be set outside manually
        self.runner_yaml = None
        self.client_yaml = None
        self.edge_yaml = None
        self.server_yaml = None

    # override
    def run(self, device: str = "cpu", training_rounds: int = 50):

        # Set deterministic seeds early (before any model/data construction)
        try:
            import random
            import numpy as np
            import torch
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
        except Exception:
            pass

        # Yamls - if yamls are None, get yaml from app config file automatically
        if self.runner_yaml is None:
            self.runner_yaml = self.get_app_object("runner")
        if self.client_yaml is None:
            self.client_yaml = self.get_app_object("client_yaml")
        if self.edge_yaml is None:
            self.edge_yaml = self.get_app_object("edge_yaml")
        if self.server_yaml is None:
            self.server_yaml = self.get_app_object("server_yaml")

        # Training rounds
        self.fed_runner = FedRunner()  # Runner
        self.fed_runner.training_rounds = training_rounds
        self.fed_runner.with_yaml(self.runner_yaml)
        self.fed_runner.create_nodes()
        self.fed_runner.create_run_strategy()

        # Prepare server node and var
        server_var = FedNodeVars(self.server_yaml)
        server_var.prepare()  # create loader/model/optimizer/loss, etc.
        self.__attach_event_handler(server_var)
        server_var.owner_nodes = self.fed_runner.server_node  # Two way binding
        server_var.set_device(device)
        self.fed_runner.server_node.node_var = server_var
        self.fed_runner.server_node.prepare_strategy()
        self.fed_runner.server_node.node_var = server_var

        # Load data (train loader sits in server_var.data_loader)
        train_loader = server_var.data_loader

        # Feature-shifted handler: prefer explicit counts from YAML; otherwise split evenly
        counts = server_var.data_distribution
        partitioner = FeatureShiftedPartitioner(train_loader.data_loader)
        part_args = FeatureShiftedArgs(
            batch_size=64, shuffle=True, num_workers=0, return_loaders=False
        )

        allocated_data = None
        try:
            # Ensure label-wise shuffling inside the partitioner is deterministic
            try:
                import torch as _torch
                _torch.manual_seed(42)
            except Exception:
                pass
            if isinstance(counts, list) and len(counts) > 0 and isinstance(counts[0], list):
                allocated_data = partitioner.partition_from_counts(counts, part_args)
        except Exception as ex:
            console.warn(f"Feature-shifted explicit counts failed: {ex}")
            allocated_data = None

        if allocated_data is None:
            # Fallback to an even per-label split across clients
            num_clients = len(self.fed_runner.client_node_list)
            try:
                import torch as _torch
                _torch.manual_seed(42)
            except Exception:
                pass
            allocated_data = partitioner.partition_evenly(num_clients, part_args)

        # Wrap each allocated CustomDataset in DatasetLoaderFactory for trainers
        for i in range(len(allocated_data)):
            args = DatasetLoaderArgs(
                {
                    "name": "custom",
                    "root": "../../../.dataset",
                    "split": "",
                    "batch_size": 64,
                    "shuffle": True,
                    "num_workers": 0,
                    "is_download": True,
                    "is_load_train_set": True,
                    "is_load_test_set": True,
                    "dataset": allocated_data[i],
                }
            )
            allocated_data[i] = DatasetLoaderFactory().create(args)

        # Prepare each client node and var
        client_var_list = []
        for index, node in enumerate(self.fed_runner.client_node_list):
            client_var = FedNodeVars(self.client_yaml)
            # Set client-specific rank ratio if provided
            try:
                client_var.config_dict["nn_model"]["rank_ratio"] = self.server_yaml[
                    "rank_distribution"
                ]["rank_ratio_list"][index]
            except Exception:
                pass

            client_var.prepare()
            client_var.data_loader = allocated_data[index]
            client_var.data_sample_num = client_var.data_loader.data_sample_num
            client_var.set_device(device)
            client_var.trainer.set_train_loader(client_var.data_loader)
            self.__attach_event_handler(client_var)

            # Two way binding
            client_var.owner_nodes = node
            node.node_var = client_var
            node.prepare_strategy()
            client_var_list.append(client_var)

        # Reset RNG before training so DataLoader shuffles align across runs
        try:
            import torch as _torch
            _torch.manual_seed(42)
        except Exception:
            pass

        self.fed_runner.run()

        return

    # Attach events to node variable object
    def __attach_event_handler(self, node_var):
        node_var.attach_event("on_prepare_data_loader", self.on_prepare_data_loader)
        node_var.attach_event("on_prepare_model", self.on_prepare_model)
        node_var.attach_event("on_prepare_loss_func", self.on_prepare_loss_func)
        node_var.attach_event("on_prepare_optimizer", self.on_prepare_optimizer)
        node_var.attach_event("on_prepare_strategy", self.on_prepare_strategy)
        node_var.attach_event("on_prepare_extractor", self.on_prepare_extractor)
        node_var.attach_event(
            "on_prepare_data_distribution", self.on_prepare_data_distribution
        )
        node_var.attach_event("on_prepare_data_handler", self.on_prepare_data_handler)
        node_var.attach_event(
            "on_prepare_client_selection", self.on_prepare_client_selection
        )
        node_var.attach_event("on_prepare_trainer", self.on_prepare_trainer)
        node_var.attach_event("on_prepare_aggregation", self.on_prepare_aggregation)
        node_var.attach_event(
            "on_prepare_training_logger", self.on_prepare_training_logger
        )

    # Event handlers (stubs)
    # region
    def on_prepare_data_loader(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_data_loader event")
        return

    def on_prepare_model(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_model event")
        return

    def on_prepare_optimizer(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_optimizer event")
        return

    def on_prepare_loss_func(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_loss_func event")
        return

    def on_prepare_data_distribution(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_data_distribution event")
        return

    def on_prepare_data_handler(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_data_handler event")
        return

    def on_prepare_client_selection(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_client_selection event")
        return

    def on_prepare_trainer(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_training event")
        return

    def on_prepare_aggregation(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_aggregation event")
        return

    def on_prepare_strategy(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_strategy event")
        return

    def on_prepare_extractor(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_extractor event")
        return

    def on_prepare_training_logger(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_training_logger event")
        return
    # endregion
