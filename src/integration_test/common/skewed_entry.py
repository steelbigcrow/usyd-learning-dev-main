from __future__ import annotations

"""
Skewed-sample entry used only for integration tests.
It mirrors SampleAppEntry but replaces the non-IID split with the
SkewedLongtailPartitioner to keep decoupled from existing nonIID code.
"""

from typing import Any, List

from usyd_learning.fed_node import FedNodeVars, FedNodeEventArgs
from usyd_learning.fed_runner import FedRunner
from fl_lora_sample.lora_sample_entry import SampleAppEntry
from usyd_learning.ml_data_loader.dataset_loader_factory import DatasetLoaderFactory
from usyd_learning.ml_data_loader.dataset_loader_args import DatasetLoaderArgs

from usyd_learning.fl_algorithms.skewed_longtail_noniid import (
    SkewedLongtailPartitioner,
    SkewedLongtailArgs,
)


class SkewedSampleAppEntry(SampleAppEntry):
    def __init__(self):
        # Inherit SampleAppEntry so default event handlers exist
        super().__init__()

    def run(self, device: str = "cpu", training_rounds: int = 1):
        # Load yamls from app config if absent
        if self.runner_yaml is None:
            self.runner_yaml = self.get_app_object("runner")
        if self.client_yaml is None:
            self.client_yaml = self.get_app_object("client_yaml")
        if self.edge_yaml is None:
            self.edge_yaml = self.get_app_object("edge_yaml")
        if self.server_yaml is None:
            self.server_yaml = self.get_app_object("server_yaml")

        # Runner setup
        self.fed_runner = FedRunner()
        self.fed_runner.training_rounds = training_rounds
        self.fed_runner.with_yaml(self.runner_yaml)
        self.fed_runner.create_nodes()
        self.fed_runner.create_run_strategy()

        # Server vars
        server_var = FedNodeVars(self.server_yaml)
        server_var.prepare()
        self._attach_handlers(server_var)
        server_var.owner_nodes = self.fed_runner.server_node
        server_var.set_device(device)
        self.fed_runner.server_node.node_var = server_var
        self.fed_runner.server_node.prepare_strategy()
        self.fed_runner.server_node.node_var = server_var

        # Use server's dataset loader to get the base DataLoader
        train_loader = server_var.data_loader  # type: ignore[attr-defined]
        base_dl = train_loader.data_loader

        # Use the YAML-provided client-by-class count matrix prepared in server_var
        partitioner = SkewedLongtailPartitioner(base_dl)
        part_args = SkewedLongtailArgs(batch_size=64, shuffle=True, num_workers=0, return_loaders=False)
        counts = server_var.data_distribution  # type: ignore[attr-defined]
        client_datasets = partitioner.partition(counts, part_args)

        # Wrap client datasets into DatasetLoader objects (custom)
        for i in range(len(client_datasets)):
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
                    "dataset": client_datasets[i],
                }
            )
            client_datasets[i] = DatasetLoaderFactory().create(args)

        # Prepare client nodes
        for index, node in enumerate(self.fed_runner.client_node_list):
            client_var = FedNodeVars(self.client_yaml)
            # Keep parity with SampleAppEntry: propagate rank_ratio for each client
            try:
                client_var.config_dict["nn_model"]["rank_ratio"] = self.server_yaml["rank_distribution"]["rank_ratio_list"][index]
            except Exception:
                # If rank distribution not present, continue without overriding
                pass
            client_var.prepare()
            client_var.data_loader = client_datasets[index]  # type: ignore[index]
            client_var.data_sample_num = client_var.data_loader.data_sample_num
            client_var.set_device(device)
            client_var.trainer.set_train_loader(client_var.data_loader)
            self._attach_handlers(client_var)

            client_var.owner_nodes = node
            node.node_var = client_var
            node.prepare_strategy()

        # Run training
        self.fed_runner.run()

    def _attach_handlers(self, node_var: FedNodeVars):
        # Delegate to default event hooks (same as sample entries)
        node_var.attach_event("on_prepare_data_loader", self.on_prepare_data_loader)
        node_var.attach_event("on_prepare_model", self.on_prepare_model)
        node_var.attach_event("on_prepare_loss_func", self.on_prepare_loss_func)
        node_var.attach_event("on_prepare_optimizer", self.on_prepare_optimizer)
        node_var.attach_event("on_prepare_strategy", self.on_prepare_strategy)
        node_var.attach_event("on_prepare_extractor", self.on_prepare_extractor)
        node_var.attach_event("on_prepare_data_distribution", self.on_prepare_data_distribution)
        node_var.attach_event("on_prepare_data_handler", self.on_prepare_data_handler)
        node_var.attach_event("on_prepare_client_selection", self.on_prepare_client_selection)
        node_var.attach_event("on_prepare_trainer", self.on_prepare_trainer)
        node_var.attach_event("on_prepare_aggregation", self.on_prepare_aggregation)
        node_var.attach_event("on_prepare_training_logger", self.on_prepare_training_logger)
