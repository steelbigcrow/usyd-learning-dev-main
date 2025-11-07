"""
Entry class for feature-shift distributions (IID-like):
- Focused on scenarios where each label's samples are evenly split across clients.
- Works for any dataset where the server YAML provides a dense distribution matrix
  (via data_distribution.use + custom_define) or when such a matrix is absent,
  it falls back to uniform weights per (client, label).

This entry is decoupled from legacy non-IID generators and uses the standalone
SkewedLongtailPartitioner so that explicit matrices (feature-shift) are respected
and the remaining rounding is handled robustly by largest-remainder allocation.
"""

from usyd_learning.fed_node import FedNodeVars, FedNodeEventArgs
from usyd_learning.fed_runner import FedRunner
from usyd_learning.ml_utils import AppEntry, console
from usyd_learning.fl_algorithms.skewed_longtail_noniid import (
    SkewedLongtailPartitioner,
    SkewedLongtailArgs,
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
        server_var.prepare()
        self.__attach_event_handler(server_var)
        server_var.owner_nodes = self.fed_runner.server_node  # Two way binding
        server_var.set_device(device)
        self.fed_runner.server_node.node_var = server_var
        self.fed_runner.server_node.prepare_strategy()
        self.fed_runner.server_node.node_var = server_var

        # Base data loader
        train_loader = server_var.data_loader
        base_dl = train_loader.data_loader

        # Build sparse spec from a dense matrix in YAML if available; otherwise, fallback
        dist_matrix = server_var.data_distribution
        sparse_spec: dict[int, dict[int, float]] = {}
        if isinstance(dist_matrix, list) and len(dist_matrix) > 0 and isinstance(dist_matrix[0], list):
            for ci, row in enumerate(dist_matrix):
                row_map: dict[int, float] = {}
                for lbl, v in enumerate(row):
                    try:
                        val = float(v)
                    except Exception:
                        continue
                    if val > 0:
                        row_map[int(lbl)] = val
                sparse_spec[int(ci)] = row_map
        else:
            # Fallback to uniform feature-shift weights (each client sees all labels equally)
            partitioner_probe = SkewedLongtailPartitioner(base_dl)
            label_ids = partitioner_probe.label_ids
            num_clients = len(self.fed_runner.client_node_list)
            for ci in range(num_clients):
                sparse_spec[ci] = {lbl: 1.0 for lbl in label_ids}

        # Partition using the decoupled partitioner; produce CustomDatasets
        partitioner = SkewedLongtailPartitioner(base_dl)
        part_args = SkewedLongtailArgs(batch_size=64, shuffle=True, num_workers=0, return_loaders=False)
        client_datasets = partitioner.partition(sparse_spec, part_args)

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
            # Fill client-specific rank ratio if provided
            try:
                client_var.config_dict["nn_model"]["rank_ratio"] = self.server_yaml["rank_distribution"]["rank_ratio_list"][index]
            except Exception:
                pass
            client_var.prepare()
            client_var.data_loader = client_datasets[index]  # type: ignore[index]
            client_var.data_sample_num = client_var.data_loader.data_sample_num
            client_var.set_device(device)
            client_var.trainer.set_train_loader(client_var.data_loader)
            self.__attach_event_handler(client_var)

            client_var.owner_nodes = node
            node.node_var = client_var
            node.prepare_strategy()

        # Run
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
        node_var.attach_event("on_prepare_model", self.on_prepare_model)
        node_var.attach_event("on_prepare_loss_func", self.on_prepare_loss_func)
        node_var.attach_event("on_prepare_data_distribution", self.on_prepare_data_distribution)
        node_var.attach_event("on_prepare_data_handler", self.on_prepare_data_handler)
        node_var.attach_event("on_prepare_client_selection", self.on_prepare_client_selection)
        node_var.attach_event("on_prepare_trainer", self.on_prepare_trainer)
        node_var.attach_event("on_prepare_aggregation", self.on_prepare_aggregation)
        node_var.attach_event("on_prepare_extractor", self.on_prepare_extractor)
        node_var.attach_event("on_prepare_strategy", self.on_prepare_strategy)
        node_var.attach_event("on_prepare_training_logger", self.on_prepare_training_logger)

    # Event handdlers (placeholders)
    # region
    def on_prepare_data_loader(self, args: FedNodeEventArgs):
        console.warn("TODO: on_prepare_data_loader event")
        return

    def on_prepare_model(self, args: FedNodeEventArgs):
        console.warn("TODO: on_prepare_model event")
        return

    def on_prepare_optimizer(self, args: FedNodeEventArgs):
        console.warn("TODO: on_prepare_optimizer event")
        return

    def on_prepare_loss_func(self, args: FedNodeEventArgs):
        console.warn("TODO: on_prepare_loss_func event")
        return

    def on_prepare_data_distribution(self, args: FedNodeEventArgs):
        console.warn("TODO: on_prepare_data_distribution event")
        return

    def on_prepare_data_handler(self, args: FedNodeEventArgs):
        console.warn("TODO: on_prepare_data_handler event")
        return

    def on_prepare_client_selection(self, args: FedNodeEventArgs):
        console.warn("TODO: on_prepare_client_selection event")
        return

    def on_prepare_trainer(self, args: FedNodeEventArgs):
        console.warn("TODO: on_prepare_training event")
        return

    def on_prepare_aggregation(self, args: FedNodeEventArgs):
        console.warn("TODO: on_prepare_aggregation event")
        return

    def on_prepare_strategy(self, args: FedNodeEventArgs):
        console.warn("TODO: on_prepare_strategy event")
        return

    def on_prepare_extractor(self, args: FedNodeEventArgs):
        console.warn("TODO: on_prepare_extractor event")
        return

    def on_prepare_training_logger(self, args: FedNodeEventArgs):
        console.warn("TODO: on_prepare_training_logger event")
        return
    # endregion
