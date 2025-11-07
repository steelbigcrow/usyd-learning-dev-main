from tqdm import tqdm

from usyd_learning.fed_strategy.strategy_args import StrategyArgs
from ...ml_utils import console
from ...fl_algorithms.aggregation.fed_aggregator_facotry import FedAggregatorFactory
from ...fl_algorithms.selection.fed_client_selector_factory import FedClientSelectorFactory
from ...fed_runner import FedRunner
from ...fed_strategy.runner_strategy import RunnerStrategy 
from ...fed_node import FedNodeClient, FedNodeServer


class ZpRunnerStrategy(RunnerStrategy):
    """
    Runner loop for Zero-Pad (ZP) aggregation, mirroring the RBLA runner
    structure but fully decoupled from RBLA code paths. The server-side ZP
    strategy handles preprocessing and aggregation.
    """

    def __init__(self, runner: FedRunner, args: StrategyArgs, client_node, server_node) -> None:
        super().__init__(runner)
        self._strategy_type = "zp"
        self.args = args
        self.client_nodes: list[FedNodeClient] = client_node
        self.server_node: FedNodeServer = server_node
        self.set_node_connection()

    def _create_inner(self, client_node, server_node) -> None:
        return self
    
    def prepare(self, logger_header) -> None:
        self.server_node.prepare(logger_header, self.client_nodes)
        return

    def set_node_connection(self) -> None:
        self.server_node.set_client_nodes(self.client_nodes)
        for client in self.client_nodes:
            client.set_server_node(self.server_node)
        return
    
    def simulate_client_local_training_process(self, participants):
        for client in participants:
            console.info(f"\n[{client.node_id}] Local training started")
            updated_weights, train_record = client.strategy.run_local_training()
            yield {
                "updated_weights": updated_weights,
                "train_record": train_record
            }

    def simulate_server_broadcast_process(self):
        self.server_node.broadcast_weight(self.client_nodes)
        return
    
    def simulate_server_update_process(self, weight):
        self.server_node.strategy.server_update(weight)
        return

    def run(self) -> None:
        print("Running [ZP] strategy...")
        # Compose header robustly: tolerate missing optional 'rank_distribution'
        cfg = self.server_node.node_var.config_dict
        rank_cfg = cfg.get('rank_distribution', None)
        rank_str = "N/A"
        try:
            if isinstance(rank_cfg, dict) and 'rank_ratio_list' in rank_cfg:
                rank_str = str(rank_cfg['rank_ratio_list'])
        except Exception:
            rank_str = "N/A"

        header_data = {
            "general": cfg['general'],
            "aggregation": cfg['aggregation']['method'],
            "rank_distribution": rank_str,
            'epoch': cfg['training']['epochs'],
            "dataset": cfg['data_loader']['name'],
            "batch_size": cfg['data_loader']['batch_size'],
            "model": cfg['nn_model']['name'],
            "loss_function": cfg['loss_func']['type'],
            "client_selection": cfg['client_selection']
        }
        self.server_node.prepare(header_data, self.client_nodes)
        self.server_node.broadcast()
        for round in tqdm(range(self.args.key_value_dict.data['training_rounds'] + 1)):
           
            console.out(f"\n{'='*10} Training round {round}/{self.args.key_value_dict.data['training_rounds']}, Total participants: {len(self.client_nodes)} {'='*10}")
            
            self.participants = self.server_node.select_clients(self.client_nodes)
            
            console.info(f"Round: {round}, Select {len(self.participants)} clients: ', '").ok(f"{', '.join(map(str, self.participants))}")

            client_updates = list(self.simulate_client_local_training_process(self.participants))         

            self.server_node.receive_client_updates(client_updates)

            self.server_node.aggregation()

            self.server_node.apply_weight()

            self.server_node.broadcast()

            self.server_node.evaluate()

            self.server_node.record_evaluation()

            console.out(f"{'='*10} Round {round}/{self.args.key_value_dict.data['training_rounds']} End{'='*10}")

        return
        
