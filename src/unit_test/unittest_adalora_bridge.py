import os
import unittest
from typing import Dict, List, Tuple

import torch

from startup_init import startup_init_path

startup_init_path(os.path.dirname(os.path.abspath(__file__)))

from usyd_learning.ml_algorithms.adalora.adalora_rbla_bridge import (
    peft_to_plain_lora_shrunk,
    plain_lora_to_peft,
    select_template_with_max_rank,
)
from usyd_learning.fl_algorithms.aggregation.methods._fed_aggregator_zeropad import (
    FedAggregator_ZeroPad,
)
from usyd_learning.fl_algorithms.aggregation.fed_aggregator_args import (
    FedAggregatorArgs,
)
from usyd_learning.fl_algorithms.aggregation.methods._fed_aggregator_rbla import (
    FedAggregator_RBLA,
)


BASE_PREFIX = "base_model.model"


def _client_layer(
    layer: str,
    a_rows: List[List[float]],
    b_rows: List[List[float]],
    mask: List[float],
) -> Dict[str, torch.Tensor]:
    return {
        f"{BASE_PREFIX}.{layer}.lora_A.default": torch.tensor(a_rows, dtype=torch.float32),
        f"{BASE_PREFIX}.{layer}.lora_B.default": torch.tensor(b_rows, dtype=torch.float32),
        f"{BASE_PREFIX}.{layer}.lora_E.default": torch.tensor(mask, dtype=torch.float32).view(-1, 1),
        f"{BASE_PREFIX}.{layer}.ranknum": torch.tensor(float(sum(mask)), dtype=torch.float32),
    }


def _make_simple_clients() -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]:
    layer = "layer"
    # Client 0 -> effective rank 2
    c0 = _client_layer(
        layer,
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [30.0, 30.0, 30.0],
            [40.0, 40.0, 40.0],
        ],
        [
            [10.0, 20.0, 30.0, 40.0],
            [10.0, 20.0, 30.0, 40.0],
        ],
        [1.0, 1.0, 0.0, 0.0],
    )
    # Client 1 -> effective rank 3
    c1 = _client_layer(
        layer,
        [
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],
            [6.0, 6.0, 6.0],
        ],
        [
            [7.0, 8.0, 9.0, 10.0],
            [7.0, 8.0, 9.0, 10.0],
        ],
        [1.0, 1.0, 1.0, 0.0],
    )
    return c0, c1, layer


class TestAdaLoRABridgeBasics(unittest.TestCase):
    def test_scalar_and_vector_rank_hints(self):
        sd, _, layer = _make_simple_clients()
        # Scalar rank hint -> shrink to r=2
        scalar_sd = {
            f"{BASE_PREFIX}.{layer}.lora_A.default": sd[f"{BASE_PREFIX}.{layer}.lora_A.default"].clone(),
            f"{BASE_PREFIX}.{layer}.lora_B.default": sd[f"{BASE_PREFIX}.{layer}.lora_B.default"].clone(),
            f"{BASE_PREFIX}.{layer}.ranknum": torch.tensor(2.0),
        }
        plain_scalar = peft_to_plain_lora_shrunk(scalar_sd)
        self.assertEqual(tuple(plain_scalar[f"{layer}.lora_A"].shape), (2, 3))
        self.assertEqual(tuple(plain_scalar[f"{layer}.lora_B"].shape), (2, 2))
        self.assertEqual(int(plain_scalar[f"{layer}.rank_rr"].item()), 2)

        # Vector rank hint -> keep custom mask
        vector_sd = {
            f"{BASE_PREFIX}.{layer}.lora_A.default": sd[f"{BASE_PREFIX}.{layer}.lora_A.default"].clone(),
            f"{BASE_PREFIX}.{layer}.lora_B.default": sd[f"{BASE_PREFIX}.{layer}.lora_B.default"].clone(),
            f"{BASE_PREFIX}.{layer}.ranknum": torch.tensor([1.0, 1.0, 0.0, 0.0]),
        }
        plain_vector = peft_to_plain_lora_shrunk(vector_sd)
        self.assertTrue(
            torch.allclose(
                plain_vector[f"{layer}.rank_mask"].view(-1),
                torch.tensor([1.0, 1.0, 0.0, 0.0]),
                atol=1e-6,
            )
        )

    def test_plain_roundtrip_without_hints(self):
        sd, _, layer = _make_simple_clients()
        # Remove masks -> expect full rank
        del sd[f"{BASE_PREFIX}.{layer}.lora_E.default"]
        del sd[f"{BASE_PREFIX}.{layer}.ranknum"]
        plain = peft_to_plain_lora_shrunk(sd)
        self.assertEqual(tuple(plain[f"{layer}.lora_A"].shape), (4, 3))
        self.assertTrue(torch.allclose(plain[f"{layer}.rank_mask"], torch.ones(4), atol=1e-6))
        mapped = plain_lora_to_peft(plain, sd)
        self.assertEqual(tuple(mapped[f"{BASE_PREFIX}.{layer}.lora_A.default"].shape), (4, 3))


class TestAdaLoRABridgeAggregators(unittest.TestCase):
    def test_zero_pad_pipeline(self):
        c0, c1, layer = _make_simple_clients()
        sd0_plain = peft_to_plain_lora_shrunk(c0)
        sd1_plain = peft_to_plain_lora_shrunk(c1)

        args = FedAggregatorArgs({"aggregation": {"device": "cpu", "method": "zp"}})
        aggregator = FedAggregator_ZeroPad(args)
        aggregated_plain = aggregator.aggregate([(sd0_plain, 1.0), (sd1_plain, 1.0)])

        self.assertEqual(tuple(aggregated_plain[f"{layer}.lora_A"].shape), (3, 3))
        self.assertEqual(tuple(aggregated_plain[f"{layer}.lora_B"].shape), (2, 3))

        template = select_template_with_max_rank([
            {"updated_weights": c0},
            {"updated_weights": c1},
        ])
        peft = plain_lora_to_peft(aggregated_plain, template)
        self.assertIn(f"{BASE_PREFIX}.{layer}.lora_A.default", peft)
        self.assertTrue(torch.all(peft[f"{BASE_PREFIX}.{layer}.lora_A.default"][2] == 0))

    def test_rbla_pipeline_multi_client(self):
        c0, c1, layer = _make_simple_clients()
        # Third client with rr=1
        c2 = _client_layer(
            layer,
            [
                [11.0, 11.0, 11.0],
                [12.0, 12.0, 12.0],
                [13.0, 13.0, 13.0],
                [14.0, 14.0, 14.0],
            ],
            [
                [15.0, 16.0, 17.0, 18.0],
                [15.0, 16.0, 17.0, 18.0],
            ],
            [1.0, 0.0, 0.0, 0.0],
        )
        vols = [1.0, 2.0, 1.0]
        args = FedAggregatorArgs({"aggregation": {"device": "cpu", "method": "rbla"}})
        aggregator = FedAggregator_RBLA(args)
        plains = [
            peft_to_plain_lora_shrunk(c)
            for c in (c0, c1, c2)
        ]
        aggregated_plain = aggregator.aggregate(list(zip(plains, vols)))
        self.assertEqual(tuple(aggregated_plain[f"{layer}.lora_A"].shape), (3, 3))

        template = select_template_with_max_rank([
            {"updated_weights": c0},
            {"updated_weights": c1},
            {"updated_weights": c2},
        ])
        peft = plain_lora_to_peft(aggregated_plain, template)
        self.assertEqual(int(round(float(peft[f"{BASE_PREFIX}.{layer}.ranknum"].item()))), 2)


class TestAdaLoRAMultiLayerBroadcast(unittest.TestCase):
    def test_unified_budget_per_layer(self):
        layers = ("L1", "L2")
        # Build three clients with different rank budgets per layer
        def build_client(offset: int) -> Dict[str, torch.Tensor]:
            sd: Dict[str, torch.Tensor] = {}
            masks = {
                "L1": [[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]][offset],
                "L2": [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]][offset],
            }
            base = 1 + offset * 10
            for idx, layer in enumerate(layers):
                rows = [
                    [base + 0 + idx, base + 0 + idx, base + 0 + idx],
                    [base + 1 + idx, base + 1 + idx, base + 1 + idx],
                    [base + 2 + idx, base + 2 + idx, base + 2 + idx],
                    [base + 3 + idx, base + 3 + idx, base + 3 + idx],
                ]
                cols = [
                    [base + 10 + idx, base + 11 + idx, base + 12 + idx, base + 13 + idx],
                    [base + 10 + idx, base + 11 + idx, base + 12 + idx, base + 13 + idx],
                ]
                sd.update(_client_layer(layer, rows, cols, masks[layer]))
            return sd

        clients = [build_client(i) for i in range(3)]
        vols = [1.0, 2.0, 1.0]
        plains = [peft_to_plain_lora_shrunk(sd) for sd in clients]

        args = FedAggregatorArgs({"aggregation": {"device": "cpu", "method": "rbla"}})
        aggregator = FedAggregator_RBLA(args)
        aggregated_plain = aggregator.aggregate(list(zip(plains, vols)))

        for layer in layers:
            self.assertEqual(tuple(aggregated_plain[f"{layer}.lora_A"].shape), (3, 3))
            self.assertTrue(torch.all(aggregated_plain[f"{layer}.rank_mask"][2:] <= 1.0 + 1e-6))

        template = select_template_with_max_rank([
            {"updated_weights": sd} for sd in clients
        ])
        peft = plain_lora_to_peft(aggregated_plain, template)
        for layer in layers:
            mask = peft[f"{BASE_PREFIX}.{layer}.lora_E.default"].view(-1)
            self.assertTrue(torch.allclose(mask, torch.tensor([1.0, 1.0, 0.0, 0.0]), atol=1e-6))
            self.assertEqual(int(peft[f"{BASE_PREFIX}.{layer}.ranknum"].item()), 2)


if __name__ == "__main__":
    unittest.main()
