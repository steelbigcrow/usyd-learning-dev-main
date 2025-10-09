import os
import unittest
import math
from typing import List

from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

from usyd_learning.fl_algorithms import FedAggregatorFactory, FedAggregatorArgs
from usyd_learning.fl_algorithms.aggregation.methods._fed_aggregator_rbla import (
    FedAggregator_RBLA,
)


class TinyLoRANet(nn.Module):
    def __init__(self, in_features: int, hidden: int, out_features: int, r: int):
        super().__init__()
        # Use MSLoRALinear which follows A:[r,in], B:[out,r] convention used by broadcast helper
        from usyd_learning.ml_algorithms.lora.impl.lora_ms import MSLoRALinear
        self.fc1 = MSLoRALinear(in_features, hidden, r=r)
        self.fc2 = MSLoRALinear(hidden, out_features, r=r)

    def lora_prefixes(self) -> List[str]:
        return ["fc1", "fc2"]


class TestSVDAggregator(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        return super().setUp()

    def _make_clients(self, ranks: List[int], in_features=8, hidden=10, out_features=6):
        models = [TinyLoRANet(in_features, hidden, out_features, r) for r in ranks]
        sds: List[dict] = []
        for m, r in zip(models, ranks):
            sd = m.state_dict()
            # Fill LoRA A/B with random values to create meaningful deltas
            for k in list(sd.keys()):
                if k.endswith("lora_A") and sd[k].dtype.is_floating_point:
                    sd[k] = torch.randn_like(sd[k])
                if k.endswith("lora_B") and sd[k].dtype.is_floating_point:
                    sd[k] = torch.randn_like(sd[k])
            sds.append(sd)
        return models, sds

    def test_svd_aggregator_shapes_and_reconstruction(self):
        # Mixed ranks across clients
        ranks = [1, 2, 3]
        models, state_dicts = self._make_clients(ranks)
        weights = [0.2, 0.3, 0.5]

        args = FedAggregatorArgs({
            "aggregation": {"method": "svd", "device": "cpu"}
        })
        agg = FedAggregatorFactory.create_aggregator(args)

        # Aggregate
        global_sd = agg.aggregate({"state_dicts": state_dicts, "weights": weights})

        # For each layer, check that shapes equal r_max
        for pref in models[0].lora_prefixes():
            A_key = f"{pref}.lora_A"
            B_key = f"{pref}.lora_B"
            self.assertIn(A_key, global_sd)
            self.assertIn(B_key, global_sd)
            r_max = max(ranks)
            A = global_sd[A_key]
            B = global_sd[B_key]
            print(f"[SVD] Layer={pref}: aggregated A:{tuple(A.shape)}, B:{tuple(B.shape)} (r_max={r_max})")
            self.assertEqual(list(A.shape), [r_max, models[0]._modules[pref].in_features])
            self.assertEqual(list(B.shape), [models[0]._modules[pref].out_features, r_max])

            # Reconstruct reference W_g and compare with B@A
            W_sum = torch.zeros(B.shape[0], A.shape[1])
            tw = sum(weights)
            for sd, w in zip(state_dicts, weights):
                Ai = sd[A_key]
                Bi = sd[B_key]
                if Ai.shape[0] == Bi.shape[1]:
                    W_delta = Bi @ Ai
                else:
                    W_delta = Ai @ Bi
                W_sum += float(w) * W_delta
            W_g = W_sum / tw

            W_hat = B @ A
            # Full split at r_max should match best rank-r approximation (Eckart–Young)
            U, S, Vh = torch.linalg.svd(W_g, full_matrices=False)
            U_r = U[:, :A.shape[0]]
            S_r = torch.diag(S[:A.shape[0]])
            V_r = Vh[:A.shape[0], :]
            W_trunc = U_r @ S_r @ V_r
            self.assertTrue(torch.allclose(W_trunc, W_hat, atol=1e-5, rtol=1e-4))

        # Simulate broadcast: adapt back to each client's local rank
        for i, m in enumerate(models):
            local_sd = m.state_dict()
            new_sd = FedAggregator_RBLA.broadcast_lora_state_dict(global_sd, local_sd)
            # Shapes must match local ranks
            for pref in m.lora_prefixes():
                A_key = f"{pref}.lora_A"
                B_key = f"{pref}.lora_B"
                print(f"[Broadcast] Client#{i} layer={pref}: before A:{tuple(local_sd[A_key].shape)}, B:{tuple(local_sd[B_key].shape)}; after A:{tuple(new_sd[A_key].shape)}, B:{tuple(new_sd[B_key].shape)}")
                self.assertEqual(new_sd[A_key].shape[0], local_sd[A_key].shape[0])
                self.assertEqual(new_sd[B_key].shape[1], local_sd[B_key].shape[1])

            # Numeric check: B_i A_i equals rank-r_i truncation of W_g
            for pref in m.lora_prefixes():
                A_key = f"{pref}.lora_A"
                B_key = f"{pref}.lora_B"
                A_loc = new_sd[A_key]
                B_loc = new_sd[B_key]
                W_hat_i = B_loc @ A_loc

                # Build W_g and truncate to rank r_i
                r_i = A_loc.shape[0]
                # Recompute W_g from originals for this layer
                A_key_g = A_key
                B_key_g = B_key
                tw = sum(weights)
                W_sum = torch.zeros_like(W_hat_i)
                for sd, w in zip(state_dicts, weights):
                    Aig = sd[A_key_g]
                    Big = sd[B_key_g]
                    if Aig.shape[0] == Big.shape[1]:
                        W_delta = Big @ Aig
                    else:
                        W_delta = Aig @ Big
                    W_sum += float(w) * W_delta
                W_g = W_sum / tw
                # SVD truncation to r_i
                U, S, Vh = torch.linalg.svd(W_g, full_matrices=False)
                U_r = U[:, :r_i]
                S_r = torch.diag(S[:r_i])
                V_r = Vh[:r_i, :]
                W_trunc = U_r @ S_r @ V_r

                self.assertTrue(torch.allclose(W_trunc, W_hat_i, atol=1e-5, rtol=1e-4))

    def test_svd_aggregator_with_adalora_style_keys(self):
        # Prepare one client sd in PEFT/AdaLoRA-like shape
        ranks = [2, 3]
        models, sds_plain = self._make_clients(ranks, in_features=7, hidden=9, out_features=5)

        # Convert first client's plain sd to minimal PEFT-like naming
        sd_peft = {}
        base = "base_model.model"
        for k, v in sds_plain[0].items():
            if k.endswith("lora_A"):
                pref = k[:-len(".lora_A")]
                sd_peft[f"{base}.{pref}.lora_A.default"] = v
            elif k.endswith("lora_B"):
                pref = k[:-len(".lora_B")]
                sd_peft[f"{base}.{pref}.lora_B.default"] = v
            else:
                sd_peft[f"{base}.{k}"] = v

        weights = [0.4, 0.6]
        args = FedAggregatorArgs({
            "aggregation": {"method": "svd", "device": "cpu"}
        })
        agg = FedAggregatorFactory.create_aggregator(args)

        # Provide one PEFT-style sd and one plain sd
        global_sd = agg.aggregate({"state_dicts": [sd_peft, sds_plain[1]], "weights": weights})

        # Verify keys exist and ranks equal r_max
        r_max = max(ranks)
        for pref in models[0].lora_prefixes():
            A_key = f"{pref}.lora_A"
            B_key = f"{pref}.lora_B"
            self.assertIn(A_key, global_sd)
            self.assertIn(B_key, global_sd)
            self.assertEqual(global_sd[A_key].shape[0], r_max)
            self.assertEqual(global_sd[B_key].shape[1], r_max)


if __name__ == "__main__":
    unittest.main()
