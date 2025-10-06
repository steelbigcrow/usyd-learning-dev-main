import unittest
from collections import OrderedDict

import torch
import torch.nn as nn

# Init startup path for local imports
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))

from usyd_learning.ml_algorithms import AdaLoRAOptions, wrap_with_adalora
from usyd_learning.fl_algorithms.aggregation.methods._fed_aggregator_rbla import (
    FedAggregator_RBLA,
)
from usyd_learning.ml_algorithms.adalora.adalora_rbla_bridge import (
    peft_to_plain_lora_shrunk,
    plain_lora_to_peft,
    select_template_with_max_rank,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 3)
        self.linear2 = nn.Linear(3, 2)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


def build_adalora_model(r: int = 6) -> nn.Module:
    base = TinyModel()
    opts = AdaLoRAOptions(
        r=r,
        lora_alpha=int(max(1, r // 2)),
        lora_dropout=0.0,
        target_modules=["linear1", "linear2"],
        total_step=10,
        extra_kwargs={
            "init_r": r,
            "target_r": r,
        },
    )
    return wrap_with_adalora(base, opts)


def set_layer_mask_in_state_dict(sd: OrderedDict, layer_name: str, mask: list[float]) -> None:
    """Set lora_E for a given layer to a binary mask (1.0 selected, 0.0 pruned)."""
    # Find lora_E key for layer
    candidates = [
        k for k in sd.keys() if (f".{layer_name}." in k and ".lora_E." in k)
    ]
    assert len(candidates) >= 1, f"No lora_E found for layer {layer_name}"
    # Use the first matching key (AdaLoRA puts lora_E on the adapter container)
    k = candidates[0]
    E = sd[k]
    r = E.shape[0]
    assert r == len(mask), f"Mask length {len(mask)} != r {r} for key {k}"
    new_E = torch.tensor(mask, dtype=E.dtype).view(r, 1)
    sd[k] = new_E


class TestAdaLoRAMaskUnify(unittest.TestCase):
    def test_peft_to_plain_carries_mask_and_rr(self):
        r = 6
        model = build_adalora_model(r=r)
        sd = OrderedDict({k: v.clone().detach() for k, v in model.state_dict().items()})

        # Define per-layer masks (sum -> effective ranks)
        mask_l1 = [1, 1, 0, 0, 0, 0]  # rr=2
        mask_l2 = [1, 0, 1, 0, 0, 0]  # rr=2
        set_layer_mask_in_state_dict(sd, "linear1", mask_l1)
        set_layer_mask_in_state_dict(sd, "linear2", mask_l2)

        out = peft_to_plain_lora_shrunk(sd)

        # Check presence of keys and shapes follow rr
        for prefix, m in {"linear1": mask_l1, "linear2": mask_l2}.items():
            rr = sum(m)
            self.assertIn(f"{prefix}.lora_A", out)
            self.assertIn(f"{prefix}.lora_B", out)
            self.assertIn(f"{prefix}.rank_mask", out)
            self.assertIn(f"{prefix}.rank_rr", out)

            A = out[f"{prefix}.lora_A"]
            B = out[f"{prefix}.lora_B"]
            mask_vec = out[f"{prefix}.rank_mask"].view(-1)
            rr_scalar = float(out[f"{prefix}.rank_rr"].item())

            self.assertEqual(A.shape[0], rr)
            self.assertEqual(B.shape[1], rr)
            self.assertEqual(int(round(mask_vec.sum().item())), rr)
            self.assertEqual(int(round(rr_scalar)), rr)

    def test_plain_to_peft_unify_mask_and_zero_ab(self):
        r = 6
        # Client 1 with rr=2
        m1 = build_adalora_model(r=r)
        sd1 = OrderedDict({k: v.clone().detach() for k, v in m1.state_dict().items()})
        mask1 = [1, 1, 0, 0, 0, 0]
        set_layer_mask_in_state_dict(sd1, "linear1", mask1)
        set_layer_mask_in_state_dict(sd1, "linear2", mask1)
        p1 = peft_to_plain_lora_shrunk(sd1)

        # Client 2 with rr=3
        m2 = build_adalora_model(r=r)
        sd2 = OrderedDict({k: v.clone().detach() for k, v in m2.state_dict().items()})
        mask2 = [1, 1, 1, 0, 0, 0]
        set_layer_mask_in_state_dict(sd2, "linear1", mask2)
        set_layer_mask_in_state_dict(sd2, "linear2", mask2)
        p2 = peft_to_plain_lora_shrunk(sd2)

        # Aggregate plain dicts using RBLA (rank_mask gets averaged as a normal tensor)
        agg = FedAggregator_RBLA()
        clients = [
            {"updated_weights": p1, "train_record": {"data_sample_num": 1}},
            {"updated_weights": p2, "train_record": {"data_sample_num": 1}},
        ]
        aggregated_plain = agg.aggregate(clients)

        # Map back to PEFT using max-r template (either client's peft sd works here since both r are equal)
        template_sd = select_template_with_max_rank([
            {"updated_weights": sd1}, {"updated_weights": sd2}
        ])
        out_peft = plain_lora_to_peft(aggregated_plain, template_sd)

        # Expect unified K = round((2 + 3) / 2) = 2, and top-2 channels are 0 and 1
        K_expected = 2
        for layer in ("linear1", "linear2"):
            # Find keys
            e_keys = [k for k in out_peft.keys() if f".{layer}." in k and ".lora_E." in k]
            a_keys = [k for k in out_peft.keys() if f".{layer}." in k and ".lora_A." in k]
            b_keys = [k for k in out_peft.keys() if f".{layer}." in k and ".lora_B." in k]
            self.assertTrue(len(e_keys) >= 1)
            self.assertTrue(len(a_keys) >= 1)
            self.assertTrue(len(b_keys) >= 1)
            E = out_peft[e_keys[0]]
            A = out_peft[a_keys[0]]
            B = out_peft[b_keys[0]]
            # E gate should be binary {0,1} and sum to K_expected
            e_bin = (E.view(-1) > 0.5).to(torch.int64)
            self.assertEqual(int(e_bin.sum().item()), K_expected)
            # By construction we unify first-K channels
            self.assertTrue(e_bin[0].item() == 1 and e_bin[1].item() == 1)
            # A rows beyond unified K are zero; B cols beyond unified K are zero
            ra = A.shape[0]
            rb = B.shape[1]
            if ra > K_expected:
                self.assertTrue(torch.allclose(A[K_expected:, :], torch.zeros_like(A[K_expected:, :])))
            if rb > K_expected:
                self.assertTrue(torch.allclose(B[:, K_expected:], torch.zeros_like(B[:, K_expected:])))


if __name__ == "__main__":
    unittest.main()
