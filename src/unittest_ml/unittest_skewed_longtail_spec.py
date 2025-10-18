from __future__ import annotations

import os
import unittest
import torch

# Init startup path so relative imports resolve during unittest
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))

from usyd_learning.fl_algorithms.skewed_longtail_noniid.skewed_longtail_spec import (
    SkewedLongtailSpec,
)


class TestSkewedLongtailSpec(unittest.TestCase):
    def test_to_dense_weights_handles_sparse_and_unknown(self):
        # Sparse client ids and an unknown label 99 that should be ignored
        spec = {
            0: {0: 2.0},
            2: {1: 3.5, 99: 10.0},
            5: {0: 1.0, 1: 1.0},
        }

        s = SkewedLongtailSpec(spec)
        # Explicit label ids (only known labels)
        weights, label_ids = s.to_dense_weights(label_ids=[0, 1], num_clients=6)

        self.assertEqual(weights.shape, (6, 2))
        # Row 0: (2, 0)
        self.assertTrue(torch.allclose(weights[0], torch.tensor([2.0, 0.0])))
        # Row 1: zeros
        self.assertTrue(torch.allclose(weights[1], torch.tensor([0.0, 0.0])))
        # Row 2: (0, 3.5) — 99 ignored
        self.assertTrue(torch.allclose(weights[2], torch.tensor([0.0, 3.5])))
        # Row 5: (1, 1)
        self.assertTrue(torch.allclose(weights[5], torch.tensor([1.0, 1.0])))

    def test_normalize_weights_exact_sums_and_stability(self):
        # Two labels, three clients
        weights = torch.tensor([
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
        ])
        available = [80, 40]

        c1 = SkewedLongtailSpec.normalize_weights_to_counts(weights, available)
        c2 = SkewedLongtailSpec.normalize_weights_to_counts(weights, available)

        # Column sums match available
        self.assertEqual(int(c1[:, 0].sum().item()), 80)
        self.assertEqual(int(c1[:, 1].sum().item()), 40)
        # Deterministic
        self.assertTrue(torch.equal(c1, c2))

        # Proportion sanity: client0 should get the largest share of label0
        self.assertGreater(int(c1[0, 0].item()), int(c1[1, 0].item()))
        self.assertGreater(int(c1[0, 0].item()), int(c1[2, 0].item()))

    def test_normalize_zero_cases(self):
        # One label has zero weight; one label has zero available
        weights = torch.tensor([
            [0.0, 1.0],
            [0.0, 3.0],
        ])
        available = [50, 0]  # label0 has samples but zero weight; label1 has zero samples

        counts = SkewedLongtailSpec.normalize_weights_to_counts(weights, available)

        # First column should allocate zero to all (no weights)
        self.assertTrue(torch.all(counts[:, 0] == 0))
        # Second column also zero (no available)
        self.assertTrue(torch.all(counts[:, 1] == 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)

