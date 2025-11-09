from __future__ import annotations

import os
import unittest
import torch
from torch.utils.data import Dataset, DataLoader

# Init startup path so relative imports resolve during unittest
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))

from usyd_learning.fl_algorithms.skewed_longtail_noniid import (
    SkewedLongtailPartitioner,
    SkewedLongtailArgs,
)


class _IdDataset(Dataset):
    """
    Minimal dataset that exposes a unique integer id as the data tensor and a label.
    """
    def __init__(self, n: int, num_labels: int):
        self.ids = torch.arange(n, dtype=torch.int64)
        # Round-robin labels 0..num_labels-1
        self.labels = torch.arange(n, dtype=torch.int64) % num_labels

    def __len__(self):
        return self.ids.numel()

    def __getitem__(self, idx):
        return self.ids[idx], self.labels[idx]


class TestSkewedLongtailPartitioner(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def _make_base_loader(self, n=100, num_labels=10, batch=32):
        ds = _IdDataset(n=n, num_labels=num_labels)
        return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)

    def _label_histogram(self, data_tensor, label_tensor, num_labels):
        hist = torch.zeros(num_labels, dtype=torch.int64)
        for l in label_tensor.tolist():
            hist[l] += 1
        return hist

    def test_partition_exact_counts_and_uniqueness(self):
        base = self._make_base_loader(n=120, num_labels=10)
        part = SkewedLongtailPartitioner(base)

        # Explicit counts over 4 clients, 10 labels. Each label appears 12 times
        # in the synthetic dataset, so split evenly as 3 per client.
        num_clients = 4
        num_labels = 10
        counts = torch.full((num_clients, num_labels), 3, dtype=torch.int64)

        datasets = part.partition(counts.tolist(), SkewedLongtailArgs(return_loaders=False))
        self.assertEqual(len(datasets), num_clients)

        # Build expected counts per client from the counts matrix
        expected = counts
        label_ids = part.label_ids

        # Check label histograms match expected counts per client
        for ci, ds in enumerate(datasets):
            ys = torch.stack([ds[i][1] for i in range(len(ds))]) if len(ds) > 0 else torch.empty(0, dtype=torch.int64)
            hist = torch.zeros(len(label_ids), dtype=torch.int64)
            for lbl in ys.tolist():
                hist[label_ids.index(int(lbl))] += 1
            self.assertTrue(torch.equal(hist, expected[ci]))

        # Check global uniqueness of ids across clients
        seen = set()
        total = 0
        for ds in datasets:
            for i in range(len(ds)):
                sample_id = int(ds[i][0].item())
                self.assertNotIn(sample_id, seen)
                seen.add(sample_id)
                total += 1

        # All consumed samples equal sum of expected counts
        self.assertEqual(total, int(expected.sum().item()))

    def test_partition_return_loaders(self):
        base = self._make_base_loader(n=60, num_labels=5)
        part = SkewedLongtailPartitioner(base)
        counts = [
            [12, 0, 0, 0, 0],
            [0, 12, 0, 0, 0],
            [0, 0, 12, 0, 0],
            [0, 0, 0, 12, 0],
            [0, 0, 0, 0, 12],
        ]
        loaders = part.partition(counts, SkewedLongtailArgs(batch_size=8, shuffle=True, num_workers=0, return_loaders=True))
        # Iterate a couple of batches from the first loader
        it = iter(loaders[0])
        try:
            _ = next(it)
        except StopIteration:
            self.fail("Unexpected empty loader for client 0")


if __name__ == "__main__":
    unittest.main(verbosity=2)

