import os
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from startup_init import startup_init_path

startup_init_path(os.path.dirname(os.path.abspath(__file__)))

from usyd_learning.fl_algorithms.feature_shifted.feature_shifted_partitioner import (
    FeatureShiftedArgs,
    FeatureShiftedPartitioner,
)


def _build_loader():
    # 12 samples, labels 0/1/2 with uneven counts to test remainder logic.
    images = torch.arange(12).float().view(-1, 1)
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2], dtype=torch.long)
    ds = TensorDataset(images, labels)
    return DataLoader(ds, batch_size=4, shuffle=False)


class TestFeatureShiftedPartitioner(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.loader = _build_loader()
        self.partitioner = FeatureShiftedPartitioner(self.loader)

    def test_make_balanced_counts_assigns_remainders_to_last_clients(self):
        counts = self.partitioner.make_balanced_counts(num_clients=3)
        # Each of 12 samples split across 3 clients -> base 4 per label 0? verifying specific pattern.
        self.assertEqual(counts.shape, (3, len(self.partitioner.label_ids)))
        # Label 0 has 3 samples -> pattern [1,1,1] with remainder on last clients.
        self.assertListEqual(counts[:, 0].tolist(), [1, 1, 1])
        # Label 2 has 5 samples -> base=1 remainder=2 -> last two clients receive +1.
        self.assertListEqual(counts[:, 2].tolist(), [1, 2, 2])

    def test_partition_from_counts_validates_and_returns_datasets(self):
        counts = [
            [1, 0, 2],
            [1, 2, 1],
        ]
        datasets = self.partitioner.partition_from_counts(counts)
        self.assertEqual(len(datasets), 2)
        self.assertEqual(len(datasets[0]), 3)
        self.assertEqual(len(datasets[1]), 4)
        # Request loaders
        args = FeatureShiftedArgs(batch_size=2, shuffle=False, return_loaders=True)
        loaders = self.partitioner.partition_from_counts(counts, args=args)
        for loader, row in zip(loaders, counts):
            total = sum(row)
            self.assertEqual(len(loader.dataset), total)

        with self.assertRaisesRegex(ValueError, "counts must be non-negative"):
            self.partitioner.partition_from_counts([[1, -1, 0]])
        with self.assertRaisesRegex(ValueError, "exposes"):
            self.partitioner.partition_from_counts([[1, 1]])

    def test_partition_evenly_fallback(self):
        datasets = self.partitioner.partition_evenly(num_clients=2)
        lengths = [len(ds) for ds in datasets]
        self.assertEqual(sum(lengths), len(self.partitioner.x_train))
        self.assertTrue(all(length > 0 for length in lengths))


if __name__ == "__main__":
    unittest.main()
