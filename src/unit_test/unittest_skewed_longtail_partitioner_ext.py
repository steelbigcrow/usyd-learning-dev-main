import os
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from startup_init import startup_init_path

startup_init_path(os.path.dirname(os.path.abspath(__file__)))

from usyd_learning.fl_algorithms.skewed_longtail_noniid.skewed_longtail_partitioner import (
    SkewedLongtailArgs,
    SkewedLongtailPartitioner,
)


def _loader():
    images = torch.arange(20).float().view(-1, 1)
    labels = torch.tensor([0] * 5 + [1] * 5 + [2] * 10, dtype=torch.long)
    return DataLoader(TensorDataset(images, labels), batch_size=5, shuffle=False)


class TestSkewedLongtailPartitionerExt(unittest.TestCase):
    def setUp(self):
        self.partitioner = SkewedLongtailPartitioner(_loader())

    def test_partition_from_counts_honours_cap(self):
        counts = [
            [2, 1, 3],
            [2, 2, 4],
        ]
        datasets = self.partitioner.partition_from_counts(counts, SkewedLongtailArgs(return_loaders=False))
        self.assertEqual(len(datasets), 2)
        self.assertEqual(len(datasets[0]), 6)
        self.assertEqual(len(datasets[1]), 8)

        with self.assertRaisesRegex(ValueError, "Requested counts exceed available"):
            self.partitioner.partition_from_counts([[10, 0, 0]])

    def test_empty_row_returns_placeholder_dataset(self):
        counts = [
            [0, 0, 0],
            [1, 1, 1],
        ]
        datasets = self.partitioner.partition_from_counts(counts)
        self.assertEqual(len(datasets[0]), 0)
        self.assertEqual(len(datasets[1]), 3)


if __name__ == "__main__":
    unittest.main()
