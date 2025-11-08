import os
import unittest
from types import SimpleNamespace
from unittest import mock

from startup_init import startup_init_path

startup_init_path(os.path.dirname(os.path.abspath(__file__)))
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test"))
)

from fl_lora_sample.lora_sample_entry_skewed_longtail_noniid import SampleAppEntry


class FakeLoaderFactory:
    def __init__(self):
        self.datasets = []

    def create(self, args):
        self.datasets.append(args.dataset)
        return SimpleNamespace(dataset=args.dataset)


class TestSkewedLongtailEntry(unittest.TestCase):
    def setUp(self):
        self.entry = SampleAppEntry()
        self.base_loader = SimpleNamespace(data_loader=object())

    def test_partition_called_with_counts(self):
        fake_factory = FakeLoaderFactory()
        fake_partitioner = mock.MagicMock()
        fake_partitioner.partition.return_value = ["c0", "c1", "c2"]
        with mock.patch(
            "fl_lora_sample.lora_sample_entry_skewed_longtail_noniid.SkewedLongtailPartitioner",
            return_value=fake_partitioner,
        ):
            result = self.entry._allocate_skewed_data(
                self.base_loader,
                counts=[[1, 0], [0, 1], [1, 1]],
                loader_factory=fake_factory,
            )
        fake_partitioner.partition.assert_called_once()
        self.assertEqual(len(result), 3)
        self.assertEqual(fake_factory.datasets, ["c0", "c1", "c2"])


if __name__ == "__main__":
    unittest.main()
