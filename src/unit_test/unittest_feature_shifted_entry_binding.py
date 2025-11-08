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

from fl_lora_sample.lora_sample_entry_feature_shifted import SampleAppEntry


class FakeLoaderFactory:
    def __init__(self):
        self.datasets = []

    def create(self, args):
        self.datasets.append(args.dataset)
        return SimpleNamespace(dataset=args.dataset)


class TestFeatureShiftedEntryBinding(unittest.TestCase):
    def setUp(self):
        self.entry = SampleAppEntry()
        self.base_loader = SimpleNamespace(data_loader=object())

    def test_explicit_counts_path(self):
        fake_factory = FakeLoaderFactory()
        fake_partitioner = mock.MagicMock()
        fake_partitioner.partition_from_counts.return_value = ["ds0", "ds1"]
        with mock.patch(
            "fl_lora_sample.lora_sample_entry_feature_shifted.FeatureShiftedPartitioner",
            return_value=fake_partitioner,
        ):
            result = self.entry._allocate_feature_shifted_data(
                self.base_loader,
                counts=[[1], [1]],
                num_clients=2,
                loader_factory=fake_factory,
            )
        self.assertEqual(len(result), 2)
        fake_partitioner.partition_from_counts.assert_called_once()
        fake_partitioner.partition_evenly.assert_not_called()
        self.assertEqual(fake_factory.datasets, ["ds0", "ds1"])

    def test_fallback_even_split_when_counts_missing(self):
        fake_factory = FakeLoaderFactory()
        fake_partitioner = mock.MagicMock()
        fake_partitioner.partition_from_counts.side_effect = ValueError("bad counts")
        fake_partitioner.partition_evenly.return_value = ["even0", "even1", "even2"]
        with mock.patch(
            "fl_lora_sample.lora_sample_entry_feature_shifted.FeatureShiftedPartitioner",
            return_value=fake_partitioner,
        ):
            result = self.entry._allocate_feature_shifted_data(
                self.base_loader,
                counts=None,
                num_clients=3,
                loader_factory=fake_factory,
            )
        fake_partitioner.partition_evenly.assert_called_once_with(3, mock.ANY)
        self.assertEqual(len(result), 3)
        self.assertEqual(fake_factory.datasets, ["even0", "even1", "even2"])


if __name__ == "__main__":
    unittest.main()
