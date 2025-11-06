"""
Feature-shifted (label-balanced) partitioning utilities.

This module provides a lightweight, decoupled partitioner that splits each
label's samples evenly across a target number of clients (with largest-remainder
handling), or partitions according to an explicit client-by-class count matrix.

It mirrors the structure of the skewed_longtail_noniid utilities so it can be
used independently wherever a balanced per-label split is desired.
"""

from .feature_shifted_partitioner import FeatureShiftedPartitioner, FeatureShiftedArgs

__all__ = [
    "FeatureShiftedPartitioner",
    "FeatureShiftedArgs",
]

