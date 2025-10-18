"""
Skewed long-tail non-IID utilities.

This module provides a decoupled implementation for constructing and applying
skewed long-tail non-IID client distributions from sparse specifications.
"""

from .skewed_longtail_spec import SkewedLongtailSpec
from .skewed_longtail_partitioner import SkewedLongtailPartitioner, SkewedLongtailArgs

__all__ = [
    "SkewedLongtailSpec",
    "SkewedLongtailPartitioner",
    "SkewedLongtailArgs",
]

