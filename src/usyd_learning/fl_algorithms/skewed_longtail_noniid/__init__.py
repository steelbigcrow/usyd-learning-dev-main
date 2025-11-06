"""
Skewed long-tail non-IID utilities (YAML-count-driven).

This package exposes a partitioner that consumes an explicit client-by-class
count matrix (typically loaded from YAML). It no longer uses a separate
"spec" layer, since distributions are predefined in configs.
"""

from .skewed_longtail_partitioner import SkewedLongtailPartitioner, SkewedLongtailArgs

__all__ = [
    "SkewedLongtailPartitioner",
    "SkewedLongtailArgs",
]
