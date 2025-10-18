"""
Lightweight monitoring package for the training framework.

Importing `usyd_learning.monitoring.auto_enable` activates runtime instrumentation
according to src/Monitor Plan.md, without modifying core training code.

Typical usage (single line at entry):

    import usyd_learning.monitoring.auto_enable  # noqa: F401

The module will auto-discover optional monitoring.yaml and default to
CSV + console sinks writing under ./.monitor/.
"""

from .hub import MonitorHub, get_hub  # re-export

__all__ = [
    "MonitorHub",
    "get_hub",
]

