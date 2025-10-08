"""
AdaLoRA model adapters.

These classes provide model-type names that mirror the LoRA directory for
organizational symmetry. Architectures use standard nn.Modules, and AdaLoRA
adapters are injected at trainer build time (see fed_node_vars.prepare_trainer).
"""

