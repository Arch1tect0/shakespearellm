"""Model-size configurations for the baseline and scaled transformers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelA:
    """Baseline transformer configuration."""

    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 64
    vocab_size: int = 256


@dataclass(frozen=True)
class ModelB:
    """Scaled transformer configuration."""

    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 256
    block_size: int = 128
    vocab_size: int = 256
