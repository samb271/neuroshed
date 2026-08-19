"""Reusable PyTorch building blocks.

Everything listed in ``__all__`` is the public API: import it as
``from neuroshed import SelfAttention``. Anything else is internal and may
move between modules without notice.
"""

from neuroshed.attention import CrossAttention, MultiheadAttention, SelfAttention
from neuroshed.blocks import MLP, CrossAttentionBlock, TransformerBlock
from neuroshed.embeddings import RotaryEmbedding, apply_rotary_emb

__version__ = "0.1.0"

__all__ = [
    "MultiheadAttention",
    "SelfAttention",
    "CrossAttention",
    "MLP",
    "TransformerBlock",
    "CrossAttentionBlock",
    "RotaryEmbedding",
    "apply_rotary_emb",
]
