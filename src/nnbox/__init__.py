"""Reusable PyTorch building blocks.

Everything listed in ``__all__`` is the public API: import it as
``from nnbox import SelfAttention``. Anything else is internal and may
move between modules without notice.
"""

from nnbox.attention import CrossAttention, MultiheadAttention, SelfAttention
from nnbox.blocks import MLP, CrossAttentionBlock, TransformerBlock
from nnbox.embeddings import RotaryEmbedding, apply_rotary_emb

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
