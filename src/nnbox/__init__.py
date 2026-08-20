"""Reusable PyTorch building blocks.

Everything listed in ``__all__`` is the public API: import it as
``from nnbox import SelfAttention``. Anything else is internal and may
move between modules without notice.
"""

from nnbox.attention import CrossAttention, MultiheadAttention, SelfAttention
from nnbox.convolution import ConvStack, ResidualConvBlock
from nnbox.embeddings import RotaryEmbedding, apply_rotary_emb, build_2d_sincos_pos_embed
from nnbox.mlp import MLP
from nnbox.transformer import CrossAttentionBlock, TransformerBlock

__version__ = "0.2.0"

__all__ = [
    "MLP",
    "ConvStack",
    "CrossAttention",
    "CrossAttentionBlock",
    "MultiheadAttention",
    "ResidualConvBlock",
    "RotaryEmbedding",
    "SelfAttention",
    "TransformerBlock",
    "apply_rotary_emb",
    "build_2d_sincos_pos_embed",
]
