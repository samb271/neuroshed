"""Feed-forward and transformer blocks.

Both blocks are pre-norm and residual: every sublayer sees a normalized input
and writes back through a residual add.
"""

import torch.nn as nn

from neuroshed.attention import CrossAttention, SelfAttention

__all__ = ["MLP", "TransformerBlock", "CrossAttentionBlock"]


class MLP(nn.Module):
    """Two-layer GELU feed-forward network.

    Args:
        in_dim: input dimension.
        out_dim: output dimension; defaults to `in_dim`.
        hidden_dim: hidden dimension; defaults to `in_dim * mlp_ratio`.
        mlp_ratio: hidden width as a multiple of `in_dim`, ignored if
            `hidden_dim` is given.
        dropout: dropout after each linear.
    """

    def __init__(self, in_dim, out_dim=None, hidden_dim=None, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(in_dim * mlp_ratio)
        if out_dim is None:
            out_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: self-attention then MLP.

    Args:
        dim: model dimension.
        num_heads: attention heads.
        mlp_ratio: MLP hidden width as a multiple of `dim`.
        dropout: dropout for both the attention and the MLP.
        rope: a shared `RotaryEmbedding`, or None for no position embedding.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1, rope=None):
        super().__init__()
        self.norm_self = nn.RMSNorm(dim)
        self.norm_mlp = nn.RMSNorm(dim)

        self.self_attn = SelfAttention(dim, num_heads, dropout=dropout, rope=rope)

        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x, key_padding_mask=None, positions=None):
        """
        Args:
            x: (B, T, dim)
            key_padding_mask: (B, T) bool, True where the token is padding.
            positions: (T,) or (B, T) positions for RoPE; defaults to 0..T-1.

        Returns:
            (B, T, dim)
        """
        x = x + self.self_attn(
            self.norm_self(x), key_padding_mask=key_padding_mask, positions=positions
        )
        x = x + self.mlp(self.norm_mlp(x))
        return x


class CrossAttentionBlock(nn.Module):
    """Pre-norm transformer block: self-attention, cross-attention, then MLP.

    `x` attends to itself, then to `context`. Query and context are normalized
    separately before the cross-attention, so a context produced by a different
    trunk (a frozen encoder, a text tower) does not have to arrive pre-scaled.

    Args:
        dim: model dimension, shared by `x` and `context`.
        num_heads: attention heads.
        mlp_ratio: MLP hidden width as a multiple of `dim`.
        dropout: dropout for both attentions and the MLP.
        rope: a shared `RotaryEmbedding` applied to the self-attention only, or
            None. The cross-attention is left unrotated: `x` and `context` are
            separate coordinate systems, so a relative offset between them has
            no meaning.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1, rope=None):
        super().__init__()
        self.norm_self = nn.RMSNorm(dim)
        self.norm_cross = nn.RMSNorm(dim)
        self.norm_context = nn.RMSNorm(dim)
        self.norm_mlp = nn.RMSNorm(dim)

        self.self_attn = SelfAttention(dim, num_heads, dropout=dropout, rope=rope)
        self.cross_attn = CrossAttention(dim, num_heads, dropout=dropout)

        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(
        self, x, context, key_padding_mask=None, context_padding_mask=None, positions=None
    ):
        """
        Args:
            x: (B, Tq, dim)
            context: (B, Tk, dim)
            key_padding_mask: (B, Tq) bool, True where `x` is padding.
            context_padding_mask: (B, Tk) bool, True where `context` is padding.
            positions: (Tq,) or (B, Tq) positions for RoPE in the
                self-attention; defaults to 0..Tq-1.

        Returns:
            (B, Tq, dim)
        """
        x = x + self.self_attn(
            self.norm_self(x), key_padding_mask=key_padding_mask, positions=positions
        )
        x = x + self.cross_attn(
            self.norm_cross(x),
            self.norm_context(context),
            context_padding_mask=context_padding_mask,
        )
        x = x + self.mlp(self.norm_mlp(x))
        return x
