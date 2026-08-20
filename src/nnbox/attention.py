"""Multi-head attention.

All masks in this package are *padding* masks with the convention
``True == ignore this position``, matching ``nn.MultiheadAttention``'s
``key_padding_mask``.
"""

import torch.nn.functional as F
from torch import nn

from nnbox.embeddings import apply_rotary_emb
from nnbox.utils import padding_mask_to_bias

__all__ = ["MultiheadAttention", "SelfAttention", "CrossAttention"]


class MultiheadAttention(nn.Module):
    """Multi-head attention with QK normalization and optional RoPE.

    Queries and keys are RMS-normalized per head before the dot product, which
    keeps attention logits from drifting in scale during long training runs.

    Args:
        dim: model dimension; must be divisible by `num_heads`.
        num_heads: number of attention heads.
        dropout: attention dropout, applied during training only.
        rope: a shared `RotaryEmbedding`, or None for no position embedding.
            Its `head_dim` must equal `dim // num_heads`.
    """

    def __init__(self, dim, num_heads, dropout=0.1, rope=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if rope is not None:
            assert rope.head_dim == self.head_dim, (
                f"rope.head_dim ({rope.head_dim}) must equal dim // num_heads ({self.head_dim})"
            )

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.dropout = dropout

        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

        # Registered as a submodule, but it owns no parameters and only a
        # non-persistent buffer, so sharing one instance across every block
        # costs nothing and adds nothing to the state dict.
        self.rope = rope

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        query_positions=None,
        key_positions=None,
    ):
        """
        Args:
            query: (B, Tq, dim)
            key: (B, Tk, dim)
            value: (B, Tk, dim)
            key_padding_mask: (B, Tk) bool, True where the key is padding.
            query_positions: (Tq,) or (B, Tq) positions for RoPE; defaults to
                0..Tq-1. Ignored when `rope` is None.
            key_positions: (Tk,) or (B, Tk) positions for RoPE; defaults to
                0..Tk-1. Ignored when `rope` is None.

        Returns:
            (B, Tq, dim)
        """
        B, Tq, _ = query.shape
        Tk = key.shape[1]
        H, Dh = self.num_heads, self.head_dim

        q = self.q_proj(query).view(B, Tq, H, Dh).transpose(1, 2)
        k = self.k_proj(key).view(B, Tk, H, Dh).transpose(1, 2)
        v = self.v_proj(value).view(B, Tk, H, Dh).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.rope is not None:
            q = apply_rotary_emb(q, *self._rope_tables(Tq, query_positions, q))
            k = apply_rotary_emb(k, *self._rope_tables(Tk, key_positions, k))

        attn_bias = None
        if key_padding_mask is not None:
            attn_bias = padding_mask_to_bias(key_padding_mask, query.dtype)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, Tq, self.dim)
        return self.out_proj(out)

    def _rope_tables(self, seq_len, positions, ref):
        return self.rope(
            seq_len=seq_len if positions is None else None,
            positions=positions,
            device=ref.device,
            dtype=ref.dtype,
        )


class SelfAttention(MultiheadAttention):
    """Attention of a sequence over itself."""

    def forward(self, x, key_padding_mask=None, positions=None):
        """
        Args:
            x: (B, T, dim)
            key_padding_mask: (B, T) bool, True where the token is padding.
            positions: (T,) or (B, T) positions for RoPE; defaults to 0..T-1.

        Returns:
            (B, T, dim)
        """
        return super().forward(
            x, x, x,
            key_padding_mask=key_padding_mask,
            query_positions=positions,
            key_positions=positions,
        )


class CrossAttention(MultiheadAttention):
    """Attention from a sequence to a separate context sequence.

    Queries come from `x`, keys and values from `context`; the two may differ in
    length but must share `dim`. RoPE is off by default here and should stay off
    unless `x` and `context` genuinely share a coordinate system — rotating two
    unrelated position spaces makes the relative offset between them meaningless.
    """

    def forward(
        self, x, context, context_padding_mask=None, positions=None, context_positions=None
    ):
        """
        Args:
            x: (B, Tq, dim)
            context: (B, Tk, dim)
            context_padding_mask: (B, Tk) bool, True where the context is padding.
            positions: (Tq,) or (B, Tq) query positions for RoPE.
            context_positions: (Tk,) or (B, Tk) context positions for RoPE.

        Returns:
            (B, Tq, dim)
        """
        return super().forward(
            x, context, context,
            key_padding_mask=context_padding_mask,
            query_positions=positions,
            key_positions=context_positions,
        )
