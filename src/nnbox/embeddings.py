"""Positional embeddings."""

import torch
from torch import nn

__all__ = ["RotaryEmbedding", "apply_rotary_emb", "build_2d_sincos_pos_embed"]


def rotate_half(x):
    """Rotate the halves of the last dimension: (x1, x2) -> (-x2, x1)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x, cos, sin):
    """Apply rotary position embeddings to a projected q or k tensor.

    Args:
        x: (B, H, T, Dh)
        cos: (..., T, Dh) broadcastable against `x`, from `RotaryEmbedding`.
        sin: same shape as `cos`.

    Returns:
        (B, H, T, Dh)
    """
    return x * cos + rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings (RoPE).

    Rotates pairs of channels by an angle proportional to a token's position, so
    that the q-k dot product depends only on the *relative* offset between two
    tokens. Holds no learnable parameters: one instance can be shared across
    every block in a model, which is the intended use.

    Args:
        head_dim: per-head dimension to rotate; must be even.
        base: geometric base for the frequency ladder. Larger values stretch the
            usable context; 10000 is the standard setting.
    """

    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim ({head_dim}) must be even"
        self.head_dim = head_dim
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len=None, positions=None, device=None, dtype=torch.float32):
        """Build the (cos, sin) tables for a set of positions.

        Pass either `seq_len` (positions 0..seq_len-1) or explicit `positions`,
        which lets callers handle packed sequences, offsets during cached
        decoding, or a flattened 2D grid.

        Args:
            seq_len: sequence length, for contiguous positions from 0.
            positions: (T,) or (B, T) integer or float positions.
            device: device for the tables; defaults to the module's buffer.
            dtype: dtype of the returned tables; match the tensors you rotate.

        Returns:
            (cos, sin), each (1, 1, T, head_dim) for 1D positions or
            (B, 1, T, head_dim) for batched positions.
        """
        if (seq_len is None) == (positions is None):
            raise ValueError("pass exactly one of seq_len or positions")

        if positions is None:
            device = device if device is not None else self.inv_freq.device
            positions = torch.arange(seq_len, device=device)
        elif device is not None:
            positions = positions.to(device)

        if positions.dim() not in (1, 2):
            raise ValueError(f"positions must be (T,) or (B, T), got {tuple(positions.shape)}")

        # Angles in float32 regardless of the output dtype: under bf16 the
        # position-frequency product loses enough precision to distort the
        # rotation at long context lengths.
        freqs = positions.to(torch.float32).unsqueeze(-1) * self.inv_freq.to(positions.device)
        emb = torch.cat((freqs, freqs), dim=-1)

        # (T, Dh) -> (1, 1, T, Dh); (B, T, Dh) -> (B, 1, T, Dh)
        emb = emb.unsqueeze(0) if emb.dim() == 2 else emb
        emb = emb.unsqueeze(1)

        return emb.cos().to(dtype), emb.sin().to(dtype)


def build_2d_sincos_pos_embed(grid_h, grid_w, dim, *, device, dtype):
    """Fixed (non-learned) 2D sin-cos position embedding for a patch grid.

    Unlike a learned position embedding, this needs no interpolation to
    support a grid size other than the one training started with: it's
    recomputed directly at whatever `grid_h, grid_w` the caller passes.

    Args:
        grid_h: grid height, in patches.
        grid_w: grid width, in patches.
        dim: embedding dimension; must be divisible by 4 (split evenly
            between sin/cos and the height/width axes).
        device: device for the returned tensor.
        dtype: dtype for the returned tensor.

    Returns:
        (1, grid_h * grid_w, dim), row-major over the grid.
    """
    assert dim % 4 == 0, f"dim ({dim}) must be divisible by 4"

    y, x = torch.meshgrid(
        torch.arange(grid_h, device=device, dtype=torch.float32),
        torch.arange(grid_w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    y, x = y.reshape(-1, 1), x.reshape(-1, 1)

    omega = torch.arange(dim // 4, device=device, dtype=torch.float32)
    omega = 1.0 / (10000.0 ** (omega / (dim // 4)))

    out_x, out_y = x * omega[None, :], y * omega[None, :]
    emb = torch.cat(
        [torch.sin(out_x), torch.cos(out_x), torch.sin(out_y), torch.cos(out_y)], dim=-1
    )
    return emb.unsqueeze(0).to(dtype=dtype)
