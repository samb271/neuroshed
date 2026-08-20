"""Patch-token decoder: reconstructs an image from a sequence of patch tokens.

The inverse of `nnbox.models.vit.PatchEmbed`: projects each token back to a
pixel patch, then refines the tiled result with a residual conv stack in pixel
space to smooth patch-boundary seams. Takes any patch-token sequence at
`embed_dim` — from `ViT(use_cls_token=False)` or an external frozen encoder —
not just nnbox's own `ViT`.
"""

import math

import torch.nn as nn

from nnbox.convolution import ConvStack
from nnbox.embeddings import build_2d_sincos_pos_embed
from nnbox.transformer import TransformerBlock

__all__ = ["PatchDecoder"]


class PatchDecoder(nn.Module):
    """Reconstructs an image from a sequence of patch tokens.

    Projects tokens to `model_dim`, adds a fixed 2D sin-cos position embedding
    sized to the input's token grid, runs a `TransformerBlock` stack, unfolds
    each output token into a pixel patch, then refines the tiled image with a
    residual conv stack.

    Args:
        img_size: output image side length; must be divisible by `patch_size`.
        patch_size: pixel width/height reconstructed per token.
        out_channels: output image channels.
        embed_dim: dimension of the input token sequence.
        model_dim: internal transformer width; must be divisible by
            `num_heads`. Kept separate from `embed_dim` via a linear
            projection, so the decoder can run wider or narrower than
            whatever produced its input tokens.
        depth: number of transformer blocks.
        num_heads: attention heads.
        mlp_ratio: MLP hidden width as a multiple of `model_dim`.
        dropout: dropout in the transformer blocks.
        refine_channels: hidden channels in the conv refinement stack.
        refine_blocks: number of residual conv blocks; 0 skips refinement
            beyond the initial projection.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        out_channels=3,
        embed_dim=768,
        model_dim=768,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        refine_channels=128,
        refine_blocks=4,
    ):
        super().__init__()
        assert img_size % patch_size == 0, (
            f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
        )
        assert model_dim % num_heads == 0, (
            f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.expected_grid = img_size // patch_size
        self.out_channels = out_channels

        self.input_proj = nn.Linear(embed_dim, model_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(model_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.RMSNorm(model_dim)
        self.patch_head = nn.Linear(model_dim, out_channels * patch_size * patch_size)

        self.refine = ConvStack(out_channels, refine_channels, refine_blocks)

    def forward(self, x):
        """
        Args:
            x: (B, N, embed_dim) patch tokens; N must be a perfect square
                equal to `(img_size // patch_size) ** 2`.

        Returns:
            (B, out_channels, img_size, img_size), raw (no output
            activation) — apply sigmoid/clamp yourself if your target range
            calls for it.
        """
        B, N, _ = x.shape
        grid = math.isqrt(N)
        assert grid * grid == N, f"PatchDecoder expected a square token grid, got N={N} tokens"
        assert grid == self.expected_grid, (
            f"PatchDecoder grid mismatch: got {grid}x{grid} tokens, expected "
            f"{self.expected_grid}x{self.expected_grid}. Match patch_size with the "
            "encoder tokenization."
        )

        x = self.input_proj(x)
        x = x + build_2d_sincos_pos_embed(
            grid, grid, x.shape[-1], device=x.device, dtype=x.dtype
        )

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.patch_head(x)

        p = self.patch_size
        x = x.view(B, grid, grid, self.out_channels, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, self.out_channels, grid * p, grid * p)

        return x + self.refine(x)
