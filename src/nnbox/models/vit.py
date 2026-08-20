"""Vision Transformer (Dosovitskiy et al., 2020).

Patchify, prepend a CLS token, add a learned position embedding, then run the
generic `TransformerBlock` stack. The transformer itself is not ViT-specific
and is reused as-is; only the patch/position embedding here is.
"""

from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnbox.transformer import TransformerBlock

__all__ = ["ViT", "ViTConfig", "VIT_CONFIGS", "vit_config"]


@dataclass(frozen=True)
class ViTConfig:
    """Size knobs for a ViT encoder. Build one from a preset via `vit_config`."""

    dim: int
    depth: int
    num_heads: int
    mlp_ratio: float = 4.0
    patch_size: int = 16
    dropout: float = 0.1


# Base/Large/Huge are Table 1 of Dosovitskiy et al., 2020. Tiny/Small are from
# Touvron et al., 2021 (DeiT), now the de facto standard rest of the family.
VIT_CONFIGS = {
    "tiny": ViTConfig(dim=192, depth=12, num_heads=3),
    "small": ViTConfig(dim=384, depth=12, num_heads=6),
    "base": ViTConfig(dim=768, depth=12, num_heads=12),
    "large": ViTConfig(dim=1024, depth=24, num_heads=16),
    "huge": ViTConfig(dim=1280, depth=32, num_heads=16),
}


def vit_config(name, **overrides):
    """Look up a named preset (e.g. "base"), with optional field overrides.

    Common overrides: `patch_size` (16 or 32 in the paper), `dropout`.
    """
    return replace(VIT_CONFIGS[name], **overrides)


class PatchEmbed(nn.Module):
    """Splits an image into non-overlapping patches and projects each to `dim`.

    A single strided convolution does both: it is equivalent to unfolding into
    flat patches and applying a shared linear projection.
    """

    def __init__(self, patch_size, in_chans, dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W); H and W must be divisible by `patch_size`.

        Returns:
            patches: (B, grid_h * grid_w, dim)
            grid_size: (grid_h, grid_w)
        """
        H, W = x.shape[-2:]
        assert H % self.patch_size == 0 and W % self.patch_size == 0, (
            f"image size ({H}, {W}) must be divisible by patch_size ({self.patch_size})"
        )
        x = self.proj(x)
        grid_size = x.shape[-2:]
        return x.flatten(2).transpose(1, 2), grid_size


class ViT(nn.Module):
    """Vision Transformer encoder.

    Encoder only: patchify, optionally prepend a CLS token, add a position
    embedding, run the transformer stack, final norm. Returns the full token
    sequence with no pooling or head — compose those yourself for whatever
    task you're using it for, e.g. an `MLP` from `nnbox.mlp` on
    `out[:, 0]` for classification, or `out[:, 1:]` for dense/patch-level
    tasks.

    Args:
        config: a `ViTConfig`, e.g. `vit_config("base", patch_size=32)`.
        img_size: image side length used to size the position embedding; other
            square resolutions divisible by `config.patch_size` also work at
            forward time, via bicubic interpolation of the position embedding.
        in_chans: input channels.
        use_cls_token: prepend a learnable CLS token (paper default). Turn off
            for encoders meant to be consumed patch-token-only, e.g. via mean
            pooling or a dense per-patch head.
    """

    def __init__(self, config, img_size=224, in_chans=3, use_cls_token=True):
        super().__init__()
        assert img_size % config.patch_size == 0, (
            f"img_size ({img_size}) must be divisible by patch_size ({config.patch_size})"
        )
        self.config = config
        self.use_cls_token = use_cls_token
        self.num_prefix_tokens = 1 if use_cls_token else 0

        self.patch_embed = PatchEmbed(config.patch_size, in_chans, config.dim)
        grid_size = img_size // config.patch_size
        self.default_grid_size = (grid_size, grid_size)
        num_patches = grid_size * grid_size

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        num_tokens = self.num_prefix_tokens + num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, config.dim))
        self.pos_drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.dim, config.num_heads,
                    mlp_ratio=config.mlp_ratio, dropout=config.dropout,
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.RMSNorm(config.dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _position_embed(self, grid_size):
        """Position embedding for `grid_size`, bicubic-resized if it differs
        from `self.default_grid_size`."""
        if grid_size == self.default_grid_size:
            return self.pos_embed

        n = self.num_prefix_tokens
        prefix_pos, patch_pos = self.pos_embed[:, :n], self.pos_embed[:, n:]
        dim = patch_pos.shape[-1]
        gh0, gw0 = self.default_grid_size
        patch_pos = patch_pos.reshape(1, gh0, gw0, dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=grid_size, mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], dim)
        return torch.cat((prefix_pos, patch_pos), dim=1)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)

        Returns:
            (B, num_prefix_tokens + num_patches, dim) tokens after the final
            norm. If `use_cls_token`, index 0 is the CLS token and the rest
            are patch tokens in row-major grid order (matches e.g. DINOv2's
            `x_norm_clstoken` / `x_norm_patchtokens` split via `out[:, 0]` /
            `out[:, 1:]`); otherwise every token is a patch token.
        """
        B = x.shape[0]
        x, grid_size = self.patch_embed(x)
        if self.use_cls_token:
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        x = x + self._position_embed(grid_size)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        return self.norm(x)
