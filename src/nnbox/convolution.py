"""Convolutional blocks for NCHW feature maps.

Residual like the transformer blocks, but normalized with GroupNorm rather
than RMSNorm, since these operate on spatial feature maps rather than a token
sequence.
"""

from torch import nn

__all__ = ["ConvStack", "ResidualConvBlock"]


def group_norm(channels, max_groups=32):
    """Build an `nn.GroupNorm` over `channels`, choosing a valid group count.

    `nn.GroupNorm` requires its group count to divide `channels`, so a fixed
    32 breaks on channel counts that aren't multiples of it. This picks the
    largest divisor of `channels` that is at most `max_groups`.
    """
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualConvBlock(nn.Module):
    """Residual conv block: two 3x3 convs with GroupNorm, GELU in between.

    Args:
        channels: input and output channel count (unchanged by this block).
    """

    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            group_norm(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            group_norm(channels),
        )

    def forward(self, x):
        """
        Args:
            x: (B, channels, H, W)

        Returns:
            (B, channels, H, W)
        """
        return x + self.net(x)


class ConvStack(nn.Module):
    """Channel-preserving stack of `ResidualConvBlock`s, bracketed by
    projections to and from a hidden width.

    Input and output channel counts match, so callers are free to apply the
    result residually.

    Args:
        channels: input and output channels.
        hidden_channels: width the residual blocks run at.
        depth: number of `ResidualConvBlock`s; 0 leaves just the two
            projections.
    """

    def __init__(self, channels, hidden_channels, depth):
        super().__init__()
        assert hidden_channels > 0, f"hidden_channels must be > 0, got {hidden_channels}"
        assert depth >= 0, f"depth must be >= 0, got {depth}"

        layers = [
            nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1),
            group_norm(hidden_channels),
            nn.GELU(),
        ]
        layers.extend(ResidualConvBlock(hidden_channels) for _ in range(depth))
        layers.append(nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (B, channels, H, W)

        Returns:
            (B, channels, H, W)
        """
        return self.net(x)
