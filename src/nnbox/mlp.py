"""Feed-forward networks."""

import torch.nn as nn

__all__ = ["MLP"]


class MLP(nn.Module):
    """Two-layer GELU feed-forward network.

    Used as the feed-forward sublayer of the transformer blocks, and standalone
    as a task head on an encoder's output.

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
