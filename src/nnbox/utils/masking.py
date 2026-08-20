"""Attention mask conversions.

All masks in this package are *padding* masks with the convention
``True == ignore this position``, matching ``nn.MultiheadAttention``'s
``key_padding_mask``.
"""

import torch

__all__ = ["padding_mask_to_bias"]


def padding_mask_to_bias(mask, dtype):
    """Convert a bool padding mask to an additive attention bias.

    `scaled_dot_product_attention` takes either a bool mask or an additive
    float bias; the float form is used here so callers can sum it with other
    biases before the softmax.

    Args:
        mask: (B, Tk) bool, True where the key is padding.
        dtype: dtype of the returned bias; match the query it is added to.

    Returns:
        (B, 1, 1, Tk), 0 where attended and -inf where masked out,
        broadcastable across heads and query positions.
    """
    bias = torch.zeros(mask.shape, device=mask.device, dtype=dtype)
    return bias.masked_fill(mask, float("-inf"))[:, None, None, :]
