"""Padding-mask tests.

The convention (`True == ignore`) is stated in three docstrings and relied on
by every attention path, so it gets a direct test rather than only indirect
coverage through the layers.
"""

import torch

from nnbox.utils import padding_mask_to_bias

B, TK = 2, 4


def test_bias_is_zero_where_attended_and_neg_inf_where_masked():
    mask = torch.tensor([[False, True, False, True]])

    bias = padding_mask_to_bias(mask, torch.float32)

    assert bias.shape == (1, 1, 1, 4)
    torch.testing.assert_close(bias[0, 0, 0, 0], torch.tensor(0.0))
    assert bias[0, 0, 0, 1] == float("-inf")
    assert bias[0, 0, 0, 3] == float("-inf")


def test_bias_broadcasts_over_heads_and_queries():
    """(B, 1, 1, Tk) so it can be added to a (B, H, Tq, Tk) logit tensor."""
    mask = torch.zeros(B, TK, dtype=torch.bool)

    bias = padding_mask_to_bias(mask, torch.float32)

    assert bias.shape == (B, 1, 1, TK)
    assert (bias + torch.zeros(B, 3, 5, TK)).shape == (B, 3, 5, TK)


def test_bias_follows_requested_dtype():
    mask = torch.zeros(B, TK, dtype=torch.bool)
    assert padding_mask_to_bias(mask, torch.bfloat16).dtype == torch.bfloat16
