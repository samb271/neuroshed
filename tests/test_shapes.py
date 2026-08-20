"""Forward-pass shape and masking tests.

Cheap insurance: a block whose output shape or mask handling silently changed is
the kind of bug that only shows up several hours into a training run.
"""

import pytest
import torch

from nnbox import (
    MLP,
    ConvStack,
    CrossAttention,
    CrossAttentionBlock,
    MultiheadAttention,
    ResidualConvBlock,
    SelfAttention,
    TransformerBlock,
)

B, TQ, TK, DIM, HEADS = 2, 5, 7, 32, 4
CHANNELS, HW = 6, 8


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def test_multihead_attention_shape():
    attn = MultiheadAttention(DIM, HEADS).eval()
    q, kv = torch.randn(B, TQ, DIM), torch.randn(B, TK, DIM)
    assert attn(q, kv, kv).shape == (B, TQ, DIM)


def test_dim_must_divide_heads():
    with pytest.raises(AssertionError):
        MultiheadAttention(30, HEADS)


def test_self_attention_shape():
    assert SelfAttention(DIM, HEADS).eval()(torch.randn(B, TQ, DIM)).shape == (B, TQ, DIM)


def test_cross_attention_shape():
    attn = CrossAttention(DIM, HEADS).eval()
    out = attn(torch.randn(B, TQ, DIM), torch.randn(B, TK, DIM))
    assert out.shape == (B, TQ, DIM)


def test_padded_context_is_ignored():
    """Changing masked-out context must not change the output."""
    attn = CrossAttention(DIM, HEADS).eval()
    x = torch.randn(B, TQ, DIM)
    context = torch.randn(B, TK, DIM)
    mask = torch.zeros(B, TK, dtype=torch.bool)
    mask[:, -2:] = True

    out = attn(x, context, context_padding_mask=mask)
    context[:, -2:] = torch.randn(B, 2, DIM)
    out_perturbed = attn(x, context, context_padding_mask=mask)

    torch.testing.assert_close(out, out_perturbed)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, DIM),
        ({"out_dim": 8}, 8),
        ({"hidden_dim": 3, "out_dim": 8}, 8),
    ],
)
def test_mlp_shapes(kwargs, expected):
    assert MLP(DIM, **kwargs).eval()(torch.randn(B, TQ, DIM)).shape == (B, TQ, expected)


def test_mlp_hidden_dim_overrides_ratio():
    mlp = MLP(DIM, hidden_dim=3, mlp_ratio=4.0)
    assert mlp.net[0].out_features == 3


def test_transformer_block_shape():
    block = TransformerBlock(DIM, HEADS).eval()
    mask = torch.zeros(B, TQ, dtype=torch.bool)
    mask[:, -1] = True
    assert block(torch.randn(B, TQ, DIM), key_padding_mask=mask).shape == (B, TQ, DIM)


def test_cross_attention_block_shape():
    block = CrossAttentionBlock(DIM, HEADS).eval()
    out = block(torch.randn(B, TQ, DIM), torch.randn(B, TK, DIM))
    assert out.shape == (B, TQ, DIM)


def test_cross_attention_block_masks_the_query_side():
    """`key_padding_mask` marks padding in `x` itself: a masked position must
    not leak into the other positions' outputs through the self-attention."""
    block = CrossAttentionBlock(DIM, HEADS).eval()
    x = torch.randn(B, TQ, DIM)
    context = torch.randn(B, TK, DIM)
    mask = torch.zeros(B, TQ, dtype=torch.bool)
    mask[:, -1] = True

    out = block(x, context, key_padding_mask=mask)
    x[:, -1] = torch.randn(B, DIM)
    out_perturbed = block(x, context, key_padding_mask=mask)

    torch.testing.assert_close(out[:, :-1], out_perturbed[:, :-1])


def test_cross_attention_block_masks_the_context_side():
    block = CrossAttentionBlock(DIM, HEADS).eval()
    x = torch.randn(B, TQ, DIM)
    context = torch.randn(B, TK, DIM)
    mask = torch.zeros(B, TK, dtype=torch.bool)
    mask[:, -2:] = True

    out = block(x, context, context_padding_mask=mask)
    context[:, -2:] = torch.randn(B, 2, DIM)
    out_perturbed = block(x, context, context_padding_mask=mask)

    torch.testing.assert_close(out, out_perturbed)


def test_cross_attention_block_backward():
    """Gradients must reach both the query and the context path."""
    block = CrossAttentionBlock(DIM, HEADS)
    x = torch.randn(B, TQ, DIM, requires_grad=True)
    context = torch.randn(B, TK, DIM, requires_grad=True)

    block(x, context).sum().backward()

    assert x.grad is not None and x.grad.abs().sum() > 0
    assert context.grad is not None and context.grad.abs().sum() > 0


def test_residual_conv_block_preserves_shape():
    block = ResidualConvBlock(CHANNELS).eval()
    x = torch.randn(B, CHANNELS, HW, HW)
    assert block(x).shape == x.shape


@pytest.mark.parametrize("channels", [3, 6, 100, 128])
def test_group_norm_sized_for_any_channel_count(channels):
    """GroupNorm needs a group count dividing `channels`; odd widths like
    3 or 100 must not blow up."""
    block = ResidualConvBlock(channels).eval()
    assert block(torch.randn(1, channels, 4, 4)).shape == (1, channels, 4, 4)


@pytest.mark.parametrize("depth", [0, 1, 3])
def test_conv_stack_preserves_channels(depth):
    stack = ConvStack(CHANNELS, 16, depth).eval()
    x = torch.randn(B, CHANNELS, HW, HW)
    assert stack(x).shape == x.shape


def test_conv_stack_rejects_bad_sizes():
    with pytest.raises(AssertionError):
        ConvStack(CHANNELS, 0, 2)
    with pytest.raises(AssertionError):
        ConvStack(CHANNELS, 16, -1)


def test_conv_stack_backward():
    stack = ConvStack(CHANNELS, 16, 2)
    x = torch.randn(B, CHANNELS, HW, HW, requires_grad=True)

    stack(x).sum().backward()

    assert x.grad is not None and x.grad.abs().sum() > 0
