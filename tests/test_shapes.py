"""Forward-pass shape and masking tests.

Cheap insurance: a block whose output shape or mask handling silently changed is
the kind of bug that only shows up several hours into a training run.
"""

import pytest
import torch

from nnbox import (
    MLP,
    CrossAttention,
    CrossAttentionBlock,
    MultiheadAttention,
    SelfAttention,
    TransformerBlock,
)

B, TQ, TK, DIM, HEADS = 2, 5, 7, 32, 4


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


def test_cross_attention_block_backward():
    """Gradients must reach both the query and the context path."""
    block = CrossAttentionBlock(DIM, HEADS)
    x = torch.randn(B, TQ, DIM, requires_grad=True)
    context = torch.randn(B, TK, DIM, requires_grad=True)

    block(x, context).sum().backward()

    assert x.grad is not None and x.grad.abs().sum() > 0
    assert context.grad is not None and context.grad.abs().sum() > 0
