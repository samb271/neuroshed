"""Rotary embedding tests.

The shape tests are routine; `test_attention_logits_depend_only_on_offset` is
the one that matters — it is the property RoPE exists to provide, and it fails
loudly if the rotation is applied to the wrong axis or the halves are paired
incorrectly.
"""

import pytest
import torch

from nnbox import RotaryEmbedding, SelfAttention, TransformerBlock, apply_rotary_emb

B, T, DIM, HEADS = 2, 6, 32, 4
HEAD_DIM = DIM // HEADS


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def test_tables_shape_from_seq_len():
    cos, sin = RotaryEmbedding(HEAD_DIM)(seq_len=T)
    assert cos.shape == sin.shape == (1, 1, T, HEAD_DIM)


def test_tables_shape_from_batched_positions():
    positions = torch.arange(T).expand(B, T)
    cos, sin = RotaryEmbedding(HEAD_DIM)(positions=positions)
    assert cos.shape == sin.shape == (B, 1, T, HEAD_DIM)


def test_requires_exactly_one_of_seq_len_or_positions():
    rope = RotaryEmbedding(HEAD_DIM)
    with pytest.raises(ValueError):
        rope()
    with pytest.raises(ValueError):
        rope(seq_len=T, positions=torch.arange(T))


def test_odd_head_dim_rejected():
    with pytest.raises(AssertionError):
        RotaryEmbedding(7)


def test_rope_head_dim_must_match_attention():
    with pytest.raises(AssertionError):
        SelfAttention(DIM, HEADS, rope=RotaryEmbedding(HEAD_DIM * 2))


def test_rotation_preserves_norm():
    rope = RotaryEmbedding(HEAD_DIM)
    x = torch.randn(B, HEADS, T, HEAD_DIM)
    rotated = apply_rotary_emb(x, *rope(seq_len=T))
    torch.testing.assert_close(rotated.norm(dim=-1), x.norm(dim=-1))


def test_position_zero_is_identity():
    rope = RotaryEmbedding(HEAD_DIM)
    x = torch.randn(B, HEADS, 1, HEAD_DIM)
    torch.testing.assert_close(apply_rotary_emb(x, *rope(seq_len=1)), x)


def test_attention_logits_depend_only_on_offset():
    """q·k for a pair of positions must be unchanged by shifting both."""
    rope = RotaryEmbedding(HEAD_DIM)
    q = torch.randn(1, 1, 1, HEAD_DIM)
    k = torch.randn(1, 1, 1, HEAD_DIM)

    def logit(i, j):
        qi = apply_rotary_emb(q, *rope(positions=torch.tensor([i])))
        kj = apply_rotary_emb(k, *rope(positions=torch.tensor([j])))
        return (qi * kj).sum()

    torch.testing.assert_close(logit(0, 3), logit(5, 8))
    torch.testing.assert_close(logit(2, 1), logit(11, 10))


def test_absolute_position_changes_logits():
    """Sanity check the previous test is not passing because RoPE is a no-op."""
    rope = RotaryEmbedding(HEAD_DIM)
    q, k = torch.randn(1, 1, 1, HEAD_DIM), torch.randn(1, 1, 1, HEAD_DIM)

    def logit(i, j):
        qi = apply_rotary_emb(q, *rope(positions=torch.tensor([i])))
        kj = apply_rotary_emb(k, *rope(positions=torch.tensor([j])))
        return (qi * kj).sum()

    assert not torch.allclose(logit(0, 1), logit(0, 4))


def test_self_attention_with_rope_shape():
    attn = SelfAttention(DIM, HEADS, rope=RotaryEmbedding(HEAD_DIM)).eval()
    assert attn(torch.randn(B, T, DIM)).shape == (B, T, DIM)


def test_explicit_positions_match_default_arange():
    attn = SelfAttention(DIM, HEADS, rope=RotaryEmbedding(HEAD_DIM)).eval()
    x = torch.randn(B, T, DIM)
    torch.testing.assert_close(attn(x), attn(x, positions=torch.arange(T)))


def test_rope_changes_attention_output():
    torch.manual_seed(0)
    plain = SelfAttention(DIM, HEADS).eval()
    torch.manual_seed(0)
    rotary = SelfAttention(DIM, HEADS, rope=RotaryEmbedding(HEAD_DIM)).eval()

    x = torch.randn(B, T, DIM)
    assert not torch.allclose(plain(x), rotary(x))


def test_shared_rope_adds_nothing_to_state_dict():
    rope = RotaryEmbedding(HEAD_DIM)
    with_rope = TransformerBlock(DIM, HEADS, rope=rope).state_dict()
    without = TransformerBlock(DIM, HEADS).state_dict()
    assert with_rope.keys() == without.keys()


def test_block_with_rope_shape():
    block = TransformerBlock(DIM, HEADS, rope=RotaryEmbedding(HEAD_DIM)).eval()
    assert block(torch.randn(B, T, DIM)).shape == (B, T, DIM)


def test_rope_follows_module_dtype():
    """Tables must come back in the tensor's dtype, not float32."""
    rope = RotaryEmbedding(HEAD_DIM)
    x = torch.randn(B, HEADS, T, HEAD_DIM, dtype=torch.bfloat16)
    cos, sin = rope(seq_len=T, dtype=x.dtype)
    assert apply_rotary_emb(x, cos, sin).dtype == torch.bfloat16
