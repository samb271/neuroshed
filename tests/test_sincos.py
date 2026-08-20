"""2D sin-cos position embedding tests."""

import pytest
import torch

from nnbox import build_2d_sincos_pos_embed


def test_shape_covers_the_full_grid():
    emb = build_2d_sincos_pos_embed(3, 5, 8, device="cpu", dtype=torch.float32)
    assert emb.shape == (1, 3 * 5, 8)


def test_dim_must_be_divisible_by_4():
    with pytest.raises(AssertionError):
        build_2d_sincos_pos_embed(3, 3, 6, device="cpu", dtype=torch.float32)


def test_is_deterministic_and_position_dependent():
    emb1 = build_2d_sincos_pos_embed(2, 2, 8, device="cpu", dtype=torch.float32)
    emb2 = build_2d_sincos_pos_embed(2, 2, 8, device="cpu", dtype=torch.float32)
    torch.testing.assert_close(emb1, emb2)
    assert not torch.allclose(emb1[:, 0], emb1[:, 1])


def test_follows_requested_dtype():
    emb = build_2d_sincos_pos_embed(2, 2, 8, device="cpu", dtype=torch.bfloat16)
    assert emb.dtype == torch.bfloat16
