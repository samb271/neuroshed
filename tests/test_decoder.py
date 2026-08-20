"""PatchDecoder shape and reconstruction tests."""

import pytest
import torch

from nnbox.models.decoder import PatchDecoder

B, IMG_SIZE, PATCH_SIZE, EMBED_DIM = 2, 32, 16, 24
GRID = IMG_SIZE // PATCH_SIZE
NUM_TOKENS = GRID * GRID


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def tiny_decoder(**overrides):
    kwargs = {
        "img_size": IMG_SIZE,
        "patch_size": PATCH_SIZE,
        "embed_dim": EMBED_DIM,
        "model_dim": 16,
        "depth": 2,
        "num_heads": 2,
        "refine_channels": 8,
        "refine_blocks": 1,
    }
    kwargs.update(overrides)
    return PatchDecoder(**kwargs)


def test_decoder_shape():
    model = tiny_decoder().eval()
    out = model(torch.randn(B, NUM_TOKENS, EMBED_DIM))
    assert out.shape == (B, 3, IMG_SIZE, IMG_SIZE)


def test_img_size_must_be_divisible_by_patch_size():
    with pytest.raises(AssertionError):
        tiny_decoder(img_size=IMG_SIZE + 1)


def test_model_dim_must_divide_num_heads():
    with pytest.raises(AssertionError):
        tiny_decoder(model_dim=15, num_heads=2)


def test_token_count_must_be_a_perfect_square():
    model = tiny_decoder().eval()
    with pytest.raises(AssertionError):
        model(torch.randn(B, NUM_TOKENS + 1, EMBED_DIM))


def test_model_dim_must_be_divisible_by_4():
    """Required by the 2D sin-cos position embedding, and caught at build
    time rather than several forward passes later."""
    with pytest.raises(AssertionError):
        tiny_decoder(model_dim=18, num_heads=2)


def test_decodes_a_larger_grid_at_its_own_resolution():
    """Resolution-agnostic: output size follows the token grid, not img_size."""
    model = tiny_decoder().eval()
    grid = GRID + 1

    out = model(torch.randn(B, grid * grid, EMBED_DIM))

    assert out.shape == (B, 3, grid * PATCH_SIZE, grid * PATCH_SIZE)


def test_decodes_a_smaller_grid_at_its_own_resolution():
    model = tiny_decoder(img_size=PATCH_SIZE * 4).eval()

    out = model(torch.randn(B, 2 * 2, EMBED_DIM))

    assert out.shape == (B, 3, 2 * PATCH_SIZE, 2 * PATCH_SIZE)


def test_off_grid_decode_uses_every_parameter():
    """No resolution-specific weights: gradients must still reach the whole
    model at a grid the decoder was not built around."""
    model = tiny_decoder()
    grid = GRID + 1

    model(torch.randn(B, grid * grid, EMBED_DIM)).sum().backward()

    ungrad = [n for n, p in model.named_parameters() if p.grad is None]
    assert ungrad == []


def test_default_grid_position_embed_is_cached():
    """The hot path must reuse the buffer instead of rebuilding the table."""
    model = tiny_decoder().eval()
    assert model._position_embed(GRID, torch.device("cpu"), torch.float32) is model.pos_embed


def test_off_grid_position_embed_is_built_on_demand():
    model = tiny_decoder().eval()
    pos = model._position_embed(GRID + 1, torch.device("cpu"), torch.float32)
    assert pos.shape == (1, (GRID + 1) ** 2, model.model_dim)


def test_position_embed_follows_requested_dtype():
    model = tiny_decoder().eval()
    device = torch.device("cpu")
    assert model._position_embed(GRID, device, torch.bfloat16).dtype == torch.bfloat16
    assert model._position_embed(GRID + 1, device, torch.bfloat16).dtype == torch.bfloat16


def test_position_embed_is_not_learned_state():
    """A fixed table is a pure function of the config, so it belongs in
    neither the state dict nor the parameters."""
    model = tiny_decoder()
    assert "pos_embed" not in model.state_dict()
    assert all(p is not model.pos_embed for p in model.parameters())


def test_projects_from_a_different_embed_dim():
    """embed_dim and model_dim are decoupled by the input projection."""
    model = tiny_decoder(embed_dim=EMBED_DIM * 2).eval()

    out = model(torch.randn(B, NUM_TOKENS, EMBED_DIM * 2))

    assert out.shape == (B, 3, IMG_SIZE, IMG_SIZE)


def test_out_channels_is_respected():
    model = tiny_decoder(out_channels=1).eval()

    out = model(torch.randn(B, NUM_TOKENS, EMBED_DIM))

    assert out.shape == (B, 1, IMG_SIZE, IMG_SIZE)


def test_refine_blocks_zero_still_runs():
    model = tiny_decoder(refine_blocks=0).eval()
    out = model(torch.randn(B, NUM_TOKENS, EMBED_DIM))
    assert out.shape == (B, 3, IMG_SIZE, IMG_SIZE)


def test_output_has_no_activation_applied():
    """Raw conv output isn't clamped to any particular range."""
    model = tiny_decoder().eval()
    out = model(torch.randn(B, NUM_TOKENS, EMBED_DIM) * 10)
    assert out.abs().max() > 1.0


def test_decoder_backward():
    model = tiny_decoder()
    x = torch.randn(B, NUM_TOKENS, EMBED_DIM, requires_grad=True)

    model(x).sum().backward()

    assert x.grad is not None and x.grad.abs().sum() > 0


def test_checkpoint_loads_across_resolutions():
    """Nothing in the state dict is resolution-specific, so weights trained at
    one img_size load into a decoder built around another."""
    trained = tiny_decoder()
    other = tiny_decoder(img_size=PATCH_SIZE * 8)

    other.load_state_dict(trained.state_dict())

    assert other.default_grid == 8
