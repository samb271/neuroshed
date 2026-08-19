"""ViT shape and position-embedding-interpolation tests."""

import pytest
import torch

from ml_modules import MLP
from ml_modules.models.vit import VIT_CONFIGS, ViT, vit_config

B, IMG_SIZE = 2, 32


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def tiny_config(**overrides):
    return vit_config("tiny", **overrides)


@pytest.mark.parametrize("name", VIT_CONFIGS)
def test_presets_build_and_run(name):
    config = vit_config(name, patch_size=16, dropout=0.0)
    model = ViT(config, img_size=IMG_SIZE).eval()
    num_patches = (IMG_SIZE // config.patch_size) ** 2

    out = model(torch.randn(B, 3, IMG_SIZE, IMG_SIZE))

    assert out.shape == (B, 1 + num_patches, config.dim)


def test_image_size_must_be_divisible_by_patch_size():
    with pytest.raises(AssertionError):
        ViT(tiny_config(patch_size=16), img_size=IMG_SIZE + 1)


def test_cls_and_patch_tokens_are_distinguishable():
    """Index 0 (CLS) and the patch tokens must carry different information,
    i.e. the model isn't collapsing them to the same representation."""
    model = ViT(tiny_config(patch_size=16), img_size=IMG_SIZE).eval()
    out = model(torch.randn(B, 3, IMG_SIZE, IMG_SIZE))

    assert not torch.allclose(out[:, 0], out[:, 1:].mean(dim=1))


def test_forward_at_different_resolution_interpolates_pos_embed():
    """A resolution other than img_size must not error and must produce a
    different position embedding than the default grid (i.e. interpolation
    actually ran, not just reused the original table)."""
    model = ViT(tiny_config(patch_size=16), img_size=IMG_SIZE).eval()

    default_pos = model._position_embed(model.default_grid_size)
    larger_pos = model._position_embed((4, 4))

    assert larger_pos.shape == (1, 4 * 4 + 1, model.config.dim)
    assert default_pos.shape != larger_pos.shape

    num_patches = (IMG_SIZE * 2 // 16) ** 2
    out = model(torch.randn(B, 3, IMG_SIZE * 2, IMG_SIZE * 2))
    assert out.shape == (B, 1 + num_patches, model.config.dim)


def test_use_cls_token_false_returns_patch_tokens_only():
    model = ViT(tiny_config(patch_size=16), img_size=IMG_SIZE, use_cls_token=False).eval()
    num_patches = (IMG_SIZE // 16) ** 2

    out = model(torch.randn(B, 3, IMG_SIZE, IMG_SIZE))

    assert out.shape == (B, num_patches, model.config.dim)
    assert not hasattr(model, "cls_token")


def test_use_cls_token_false_interpolates_pos_embed():
    model = ViT(tiny_config(patch_size=16), img_size=IMG_SIZE, use_cls_token=False).eval()
    num_patches = (IMG_SIZE * 2 // 16) ** 2

    out = model(torch.randn(B, 3, IMG_SIZE * 2, IMG_SIZE * 2))

    assert out.shape == (B, num_patches, model.config.dim)


def test_vit_backward():
    """Gradients must reach the input pixels through the whole stack."""
    model = ViT(tiny_config(patch_size=16), img_size=IMG_SIZE)
    x = torch.randn(B, 3, IMG_SIZE, IMG_SIZE, requires_grad=True)

    model(x).sum().backward()

    assert x.grad is not None and x.grad.abs().sum() > 0


def test_composes_with_mlp_classification_head():
    """The encoder-only design: a classifier is the encoder's CLS token plus
    a caller-assembled MLP, not something ViT builds itself."""
    model = ViT(tiny_config(patch_size=16), img_size=IMG_SIZE).eval()
    head = MLP(model.config.dim, out_dim=10).eval()

    logits = head(model(torch.randn(B, 3, IMG_SIZE, IMG_SIZE))[:, 0])

    assert logits.shape == (B, 10)
