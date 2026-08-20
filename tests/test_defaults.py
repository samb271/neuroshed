"""Default-value tests.

The package hardcodes its design decisions to limit drift between projects,
so the defaults are part of the contract. The dropout sweep is introspective
on purpose: a layer added later with a different default fails here without
anyone remembering to extend this file.
"""

import inspect

import pytest
import torch

import nnbox
from nnbox import MLP, MultiheadAttention
from nnbox.models.decoder import PatchDecoder
from nnbox.models.vit import ViT, ViTConfig, vit_config

DROPOUT_DEFAULT = 0.1
B, T, DIM, HEADS = 2, 5, 32, 4


def _dropout_defaults():
    """Every public constructor taking a `dropout`, with its default."""
    public = [getattr(nnbox, name) for name in nnbox.__all__]
    found = []
    for obj in public + [ViTConfig, PatchDecoder]:
        if not inspect.isclass(obj):
            continue
        param = inspect.signature(obj).parameters.get("dropout")
        if param is not None and param.default is not inspect.Parameter.empty:
            found.append((obj.__name__, param.default))
    return found


DROPOUT_DEFAULTS = _dropout_defaults()


def test_the_dropout_sweep_actually_found_something():
    """Guard against the introspection silently matching nothing."""
    names = {name for name, _ in DROPOUT_DEFAULTS}
    assert {"MultiheadAttention", "MLP", "TransformerBlock", "ViTConfig"} <= names


@pytest.mark.parametrize("name, default", DROPOUT_DEFAULTS, ids=[n for n, _ in DROPOUT_DEFAULTS])
def test_dropout_defaults_to_the_package_value(name, default):
    assert default == DROPOUT_DEFAULT


def test_vit_preset_dropout_defaults_to_the_package_value():
    assert vit_config("tiny").dropout == DROPOUT_DEFAULT


def test_vit_wires_config_dropout_into_the_stack():
    model = ViT(vit_config("tiny", patch_size=16), img_size=32)

    assert model.pos_drop.p == DROPOUT_DEFAULT
    assert model.blocks[0].self_attn.dropout == DROPOUT_DEFAULT


def test_decoder_wires_dropout_into_the_stack():
    model = PatchDecoder(img_size=32, patch_size=16, embed_dim=24, model_dim=16, depth=1,
                         num_heads=2, refine_channels=8, refine_blocks=1)

    assert model.blocks[0].self_attn.dropout == DROPOUT_DEFAULT


def test_attention_dropout_is_live_in_training_by_default():
    """A nonzero default is only meaningful if it reaches the attention."""
    attn = MultiheadAttention(DIM, HEADS).train()
    x = torch.randn(B, T, DIM)

    torch.manual_seed(1)
    first = attn(x, x, x)
    torch.manual_seed(2)
    second = attn(x, x, x)

    assert not torch.allclose(first, second)


def test_attention_dropout_is_off_in_eval():
    attn = MultiheadAttention(DIM, HEADS).eval()
    x = torch.randn(B, T, DIM)

    torch.testing.assert_close(attn(x, x, x), attn(x, x, x))


def test_mlp_dropout_is_live_in_training_by_default():
    mlp = MLP(DIM).train()
    x = torch.randn(B, T, DIM)

    torch.manual_seed(1)
    first = mlp(x)
    torch.manual_seed(2)
    second = mlp(x)

    assert not torch.allclose(first, second)
