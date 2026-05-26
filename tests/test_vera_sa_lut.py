import pytest
import torch

from model.vera_sa_lut import VeraSALUT, context_tv, lut_monotonicity


def _small_model(**overrides):
    kwargs = dict(
        vision_channels=16,
        hidden_dim=24,
        exec_dim=12,
        feature_channels=16,
        grid_size=5,
        context_size=2,
        basis_count=4,
        semantic_layers=2,
        attention_channels=16,
        attention_heads=4,
        max_attention_tokens=64,
    )
    kwargs.update(overrides)
    return VeraSALUT(**kwargs)


def _inputs(batch_size=2, height=32, width=32):
    return dict(
        i_in=torch.rand(batch_size, 3, height, width),
        i_base=torch.rand(batch_size, 3, height, width),
        vision_spatial=torch.rand(batch_size, 8, 8, 16),
        hidden_bank=torch.rand(batch_size, 2, 3, 24),
        z_exec=torch.rand(batch_size, 3, 12),
    )


def test_vera_sa_lut_forward_shapes_and_gradients():
    torch.manual_seed(0)
    model = _small_model()
    inputs = _inputs()

    out = model(
        inputs["i_in"],
        vision_spatial=inputs["vision_spatial"],
        hidden_bank=inputs["hidden_bank"],
        z_exec=inputs["z_exec"],
        return_aux=True,
    )

    assert out.image.shape == inputs["i_in"].shape
    assert out.lut_image.shape == inputs["i_in"].shape
    assert out.context.shape == (2, 1, 32, 32)
    assert out.lut.shape == (2, 5, 5, 5, 2, 3)
    assert out.alpha.shape == (2, 4)
    assert torch.allclose(out.alpha.sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.all((0.0 <= out.context) & (out.context <= 1.0))
    assert [tuple(x.shape) for x in out.affinity_maps] == [(2, 3, 8, 8), (2, 3, 4, 4), (2, 3, 2, 2)]

    (out.image.mean() + out.context.mean() + out.alpha.mean()).backward()
    assert model.delta_basis.grad is not None
    assert model.delta_basis.grad.abs().sum() > 0


@pytest.mark.parametrize(
    ("variant", "needs_vision", "needs_hidden", "has_affinity"),
    [
        ("E1", False, False, False),
        ("E2", True, False, False),
        ("E3", True, True, True),
        ("E4", True, True, True),
    ],
)
def test_ablation_variants_match_pipeline_contract(variant, needs_vision, needs_hidden, has_affinity):
    model = VeraSALUT.from_ablation(
        variant,
        vision_channels=16,
        hidden_dim=24 if needs_hidden else None,
        exec_dim=12 if needs_hidden else None,
        feature_channels=16,
        grid_size=5,
        basis_count=4,
        semantic_layers=2,
        attention_channels=16,
        attention_heads=4,
        max_attention_tokens=64,
    )
    inputs = _inputs()
    kwargs = {}
    if needs_vision:
        kwargs["vision_spatial"] = inputs["vision_spatial"]
    if needs_hidden:
        kwargs["hidden_bank"] = inputs["hidden_bank"]
        kwargs["z_exec"] = inputs["z_exec"]

    out = model(inputs["i_in"], return_aux=True, **kwargs)

    assert out.image.shape == inputs["i_in"].shape
    assert out.context.shape == (2, 1, 32, 32)
    assert all(x is not None for x in out.affinity_maps) is has_affinity
    if variant != "E4":
        assert out.exec_tokens is not None if needs_hidden else out.exec_tokens is None
        assert model.alpha_head.use_exec_tokens is False


def test_hidden_alpha_and_semantic_modes_are_supported():
    torch.manual_seed(1)
    model = _small_model(alpha_source="hidden")
    inputs = _inputs(batch_size=3)

    normal = model(
        inputs["i_in"],
        vision_spatial=inputs["vision_spatial"],
        hidden_bank=inputs["hidden_bank"],
        z_exec=inputs["z_exec"],
        return_aux=True,
    )
    zero = model(
        inputs["i_in"],
        vision_spatial=inputs["vision_spatial"],
        hidden_bank=inputs["hidden_bank"],
        z_exec=inputs["z_exec"],
        semantic_mode="zero",
        return_aux=True,
    )
    shuffled = model(
        inputs["i_in"],
        vision_spatial=inputs["vision_spatial"],
        hidden_bank=inputs["hidden_bank"],
        z_exec=inputs["z_exec"],
        semantic_mode="shuffle",
        return_aux=True,
    )

    assert normal.semantic_tokens is not None
    assert torch.allclose(zero.semantic_tokens, torch.zeros_like(zero.semantic_tokens))
    assert torch.allclose(shuffled.semantic_tokens, normal.semantic_tokens.roll(shifts=1, dims=0))
    assert torch.allclose(normal.alpha.sum(dim=1), torch.ones(3), atol=1e-6)


def test_terminal_residual_mode_requires_base_image_and_returns_gate():
    model = _small_model(output_mode="terminal_residual")
    inputs = _inputs()

    with pytest.raises(ValueError, match="i_base is required"):
        model(
            inputs["i_in"],
            vision_spatial=inputs["vision_spatial"],
            hidden_bank=inputs["hidden_bank"],
        )

    out = model(
        inputs["i_in"],
        i_base=inputs["i_base"],
        vision_spatial=inputs["vision_spatial"],
        hidden_bank=inputs["hidden_bank"],
        return_aux=True,
    )
    assert out.image.shape == inputs["i_in"].shape
    assert out.gate is not None
    assert 0.0 < float(out.gate) < 1.0


def test_regularizers_accept_vera_sa_lut_outputs():
    model = _small_model()
    inputs = _inputs()
    out = model(
        inputs["i_in"],
        vision_spatial=inputs["vision_spatial"],
        hidden_bank=inputs["hidden_bank"],
        return_aux=True,
    )

    assert context_tv(out.context).ndim == 0
    assert lut_monotonicity(out.lut).ndim == 0
