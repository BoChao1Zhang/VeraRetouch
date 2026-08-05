"""Protocol 7.2 -- aligned pooling: exact, mask-aware, NaN-free, differentiable."""

from __future__ import annotations

import torch

from q3vl.what.config import LutConfig, N_SLOTS, POOL_FEATURE_DIM, V_PROJ_DIM
from q3vl.what.gaussians import anchor_points, decode_geometry, gaussian_log_density
from q3vl.what.pooling import (
    LOG_MASS_CLAMP,
    LOG_MASS_SCALE,
    VisionProjector,
    aligned_pool,
    global_visual_pool,
    roi_bg_pool,
)

CFG = LutConfig()
P = 40


def geometry(batch: int = 2, sigma_raw: float = 0.0) -> dict[str, torch.Tensor]:
    z = torch.zeros(batch, N_SLOTS, 9)
    z[..., 3:6] = sigma_raw
    return decode_geometry(z, anchor_points(), CFG)


def test_matches_the_naive_formula_when_it_is_well_conditioned():
    """The log-domain implementation is the literal
    ``sum a_i x / sum a_i`` -- checked against the literal form at a sigma where
    the literal form does not underflow."""
    torch.manual_seed(0)
    g = geometry(1, sigma_raw=2.0)                      # wide, well conditioned
    rgb = torch.rand(1, P, 3)
    v_feat = torch.randn(1, P, V_PROJ_DIM)
    m = torch.rand(1, P).clamp(0.1, 1.0)
    v, _ = aligned_pool(g, rgb, v_feat, m_pred=m)

    ld = gaussian_log_density(rgb, g["mu"], g["sigma"], g["off"])
    a = torch.exp(ld) * m.unsqueeze(1)                   # (1, N, P)
    feat = torch.cat([rgb, torch.zeros(1, P, 0)], -1)
    naive = torch.einsum("bnp,bpf->bnf", a, feat) / a.sum(-1, keepdim=True)
    assert torch.allclose(v[..., :3], naive, atol=1e-4)
    # log(sum a_i), scaled the way the feature carries it
    want = torch.log(a.sum(-1)).clamp(-LOG_MASS_CLAMP, LOG_MASS_CLAMP) / LOG_MASS_SCALE
    assert torch.allclose(v[..., -2], want, atol=1e-4)


def test_feature_width_and_valid_bit():
    g = geometry(2)
    v, stats = aligned_pool(g, torch.rand(2, P, 3), torch.randn(2, P, V_PROJ_DIM))
    assert v.shape == (2, N_SLOTS, POOL_FEATURE_DIM)
    assert POOL_FEATURE_DIM == 3 + 3 + V_PROJ_DIM + 2
    assert set(v[..., -1].unique().tolist()) <= {0.0, 1.0}
    assert "invalid_fraction" in stats


def test_no_valid_pixel_falls_back_to_the_masked_global_pool_without_nan():
    """A slot whose Gaussian sits where the image has no pixels at all: the
    protocol says fall back to the masked global pool, not produce NaN."""
    g = geometry(1, sigma_raw=-8.0)                      # sigma at its floor
    rgb = torch.full((1, P, 3), 0.98)                    # all pixels in one corner
    v_feat = torch.randn(1, P, V_PROJ_DIM)
    m = torch.rand(1, P).clamp(0.1, 1.0)
    v, stats = aligned_pool(g, rgb, v_feat, m_pred=m)
    assert torch.isfinite(v).all()
    assert stats["n_invalid_slots"] > 0
    invalid = v[..., -1] == 0.0
    fallback = torch.einsum("bp,bpf->bf", m / m.sum(-1, keepdim=True), rgb)
    assert torch.allclose(v[..., :3][invalid], fallback.expand(N_SLOTS, 3)[invalid[0]],
                          atol=1e-4)


def test_all_padding_masked_out_does_not_nan():
    g = geometry(1)
    valid = torch.zeros(1, P, dtype=torch.bool)
    v, stats = aligned_pool(g, torch.rand(1, P, 3), torch.randn(1, P, V_PROJ_DIM),
                            valid=valid)
    assert torch.isfinite(v).all()
    assert stats["invalid_fraction"] == 1.0


def test_mask_conditioning_changes_the_result_and_is_recorded():
    torch.manual_seed(1)
    g = geometry(1, sigma_raw=1.0)
    rgb, vf = torch.rand(1, P, 3), torch.randn(1, P, V_PROJ_DIM)
    m = torch.rand(1, P).clamp(0.01, 1.0)
    a, sa = aligned_pool(g, rgb, vf, m_pred=None)
    b, sb = aligned_pool(g, rgb, vf, m_pred=m)
    assert not torch.allclose(a, b)
    assert sa["mask_conditioned"] is False and sb["mask_conditioned"] is True


def test_gradient_reaches_the_geometry():
    """Protocol 7.4: the gradient must pass through the pooling back to mu/Sigma."""
    z = torch.zeros(1, N_SLOTS, 9, requires_grad=True)
    g = decode_geometry(z, anchor_points(), CFG)
    v, _ = aligned_pool(g, torch.rand(1, P, 3), torch.randn(1, P, V_PROJ_DIM))
    v.sum().backward()
    assert torch.isfinite(z.grad).all()
    assert float(z.grad[..., 0:3].abs().sum()) > 0.0      # mu
    assert float(z.grad[..., 3:6].abs().sum()) > 0.0      # Cholesky diagonal


def test_roi_and_bg_pools_are_the_protocol_6_formula():
    torch.manual_seed(2)
    f = torch.randn(2, P, 1024)
    m = torch.rand(2, P)
    roi, bg = roi_bg_pool(f, m)
    want_roi = (f * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True)
    want_bg = (f * (1 - m).unsqueeze(-1)).sum(1) / (1 - m).sum(1, keepdim=True)
    assert torch.allclose(roi, want_roi, atol=1e-4)
    assert torch.allclose(bg, want_bg, atol=1e-4)


def test_global_visual_pool_respects_padding():
    f = torch.randn(1, 4, 1024)
    valid = torch.tensor([[True, True, False, False]])
    assert torch.allclose(global_visual_pool(f, valid), f[:, :2].mean(1), atol=1e-4)


def test_vision_projector_shape():
    v = VisionProjector()
    assert v(torch.randn(2, P, 1024)).shape == (2, P, V_PROJ_DIM)
