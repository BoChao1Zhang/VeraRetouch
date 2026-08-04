"""The analytic path: phi_dir parity with Where-A, s, the readout, one upsample."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from q3vl.where.basis import Latent, s_low
from q3vl.where.phi import build_phi_dir
from q3vl.where.readout import apply_readout, param_shapes
from q3vl.where.upsample import ChannelOrderError, guided_upsample
from q3vl.whereb.fields import (
    FrozenBasis,
    load_basis,
    oracle_fields,
    phi_dir_fast,
    predict_fields,
    s_from_params,
)


@pytest.mark.parametrize("shape", [(6, 8), (5, 5), (4, 12)])
def test_phi_fast_matches_where_a_bit_for_bit(shape):
    """The training loop's phi may not drift from the calibrated one."""
    gh, gw = shape
    g = torch.Generator().manual_seed(0)
    sem = torch.randn(gh * gw, 64, generator=g, dtype=torch.float64)
    img = torch.rand(3, gh, gw, generator=g, dtype=torch.float64)
    ref = build_phi_dir(sem, img, gh, gw).phi_dir
    got = phi_dir_fast(sem, img, gh, gw)
    assert torch.equal(ref, got)


def test_phi_fast_is_differentiable_into_the_semantic_block():
    sem = torch.randn(24, 64, dtype=torch.float64, requires_grad=True)
    img = torch.rand(3, 4, 6, dtype=torch.float64)
    phi_dir_fast(sem, img, 4, 6).sum().backward()
    assert sem.grad is not None and torch.isfinite(sem.grad).all()


def test_phi_fast_rejects_a_shape_mismatch():
    with pytest.raises(ValueError):
        phi_dir_fast(torch.randn(10, 64), torch.rand(3, 4, 6), 4, 6)
    with pytest.raises(ValueError):
        phi_dir_fast(torch.randn(24, 64), torch.rand(3, 5, 6), 4, 6)


# --- the frozen basis -------------------------------------------------------

def test_frozen_basis_is_a_buffer_not_a_parameter():
    b = FrozenBasis(torch.randn(64, 1024))
    assert list(b.parameters()) == []
    assert "weight" in dict(b.named_buffers())
    out = b(torch.randn(7, 1024))
    assert out.shape == (7, 64)


def test_frozen_basis_shape_is_enforced():
    with pytest.raises(ValueError):
        FrozenBasis(torch.randn(32, 1024))


def test_load_basis_checks_the_published_digest(tmp_path):
    d = tmp_path / "BA-3-Joint"
    d.mkdir(parents=True)
    w = np.random.RandomState(0).randn(64, 1024).astype(np.float32)
    import io

    buf = io.BytesIO()
    np.save(buf, w, allow_pickle=False)
    raw = buf.getvalue()
    (d / "B.npy").write_bytes(raw)
    (d / "basis.json").write_text(json.dumps(
        {"sha256": hashlib.sha256(raw).hexdigest(), "arm": "BA-3-Joint"}))
    b = load_basis("BA-3-Joint", tmp_path)
    assert b.facts()["shape"] == [64, 1024]

    (d / "basis.json").write_text(json.dumps({"sha256": "0" * 64}))
    with pytest.raises(RuntimeError, match="sha256"):
        load_basis("BA-3-Joint", tmp_path)


def test_load_basis_refuses_to_invent_an_uncalibrated_basis(tmp_path):
    with pytest.raises(FileNotFoundError, match="calibrated"):
        load_basis("BA-3-Joint", tmp_path)


# --- s and the mask ---------------------------------------------------------

def test_s_formula_matches_the_protocol_and_is_bounded():
    phi = torch.randn(40, 71, dtype=torch.float64)
    w_dir = torch.nn.functional.normalize(torch.randn(71, dtype=torch.float64), dim=0)
    w0 = torch.tensor(0.4, dtype=torch.float64)
    alpha = torch.tensor(2.0, dtype=torch.float64)
    want = 3.0 * torch.tanh((w0 + alpha * (phi @ w_dir)) / 3.0)
    got = s_from_params(phi, w0, alpha, w_dir)
    assert torch.allclose(got, want)
    assert float(got.abs().max()) < 3.0


def test_predict_fields_agrees_with_the_where_a_latent_path():
    torch.manual_seed(0)
    gh, gw = 5, 7
    phi = torch.randn(gh * gw, 71, dtype=torch.float64)
    rho = {k: (torch.randn((), dtype=torch.float64) if s == ()
               else torch.randn(12, dtype=torch.float64))
           for k, s in param_shapes("cband12").items()}
    lat = Latent("cband12", torch.tensor(0.1, dtype=torch.float64),
                 torch.tensor(0.5, dtype=torch.float64),
                 torch.randn(71, dtype=torch.float64), rho)
    params = {"w0": lat.w0, "w_raw": lat.w_raw, "alpha_raw": lat.alpha_raw, **rho}
    got = predict_fields(phi, params, "cband12", gh, gw)
    assert torch.allclose(got["s_low"], s_low(phi, lat))
    assert torch.allclose(got["m_low"], apply_readout("cband12", s_low(phi, lat), rho))


def test_guided_upsample_is_applied_once_to_the_scalar_only():
    torch.manual_seed(0)
    gh, gw = 4, 6
    phi = torch.randn(gh * gw, 71)
    rho = {k: (torch.randn(()) if s == () else torch.randn(12))
           for k, s in param_shapes("band").items()}
    params = {"w0": torch.tensor(0.0), "w_raw": torch.randn(71),
              "alpha_raw": torch.tensor(0.5), **rho}
    guide = torch.rand(1, 1, gh * 8, gw * 8)
    out = predict_fields(phi, params, "band", gh, gw, guide_hi=guide)
    assert out["s_hi"].shape == (1, 1, gh * 8, gw * 8)
    assert out["m_hi"].shape == out["s_hi"].shape
    with pytest.raises(ChannelOrderError):
        guided_upsample(torch.randn(1, 64, gh, gw), guide)


def test_predict_fields_backpropagates_to_every_predicted_parameter():
    gh, gw = 4, 5
    phi = torch.randn(gh * gw, 71)
    params = {
        "w0": torch.tensor(0.1, requires_grad=True),
        "w_raw": torch.randn(71, requires_grad=True),
        "alpha_raw": torch.tensor(0.3, requires_grad=True),
        "mu": torch.tensor(0.0, requires_grad=True),
        "h_raw": torch.tensor(0.0, requires_grad=True),
        "k_raw": torch.tensor(0.0, requires_grad=True),
        "pi_raw": torch.tensor(2.0, requires_grad=True),
    }
    out = predict_fields(phi, params, "band", gh, gw,
                         guide_hi=torch.rand(1, 1, gh * 4, gw * 4))
    out["m_hi"].sum().backward()
    for k, v in params.items():
        assert v.grad is not None, k
        assert torch.isfinite(v.grad).all(), k


def test_oracle_fields_recompute_s_star_and_r_star_from_the_latent():
    torch.manual_seed(0)
    phi = torch.randn(30, 71)
    rho = {k: (torch.randn(()) if s == () else torch.randn(12))
           for k, s in param_shapes("band").items()}
    lat = Latent("band", torch.tensor(0.2), torch.tensor(0.4), torch.randn(71), rho)
    z = torch.linspace(-3, 3, 257)
    o = oracle_fields(phi, lat, "band", z)
    assert torch.allclose(o["s_star"], s_low(phi, lat))
    assert torch.allclose(o["r_star"], apply_readout("band", z, rho))
    assert abs(float(o["w_dir_star"].norm()) - 1.0) < 1e-6
