"""Protocol 4.4 / 10.2 -- multi-start L-BFGS oracle fit and its fit report."""

from __future__ import annotations

import pytest
import torch

from q3vl.where.basis import Latent, canonicalize, is_canonical, mask_from_latent
from q3vl.where.config import FitConfig, PHI_DIR_DIM, SEM_DIM, UpsampleConfig
from q3vl.where.oracle import (
    OBJECTIVES, evaluate_latent, fit_latent, mask_metrics, objective_value,
    soft_iou_minmax,
)
from q3vl.where.phi import build_phi_dir

DT = torch.float64


def _phi(gh=16, gw=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    sem = torch.randn(gh * gw, SEM_DIM, generator=g, dtype=DT)
    img = torch.rand(3, gh, gw, generator=g, dtype=DT)
    return build_phi_dir(sem, img, gh, gw).phi_dir, gh, gw


def _planted(readout: str, seed=0) -> Latent:
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(PHI_DIR_DIM, generator=g, dtype=DT)
    if readout == "band":
        rho = {"mu": torch.tensor(0.4, dtype=DT), "h_raw": torch.tensor(0.0, dtype=DT),
               "k_raw": torch.tensor(1.0, dtype=DT), "pi_raw": torch.tensor(3.0, dtype=DT)}
    else:
        rho = {"sig_raw": torch.zeros(12, dtype=DT),
               "o_raw": torch.zeros(12, dtype=DT),
               "c_raw": torch.tensor([-3.0] * 4 + [3.0] * 4 + [-3.0] * 4, dtype=DT)}
    return canonicalize(Latent(readout, torch.tensor(0.2, dtype=DT),
                               torch.tensor(0.5, dtype=DT), w, rho))


def test_objectives_are_sane():
    a = torch.tensor([0.0, 1.0, 0.5], dtype=DT)
    assert float(objective_value("soft_iou_minmax", a, a)) == pytest.approx(0.0, abs=1e-5)
    assert float(objective_value("mse", a, a)) == 0.0
    assert float(soft_iou_minmax(a, a)) == pytest.approx(1.0, abs=1e-5)
    with pytest.raises(ValueError):
        objective_value("nope", a, a)
    assert set(OBJECTIVES) == {"soft_iou_minmax", "mse", "bce"}


@pytest.mark.parametrize("readout", ["band", "cband12"])
def test_fit_recovers_a_planted_mask(readout):
    phi, gh, gw = _phi(seed=1)
    lat = _planted(readout, seed=1)
    target, _ = mask_from_latent(phi, lat)
    res = fit_latent(phi, target, readout, FitConfig(seed=0))
    assert res.status == "ok", res.reject_reason
    assert res.metrics["soft_iou_minmax"] > 0.95, res.metrics
    assert res.loss < 0.05


@pytest.mark.parametrize("readout", ["band", "cband12"])
def test_fit_returns_a_canonical_latent(readout):
    phi, _, _ = _phi(seed=2)
    lat = _planted(readout, seed=2)
    target, _ = mask_from_latent(phi, lat)
    res = fit_latent(phi, target, readout, FitConfig(seed=1))
    assert is_canonical(res.latent), "the fit must apply the protocol sign rule"
    assert abs(float(res.latent.w_dir.norm()) - 1.0) < 1e-12
    assert float(res.latent.alpha) > 0


def test_fit_is_deterministic_in_the_seed():
    phi, _, _ = _phi(seed=3)
    target, _ = mask_from_latent(phi, _planted("band", seed=3))
    a = fit_latent(phi, target, "band", FitConfig(seed=5))
    b = fit_latent(phi, target, "band", FitConfig(seed=5))
    assert a.loss == b.loss
    assert torch.allclose(a.latent.w_raw, b.latent.w_raw, atol=0)


def test_unfittable_target_is_rejected_not_zeroed():
    """Protocol 10.2: a failed sample enters an explicit rejection report and is
    never silently replaced by a zero vector."""
    # a realistic 32x48 F_pre grid: 1536 points, 71 direction dims, so a random
    # target genuinely cannot be represented
    phi, _, _ = _phi(gh=32, gw=48, seed=4)
    g = torch.Generator().manual_seed(9)
    target = (torch.rand(phi.shape[0], generator=g, dtype=DT) > 0.5).to(DT)  # pure noise
    res = fit_latent(phi, target, "band",
                     FitConfig(seed=0, n_random=2, max_iter=60, reject_loss=0.35))
    assert res.status == "rejected"
    assert res.reject_reason == "loss_above_threshold"
    # the latent is still the best fit found, not a zero vector
    assert float(res.latent.w_raw.abs().max()) > 0
    d = res.to_dict()
    assert d["status"] == "rejected" and d["reject_reason"] == "loss_above_threshold"
    assert len(d["start_losses"]) == res.n_starts


def test_constant_mask_is_flagged():
    phi, _, _ = _phi(gh=8, gw=8, seed=5)
    target = torch.full((phi.shape[0],), 0.5, dtype=DT)
    res = fit_latent(phi, target, "band", FitConfig(seed=0, n_random=2, max_iter=40))
    assert res.status in ("ok", "rejected")
    assert res.metrics["pred_std"] < 0.2


def test_multiple_starts_are_actually_tried():
    phi, _, _ = _phi(gh=8, gw=8, seed=6)
    target, _ = mask_from_latent(phi, _planted("band", seed=6))
    res = fit_latent(phi, target, "band", FitConfig(seed=0, n_random=3))
    # lsq + radial + 3 random + small-alpha = 6 seeds, x2 polarities for band
    assert res.n_starts == 12
    assert len(res.start_losses) == res.n_starts
    assert res.best_start >= 0
    res_c = fit_latent(phi, target, "cband12", FitConfig(seed=0, n_random=3))
    assert res_c.n_starts == 6


def test_fit_report_serialises():
    phi, _, _ = _phi(gh=8, gw=8, seed=7)
    target, _ = mask_from_latent(phi, _planted("cband12", seed=7))
    res = fit_latent(phi, target, "cband12", FitConfig(seed=0, n_random=1, max_iter=30))
    d = res.to_dict()
    assert d["latent"]["readout"] == "cband12"
    assert len(d["latent"]["rho"]["sigma"]) == 12
    assert d["objective"] == "soft_iou_minmax"
    back = Latent.from_dict(d["latent"])
    assert torch.allclose(back.w_raw, res.latent.w_raw, atol=1e-12)


def test_evaluate_latent_reports_low_and_high_res():
    phi, gh, gw = _phi(gh=8, gw=12, seed=8)
    lat = _planted("band", seed=8)
    target_low, _ = mask_from_latent(phi, lat)
    H, W = gh * 16, gw * 16
    g = torch.Generator().manual_seed(11)
    guide = torch.rand(1, 1, H, W, generator=g, dtype=DT)
    target_hi = torch.rand(H, W, generator=g, dtype=DT)
    out = evaluate_latent(phi, lat, target_low, gh, gw, guide, target_hi,
                          UpsampleConfig())
    assert "low" in out and "hi" in out
    assert out["low"]["soft_iou_minmax"] > 0.99
    assert out["readout_bounds"]["k_in_bounds"]
    assert abs(out["s_low_range"][0]) <= 3.0 and abs(out["s_low_range"][1]) <= 3.0


def test_mask_metrics_keys():
    a = torch.rand(50, dtype=DT)
    b = torch.rand(50, dtype=DT)
    m = mask_metrics(a, b)
    for k in ("soft_iou_minmax", "soft_iou_prod", "mae", "mse", "pred_mean", "pred_std"):
        assert k in m


def test_bad_shapes_are_rejected():
    phi, _, _ = _phi(gh=4, gw=4, seed=9)
    with pytest.raises(ValueError):
        fit_latent(phi[:, :10], torch.zeros(16, dtype=DT), "band")
    with pytest.raises(ValueError):
        fit_latent(phi, torch.zeros(15, dtype=DT), "band")
    with pytest.raises(ValueError):
        fit_latent(phi, torch.zeros(16, dtype=DT), "nope")


# --- REVIEW-impl-WhereA B-1: no fabricated latent ---------------------------

def test_all_starts_failed_returns_no_latent_at_all():
    """Protocol 10.2 forbids substituting a zero vector.  A `None` makes any
    consumer that skips the status check crash at once instead of quietly
    training B on the constant mask a zero latent produces."""
    phi, _, _ = _phi(gh=8, gw=8, seed=20)
    phi = phi.clone()
    phi[0, 0] = float("nan")                    # every start's loss is NaN
    target = torch.rand(phi.shape[0], dtype=DT)
    res = fit_latent(phi, target, "band", FitConfig(seed=0, n_random=1, max_iter=10))
    assert res.status == "rejected"
    assert res.reject_reason == "all_starts_failed"
    assert res.latent is None, "a fabricated latent is exactly what 10.2 forbids"
    assert res.usable is False
    d = res.to_dict()
    assert d["latent"] is None and d["usable"] is False


def test_usable_is_the_single_gate():
    phi, _, _ = _phi(seed=21)
    target, _ = mask_from_latent(phi, _planted("band", seed=21))
    ok = fit_latent(phi, target, "band", FitConfig(seed=0, n_random=1, max_iter=40))
    assert ok.usable is (ok.status == "ok" and ok.latent is not None)
    assert ok.usable is True
    rej = fit_latent(phi, (torch.rand(phi.shape[0], dtype=DT) > .5).to(DT), "band",
                     FitConfig(seed=0, n_random=1, max_iter=40, reject_loss=0.0))
    assert rej.status == "rejected" and rej.usable is False
    # the best-effort latent is still there for forensics, just not usable
    assert rej.latent is not None


def test_rejection_row_is_compact_and_carries_the_reason():
    phi, _, _ = _phi(gh=8, gw=8, seed=22)
    res = fit_latent(phi, (torch.rand(phi.shape[0], dtype=DT) > .5).to(DT), "band",
                     FitConfig(seed=0, n_random=1, max_iter=20, reject_loss=0.0))
    row = res.rejection_row(sample_id="sft_x", step=7, build="l3")
    assert row["sample_id"] == "sft_x" and row["step"] == 7 and row["build"] == "l3"
    assert row["status"] == "rejected" and row["reject_reason"]
    assert "start_losses" not in row              # compact: no per-start dump
    import json
    json.dumps(row)                               # must be JSONL-serialisable


def test_evaluate_latent_reports_the_s_domain():
    phi, gh, gw = _phi(gh=8, gw=12, seed=23)
    lat = _planted("band", seed=23)
    target_low, _ = mask_from_latent(phi, lat)
    H, W = gh * 16, gw * 16
    g = torch.Generator().manual_seed(24)
    guide = torch.rand(1, 1, H, W, generator=g, dtype=DT)
    out = evaluate_latent(phi, lat, target_low, gh, gw, guide,
                          torch.rand(H, W, generator=g, dtype=DT), UpsampleConfig())
    assert "s_domain" in out
    assert out["s_domain"]["domain"] == [-3.0, 3.0]
    assert "hi_minus_low_soft_iou" in out
    assert abs(out["s_hi_range"][0]) <= 3.0 + 1e-12
    assert abs(out["s_hi_range"][1]) <= 3.0 + 1e-12
