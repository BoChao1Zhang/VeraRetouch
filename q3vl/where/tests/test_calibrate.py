"""Protocol 4.4 / 10.2 -- the four arms, end to end on mock data."""

from __future__ import annotations

import json
import math

import pytest
import torch

from q3vl.where.calibrate import (
    Calibrator, WhereASample, arm_readouts, make_scheduler, percentiles,
)
from q3vl.where.config import ARMS, CalibConfig, FitConfig, FPRE_DIM, SEM_DIM
from q3vl.where.oracle import FitResult
from q3vl.where.projector import BasisProjector

DT = torch.float32


def _mock_sample(i: int, gh: int = 8, gw: int = 12) -> WhereASample:
    """A soft elliptical region: expressible from geo5 alone, so the oracle fit
    is well posed no matter what the projector currently is."""
    g = torch.Generator().manual_seed(100 + i)
    ys = torch.linspace(-1, 1, gh, dtype=DT)
    xs = torch.linspace(-1, 1, gw, dtype=DT)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    cx = 0.3 * math.cos(i)
    cy = 0.3 * math.sin(i)
    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    mask = torch.sigmoid((0.45 - r2) * 8.0)
    return WhereASample(
        sample_id=f"mock_{i:03d}",
        fpre=torch.randn(gh * gw, FPRE_DIM, generator=g, dtype=DT),
        img_low=torch.rand(3, gh, gw, generator=g, dtype=DT),
        mask_low=mask.reshape(-1),
        grid_h=gh, grid_w=gw,
        meta={"build": "l1", "winner_confidence": "normal", "upscaled": False},
    )


def _cfg(arm: str) -> CalibConfig:
    return CalibConfig(arm=arm, batch_size=2,
                       inner_fit=FitConfig(n_random=1, max_iter=20, seed=0))


def test_arm_table_matches_the_protocol():
    assert ARMS == ("BA-0-Fixed", "BA-1-Band", "BA-2-CBand12", "BA-3-Joint")
    assert arm_readouts("BA-0-Fixed") == ()
    assert arm_readouts("BA-1-Band") == ("band",)
    assert arm_readouts("BA-2-CBand12") == ("cband12",)
    assert arm_readouts("BA-3-Joint") == ("band", "cband12")
    with pytest.raises(ValueError):
        arm_readouts("BA-9")


def test_projector_is_seeded_orthogonal():
    a = BasisProjector(seed=7)
    b = BasisProjector(seed=7)
    c = BasisProjector(seed=8)
    assert torch.allclose(a.weight, b.weight, atol=0), "same seed must give the same B"
    assert not torch.allclose(a.weight, c.weight)
    assert a.weight.shape == (SEM_DIM, FPRE_DIM)
    assert a.orthogonality_error() < 1e-5
    facts = a.facts()
    assert facts["condition_number"] == pytest.approx(1.0, abs=1e-4)
    assert len(facts["digest"]) == 64


def test_ba3_joint_step_closes_the_loop():
    cfg = _cfg("BA-3-Joint")
    cal = Calibrator(cfg, total_steps=4)
    before = cal.projector.weight.detach().clone()
    batch = [_mock_sample(i) for i in range(2)]
    out = cal.step(batch, record_fits=True)

    assert math.isfinite(out["loss"]), out
    assert set(out["loss_per_readout"]) == {"band", "cband12"}
    assert all(math.isfinite(v) for v in out["loss_per_readout"].values())
    assert all(math.isfinite(v) for v in out["fit_loss_per_readout"].values())
    assert out["grad_norm"] is not None and math.isfinite(out["grad_norm"])
    assert not torch.allclose(before, cal.projector.weight), "B must have moved"
    # one fit report row per (sample, readout)
    assert len(cal.fit_report) == 4
    assert {r["readout"] for r in cal.fit_report} == {"band", "cband12"}
    assert all(r["sample_id"].startswith("mock_") for r in cal.fit_report)


@pytest.mark.parametrize("arm", ["BA-1-Band", "BA-2-CBand12"])
def test_single_readout_arms_train_only_their_readout(arm):
    cal = Calibrator(_cfg(arm), total_steps=2)
    before = cal.projector.weight.detach().clone()
    out = cal.step([_mock_sample(0)])
    assert set(out["loss_per_readout"]) == set(arm_readouts(arm))
    assert not torch.allclose(before, cal.projector.weight)


def test_ba0_never_trains_the_projector():
    cal = Calibrator(_cfg("BA-0-Fixed"))
    before = cal.projector.weight.detach().clone()
    out = cal.step([_mock_sample(0)])
    assert cal.optimizer is None
    assert out["loss_per_readout"] == {}
    assert torch.allclose(before, cal.projector.weight, atol=0)
    assert not any(p.requires_grad for p in cal.projector.parameters())


def test_evaluate_reports_the_oracle_ceiling():
    cal = Calibrator(_cfg("BA-3-Joint"))
    report = cal.evaluate([_mock_sample(i) for i in range(3)])
    assert report["n_samples"] == 3
    assert set(report["per_readout"]) == {"band", "cband12"}
    for r, stats in report["per_readout"].items():
        assert 0.0 <= stats["all_ok_low"]["soft_iou_minmax"]["median"] <= 1.0
        assert 0.0 <= stats["fit_success_rate"] <= 1.0
        assert stats["n_ok"] + stats["n_rejected"] == 3
    assert len(report["rows"]) == 6
    assert "phi_diag" in report["rows"][0]
    assert report["projector"]["digest"]
    assert report["objective"] == "soft_iou_minmax"


def test_evaluate_on_ba0_still_uses_both_readouts():
    cal = Calibrator(_cfg("BA-0-Fixed"))
    report = cal.evaluate([_mock_sample(0)])
    assert set(report["per_readout"]) == {"band", "cband12"}


def test_scheduler_warms_up_then_decays():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1.0)
    sched = make_scheduler(opt, total_steps=100, warmup_ratio=0.03, kind="cosine")
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[3] <= 1.0 + 1e-9
    assert lrs[3] == pytest.approx(1.0, abs=1e-9)
    assert lrs[-1] < 0.01
    assert all(lrs[i] >= lrs[i + 1] - 1e-12 for i in range(3, 99))


def test_sample_shape_validation():
    with pytest.raises(ValueError):
        WhereASample("x", torch.zeros(10, FPRE_DIM), torch.zeros(3, 4, 4),
                     torch.zeros(16), 4, 4)
    with pytest.raises(ValueError):
        WhereASample("x", torch.zeros(16, FPRE_DIM), torch.zeros(3, 4, 4),
                     torch.zeros(10), 4, 4)


def test_state_is_checkpointable():
    cal = Calibrator(_cfg("BA-1-Band"), total_steps=1)
    cal.step([_mock_sample(0)])
    st = cal.state()
    assert st["arm"] == "BA-1-Band" and st["step"] == 1
    assert "weight" in st["projector"]
    fresh = BasisProjector()
    fresh.load_state_dict(st["projector"])
    assert fresh.digest() == st["projector_facts"]["digest"]


# --- REVIEW-impl-WhereA B-1 / B-2 / B-3 / B-4 / B-6 / N-11 ------------------

def _mock_hi(i: int, gh: int = 8, gw: int = 12) -> WhereASample:
    """Mock sample carrying the delivery-resolution branch."""
    s = _mock_sample(i, gh, gw)
    H, W = gh * 16, gw * 16
    mask_hi = torch.nn.functional.interpolate(
        s.mask_low.reshape(1, 1, gh, gw), size=(H, W), mode="bilinear", align_corners=False
    )[0, 0]
    guide = torch.rand(1, 1, H, W, generator=torch.Generator().manual_seed(500 + i), dtype=DT)
    return WhereASample(s.sample_id, s.fpre, s.img_low, s.mask_low, gh, gw,
                        meta=s.meta, mask_hi=mask_hi, guide_hi=guide)


class _RejectingCalibrator(Calibrator):
    """Forces every fit of one readout to be rejected, latent and all."""

    def __init__(self, *a, reject: str = "band", **kw):
        super().__init__(*a, **kw)
        self.reject = reject

    def fit_sample(self, sample, phi_dir, readout, **kw):
        fit = super().fit_sample(sample, phi_dir, readout, **kw)
        if readout == self.reject:
            return FitResult(latent=None, readout=readout, loss=float("inf"),
                             status="rejected", reject_reason="all_starts_failed",
                             n_starts=fit.n_starts, objective=fit.objective)
        return fit


def test_rejected_fits_never_reach_the_projector():
    """B-1: a rejected fit -- including all_starts_failed, whose latent is None --
    must contribute no loss and no gradient."""
    cal = _RejectingCalibrator(_cfg("BA-3-Joint"), total_steps=2, reject="band")
    out = cal.step([_mock_sample(0), _mock_sample(1)])
    assert out["n_rejected_fits"] == 2                      # both samples' band fits
    assert out["n_used_per_readout"]["band"] == 0
    assert out["n_used_per_readout"]["cband12"] == 2
    assert math.isnan(out["loss_per_readout"]["band"])
    assert math.isfinite(out["loss"])                       # cband still trains


def test_all_readouts_rejected_leaves_the_projector_untouched():
    """The degenerate case: nothing usable in the batch -> no gradient at all,
    rather than a gradient computed from a fabricated latent."""
    cal = _RejectingCalibrator(_cfg("BA-1-Band"), total_steps=2, reject="band")
    before = cal.projector.weight.detach().clone()
    out = cal.step([_mock_sample(0)])
    assert out["n_rejected_fits"] == 1
    assert out["grad_norm"] is None
    assert torch.allclose(before, cal.projector.weight, atol=0)


def test_rejected_fits_are_written_out_sample_level():
    """B-2: the calibration epoch must leave a sample-level rejection trail,
    not just a per-step integer."""
    cal = _RejectingCalibrator(_cfg("BA-3-Joint"), total_steps=2, reject="cband12")
    out = cal.step([_mock_sample(0), _mock_sample(1)])
    rows = out["fit_rows"]
    assert len(rows) == 2
    for row in rows:
        assert row["sample_id"].startswith("mock_")
        assert row["readout"] == "cband12"
        assert row["status"] == "rejected"
        assert row["reject_reason"] == "all_starts_failed"
        assert row["build"] == "l1" and row["winner_confidence"] == "normal"
        json.dumps(row)
    summary = cal.rejection_summary()
    assert summary["n_fits"] == 4 and summary["n_rejected"] == 2
    assert summary["reject_reasons"]["all_starts_failed"] == 2
    assert summary["reject_rate"] == 0.5


def test_sample_every_records_healthy_fits_too():
    cal = Calibrator(_cfg("BA-1-Band"), total_steps=2)
    out = cal.step([_mock_sample(0)], sample_every=1)
    rows = out["fit_rows"]
    assert len(rows) == 1 and rows[0].get("sampled") is True
    assert rows[0]["status"] == "ok"
    # ...and not on the other steps
    assert cal.step([_mock_sample(1)], sample_every=5)["fit_rows"] == []


def test_rejected_fits_do_not_enter_the_ceiling():
    """B-1: a rejected sample's metrics would drag the oracle ceiling down and
    thereby loosen the protocol 5.6 '>= 85% of oracle' gate."""
    cal = _RejectingCalibrator(_cfg("BA-3-Joint"), reject="band")
    report = cal.evaluate([_mock_sample(i) for i in range(3)])
    band = report["per_readout"]["band"]
    cband = report["per_readout"]["cband12"]
    assert band["n_ok"] == 0 and band["n_rejected"] == 3
    assert band["fit_success_rate"] == 0.0
    assert band["all_ok_low"]["n"] == 0
    assert band["reject_reasons"] == {"all_starts_failed": 3}
    assert cband["n_ok"] == 3 and cband["all_ok_low"]["n"] == 3


def test_evaluate_measures_the_delivered_resolution():
    """B-4: the ceiling has to be measured after the one guided upsample, on the
    real path, with the s-domain reported."""
    cal = Calibrator(_cfg("BA-3-Joint"))
    report = cal.evaluate([_mock_hi(i) for i in range(2)])
    assert report["n_with_hi_res"] == 2
    for r, v in report["per_readout"].items():
        assert v["all_ok_hi"]["n"] == 2, r
        assert 0.0 <= v["all_ok_hi"]["soft_iou_minmax"]["median"] <= 1.0
        dom = v["s_domain"]
        assert dom["clamped"] is True
        assert dom["max_frac_out_of_domain"] is not None
        assert dom["raw_min"] is not None and dom["raw_max"] is not None
    assert report["upsample"]["domain"] == [-3.0, 3.0]


def test_evaluate_without_hi_res_says_so():
    cal = Calibrator(_cfg("BA-1-Band"))
    report = cal.evaluate([_mock_sample(0)])
    assert report["n_with_hi_res"] == 0
    assert report["per_readout"]["band"]["all_ok_hi"]["n"] == 0


def test_headline_stratum_is_normal_only_but_low_still_reported():
    """D1: `low` trains, the headline reports `normal`, and `low` is its own row."""
    samples = []
    for i in range(4):
        s = _mock_sample(i)
        s.meta["winner_confidence"] = "normal" if i < 2 else "low"
        samples.append(s)
    cal = Calibrator(_cfg("BA-1-Band"))
    report = cal.evaluate(samples)
    band = report["per_readout"]["band"]
    assert report["headline_winner_confidence"] == ["normal"]
    assert band["headline_low"]["n"] == 2
    assert band["all_ok_low"]["n"] == 4
    strata = band["by_winner_confidence_low_res"]
    assert strata["normal"]["n"] == 2 and strata["low"]["n"] == 2


def test_sample_rejects_a_half_supplied_high_res_branch():
    s = _mock_sample(0)
    with pytest.raises(ValueError, match="together"):
        WhereASample(s.sample_id, s.fpre, s.img_low, s.mask_low, s.grid_h, s.grid_w,
                     mask_hi=torch.zeros(128, 192))
    with pytest.raises(ValueError, match="together"):
        WhereASample(s.sample_id, s.fpre, s.img_low, s.mask_low, s.grid_h, s.grid_w,
                     guide_hi=torch.zeros(1, 1, 128, 192))


def test_inner_and_outer_objective_are_the_same():
    """B-6: the envelope-theorem argument for the fixed-latent gradient only
    holds when both levels minimise the same functional."""
    from q3vl.where.config import CALIB_OBJECTIVE, FIT_OBJECTIVE
    assert CALIB_OBJECTIVE == FIT_OBJECTIVE == "soft_iou_minmax"
    cfg = CalibConfig()
    assert cfg.objective == cfg.inner_fit.objective


def test_percentile_convention_is_pinned():
    """N-11: protocol 5.6 gates on p10, so the convention cannot be incidental."""
    xs = [float(i) for i in range(1, 101)]
    p = percentiles(xs)
    assert p["p10"] == 10.0 and p["median"] == 50.0 and p["p90"] == 90.0
    assert p["min"] == 1.0 and p["max"] == 100.0 and p["mean"] == 50.5
    assert percentiles([]) == {}
    assert p["n"] == 100
    assert percentiles([7.0]) == {"n": 1, "mean": 7.0, "p10": 7.0, "median": 7.0,
                                  "p90": 7.0, "min": 7.0, "max": 7.0}
    # k=2: nearest-rank makes p10 == median; `n` is what stops that reading as
    # a tight distribution (N-23)
    two = percentiles([1.0, 9.0])
    assert two["n"] == 2 and two["p10"] == two["median"] == 1.0
    # nearest-rank, spelled out: idx = clip(ceil(q*k) - 1, 0, k-1) on the sorted
    # array.  (Nearest-rank p10/p90 are order statistics, not reflections of one
    # another; the point of pinning it is that the 5.6 gate reads the same number
    # every time, not that it satisfies a symmetry.)
    for k in (7, 13, 100):
        ys = [float(i) for i in range(k)]
        q = percentiles(ys)
        assert q["p10"] == ys[max(0, math.ceil(0.10 * k) - 1)]
        assert q["p90"] == ys[min(k - 1, math.ceil(0.90 * k) - 1)]
    # unsorted input must give the same answer as sorted input
    import random
    shuffled = xs[:]
    random.Random(0).shuffle(shuffled)
    assert percentiles(shuffled) == p


def test_scheduler_uses_the_eligible_step_count():
    """B-3: with the *unfiltered* count (75544 vs 42752 eligible) the cosine
    stopped at ~42% of peak and warmup covered 5.3% instead of 3%."""
    p = torch.nn.Parameter(torch.zeros(1))
    n_eligible, batch = 42752, 8
    total = math.ceil(n_eligible / batch)
    opt = torch.optim.AdamW([p], lr=1.0)
    sched = make_scheduler(opt, total_steps=total, warmup_ratio=0.03, kind="cosine")
    lrs = []
    for _ in range(total):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[-1] < 1e-4, "cosine must anneal to ~0 by the last real step"
    warmup = max(1, round(total * 0.03))
    assert abs(warmup / total - 0.03) < 0.001

    # the bug: schedule built for the unfiltered population
    wrong_total = math.ceil(75544 / batch)
    opt2 = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched2 = make_scheduler(opt2, total_steps=wrong_total, warmup_ratio=0.03, kind="cosine")
    for _ in range(total):
        opt2.step()
        sched2.step()
    assert opt2.param_groups[0]["lr"] > 0.3, "reproduces the never-annealing schedule"


def test_calibrator_exposes_the_schedule_it_was_built_with():
    cal = Calibrator(_cfg("BA-1-Band"), total_steps=100)
    assert cal.total_steps == 100
    assert cal.warmup_steps == 3
    st = cal.state()
    assert st["total_steps"] == 100 and st["warmup_steps"] == 3
    assert "rejections" in st


# --- 复审 N-18 / N-20 / N-21 / N-24 ----------------------------------------

def test_mismatched_objectives_are_refused_at_construction():
    """N-18: two module constants that happen to agree is not a guarantee --
    `CalibConfig(objective=...)` is a public argument four scripts construct."""
    bad = CalibConfig(arm="BA-1-Band", objective="mse",
                      inner_fit=FitConfig(objective="soft_iou_minmax"))
    with pytest.raises(ValueError, match="inner/outer objective mismatch"):
        Calibrator(bad)
    bad2 = CalibConfig(arm="BA-1-Band", objective="soft_iou_minmax",
                       inner_fit=FitConfig(objective="mse"))
    with pytest.raises(ValueError, match="inner/outer objective mismatch"):
        Calibrator(bad2)
    # matching pair is fine, whichever functional it is
    Calibrator(CalibConfig(arm="BA-1-Band", objective="mse",
                           inner_fit=FitConfig(objective="mse")))


def test_fit_sample_rejects_an_off_objective_fit_cfg():
    """The other way in: an explicit fit_cfg passed at call time."""
    cal = Calibrator(_cfg("BA-1-Band"))
    s = _mock_sample(0)
    parts = cal.phi_for(s)
    with pytest.raises(ValueError, match="must minimise the same"):
        cal.fit_sample(s, parts.phi_dir, "band",
                       fit_cfg=FitConfig(objective="mse", n_random=1, max_iter=5))


def test_freeze_projector_makes_b_untouchable():
    """N-24: the oracle-latent job's 'B is never touched' must be structural."""
    cal = Calibrator(_cfg("BA-3-Joint"), total_steps=2)
    assert cal.optimizer is not None and cal.trains_projector
    cal.freeze_projector()
    assert cal.optimizer is None and cal.scheduler is None
    assert not cal.trains_projector
    assert not any(p.requires_grad for p in cal.projector.parameters())
    before = cal.projector.digest()
    out = cal.step([_mock_sample(0)])
    assert out["grad_norm"] is None
    assert cal.projector.digest() == before


def test_domain_report_carries_per_sample_percentiles():
    """N-20: the pre-registered gate is on the *median* over samples, so the
    distribution has to be there, not just a max."""
    cal = Calibrator(_cfg("BA-1-Band"))
    report = cal.evaluate([_mock_hi(i) for i in range(3)])
    dom = report["per_readout"]["band"]["s_domain"]
    assert set(dom["frac_out_of_domain"]) >= {"median", "p90", "mean", "max"}
    assert dom["frac_out_of_domain"]["n"] == 3
    assert dom["frac_out_of_domain"]["max"] == pytest.approx(dom["max_frac_out_of_domain"])


def test_headline_summaries_carry_their_sample_count():
    """N-23: p10==median at k=2 reads as 'tight' unless n is right there."""
    cal = Calibrator(_cfg("BA-1-Band"))
    samples = []
    for i in range(3):
        s = _mock_sample(i)
        s.meta["winner_confidence"] = "normal" if i == 0 else "low"
        samples.append(s)
    band = cal.evaluate(samples)["per_readout"]["band"]
    assert band["headline_low"]["n"] == 1
    assert band["all_ok_low"]["n"] == 3
