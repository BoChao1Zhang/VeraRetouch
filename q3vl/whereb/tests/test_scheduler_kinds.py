"""``make_scheduler``'s EPR-018..023 kinds, and that the old ones did not move.

CPU only.  The first test is the important one: the live arms pass
``kind="cosine"`` with no keyword arguments, and the extension must leave that
curve bit-identical -- a schedule that shifted would break U4 step-matching
against every published board without changing a single line of arm code.
"""

from __future__ import annotations

import math

import pytest
import torch

from q3vl.where.calibrate import SCHEDULER_KINDS, make_scheduler, scale_milestones


def _factors(kind, total=100, ratio=0.03, **kw):
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    sch = make_scheduler(opt, total, ratio, kind, **kw)
    out = []
    for _ in range(total):
        out.append(opt.param_groups[0]["lr"])
        opt.step()
        sch.step()
    return out


def _original(kind, total, ratio):
    """The pre-extension two-line body, transcribed."""
    warmup = max(1, int(round(total * ratio)))
    out = []
    for step in range(total):
        if step < warmup:
            out.append((step + 1) / warmup)
        elif kind != "cosine":
            out.append(1.0)
        else:
            p = (step - warmup) / max(1, total - warmup)
            out.append(0.5 * (1.0 + math.cos(math.pi * min(1.0, p))))
    return out


@pytest.mark.parametrize("kind", ["cosine", "constant", "whatever"])
@pytest.mark.parametrize("total,ratio", [(100, 0.03), (1200, 0.03), (7, 0.5)])
def test_existing_kinds_are_bit_identical(kind, total, ratio):
    got = _factors(kind, total=total, ratio=ratio)
    want = _original(kind, total, ratio)
    assert got == pytest.approx(want, abs=0.0, rel=0.0)


def test_kind_list_is_the_documented_one():
    assert SCHEDULER_KINDS == ("cosine", "constant", "linear", "multistep",
                               "warmup_multistep")


# --- linear (LISA WarmupDecayLR) -------------------------------------------

def test_linear_warms_up_then_decays_to_zero():
    f = _factors("linear", total=100, ratio=0.24)     # warmup 24, as EPR-018
    assert f[0] == pytest.approx(1 / 24)
    assert f[23] == pytest.approx(1.0)
    assert f[24] == pytest.approx(1.0)                # first post-warmup step
    assert f[-1] < 0.02 and f[-1] >= 0.0
    post = f[24:]
    assert all(post[i] >= post[i + 1] - 1e-12 for i in range(len(post) - 1))


def test_linear_never_goes_negative():
    assert min(_factors("linear", total=50, ratio=0.06)) >= 0.0


# --- multistep --------------------------------------------------------------

def test_multistep_gamma_ladder():
    f = _factors("multistep", total=100, ratio=0.03, milestones=[40, 70],
                 gamma=0.1)
    assert f[39] == pytest.approx(1.0)
    assert f[40] == pytest.approx(0.1)
    assert f[69] == pytest.approx(0.1)
    assert f[70] == pytest.approx(0.01)


def test_multistep_values_ladder_is_vitmattes_form():
    """``MultiStepParamScheduler(values=[1.0, 0.1, 0.05])`` at 30% / 90%."""
    f = _factors("multistep", total=100, ratio=0.02, milestones=[30, 90],
                 values=[1.0, 0.1, 0.05])
    assert (f[29], f[30], f[89], f[90]) == pytest.approx((1.0, 0.1, 0.1, 0.05))


def test_multistep_values_length_is_checked():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    with pytest.raises(ValueError, match="len\\(milestones\\) \\+ 1"):
        make_scheduler(opt, 100, 0.03, "multistep", milestones=[30, 90],
                       values=[1.0, 0.1])


def test_multistep_without_milestones_is_refused():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    with pytest.raises(ValueError, match="needs milestones"):
        make_scheduler(opt, 100, 0.03, "multistep")


# --- warmup_multistep (detectron2) -----------------------------------------

def test_warmup_multistep_uses_the_warmup_factor():
    f = _factors("warmup_multistep", total=100, ratio=0.0, warmup_steps=10,
                 warmup_factor=1e-3, milestones=[60, 80], gamma=0.1)
    assert f[0] == pytest.approx(1e-3)                    # WARMUP_FACTOR
    assert f[5] == pytest.approx(1e-3 * 0.5 + 0.5)
    assert f[10] == pytest.approx(1.0)
    assert f[60] == pytest.approx(0.1)
    assert f[80] == pytest.approx(0.01)


def test_warmup_steps_overrides_the_ratio():
    f = _factors("cosine", total=100, ratio=0.5, warmup_steps=4)
    assert f[3] == pytest.approx(1.0)
    assert f[4] == pytest.approx(1.0)      # first post-warmup step, cos(0) = 1
    assert f[5] < 1.0                      # ... and the cosine has started
    assert f[49] < 0.7                     # ratio=0.5 would still be warming up


# --- milestone carry-over ---------------------------------------------------

def test_scale_milestones_matches_the_proposals_arithmetic():
    # EPR-023: 800 / 1067 from CondInst's (60000, 80000) over 90000
    assert scale_milestones((60000 / 90000, 80000 / 90000), 1200) == [800, 1067]
    # EPR-019: SAM's 60k / 86666 over 90k
    assert scale_milestones((60000 / 90000, 86666 / 90000), 1200) == [800, 1156]
    # EPR-020: PointRend cityscapes (40000, 55000) over 65000
    assert scale_milestones((40000 / 65000, 55000 / 65000), 1200) == [738, 1015]
