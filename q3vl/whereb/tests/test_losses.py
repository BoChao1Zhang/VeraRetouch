"""Protocol 5.5 checked against hand computation, symbol by symbol."""

from __future__ import annotations

import math

import pytest
import torch

from q3vl.where.readout import apply_readout, param_shapes
from q3vl.whereb.config import MASK_BCE_W, MASK_BF1_W, MASK_IOU_W
from q3vl.whereb.losses import (
    aggregate,
    balanced_bce,
    boundary_f1_loss,
    boundary_map,
    curve_grid,
    loss_curve,
    loss_dir,
    loss_s,
    mask_loss,
    sample_loss,
    schedule_weights,
    soft_iou,
)


# --- soft IoU ---------------------------------------------------------------

def test_soft_iou_minmax_matches_hand_computation():
    m = torch.tensor([0.8, 0.2, 1.0, 0.0])
    t = torch.tensor([1.0, 0.0, 0.5, 0.4])
    want = (0.8 + 0.0 + 0.5 + 0.0) / (1.0 + 0.2 + 1.0 + 0.4)
    assert abs(float(soft_iou(m, t)) - want) < 1e-5


def test_soft_iou_is_one_for_a_perfect_match_and_small_for_a_miss():
    t = torch.tensor([1.0, 1.0, 0.0, 0.0])
    assert abs(float(soft_iou(t, t)) - 1.0) < 1e-5
    assert float(soft_iou(1 - t, t)) < 1e-4


# --- balanced BCE -----------------------------------------------------------

def test_balanced_bce_matches_hand_computation():
    m = torch.tensor([0.9, 0.1])
    t = torch.tensor([1.0, 0.0])
    pos = 0.5
    w_pos, w_neg = 0.5 / (pos + 1e-6), 0.5 / (1 - pos + 1e-6)
    want = -(w_pos * 1.0 * math.log(0.9) + w_neg * 1.0 * math.log(0.9)) / 2
    assert abs(float(balanced_bce(m, t)) - want) < 1e-4


def test_balanced_bce_rewards_a_tiny_region_over_predicting_zero():
    t = torch.zeros(1000)
    t[:20] = 1.0
    all_zero = torch.full((1000,), 1e-3)
    correct = t.clone().clamp(1e-3, 1 - 1e-3)
    assert float(balanced_bce(correct, t)) < float(balanced_bce(all_zero, t))


def test_balanced_bce_is_finite_for_an_all_ones_global_mask():
    t = torch.ones(64)
    v = float(balanced_bce(torch.full((64,), 0.9), t))
    assert math.isfinite(v) and v > 0


# --- boundary F1 ------------------------------------------------------------

def test_boundary_map_is_the_inner_edge_of_the_region():
    y = torch.zeros(1, 1, 7, 7)
    y[..., 2:5, 2:5] = 1.0
    b = boundary_map(y, 3)
    assert float(b[0, 0, 3, 3]) == 0.0                # interior is not boundary
    assert float(b[0, 0, 2, 2]) == 1.0                # corner is
    assert float(b[0, 0, 0, 0]) == 0.0                # background is not


def test_boundary_f1_loss_is_zero_for_an_exact_match():
    y = torch.zeros(1, 1, 16, 16)
    y[..., 4:12, 4:12] = 1.0
    assert float(boundary_f1_loss(y, y)) < 1e-4


def test_boundary_f1_loss_grows_with_displacement_beyond_the_tolerance():
    a = torch.zeros(1, 1, 48, 48)
    a[..., 8:24, 8:24] = 1.0
    near = torch.zeros_like(a)
    near[..., 14:30, 14:30] = 1.0          # 6 px: outside the 3 px tolerance
    far = torch.zeros_like(a)
    far[..., 26:42, 26:42] = 1.0           # 18 px: no overlap at all
    assert float(boundary_f1_loss(a, a)) < float(boundary_f1_loss(near, a))
    assert float(boundary_f1_loss(near, a)) < float(boundary_f1_loss(far, a))
    assert float(boundary_f1_loss(far, a)) > 0.95


def test_boundary_f1_tolerance_forgives_a_shift_inside_3px():
    a = torch.zeros(1, 1, 40, 40)
    a[..., 10:30, 10:30] = 1.0
    shift2 = torch.zeros_like(a)
    shift2[..., 12:32, 12:32] = 1.0
    assert float(boundary_f1_loss(shift2, a, tol_px=3)) < \
        float(boundary_f1_loss(shift2, a, tol_px=0))


def test_boundary_f1_loss_is_zero_when_neither_side_has_a_boundary():
    ones = torch.ones(1, 1, 12, 12)
    assert float(boundary_f1_loss(ones, ones)) == 0.0


def test_boundary_f1_loss_punishes_an_invented_boundary_on_a_global_mask():
    ones = torch.ones(1, 1, 24, 24)
    pred = torch.ones_like(ones)
    pred[..., 8:16, 8:16] = 0.0
    assert float(boundary_f1_loss(pred, ones)) > 0.9


# --- L_mask -----------------------------------------------------------------

def test_mask_loss_is_the_protocol_weighted_sum():
    torch.manual_seed(0)
    m = torch.rand(1, 1, 24, 24)
    t = (torch.rand(1, 1, 24, 24) > 0.5).float()
    d = mask_loss(m, t)
    want = (MASK_IOU_W * (1 - float(d["soft_iou"]))
            + MASK_BCE_W * float(d["bce"])
            + MASK_BF1_W * float(d["bf1_loss"]))
    assert abs(float(d["L_mask"]) - want) < 1e-5


def test_mask_loss_backpropagates():
    m = torch.rand(1, 1, 16, 16, requires_grad=True)
    t = (torch.rand(1, 1, 16, 16) > 0.5).float()
    mask_loss(m, t)["L_mask"].backward()
    assert m.grad is not None and torch.isfinite(m.grad).all()


# --- auxiliaries ------------------------------------------------------------

def test_loss_s_is_huber_on_the_scaled_field():
    s_pred = torch.tensor([0.0, 1.5, -3.0])
    s_star = torch.tensor([0.0, 0.0, 3.0])
    want = torch.nn.functional.huber_loss(s_pred / 3, s_star / 3, delta=1.0)
    assert abs(float(loss_s(s_pred, s_star)) - float(want)) < 1e-6
    assert float(loss_s(s_star, s_star)) == 0.0


def test_curve_grid_is_linspace_minus3_3_257():
    z = curve_grid()
    assert z.numel() == 257
    assert float(z[0]) == -3.0 and float(z[-1]) == 3.0
    assert abs(float(z[1] - z[0]) - 6.0 / 256) < 1e-6


@pytest.mark.parametrize("readout", ["band", "cband12"])
def test_loss_curve_is_the_mean_absolute_readout_difference(readout):
    torch.manual_seed(1)
    rho_a = {k: (torch.randn(()) if s == () else torch.randn(12))
             for k, s in param_shapes(readout).items()}
    rho_b = {k: (torch.randn(()) if s == () else torch.randn(12))
             for k, s in param_shapes(readout).items()}
    z = curve_grid()
    want = (apply_readout(readout, z, rho_a) - apply_readout(readout, z, rho_b)).abs().mean()
    got = loss_curve(readout, rho_a, apply_readout(readout, z, rho_b))
    assert abs(float(got) - float(want)) < 1e-6
    assert float(loss_curve(readout, rho_a, apply_readout(readout, z, rho_a))) < 1e-7


def test_loss_dir_is_one_minus_cosine():
    a = torch.tensor([1.0, 0.0, 0.0])
    assert abs(float(loss_dir(a, a))) < 1e-6
    assert abs(float(loss_dir(a, -a)) - 2.0) < 1e-6
    b = torch.tensor([0.0, 1.0, 0.0])
    assert abs(float(loss_dir(a, b)) - 1.0) < 1e-6


# --- schedule ---------------------------------------------------------------

def test_schedule_switches_at_exactly_30_percent():
    total = 1000
    assert schedule_weights(0, total)["stage"] == 1
    assert schedule_weights(299, total)["stage"] == 1
    assert schedule_weights(300, total)["stage"] == 2
    assert schedule_weights(999, total)["stage"] == 2
    a = schedule_weights(0, total)
    b = schedule_weights(999, total)
    assert (a["s"], a["curve"], a["dir"]) == (1.00, 1.00, 0.10)
    assert (b["s"], b["curve"], b["dir"]) == (0.25, 0.25, 0.05)
    assert a["mask"] == b["mask"] == 1.00


def test_schedule_boundary_rounds_and_is_reported():
    w = schedule_weights(0, 7)
    assert w["boundary_step"] == 2                      # round(0.3 * 7) = 2
    assert schedule_weights(1, 7)["stage"] == 1
    assert schedule_weights(2, 7)["stage"] == 2


# --- assembly ---------------------------------------------------------------

def test_sample_loss_skips_the_auxiliaries_without_an_oracle():
    torch.manual_seed(0)
    fields = {"m_hi": torch.rand(1, 1, 12, 12), "s_low": torch.randn(20),
              "w_dir": torch.randn(71)}
    rho = {k: (torch.randn(()) if s == () else torch.randn(12))
           for k, s in param_shapes("band").items()}
    w = schedule_weights(0, 10)
    t = torch.ones(1, 1, 12, 12)
    sl = sample_loss(readout="band", fields=fields, rho_pred=rho, mask_target=t,
                     weights=w)
    assert not sl.has_oracle
    assert set(sl.parts) == {"L_mask"}
    assert torch.isfinite(sl.total)


def test_sample_loss_adds_the_three_auxiliaries_with_an_oracle():
    torch.manual_seed(0)
    fields = {"m_hi": torch.rand(1, 1, 12, 12), "s_low": torch.randn(20),
              "w_dir": torch.nn.functional.normalize(torch.randn(71), dim=0)}
    rho = {k: (torch.randn(()) if s == () else torch.randn(12))
           for k, s in param_shapes("band").items()}
    w = schedule_weights(0, 10)
    sl = sample_loss(
        readout="band", fields=fields, rho_pred=rho,
        mask_target=torch.ones(1, 1, 12, 12), weights=w,
        s_star=torch.randn(20), r_star=torch.rand(257),
        w_dir_star=torch.nn.functional.normalize(torch.randn(71), dim=0),
    )
    assert sl.has_oracle
    assert set(sl.parts) == {"L_mask", "L_s", "L_curve", "L_dir"}
    manual = (w["mask"] * sl.parts["L_mask"] + w["s"] * sl.parts["L_s"]
              + w["curve"] * sl.parts["L_curve"] + w["dir"] * sl.parts["L_dir"])
    assert abs(float(sl.total) - float(manual)) < 1e-6


def test_aggregate_reports_both_context_sub_batches():
    from q3vl.whereb.losses import SampleLoss

    losses = [SampleLoss(torch.tensor(1.0), {}, {"soft_iou": 0.5}, True),
              SampleLoss(torch.tensor(3.0), {}, {"soft_iou": 0.1}, False)]
    total, stats = aggregate(losses, ["gt", "generated"])
    assert abs(float(total) - 2.0) < 1e-6
    assert stats["by_context"]["gt"]["loss"] == 1.0
    assert stats["by_context"]["generated"]["loss"] == 3.0
    assert stats["n_with_oracle"] == 1


# --- B5 / ruling D-B16: the auxiliary denominator ---------------------------

def _split_losses(n: int, n_oracle: int, mask: float = 1.0, aux: float = 4.0):
    from q3vl.whereb.losses import SampleLoss

    out = []
    for i in range(n):
        has = i < n_oracle
        out.append(SampleLoss(
            total=torch.tensor(mask + (aux if has else 0.0)),
            parts={}, scalars={}, has_oracle=has,
            mask_term=torch.tensor(mask),
            aux_term=torch.tensor(aux) if has else None,
        ))
    return out


def test_aux_losses_are_normalised_by_the_samples_that_have_an_oracle():
    """Stage-1's nominal 1.00 must actually be 1.00 (review blocker B5)."""
    losses = _split_losses(n=4, n_oracle=1, mask=1.0, aux=4.0)
    total, stats = aggregate(losses)
    assert abs(float(total) - (1.0 + 4.0)) < 1e-6      # not 1.0 + 4.0/4
    assert stats["n_with_oracle"] == 1
    assert stats["oracle_fraction"] == 0.25
    assert stats["aux_effective_scale"] == 1.0
    assert stats["aux_denominator"] == "with_oracle"


def test_batch_denominator_reproduces_the_dilution_and_reports_it():
    losses = _split_losses(n=4, n_oracle=1, mask=1.0, aux=4.0)
    total, stats = aggregate(losses, aux_denominator="batch")
    assert abs(float(total) - (1.0 + 4.0 / 4)) < 1e-6
    assert stats["aux_effective_scale"] == 0.25        # the silent 0.47x, exposed


def test_mask_loss_is_still_averaged_over_the_whole_batch():
    from q3vl.whereb.losses import SampleLoss

    losses = [
        SampleLoss(torch.tensor(1.0), {}, {}, False, torch.tensor(1.0), None),
        SampleLoss(torch.tensor(3.0), {}, {}, False, torch.tensor(3.0), None),
    ]
    total, stats = aggregate(losses)
    assert abs(float(total) - 2.0) < 1e-6              # every sample has a mask
    assert stats["aux_effective_scale"] == 0.0


def test_aux_denominator_does_not_change_a_fully_covered_batch():
    losses = _split_losses(n=4, n_oracle=4, mask=1.0, aux=4.0)
    a, _ = aggregate(losses)
    b, _ = aggregate(losses, aux_denominator="batch")
    assert abs(float(a) - float(b)) < 1e-6
    assert abs(float(a) - 5.0) < 1e-6


def test_unknown_aux_denominator_is_rejected():
    with pytest.raises(ValueError, match="aux_denominator"):
        aggregate(_split_losses(2, 1), aux_denominator="whatever")


def test_sample_loss_exposes_the_two_terms_separately():
    torch.manual_seed(0)
    fields = {"m_hi": torch.rand(1, 1, 12, 12), "s_low": torch.randn(20),
              "w_dir": torch.nn.functional.normalize(torch.randn(71), dim=0)}
    rho = {k: (torch.randn(()) if s == () else torch.randn(12))
           for k, s in param_shapes("band").items()}
    w = schedule_weights(0, 10)
    sl = sample_loss(
        readout="band", fields=fields, rho_pred=rho,
        mask_target=torch.ones(1, 1, 12, 12), weights=w,
        s_star=torch.randn(20), r_star=torch.rand(257),
        w_dir_star=torch.nn.functional.normalize(torch.randn(71), dim=0),
    )
    assert sl.mask_term is not None and sl.aux_term is not None
    assert abs(float(sl.total) - float(sl.mask_term + sl.aux_term)) < 1e-6
    no_oracle = sample_loss(readout="band", fields=fields, rho_pred=rho,
                            mask_target=torch.ones(1, 1, 12, 12), weights=w)
    assert no_oracle.aux_term is None
    assert torch.equal(no_oracle.total, no_oracle.mask_term)
