"""The two guards, and the two failures they exist to make impossible.

Both failures happened on the Where side within the week before this batch was
written; both are reproduced here as tests so a regression is a red test rather
than 2.6 GPU-hours.
"""

from __future__ import annotations

import json

import pytest
import torch

from q3vl.whatb.guards import (
    FIRST_STEP_COLUMNS,
    DegeneracyThresholds,
    DegenerateTransform,
    LossColumnsMissing,
    StepsRowUnavailable,
    assert_first_step_columns,
    assert_transform_not_degenerate,
    clear_step_witness,
    measure_degeneracy,
    record_step_witness,
    resolve_first_step_row,
)


@pytest.fixture(autouse=True)
def _clean_witness():
    clear_step_witness()
    yield
    clear_step_witness()


def _healthy(b: int = 4, p: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    x = torch.rand(b, p, 3, generator=g, dtype=torch.float64)
    scale = torch.rand(b, 1, 3, generator=g, dtype=torch.float64) * 0.5 + 0.6
    shift = torch.rand(b, 1, 3, generator=g, dtype=torch.float64) * 0.2
    return x, (x * scale + shift).clamp(0, 1)


# --------------------------------------------------------------------------- #
# guard 2: degenerate transform  (PRND / CONDINST, 1200 steps before anyone saw it)
# --------------------------------------------------------------------------- #
def test_a_healthy_transform_passes() -> None:
    x, y = _healthy()
    report = assert_transform_not_degenerate(y, x, exit_process=False)
    assert report.ok and report.failures == ()
    assert report.n_samples == 4 and report.n_queries == 64
    print(
        f"\nhealthy: point_std={report.point_std:.3e} identity_dev={report.identity_dev:.3e} "
        f"cross_std={report.cross_std:.3e}"
    )


def test_constant_field_is_caught() -> None:
    """The exact PRND/CONDINST shape: one output colour for every input."""
    x, _ = _healthy()
    y = torch.full_like(x, 0.42)
    with pytest.raises(DegenerateTransform) as exc:
        assert_transform_not_degenerate(y, x, exit_process=False)
    assert any("flat across query colours" in f for f in exc.value.report.failures)
    assert any("one transform for every sample" in f for f in exc.value.report.failures)


def test_identity_output_is_caught() -> None:
    x, _ = _healthy()
    with pytest.raises(DegenerateTransform) as exc:
        assert_transform_not_degenerate(x.clone(), x, exit_process=False)
    assert any("transform is the identity" in f for f in exc.value.report.failures)


def test_same_transform_for_every_sample_is_caught() -> None:
    """Passes the first two checks; fails the third.  This is the N3 look-alike."""
    x, _ = _healthy()
    shared = torch.stack([x[0]] * x.shape[0])
    y = (shared * 0.8 + 0.1)
    report = measure_degeneracy(y, shared)
    assert report.point_std > 1e-3 and report.identity_dev > 1e-3
    assert report.failures == (
        f"one transform for every sample: mean std_over_samples = {report.cross_std:.3e} <= 1.0e-04",
    )


def test_guard_exits_the_process_by_default() -> None:
    """A training loop must not be able to catch this as a warning."""
    x, _ = _healthy()
    with pytest.raises(SystemExit) as exc:
        assert_transform_not_degenerate(x.clone(), x)
    assert exc.value.code == 2


def test_thresholds_are_recorded_and_overridable() -> None:
    thr = DegeneracyThresholds()
    assert thr.as_dict() == {"point_std": 1e-3, "identity_dev": 1e-3, "cross_std": 1e-4}
    x, y = _healthy()
    strict = DegeneracyThresholds(point_std=10.0, identity_dev=1e-3, cross_std=1e-4)
    with pytest.raises(DegenerateTransform):
        assert_transform_not_degenerate(y, x, thresholds=strict, exit_process=False)
    report = measure_degeneracy(y, x)
    assert report.as_dict()["thresholds"] == thr.as_dict()


def test_single_sample_batch_skips_only_the_cross_sample_check() -> None:
    x, y = _healthy(b=1)
    report = measure_degeneracy(y, x)
    assert report.ok
    assert report.cross_std != report.cross_std  # nan: undefined with one sample


def test_shared_query_grid_broadcasts() -> None:
    x, y = _healthy()
    assert measure_degeneracy(y, x[0]).ok


def test_guard_never_leaves_the_input_device() -> None:
    """Reductions run where the tensor is: CPU/CUDA tie-breaks moved an IoU 0.296."""
    x, y = _healthy()
    tracked: list[str] = []
    orig = torch.Tensor.cpu

    def _trap(self):  # pragma: no cover - only fires on regression
        tracked.append("cpu")
        return orig(self)

    torch.Tensor.cpu = _trap  # type: ignore[method-assign]
    try:
        measure_degeneracy(y, x)
    finally:
        torch.Tensor.cpu = orig  # type: ignore[method-assign]
    assert tracked == []


# --------------------------------------------------------------------------- #
# guard 1: the first-step-row contract  (SEGSAM / PRND)
# --------------------------------------------------------------------------- #
_GOOD_ROW = {c: 1.0 for c in FIRST_STEP_COLUMNS}


def test_tier1_caller() -> None:
    row, source = resolve_first_step_row(_GOOD_ROW)
    assert source == "caller" and row == _GOOD_ROW


def test_tier2_disk(tmp_path) -> None:
    path = tmp_path / "steps.jsonl"
    path.write_text("\n" + json.dumps(_GOOD_ROW) + "\n" + json.dumps({"step": 1}) + "\n")
    row, source = resolve_first_step_row(None, steps_path=path)
    assert source == "disk" and row == _GOOD_ROW


def test_tier3_in_process_witness() -> None:
    record_step_witness(_GOOD_ROW)
    row, source = resolve_first_step_row(None, steps_path="/nonexistent/steps.jsonl")
    assert source == "witness" and row == _GOOD_ROW


def test_the_call_site_that_passes_nothing_still_works(tmp_path) -> None:
    """``evaluate.py:660`` -- the one that gates ``metrics.json`` -- passes nothing.

    That signature mismatch is what made SEGSAM and PRND report a loss that had
    in fact run.  Here the assertion fetches the row itself.
    """
    path = tmp_path / "steps.jsonl"
    path.write_text(json.dumps(_GOOD_ROW) + "\n")
    row, source = assert_first_step_columns(steps_path=path)
    assert source == "disk" and row == _GOOD_ROW


def test_not_handed_and_not_computed_raise_different_errors(tmp_path) -> None:
    with pytest.raises(StepsRowUnavailable, match="wiring failure, not a pass"):
        assert_first_step_columns(steps_path=tmp_path / "absent.jsonl")

    partial = dict(_GOOD_ROW)
    del partial["L_hc"]
    partial["n_hc_masked"] = None
    with pytest.raises(LossColumnsMissing, match=r"missing \['L_hc', 'n_hc_masked'\]"):
        assert_first_step_columns(steps_row=partial)

    assert not issubclass(StepsRowUnavailable, LossColumnsMissing)
    assert not issubclass(LossColumnsMissing, StepsRowUnavailable)


def test_unavailable_is_a_failure_not_a_pass() -> None:
    row, source = resolve_first_step_row(None)
    assert row is None and source == "unavailable"
    with pytest.raises(StepsRowUnavailable):
        assert_first_step_columns()


def test_loss_level_4_adds_l_img() -> None:
    with pytest.raises(LossColumnsMissing, match=r"\['L_img'\]"):
        assert_first_step_columns((*FIRST_STEP_COLUMNS, "L_img"), steps_row=_GOOD_ROW)
    row, _ = assert_first_step_columns((*FIRST_STEP_COLUMNS, "L_img"), steps_row={**_GOOD_ROW, "L_img": 0.1})
    assert "L_img" in row


def test_frozen_column_list() -> None:
    assert FIRST_STEP_COLUMNS == (
        "L_rec", "L_hc", "L_sparse", "n_colors", "n_luts_in_batch", "mining_ratio", "n_hc_masked",
    )


def test_malformed_steps_file_falls_through_to_the_next_tier(tmp_path) -> None:
    path = tmp_path / "steps.jsonl"
    path.write_text("{not json\n")
    record_step_witness(_GOOD_ROW)
    _, source = resolve_first_step_row(None, steps_path=path)
    assert source == "witness"
