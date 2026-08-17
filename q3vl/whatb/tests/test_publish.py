"""The publication gate, including the two failures it must keep apart."""

from __future__ import annotations

import json

import pytest
import torch

from q3vl.whatb import criteria as C
from q3vl.whatb import publish as P
from q3vl.whatb.degeneracy import (
    DegeneracyThresholds,
    DegenerateTransform,
    assert_transform_not_degenerate,
)
from q3vl.whatb.guards import (
    DegeneracyCheckNotRun,
    clear_degeneracy_check,
    clear_step_witness,
    degeneracy_check_ran,
)


FULL_ROW = {c: 0.5 for c in P.FIRST_STEP_COLUMNS}


def _board(**kw):
    rows = [{"sample_id": f"s{i}", "winner_confidence": "normal",
             "task_type": "style" if i % 2 else "local",
             "E_arm": 4.0 + 0.1 * i,
             "E_B0_identity": 30.0, "E_B1_libmean": 25.0,
             "E_B2_librandom_repeats": [35.0] * 8,
             "E_B3_bucket_retrieval_repeats": [20.0] * 8,
             "E_B4_oracle": 9.0,
             "E_N1_shuffle": 6.0, "M_N1_shuffle": 3.0,
             "E_N2_irrelevant": 7.0, "M_N2_irrelevant": 4.0,
             "E_N3_const": 8.0, "M_N3_const": 5.0} for i in range(6)]
    board = C.build_board(rows, arm="EPR-024", split="V_what")
    board.update(kw)
    return board


def _guard_ran():
    """Fire the shared degeneracy guard once, as every real run does."""
    x = torch.rand(4, 32, 3)
    assert_transform_not_degenerate(x * 0.7 + 0.1, x, exit_process=False)
    assert degeneracy_check_ran() is not None


@pytest.fixture(autouse=True)
def _no_witness():
    clear_step_witness()
    clear_degeneracy_check()
    # every test in this file publishes a board, and a published board now has
    # to show that the degenerate-solution guard really ran in this process
    # (W4).  The two tests that check the refusal itself clear it again.
    _guard_ran()
    yield
    clear_step_witness()
    clear_degeneracy_check()


# --------------------------------------------------------------------------- #
# the three-tier steps row
# --------------------------------------------------------------------------- #
def test_tier_one_caller(tmp_path):
    rep = P.assert_publishable(_board(published=True), "EPR-024",
                               steps_row=FULL_ROW, axes=())
    assert rep["steps"]["source"] == "caller"


def test_tier_two_disk(tmp_path):
    path = tmp_path / "steps.jsonl"
    path.write_text("\n" + json.dumps({**FULL_ROW, "step": 1}) + "\n"
                    + json.dumps({"step": 2}) + "\n")
    rep = P.assert_publishable(_board(published=True), "EPR-024",
                               steps_path=path, axes=())
    assert rep["steps"]["source"] == "disk"


def test_tier_three_in_process_witness(tmp_path):
    P.record_step_witness({**FULL_ROW, "step": 0})
    rep = P.assert_publishable(_board(published=True), "EPR-024",
                               steps_path=tmp_path / "missing.jsonl", axes=())
    assert rep["steps"]["source"] == "witness"


def test_not_passed_and_not_computed_are_different_exceptions(tmp_path):
    # (a) nothing anywhere -> plumbing failure
    with pytest.raises(P.StepsRowUnavailable):
        P.assert_publishable(_board(published=True), "EPR-024",
                             steps_path=tmp_path / "nope.jsonl", axes=())
    # (b) a row exists but the loss columns are not in it -> substantive failure
    with pytest.raises(P.LossColumnsMissing):
        P.assert_publishable(_board(published=True), "EPR-024",
                             steps_row={"L_rec": 1.0}, axes=())
    assert not issubclass(P.StepsRowUnavailable, P.LossColumnsMissing)
    assert not issubclass(P.LossColumnsMissing, P.StepsRowUnavailable)


def test_loss_level_four_demands_l_img():
    with pytest.raises(P.LossColumnsMissing, match="L_img"):
        P.assert_publishable(_board(published=True), "EPR-024",
                             steps_row=FULL_ROW, loss_level=4, axes=())
    rep = P.assert_publishable(_board(published=True), "EPR-024",
                               steps_row={**FULL_ROW, "L_img": 0.1},
                               loss_level=4, axes=())
    assert "L_img" in rep["steps"]["values"]


def test_arm_specific_extra_column_is_demanded():
    with pytest.raises(P.LossColumnsMissing, match="L_interp"):
        P.assert_publishable(_board(published=True), "EPR-026",
                             steps_row=FULL_ROW, axes=(),
                             extra_step_columns=("L_interp",))


def test_eval_only_skips_the_training_side_but_not_the_criteria():
    rep = P.assert_publishable(_board(published=True), "EPR-024",
                               eval_only=True, axes=())
    assert "skipped" in rep["steps"]
    empty = C.build_board([], arm="EPR-024", split="V_what")
    empty["published"] = True
    with pytest.raises(C.CriterionNotComputed):
        P.assert_publishable(empty, "EPR-024", eval_only=True, axes=())


# --------------------------------------------------------------------------- #
# the headline-presence rule (§10.2)
# --------------------------------------------------------------------------- #
def test_published_board_must_carry_the_headline():
    board = _board(published=True)
    board["contexts"]["all"]["headline_normal_only"] = {"n": 0}
    with pytest.raises(C.CriterionNotComputed, match="headline_normal_only"):
        P.assert_publishable(board, "EPR-024", steps_row=FULL_ROW, axes=())


def test_interim_board_with_no_normal_rows_records_the_absence():
    board = C.build_board(
        [{"sample_id": "s", "winner_confidence": "low", "task_type": "style",
          "E_arm": 4.0}], arm="EPR-024", split="V_what")
    board["published"] = False
    rep = P.assert_publishable(board, "EPR-024", steps_row=FULL_ROW,
                               axes=(), required=[])
    assert rep["headline"]["n"] == 0 and "interim" in rep["headline"]["note"]


def test_interim_board_that_has_normal_rows_but_no_headline_raises():
    board = _board(published=False)
    board["contexts"]["all"]["headline_normal_only"] = {"n": 0}
    with pytest.raises(P.HeadlineMissing, match="disagree"):
        P.assert_publishable(board, "EPR-024", steps_row=FULL_ROW, axes=(),
                             required=[])


# --------------------------------------------------------------------------- #
# the degeneracy guard (alias module -> guards, one implementation)
# --------------------------------------------------------------------------- #
def test_degeneracy_alias_is_the_same_object():
    from q3vl.whatb import guards

    assert assert_transform_not_degenerate is guards.assert_transform_not_degenerate


def test_healthy_transform_passes():
    x = torch.rand(8, 256, 3)
    y = (x * torch.rand(8, 1, 3) + 0.1 * torch.rand(8, 1, 3)).clamp(0, 1)
    rep = assert_transform_not_degenerate(y, x, exit_process=False)
    assert rep.ok and rep.n_samples == 8 and rep.n_queries == 256


@pytest.mark.parametrize("kind", ["constant", "identity", "same_for_all"])
def test_each_degenerate_mode_is_caught(kind):
    x = torch.rand(8, 256, 3)
    if kind == "constant":
        y = torch.full_like(x, 0.4)
    elif kind == "identity":
        y = x.clone()
    else:
        y = (x[0:1] * 0.3 + 0.2).expand_as(x).contiguous()
    with pytest.raises(DegenerateTransform) as exc:
        assert_transform_not_degenerate(y, x, exit_process=False)
    assert exc.value.report.failures


def test_degeneracy_exits_the_process_by_default():
    x = torch.rand(4, 32, 3)
    with pytest.raises(SystemExit):
        assert_transform_not_degenerate(torch.zeros_like(x), x)


def test_thresholds_travel_into_run_setup():
    t = DegeneracyThresholds()
    x = torch.rand(4, 32, 3)
    rep = assert_transform_not_degenerate(x * 0.5 + 0.1, x, thresholds=t,
                                          exit_process=False)
    assert rep.as_dict()["thresholds"] == t.as_dict()
    assert set(t.as_dict()) == {"point_std", "identity_dev", "cross_std"}


# --------------------------------------------------------------------------- #
# the degenerate-solution guard must have RUN (W4)
# --------------------------------------------------------------------------- #
def test_published_board_refuses_when_the_degeneracy_guard_never_ran():
    """--eval-only / --no-train / --quick-eval-every 0 must not be a way out."""
    clear_degeneracy_check()
    with pytest.raises(DegeneracyCheckNotRun, match="never ran"):
        P.assert_publishable(_board(published=True), "EPR-024",
                             steps_row=FULL_ROW, axes=())


def test_interim_board_and_the_explicit_escape_hatch_are_allowed():
    clear_degeneracy_check()
    # an interim (quick-eval) board is not a publication and is not refused
    rep = P.assert_publishable(_board(published=False, quick=True), "EPR-024",
                               steps_row=FULL_ROW, axes=())
    assert rep["degeneracy"]["ran"] is False
    # the escape hatch is recorded, never silent
    rep = P.assert_publishable(_board(published=True), "EPR-024",
                               steps_row=FULL_ROW, axes=(),
                               require_degeneracy_check=False)
    assert rep["degeneracy"]["ran"] is False and rep["degeneracy"]["required"] is False


def test_the_witness_carries_the_measured_numbers():
    rep = P.assert_publishable(_board(published=True), "EPR-024",
                               steps_row=FULL_ROW, axes=())
    for key in ("point_std", "identity_dev", "cross_std", "ok"):
        assert key in rep["degeneracy"]
    assert rep["degeneracy"]["ok"] is True
