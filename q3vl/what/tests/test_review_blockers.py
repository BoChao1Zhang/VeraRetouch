"""Regression tests for the six blockers of ``docs/reviews/REVIEW-impl-What.md``.

Each test names the blocker it pins and, where the reviewer supplied one, uses the
reviewer's own constructive counter-example.  A test that merely exercised the
fixed code would not stop the bug coming back; these fail on the pre-fix
behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from q3vl.what.config import GATE_FAILED_TAG, TrainConfig
from q3vl.what.evaluate import ceiling_board, main_board
from q3vl.what.losses import loss_style_dist
from q3vl.what.metrics import style_diagnostics
from q3vl.what.model import WhatModel
from q3vl.what.provenance import (
    WhereProvenanceError,
    assert_where_consistency,
    collect_where_digests,
    file_sha256,
)
from q3vl.what.trainer import WhatTrainer

from .conftest import MockBuilder, MockDataset, small_arm


def _row(arm, step, de, p90=1.0, ceiling=False, gate=True):
    return {"arm": arm, "step": step, "local_image_de00_median": de,
            "lut_de00_p90": p90, "boundary_de00_median": de,
            "n_trainable_params": 1, "latency_ms": 1.0,
            "is_ceiling": ceiling, "gate_pass": gate}


# --- B-1: the gate is a filter, not a label ---------------------------------

def test_b1_a_gate_failing_arm_cannot_top_the_board():
    """The reviewer's counter-example, verbatim: T01 fails the gate but has the
    better CIEDE2000.  Before the fix it came first with ``gate_pass: False``
    recorded in its own row."""
    board = main_board([_row("T01", 500, 1.0, gate=False),
                        _row("T02", 500, 2.0, gate=True)], split="V_what")
    assert [r["arm"] for r in board["ranked"]] == ["T02"]
    assert [r["arm"] for r in board["gate_failed"]] == ["T01"]
    assert board["top2"] == [{"arm": "T02", "step": 500}]
    assert board["any_gate_failed"] and board["selection_possible"]


def test_b1_a_gate_failing_step_cannot_represent_its_arm():
    """Same arm, two steps: the better metric fails the gate.  The arm must enter
    the board by its gate-passing step, not by its best-looking one."""
    board = main_board([_row("T01", 500, 1.0, gate=False),
                        _row("T01", 1000, 1.5, gate=True)], split="V_what")
    assert [(r["arm"], r["step"]) for r in board["ranked"]] == [("T01", 1000)]
    assert board["gate_failed"] == []          # the arm is represented already


def test_b1_all_arms_failing_produces_no_selection_but_a_diagnosis():
    """Protocol 5.6's shape, transposed: still rank for debugging, select nothing,
    tag the board.  This is the R1 scenario (bake gate may be unreachable)."""
    board = main_board([_row("T01", 500, 1.0, gate=False),
                        _row("T05", 500, 2.0, gate=False)], split="V_what")
    assert board["ranked"] == [] and board["top2"] == []
    assert board["selection_possible"] is False
    assert board["tag"] == GATE_FAILED_TAG
    assert [r["arm"] for r in board["diagnostic_ranked"]] == ["T01", "T05"]
    assert len(board["gate_failed"]) == 2


def test_b1_ceiling_arms_stay_out_of_both_tables():
    rows = [_row("T01", 1, 2.0), _row("C03", 1, 0.1, ceiling=True),
            _row("C04", 1, 0.1, ceiling=True, gate=False)]
    board = main_board(rows, split="V_what")
    assert [r["arm"] for r in board["ranked"]] == ["T01"]
    assert board["gate_failed"] == []
    assert sorted(c["arm"] for c in ceiling_board(rows)) == ["C03", "C04"]


# --- B-2: the rolling deletion must not eat the winner -----------------------

class _EvalStub:
    """Returns a pre-scripted ``V_what`` score per step."""

    def __init__(self, scores: dict[int, float], gate: dict[int, bool] | None = None):
        self.scores = scores
        self.gate = gate or {}
        self.calls: list[int] = []

    def __call__(self, step: int) -> dict:
        self.calls.append(step)
        return {"local_image_de00_median": self.scores.get(step, 99.0),
                "gate_pass": self.gate.get(step, True)}


def _trainer(tmp_path, eval_fn, *, steps: int, keep_last: int = 2,
             eval_every: int = 2) -> WhatTrainer:
    torch.manual_seed(0)
    cfg = small_arm("T01")
    ds = MockDataset(4)
    tcfg = TrainConfig(arm="T01", micro_batch=2, effective_batch=2,
                       eval_steps=eval_every, save_steps=eval_every,
                       keep_last=keep_last, grad_ratio_every=10 ** 9,
                       style_queue_size=8)
    return WhatTrainer(WhatModel(cfg), MockBuilder(cfg), ds, cfg, tcfg,
                       run_dir=tmp_path, device="cpu", eval_fn=eval_fn,
                       log_every=10 ** 9,
                       order=[i % 4 for i in range(steps * 2)])


def test_b2_an_early_best_checkpoint_survives_the_rolling_deletion(tmp_path):
    """The reviewer's scenario: best at an early step, many ordinary saves after.
    Before the fix the file was unlinked long before selection."""
    ev = _EvalStub({2: 0.5})                     # step 2 is best, everything else 99
    tr = _trainer(tmp_path, ev, steps=12, keep_last=2, eval_every=2)
    tr.train()
    best = tr.best()
    assert best is not None and best["step"] == 2
    alive = [c for c in tr.state.saved
             if c["step"] == 2 and not c.get("deleted")]
    assert alive, "every file for the selected step was deleted"
    assert any(c["protected"] or c["best_protected"] for c in alive)
    assert any(Path(c["path"]).exists() for c in alive), \
        "the selected checkpoint was deleted"
    assert tr.state.lost_best_steps == []
    # the protection is not a licence to keep everything
    assert any(c.get("deleted") for c in tr.state.saved)


def test_b2_protection_moves_with_the_best_and_the_old_one_is_reclaimed(tmp_path):
    ev = _EvalStub({2: 5.0, 4: 4.0, 6: 0.1})
    tr = _trainer(tmp_path, ev, steps=12, keep_last=1, eval_every=2)
    tr.train()
    assert tr.best()["step"] == 6
    marked = {c["step"] for c in tr.state.saved if c.get("best_protected")}
    assert marked <= {6}, marked          # never more than the incumbent
    alive6 = [c for c in tr.state.saved if c["step"] == 6 and not c.get("deleted")]
    assert alive6 and any(Path(c["path"]).exists() for c in alive6)
    # the superseded bests were reclaimed rather than accumulating
    assert not any(c.get("best_protected") for c in tr.state.saved
                   if c["step"] in (2, 4))
    assert tr.state.lost_best_steps == []


def test_b2_both_orderings_are_covered_eval_before_save_and_save_before_eval(tmp_path):
    """S0-TRAIN's lesson: one deletion path was fixed and the other was not.

    Here the two paths are the two interleavings of eval and save.  ``eval_steps
    == save_steps`` exercises "eval names a best whose file does not exist yet";
    ``save_steps < eval_steps`` exercises "saves accumulate between evals and the
    incumbent must survive them".
    """
    for eval_every, save_every in ((2, 2), (6, 2)):
        d = tmp_path / f"{eval_every}_{save_every}"
        ev = _EvalStub({2: 0.5, 6: 0.4, 12: 9.0})
        tr = _trainer(d, ev, steps=14, keep_last=1, eval_every=eval_every)
        object.__setattr__(tr.cfg, "save_steps", save_every)
        tr.train()
        best = tr.best()
        alive = [c for c in tr.state.saved
                 if c["step"] == best["step"] and not c.get("deleted")]
        assert alive, (eval_every, save_every, best)
        assert any(Path(c["path"]).exists() for c in alive), (eval_every, save_every)
        assert tr.state.lost_best_steps == [], (eval_every, save_every)


def test_b2_protected_milestones_are_never_reclaimed(tmp_path):
    tr = _trainer(tmp_path, _EvalStub({}), steps=8, keep_last=1, eval_every=10 ** 9)
    tr.train()
    protected = [c for c in tr.state.saved if c["protected"]]
    assert protected, "final/epoch milestones must be protected"
    for c in protected:
        assert not c.get("deleted") and Path(c["path"]).exists()


def test_b2_best_skips_gate_failing_checkpoints(tmp_path):
    ev = _EvalStub({2: 0.1, 4: 0.9}, gate={2: False, 4: True})
    tr = _trainer(tmp_path, ev, steps=6, keep_last=1, eval_every=2)
    tr.train()
    assert tr.best()["step"] == 4          # 0.1 is better but failed the gate


def test_b2_keep_last_none_disables_rolling(tmp_path):
    tr = _trainer(tmp_path, _EvalStub({}), steps=8, keep_last=None, eval_every=10 ** 9)
    tr.train()
    assert not any(c.get("deleted") for c in tr.state.saved)
    assert all(Path(c["path"]).exists() for c in tr.state.saved)


# --- B-5 / amendment A-3: one natural weighting for all twelve arms ----------

@pytest.mark.parametrize("arm", ["T01", "T04", "C01", "C02", "C03", "C04"])
def test_b5_every_arm_weights_the_natural_half_with_the_frozen_mask(arm):
    cfg = small_arm(arm)
    batch = MockBuilder(cfg).build([MockDataset(2)[0], MockDataset(2)[1]])
    for t in batch.targets:
        assert t["natural_weighting"] == "frozen_m_pred"
        assert "u_gt" in t


def test_b5_the_query_points_do_not_depend_on_the_arm():
    """The point of A-3: twelve arms, one loss.  Same samples, same query colours."""
    ref = None
    for arm in ("T01", "T04", "C01", "C02", "C03", "C04"):
        cfg = small_arm(arm)
        batch = MockBuilder(cfg).build([MockDataset(2)[0], MockDataset(2)[1]])
        xs = torch.stack([t["x"] for t in batch.targets])
        if ref is None:
            ref = xs
        else:
            assert torch.equal(ref, xs), arm


def test_b5_a_local_sample_without_a_frozen_mask_is_refused():
    """Silently falling back to whole-image sampling is what B-5 was."""
    from q3vl.what.data import WhatBatchBuilder

    builder = WhatBatchBuilder.__new__(WhatBatchBuilder)
    builder.seed = 0
    builder._x_uniform = torch.zeros(4, 3)
    sample = type("S", (), {"is_global": False, "sample_id": "s0",
                            "image_tensor": lambda self: torch.rand(3, 8, 8)})()
    with pytest.raises(RuntimeError, match="A-3"):
        WhatBatchBuilder.query_points(builder, sample, None)


def test_b5_a_global_sample_samples_the_whole_image():
    from q3vl.what.data import WhatBatchBuilder

    builder = WhatBatchBuilder.__new__(WhatBatchBuilder)
    builder.seed = 0
    builder._x_uniform = torch.zeros(4, 3)
    sample = type("S", (), {"is_global": True, "sample_id": "g0",
                            "image_tensor": lambda self: torch.rand(3, 8, 8)})()
    _x, weighting = WhatBatchBuilder.query_points(builder, sample, None)
    assert weighting == "global_uniform"


def test_n14_a_main_arm_rejects_an_oracle_where_signal():
    cfg = small_arm("T01")
    batch = MockBuilder(cfg).build([MockDataset(1)[0]])
    batch.inputs["where"].source = "oracle"
    with pytest.raises(AssertionError, match="8.2"):
        batch.check_inputs(expect_source="predicted")


# --- B-6: one frozen Where checkpoint for every arm --------------------------

def _write_setup(root: Path, arm: str, digest: str | None, path: str = "/ck.pt"):
    d = root / arm
    d.mkdir(parents=True, exist_ok=True)
    (d / "run_setup.json").write_text(json.dumps(
        {"arm": arm, "where": {"checkpoint_sha256": digest, "path": path}}))


def test_b6_digest_of_a_real_file_is_stable(tmp_path):
    f = tmp_path / "ck.pt"
    f.write_bytes(b"weights" * 1000)
    assert file_sha256(f) == file_sha256(f)
    f2 = tmp_path / "ck2.pt"
    f2.write_bytes(b"weights" * 1000 + b"!")
    assert file_sha256(f) != file_sha256(f2)


def test_b6_a_second_arm_on_a_different_checkpoint_is_refused(tmp_path):
    _write_setup(tmp_path, "T01", "aaa")
    ok = assert_where_consistency(tmp_path, "T02", "aaa", "/ck.pt")
    assert ok["consistent"] and ok["n_previous_arms_checked"] == 1
    with pytest.raises(WhereProvenanceError, match="disagrees"):
        assert_where_consistency(tmp_path, "T02", "bbb", "/other.pt")


def test_b6_a_declared_digest_must_exist(tmp_path):
    _write_setup(tmp_path, "T01", "aaa")
    with pytest.raises(WhereProvenanceError, match="has none"):
        assert_where_consistency(tmp_path, "C01", None, None)


def test_b6_the_no_where_controls_are_not_exempt(tmp_path):
    """Amendment A-3 conditions C01/C02 on the frozen mask, so they are in scope."""
    _write_setup(tmp_path, "T01", "aaa")
    with pytest.raises(WhereProvenanceError, match="disagrees"):
        assert_where_consistency(tmp_path, "C01", "ccc", "/other.pt")


def test_b6_an_unreadable_setup_is_a_stop_not_a_skip(tmp_path):
    d = tmp_path / "T01"
    d.mkdir(parents=True)
    (d / "run_setup.json").write_text("{not json")
    assert any(k.startswith("<unreadable:") for k in collect_where_digests(tmp_path))
    with pytest.raises(WhereProvenanceError, match="unreadable"):
        assert_where_consistency(tmp_path, "T02", "aaa", "/ck.pt")


def test_b6_the_first_arm_has_nothing_to_disagree_with(tmp_path):
    rec = assert_where_consistency(tmp_path, "T01", "aaa", "/ck.pt")
    assert rec["consistent"] and rec["n_previous_arms_checked"] == 0


def test_b6_an_arm_ignores_its_own_previous_run(tmp_path):
    _write_setup(tmp_path, "T01", "aaa")
    assert assert_where_consistency(tmp_path, "T01", "bbb", "/new.pt")["consistent"]


# --- N-16: the preflight must not silently downgrade delivered evidence ------

def test_n16_overwriting_a_complete_report_with_an_incomplete_one_is_refused(tmp_path):
    from q3vl.what.preflight import PreflightReport, run_what_preflight, write_report

    full = run_what_preflight(tmp_path, with_data=False)
    # pretend the delivered report was complete
    path = tmp_path / "preflight_what.json"
    data = json.loads(path.read_text())
    data["complete"] = True
    data["skipped"] = []
    path.write_text(json.dumps(data))

    partial = PreflightReport(checks=list(full.checks))
    assert not partial.complete                     # it has the two skips
    with pytest.raises(RuntimeError, match="silently downgraded"):
        write_report(partial, tmp_path)
    write_report(partial, tmp_path, force=True)
    assert (tmp_path / "preflight_what.json.superseded").exists()


def test_n16_default_output_is_not_the_deliverable_directory():
    import argparse
    import inspect

    from q3vl.what import preflight as pf
    from q3vl.what.config import REPORT_DIR

    src = inspect.getsource(pf.main)
    assert "REPORT_DIR" not in src
    assert "RUN_ROOT" in src
    parser = argparse.ArgumentParser()
    # rebuild main()'s parser default the same way
    assert str(REPORT_DIR) not in str(pf.RUN_ROOT)
