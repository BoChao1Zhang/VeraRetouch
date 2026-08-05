"""Review NF-2 -- the evaluation library must actually be *called* in production.

Three rounds of review had checked the library and never checked its caller.  So
the tests here are deliberately split:

* **wiring** -- assertions about ``run_what.py`` itself (does it construct an
  ``eval_fn``? does it pass it? does it gate on the boundary scan?).  These are
  the ones ``_EvalStub`` can never give, because a stub *is* the thing that was
  missing;
* **behaviour** -- a real ``make_eval_fn`` over mock data, driving a real trainer,
  showing that B-2's protection engages through the production construction path
  rather than through an injected double.
"""

from __future__ import annotations

import ast
import json
import time
from pathlib import Path

import pytest
import torch

from q3vl.what.boundary import (
    SCHEMA,
    ColorBoundaryError,
    merge_report,
    require_color_boundary_scan,
    summarise_scan,
)
from q3vl.what.config import (
    CONTEXT_GENERATED,
    CONTEXT_GT,
    EVAL_SUBSET_SEED,
    EVAL_SUBSET_SIZE,
    ONLINE_SELECTION_KEY,
    SELECTION_CONTEXT,
    TrainConfig,
)
from q3vl.what.evalloop import (
    build_eval_subset,
    make_eval_fn,
    mask_area_bin,
    stable_order_key,
    subset_digest,
)
from q3vl.what.model import WhatModel
from q3vl.what.trainer import WhatTrainer

from .conftest import MockBuilder, MockDataset, small_arm

RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_what.py"


# --- wiring: what the runner actually does -----------------------------------

def _runner_tree() -> ast.AST:
    return ast.parse(RUNNER.read_text(encoding="utf-8"))


def _call_kwargs(tree: ast.AST, func_name: str) -> set[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = getattr(f, "id", None) or getattr(f, "attr", None)
            if name == func_name:
                return {kw.arg for kw in node.keywords if kw.arg}
    return set()


def test_the_runner_passes_an_eval_fn_to_the_trainer():
    """NF-2's first point, asserted on the call site rather than on a stub.

    Before the fix this call had no ``eval_fn`` keyword at all, which made
    protocol 10.4's ``eval_steps`` unimplemented and blocker B-2's protection
    inert -- while B-2's own seven regression tests stayed green because every
    one of them injected an ``_EvalStub``.
    """
    kwargs = _call_kwargs(_runner_tree(), "WhatTrainer")
    assert kwargs, "run_what.py no longer constructs a WhatTrainer"
    assert "eval_fn" in kwargs, "run_what.py constructs the trainer without eval_fn"


def test_the_runner_builds_the_subset_and_the_eval_fn():
    src = RUNNER.read_text(encoding="utf-8")
    for needed in ("build_eval_subset", "make_eval_fn", "eval_subset_rows",
                   "eval_subset.json"):
        assert needed in src, needed
    # the subset manifest must reach the config digest (NF-2 ruling item 1)
    assert "_config_digest" in src and "eval_subset_digest" in \
        (RUNNER.parent / "run_what.py").read_text(encoding="utf-8")


def test_the_runner_gates_on_the_boundary_scan():
    """N-24: the scan is a pre-run hard gate, not a post-run job."""
    kwargs_src = RUNNER.read_text(encoding="utf-8")
    assert "require_color_boundary_scan" in kwargs_src
    tree = _runner_tree()
    # it must be called before the trainer is constructed
    lines = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name in ("require_color_boundary_scan", "WhatTrainer"):
                lines.setdefault(name, node.lineno)
    assert lines["require_color_boundary_scan"] < lines["WhatTrainer"]


def test_the_offline_evaluator_exists_and_feeds_main_board():
    """NF-2's third point: amendment A-4's boards need a producer."""
    offline = RUNNER.parent / "evaluate_what.py"
    assert offline.exists()
    src = offline.read_text(encoding="utf-8")
    for needed in ("main_board", "ceiling_board", "context_report", "arm_metrics",
                   "load_target_image", "sample_row"):
        assert needed in src, needed
    # both contexts, never averaged
    assert "CONTEXT_GT" in src and "CONTEXT_GENERATED" in src


def test_only_the_offline_evaluator_loads_i_tar():
    """Protocol 9.5: ``I_tar`` never enters ``L_what``."""
    pkg = RUNNER.resolve().parents[1]
    callers = []
    for path in sorted(pkg.rglob("*.py")):
        if "tests" in path.parts:
            continue
        if "load_target_image" in path.read_text(encoding="utf-8"):
            callers.append(path.name)
    assert sorted(callers) == ["data.py", "evaluate_what.py"], callers


# --- the deterministic subset ------------------------------------------------

def _rows(n=897):
    builds = ("g1", "g2", "g3", "g4", "l1", "l2", "l3", "l4", "l5", "l6")
    out = []
    for i in range(n):
        b = builds[i % len(builds)]
        local = b.startswith("l")
        out.append({"sample_id": f"sft_{i:05d}", "build": b,
                    "render_mode": "local" if local else "global",
                    "mask_area": (0.01 + 0.29 * ((i // 10) % 3)) if local else 1.0})
    return out


def test_the_subset_is_deterministic_across_processes():
    a, ma = build_eval_subset(_rows(), 256)
    b, mb = build_eval_subset(_rows(), 256)
    assert a == b and ma["digest"] == mb["digest"]
    # and it does not depend on the order the rows arrive in
    shuffled = list(reversed(_rows()))
    _c, mc = build_eval_subset(shuffled, 256)
    assert mc["digest"] == ma["digest"]


def test_the_subset_is_stratified_and_proportional():
    rows = _rows()
    idx, man = build_eval_subset(rows, 256)
    assert man["n"] == 256
    assert man["mask_area_used"] is True
    assert man["strata_keys"] == ["build", "render_mode", "mask_area_bin"]
    # every populated stratum is represented
    assert all(v["taken"] >= 1 for v in man["strata"].values())
    # the build mix tracks the population's within one sample per build
    from collections import Counter

    got = Counter(rows[i]["build"] for i in idx)
    want = Counter(r["build"] for r in rows)
    for b in want:
        assert abs(got[b] - want[b] * 256 / len(rows)) <= 1.5, b


def test_the_subset_records_when_mask_area_is_unavailable():
    rows = [dict(r, mask_area=None) for r in _rows(100)]
    _idx, man = build_eval_subset(rows, 32)
    assert man["mask_area_used"] is False
    assert man["strata_keys"] == ["build", "render_mode"]
    assert man["mask_area_bins"] is None


def test_subset_helpers():
    assert stable_order_key("s0", 1) == stable_order_key("s0", 1)
    assert stable_order_key("s0", 1) != stable_order_key("s0", 2)
    assert mask_area_bin(None) == "unknown"
    assert mask_area_bin(0.01) == "a0" and mask_area_bin(0.9) == "a3"
    assert subset_digest(["b", "a"]) == subset_digest(["a", "b"])
    with pytest.raises(ValueError):
        build_eval_subset(_rows(10), 0)
    small, man = build_eval_subset(_rows(10), 999)
    assert len(small) == 10 and man["n"] == 10


def test_the_default_subset_size_and_seed_are_the_ruling_s():
    assert EVAL_SUBSET_SIZE == 256
    assert EVAL_SUBSET_SEED == 20260804
    assert TrainConfig().eval_subset_size == 256
    assert TrainConfig().selection_key == ONLINE_SELECTION_KEY


# --- behaviour: the real eval_fn, driving a real trainer ---------------------

def _real_eval_trainer(tmp_path, *, n=8, keep_last=1, eval_every=2, steps=12):
    torch.manual_seed(0)
    cfg = small_arm("T01")
    model = WhatModel(cfg)
    ds = MockDataset(n)
    builder = MockBuilder(cfg)
    rows = [{"sample_id": ds[i].sample_id, "build": "l1", "render_mode": "local",
             "mask_area": 0.1 + 0.05 * i} for i in range(n)]
    idx, manifest = build_eval_subset(rows, 4)
    eval_fn = make_eval_fn(model, builder, ds, idx, cfg, n_trainable=model.n_trainable(),
                           micro_batch=2, subset_manifest=manifest, out_dir=tmp_path)
    tcfg = TrainConfig(arm="T01", micro_batch=2, effective_batch=2,
                       eval_steps=eval_every, save_steps=eval_every,
                       keep_last=keep_last, grad_ratio_every=10 ** 9,
                       style_queue_size=8)
    tr = WhatTrainer(model, builder, ds, cfg, tcfg, run_dir=tmp_path, device="cpu",
                     eval_fn=eval_fn, log_every=10 ** 9,
                     order=[i % n for i in range(steps * 2)])
    return tr, manifest


def test_the_real_eval_fn_produces_both_context_boards_and_a_gap(tmp_path):
    tr, manifest = _real_eval_trainer(tmp_path, steps=4, eval_every=2)
    tr.train()
    reports = [json.loads(l) for l in
               (tmp_path / "eval.jsonl").read_text().splitlines()]
    assert reports
    r = reports[0]
    assert set(r["contexts"]) == {CONTEXT_GT, CONTEXT_GENERATED}
    for ctx, m in r["contexts"].items():
        assert m["context"] == ctx
        assert m[ONLINE_SELECTION_KEY] is not None
        assert "gate_pass" in m
    assert r["gap"]["n_pairs"] == 1
    assert r["gap"]["metric"] == ONLINE_SELECTION_KEY
    assert r["selection_context"] == SELECTION_CONTEXT
    assert r["subset_digest"] == manifest["digest"]
    # the promoted keys are the generated board's, not an average
    assert r[ONLINE_SELECTION_KEY] == \
        r["contexts"][SELECTION_CONTEXT][ONLINE_SELECTION_KEY]
    # per-sample rows landed too
    lines = (tmp_path / "eval_per_sample.jsonl").read_text().splitlines()
    assert lines and {json.loads(l)["context"] for l in lines} == \
        {CONTEXT_GT, CONTEXT_GENERATED}


def test_b2_protection_engages_through_the_production_eval_fn(tmp_path):
    """B-2's regression, driven by the real evaluator instead of ``_EvalStub``.

    This is the assertion NF-2's second point needed: the mechanism was correct
    and the wiring was absent, so a test that injects its own ``eval_fn`` could
    not tell the difference.
    """
    tr, _ = _real_eval_trainer(tmp_path, keep_last=1, eval_every=2, steps=12)
    tr.train()
    assert tr.state.checkpoints, "no eval report was recorded"
    best = tr.best()
    assert best is not None and best.get(ONLINE_SELECTION_KEY) is not None
    alive = [c for c in tr.state.saved
             if c["step"] == best["step"] and not c.get("deleted")]
    assert alive, "the selected checkpoint was deleted"
    assert any(c["protected"] or c["best_protected"] for c in alive)
    assert any(Path(c["path"]).exists() for c in alive)
    assert tr.state.lost_best_steps == []
    # and the rolling deletion still happened -- protection is not "keep all"
    assert any(c.get("deleted") for c in tr.state.saved)


def test_best_reports_the_gate_fallback_and_unknown_counts(tmp_path):
    """Review N-18 / N-19, on the real report shape."""
    tr, _ = _real_eval_trainer(tmp_path, steps=4, eval_every=2)
    tr.train()
    best = tr.best()
    assert "gate_fallback" in best and "n_gate_unknown" in best
    assert best["n_gate_unknown"] == 0        # arm_metrics always writes gate_pass


def test_the_eval_pass_leaves_the_model_in_training_mode(tmp_path):
    tr, _ = _real_eval_trainer(tmp_path, steps=2, eval_every=2)
    tr.model.train()
    tr.eval_fn(0)
    assert tr.model.training
    assert tr.model.collect_pool_stats is True


def test_eval_wall_clock_is_recorded(tmp_path):
    """The wall-clock cost is a number in every report, not an assumption."""
    tr, _ = _real_eval_trainer(tmp_path, steps=2, eval_every=2)
    t0 = time.time()
    report = tr.eval_fn(0)
    assert report["eval_seconds"] > 0.0
    assert report["eval_seconds"] <= time.time() - t0 + 1.0
    assert report["n_subset"] == 4


# --- N-24: the boundary gate -------------------------------------------------

def _scan(tmp_path, *, over=(), missing=0, boundary=384, splits=None):
    from q3vl.what.boundary import SCANNED_SPLITS

    per = [summarise_scan(s, [100, 200, 300], list(over), missing, boundary, 1.0)
           for s in (splits or SCANNED_SPLITS)]
    report = merge_report(per, boundary=boundary)
    p = tmp_path / "color_boundary_scan.json"
    p.write_text(json.dumps(report))
    return p


def test_the_gate_passes_on_a_clean_scan(tmp_path):
    got = require_color_boundary_scan(_scan(tmp_path))
    assert got["ok"] and got["boundary"] == 384
    assert set(got["splits"]) >= {"train", "V_what", "T_lut_unseen"}


def test_a_missing_scan_stops_the_arm(tmp_path):
    with pytest.raises(ColorBoundaryError, match="does not exist"):
        require_color_boundary_scan(tmp_path / "nope.json")


def test_an_over_boundary_sample_stops_the_arm(tmp_path):
    p = _scan(tmp_path, over=[{"sample_id": "sft_x", "tokens_color": 400}])
    with pytest.raises(ColorBoundaryError, match="does not pass"):
        require_color_boundary_scan(p)


def test_a_scan_against_another_boundary_is_refused(tmp_path):
    p = _scan(tmp_path, boundary=256)
    with pytest.raises(ColorBoundaryError, match="different boundary"):
        require_color_boundary_scan(p, boundary=384)


def test_records_missing_the_tokens_field_are_not_a_silent_pass(tmp_path):
    p = _scan(tmp_path, missing=3)
    with pytest.raises(ColorBoundaryError, match="does not pass"):
        require_color_boundary_scan(p)


def test_an_incomplete_scan_is_refused(tmp_path):
    p = _scan(tmp_path, splits=["train", "V_what"])
    with pytest.raises(ColorBoundaryError, match="were not scanned"):
        require_color_boundary_scan(p)


def test_a_stale_schema_is_refused(tmp_path):
    p = _scan(tmp_path)
    d = json.loads(p.read_text())
    d["schema"] = "q3vl.what.color_boundary/0"
    p.write_text(json.dumps(d))
    with pytest.raises(ColorBoundaryError, match="schema"):
        require_color_boundary_scan(p)


def test_summarise_scan_counts_the_tags(tmp_path):
    row = summarise_scan("train", [380, 381, 382], [], 0, 384, 1.0)
    # the record counts the body; the span adds <color> and </color>
    assert row["max_span_with_tags"] == 384
    assert row["headroom"] == 0
    assert row["ok"] is True
    assert SCHEMA.endswith("/1")


# --- N-23: coverage also validates one record --------------------------------

def test_assert_covers_probes_a_record(tmp_path):
    from .test_a4_color_context import _publish, _record, _store
    from q3vl.what.config import GENCTX_MODE_WITH_WHERE

    root = _publish(tmp_path, [_record("s0", GENCTX_MODE_WITH_WHERE)],
                    GENCTX_MODE_WITH_WHERE)
    got = _store(root, GENCTX_MODE_WITH_WHERE).assert_covers(["s0"])
    assert got["probe"]["schema_version"].endswith("/2")
    assert got["probe"]["mode"] == GENCTX_MODE_WITH_WHERE


def test_assert_covers_catches_a_whole_split_in_the_wrong_mode(tmp_path):
    """N-23: before, this surfaced at the first batch instead of at startup."""
    from .test_a4_color_context import _publish, _record, _store
    from q3vl.what.config import GENCTX_MODE_FORCED_COLOR, GENCTX_MODE_WITH_WHERE

    root = _publish(tmp_path, [_record("s0", GENCTX_MODE_WITH_WHERE)],
                    GENCTX_MODE_FORCED_COLOR)
    with pytest.raises(ValueError, match="forced"):
        _store(root, GENCTX_MODE_FORCED_COLOR).assert_covers(["s0"])
