"""Amendment A-4 -- teacher / generated ``<color>`` context (review NF-1).

The property NF-1 was about is not "the code has a generated branch"; it is that
**the twelve arms train on a 50/50 mixture, select on the generated board, and
can never fall back to the GT span**.  Each of those is tested here on the real
code path, and the store contract is tested against a fake *published* payload so
the schema/mode assertions are exercised rather than described.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from q3vl.what.config import (
    ARM_IDS,
    COLOR_CONTEXT_MAX_TOKENS,
    CONTEXT_GENERATED,
    CONTEXT_GT,
    CONTEXT_MODES,
    GENCTX_MODE_FORCED_COLOR,
    GENCTX_MODE_WITH_WHERE,
    SCHEMA_COLOR_GENCTX,
    SELECTION_CONTEXT,
    TEACHER_FRACTION,
    TrainConfig,
    arm_config,
    genctx_mode_of,
)
from q3vl.what.context import (
    BalancedContextSampler,
    ColorContext,
    context_breakdown,
    generated_color_context,
    gt_color_context,
    iter_modes,
)
from q3vl.what.evaluate import arm_metrics, ceiling_board, context_report, main_board
from q3vl.what.model import WhatModel
from q3vl.what.trainer import WhatTrainer

from .conftest import MockBuilder, MockDataset, small_arm


class FakeTokenizer:
    def __init__(self, n_per_word: int = 1):
        self.n = n_per_word
        self.eos_token_id = 2

    def __call__(self, text: str, add_special_tokens: bool = False):
        return {"input_ids": [7] * (len(text.split()) * self.n)}


# --- the two builders are different functions --------------------------------

def test_the_generated_builder_cannot_reach_the_gt_text():
    import inspect

    params = set(inspect.signature(generated_color_context).parameters)
    assert "color_text" not in params and "sample" not in params
    # and the teacher builder is the only one that takes it
    assert "color_text" in set(inspect.signature(gt_color_context).parameters)


def test_a_generation_without_a_closing_tag_is_marked_never_replaced():
    ctx = generated_color_context("s0", [10, 11, 12], close_id=99)
    assert ctx.mode == CONTEXT_GENERATED
    assert ctx.token_ids == [10, 11, 12]
    assert ctx.format_failure and ctx.stop_reason == "no_close_tag"


def test_the_span_is_cut_at_the_first_close_tag():
    ctx = generated_color_context("s0", [10, 99, 12, 99], close_id=99)
    assert ctx.token_ids == [10, 99] and not ctx.format_failure
    assert ctx.stop_reason == "closed"


def test_an_overlong_closed_span_is_truncated_and_flagged():
    # start past the close id so the only 99 in the sequence is the real one
    ids = list(range(100, 100 + COLOR_CONTEXT_MAX_TOKENS + 50)) + [99]
    ctx = generated_color_context("s0", ids, close_id=99)
    assert len(ctx.token_ids) == COLOR_CONTEXT_MAX_TOKENS
    assert ctx.truncated and ctx.format_failure
    assert ctx.stop_reason == "closed_over_boundary"


def test_generation_stops_at_eos():
    ctx = generated_color_context("s0", [10, 2, 99], close_id=99, eos_id=2)
    assert ctx.token_ids == [10] and ctx.stop_reason == "no_close_tag"


def test_an_empty_generation_is_reported():
    ctx = generated_color_context("s0", [], close_id=99)
    assert ctx.format_failure and ctx.stop_reason == "empty"


def test_a_gt_span_over_the_boundary_raises_instead_of_truncating():
    tok = FakeTokenizer()
    long_text = " ".join(["w"] * (COLOR_CONTEXT_MAX_TOKENS + 1))
    with pytest.raises(ValueError, match="must be re-derived"):
        gt_color_context(tok, "s0", long_text)
    ok = gt_color_context(tok, "s0", "short body")
    assert ok.mode == CONTEXT_GT and not ok.format_failure


def test_a_teacher_context_may_not_carry_a_generation_mode():
    with pytest.raises(ValueError):
        ColorContext(mode=CONTEXT_GT, token_ids=[1],
                     genctx_mode=GENCTX_MODE_WITH_WHERE)
    with pytest.raises(ValueError):
        ColorContext(mode="teacher", token_ids=[1])
    with pytest.raises(ValueError):
        ColorContext(mode=CONTEXT_GENERATED, token_ids=[1], genctx_mode="nonsense")


# --- the 50/50 ratio ---------------------------------------------------------

@pytest.mark.parametrize("micro", [2, 4, 8, 32])
def test_every_micro_batch_is_exactly_half_teacher(micro):
    sampler = BalancedContextSampler(256, micro, seed=0,
                                     teacher_fraction=TEACHER_FRACTION)
    batches = list(sampler)
    assert batches
    for b in batches:
        assert len(b) == micro
        assert sum(1 for _, m in b if m == CONTEXT_GT) == micro // 2


def test_an_odd_micro_batch_is_rejected():
    with pytest.raises(ValueError, match="50/50"):
        BalancedContextSampler(64, 3, seed=0)


def test_the_two_stages_agree_on_the_mode_strings():
    """``iter_modes`` asserts it; a rename on either side must be loud."""
    from q3vl.whereb.context import GENERATED as WB_GEN, GT as WB_GT

    assert (WB_GT, WB_GEN) == (CONTEXT_GT, CONTEXT_GENERATED)
    batches = list(iter_modes(BalancedContextSampler(16, 4, seed=0)))
    assert all(m in CONTEXT_MODES for b in batches for _, m in b)


def test_a_sample_is_seen_under_exactly_one_context_per_epoch():
    """The pools are disjoint, so 1 epoch means 1 context per sample."""
    sampler = BalancedContextSampler(64, 4, seed=1)
    assert not set(sampler.teacher_pool) & set(sampler.generated_pool)
    seen: dict[int, set[str]] = {}
    for b in sampler:
        for i, m in b:
            seen.setdefault(i, set()).add(m)
    assert all(len(v) == 1 for v in seen.values())


def test_context_breakdown_reports_the_ratio():
    ctxs = [ColorContext(mode=CONTEXT_GT, token_ids=[1]),
            ColorContext(mode=CONTEXT_GENERATED, token_ids=[1],
                         genctx_mode=GENCTX_MODE_WITH_WHERE, format_failure=True)]
    b = context_breakdown(ctxs)
    assert b["n_gt"] == 1 and b["n_generated"] == 1 and b["teacher_fraction"] == 0.5
    assert b["format_failure_rate_generated"] == 1.0


# --- the forced-prefix semantics for the no-where controls -------------------

def test_the_control_arms_use_the_forced_color_prefix_generation():
    assert genctx_mode_of("C01") == GENCTX_MODE_FORCED_COLOR
    assert genctx_mode_of("C02") == GENCTX_MODE_FORCED_COLOR
    for arm in ARM_IDS:
        if arm not in ("C01", "C02"):
            assert genctx_mode_of(arm) == GENCTX_MODE_WITH_WHERE, arm


def test_the_generation_mode_tracks_the_where_prefix_not_the_arm_name():
    """The rule is derived from ``where_prefix``, so a future no-where arm gets
    the right generation without anyone remembering to add it to a list."""
    for arm in ARM_IDS:
        cfg = arm_config(arm)
        expect = (GENCTX_MODE_WITH_WHERE if cfg.where_prefix
                  else GENCTX_MODE_FORCED_COLOR)
        assert cfg.genctx_mode == expect, arm


# --- the published store contract --------------------------------------------

def _publish(tmp_path: Path, records, mode: str) -> Path:
    # imported lazily: q3vl.data.shardio imports sqlite3 at module level, and
    # some conda environments on this box cannot load it (libstdc++ mismatch).
    # A module-level import would make the *whole* file uncollectable there,
    # taking 30 unrelated A-4 tests down with the 4 that need published shards.
    try:
        from q3vl.data.shardio import build_from_memory
    except ImportError as exc:                                  # pragma: no cover
        pytest.skip(f"shardio not importable here: {exc}")

    root = tmp_path / mode

    def gen():
        for r in records:
            blob = json.dumps(r, ensure_ascii=False, sort_keys=True).encode() + b"\n"
            yield f"{r['sample_id']}.genwhere.json", blob

    build_from_memory(gen(), root, shard_size_bytes=1 << 20,
                      producer="test", source_label="test")
    return root


def _record(sid: str, mode: str, **kw) -> dict:
    base = {"schema_version": SCHEMA_COLOR_GENCTX, "sample_id": sid, "mode": mode,
            "color_ids": [5, 6, 7], "color_text": "<color>x</color>",
            "color_stop_reason": "closed", "color_format_failure": False,
            "color_truncated": False}
    base.update(kw)
    return base


def _store(root, mode):
    """Imported lazily for the same reason ``_publish`` is: the store's import
    chain reaches ``sqlite3``, which cannot load after torch in some envs."""
    from q3vl.what.stores import ColorGenContextStore

    return ColorGenContextStore(root, mode=mode)


def test_the_store_reads_a_v2_record(tmp_path):
    root = _publish(tmp_path, [_record("s0", GENCTX_MODE_WITH_WHERE)],
                    GENCTX_MODE_WITH_WHERE)
    store = _store(root, GENCTX_MODE_WITH_WHERE)
    assert store.color_ids("s0") == [5, 6, 7]
    assert store.assert_covers(["s0"])["coverage"] == 1.0
    assert store.summary()["mode"] == GENCTX_MODE_WITH_WHERE


def test_the_store_refuses_a_v1_record(tmp_path):
    """A v1 ``genwhere`` record has no ``<color>`` segment; reading one would give
    every sample an empty generated context and nothing would complain."""
    v1 = {"schema_version": "q3vl.where_b.genwhere/1", "sample_id": "s0",
          "where_ids": [1, 2]}
    root = _publish(tmp_path, [v1], GENCTX_MODE_WITH_WHERE)
    store = _store(root, GENCTX_MODE_WITH_WHERE)
    with pytest.raises(KeyError, match="missing"):
        store.record("s0")


def test_the_store_refuses_the_wrong_generation_mode(tmp_path):
    """C01/C02 must not be served a context generated after a ``<where>`` span."""
    root = _publish(tmp_path, [_record("s0", GENCTX_MODE_WITH_WHERE)],
                    GENCTX_MODE_FORCED_COLOR)
    store = _store(root, GENCTX_MODE_FORCED_COLOR)
    with pytest.raises(ValueError, match="forced"):
        store.record("s0")


def test_the_store_refuses_an_unknown_mode(tmp_path):
    root = _publish(tmp_path, [_record("s0", GENCTX_MODE_WITH_WHERE)], "x")
    with pytest.raises(ValueError, match="unknown genctx mode"):
        _store(root, "not_a_mode")


def test_missing_coverage_is_a_hard_stop_not_a_gt_fallback(tmp_path):
    root = _publish(tmp_path, [_record("s0", GENCTX_MODE_WITH_WHERE)],
                    GENCTX_MODE_WITH_WHERE)
    store = _store(root, GENCTX_MODE_WITH_WHERE)
    with pytest.raises(RuntimeError, match="forbids falling back"):
        store.assert_covers(["s0", "s1", "s2"])


# --- the builder's context branch --------------------------------------------

class _StubStore:
    def __init__(self, mode: str, ids=(5, 6, 99)):
        self.mode = mode
        self.ids = list(ids)

    def record(self, sample_id: str) -> dict:
        return {"sample_id": sample_id, "mode": self.mode,
                "color_ids": self.ids, "color_text": "gen"}

    def summary(self) -> dict:
        return {"mode": self.mode}


def _builder(arm: str, store=None):
    from q3vl.what.data import WhatBatchBuilder

    b = WhatBatchBuilder.__new__(WhatBatchBuilder)
    b.cfg = arm_config(arm)
    b.tokenizer = FakeTokenizer()
    b.color_genctx = store
    b.close_color_id = 99
    b.eos_id = 2
    b.format_stats = {}
    return b


def _sample(sid="s0", text="a b c"):
    return type("S", (), {"sample_id": sid, "color_text": text})()


def test_the_builder_serves_both_contexts_and_records_the_stats():
    b = _builder("T01", _StubStore(GENCTX_MODE_WITH_WHERE))
    gt = b.color_context(_sample(), CONTEXT_GT)
    gen = b.color_context(_sample(), CONTEXT_GENERATED)
    assert gt.mode == CONTEXT_GT and gen.mode == CONTEXT_GENERATED
    assert gt.token_ids != gen.token_ids
    assert gen.genctx_mode == GENCTX_MODE_WITH_WHERE
    assert set(b.format_stats) == {CONTEXT_GT, CONTEXT_GENERATED}


def test_the_builder_refuses_a_generated_context_with_no_store():
    b = _builder("T01", None)
    with pytest.raises(RuntimeError, match="never fall"):
        b.color_context(_sample(), CONTEXT_GENERATED)


def test_the_builder_refuses_a_context_from_the_wrong_generation():
    """A C01 builder handed the with-<where> generation must stop."""
    b = _builder("C01", _StubStore(GENCTX_MODE_WITH_WHERE))
    with pytest.raises(AssertionError, match="needs"):
        b.color_context(_sample(), CONTEXT_GENERATED)
    ok = _builder("C01", _StubStore(GENCTX_MODE_FORCED_COLOR))
    assert ok.color_context(_sample(), CONTEXT_GENERATED).genctx_mode == \
        GENCTX_MODE_FORCED_COLOR


def test_an_unknown_context_mode_is_rejected():
    b = _builder("T01", _StubStore(GENCTX_MODE_WITH_WHERE))
    with pytest.raises(ValueError, match="unknown context mode"):
        b.color_context(_sample(), "teacher_forced")


# --- training uses both, and refuses to train on one -------------------------

def test_the_trainer_refuses_a_builder_with_no_generated_context(tmp_path):
    cfg = small_arm("T01")
    builder = MockBuilder(cfg)
    builder.color_genctx = None
    with pytest.raises(ValueError, match="50% generated"):
        WhatTrainer(WhatModel(cfg), builder, MockDataset(4), cfg,
                    TrainConfig(arm="T01", micro_batch=2, effective_batch=2),
                    run_dir=tmp_path, device="cpu")


def test_a_mock_run_logs_the_ratio_and_both_context_losses(tmp_path):
    torch.manual_seed(0)
    cfg = small_arm("T01")
    tcfg = TrainConfig(arm="T01", micro_batch=2, effective_batch=2,
                       eval_steps=10 ** 9, save_steps=10 ** 9,
                       grad_ratio_every=10 ** 9, style_queue_size=8)
    tr = WhatTrainer(WhatModel(cfg), MockBuilder(cfg), MockDataset(8), cfg, tcfg,
                     run_dir=tmp_path, device="cpu", log_every=10 ** 9,
                     order=[i % 8 for i in range(16)])
    tr.train()
    rows = [json.loads(l) for l in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert rows
    for r in rows:
        assert r["n_gt"] == r["n_generated"] == 1
        assert r["teacher_fraction"] == 0.5
        assert "L_func_ctx_gt" in r and "L_func_ctx_generated" in r
    setup = tr.setup()
    assert setup["teacher_fraction"] == TEACHER_FRACTION
    assert setup["generated_mode"] == CONTEXT_GENERATED
    assert setup["sampler"]["n_teacher_pool"] == setup["sampler"]["n_generated_pool"]


def test_the_mock_builder_actually_changes_the_conditioning():
    cfg = small_arm("T01")
    b = MockBuilder(cfg)
    ds = MockDataset(2)
    gt = b.build([ds[0], ds[1]], [CONTEXT_GT] * 2)
    gen = b.build([ds[0], ds[1]], [CONTEXT_GENERATED] * 2)
    assert not torch.equal(gt.inputs["h_color"], gen.inputs["h_color"])
    assert [c.mode for c in gen.contexts] == [CONTEXT_GENERATED] * 2


# --- evaluation reports both, selection reads generated ----------------------

def _rows(context, de):
    return [{"sample_id": f"s{i}", "render_mode": "local", "lut_de00_mean": de,
             "lut_de00_p90": de, "img_de00_median": de, "bake_mae_mean": 1e-5,
             "bake_err_p99": 1e-5, "bake_non_finite": 0.0, "lut_non_finite": 0.0,
             "img_boundary_de00_median": de}
            for i in range(4)]


def test_arm_metrics_labels_its_context_and_rejects_an_unknown_one():
    m = arm_metrics(_rows("gt", 1.0), arm="T01", step=500, n_trainable=1,
                    context=CONTEXT_GT)
    assert m["context"] == CONTEXT_GT
    with pytest.raises(ValueError, match="unknown context"):
        arm_metrics(_rows("gt", 1.0), arm="T01", step=500, n_trainable=1,
                    context="teacher")


def test_selection_reads_the_generated_board_only():
    """A teacher-context row must not be selectable, however good it looks."""
    rows = [
        {"arm": "T01", "step": 500, "context": CONTEXT_GT, "gate_pass": True,
         "local_image_de00_median": 0.1, "lut_de00_p90": 0.1,
         "boundary_de00_median": 0.1, "n_trainable_params": 1, "latency_ms": 1.0,
         "is_ceiling": False},
        {"arm": "T01", "step": 500, "context": CONTEXT_GENERATED, "gate_pass": True,
         "local_image_de00_median": 2.0, "lut_de00_p90": 2.0,
         "boundary_de00_median": 2.0, "n_trainable_params": 1, "latency_ms": 1.0,
         "is_ceiling": False},
    ]
    board = main_board(rows, split="V_what")
    assert board["context"] == SELECTION_CONTEXT == CONTEXT_GENERATED
    assert board["n_candidates_in_context"] == 1
    assert board["ranked"][0]["local_image_de00_median"] == 2.0
    gt_board = main_board(rows, split="V_what", context=CONTEXT_GT)
    assert gt_board["ranked"][0]["local_image_de00_median"] == 0.1


def test_a_row_without_a_context_label_is_refused():
    rows = [{"arm": "T01", "step": 1, "gate_pass": True,
             "local_image_de00_median": 1.0, "is_ceiling": False}]
    with pytest.raises(ValueError, match="no 'context' field"):
        main_board(rows, split="V_what")


def test_context_report_pairs_the_two_boards_and_computes_the_gap():
    rows = [
        {"arm": "T01", "step": 500, "context": CONTEXT_GT,
         "local_image_de00_median": 1.0},
        {"arm": "T01", "step": 500, "context": CONTEXT_GENERATED,
         "local_image_de00_median": 1.4},
        {"arm": "T05", "step": 500, "context": CONTEXT_GENERATED,
         "local_image_de00_median": 1.1},
    ]
    rep = context_report(rows)
    assert rep["metric"] == "local_image_de00_median"
    assert rep["n_pairs"] == 1
    t01 = [r for r in rep["rows"] if r["arm"] == "T01"][0]
    assert abs(t01["gap"] - 0.4) < 1e-9
    t05 = [r for r in rep["rows"] if r["arm"] == "T05"][0]
    assert t05["gt"] is None and "gap" not in t05


def test_the_ceiling_board_is_also_per_context():
    rows = [{"arm": "C03", "step": 1, "context": CONTEXT_GENERATED,
             "is_ceiling": True, "local_image_de00_median": 0.1},
            {"arm": "C03", "step": 1, "context": CONTEXT_GT,
             "is_ceiling": True, "local_image_de00_median": 0.05}]
    assert len(ceiling_board(rows)) == 1
    assert ceiling_board(rows)[0]["context"] == CONTEXT_GENERATED


# --- environment guard (found while verifying A-4 in the campaign env) -------

def test_every_entry_point_imports_sqlite3_before_torch():
    """``import torch`` poisons ``import sqlite3`` in the campaign env.

    Measured 2026-08-05 on ``/home/bc/envs/q3vl_sft``::

        import sqlite3; import torch   -> fine
        import torch;   import sqlite3 -> ImportError (libstdc++ CXXABI_1.3.15)

    torch loads a libstdc++ that shadows the one ``_sqlite3``'s dependency chain
    needs.  Every published-shard store in this campaign reaches ``sqlite3``
    through ``q3vl.data.shardio``, so a process that imports torch first can
    never open a shard afterwards -- which is exactly what ``run_what.py`` does
    at ``ColorGenContextStore`` construction.  The guard is one import at the top
    of each entry point; this test is what stops someone tidying it away.
    """
    import ast

    from q3vl.what import scripts

    root = Path(scripts.__file__).parent
    for name in ("run_what.py", "pack_gt_luts.py", "make_zgt_center.py"):
        src = (root / name).read_text()
        tree = ast.parse(src)
        order = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                order.extend((a.name.split(".")[0], node.lineno) for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                order.append((node.module.split(".")[0], node.lineno))
        lines = dict()
        for mod, lineno in order:
            lines.setdefault(mod, lineno)
        assert "sqlite3" in lines, f"{name} lost its sqlite3 guard"
        if "torch" in lines:
            assert lines["sqlite3"] < lines["torch"], (
                f"{name}: sqlite3 must be imported before torch "
                f"(sqlite3 at line {lines['sqlite3']}, torch at {lines['torch']})")


def test_the_producer_and_the_consumer_share_one_vocabulary():
    """Amendment A-4's interface with WB-IMPL, asserted rather than assumed.

    The mode strings, the ``<color>`` boundary and the schema id all live on the
    producer (``q3vl.whereb.config``) and are *imported* here.  A test is still
    worth it: it names the four things that must not drift, so a future rename on
    either side fails here with an explanation instead of failing at the first
    real record with a mode mismatch.
    """
    from q3vl.whereb import config as wb

    assert set(wb.GENCTX_MODES) == {GENCTX_MODE_WITH_WHERE, GENCTX_MODE_FORCED_COLOR}
    assert SCHEMA_COLOR_GENCTX == wb.SCHEMA_GENCTX == "q3vl.where_b.genwhere/2"
    assert COLOR_CONTEXT_MAX_TOKENS == wb.COLOR_CONTEXT_MAX_TOKENS == 384


def test_the_consumer_reads_the_fields_the_producer_writes():
    """The record shape, pinned against the producer's own payload builder."""
    # Both sides are read as *text*.  Importing either one pulls in
    # ``q3vl.data.shardio`` -> ``sqlite3``, which cannot load after torch in this
    # environment (see R6); and a contract about which field names appear in a
    # payload does not need the modules to be importable to be checkable.
    import q3vl.whereb as wb_pkg
    import q3vl.what as what_pkg

    producer = (Path(wb_pkg.__file__).parent / "gencontext.py").read_text()
    consumer = (Path(what_pkg.__file__).parent / "stores.py").read_text()

    required = ("sample_id", "schema_version", "mode", "color_ids")
    assert f"REQUIRED_FIELDS = {required}".replace("'", '"') in \
        consumer.replace("'", '"'), "the consumer's REQUIRED_FIELDS moved"
    for field in required:
        assert f'"{field}"' in producer, f"producer does not emit {field!r}"
    for optional in ("color_text", "color_stop_reason", "color_format_failure",
                     "color_truncated"):
        assert f'"{optional}"' in producer, optional
