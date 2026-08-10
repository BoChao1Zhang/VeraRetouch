"""The online (quick) eval: three numbers, two contexts, no gate.

User ruling 2026-08-10: the Where-A oracle is verified (0.97 ceiling), so the
every-500-step reading only has to say how close this checkpoint is to it.  The
full protocol 5.6 board stays exactly as it was and remains what the gate is
decided on -- it just runs once, offline, at selection time.
"""

from __future__ import annotations

import json

import torch

from q3vl.whereb import config as C
from q3vl.whereb.evaluate import (
    QUICK_CONTEXTS,
    QUICK_KEYS,
    evaluate_arm,
    evaluate_context,
)
from q3vl.whereb.metrics import sample_metrics, soft_iou_value, summarise
from q3vl.whereb.model import WhereBModel

from .test_evaluate_and_trainer import FakeBuilder, FakeDataset, _cfg


# --- the new column ---------------------------------------------------------

def test_soft_iou_vs_oracle_is_prediction_against_oracle_not_oracle_quality():
    """`oracle_soft_iou` is the ORACLE's own quality (oracle vs GT).  The
    monitoring reading is a different pair: prediction vs oracle."""
    pred = torch.zeros(8, 8); pred[:5] = 1.0          # over-covers
    orac = torch.zeros(8, 8); orac[:4, :6] = 1.0      # oracle: nearly GT
    gt = torch.zeros(8, 8); gt[:4] = 1.0
    m = sample_metrics(pred, gt, m_oracle=orac)
    assert m["soft_iou_vs_oracle"] == soft_iou_value(pred, orac)
    assert m["oracle_soft_iou"] == soft_iou_value(orac, gt)
    assert m["soft_iou_vs_oracle"] != m["oracle_soft_iou"]


def test_summarise_reports_the_direct_value_next_to_the_ratio():
    rows = [{"render_mode": "local", "soft_iou": 0.8, "oracle_soft_iou": 0.9,
             "soft_iou_vs_oracle": 0.75}]
    s = summarise(rows)
    assert s["soft_iou_vs_oracle"] == 0.75
    assert abs(s["soft_iou_vs_oracle_ratio"] - 0.8 / 0.9) < 1e-9


# --- the quick board --------------------------------------------------------

def test_quick_runs_two_contexts_and_reports_three_numbers(tmp_path):
    cfg = _cfg("W02")
    ds = FakeDataset(6, cfg.readout, n_global=0)
    builder = FakeBuilder(ds)
    rep = evaluate_arm(WhereBModel(cfg), builder, ds, cfg, batch_size=3,
                       out_dir=tmp_path, quick=True)
    assert rep["eval_mode"] == "quick"
    assert set(rep["contexts"]) == set(QUICK_CONTEXTS) == {"generated", "gt"}
    assert [m for call in builder.calls for m in call].count("shuffled") == 0
    for k in QUICK_KEYS:
        assert k in rep and rep[k] is not None, k
    # `best()` ranks on this key; it must survive the simplification
    assert rep["local_soft_iou_median"] == rep["contexts"]["generated"][
        "local_soft_iou_median"]


def test_quick_carries_no_gate_and_no_strata(tmp_path):
    """A progress reading must not be able to pass for a 5.6 decision."""
    cfg = _cfg("W02")
    ds = FakeDataset(4, cfg.readout, n_global=0)
    rep = evaluate_arm(WhereBModel(cfg), FakeBuilder(ds), ds, cfg, batch_size=2,
                       out_dir=tmp_path, quick=True)
    for banned in ("gate", "strata", "antonym_invariance"):
        assert banned not in rep, banned


def test_quick_still_writes_per_sample_rows_with_the_strata_keys(tmp_path):
    """The offline strata tooling (WEVAL-1) is why the online board can afford
    to report only three numbers."""
    cfg = _cfg("W02")
    ds = FakeDataset(4, cfg.readout, n_global=0)
    evaluate_arm(WhereBModel(cfg), FakeBuilder(ds), ds, cfg, batch_size=2,
                 out_dir=tmp_path, quick=True)
    rows = [json.loads(l) for l in
            (tmp_path / "per_sample.jsonl").read_text().splitlines() if l.strip()]
    assert len(rows) == 8                                # 4 samples x 2 contexts
    for r in rows:
        assert r["active_primitive_bucket"] in ("1", "2", ">=3")
        for k in ("soft_iou", "soft_iou_vs_oracle", "grid_hard_iou"):
            assert k in r, k


def test_full_board_is_unchanged(tmp_path):
    """The simplification is additive: the full board still runs all seven
    contexts and still produces the gate."""
    cfg = _cfg("W02")
    ds = FakeDataset(4, cfg.readout, n_global=0)
    rep = evaluate_arm(WhereBModel(cfg), FakeBuilder(ds), ds, cfg, batch_size=2,
                       out_dir=tmp_path, quick=False)
    assert "gate" in rep and "strata" in rep
    from q3vl.whereb.context import CONTEXT_MODES
    assert set(rep["strata"]) == set(CONTEXT_MODES)          # all seven


def test_gate_rows_are_untouched_by_the_split():
    """The ruling changes what runs online, never what the gate means."""
    assert len(C.GATES) == 10
    assert any(k == "grid_boundary_f1_vs_oracle_ratio" for k, _, _ in C.GATES)


# --- the monitoring reading reaches steps.jsonl ------------------------------

def test_eval_summary_row_lands_in_steps_jsonl(tmp_path):
    from q3vl.whereb.config import TrainConfig
    from q3vl.whereb.trainer import WhereBTrainer

    cfg = _cfg("W02")
    ds = FakeDataset(4, cfg.readout, n_global=0)
    tcfg = TrainConfig(arm="W02", micro_batch=2, prefetch_workers=0)
    tr = WhereBTrainer(WhereBModel(cfg), FakeBuilder(ds), ds, cfg, tcfg,
                       run_dir=tmp_path, device="cpu",
                       eval_fn=lambda step: {"eval_mode": "quick",
                                             "soft_iou_vs_oracle": 0.61,
                                             "local_soft_iou_median": 0.55,
                                             "grid_hard_iou": 0.44})
    log = (tmp_path / "steps.jsonl").open("a")
    tr._eval_and_record(log)
    log.close()
    rows = [json.loads(l) for l in
            (tmp_path / "steps.jsonl").read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    assert rows[0]["event"] == "eval"
    assert rows[0]["soft_iou_vs_oracle"] == 0.61
    assert rows[0]["eval_mode"] == "quick"


# --- the entry script defaults ----------------------------------------------

def test_online_eval_defaults_to_quick_and_final_to_full():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_where_b.py").read_text()
    assert '"--online-eval", choices=("quick", "full"), default="quick"' in src
    assert '"--final-eval", choices=("quick", "full"), default="full"' in src
    assert "quick=args.online_eval == \"quick\"" in src
    assert "quick=args.final_eval == \"quick\"" in src
