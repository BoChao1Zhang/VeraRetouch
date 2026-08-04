"""The evaluation pass and the training loop, driven by stubs.

The real :class:`BatchBuilder` needs a frozen 4B VLM; these tests substitute a
builder that hands back synthetic batches so the *loop* logic -- gradient
accumulation, the two-stage schedule, checkpointing, the four-context board and
the strata group-by -- is covered without a GPU.
"""

from __future__ import annotations

import json

import pytest
import torch

from q3vl.whereb.config import ConnectorConfig, TrainConfig, arm_config
from q3vl.whereb.context import ShuffleIndex
from q3vl.whereb.evaluate import evaluate_arm, evaluate_context, strata_report
from q3vl.whereb.model import WhereBModel
from q3vl.whereb.trainer import WhereBTrainer

from .conftest import make_mock_sample, mock_batch


class FakeDataset:
    def __init__(self, n: int, readout: str, n_global: int = 2):
        self.split = "V_where"
        self.samples = []
        for i in range(n):
            s = make_mock_sample(f"sft_{i:03d}", readout, seed=i,
                                 is_global=i >= n - n_global)
            self.samples.append(s)
        self.readout = readout

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


class FakeOracle:
    def __init__(self, ds: FakeDataset):
        self.by_id = {s.sample_id: s.latent for s in ds.samples if not s.is_global}

    def latent(self, sample_id: str, readout: str):
        return self.by_id.get(sample_id)


class FakeBuilder:
    def __init__(self, ds: FakeDataset, shuffle_index=None):
        self.readout = ds.readout
        self.oracle = FakeOracle(ds)
        self.shuffle_index = shuffle_index
        self.basis = type("B", (), {"facts": lambda self: {}, "digest": lambda self: "x"})()
        self.vlm = type("V", (), {"facts": lambda self: {}})()
        self.calls: list[list[str]] = []

    def build(self, samples, modes):
        self.calls.append(list(modes))
        return mock_batch(list(samples), self.readout, modes=list(modes))

    def stats(self):
        return {}


def _cfg(arm="W01"):
    c = arm_config(arm)
    return type(c)(arm=arm, seed=c.seed,
                   connector=ConnectorConfig(dim=32, n_blocks=2, n_heads=4, ffn=64,
                                             pos_bands=4))


# --- evaluation -------------------------------------------------------------

def test_evaluate_context_produces_one_row_per_sample():
    cfg = _cfg()
    ds = FakeDataset(6, cfg.readout)
    rows, summary = evaluate_context(WhereBModel(cfg), FakeBuilder(ds), ds, cfg,
                                     "gt", batch_size=2)
    assert len(rows) == 6
    assert summary["n_local"] == 4 and summary["n_global"] == 2
    assert 0.0 <= summary["local_soft_iou_median"] <= 1.0
    assert all(r["context"] == "gt" for r in rows)
    assert all("oracle_soft_iou" in r for r in rows if r["render_mode"] == "local")


def test_evaluate_context_skips_samples_without_a_shuffle_partner():
    cfg = _cfg()
    ds = FakeDataset(4, cfg.readout, n_global=0)
    records = [{"sample_id": s.sample_id, "source_image_id": "img" if i < 2 else f"solo{i}",
                "render_mode": "local", "where": f"t{i}", "instruction": f"do {i}"}
               for i, s in enumerate(ds.samples)]
    shuf = ShuffleIndex(records, seed=0)
    rows, summary = evaluate_context(WhereBModel(cfg), FakeBuilder(ds, shuf), ds, cfg,
                                     "shuffled", batch_size=4)
    assert len(rows) == 2                          # only the paired image group
    assert summary["n_skipped"] == 2
    assert summary["skipped"][0]["reason"] == "no_shuffle_partner"


def test_evaluate_arm_reports_four_contexts_and_the_gate_board(tmp_path):
    cfg = _cfg()
    ds = FakeDataset(6, cfg.readout)
    records = [{"sample_id": s.sample_id, "source_image_id": "img",
                "render_mode": "global" if s.is_global else "local",
                "where": f"t{i}", "instruction": f"do {i}"}
               for i, s in enumerate(ds.samples)]
    builder = FakeBuilder(ds, ShuffleIndex(records, seed=0))
    m = evaluate_arm(WhereBModel(cfg), builder, ds, cfg, batch_size=3,
                     out_dir=tmp_path)
    assert set(m["per_context"]) == {"gt", "generated", "null", "shuffled"}
    assert m["main_context"] == "generated"
    assert "gt_generated_iou_gap" in m and "instruction_shuffle_iou_drop" in m
    assert "null_context_gap" in m
    assert m["gate"]["n_gates"] == 9
    # ruling D-B1 addendum: the attribution note travels with every board
    att = m["attribution"]
    assert "local_soft_iou_median" in att["directly_optimised"]
    assert "auc_target" in att["not_optimised"]
    assert "boundary_f1" in att["weakly_optimised"]
    # ... and the ready-to-paste REPORT.md section ships next to metrics.json
    section = (tmp_path / "ATTRIBUTION.md").read_text()
    assert "不是独立检验" in section
    assert "PRIMARY attribution" in section
    for key in ("auc_target", "instruction_shuffle_iou_drop", "null_context_gap",
                "boundary_f1", "local_soft_iou_median"):
        assert f"`{key}`" in section
    assert m["arm"] == cfg.arm
    lines = (tmp_path / "per_sample.jsonl").read_text().strip().split("\n")
    assert len(lines) >= 6 * 3
    assert json.loads(lines[0])["sample_id"].startswith("sft_")
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["gate"]["n_gates"] == 9


def test_strata_report_groups_by_every_requested_key():
    rows = [
        {"soft_iou": 0.8, "boundary_f1": 0.7, "auc_target": 0.9, "render_mode": "local",
         "upscaled": True, "winner_confidence": "normal", "build": "l1"},
        {"soft_iou": 0.6, "boundary_f1": 0.5, "auc_target": 0.8, "render_mode": "local",
         "upscaled": False, "winner_confidence": "low", "build": "l2"},
    ]
    rep = strata_report(rows)
    assert set(rep) == {"upscaled", "winner_confidence", "build", "render_mode"}
    assert set(rep["winner_confidence"]) == {"normal", "low"}
    assert rep["upscaled"]["True"]["n"] == 1


# --- the training loop ------------------------------------------------------

def test_trainer_accumulates_gradients_to_the_effective_batch(tmp_path):
    cfg = _cfg()
    ds = FakeDataset(48, cfg.readout, n_global=0)
    builder = FakeBuilder(ds)
    tcfg = TrainConfig(arm=cfg.arm, micro_batch=4, effective_batch=16,
                       eval_steps=2, save_steps=10**9)
    tr = WhereBTrainer(WhereBModel(cfg), builder, ds, cfg, tcfg,
                       run_dir=tmp_path, device="cpu", log_every=10**9)
    assert tr.gas == 4
    setup = tr.setup()
    assert setup["grad_accum"] == 4
    assert setup["total_optimizer_steps"] == len(tr.sampler) // 4
    state = tr.train()
    assert state.step == setup["total_optimizer_steps"]
    assert state.micro_step == state.step * 4
    rows = [json.loads(l) for l in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert len(rows) == state.step
    assert all(r["n"] == 4 for r in rows)
    assert all(set(r["by_context"]) == {"gt", "generated"} for r in rows)


def test_trainer_micro_batches_are_all_fifty_fifty(tmp_path):
    cfg = _cfg()
    ds = FakeDataset(32, cfg.readout, n_global=0)
    builder = FakeBuilder(ds)
    tcfg = TrainConfig(arm=cfg.arm, micro_batch=4, effective_batch=8,
                       save_steps=10**9)
    WhereBTrainer(WhereBModel(cfg), builder, ds, cfg, tcfg, run_dir=tmp_path,
                  device="cpu", log_every=10**9).train()
    assert builder.calls
    for modes in builder.calls:
        assert modes.count("gt") == modes.count("generated") == 2


def test_trainer_switches_the_loss_schedule_and_saves(tmp_path):
    cfg = _cfg()
    ds = FakeDataset(80, cfg.readout, n_global=0)
    tcfg = TrainConfig(arm=cfg.arm, micro_batch=4, effective_batch=8,
                       save_steps=5, eval_steps=10**9)
    tr = WhereBTrainer(WhereBModel(cfg), FakeBuilder(ds), ds, cfg, tcfg,
                       run_dir=tmp_path, device="cpu", log_every=10**9)
    state = tr.train()
    stages = {r["stage"] for r in state.history}
    assert stages == {1, 2}
    boundary = round(0.30 * state.total_steps)
    assert state.history[boundary - 1]["stage"] == 1
    assert state.history[boundary]["stage"] == 2
    assert (tmp_path / "where_b_final.pt").exists()
    ck = torch.load(tmp_path / "where_b_final.pt", weights_only=False)
    assert ck["arm"] == cfg.arm and ck["step"] == state.step


def test_checkpoint_selection_uses_a_metric_not_val_loss(tmp_path):
    cfg = _cfg()
    ds = FakeDataset(16, cfg.readout, n_global=0)
    tcfg = TrainConfig(arm=cfg.arm, micro_batch=4, effective_batch=8,
                       eval_steps=1, save_steps=10**9)
    scores = iter([0.4, 0.9, 0.6, 0.5, 0.3, 0.2])
    tr = WhereBTrainer(WhereBModel(cfg), FakeBuilder(ds), ds, cfg, tcfg,
                       run_dir=tmp_path, device="cpu", log_every=10**9,
                       eval_fn=lambda step: {"local_soft_iou_median": next(scores),
                                             "eval_loss": -step})
    tr.train()
    best = tr.best()
    assert best is not None
    assert best["local_soft_iou_median"] == 0.9      # not the lowest eval_loss
    assert (tmp_path / "eval.jsonl").exists()


def test_trainer_warmup_and_cosine_touch_the_protocol_endpoints(tmp_path):
    cfg = _cfg()
    ds = FakeDataset(120, cfg.readout, n_global=0)
    tcfg = TrainConfig(arm=cfg.arm, micro_batch=4, effective_batch=8,
                       save_steps=10**9)
    tr = WhereBTrainer(WhereBModel(cfg), FakeBuilder(ds), ds, cfg, tcfg,
                       run_dir=tmp_path, device="cpu", log_every=10**9)
    state = tr.train()
    lrs = [r["lr"] for r in state.history]
    assert lrs[0] <= tcfg.learning_rate
    assert max(lrs) <= tcfg.learning_rate + 1e-12
    assert lrs[-1] < lrs[len(lrs) // 2]              # cosine decays
    assert lrs[-1] < 1e-5


def test_odd_micro_batch_is_rejected_by_the_trainer(tmp_path):
    cfg = _cfg()
    ds = FakeDataset(16, cfg.readout, n_global=0)
    with pytest.raises(ValueError, match="50/50"):
        WhereBTrainer(WhereBModel(cfg), FakeBuilder(ds), ds, cfg,
                      TrainConfig(arm=cfg.arm, micro_batch=3, effective_batch=9),
                      run_dir=tmp_path, device="cpu")
