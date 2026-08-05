"""A closed mock loop for ``T01`` (WC-0 + FG48) and ``T08`` (WC-3 + SB48).

What this proves and what it does not:

* it **does** prove that the whole path -- ``H_color`` -> colour connector ->
  ``z_style`` -> WC tokens -> seed blocks -> provisional/shared geometry ->
  aligned pooling -> refinement blocks -> 48 heads -> constrained parameters ->
  ``T_pred`` -> the protocol 9.5 loss -> optimiser -> back -- is connected, finite
  and descends on a target that only ``H_color`` identifies;
* it does **not** prove anything about the real task.  The mock GT is an affine
  LUT and ``H_color`` is a linear embedding of that LUT's own parameters, chosen
  precisely so that "the loss went down" means "the parameter path carries
  signal" rather than "the model learned the dataset mean".

Run at reduced widths (48 slots, 16 colour queries, 2+4 blocks -- every structural
number unchanged) so it is a test rather than a job.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from q3vl.what.config import TrainConfig
from q3vl.what.losses import StyleQueue
from q3vl.what.model import WhatModel
from q3vl.what.trainer import WhatTrainer, build_optimizer, compute_batch, param_groups

from .conftest import MockBuilder, MockDataset, small_arm

#: 8 mock samples at micro-batch 2 = 4 distinct batches, so 4 steps is one pass.
#: Comparing whole passes is the only honest comparison: consecutive steps see
#: *different* batches, and their losses differ by more than any 24-step trend.
N_SAMPLES, MICRO = 8, 2
STEPS_PER_PASS = N_SAMPLES // MICRO
STEPS = 24
#: DECISION (mock only): the protocol 10.4 learning rate is 1e-4 over ~5k steps.
#: With zero-initialised gates and output layers, 24 steps at 1e-4 move the loss
#: by ~0.5%, which is a real but unconvincing signal.  The mock runs at 1e-3 so
#: the descent is unambiguous; the protocol values themselves are asserted by
#: ``test_optimizer_groups_match_protocol_10_4``, which is where they belong.
MOCK_LR = 1e-3


def _run(arm: str, tmp_path: Path, steps: int = STEPS) -> dict:
    torch.manual_seed(0)
    cfg = small_arm(arm)
    model = WhatModel(cfg)
    ds = MockDataset(N_SAMPLES)
    builder = MockBuilder(cfg)
    tcfg = TrainConfig(arm=arm, micro_batch=MICRO, effective_batch=MICRO,
                       eval_steps=10 ** 9, save_steps=10 ** 9, grad_ratio_every=4,
                       style_queue_size=32, backend_lr=MOCK_LR, head_lr=MOCK_LR,
                       geometry_lr=MOCK_LR / 2)
    tr = WhatTrainer(model, builder, ds, cfg, tcfg, run_dir=tmp_path / arm,
                     device="cpu", log_every=10 ** 9,
                     order=[i % N_SAMPLES for i in range(steps * MICRO)])
    state = tr.train()
    rows = [json.loads(l) for l in (tmp_path / arm / "steps.jsonl").read_text().splitlines()]
    return {"cfg": cfg, "model": model, "state": state, "rows": rows,
            "setup": tr.setup()}


def _pass_means(rows, key: str = "loss") -> list[float]:
    return [sum(r[key] for r in rows[i:i + STEPS_PER_PASS]) / STEPS_PER_PASS
            for i in range(0, len(rows) - STEPS_PER_PASS + 1, STEPS_PER_PASS)]


@pytest.mark.parametrize("arm", ["T01", "T08"])
def test_mock_closed_loop_descends_and_stays_finite(arm, tmp_path):
    r = _run(arm, tmp_path)
    rows = r["rows"]
    assert len(rows) >= STEPS - 1
    assert all(torch.isfinite(torch.tensor(x["loss"])) for x in rows)
    assert all(torch.isfinite(torch.tensor(x["grad_norm"])) for x in rows)

    passes = _pass_means(rows)
    assert len(passes) >= 4
    assert passes[-1] < passes[0], (arm, passes)
    # the two supervision terms that carry the target must both improve, not just
    # the anti-collapse regularisers
    assert _pass_means(rows, "L_func")[-1] < _pass_means(rows, "L_func")[0]
    assert _pass_means(rows, "L_hc")[-1] < _pass_means(rows, "L_hc")[0]
    assert _pass_means(rows, "L_style_cos")[-1] < _pass_means(rows, "L_style_cos")[0]

    # the two protocol 9.1 halves are reported separately, every step
    assert all("L_func_uniform" in x and "L_func_natural" in x for x in rows)
    # protocol 9.2's mandatory ratio is present on its schedule and finite
    ratios = [x["grad_ratio_hc_over_func"] for x in rows if "grad_ratio_hc_over_func" in x]
    assert ratios and all(0.0 < v < 1e4 for v in ratios)


@pytest.mark.parametrize("arm", ["T01", "T08"])
def test_mock_loop_keeps_every_parameter_domain_valid(arm, tmp_path):
    r = _run(arm, tmp_path, steps=6)
    cfg, model = r["cfg"], r["model"]
    ds, builder = MockDataset(2), MockBuilder(cfg)
    out = model(**builder.build([ds[0], ds[1]]).inputs)
    rep = out.report()
    assert rep["mu_in_cube"] and rep["sigma_positive"] and rep["spd"]
    assert rep["opacity_range"][0] > 0.0 and rep["opacity_range"][1] < 1.0
    assert rep["all_finite"]
    rows = r["rows"]
    assert all(x["bake_non_finite"] == 0 for x in rows)
    assert all(x["n_active_mean"] >= 0 for x in rows)


def test_the_style_gate_opens_during_training(tmp_path):
    """Zero-init gates must not stay zero: if they did, ``H_color`` could never
    influence anything and every arm would be a constant LUT."""
    r = _run("T01", tmp_path, steps=6)
    gates = r["model"].gate_values()
    assert any(abs(v) > 0.0 for v in gates["color"]["gate_text"])
    assert any(abs(v) > 0.0 for v in gates["backend"]["gate_v"])


def test_optimizer_groups_match_protocol_10_4(tmp_path):
    for arm, has_geometry in (("T01", False), ("T08", True)):
        cfg = small_arm(arm)
        model = WhatModel(cfg)
        tcfg = TrainConfig(arm=arm)
        groups = param_groups(model, tcfg)
        by = {g["name"]: g for g in groups}
        assert any(n.startswith("backend") for n in by)
        assert any(n.startswith("head") for n in by)
        assert ("geometry" in by or "geometry_no_decay" in by) is has_geometry, arm
        if has_geometry:
            geo = by.get("geometry_no_decay") or by["geometry"]
            assert geo["lr"] == tcfg.geometry_lr
            assert geo["weight_decay"] == 0.0          # geometry: no decay
        # no-decay groups really carry zero decay, decayed groups the protocol's
        for g in groups:
            assert g["weight_decay"] in (0.0, tcfg.weight_decay)
            if g["name"].endswith("no_decay"):
                assert g["weight_decay"] == 0.0
        build_optimizer(model, tcfg)                  # must not raise


def test_modln_and_layernorm_are_excluded_from_weight_decay():
    model = WhatModel(small_arm("T01"))
    groups = param_groups(model, TrainConfig())
    decayed = {id(p) for g in groups if g["weight_decay"] > 0 for p in g["params"]}
    for name, p in model.named_parameters():
        if ".mod_" in name or "norm" in name or p.dim() <= 1:
            assert id(p) not in decayed, name


def test_setup_record_is_complete(tmp_path):
    r = _run("T08", tmp_path, steps=2)
    s = r["setup"]
    for key in ("arm", "wc", "generator", "where_source", "model", "train",
                "param_groups", "grad_accum", "total_optimizer_steps",
                "style_queue"):
        assert key in s, key
    assert s["model"]["generator_facts"]["n_out_per_slot"] == 14


def test_compute_batch_is_reproducible_under_a_fixed_seed():
    cfg = small_arm("T01")
    torch.manual_seed(3)
    model = WhatModel(cfg)
    batch = MockBuilder(cfg).build([MockDataset(2)[0], MockDataset(2)[1]])
    c = MockBuilder(cfg).d_func_scale
    a, sa = compute_batch(model, batch, cfg, StyleQueue(8, 1), d_func_scale=c)
    b, sb = compute_batch(model, batch, cfg, StyleQueue(8, 1), d_func_scale=c)
    assert torch.allclose(a, b)
    assert abs(sa["L_func"] - sb["L_func"]) < 1e-9
