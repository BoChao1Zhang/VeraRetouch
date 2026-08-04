"""Protocol 14 items 7/8/9 as unit tests (the CLI runs the same functions)."""

from __future__ import annotations

import inspect

import torch

from q3vl.whereb.config import ConnectorConfig, arm_config
from q3vl.whereb.data import FORBIDDEN_INPUT_SUBSTRINGS, META_KEYS, Batch
from q3vl.whereb.model import MODEL_INPUT_KEYS, WhereBModel
from q3vl.whereb.preflight import (
    check_context_flows,
    check_no_h_color,
    check_no_target_leak,
    check_parameter_table,
    check_zero_init_gates,
)

from .conftest import make_mock_sample, mock_batch


def _small(arm="W01"):
    cfg = arm_config(arm)
    return WhereBModel(type(cfg)(
        arm=arm, seed=cfg.seed,
        connector=ConnectorConfig(dim=32, n_blocks=2, n_heads=4, ffn=64,
                                  pos_bands=4),
    ))


def test_item7_context_flows(tokenizer):
    c = check_context_flows(tokenizer)
    assert c.status == "pass", c.message
    assert c.detail["generated_unclosed"]["format_failure"] is True
    assert c.detail["sampler"]["n_teacher_pool"] == c.detail["sampler"]["n_generated_pool"]


def test_item8_structural_no_h_color():
    c = check_no_h_color(_small())
    assert c.status == "pass", c.message
    assert all(not v for v in c.detail["identifier_scan"].values()), c.detail
    for params in c.detail["signatures"].values():
        assert not any("color" in p for p in params)


def test_item9_no_target_leak_structural():
    c = check_no_target_leak()
    assert c.status == "pass", c.message
    assert "image" not in META_KEYS          # image.baked is I_tar
    assert set(c.detail["model_input_keys"]) == set(MODEL_INPUT_KEYS)


def test_item9_taint_test_on_a_real_batch():
    cfg = arm_config("W01")
    model = _small("W01")
    samples = [make_mock_sample("a", "band", seed=1),
               make_mock_sample("b", "band", seed=2)]
    batch = mock_batch(samples, "band")
    c = check_no_target_leak(model, batch)
    assert c.status == "pass", c.message
    assert c.detail["taint_test_output_unchanged"] is True


def test_batch_input_whitelist_rejects_a_smuggled_target():
    samples = [make_mock_sample("a", "band", seed=1)]
    batch = mock_batch(samples, "band")
    batch.check_inputs()
    batch.inputs["mask_hi"] = torch.zeros(1)
    try:
        batch.check_inputs()
    except AssertionError as exc:
        assert "14.9" in str(exc)
    else:
        raise AssertionError("a target key passed the whitelist")


def test_forbidden_substrings_cover_i_tar_and_the_baked_render():
    assert "baked" in FORBIDDEN_INPUT_SUBSTRINGS
    assert "i_tar" in FORBIDDEN_INPUT_SUBSTRINGS


def test_zero_init_gate_check_runs_on_the_full_size_arms():
    c = check_zero_init_gates()
    assert c.status == "pass", c.message
    assert c.detail["W01_input_independence_max_diff"] == 0.0
    assert c.detail["W08_input_independence_max_diff"] == 0.0


def test_parameter_table_check_reports_all_eight_arms():
    c = check_parameter_table()
    assert c.status == "pass", c.message
    rows = c.detail["rows"]
    assert len(rows) == 8
    assert {r["arm"] for r in rows} == {f"W0{i}" for i in range(1, 9)}
    for r in rows:
        assert r["n_trainable_params"] > 0
        assert set(r["params_by_group"]) == {"banks", "streams", "heads"}


def test_model_forward_signature_is_exactly_the_whitelist():
    params = list(inspect.signature(WhereBModel.forward).parameters)[1:]
    assert tuple(params) == MODEL_INPUT_KEYS
