"""Protocol 14 rows and protocol 12.4 selection, as tests rather than as a script."""

from __future__ import annotations

import json

import pytest

from q3vl.what.config import (
    ARM_IDS,
    BAKE_GATE,
    CEILING_ARM_IDS,
    EFFECTIVE_BATCH,
    EPOCHS,
    EVAL_STEPS,
    MAX_GRAD_NORM,
    QUERY_CONNECTOR_BACKEND_LR,
    SAVE_STEPS,
    SHARED_GEOMETRY_LR,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    Z_STYLE_DIM,
    ZGT_DIM,
    arm_config,
)
from q3vl.what.evaluate import ceiling_board, main_board, summarise, write_per_sample
from q3vl.what.preflight import (
    REQUIRED_CHECKS,
    check_arm_matrix,
    check_bake_readback,
    check_gaussian_constraints,
    check_lab_units_and_grad_norms,
    check_no_h_where,
    check_no_target_leak,
    check_param_match,
    check_srht_deterministic,
    check_zero_init_identity,
    run_what_preflight,
)


# --- config consistency ------------------------------------------------------

def test_the_style_code_and_the_target_code_are_one_dimension():
    """``L_style_cos`` compares them directly; different widths is not a config
    choice, it is a crash waiting for the first batch."""
    assert Z_STYLE_DIM == ZGT_DIM == 1024


def test_stage_what_imports_the_hidden_state_contract_and_never_redeclares_it():
    """Ruling D-B2 / review nit N1: ``H_color`` and ``H_where`` are read from the
    same place, and the constants live in ``q3vl.whereb.contracts``.  Where-B has
    a package-wide scan for a re-declaration; this is its Stage-What twin."""
    from pathlib import Path

    from q3vl.whereb import contracts

    from q3vl.what import config as what_config
    from q3vl.what import hiddens as what_hiddens

    assert what_config.COLOR_HIDDEN_LAYER is contracts.SEGMENT_HIDDEN_LAYER
    assert what_config.COLOR_HIDDEN_FINAL_NORM is contracts.SEGMENT_HIDDEN_FINAL_NORM
    sig = __import__("inspect").signature(what_hiddens.WhatVLM.__init__)
    assert sig.parameters["layer"].default == contracts.SEGMENT_HIDDEN_LAYER
    assert sig.parameters["final_norm"].default == contracts.SEGMENT_HIDDEN_FINAL_NORM

    root = Path(what_config.__file__).parent
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if "tests" in path.parts:
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            s = line.strip()
            if s.startswith("#") or "import" in s:
                continue
            for name in ("SEGMENT_HIDDEN_LAYER", "SEGMENT_HIDDEN_FINAL_NORM",
                         "COLOR_HIDDEN_LAYER", "COLOR_HIDDEN_FINAL_NORM"):
                if f"{name} =" in s:
                    offenders.append(f"{path.name}:{i}")
    assert not offenders, f"hidden-state contract re-declared at {offenders}"


def test_optimiser_constants_are_the_protocol_10_4_numbers():
    assert QUERY_CONNECTOR_BACKEND_LR == 1.0e-4
    assert SHARED_GEOMETRY_LR == 5.0e-5
    assert WEIGHT_DECAY == 0.01 and WARMUP_RATIO == 0.03
    assert MAX_GRAD_NORM == 1.0 and EFFECTIVE_BATCH == 32 and EPOCHS == 1.0
    assert EVAL_STEPS == SAVE_STEPS == 500


def test_bake_gate_thresholds_are_the_protocol_12_1_numbers():
    assert dict((k, (op, thr)) for k, op, thr in BAKE_GATE) == {
        "bake_mae_mean": ("<=", 1e-4),
        "bake_err_p99": ("<=", 5e-4),
        "bake_non_finite": ("<=", 0.0),
    }


# --- preflight rows ----------------------------------------------------------

def test_individual_checks_pass():
    for check in (check_no_h_where(), check_no_target_leak(),
                  check_gaussian_constraints(), check_zero_init_identity(),
                  check_lab_units_and_grad_norms(), check_bake_readback(),
                  check_srht_deterministic(), check_param_match(),
                  check_arm_matrix()):
        assert check.status == "pass", (check.id, check.detail)


def test_the_zero_init_row_records_the_2x_measurement():
    d = check_zero_init_identity().detail
    assert d["max_abs_identity_error"] < 1e-4
    # the literal formula really is f(x) = 2x: the mean ratio T(x)/x is 2, and
    # the clamped worst-case error is the 0.5 the cube's ceiling allows
    assert abs(d["literal_formula_unclamped_ratio"] - 2.0) < 1e-3
    assert d["literal_formula_unclamped_max_abs_error"] > 0.9
    assert d["literal_formula_max_abs_error"] >= 0.5


def test_the_bake_row_separates_the_gate_from_the_measurement():
    d = check_bake_readback().detail
    assert d["lattice_readback_max_abs"] < 1e-5
    assert d["affine_readback_max_abs"] < 1e-4
    assert d["gaussian_mixture_p99"] > 0.0             # random params are not free


def test_a_missing_required_row_is_a_failure_not_a_silent_pass():
    from q3vl.what.preflight import Check, PreflightReport

    rep = PreflightReport()
    rep.add(Check("WT-P8-no-h-where", "pass"))
    assert not rep.ok
    assert len(rep.missing) == len(REQUIRED_CHECKS) - 1


def test_full_preflight_without_data_writes_a_report(tmp_path):
    rep = run_what_preflight(tmp_path, with_data=False)
    assert rep.ok
    assert not rep.missing
    d = json.loads((tmp_path / "preflight_what.json").read_text())
    assert d["ok"] and d["n_fail"] == 0
    assert set(d["skipped"]) == {"WT-W1-gt-lut-resolves", "WT-W2-lut-unseen-disjoint"}


def test_the_structural_scan_would_catch_a_real_leak(tmp_path, monkeypatch):
    """The identifier scan must fire on a genuine ``where`` identifier and not on
    ``torch.where``; both halves are exercised."""
    from q3vl.what import preflight as pf

    src = tmp_path / "fake_color.py"
    src.write_text("import torch\n\ndef f(x):\n    return torch.where(x > 0, x, x)\n")
    assert not any("where" in n and n != "torch.where"
                   for n in pf._code_identifiers(src))
    src.write_text("def f(h_where):\n    return h_where\n")
    assert any("where" in n for n in pf._code_identifiers(src))


# --- protocol 12.4 selection -------------------------------------------------

def _row(arm, step, de, p90, ceiling=False, gate=True):
    return {"arm": arm, "step": step, "local_image_de00_median": de,
            "lut_de00_p90": p90, "boundary_de00_median": de,
            "n_trainable_params": 1, "latency_ms": 1.0,
            "is_ceiling": ceiling, "gate_pass": gate}


def test_main_board_excludes_the_ceiling_arms():
    rows = [_row("T01", 500, 3.0, 5.0), _row("T05", 500, 2.0, 4.0),
            _row("C03", 500, 0.1, 0.2, ceiling=True)]
    board = main_board(rows, split="V_what")
    assert [r["arm"] for r in board["ranked"]] == ["T05", "T01"]
    assert all(r["arm"] != "C03" for r in board["ranked"])
    assert [c["arm"] for c in ceiling_board(rows)] == ["C03"]


def test_main_board_keeps_one_checkpoint_per_arm():
    rows = [_row("T01", 500, 3.0, 5.0), _row("T01", 1000, 2.5, 4.0),
            _row("T05", 500, 2.9, 4.5)]
    board = main_board(rows, split="V_what")
    assert board["n_arms"] == 2
    assert [r["step"] for r in board["ranked"] if r["arm"] == "T01"] == [1000]
    assert board["top2_distinct_arms"]


def test_selection_refuses_the_test_splits():
    for split in ("T_final", "T_lut_unseen", "V_where"):
        with pytest.raises(PermissionError):
            main_board([_row("T01", 1, 1.0, 1.0)], split=split)


def test_gate_failure_is_carried_into_the_board():
    rows = [_row("T01", 1, 1.0, 1.0, gate=False), _row("T05", 1, 2.0, 2.0)]
    board = main_board(rows, split="V_what")
    assert board["any_gate_failed"]


def test_summarise_and_per_sample_writer(tmp_path):
    rows = [{"sample_id": f"s{i}", "lut_mae": 0.1 * i, "render_mode": "local"}
            for i in range(5)]
    s = summarise(rows)
    assert s["n"] == 5 and abs(s["lut_mae_mean"] - 0.2) < 1e-6
    p = write_per_sample(rows, tmp_path / "per_sample.jsonl")
    assert len(p.read_text().strip().splitlines()) == 5


def test_ceiling_arms_are_exactly_the_protocol_8_2_pair():
    assert CEILING_ARM_IDS == ("C03", "C04")
    assert [arm_config(a).is_ceiling for a in ARM_IDS].count(True) == 2
