"""Review blocker B1 (preflight wiring) and nit N1 (shared hidden-state contract)."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

from q3vl.whereb import config as C
from q3vl.whereb import contracts, preflight as P


# --- N1: one contract module, no per-stage re-declaration ------------------

def test_hidden_state_contract_lives_in_a_shared_module():
    assert contracts.SEGMENT_HIDDEN_LAYER == -1
    assert contracts.SEGMENT_HIDDEN_FINAL_NORM is True
    assert "D-B2" in contracts.SEGMENT_HIDDEN_RULING
    # whereb re-exports rather than re-declaring
    assert C.WHERE_HIDDEN_LAYER is contracts.SEGMENT_HIDDEN_LAYER
    assert C.WHERE_HIDDEN_FINAL_NORM is contracts.SEGMENT_HIDDEN_FINAL_NORM


def test_only_contracts_py_assigns_the_hidden_state_constants():
    """Stage-What must import these, not invent its own (review nit N1)."""
    root = Path(contracts.__file__).parent
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path.name in ("contracts.py",) or "tests" in path.parts:
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for name in ("SEGMENT_HIDDEN_LAYER", "SEGMENT_HIDDEN_FINAL_NORM"):
                if f"{name} =" in stripped and "import" not in stripped:
                    offenders.append(f"{path.name}:{i}")
    assert not offenders, f"hidden-state contract re-declared at {offenders}"


def test_frozen_vlm_defaults_come_from_the_contract():
    from q3vl.whereb.hiddens import FrozenVLM

    sig = inspect.signature(FrozenVLM.__init__)
    assert sig.parameters["layer"].default == contracts.SEGMENT_HIDDEN_LAYER
    assert sig.parameters["final_norm"].default == contracts.SEGMENT_HIDDEN_FINAL_NORM


# --- B1: the preflight can no longer report PASS for checks it never ran ---

def test_required_checks_include_the_two_model_checks():
    assert set(P.MODEL_CHECKS) <= set(P.REQUIRED_CHECKS)
    assert "WB-P7b-hidden-contract" in P.REQUIRED_CHECKS
    assert "WB-P8b-h-where-causal-independence" in P.REQUIRED_CHECKS


def test_a_report_missing_a_required_check_is_not_ok():
    rep = P.PreflightReport()
    for cid in P.REQUIRED_CHECKS:
        if cid in P.MODEL_CHECKS:
            continue
        rep.add(P.Check(cid, "pass"))
    assert rep.missing == list(P.MODEL_CHECKS)
    assert rep.ok is False                     # this was True before the fix
    assert rep.complete is False
    d = rep.to_dict()
    assert d["missing_required"] == list(P.MODEL_CHECKS)


def test_explicit_skips_are_visible_and_block_completeness():
    rep = P.PreflightReport()
    for cid in P.REQUIRED_CHECKS:
        rep.add(P.Check(cid, "skip" if cid in P.MODEL_CHECKS else "pass"))
    assert rep.ok is True                      # nothing failed, nothing missing
    assert rep.complete is False               # but it is not a full preflight
    d = rep.to_dict()
    assert d["n_skip"] == 2 and d["skipped"] == list(P.MODEL_CHECKS)
    assert d["missing_required"] == []


def test_all_pass_is_complete():
    rep = P.PreflightReport()
    for cid in P.REQUIRED_CHECKS:
        rep.add(P.Check(cid, "pass"))
    assert rep.ok and rep.complete


def test_with_model_path_records_a_failure_instead_of_vanishing(tmp_path):
    """A model-check crash must produce fail rows, never a missing row."""
    rep = P.PreflightReport()
    P.run_model_checks(rep, model_dir=Path("/nonexistent-model-dir"),
                       checkpoint=None, device="cpu", dtype="float32",
                       attn="eager", split="V_where", sample_index=0)
    ids = {c.id: c for c in rep.checks}
    assert set(ids) == set(P.MODEL_CHECKS)
    assert all(c.status == "fail" for c in ids.values())
    assert all("error" in c.detail for c in ids.values())


def test_driver_signature_exposes_the_model_check_knobs():
    sig = inspect.signature(P.run_where_b_preflight)
    for name in ("checkpoint", "device", "dtype", "attn", "split", "sample_index"):
        assert name in sig.parameters, name


def test_cpu_driver_marks_the_model_checks_as_skipped(tmp_path):
    out = tmp_path / "pf.json"
    rep = P.run_where_b_preflight(out=out, skip_model=True, arms=("W01",))
    d = json.loads(out.read_text())
    assert d["missing_required"] == []
    assert d["skipped"] == list(P.MODEL_CHECKS)
    assert d["ok"] is True and d["complete"] is False
    assert rep.ids >= set(P.REQUIRED_CHECKS)


def test_context_flow_check_asserts_the_instruction_swap(tmp_path):
    from .conftest import FakeTokenizer

    c = P.check_context_flows(FakeTokenizer())
    assert c.status == "pass", c.message
    assert c.detail["shuffled_swaps_instruction"] is True
    assert c.detail["shuffle_record_requires_instruction"] is True
    assert c.detail["empty_partner_instruction_refused"] is True
