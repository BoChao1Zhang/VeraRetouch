"""Protocol 14 items 4/5/6 -- the checks themselves, and their failure modes.

The model-dependent checks are exercised by ``preflight.py --device cpu`` on real
data; what is pinned here is the part that must hold whatever the model does:
the data-free checks, and the guarantee that a preflight always leaves evidence
behind.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from q3vl.where.preflight import (
    Check, PreflightReport, check_fpre_invariant_to_sft, check_readout_boundaries,
    check_upsample_order, run_where_a_preflight,
)


def test_readout_boundary_check_passes():
    c = check_readout_boundaries()
    assert c.status == "pass", c.message
    assert c.detail["mu_grid"][0] == -3.0 and c.detail["mu_grid"][-1] == 3.0
    for raw in ("-1000000.0", "0.0", "1000000.0"):
        d = c.detail[raw]
        assert d["h"] > 0
        assert 1.0 <= d["k"] <= 40.0
        assert 0.025 <= d["sigma"] <= 0.30


def test_upsample_order_check_passes_and_proves_the_guard_fired():
    c = check_upsample_order()
    assert c.status == "pass", c.message
    assert c.detail["multichannel_refused"] is True
    assert c.detail["constant_preserved_max_err"] < 1e-6


def test_missing_checkpoint_is_a_skip_not_a_crash():
    c = check_fpre_invariant_to_sft(Path("/nonexistent/model"),
                                    Path("/nonexistent/ckpt"), "cpu")
    assert c.status == "skip"


def test_unloadable_checkpoint_is_a_fail_not_an_exception(tmp_path):
    """N-5b: `load_vision_tower` raises on a layout it does not recognise; that
    must become a `fail` row, never an exception that escapes the driver."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{ not json")
    c = check_fpre_invariant_to_sft(Path("/home/bc/data/models/Qwen3-VL-4B-Instruct"),
                                    ckpt, "cpu")
    assert c.status == "fail"
    assert "error" in c.detail


def test_report_ok_flag_and_counts():
    rep = PreflightReport()
    rep.add(Check("a", "pass"))
    rep.add(Check("b", "skip"))
    assert rep.ok
    d = rep.to_dict()
    assert d["n_pass"] == 1 and d["n_skip"] == 1 and d["n_fail"] == 0
    rep.add(Check("c", "fail", message="boom"))
    assert not rep.ok
    assert rep.to_dict()["n_fail"] == 1


def test_skip_model_run_writes_json_and_lists_every_check(tmp_path):
    out = tmp_path / "pf.json"
    rep = run_where_a_preflight(skip_model=True, out=out)
    assert out.exists()
    data = json.loads(out.read_text())
    ids = {c["id"] for c in data["checks"]}
    for expected in ("WA-P6a-readout-bounds", "WA-P4c-upsample-order",
                     "WA-P4a-fpre-geometry", "WA-P4d-position-encoding",
                     "WA-P5-basis-conditioning", "WA-P4e-highres-path",
                     "WA-P6b-latent-invariants", "WA-P4b-fpre-sft-invariance"):
        assert expected in ids, expected
    assert data["env"]["torch"] == torch.__version__
    assert rep.ok


def test_driver_writes_json_even_when_a_check_explodes(tmp_path, monkeypatch):
    """N-5b again, at the driver level: a preflight that dies without a report is
    indistinguishable from one that was never run."""
    import q3vl.where.preflight as pf

    def boom() -> Check:
        raise RuntimeError("simulated loader failure")

    monkeypatch.setattr(pf, "check_upsample_order", boom)
    out = tmp_path / "pf.json"
    with pytest.raises(RuntimeError, match="simulated"):
        pf.run_where_a_preflight(skip_model=True, out=out)
    assert out.exists(), "the partial report must still be on disk"
    data = json.loads(out.read_text())
    assert data["ok"] is False
    driver = [c for c in data["checks"] if c["id"] == "WA-P0-preflight-driver"]
    assert driver and driver[0]["status"] == "fail"
    assert "simulated loader failure" in driver[0]["detail"]["error"]
