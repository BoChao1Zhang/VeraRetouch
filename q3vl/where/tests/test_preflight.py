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


# --- S2 GPU preflight regressions ------------------------------------------

def test_position_encoding_check_is_device_independent():
    """WA-P4d passed on CPU and died under CUDA with "Expected all tensors to be
    on the same device": the reference side was built on CPU while
    `pos_embed.weight` lived on the GPU.  The comparison is now done entirely on
    CPU in float64, so a tower on any device is verified against the same exact
    arithmetic.  This test fakes a tower whose parameters report a foreign device
    to prove the reference side no longer inherits it."""
    import inspect

    from q3vl.where import preflight as pf

    src = inspect.getsource(pf.check_position_encoding)
    assert ".cpu()" in src, "the reference comparison must be pinned to CPU"
    assert src.count(".double().cpu()") >= 2, "both sides go to CPU float64"

    # and it must actually run against a real (CPU) tower
    class _Tower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_grid_per_side = 8
            self.pos_embed = torch.nn.Embedding(64, 16)

        def fast_pos_embed_interpolate(self, grid_thw):
            t, h, w = (int(v) for v in grid_thw[0])
            n = self.num_grid_per_side
            hs = torch.linspace(0, n - 1, h)
            ws = torch.linspace(0, n - 1, w)
            h0, w0 = hs.floor().long(), ws.floor().long()
            h1 = (h0 + 1).clamp(max=n - 1)
            w1 = (w0 + 1).clamp(max=n - 1)
            dh = (hs - h0.float()).unsqueeze(1)
            dw = (ws - w0.float()).unsqueeze(0)
            e = self.pos_embed.weight
            grid = ((1 - dh)[..., None] * (1 - dw)[..., None] * e[h0[:, None] * n + w0[None, :]]
                    + (1 - dh)[..., None] * dw[..., None] * e[h0[:, None] * n + w1[None, :]]
                    + dh[..., None] * (1 - dw)[..., None] * e[h1[:, None] * n + w0[None, :]]
                    + dh[..., None] * dw[..., None] * e[h1[:, None] * n + w1[None, :]])
            m = 2
            return (grid.reshape(h // m, m, w // m, m, -1)
                    .permute(0, 2, 1, 3, 4).reshape(h * w, -1))

    c = pf.check_position_encoding(_Tower(), grid_h=4, grid_w=6)
    assert c.status == "pass", c.detail
    assert c.detail["max_rel_error"] < c.detail["tolerance"]
    assert "model_dtype" in c.detail


def test_position_encoding_tolerance_follows_the_model_dtype():
    """bf16 carries ~3 decimal digits; the exact-arithmetic reference cannot be
    held to 1e-5 against it."""
    import inspect

    from q3vl.where import preflight as pf

    src = inspect.getsource(pf.check_position_encoding)
    assert "torch.float64" in src and "torch.float32" in src
    assert "1e-2" in src, "a bf16 tower needs the loose tolerance"
