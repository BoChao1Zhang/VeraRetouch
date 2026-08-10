"""Unattended-run defences: read mount, eval failure, PID discipline.

All three exist because the failure they prevent is invisible while it happens:
a hard-mount read parks the process in D state forever, an eval exception used to
kill ten hours of training, and a wrapper PID reports "dead" while the arm is
still running.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch

from q3vl.where.config import (
    NFS_RO_ROOT,
    NFS_RW_ROOT,
    ReadMountError,
    assert_read_mount,
)
from q3vl.whereb import config as C
from q3vl.whereb.config import TrainConfig, arm_config, ConnectorConfig
from q3vl.whereb.model import WhereBModel
from q3vl.whereb.trainer import WhereBTrainer

from .test_evaluate_and_trainer import FakeBuilder, FakeDataset, _cfg


# --- D-B17: reads go through the soft mount ---------------------------------

def test_read_roots_are_on_the_soft_read_mount():
    """Every path an unattended arm streams from must be under /mnt/nfs-ro."""
    from q3vl.data import config as dcfg
    from q3vl.where import config as wcfg

    for name, p in (("data.BUILD_ROOT", dcfg.BUILD_ROOT),
                    ("data.DATASET_ROOT", dcfg.DATASET_ROOT),
                    ("where.SFT2SEG_ROOT", wcfg.SFT2SEG_ROOT),
                    ("where.SPLIT_DIR", wcfg.SPLIT_DIR),
                    ("where.BUILD_DATASET_ROOT", wcfg.BUILD_DATASET_ROOT),
                    ("where.MASKVIEW_READ_DIR", wcfg.MASKVIEW_READ_DIR),
                    ("where.ORACLE_READ_DIR", wcfg.ORACLE_READ_DIR),
                    ("where.BASIS_READ_DIR", wcfg.BASIS_READ_DIR),
                    ("whereb.GENCTX_DIR", C.GENCTX_DIR)):
        assert str(p).startswith(str(NFS_RO_ROOT)), f"{name} reads from {p}"


def test_whereb_consumes_the_read_mirrors():
    """The re-exports Where-B actually opens must be the READ ones."""
    for p in (C.WHERE_A_MASKVIEW_DIR, C.WHERE_A_ORACLE_DIR, C.WHERE_A_BASIS_DIR):
        assert str(p).startswith(str(NFS_RO_ROOT)), p


def test_write_roots_stay_on_the_rw_mount():
    """Publication still goes to the hard mount (through nfsx); only reads moved."""
    from q3vl.data import config as dcfg
    from q3vl.where import config as wcfg

    for name, p in (("data.OUT_ROOT", dcfg.OUT_ROOT),
                    ("where.WHERE_A_ROOT", wcfg.WHERE_A_ROOT),
                    ("where.MASKVIEW_DIR", wcfg.MASKVIEW_DIR),
                    ("where.ORACLE_DIR", wcfg.ORACLE_DIR),
                    ("where.BASIS_DIR", wcfg.BASIS_DIR),
                    ("whereb.WHERE_B_ROOT", C.WHERE_B_ROOT),
                    ("whereb.GENCTX_WRITE_DIR", C.GENCTX_WRITE_DIR)):
        assert str(p).startswith(str(NFS_RW_ROOT)), f"{name} writes to {p}"
        assert not str(p).startswith(str(NFS_RO_ROOT)), f"{name} would EROFS"


def test_the_producer_writes_to_the_rw_root():
    src = (Path(__file__).resolve().parents[1] / "scripts"
           / "make_generated_context.py").read_text()
    assert "GENCTX_WRITE_DIR" in src and "default=str(GENCTX_DIR)" not in src


def test_assert_read_mount_rejects_a_missing_mount(monkeypatch, tmp_path):
    fake = tmp_path / "proc_mounts"
    fake.write_text("172.25.76.194:/rwq /mnt/nfs nfs4 rw,hard 0 0\n")
    real_open = open

    def fake_open(p, *a, **k):
        return real_open(fake if str(p) == "/proc/mounts" else p, *a, **k)

    monkeypatch.setattr("builtins.open", fake_open)
    with pytest.raises(ReadMountError, match="not mounted"):
        assert_read_mount(tmp_path)


def test_assert_read_mount_rejects_a_hard_read_mount(monkeypatch, tmp_path):
    """A hard read mount defeats the purpose -- it hangs instead of erroring."""
    fake = tmp_path / "proc_mounts"
    fake.write_text("172.25.76.194:/rwq /mnt/nfs-ro nfs ro,hard,vers=3 0 0\n")
    real_open = open

    def fake_open(p, *a, **k):
        return real_open(fake if str(p) == "/proc/mounts" else p, *a, **k)

    monkeypatch.setattr("builtins.open", fake_open)
    with pytest.raises(ReadMountError, match="without `soft`"):
        assert_read_mount(tmp_path)


def test_assert_read_mount_rejects_an_unreachable_deep_path():
    with pytest.raises(ReadMountError):
        assert_read_mount(NFS_RO_ROOT / "definitely" / "not" / "here")


def test_assert_read_mount_cannot_hang(monkeypatch):
    """A wedged stat must abandon the probe, not block the caller forever."""
    import q3vl.where.config as wcfg

    def never_returns(_p):
        import time

        time.sleep(60)

    monkeypatch.setattr(wcfg.os, "stat", never_returns)
    with pytest.raises(ReadMountError, match="did not return within"):
        assert_read_mount(NFS_RO_ROOT, timeout_s=0.5)


def test_live_deep_path_probe_passes():
    """The real thing, on the real mount, at depth."""
    from q3vl.where.config import SPLIT_DIR

    info = assert_read_mount(SPLIT_DIR)
    assert "soft" in info["options"]
    assert info["probed"][str(SPLIT_DIR)] == "ok"


# --- eval must never kill an unattended arm ---------------------------------

def _trainer(tmp_path, eval_fn, n=16):
    cfg = _cfg()
    ds = FakeDataset(n, cfg.readout, n_global=0)
    tcfg = TrainConfig(arm=cfg.arm, micro_batch=4, effective_batch=8,
                       eval_steps=1, save_steps=10**9)
    return WhereBTrainer(WhereBModel(cfg), FakeBuilder(ds), ds, cfg, tcfg,
                         run_dir=tmp_path, device="cpu", log_every=10**9,
                         eval_fn=eval_fn)


def test_training_continues_when_eval_raises(tmp_path):
    calls = {"n": 0}

    def boom(step):
        calls["n"] += 1
        raise RuntimeError("wedged read mount during eval")

    tr = _trainer(tmp_path, boom)
    state = tr.train()                      # must NOT raise
    assert state.step > 0, "training did not progress"
    assert calls["n"] >= 1
    assert len(state.eval_failures) == calls["n"]


def test_eval_failure_writes_a_status_file_with_the_traceback(tmp_path):
    def boom(step):
        raise ValueError("genctx record missing")

    tr = _trainer(tmp_path, boom)
    tr.train()
    markers = sorted(tmp_path.glob("EVAL_FAILED_step*.json"))
    assert markers, "no EVAL_FAILED status file was written"
    d = json.loads(markers[0].read_text())
    assert d["error"].startswith("ValueError")
    assert "Traceback" in d["traceback"] and "genctx record missing" in d["traceback"]
    assert "TRAINING CONTINUED" in d["note"]
    assert "not selectable" in d["note"]


def test_a_failed_eval_leaves_no_selectable_checkpoint(tmp_path):
    """Not silent: the checkpoint simply has no record, so §5.6 cannot pick it."""
    tr = _trainer(tmp_path, lambda step: (_ for _ in ()).throw(RuntimeError("x")))
    tr.train()
    assert tr.state.checkpoints == []
    assert tr.best() is None
    assert not (tmp_path / "eval.jsonl").exists()


def test_a_healthy_eval_still_records_normally(tmp_path):
    tr = _trainer(tmp_path, lambda step: {"local_soft_iou_median": 0.5 + step * 0.01})
    tr.train()
    assert tr.state.eval_failures == []
    assert tr.state.checkpoints and (tmp_path / "eval.jsonl").exists()
    assert tr.best() is not None


def test_operator_interrupts_are_not_swallowed(tmp_path):
    def interrupted(step):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _trainer(tmp_path, interrupted).train()


# --- D-B18: PID discipline in submit ----------------------------------------

def _submit_src() -> str:
    return (Path(__file__).resolve().parents[1] / "scripts"
            / "run_where_b.sh").read_text()


def test_marker_records_the_python_pid_as_authoritative():
    src = _submit_src()
    assert "python_pid=%s" in src and "wrapper_pid=%s" in src
    assert "liveness_check=ps -p" in src


def test_submit_resolves_the_child_and_never_greps_a_pattern():
    """`pgrep -f <pattern>` matches the very shell running the grep (four
    separate incidents).  Scanned over CODE lines only: the script's header
    comment deliberately quotes the banned form to explain the ban, and a whole
    -file scan would flag the explanation as the offence."""
    src = _submit_src()
    assert "pgrep -P" in src, "the python child is not resolved from the wrapper"
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))
    assert not re.search(r"pgrep\s+-[a-zA-Z]*f", code), (
        "pattern-matching pgrep used in an executed line"
    )


def test_liveness_still_uses_ps_p():
    assert 'ps -p "$pid"' in _submit_src()
