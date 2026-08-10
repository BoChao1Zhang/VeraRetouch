"""Unattended-run defences: read mount, eval failure, PID discipline.

All three exist because the failure they prevent is invisible while it happens:
a hard-mount read parks the process in D state forever, an eval exception used to
kill ten hours of training, and a wrapper PID reports "dead" while the arm is
still running.
"""

from __future__ import annotations

import json
import os
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


# --- W-B1: the streamed bytes, not just the root constants ------------------
#
# D-B17 moved the root *constants* to /mnt/nfs-ro, but the frozen split indexes
# record ABSOLUTE shard paths under /mnt/nfs, so every record and every image --
# 100% of the training loop's IO -- kept opening the hard mount while the
# startup assertion (which probed only the moved constants) reported green.


def _mirror(monkeypatch, tmp_path):
    """Stand in for the two mounts: <tmp>/nfs (hard, absent) -> <tmp>/nfs-ro."""
    from q3vl.train import shards

    hard, ro = tmp_path / "nfs", tmp_path / "nfs-ro"
    monkeypatch.setattr(shards, "READ_PREFIX_REWRITE",
                        ((f"{hard}/", f"{ro}/"),))
    monkeypatch.setattr(shards, "_ro_mounted", True)
    return hard, ro


def test_shardstore_opens_the_read_mirror_for_a_baked_absolute_path(monkeypatch, tmp_path):
    """The index says /mnt/nfs/...; the file only exists on the mirror."""
    from q3vl.train.shards import MemberRef, ShardStore

    hard, ro = _mirror(monkeypatch, tmp_path)
    rel = "bc/data/datasets/sft2seg-20260804/images/shards/shard-00005.tar"
    (ro / rel).parent.mkdir(parents=True)
    (ro / rel).write_bytes(b"\0" * 512 + b"PAYLOAD")
    assert not (hard / rel).exists(), "the hard-mount path must not exist here"

    store = ShardStore("/", verify="none")
    ref = MemberRef(shard=str(hard / rel), member="x.jpg", offset=512,
                    length=7, size=7, checksum=None)
    # Without the rewrite this raises ShardIntegrityError: shard not found.
    assert store.read(ref) == b"PAYLOAD"
    opened = Path(os.readlink(f"/proc/self/fd/{store._fds[ref.shard]}"))
    assert str(opened).startswith(str(ro)), opened
    store.close()


def test_rewrite_maps_the_campaign_mounts():
    from q3vl.train.shards import READ_PREFIX_REWRITE

    assert READ_PREFIX_REWRITE == ((f"{NFS_RW_ROOT}/", f"{NFS_RO_ROOT}/"),)


def test_rewrite_is_inert_without_the_mirror(monkeypatch):
    """A box without /mnt/nfs-ro keeps working exactly as before."""
    from q3vl.train import shards

    monkeypatch.setattr(shards, "_ro_mounted", False)
    p = f"{NFS_RW_ROOT}/bc/data/x.tar"
    assert shards.rewrite_read_path(p) == p


def test_rewrite_leaves_other_paths_alone(monkeypatch):
    from q3vl.train import shards

    monkeypatch.setattr(shards, "_ro_mounted", True)
    for p in ("/home/bc/data/runs/where_b/W03/train.log",
              "/mnt/nfs-ro/bc/already/read.tar", "relative/shard.tar"):
        assert shards.rewrite_read_path(p) == p


def test_mask_resolver_reaches_the_catalog_through_the_mirror(monkeypatch, tmp_path):
    """`image.origin.root` is a build path baked into the records; the
    `catalog.sqlite3` probe on it is itself a blocking stat on the hard mount."""
    import sqlite3

    from q3vl.where.maskdata import MaskResolver

    hard, ro = _mirror(monkeypatch, tmp_path)
    root = "bc/data/builds/prod-l1-local17k-0001"
    (ro / root / "indexes").mkdir(parents=True)
    (ro / root / "shards").mkdir(parents=True)
    con = sqlite3.connect(ro / root / "indexes" / "catalog.sqlite3")
    con.execute("create table members (sample_id text, suffix text, shard text, "
                "member text, offset_data int, size int, sha256 text)")
    con.execute("insert into members values (?,?,?,?,?,?,?)",
                ("src-1", ".cgt.png", "shard-00000", "src-1.cgt.png", 512, 7, None))
    con.commit(); con.close()

    r = MaskResolver(verify="none")
    ref = r.resolve({"sample_id": "s1", "source_sample_id": "src-1",
                     "image": {"origin": {"root": str(hard / root)}}})
    assert r.n_catalog_hits == 1, "catalog was not found through the mirror"
    # provenance keeps the logical (hard-mount) root; only the IO moves
    assert ref.root == str(hard / root)

    (ro / root / "shards" / "shard-00000.tar").write_bytes(b"\0" * 512 + b"MASKPNG")
    assert r._store.read(ref.member) == b"MASKPNG"
    r.close()


def test_startup_probes_the_streamed_shards_not_only_the_moved_constants():
    """The indicator must be attached to the thing being protected: the probe
    list has to contain the rewritten records/images shard paths."""
    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_where_b.py").read_text()
    assert "_streamed_shard_paths(" in src
    assert "assert_read_mount(*streamed)" in src
    assert "streamed_shards_rewritten" in src


def test_streamed_shard_paths_rewrites_a_baked_index(monkeypatch, tmp_path):
    import importlib

    from q3vl.train import shards

    run = importlib.import_module("q3vl.whereb.scripts.run_where_b")
    hard, ro = _mirror(monkeypatch, tmp_path)
    splits = tmp_path / "splits"
    splits.mkdir()
    img = f"{hard}/bc/data/datasets/sft2seg-20260804/images/shards/shard-00005.tar"
    rec = f"{hard}/bc/data/datasets/sft2seg-20260804/records/shards/shard-00000.tar"
    (splits / "V_where.index.jsonl").write_text(json.dumps({
        "sample_id": "s1",
        "members": {"image": {"shard": img, "member": "a.jpg", "offset": 0,
                              "length": 1, "size": 1},
                    "record": {"shard": rec, "member": "a.json", "offset": 0,
                               "length": 1, "size": 1}}}) + "\n")
    monkeypatch.setattr(run, "SPLIT_DIR", splits)

    got = [str(p) for p in run._streamed_shard_paths("V_where")]
    assert got == [shards.rewrite_read_path(img), shards.rewrite_read_path(rec)]
    assert all(p.startswith(str(ro)) for p in got), got


# --- N33: `soft` is a mount option, not a substring of the line -------------

def test_soft_option_is_parsed_exactly(monkeypatch, tmp_path):
    """A device or mount point containing "soft" must not pass for `soft`."""
    mounts = tmp_path / "mounts"
    mounts.write_text(
        f"1.2.3.4:/soft-export {NFS_RO_ROOT} nfs ro,hard,vers=3 0 0\n")
    real_open = open

    def fake_open(path, *a, **kw):
        return real_open(mounts if str(path) == "/proc/mounts" else path, *a, **kw)

    monkeypatch.setattr("builtins.open", fake_open)
    with pytest.raises(ReadMountError, match="without `soft`"):
        assert_read_mount(tmp_path)


# --- N34/N35: the report survives a failing final evaluation -----------------

def test_final_evaluation_is_guarded_and_failures_are_reported():
    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_where_b.py").read_text()
    body = src[src.index("state = trainer.train()"):]
    assert "FINAL_EVAL_FAILED.json" in body
    assert '"eval_failures": state.eval_failures' in body
    # arm_*.json must still be written on the failure path
    assert body.index("FINAL_EVAL_FAILED.json") < body.index('f"arm_{args.arm}.json"')
    assert "DO NOT re-run the arm" in body
