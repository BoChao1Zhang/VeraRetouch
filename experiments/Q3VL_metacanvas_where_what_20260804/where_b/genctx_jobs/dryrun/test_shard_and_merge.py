#!/usr/bin/env python
"""Dry-run for the genctx sharding + merge path.  CPU only, no GPU, no weights.

What it proves
--------------
1.  the partition is exhaustive, disjoint and balanced, at the real train size;
2.  the wrapper's forced ``--out-root`` / ``--report-dir`` suffixing (two shards
    cannot collide, and eight shard reports cannot overwrite one another);
3.  ``--help`` works on all three entry points and ``--plan-only`` costs no torch;
4.  **end to end**: real ``publish_generated`` writes two disjoint shard
    publications for the real ``V_where`` sample set, ``merge_genctx.py`` merges
    them, and both consumers -- Where-B's ``GenContextStore`` and Stage-What's
    ``ColorGenContextStore`` (through the compat symlink) -- read the result back
    in full;
5.  the merge refuses: a missing shard, a mis-partitioned shard, a wrong ``mode``,
    a mixed checkpoint, and an existing output root.

No GPU is touched: nothing here loads a model.
"""

from __future__ import annotations

import sqlite3  # noqa: F401  (R6 guard: before torch)

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOBS = HERE.parent
REPO = JOBS.parents[3]
sys.path.insert(0, str(JOBS))
sys.path.insert(0, str(REPO))

from genctx_shard import build_inner_argv, shard_indices, shard_tag  # noqa: E402

PY = os.environ.get("PY", sys.executable)
FAILURES: list[str] = []
CHECKS = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""),
          flush=True)
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "TOKENIZERS_PARALLELISM": "false"}
    return subprocess.run(cmd, capture_output=True, text=True, env=env,
                          cwd=str(REPO), timeout=600, **kw)


# ---------------------------------------------------------------- 1. partition
def test_partition() -> None:
    for n_total, n_shards in ((159215, 8), (896, 2), (897, 8), (10, 3), (5, 8), (1, 1)):
        for mode in ("interleave", "contiguous"):
            parts = [shard_indices(n_total, i, n_shards, mode) for i in range(n_shards)]
            flat = [i for p in parts for i in p]
            sizes = [len(p) for p in parts]
            check(f"partition n={n_total} k={n_shards} {mode}: exhaustive+disjoint",
                  sorted(flat) == list(range(n_total)) and len(flat) == n_total,
                  f"sizes={sizes}")
            check(f"partition n={n_total} k={n_shards} {mode}: balanced",
                  max(sizes) - min(sizes) <= 1, f"sizes={sizes}")
    for bad, why in (((10, 0, 0, "interleave"), "shard 0 of 0"),
                     ((10, 3, 3, "interleave"), "shard index == num_shards"),
                     ((10, -1, 4, "interleave"), "negative shard"),
                     ((10, 0, 4, "nope"), "unknown mode")):
        try:
            shard_indices(*bad)
            check(f"partition rejects {why}", False, "no exception")
        except ValueError:
            check(f"partition rejects {why}", True)
    check("shard_tag zero-pads", shard_tag(3, 8) == "shard03of08", shard_tag(3, 8))


# ------------------------------------------------------- 2. forced suffixing
def test_inner_argv() -> None:
    class A:
        split, shard, num_shards = "train", 3, 8
        out_root, report_dir = "/o", "/r"

    argv = build_inner_argv(A, ["--checkpoint", "/c", "--batch-size", "64"])
    check("inner argv suffixes out-root", "/o/shard03of08" in argv, str(argv))
    check("inner argv suffixes report-dir", "/r/shard03of08" in argv, str(argv))
    check("inner argv keeps passthrough", "--checkpoint" in argv and "64" in argv)
    for bad in (["--out-root", "/x"], ["--report-dir=/x"], ["--split", "V_what"]):
        try:
            build_inner_argv(A, bad)
            check(f"inner argv rejects {bad[0]}", False, "no exception")
        except SystemExit:
            check(f"inner argv rejects {bad[0]}", True)
    # the collision the suffixing exists to prevent
    roots = {build_inner_argv(type("A2", (), {**A.__dict__, "shard": i})(),
                              [])[3] for i in range(8)}
    check("eight shards -> eight distinct out-roots", len(roots) == 8, str(sorted(roots)))


# ------------------------------------------------------------------ 3. CLIs
def test_clis() -> None:
    r = run([PY, str(JOBS / "genctx_shard.py"), "--help"])
    check("genctx_shard.py --help", r.returncode == 0, r.stderr[-300:])
    r = run([PY, str(JOBS / "merge_genctx.py"), "--help"])
    check("merge_genctx.py --help", r.returncode == 0, r.stderr[-300:])
    r = run([PY, "-m", "q3vl.whereb.scripts.make_generated_context", "--help"])
    check("producer --help", r.returncode == 0, r.stderr[-300:])
    check("producer exposes no shard flag (why this wrapper exists)",
          "--shard" not in r.stdout, "")
    t0 = time.time()
    r = run([PY, str(JOBS / "genctx_shard.py"), "--split", "train", "--shard", "1",
             "--num-shards", "8", "--out-root", "/o", "--report-dir", "/r",
             "--plan-only", "--checkpoint", "/c", "--batch-size", "64"])
    dt = time.time() - t0
    ok = r.returncode == 0 and "/o/shard01of08" in r.stdout
    check("--plan-only prints the plan without torch", ok and dt < 8,
          f"rc={r.returncode} {dt:.1f}s {r.stdout[:160]}")
    r = run(["bash", str(JOBS / "genctx_dual.sh")])
    check("genctx_dual.sh without a checkpoint exits 2", r.returncode == 2,
          f"rc={r.returncode}")


# ------------------------------------------------------- 4/5. end-to-end merge
def fake_record(sid: str, split: str, mode: str, ckpt: str) -> dict:
    where = [151669, 1, 2, 151670] if mode == "two_segment" else []
    color = [151671, 3, 4, 151672]
    return {
        "schema_version": "q3vl.where_b.genwhere/2",
        "sample_id": sid, "split": split, "build": "l1", "render_mode": "local",
        "winner_confidence": "normal", "checkpoint": ckpt,
        "generated_ids": where + color, "generated_text": "<where>x</where><color>y</color>",
        "n_generated_tokens": len(where) + len(color),
        "where_ids": where, "where_text": "<where>x</where>",
        "format_failure": False, "truncated": False,
        "stop_reason": "closed" if mode == "two_segment" else "suppressed_by_mode",
        "starts_with_where_open": mode == "two_segment",
        "mode": mode, "where_suppressed": mode == "forced_color",
        "color_ids": color, "color_text": "<color>y</color>",
        "color_format_failure": False, "color_truncated": False,
        "color_stop_reason": "closed", "starts_with_color_open": True,
        "segments_overlap": False,
        "gen": {"max_new_tokens": 512, "mode": mode},
    }


def publish_fake_shard(root: Path, ids: list[str], split: str, mode: str,
                       ckpt: str) -> None:
    from q3vl.whereb.gencontext import publish_generated

    publish_generated((fake_record(s, split, mode, ckpt) for s in ids), root, split)


def test_end_to_end(tmp: Path) -> None:
    from q3vl.whereb.data import open_dataset
    from q3vl.whereb.stores import GenContextStore
    from q3vl.what.stores import ColorGenContextStore

    split, k = "V_where", 2
    ds, _ = open_dataset(split, need_mask=False)
    ids = [r.sample_id for r in ds.refs]
    check(f"{split} opens through the sanctioned factory", len(ids) > 0, f"n={len(ids)}")

    stage = tmp / "_shards" / "two_segment" / split
    for i in range(k):
        part = [ids[j] for j in shard_indices(len(ids), i, k, "interleave")]
        publish_fake_shard(stage / shard_tag(i, k) / split, part, split,
                           "two_segment", "/ckpt-4976")
    check("two shard roots published side by side",
          all((stage / shard_tag(i, k) / split / "manifest.json").exists() for i in range(k)))

    out = tmp / "genwhere" / split
    r = run([PY, str(JOBS / "merge_genctx.py"), "--split", split, "--mode", "two_segment",
             "--shard-root", str(stage), "--num-shards", str(k),
             "--out", str(out), "--report", str(tmp / "report.json")])
    check("merge exits 0", r.returncode == 0, (r.stdout + r.stderr)[-600:])
    check("merge prints MERGE-OK", "MERGE-OK" in r.stdout, r.stdout[-200:])

    st = GenContextStore(out)
    got = {sid for sid, suf in st.rows if suf == st.SUFFIX}
    check("Where-B GenContextStore reads every sample back",
          got == set(ids), f"{len(got)}/{len(ids)}")
    check("merged manifest is complete", st.manifest.get("status") == "complete")
    rep = json.loads((tmp / "report.json").read_text())
    check("merge report records exact coverage",
          rep["n_samples"] == len(ids) and rep["coverage"] == 1.0, str(rep["n_samples"]))
    check("merge report pins one checkpoint", rep["checkpoint"] == "/ckpt-4976",
          str(rep["checkpoint"]))
    check("merged summary carries both segments",
          rep["summary"].get("color", {}).get("n") == len(ids), str(rep["summary"])[:200])

    # the CX-1 compat symlink: Stage-What's <root>/<split>/<mode> path
    link = out / "two_segment"
    check("compat symlink two_segment exists", link.is_symlink(), str(link))
    check("compat symlink resolves to the published root",
          os.path.realpath(link) == os.path.realpath(out))
    what_store = ColorGenContextStore(link, mode="two_segment")
    cov = what_store.assert_covers(ids)
    check("Stage-What ColorGenContextStore covers the split through the symlink",
          cov["coverage"] == 1.0 and cov["n_present"] == len(ids), str(cov)[:200])
    check("Stage-What reads color ids", len(what_store.color_ids(ids[0])) == 4)

    # forced_color published next to it + its own symlink
    stage_fc = tmp / "_shards" / "forced_color" / split
    for i in range(k):
        part = [ids[j] for j in shard_indices(len(ids), i, k, "interleave")]
        publish_fake_shard(stage_fc / shard_tag(i, k) / f"{split}-forced_color", part,
                           split, "forced_color", "/ckpt-4976")
    r = run([PY, str(JOBS / "merge_genctx.py"), "--split", split, "--mode", "forced_color",
             "--shard-root", str(stage_fc), "--num-shards", str(k),
             "--out", str(tmp / "genwhere" / f"{split}-forced_color"),
             "--report", str(tmp / "report_fc.json")])
    check("forced_color merge exits 0", r.returncode == 0, (r.stdout + r.stderr)[-500:])
    fc_link = out / "forced_color"
    check("compat symlink forced_color created inside the two_segment root",
          fc_link.is_symlink() and os.path.realpath(fc_link) ==
          os.path.realpath(tmp / "genwhere" / f"{split}-forced_color"), str(fc_link))
    fc_store = ColorGenContextStore(fc_link, mode="forced_color")
    check("forced_color store refuses to be read as two_segment",
          _raises(lambda: ColorGenContextStore(fc_link, mode="two_segment").record(ids[0])))
    check("forced_color store covers the split",
          fc_store.assert_covers(ids)["coverage"] == 1.0)
    check("the two modes are separate roots",
          os.path.realpath(fc_link) != os.path.realpath(link))

    # -- refusals -------------------------------------------------------------
    r = run([PY, str(JOBS / "merge_genctx.py"), "--split", split, "--mode", "two_segment",
             "--shard-root", str(stage), "--num-shards", str(k),
             "--out", str(out), "--report", str(tmp / "x.json")])
    check("merge refuses an existing output root", r.returncode != 0
          and "already exists" in (r.stdout + r.stderr), (r.stdout + r.stderr)[-200:])

    r = run([PY, str(JOBS / "merge_genctx.py"), "--split", split, "--mode", "two_segment",
             "--shard-root", str(stage), "--num-shards", "4",
             "--out", str(tmp / "m2"), "--report", str(tmp / "x.json")])
    check("merge refuses a missing shard", r.returncode != 0, (r.stdout + r.stderr)[-200:])

    r = run([PY, str(JOBS / "merge_genctx.py"), "--split", split, "--mode", "two_segment",
             "--shard-root", str(stage), "--num-shards", str(k),
             "--shard-mode", "contiguous",
             "--out", str(tmp / "m3"), "--report", str(tmp / "x.json")])
    check("merge refuses shards that are not the planned slice",
          r.returncode != 0 and "planned" in (r.stdout + r.stderr),
          (r.stdout + r.stderr)[-260:])

    bad = tmp / "_shards" / "bad" / split
    for i in range(k):
        part = [ids[j] for j in shard_indices(len(ids), i, k, "interleave")]
        publish_fake_shard(bad / shard_tag(i, k) / split, part, split,
                           "two_segment", f"/ckpt-{4976 if i == 0 else 2488}")
    r = run([PY, str(JOBS / "merge_genctx.py"), "--split", split, "--mode", "two_segment",
             "--shard-root", str(bad), "--num-shards", str(k),
             "--out", str(tmp / "m4"), "--report", str(tmp / "x.json")])
    check("merge refuses a split generated by two checkpoints",
          r.returncode != 0 and "more than one checkpoint" in (r.stdout + r.stderr),
          (r.stdout + r.stderr)[-260:])
    check("the rejected merge published nothing", not (tmp / "m4").exists())

    wrongmode = tmp / "_shards" / "wrongmode" / split
    for i in range(k):
        part = [ids[j] for j in shard_indices(len(ids), i, k, "interleave")]
        publish_fake_shard(wrongmode / shard_tag(i, k) / split, part, split,
                           "forced_color", "/ckpt-4976")
    r = run([PY, str(JOBS / "merge_genctx.py"), "--split", split, "--mode", "two_segment",
             "--shard-root", str(wrongmode), "--num-shards", str(k),
             "--out", str(tmp / "m5"), "--report", str(tmp / "x.json")])
    check("merge refuses records whose mode is not the requested one",
          r.returncode != 0 and "mode" in (r.stdout + r.stderr),
          (r.stdout + r.stderr)[-200:])

    r = run([PY, str(JOBS / "merge_genctx.py"), "--split", split, "--mode", "two_segment",
             "--shard-root", str(stage), "--num-shards", str(k), "--check-only",
             "--out", str(tmp / "m6"), "--report", str(tmp / "x.json")])
    check("--check-only validates without publishing",
          r.returncode == 0 and not (tmp / "m6").exists(), (r.stdout + r.stderr)[-200:])


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:  # noqa: BLE001
        return True
    return False


# ------------------------------------------------------------- 6. the driver
def test_driver(tmp: Path) -> None:
    ckpt = tmp / "fake_ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "model-00001-of-00002.safetensors").write_bytes(b"x")
    env = {**os.environ, "DRY_RUN": "1", "MODE": "two_segment",
           "SPLITS": "V_where V_what train", "NUM_SHARDS": "8",
           "GENCTX_DIR": str(tmp / "genwhere_driver"),
           "STAGE_ROOT": str(tmp / "stage"), "LOG_DIR": str(tmp / "dlogs"),
           "REPORT_ROOT": str(tmp / "dreports"), "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run(["bash", str(JOBS / "genctx_dual.sh"), str(ckpt)],
                       capture_output=True, text=True, env=env, timeout=300)
    out = r.stdout + r.stderr
    check("driver DRY_RUN exits 0", r.returncode == 0, out[-500:])
    check("driver refuses to start a process in DRY_RUN",
          "no process started" in out, out[-200:])
    check("driver plans 12 shard jobs (2+2+8) over 2 cards",
          "PLAN: 12 shard jobs over 2 card(s)" in out,
          [l for l in out.splitlines() if "PLAN" in l][:2])
    check("driver counts the real split sizes",
          "split train: 159215 samples, 8 shards" in out,
          [l for l in out.splitlines() if "split train" in l][:1])
    check("driver puts the small splits first on both cards",
          "GPU 0 <- V_where:0:2 V_what:0:2 train:0:8" in out
          and "GPU 1 <- V_where:1:2 V_what:1:2 train:1:8" in out,
          [l for l in out.splitlines() if "GPU " in l and "<-" in l])
    check("driver prints PROBE-OK before planning", "PROBE-OK" in out)

    # A card that is not free must stop a real run.  GPU_FREE_MIB=0 makes the
    # test independent of what the box happens to be doing (`used >= 0` always),
    # and PY is a shim, so even a regression here cannot reach a real GPU.
    shim = tmp / "fake_py.sh"
    shim.write_text('#!/usr/bin/env bash\necho "FAKEPID=$$"\n'
                    'echo \'{"vlm": {"layer": -1}}\'\nsleep 12\nexit 0\n')
    shim.chmod(0o755)
    env2 = {**env, "GPU_FREE_MIB": "0", "DRY_RUN": "0", "DO_MERGE": "0",
            "PY": str(shim)}
    r = subprocess.run(["bash", str(JOBS / "genctx_dual.sh"), str(ckpt)],
                       capture_output=True, text=True, env=env2, timeout=300)
    check("driver refuses to start on a card that is not free", r.returncode != 0
          and "refusing to start" in (r.stdout + r.stderr),
          (r.stdout + r.stderr)[-200:])
    r = subprocess.run(["bash", str(JOBS / "genctx_dual.sh"), str(ckpt)],
                       capture_output=True, text=True, timeout=300,
                       env={**env2, "DRY_RUN": "1"})
    check("driver warns but still plans when DRY_RUN meets a busy card",
          r.returncode == 0 and "DRY_RUN, planning anyway" in (r.stdout + r.stderr),
          (r.stdout + r.stderr)[-200:])

    # a finished split is dropped, not redone
    done = Path(env["GENCTX_DIR"]) / "V_where"
    done.mkdir(parents=True, exist_ok=True)
    (done / "manifest.json").write_text('{"status": "complete"}')
    r = subprocess.run(["bash", str(JOBS / "genctx_dual.sh"), str(ckpt)],
                       capture_output=True, text=True, env=env, timeout=300)
    out = r.stdout + r.stderr
    check("driver drops an already-published split",
          "dropped from this run" in out and "PLAN: 10 shard jobs" in out, out[-400:])

    (done / "manifest.json").write_text('{"status": "partial"}')
    r = subprocess.run(["bash", str(JOBS / "genctx_dual.sh"), str(ckpt)],
                       capture_output=True, text=True, env=env, timeout=300)
    check("driver stops on a final root that is not a complete publication",
          r.returncode != 0 and "NOT A COMPLETE PUBLICATION" in (r.stdout + r.stderr),
          (r.stdout + r.stderr)[-300:])

    # resume: pre-published shards are skipped, and a LIMIT= calibration leftover
    # at the same path is refused instead of being mistaken for the real thing.
    # DRY_RUN is off here, so a broken skip would visibly try to launch python
    # (with an empty CUDA_VISIBLE_DEVICES and a fake checkpoint, so it cannot
    # reach a GPU and dies in seconds).
    stage = Path(env["STAGE_ROOT"]) / "two_segment" / "V_where"
    renv = {**env, "SPLITS": "V_where", "DRY_RUN": "0", "DO_MERGE": "0",
            "FORCE_BUSY_GPU": "1", "PROBE_TIMEOUT": "60", "PY": str(shim),
            "GENCTX_DIR": str(tmp / "genwhere_resume")}
    for i, tag in enumerate(("shard00of02", "shard01of02")):
        d = stage / tag / "V_where"
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text('{"status": "complete", "sample_count": 448}')
    r = subprocess.run(["bash", str(JOBS / "genctx_dual.sh"), str(ckpt)],
                       capture_output=True, text=True, env=renv, timeout=300)
    out = r.stdout + r.stderr
    check("driver skips shards that are already complete",
          r.returncode == 0 and out.count("already complete (448 samples) -- skipped") == 2,
          out[-400:])
    (stage / "shard01of02" / "V_where" / "manifest.json").write_text(
        '{"status": "complete", "sample_count": 128}')
    r = subprocess.run(["bash", str(JOBS / "genctx_dual.sh"), str(ckpt)],
                       capture_output=True, text=True, env=renv, timeout=300)
    out = r.stdout + r.stderr
    check("driver refuses a calibration leftover masquerading as a done shard",
          r.returncode != 0 and "holds 128 samples, not 448" in out, out[-400:])

    # D-20 machinery against a stand-in interpreter: the PID the driver proves
    # alive, kills and records must be the worker process ITSELF, not a wrapping
    # subshell (review nit N7 on run_where_b.sh).  The shim prints its own $$;
    # under a `( ... ) &` wrapper that value is the driver's pid instead.
    senv = {**env, "SPLITS": "V_where", "DRY_RUN": "0", "DO_MERGE": "0",
            "FORCE_BUSY_GPU": "1", "PROBE_TIMEOUT": "120", "PY": str(shim),
            "STAGE_ROOT": str(tmp / "stage_shim"),
            "GENCTX_DIR": str(tmp / "genwhere_shim"), "LOG_DIR": str(tmp / "slogs")}
    r = subprocess.run(["bash", str(JOBS / "genctx_dual.sh"), str(ckpt)],
                       capture_output=True, text=True, env=senv, timeout=300)
    slogs = Path(senv["LOG_DIR"])
    marker = slogs / "two_segment_V_where_shard00of02.job.marker"
    jlog = slogs / "two_segment_V_where_shard00of02.log"
    check("driver runs a shard job and writes its marker",
          r.returncode == 0 and marker.exists() and jlog.exists(),
          (r.stdout + r.stderr)[-400:])
    if marker.exists() and jlog.exists():
        mp = [l.split("=", 1)[1] for l in marker.read_text().splitlines()
              if l.startswith("pid=")][0]
        fp = [l.split("=", 1)[1] for l in jlog.read_text().splitlines()
              if l.startswith("FAKEPID=")][0]
        check("the recorded PID is the worker process, not a wrapper subshell",
              mp == fp, f"marker pid={mp} process $$={fp}")
        check("the shard job records its exit code",
              (slogs / "two_segment_V_where_shard00of02.rc").read_text().strip() == "0")
        check("the marker names checkpoint / batch / out_root",
              all(k in marker.read_text() for k in
                  ("checkpoint=", "batch_size=", "out_root=", "max_new_tokens=")))

    r = subprocess.run(["bash", str(JOBS / "genctx_forced_color_dual.sh"), str(ckpt)],
                       capture_output=True, text=True,
                       env={**env, "SPLITS": "V_what train"}, timeout=300)
    out = r.stdout + r.stderr
    check("forced_color wrapper flips the mode and the split list",
          "MODE=forced_color" in out and "SPLITS='V_what train'" in out, out[:400])
    check("forced_color plans 10 shard jobs", "PLAN: 10 shard jobs" in out,
          [l for l in out.splitlines() if "PLAN" in l][:1])


def main() -> int:
    print(f"repo={REPO}\npython={PY}\n", flush=True)
    tmp = Path(tempfile.mkdtemp(prefix="genctx_dryrun_"))
    try:
        test_partition()
        test_inner_argv()
        test_clis()
        test_end_to_end(tmp)
        test_driver(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed", flush=True)
    for f in FAILURES:
        print(f"  FAILED: {f}", flush=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
