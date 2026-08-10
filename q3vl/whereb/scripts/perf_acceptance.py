#!/usr/bin/env python
"""PERF-1 acceptance: did the next arm actually get the fast read path?

Run this **once, 30 minutes after W03 starts**, and again on W04.  It is the
checklist the optimisation was signed off against, so it names the thresholds
rather than leaving them to judgement:

===========================================  =========================  ==========
check                                        threshold                  why
===========================================  =========================  ==========
run_setup.shard_cache.enabled                true                       the arm found the cache
run_setup.shard_cache.n_entries              == 23                      all five sources cached
run_setup.prefetch.workers                   >= 1 (default 6)           supply is overlapped
run_setup.total_optimizer_steps              == 4975                    sampler unchanged
median s/step over the first 30 min          <= 2.8 s  (was 4.40 s)     the actual win
GPU utilisation, 3 min sample                mean >= 65%  (was 17-23%)  the card is fed
steps.jsonl rows                             n == micro_batch, gt+gen   50/50 still holds
===========================================  =========================  ==========

Where the numbers come from (all measured 2026-08-10 on this box, W01/W02
streaming concurrently, ``q3vl.whereb.scripts.bench_supply``):

    before (HEAD code, nfs-ro, serial)            554 ms/micro-batch  2.216 s/step
    fd pool only (cache off, serial)               428 ms/micro-batch  1.711 s/step
    fd pool + prefetch 6 (cache off)                97 ms/micro-batch  0.389 s/step
    local cache, pages cold, prefetch 6            103 ms/micro-batch  0.413 s/step
    local cache preloaded, prefetch 6 (default)     16 ms/micro-batch  0.063 s/step

W01's observed step was 4.40 s median of which 2.2 s was data supply, so the
GPU-side floor is about 2.2 s/step; 2.8 s is that floor plus room for W03's
larger MC16 canvas and for measurement noise.  A median above 3.3 s means the
arm is on the old path -- check ``shard_cache`` in its ``run_setup.json`` first.

    python -m q3vl.whereb.scripts.perf_acceptance --arm W03
    python -m q3vl.whereb.scripts.perf_acceptance --arm W03 --gpu 0 --no-wait
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

DEFAULT_RUN_ROOT = Path("/home/bc/data/runs/where_b")
#: pre-registered, see the table above
MAX_MEDIAN_S_PER_STEP = 2.8
HARD_FAIL_S_PER_STEP = 3.3
MIN_GPU_UTIL_PCT = 65.0
EXPECTED_CACHE_ENTRIES = 23
EXPECTED_TOTAL_STEPS = 4975
BASELINE_S_PER_STEP = 4.40
BASELINE_GPU_UTIL_PCT = 20.0


def _rows(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def step_times(rows: list[dict], window_s: float) -> list[float]:
    """Per-step deltas inside the first ``window_s`` of elapsed time.

    Eval and checkpoint pauses show up as 500 s outliers; the median is the
    statistic of record precisely so they do not have to be special-cased, but
    they are dropped here anyway so a 30-minute window that happens to contain
    an eval is still 30 minutes of training.
    """
    el = [(r["step"], r["elapsed_s"]) for r in rows if "elapsed_s" in r]
    out = []
    for (_s0, t0), (_s1, t1) in zip(el, el[1:]):
        if t1 > window_s:
            break
        d = t1 - t0
        if 0 < d < 60:
            out.append(d)
    return out


def gpu_util(index: int, seconds: float = 180.0, period: float = 2.0) -> dict:
    samples = []
    end = time.time() + seconds
    while time.time() < end:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
                 "-i", str(index)],
                capture_output=True, text=True, timeout=20, check=True).stdout.strip()
            samples.append(float(out.splitlines()[0]))
        except (subprocess.SubprocessError, ValueError, IndexError):
            pass
        time.sleep(period)
    if not samples:
        return {"n": 0, "mean": None, "p10": None, "p90": None}
    s = sorted(samples)
    return {"n": len(s), "mean": round(sum(s) / len(s), 1),
            "p10": s[int(0.1 * len(s))], "p90": s[int(0.9 * len(s))]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="W03")
    ap.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    ap.add_argument("--gpu", type=int, default=None,
                    help="card index to sample; omit to skip the utilisation check")
    ap.add_argument("--window-min", type=float, default=30.0)
    ap.add_argument("--gpu-seconds", type=float, default=180.0)
    ap.add_argument("--no-wait", action="store_true",
                    help="report on whatever is already logged")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_root) / args.arm
    setup_path = run_dir / "run_setup.json"
    steps_path = run_dir / "steps.jsonl"
    window_s = args.window_min * 60

    checks: list[dict] = []

    def check(name, ok, got, want):
        checks.append({"check": name, "pass": bool(ok), "got": got, "want": want})

    setup = json.loads(setup_path.read_text()) if setup_path.exists() else {}
    if not setup:
        check("run_setup.json exists", False, str(setup_path), "present")
    else:
        cache = setup.get("shard_cache") or {}
        check("shard cache enabled", cache.get("enabled") is True,
              cache.get("enabled"), True)
        check("shard cache complete", cache.get("n_entries") == EXPECTED_CACHE_ENTRIES,
              cache.get("n_entries"), EXPECTED_CACHE_ENTRIES)
        pf = (setup.get("prefetch") or {}).get("workers", setup.get("prefetch_workers"))
        check("prefetch workers >= 1", isinstance(pf, int) and pf >= 1, pf, ">= 1")
        check("sampler unchanged", setup.get("total_optimizer_steps") == EXPECTED_TOTAL_STEPS,
              setup.get("total_optimizer_steps"), EXPECTED_TOTAL_STEPS)

    rows = _rows(steps_path)
    if not args.no_wait:
        while rows and rows[-1].get("elapsed_s", 0) < window_s:
            time.sleep(30)
            rows = _rows(steps_path)

    d = step_times(rows, window_s)
    if d:
        med = sorted(d)[len(d) // 2]
        check(f"median s/step over the first {args.window_min:g} min",
              med <= MAX_MEDIAN_S_PER_STEP,
              round(med, 3), f"<= {MAX_MEDIAN_S_PER_STEP} (baseline {BASELINE_S_PER_STEP})")
        if med > HARD_FAIL_S_PER_STEP:
            checks[-1]["note"] = ("above the hard-fail line: the arm is probably on the "
                                  "old read path -- check run_setup.shard_cache")
        check("speedup vs the W01/W02 baseline", med < BASELINE_S_PER_STEP,
              f"{BASELINE_S_PER_STEP / med:.2f}x", "> 1.0x")
    else:
        check("steps.jsonl has timed steps", False, len(rows), ">= 2")

    if rows:
        mb = (setup.get("train") or {}).get("micro_batch")
        check("micro-batch size unchanged", all(r.get("n") == mb for r in rows),
              sorted({r.get("n") for r in rows}), mb)
        check("every micro-batch is 50/50 gt+generated",
              all(set(r.get("by_context", {})) == {"gt", "generated"} for r in rows),
              "ok" if rows else None, "{gt, generated}")

    util = None
    if args.gpu is not None:
        util = gpu_util(args.gpu, args.gpu_seconds)
        check(f"GPU{args.gpu} utilisation over {args.gpu_seconds:g}s",
              util["mean"] is not None and util["mean"] >= MIN_GPU_UTIL_PCT,
              util["mean"], f">= {MIN_GPU_UTIL_PCT} (baseline ~{BASELINE_GPU_UTIL_PCT})")

    width = max(len(c["check"]) for c in checks)
    print(f"\nPERF-1 acceptance -- {args.arm}\n")
    for c in checks:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['check']:<{width}}  "
              f"got={c['got']!r}  want={c['want']!r}"
              + (f"\n         {c['note']}" if c.get("note") else ""))
    failed = [c for c in checks if not c["pass"]]
    print(f"\n  {len(checks) - len(failed)}/{len(checks)} passed\n")
    out = {"arm": args.arm, "checks": checks, "gpu_util": util,
           "n_steps_logged": len(rows)}
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0 if not failed else 1


if __name__ == "__main__":                                     # pragma: no cover
    raise SystemExit(main())
