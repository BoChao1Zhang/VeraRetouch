"""Assemble the PR-AMORT delivery folders from the run directories.

The result reviewer reads **only** the delivery folder, so anything ambiguous in
the run directory has to be resolved here rather than explained in prose.

Two things this does beyond copying:

* **Splits `steps.jsonl` at a run seam.**  Early runs were opened in append
  mode, so a directory can hold one file containing two runs with restarting
  step numbers -- `head` shows the dead run, `tail` the live one, and nothing in
  the file marks the join.  The live run is written as `steps.jsonl` and the
  whole original is preserved as `steps.raw.jsonl`; neither is deleted.
* **Records the run/eval provenance** next to the numbers, so the board cannot
  be read without its convention (matched-area top-k IoU, normal-only).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def split_seam(src: Path, dst_dir: Path) -> dict:
    """Write only the last run's rows to `steps.jsonl`, keep the raw file too."""
    if not src.exists():
        return {"present": False}
    rows = []
    for line in src.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    seam, prev = 0, 0
    for n, r in enumerate(rows):
        if r.get("step", 0) <= prev:
            seam = n
        prev = r.get("step", 0)
    live = rows[seam:]
    (dst_dir / "steps.jsonl").write_text(
        "\n".join(json.dumps(r) for r in live) + "\n", encoding="utf-8")
    shutil.copy2(src, dst_dir / "steps.raw.jsonl")
    return {"present": True, "total_rows": len(rows), "seam_index": seam,
            "live_rows": len(live),
            "note": ("steps.jsonl holds the LIVE run only; steps.raw.jsonl is the "
                     "unmodified file, which also contains an earlier aborted run"
                     if seam else "single run, no seam")}


def deliver(run_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config").mkdir(exist_ok=True)
    (out_dir / "viz").mkdir(exist_ok=True)
    info: dict = {"run_dir": str(run_dir)}

    fin = run_dir / "eval_final" / "metrics.json"
    if fin.exists():
        shutil.copy2(fin, out_dir / "metrics.json")
        info["metrics"] = True
    ps = run_dir / "eval_final" / "per_sample.jsonl"
    if ps.exists():
        shutil.copy2(ps, out_dir / "per_sample.jsonl")
    for name in ("run_setup.json", "loss_preregistration.json"):
        p = run_dir / "config" / name
        if p.exists():
            shutil.copy2(p, out_dir / "config" / name)
    ev = run_dir / "eval.jsonl"
    if ev.exists():
        shutil.copy2(ev, out_dir / "config" / "eval_checkpoints.jsonl")
    info["steps"] = split_seam(run_dir / "steps.jsonl", out_dir / "config")

    viz_src = run_dir / "viz"
    n = 0
    if viz_src.exists():
        for p in sorted(viz_src.glob("*.png")):
            shutil.copy2(p, out_dir / "viz" / p.name)
            n += 1
    info["n_viz"] = n
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="run_dir=out_dir pairs")
    args = ap.parse_args(argv)
    out = {}
    for pair in args.pairs:
        src, dst = pair.split("=", 1)
        out[dst] = deliver(Path(src), Path(dst))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
