#!/usr/bin/env python3
"""F5 sampler: pick 100 (I_in, .cube preset, archived after) pairs from the
D-RENDER journals — 50 from the g-line, 50 from the l-line, every pair a
distinct preset (>=50 distinct presets required by the task card; we enforce
100), covering all 7 completed prod builds.

Output: <OUT_DIR>/manifest.jsonl (+ sampling_log.json with skip reasons).

Usage: /home/bc/miniconda3/bin/python3 sample.py [--seed 20260803]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def provisional_from_build(build: str, rng: random.Random, n_lines: int,
                           oversample: int) -> list[dict]:
    """Random groups from one build -> provisional candidate records
    (one random .cube lut candidate per group)."""
    wanted = set(rng.sample(range(n_lines), min(oversample, n_lines)))
    out = []
    for line_no, g in C.iter_group_lines(build, wanted):
        luts = [c for c in g.get("candidates", [])
                if c.get("format") == "lut"
                and str(c.get("recipe", {}).get("preset_path", "")).endswith(".cube")]
        if not luts:
            continue
        c = rng.choice(luts)
        out.append({
            "build": build,
            "line_no": line_no,
            "group_id": g["group_id"],
            "source_id": g["source_id"],
            "source_path": g["source_path"],
            "pool": C.pool_of(g["source_path"]),
            "render_mode": g.get("render_mode", ""),
            "candidate_id": c["candidate_id"],
            "preset_id": c["recipe"]["preset_id"],
            "preset_path": c["recipe"]["preset_path"],
            "lut_size": (c.get("render_diagnostics") or {}).get("lut_size"),
            "axis_order": (c.get("render_diagnostics") or {}).get("axis_order"),
        })
    rng.shuffle(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--per-line", type=int, default=50)
    ap.add_argument("--oversample", type=int, default=60)
    ap.add_argument("--out-dir", default=C.OUT_DIR)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    resolver = C.BankResolver()
    os.makedirs(args.out_dir, exist_ok=True)

    quotas = {}
    for line, builds in (("g", C.G_BUILDS), ("l", C.L_BUILDS)):
        base, extra = divmod(args.per_line, len(builds))
        for i, b in enumerate(builds):
            quotas[b] = (line, base + (1 if i < extra else 0))

    manifest: list[dict] = []
    used_presets: set[str] = set()
    skip_log: dict[str, int] = {}
    per_build_stats = {}

    for build, (line, quota) in quotas.items():
        n_lines = C.count_lines(build)
        prov = provisional_from_build(build, rng, n_lines, args.oversample)
        # one rg pass over the build indexes for all provisional candidates
        refs = C.find_candidates_in_build(build, [p["candidate_id"] for p in prov])
        taken = 0
        for p in prov:
            if taken >= quota:
                break
            reason = None
            after = cgt = src = None  # 仅类型层面预置；使用点均在赋值分支之后（reason 分支已 continue）
            if p["preset_id"] in used_presets:
                reason = "preset_dup"
            elif p["axis_order"] != "bgr":
                reason = f"axis_order={p['axis_order']}"
            else:
                ref = refs.get(p["candidate_id"], {})
                after = ref.get(".jpg")
                cgt = ref.get(".cgt.png")
                if after is None:
                    reason = "after_not_landed"
                elif line == "l" and cgt is None:
                    reason = "cgt_not_landed"
                else:
                    src = resolver.resolve(p["pool"], p["source_path"])
                    if src is None:
                        reason = f"source_unresolved:{p['pool']}"
            if reason is not None:
                skip_log[reason] = skip_log.get(reason, 0) + 1
                continue
            p2 = dict(p)
            p2["pair_id"] = f"{line}{len(manifest):03d}"
            p2["line"] = line
            p2["after_ref"] = after
            p2["cgt_ref"] = cgt
            p2["source_ref"] = src
            manifest.append(p2)
            used_presets.add(p["preset_id"])
            taken += 1
        per_build_stats[build] = {"quota": quota, "taken": taken,
                                  "provisional": len(prov)}
        if taken < quota:
            print(f"[sample] WARNING {build}: only {taken}/{quota} pairs",
                  file=sys.stderr)

    with open(os.path.join(args.out_dir, "manifest.jsonl"), "w") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log = {
        "seed": args.seed,
        "n_pairs": len(manifest),
        "n_distinct_presets": len(used_presets),
        "n_distinct_sources": len({r["source_id"] for r in manifest}),
        "per_build": per_build_stats,
        "skips": skip_log,
        "lut_sizes": sorted({r["lut_size"] for r in manifest}),
        "pools": sorted({r["pool"] for r in manifest}),
    }
    with open(os.path.join(args.out_dir, "sampling_log.json"), "w") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(json.dumps(log, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
