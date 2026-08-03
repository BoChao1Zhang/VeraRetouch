#!/usr/bin/env python3
"""T2 selfcheck: three registered checks over the parsed D-CUBE corpus.

1. identity  : identity LUT through both applier paths on the real hald
               train + eval images — max |err| and max ΔE00 must be at
               float32-precision level ("=0 量级", threshold 1e-4).
2. pairpath  : N random parse-ok presets (seed fixed), grid_sample vs
               tetrahedral on the hald train image and a deterministic
               2M-px subsample of the eval image; per-preset diff report.
3. nearident : every parse-ok preset, ΔE00 between its 33^3 table and the
               identity table on all 35,937 grid nodes; presets with
               max ΔE00 < 0.2 over the whole domain go to the cull list.

Also cross-checks our sRGB->Lab (colour, D65) against skimage rgb2lab once.

Usage:
  python3 selfcheck.py --npy-dir DIR --parse-report parse_report.jsonl \
      --hald-dir DIR --out-dir DIR [--n-presets 20] [--seed 0] [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cubelib import (
    apply_lut_grid_sample,
    apply_lut_tetrahedral,
    delta_e00,
    identity_table,
    srgb_to_lab,
)

IDENT_TOL = 1e-4          # "=0 量级" threshold on max ΔE00
NEARIDENT_DE = 0.2        # cull if max ΔE00 over full domain below this


def tiled_de00(a: np.ndarray, b: np.ndarray, rows: int = 256):
    """max/mean ΔE00 over big HxWx3 images without blowing RAM."""
    mx, s, n = 0.0, 0.0, 0
    for y0 in range(0, a.shape[0], rows):
        de = delta_e00(a[y0:y0 + rows], b[y0:y0 + rows])
        mx = max(mx, float(de.max()))
        s += float(de.sum())
        n += de.size
    return mx, s / n


def check_lab_reference() -> dict:
    """colour vs skimage rgb2lab agreement (NOTES.md fact 7).

    Known benign discrepancy: skimage uses the classical rounded D65 white
    (0.95047, 1, 1.08883) while colour derives it from chromaticity
    (0.950456, 1, 1.089058) -> absolute Lab dev up to ~0.021. What matters
    downstream is the ΔE00 between two images computed under either
    toolchain, where the white-point offset largely cancels (<0.01)."""
    import colour
    from skimage.color import rgb2lab

    rng = np.random.default_rng(0)
    rgb = rng.random((64, 64, 3))
    dev = float(np.abs(srgb_to_lab(rgb) - rgb2lab(rgb)).max())
    rgb2 = np.clip(rgb + rng.normal(0, 0.05, rgb.shape), 0, 1)
    de_ours = colour.difference.delta_E(
        srgb_to_lab(rgb), srgb_to_lab(rgb2), method="CIE 2000")
    de_skim = colour.difference.delta_E(
        rgb2lab(rgb), rgb2lab(rgb2), method="CIE 2000")
    de_disc = float(np.abs(de_ours - de_skim).max())
    return {"max_abs_lab_dev_vs_skimage": dev,
            "de00_toolchain_discrepancy_max": de_disc,
            "ok": bool(dev < 0.05 and de_disc < 0.01)}


def check_identity(train: np.ndarray, ev: np.ndarray, device: str) -> dict:
    ident = identity_table(33)
    res = {}
    for name, img in (("train", train), ("eval", ev)):
        gs = apply_lut_grid_sample(ident, img, device=device)
        tt = apply_lut_tetrahedral(ident, img)
        gs_mx, gs_mean = tiled_de00(np.clip(gs, 0, 1), img)
        tt_mx, tt_mean = tiled_de00(np.clip(tt, 0, 1), img)
        res[name] = {
            "grid_sample": {"max_abs": float(np.abs(gs - img).max()),
                            "de00_max": gs_mx, "de00_mean": gs_mean},
            "tetrahedral": {"max_abs": float(np.abs(tt - img).max()),
                            "de00_max": tt_mx, "de00_mean": tt_mean},
        }
    worst = max(res[n][p]["de00_max"] for n in res for p in res[n])
    res["worst_de00"] = worst
    res["pass"] = bool(worst < IDENT_TOL)
    res["threshold"] = IDENT_TOL
    return res


def eval_subsample(ev: np.ndarray, n_px: int = 2 ** 21) -> np.ndarray:
    """Deterministic stride subsample of the eval hald, reshaped to an image."""
    flat = ev.reshape(-1, 3)
    stride = flat.shape[0] // n_px
    sub = flat[::stride][:n_px]
    return sub.reshape(1024, n_px // 1024, 3)


def check_pairpath(npy_dir, ok_ids, train, ev, n_presets, seed, device):
    rng = np.random.default_rng(seed)
    picks = sorted(rng.choice(len(ok_ids), size=n_presets, replace=False).tolist())
    sub_eval = eval_subsample(ev)
    rows = []
    for i in picks:
        pid = ok_ids[i]
        table = np.load(os.path.join(npy_dir, pid + ".npy"))
        rec = {"id": pid}
        for name, img in (("train", train), ("eval_sub", sub_eval)):
            gs = apply_lut_grid_sample(table, img, device=device)
            tt = apply_lut_tetrahedral(table, img)
            diff = np.abs(gs - tt)
            de_mx, de_mean = tiled_de00(np.clip(gs, 0, 1), np.clip(tt, 0, 1))
            mse = float((diff ** 2).mean())
            rec[name] = {
                "max_abs": float(diff.max()),
                "mean_abs": float(diff.mean()),
                "psnr_between_paths": float("inf") if mse == 0
                else 10 * np.log10(1.0 / mse),
                "de00_max": de_mx,
                "de00_mean": de_mean,
            }
        rows.append(rec)
        print(f"  pairpath {pid}: eval de00_mean={rec['eval_sub']['de00_mean']:.4f} "
              f"max={rec['eval_sub']['de00_max']:.3f}", flush=True)
    agg = {
        "n": len(rows),
        "eval_de00_mean_avg": float(np.mean([r["eval_sub"]["de00_mean"] for r in rows])),
        "eval_de00_max_worst": float(np.max([r["eval_sub"]["de00_max"] for r in rows])),
        "eval_psnr_between_paths_min": float(np.min(
            [r["eval_sub"]["psnr_between_paths"] for r in rows])),
    }
    return rows, agg


_NODES_LAB: np.ndarray | None = None
_NPY_DIR: str | None = None


def _ni_init(npy_dir):
    global _NODES_LAB, _NPY_DIR
    _NPY_DIR = npy_dir
    _NODES_LAB = srgb_to_lab(identity_table(33).reshape(-1, 3))


def _ni_worker(pid):
    import colour

    assert _NPY_DIR is not None and _NODES_LAB is not None  # _ni_init 在每个 worker 进程先行执行
    table = np.load(os.path.join(_NPY_DIR, pid + ".npy"))
    lab = srgb_to_lab(np.clip(table.reshape(-1, 3), 0, 1))
    de = colour.difference.delta_E(lab, _NODES_LAB, method="CIE 2000")
    return {"id": pid, "de00_max": float(de.max()),
            "de00_mean": float(de.mean()),
            "de00_p99": float(np.percentile(de, 99)),
            "near_identity": bool(de.max() < NEARIDENT_DE)}


def check_near_identity(npy_dir, ok_ids, workers):
    rows = []
    with mp.Pool(workers, initializer=_ni_init, initargs=(npy_dir,)) as pool:
        for i, r in enumerate(pool.imap(_ni_worker, ok_ids, chunksize=64)):
            rows.append(r)
            if (i + 1) % 1000 == 0:
                print(f"  nearident {i + 1}/{len(ok_ids)}", flush=True)
    culled = [r["id"] for r in rows if r["near_identity"]]
    maxes = np.array([r["de00_max"] for r in rows])
    agg = {
        "checked": len(rows),
        "culled_near_identity": len(culled),
        "threshold_de00": NEARIDENT_DE,
        "de00_max_percentiles": {
            "p1": float(np.percentile(maxes, 1)),
            "p5": float(np.percentile(maxes, 5)),
            "p50": float(np.percentile(maxes, 50)),
            "p99": float(np.percentile(maxes, 99)),
        },
    }
    return rows, culled, agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npy-dir", required=True)
    ap.add_argument("--parse-report", required=True)
    ap.add_argument("--hald-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-presets", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--skip-nearident", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ok_ids = [json.loads(l)["id"] for l in open(args.parse_report)
              if json.loads(l)["ok"]]
    train = np.load(os.path.join(args.hald_dir, "hald_train_1024x2048.npy"))
    ev = np.load(os.path.join(args.hald_dir, "hald_eval_3584x4096.npy"))

    summary = {"n_parse_ok": len(ok_ids), "seed": args.seed}

    print("[1/4] Lab reference cross-check", flush=True)
    summary["lab_reference"] = check_lab_reference()

    print("[2/4] identity LUT both-path check", flush=True)
    summary["identity"] = check_identity(train, ev, args.device)

    print(f"[3/4] pairpath on {args.n_presets} random presets", flush=True)
    rows, agg = check_pairpath(args.npy_dir, ok_ids, train, ev,
                               args.n_presets, args.seed, args.device)
    summary["pairpath"] = agg
    with open(os.path.join(args.out_dir, "pairpath_presets.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    if not args.skip_nearident:
        print("[4/4] near-identity cull over full corpus", flush=True)
        ni_rows, culled, ni_agg = check_near_identity(
            args.npy_dir, ok_ids, args.workers)
        summary["near_identity"] = ni_agg
        with open(os.path.join(args.out_dir, "near_identity_stats.jsonl"), "w") as f:
            for r in ni_rows:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(args.out_dir, "near_identity_cull.txt"), "w") as f:
            f.write("\n".join(culled) + ("\n" if culled else ""))

    with open(os.path.join(args.out_dir, "selfcheck_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
