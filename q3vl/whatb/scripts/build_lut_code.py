#!/usr/bin/env python
"""C0 -- build the canonical LUT function code (EPR-031 §3.1 / §7-C0).

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/bc/VeraRetouch \\
    /home/bc/envs/q3vl_sft/bin/python -m q3vl.whatb.scripts.build_lut_code \\
        --out /home/bc/data/runs/whatb/lutcode_17c_pca

CPU only.  What it writes into ``--out``:

``code.npz``
    ``c_star`` ``(n_lut, d_keep)`` whitened canonical coordinates, ``lut_ids``,
    ``mean``, ``components`` ``(d_keep, d)``, ``explained_variance``,
    ``explained_variance_ratio`` (**full rank**, so the 90/95/99 % columns can be
    recomputed from the artefact), ``whiten_scale``, ``total_variance``.
``manifest.json``
    grid, n, dims, the sha256 of ``code.npz``, the fit command, the cumulative
    90/95/99 % dimensions, the three ``code_recon_de00`` rows, and the ``9^3``
    control table next to the repository's existing 15 / 28 / 99 ledger.

Both files are written to ``*.tmp`` in the destination directory and then
``os.replace``d, so a killed job leaves the previous artefact intact rather than
half a ``code.npz``.

The ``9^3`` control reproduces the ledger's protocol (2,500 random train index
rows -> unique ``lut_id`` -> PCA on the 2,187-dim space).  The seed of the
original draw was never recorded, so the id count of this re-draw is a measured
number and is reported as such -- **nothing here is tuned to land on 1,137**.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from q3vl.whatb.codec import lutcode as C

__all__ = ["build_arg_parser", "main"]


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="build_lut_code",
        description="EPR-031 C0: whitened PCA of the train LUT residuals")
    ap.add_argument("--out", default=str(C.CODE_DIR),
                    help="artefact directory (code.npz + manifest.json)")
    ap.add_argument("--bank-dir", default=None,
                    help="preset bank; default = q3vl.whatb.lutdata.BANK_DIR")
    ap.add_argument("--split", default="train",
                    help="the split whose unique lut_id form the fit set")
    ap.add_argument("--grid", type=int, default=C.GRID_MAIN,
                    help="main grid side (17 -> 3 * 17^3 = 14,739 dims)")
    ap.add_argument("--control-grid", type=int, default=C.GRID_CONTROL,
                    help="control grid side (9 -> 2,187 dims); 0 disables it")
    ap.add_argument("--d-lut", type=int, default=C.D_LUT_DEFAULT,
                    help="the main d_LUT recorded as the default in the manifest")
    ap.add_argument("--d-slices", default=",".join(str(d) for d in C.D_LUT_CHOICES),
                    help="the pre-registered ladder; all are PREFIXES of one SVD")
    ap.add_argument("--control-rows", type=int, default=C.LEDGER_9CUBED_ROWS,
                    help="index rows the ledger's Lib_tr draw samples")
    ap.add_argument("--control-seed", type=int, default=20260810,
                    help="seed of the Lib_tr re-draw (the ledger's own was never "
                         "recorded; the resulting id count is reported measured)")
    ap.add_argument("--exclude-eval-luts", action="store_true",
                    help="drop every lut_id that also appears in V_what / "
                         "T_final / V_where.  OFF by default: those ids are a "
                         "complete SUBSET of train's, so excluding them shrinks "
                         "the fit set below EPR-031 §1's 3,149 (see NOTES)")
    ap.add_argument("--limit", type=int, default=0,
                    help="fit on the first N lut_id only (smoke / unit test)")
    ap.add_argument("--no-recon", action="store_true",
                    help="skip code_recon_de00 (the slow geometric read-out)")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the pools and the plan, write nothing")
    return ap


def _parse_slices(text: str) -> tuple[int, ...]:
    out = tuple(int(t) for t in str(text).replace(" ", "").split(",") if t)
    if not out or min(out) < 1:
        raise ValueError(f"--d-slices must be positive integers; got {text!r}")
    return tuple(sorted(set(out)))


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"],
                              cwd=Path(__file__).resolve().parents[3],
                              capture_output=True, text=True, timeout=20
                              ).stdout.strip() or "unknown"
    except Exception:                                    # pragma: no cover
        return "unknown"


def _atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_npz(path: Path, **arrays: Any) -> None:
    """Write to ``<name>.tmp`` in the SAME directory, then ``os.replace``.

    A killed job then leaves the previous artefact intact instead of half a
    ``code.npz`` that loads and silently carries a truncated basis.
    """
    tmp = path.parent / (path.name + ".tmp")
    with tmp.open("wb") as fh:
        np.savez(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def control_table(bank, ids: Sequence[str], *, n_grid: int) -> dict[str, Any]:
    """The ``9^3`` control: cumulative 90/95/99 % dims of a residual PCA."""
    r, _grid = C.lut_residual_matrix(bank, ids, n_grid=int(n_grid))
    pca = C.fit_whitened_pca(r)
    return {"grid": f"{n_grid}^3", "dim": int(r.shape[1]), "n_lut": int(r.shape[0]),
            "rank": int(pca.explained_variance_ratio.size),
            "cumulative_variance_dims": C.cumulative_dims(
                pca.explained_variance_ratio)}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    t0 = time.time()
    out_dir = Path(args.out)
    slices = _parse_slices(args.d_slices)
    d_keep = max(slices + (int(args.d_lut),))

    fit_ids = C.train_lut_ids(args.split)
    pools = C.eval_only_lut_ids()
    excluded: list[str] = []
    if args.exclude_eval_luts:
        drop: set[str] = set()
        for ids in C.eval_only_lut_ids(
                splits_=("V_what", "T_final", "V_where")).values():
            drop |= set(ids)
        excluded = sorted(set(fit_ids) & drop)
        fit_ids = [i for i in fit_ids if i not in drop]
    if args.limit:
        fit_ids = fit_ids[:int(args.limit)]
    clean = C.assert_fit_set_clean(fit_ids)

    n = len(fit_ids)
    k_max = min(n - 1, 3 * int(args.grid) ** 3)
    if d_keep > k_max:
        d_keep = k_max
        slices = tuple(d for d in slices if d <= k_max) or (k_max,)

    plan: dict[str, Any] = {
        "arm": "EPR-031",
        "gate": "C0",
        "out": str(out_dir),
        "split": args.split,
        "grid": f"{args.grid}^3",
        "dim": 3 * int(args.grid) ** 3,
        "n_fit_lut": n,
        "d_keep": int(d_keep),
        "d_lut_default": int(args.d_lut),
        "d_slices": list(slices),
        "rank_cap_min_n_minus_1_d": int(k_max),
        "exclude_eval_luts": bool(args.exclude_eval_luts),
        "n_excluded": len(excluded),
        "fit_set_report": clean,
        "eval_split_lut_pools": {k: len(v) for k, v in pools.items()},
        "command": " ".join([sys.executable, "-m",
                             "q3vl.whatb.scripts.build_lut_code", *sys.argv[1:]]),
        "git_rev": _git_rev(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, **plan}, indent=2, ensure_ascii=False))
        return 0

    bank = C.open_bank(args.bank_dir) if args.bank_dir else C.open_bank()
    print(f"[EPR-031 C0] evaluating {n} LUTs on the {args.grid}^3 grid ...",
          flush=True)
    residual, grid = C.lut_residual_matrix(bank, fit_ids, n_grid=int(args.grid))
    t_lut = time.time() - t0

    print(f"[EPR-031 C0] fitting whitened PCA on {residual.shape} ...", flush=True)
    t1 = time.time()
    pca = C.fit_whitened_pca(residual, n_components=int(d_keep))
    t_pca = time.time() - t1
    c_star = pca.transform(residual, int(d_keep))

    recon: list[dict[str, Any]] = []
    if not args.no_recon:
        for d in slices:
            for clamp in (False, True):
                recon.append(C.code_recon_de00(pca, residual, grid, d_lut=int(d),
                                               clamp=clamp))
                print(f"[EPR-031 C0] code_recon_de00 d_LUT={d} clamp={clamp}: "
                      f"{recon[-1]['mean']:.6f}", flush=True)

    control: dict[str, Any] = {}
    if int(args.control_grid) >= 2:
        ctrl_ids = C.control_lut_ids(int(args.control_rows), int(args.control_seed),
                                     args.split)
        if args.limit:
            ctrl_ids = ctrl_ids[:int(args.limit)]
        control = {
            "ledger": {
                "source": "EPR-024:659 (and five more proposals, verbatim)",
                "protocol": (f"{C.LEDGER_9CUBED_ROWS} random train index rows -> "
                             f"{C.LEDGER_9CUBED_IDS} lut_id, "
                             f"{C.GRID_CONTROL}^3 grid, 2187 dims"),
                "cumulative_variance_dims": {
                    "90": C.LEDGER_9CUBED_DIMS[0], "95": C.LEDGER_9CUBED_DIMS[1],
                    "99": C.LEDGER_9CUBED_DIMS[2]},
            },
            "redraw": {"rows": int(args.control_rows),
                       "seed": int(args.control_seed),
                       "n_lut_id": len(ctrl_ids),
                       "matches_ledger_id_count":
                           len(ctrl_ids) == C.LEDGER_9CUBED_IDS},
            "measured_redraw_pool": control_table(bank, ctrl_ids,
                                                  n_grid=int(args.control_grid)),
            "measured_full_train_pool": control_table(bank, fit_ids,
                                                      n_grid=int(args.control_grid)),
        }
        print("[EPR-031 C0] 9^3 control: "
              f"redraw {control['measured_redraw_pool']['cumulative_variance_dims']}"
              f" / full train "
              f"{control['measured_full_train_pool']['cumulative_variance_dims']}",
              flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    code_path = out_dir / "code.npz"
    _atomic_write_npz(
        code_path,
        c_star=c_star.astype(np.float64),
        lut_ids=np.asarray(fit_ids, dtype=np.str_),
        mean=pca.mean, components=pca.components,
        explained_variance=pca.explained_variance,
        explained_variance_ratio=pca.explained_variance_ratio,
        whiten_scale=pca.whiten_scale,
        total_variance=np.asarray(pca.total_variance, dtype=np.float64),
        grid=grid.astype(np.float64))

    manifest = {
        **plan,
        "pca": pca.facts(),
        "cumulative_variance_dims": C.cumulative_dims(pca.explained_variance_ratio),
        "variance_targets": list(C.VARIANCE_TARGETS),
        "code_recon_de00": recon,
        "control_9cubed": control,
        "artefact": {"code.npz": {"sha256": C.sha256_file(code_path),
                                  "bytes": code_path.stat().st_size,
                                  "c_star_shape": list(c_star.shape)}},
        "timing_s": {"lut_eval": t_lut, "pca_fit": t_pca,
                     "total": time.time() - t0},
        "authority": ("EPR-031 §3.1 / §7-C0; the residual operator is "
                      "q3vl.whatb.lutdata.LutBank.apply (rendering.py:390-405)"),
    }
    _atomic_write_json(out_dir / "manifest.json", manifest)
    print(json.dumps({"n_fit_lut": n, "d_keep": int(d_keep),
                      "cumulative_variance_dims":
                          manifest["cumulative_variance_dims"],
                      "code_recon_de00": [
                          {"d_lut": r["d_lut"], "clamped": r["clamped"],
                           "mean": r["mean"]} for r in recon],
                      "sha256": manifest["artefact"]["code.npz"]["sha256"]},
                     indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
