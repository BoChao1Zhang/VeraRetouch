#!/usr/bin/env python
"""Producer for the ``--eval-bundle`` EPR-027 IDGATE consumes.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/bc/VeraRetouch \\
    /home/bc/envs/q3vl_sft/bin/python -m q3vl.whatb.scripts.build_eval_bundle \\
        --split V_what \\
        --z-cache /home/bc/data/runs/whatb/zcache_v2seg \\
        --out /home/bc/data/caches/whatb_eval_20260815/V_what

``run_idgate_arm.py:257-284`` reads ``<dir>/index.jsonl`` + one
``<dir>/<sample_id>.npz`` per row and refuses to start without them
(``:554-556``); nothing in the repository wrote them.  This does, from artefacts
that already exist -- **no VLM forward, no GPU**:

======================  =====================================================
``image`` ``(3,H,W)``   ``q3vl.whatb.evaldata.SampleStore`` (the one loader):
                        ``sft2seg-20260804`` by ``(shard, offset, length)``,
                        ``area_resize``-d to short side 512
``alpha`` ``(H,W)``     the same ``SampleStore``: ``where_a-20260805/maskviews``
                        ``.maskhi.png``; a ``style`` sample has no mask member
                        and gets ones -- the dataset's own convention
``alpha_shuffle``       another sample's GT field (§4.3 row 4 / the required
                        ``field_shuffle`` column), ``area_resize``-d to this
                        sample's grid.  Donor rule: see ``--shuffle-pool``
``z`` / ``z_N1_shuffle``  ``q3vl.whatb.zcache`` -- the four
``z_N2_irrelevant``       ``(split, control_tag)`` leaves, opened through
``z_N3_const``            ``ZCacheDir`` so the three HANDOFF 4.H start-up
                        assertions (checkpoint / readout_kind / context) run
                        here too and a cache from another base cannot be baked
                        into a bundle
======================  =====================================================

Two fields of the consumer's schema are **absent**, recorded in ``meta.json``
rather than faked:

``z_null``     the null-prompt read-out.  It is another frozen-VLM forward and
               this producer runs none.  ``EvalSample.z_null`` defaults to
               ``None`` and ``IdGateArm.z_lambda`` then returns ``z`` bit for
               bit at ``lambda = 1``, which is what training uses; only the
               ``--gate-u-source lambda`` / ``zhead`` ablation rows would read
               it, and neither is in ``REQUIRED_CRITERIA``.
``alpha_pred`` the where arm's ``m_pix`` on this split.  No such product exists
               on disk.  ``field_pred`` is **not** in ``REQUIRED_P2P3``
               (``criteria.py:96-98``), so its absence costs a diagnostic
               column, not the pre-registered table.

Row selection is ``normal_only(load_index(split))`` -- the same rows the other
five arms evaluate (``run_carrier_arm.py:284``, ``run_affonly_arm.py:818``), so
the cross-arm paired delta is on one sample set.  A row the ``none`` cache does
not carry is dropped, counted and named in ``meta.json``; it is never silent.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from q3vl.whatb import caliber as _caliber
from q3vl.whatb import splits as _splits
from q3vl.whatb.evaldata import SHORT_SIDE, SampleStore
from q3vl.whatb.zcache import CONTROL_TAGS, ZCacheDir

REPO = Path("/home/bc/VeraRetouch")
DEFAULT_CHECKPOINT = "/home/bc/data/runs/q3vl_base_sft_v2seg_20260814/checkpoint-4976"
DEFAULT_Z_ROOT = "/home/bc/data/runs/whatb/zcache_v2seg"

#: control tag on disk -> the npz key ``load_eval_bundle`` looks for
CONTROL_KEYS: dict[str, str] = {
    "shuffle": "z_N1_shuffle",
    "irrelevant": "z_N2_irrelevant",
    "const": "z_N3_const",
}

#: the schema fields this producer deliberately does not write (see the module
#: docstring); each one is recorded on the artefact with its reason.
ABSENT_FIELDS: dict[str, str] = {
    "z_null": ("the null-prompt read-out is another frozen-VLM forward and this "
               "producer runs none.  EvalSample.z_null defaults to None and "
               "z_lambda returns z bit for bit at lambda = 1 (the training "
               "condition); no REQUIRED_CRITERIA key reads it."),
    "alpha_pred": ("the where arm's m_pix on this split does not exist on disk. "
                   "field_pred is not in criteria.REQUIRED_P2P3, so this costs "
                   "a diagnostic column and no pre-registered one."),
}

SHUFFLE_POOLS: tuple[str, ...] = ("local", "all")


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def provenance() -> dict[str, Any]:
    out: dict[str, Any] = {"git_commit": "unknown", "working_tree_dirty": None,
                           "producer": __file__, "producer_sha256": None,
                           "python": sys.version.split()[0],
                           "torch": torch.__version__}
    try:
        out["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
        out["working_tree_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO).decode().strip())
    except Exception:                                          # pragma: no cover
        pass
    try:
        h = hashlib.sha256()
        for rel in ("q3vl/whatb/scripts/build_eval_bundle.py",
                    "q3vl/whatb/evaldata.py", "q3vl/whatb/zcache.py",
                    "q3vl/whatb/splits.py"):
            p = REPO / rel
            if p.is_file():
                h.update(rel.encode())
                h.update(p.read_bytes())
        out["producer_sha256"] = h.hexdigest()
    except Exception:                                          # pragma: no cover
        pass
    return out


# --------------------------------------------------------------------------- #
# arguments
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="build_eval_bundle",
        description="Write the index.jsonl + <sample_id>.npz bundle EPR-027 "
                    "consumes, from the split index, the maskviews and the z cache.")
    # the index口径, spelled / defaulted exactly as on the eight arm runners:
    # the bundle's row set IS the split index of one口径, so a bundle that does
    # not say which one cannot be re-made.
    _caliber.add_caliber_arguments(
        ap, group="index口径 (shared with the arm runners)",
        data=False, batch_split=False, base_lr=False)
    ap.add_argument("--split", default="V_what", choices=_splits.SPLITS)
    ap.add_argument("--out", type=Path, required=True,
                    help="destination directory (LOCAL disk; /mnt/nfs is refused)")
    ap.add_argument("--z-cache", default=DEFAULT_Z_ROOT,
                    help="root holding <split>__<tag>/ or <split>.<ctx>.<tag>.zcache.pt")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                    help="asserted against every cache's meta.checkpoint")
    ap.add_argument("--context", choices=("teacher", "generated"), default="generated",
                    help="context of the 'none' cache; the three controls are "
                         "always 'generated' (frozen block 6)")
    ap.add_argument("--readout", default="seg_color")
    ap.add_argument("--short-side", type=int, default=SHORT_SIDE)
    ap.add_argument("--limit", type=int, default=0,
                    help="first N rows only (a smoke bundle; recorded in meta.json)")
    ap.add_argument("--shuffle-pool", choices=SHUFFLE_POOLS, default="local",
                    help="donor pool for alpha_shuffle.  'local' (default) draws "
                         "only from rows that carry a real mask: a style donor's "
                         "field is ones everywhere, which for a style row makes "
                         "field_shuffle == field_gt and the column vacuous.  "
                         "'all' is the literal 'another sample's field'.")
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--resume", action="store_true",
                    help="keep .npz files that already exist (still re-indexed)")
    ap.add_argument("--self-check", type=int, default=4,
                    help="after writing, read back this many rows through the "
                         "consumer's own loader (run_idgate_arm.load_eval_bundle); "
                         "0 disables it")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve everything and print the plan, write nothing")
    return ap


# --------------------------------------------------------------------------- #
# alpha_shuffle donors
# --------------------------------------------------------------------------- #
def shuffle_donors(rows: Sequence[_splits.IndexRow], *, pool: str = "local",
                   seed: int = 20260810) -> dict[str, str]:
    """``sample_id -> donor sample_id`` for the ``field_shuffle`` column.

    Deterministic (a seeded permutation, no global RNG touched) and constrained
    to ``donor.source_image_id != own.source_image_id`` -- a donor from the same
    source image is that image's own mask again, which would make the shuffled
    field a near-copy of GT and the column silently vacuous.  The fallback when
    no such donor exists after a full scan is recorded per row, never silent.
    """
    if pool not in SHUFFLE_POOLS:
        raise ValueError(f"--shuffle-pool must be one of {SHUFFLE_POOLS}")
    donors = [r for r in rows if (pool == "all" or not r.is_style)]
    if not donors:
        raise ValueError(
            f"the {pool!r} donor pool is empty on split {rows[0].split if rows else '?'}; "
            "field_shuffle cannot be built and it is a required column")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(donors))
    out: dict[str, str] = {}
    for i, row in enumerate(rows):
        for k in range(len(donors)):
            cand = donors[int(order[(i + k) % len(donors)])]
            if (cand.sample_id != row.sample_id
                    and cand.source_image_id != row.source_image_id):
                out[row.sample_id] = cand.sample_id
                break
        else:                                                  # pragma: no cover
            raise ValueError(
                f"{row.sample_id}: no donor with a different source_image_id in "
                f"the {pool!r} pool ({len(donors)} rows); the shuffled-field "
                "column would compare a mask against itself")
    return out


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #
def _as_hw(alpha: Any, hw: tuple[int, int]) -> torch.Tensor:
    """``SampleStore.alpha`` -> the ``(H, W)`` field the consumer's schema names."""
    if isinstance(alpha, float):
        return torch.ones(hw, dtype=torch.float32)
    a = alpha
    while a.dim() > 2:
        if a.shape[0] != 1:
            raise ValueError(f"alpha has shape {tuple(a.shape)}, expected (H, W)")
        a = a[0]
    return a.to(dtype=torch.float32)


def _resize_field(a: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    """The one sanctioned operator (``q3vl/where/upsample.py:54-62``)."""
    if tuple(a.shape[-2:]) == tuple(hw):
        return a
    from q3vl.where.upsample import area_resize

    return area_resize(a.reshape(1, 1, *a.shape[-2:]), tuple(hw))[0, 0]


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> tuple[str, int]:
    """``np.savez`` through a buffer so the bytes can be hashed before they land."""
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    blob = buf.getvalue()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(blob)
    tmp.replace(path)                      # atomic: a half-written npz never appears
    return hashlib.sha256(blob).hexdigest(), len(blob)


def build(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    if str(out).startswith("/mnt/nfs"):
        raise SystemExit(
            f"--out {out} is on the NFS mount.  Campaign rule: reads go to "
            "/mnt/nfs-ro, writes go through `nfsx`, and a 4 GB bundle the arms "
            "open at start-up belongs on local disk.")

    t0 = time.time()
    # the口径 is resolved BEFORE the first load_index: the bundle's rows are the
    #口径's rows, so the选择 has to happen before anything is read.
    ver = _caliber.apply_dataset_version(args)
    all_rows = _splits.load_index(args.split)
    normal_rows = _splits.normal_only(all_rows)
    rows = normal_rows[: int(args.limit)] if args.limit else normal_rows

    caches = ZCacheDir(args.z_cache, split=args.split, checkpoint=args.checkpoint,
                       readout_kind=args.readout, context_source=args.context,
                       required=CONTROL_TAGS, seed=args.seed)
    none_cache = caches.cache("none")

    kept = [r for r in rows if r.sample_id in none_cache]
    dropped = [r.sample_id for r in rows if r.sample_id not in none_cache]
    missing_ctrl: dict[str, list[str]] = {}
    for tag in CONTROL_KEYS:
        c = caches.cache(tag)
        miss = [r.sample_id for r in kept if r.sample_id not in c]
        if miss:
            missing_ctrl[tag] = miss
    if missing_ctrl:
        raise SystemExit(
            "the control caches do not cover every row: "
            + json.dumps({k: len(v) for k, v in missing_ctrl.items()})
            + ".  N1/N2/N3 are required columns; a bundle with holes in them "
              "publishes a control computed on a different sample set.  First "
              f"missing: {json.dumps({k: v[:3] for k, v in missing_ctrl.items()})}")
    if not kept:
        raise SystemExit(f"no row of {args.split} normal-only is in the z cache")

    donors = shuffle_donors(kept, pool=args.shuffle_pool, seed=args.seed)
    by_id = {r.sample_id: r for r in kept}

    plan = {
        "split": args.split, "n_index": len(all_rows),
        "n_normal": len(normal_rows), "n_selected": len(rows),
        "limit": int(args.limit) or None,
        "n_in_z_cache": len(kept), "n_dropped_no_z": len(dropped),
        "out": str(out), "z_cache": caches.facts(),
        "shuffle_pool": args.shuffle_pool,
        # which index口径 those n_index / n_normal were counted in
        "dataset_version": ver.name,
        "dataset_root": str(ver.root),
        "dataset_version_facts": ver.facts(),
    }
    if args.dry_run:
        return {"dry_run": True, **plan}

    out.mkdir(parents=True, exist_ok=True)
    store = SampleStore(args.split, short_side=int(args.short_side))
    records = {r.sample_id: rec for r, rec in
               zip(kept, _splits.iter_records(kept))}

    index_lines: list[str] = []
    n_written = n_reused = 0
    total_bytes = 0

    def donor_field(sid: str, hw: tuple[int, int]) -> torch.Tensor:
        """The donor's GT field, ``area_resize``-d onto the *borrower's* grid.

        Read per use rather than cached: 567 mask decodes cost about 3 s and a
        cache of every donor mask is ~350 MB of resident memory for nothing.
        """
        donor = by_id[sid]
        if donor.is_style:
            # a style row has no mask member and its field is ones everywhere;
            # "resampling" ones is the same ones on any grid
            return torch.ones(hw, dtype=torch.float32)
        return _resize_field(_as_hw(store.alpha(sid, donor.task_type), hw), hw)

    for i, row in enumerate(kept):
        npz_path = out / f"{row.sample_id}.npz"
        img, alpha_raw = store.load(row)
        hw = (int(img.shape[-2]), int(img.shape[-1]))
        if min(hw) != int(args.short_side):
            raise SystemExit(
                f"{row.sample_id}: image short side is {min(hw)} after "
                f"SampleStore, the frozen headline resolution is {args.short_side}")
        alpha = _as_hw(alpha_raw, hw)
        if tuple(alpha.shape) != hw:
            raise SystemExit(
                f"{row.sample_id}: GT alpha {tuple(alpha.shape)} does not match the "
                f"image grid {hw}; this producer will not resample GT (the section "
                "E strata are defined on GT alpha at short side 512)")
        donor_id = donors[row.sample_id]
        donor_alpha = donor_field(donor_id, hw)

        rec = records.get(row.sample_id) or {}
        line: dict[str, Any] = {
            "sample_id": row.sample_id,
            "lut_id": row.lut_id,
            "task_type": row.task_type,
            "winner_confidence": row.winner_confidence,
            "minor": rec.get("minor"),
            "source_image_id": row.source_image_id,
            "height": hw[0], "width": hw[1],
            "alpha_mean": float(alpha.mean()),
            "alpha_shuffle_from": donor_id,
        }

        if args.resume and npz_path.is_file():
            n_reused += 1
            line["npz_reused"] = True
            index_lines.append(json.dumps(line, ensure_ascii=False))
            continue

        arrays = {
            "image": img.numpy().astype(np.float32),
            "alpha": alpha.numpy().astype(np.float32),
            "alpha_shuffle": donor_alpha.numpy().astype(np.float32),
            "z": none_cache.vector(row.sample_id).numpy().astype(np.float32),
        }
        for tag, key in CONTROL_KEYS.items():
            arrays[key] = caches.cache(tag).vector(
                row.sample_id).numpy().astype(np.float32)
        sha, nbytes = write_npz(npz_path, arrays)
        total_bytes += nbytes
        n_written += 1
        line["npz_sha256"] = sha
        line["npz_bytes"] = nbytes
        index_lines.append(json.dumps(line, ensure_ascii=False))
        if (i + 1) % 100 == 0:
            print(json.dumps({"done": i + 1, "of": len(kept),
                              "gib": round(total_bytes / 2 ** 30, 3),
                              "s": round(time.time() - t0, 1)}), flush=True)

    (out / "index.jsonl").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    meta = {
        "bundle": "whatb eval bundle (run_idgate_arm.py --eval-bundle)",
        "consumer": "q3vl/whatb/scripts/run_idgate_arm.py:257-284",
        **plan,
        "n_rows": len(index_lines), "n_npz_written": n_written,
        "n_npz_reused": n_reused,
        "bytes_written": total_bytes,
        "short_side": int(args.short_side),
        "row_selection": ("normal_only(load_index(split)) -- the same rows the "
                          "other five arms evaluate; low rows are not in the "
                          "bundle, so a board built on it has n_low_excluded = 0"),
        "dropped_no_z": dropped[:50],
        "npz_keys": ["image (3,H,W) f32 sRGB", "alpha (H,W) f32 GT",
                     "alpha_shuffle (H,W) f32", "z (2560,) f32",
                     *[f"{k} (2560,) f32" for k in CONTROL_KEYS.values()]],
        "absent_fields": ABSENT_FIELDS,
        "alpha_shuffle_rule": {
            "pool": args.shuffle_pool,
            "constraint": "donor.source_image_id != own.source_image_id",
            "draw": f"seeded permutation, seed={args.seed}, no global RNG",
            "resampled_with": "q3vl/where/upsample.py area_resize",
            "open_decision": (
                "'local' is this producer's conservative default and is NOT a "
                "ruling: 321 of the 567 V_what normal rows are style rows whose "
                "GT field is ones everywhere, so an unrestricted donor pool "
                "would give 57% of the rows a shuffled field identical to GT and "
                "field_shuffle would silently measure nothing.  --shuffle-pool "
                "all reproduces the literal 'another sample's field'."),
        },
        "image_source": str(_splits.SFT2SEG_SHARD_ROOT),
        # ``dataset_version`` / ``dataset_root`` / ``dataset_version_facts``
        # arrive through ``**plan`` above, so --dry-run prints them too
        "alpha_source": str(store.mask_dir),
        "provenance": provenance(),
        "elapsed_s": round(time.time() - t0, 1),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    return meta


# --------------------------------------------------------------------------- #
# self-check: the consumer's own loader
# --------------------------------------------------------------------------- #
def self_check(out: Path, n: int, *, short_side: int = SHORT_SIDE) -> dict[str, Any]:
    """Read the bundle back through ``run_idgate_arm.load_eval_bundle``.

    Not a re-implementation of the reader: the *consumer's* function, so a
    schema drift between producer and arm fails here rather than 2 GPU-hours
    into a run.
    """
    from q3vl.whatb.scripts.run_idgate_arm import load_eval_bundle

    samples = load_eval_bundle(out, device="cpu", limit=int(n),
                               short_side=int(short_side))
    want_ctrl = {"N1_shuffle", "N2_irrelevant", "N3_const"}
    for s in samples:
        if s.image.dtype != torch.float32 or s.image.dim() != 3 or s.image.shape[0] != 3:
            raise AssertionError(f"{s.sample_id}: image {tuple(s.image.shape)} "
                                 f"{s.image.dtype}")
        if tuple(s.alpha.shape) != tuple(s.image.shape[-2:]):
            raise AssertionError(f"{s.sample_id}: alpha {tuple(s.alpha.shape)} vs "
                                 f"image {tuple(s.image.shape[-2:])}")
        if min(s.image.shape[-2:]) != int(short_side):
            raise AssertionError(f"{s.sample_id}: short side {min(s.image.shape[-2:])}")
        if set(s.z_ctrl) != want_ctrl:
            raise AssertionError(f"{s.sample_id}: z_ctrl keys {sorted(s.z_ctrl)}")
        if tuple(s.z.shape) != (2560,):
            raise AssertionError(f"{s.sample_id}: z {tuple(s.z.shape)}")
        if s.alpha_shuffle is None or tuple(s.alpha_shuffle.shape) != tuple(
                s.image.shape[-2:]):
            raise AssertionError(f"{s.sample_id}: alpha_shuffle missing/mis-shaped")
        if float(s.image.min()) < 0.0 or float(s.image.max()) > 1.0:
            raise AssertionError(f"{s.sample_id}: image outside [0,1]")
    ctrl_delta = [float((s.z - s.z_ctrl["N1_shuffle"]).abs().max()) for s in samples]
    return {
        "loader": "q3vl.whatb.scripts.run_idgate_arm.load_eval_bundle",
        "n_checked": len(samples),
        "sample_ids": [s.sample_id for s in samples],
        "shapes": [{"image": list(s.image.shape), "alpha": list(s.alpha.shape)}
                   for s in samples],
        "task_types": [s.task_type for s in samples],
        "alpha_mean": [round(float(s.alpha.mean()), 6) for s in samples],
        "alpha_shuffle_mean": [round(float(s.alpha_shuffle.mean()), 6)
                               for s in samples],
        "max_abs_z_minus_zN1": [round(v, 6) for v in ctrl_delta],
        "z_ctrl_is_not_z": all(v > 0 for v in ctrl_delta),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    meta = build(args)
    if args.dry_run:
        print(json.dumps(meta, indent=2, ensure_ascii=False))
        return 0
    report: dict[str, Any] = {
        "out": str(args.out), "n_rows": meta["n_rows"],
        "n_npz_written": meta["n_npz_written"],
        "n_npz_reused": meta["n_npz_reused"],
        "gib": round(meta["bytes_written"] / 2 ** 30, 3),
        "elapsed_s": meta["elapsed_s"],
    }
    if args.self_check:
        report["self_check"] = self_check(Path(args.out), args.self_check,
                                          short_side=args.short_side)
        (Path(args.out) / "self_check.json").write_text(
            json.dumps(report["self_check"], indent=2, ensure_ascii=False),
            encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":                                     # pragma: no cover
    raise SystemExit(main())
