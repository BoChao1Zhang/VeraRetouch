"""D0-7 -- generator-randomness replay upper bound for V_where local samples.

The question
------------
Every local GT mask in this campaign was *generated*, not annotated: for a fixed
(image, subject instance, slot), ``construct.canonical_masks.build_mask_plan``
draws the geometry from a ``random.Random`` seeded by
``(build_id, seed, source_id, mode, index)``.  If that draw is wide, then a
sizeable part of the residual error every Where head is charged with is not a
perception failure at all -- it is the generator's own coin flip, which no model
conditioned on (image, instruction) can possibly predict.

D0-7 measures that coin flip directly.  Hold the image, the subject instance and
the slot (family + index) FIXED; re-roll the generator ``k = 8`` times; measure
how much the resulting mask geometry moves.

``U_replay`` = mean pairwise hard-IoU across the 8 re-rolls, on the H/16 grid,
with each mask binarised by its OWN matched-area top-k.  It is an *upper bound*
on any conditional predictor's achievable IoU on that sample.

Pre-registered thresholds (RESEARCH_ceiling-push_2026-08-11 D0-7; NOT tuned here)
--------------------------------------------------------------------------------
    U_replay <= 0.85   -> (ii) intrinsic randomness dominates
    0.85 .. 0.93       -> (ii) is real, must be handled in parallel
    U_replay >= 0.95   -> (ii) essentially excluded

Honest wrinkle, found by the D0-1 mechanism audit and re-stated in the output
-----------------------------------------------------------------------------
The ``semantic`` family's re-roll is **cosmetic**.  ``_semantic_alpha``
(``canonical_masks.py``) blurs the hard SAM3 instance mask and re-multiplies by
it; the only rng use is ``radius *= rng.uniform(0.7, 1.4)``, a blur-radius
jitter.  The silhouette itself is deterministic given the instance.  So the
semantic ``U_replay`` will sit near 1.0, and that number is **not** evidence
about semantic ambiguity -- it is evidence that the generator does not randomise
semantic geometry.  It is reported separately and excluded from the geometric
read-out.

Discipline
----------
* No AUC anywhere (CLAUDE.md red line).  IoU on the grid, matched-area top-k.
* Thresholding is always "top-k matching the mask's own area", never a tuned
  threshold (``q3vl.whereb.metrics.topk_mask`` docstring).
* Grid convention is the campaign's: ``grid_from_geometry(out_h, out_w)`` and
  ``area_resize``, i.e. exactly what every Where-B criterion already uses.
* Nothing that cannot be resolved is silently dropped: every skip is counted
  with its reason in ``metrics.json``.

Run
---
    export LD_LIBRARY_PATH=/home/bc/miniconda3/envs/llm_factory/lib
    /home/bc/envs/q3vl_sft/bin/python q3vl/whereb/scripts/run_d0_7_replay.py \
        --out experiments/.../d0_7 --limit 40      # validation
    ... --out experiments/.../d0_7                  # full 400
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]

#: the fixed eight-slot protocol's four families (canonical_masks.MODE_COUNTS)
FAMILIES = ("radial", "linear", "band", "semantic")
#: the three families whose geometry the generator actually randomises
GEOMETRIC_FAMILIES = ("radial", "linear", "band")

#: published subject cache (indexes/ shards/ manifest.json metadata.jsonl)
SUBJECT_CACHE = Path("/mnt/nfs-ro/bc/data/datasets/cache/subject")

#: production render short edge for every local build (databuild.prod-l*.toml
#: [render] short_edge = 1024).  ``preprocess_source`` scales the EXIF-oriented
#: image to this short edge, and that is the raster size ``build_mask_plan`` was
#: called with -- and therefore the size of the published ``.cgt.png``.
RENDER_SHORT_EDGE = 1024
#: [masks] linear_target_alpha_mass, pinned at 0.50 by construct.config
LINEAR_TARGET = 0.5

#: pre-registered verdict bands.  Do not tune.
THRESHOLDS = {
    "dominates_at_or_below": 0.85,
    "real_band": [0.85, 0.93],
    "excluded_at_or_above": 0.95,
}


def verdict_for(u: float | None) -> str:
    if u is None:
        return "no_data"
    if u <= THRESHOLDS["dominates_at_or_below"]:
        return "(ii) intrinsic randomness DOMINATES"
    if u >= THRESHOLDS["excluded_at_or_above"]:
        return "(ii) essentially EXCLUDED"
    if u < THRESHOLDS["real_band"][1]:
        return "(ii) REAL, handle in parallel"
    return "between the 0.93 band top and the 0.95 exclusion floor (unregistered gap)"


def family_of(slot_id: str | None) -> str:
    return str(slot_id).rsplit("-", 1)[0] if slot_id else "unknown"


# --------------------------------------------------------------------------
# worker: one sample -> k replayed grid masks
# --------------------------------------------------------------------------

_W_STATE: dict[str, Any] = {}


def _worker_init() -> None:
    # numpy/PIL only in here; one thread per process, the pool provides the
    # parallelism and BLAS oversubscription would just thrash a loaded box.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    for entry in (str(REPO), str(REPO / "dataset_build" / "src")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    import torch  # the grid projection below must be the campaign's area_resize

    torch.set_num_threads(1)
    from construct.canonical_masks import MaskPlanError, build_mask_plan  # noqa: F401
    from construct.sources import SourceRecord  # noqa: F401

    _W_STATE["build_mask_plan"] = build_mask_plan
    _W_STATE["SourceRecord"] = SourceRecord
    _W_STATE["MaskPlanError"] = MaskPlanError


def _grid_downsample(mask_hw: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """(H, W) float32 in [0,1] -> (gh, gw), the campaign's area projection."""
    import torch

    from q3vl.where.upsample import area_resize

    t = torch.from_numpy(np.ascontiguousarray(mask_hw, dtype=np.float32))[None, None]
    return area_resize(t, (gh, gw))[0, 0].clamp(0, 1).numpy()


def replay_one(job: dict[str, Any]) -> dict[str, Any]:
    """Re-roll one sample's generator ``k`` times, return the ``k`` grid masks.

    The (H, W) raster is projected to the H/16 grid *inside the worker*: a
    1024x1546 float32 array is 6 MB and shipping eight of them back per sample
    would cost more than the geometry sampling itself.
    """
    build_mask_plan = _W_STATE["build_mask_plan"]
    SourceRecord = _W_STATE["SourceRecord"]
    MaskPlanError = _W_STATE["MaskPlanError"]

    source = SourceRecord(
        source_id=job["source_id"],
        source_path=Path(job["source_path"]),
        cache_dir=Path(job["cache_dir"]),
        subject_path=Path(job["subject_png"]),
        subject_meta_path=Path(job["subject_png"]).with_suffix(".json"),
        scene=job.get("scene") or "unknown",
        subject=job.get("subject") or {},
        mask_area=float(job.get("mask_area") or 0.0),
    )
    gh, gw = int(job["gh"]), int(job["gw"])
    grids: list[list[list[float]]] = []
    areas: list[float] = []
    errors: list[str] = []
    for seed in range(int(job["k"])):
        # Fresh ``build_id`` per replay, per the main agent's ruling.  ``mask_id``
        # = stable_id("mask", build_id, source_id, physical_key) omits the seed,
        # so two replays under the production ``build_id`` would collide on
        # ``mask_id`` and the production writer short-circuits on an existing
        # ``.cgt.png``.  Belt-and-braces here: this script calls build_mask_plan
        # in memory and never writes a ``.cgt.png`` -- but reusing the production
        # ``build_id`` is exactly the habit that would make a later, writing
        # variant of this script silently replay nothing.
        try:
            plan = build_mask_plan(
                source,
                build_id=f"d0_7_replay_s{seed}",
                seed=seed,
                width=int(job["width"]),
                height=int(job["height"]),
                linear_target=LINEAR_TARGET,
            )
        except MaskPlanError as exc:
            errors.append(f"seed{seed}:MaskPlanError:{exc.code}")
            continue
        except Exception as exc:  # noqa: BLE001
            errors.append(f"seed{seed}:{type(exc).__name__}:{str(exc)[:60]}")
            continue

        slot = None
        for s in plan.slots:
            if s.slot_id == job["slot_id"]:
                slot = s
                break
        if slot is None and job.get("mode") and job.get("mode_index") is not None:
            for s in plan.slots:
                if s.mode == job["mode"] and s.mode_index == job["mode_index"]:
                    slot = s
                    break
        if slot is None:
            errors.append(f"seed{seed}:slot_not_in_plan:{job['slot_id']}")
            continue

        alpha = np.asarray(slot.mask.effective_alpha, dtype=np.float32)
        if alpha.shape != (int(job["height"]), int(job["width"])):
            errors.append(f"seed{seed}:raster_shape:{alpha.shape}")
            continue
        grids.append(_grid_downsample(alpha, gh, gw).tolist())
        areas.append(float(alpha.mean()))

    return {
        "sample_id": job["sample_id"],
        "grids": grids,
        "raster_area_mean": areas,
        "errors": errors,
    }


# --------------------------------------------------------------------------
# aggregation helpers
# --------------------------------------------------------------------------

def quantiles(xs: list[float]) -> dict[str, float | None]:
    v = np.asarray([x for x in xs if x is not None and np.isfinite(x)], dtype=np.float64)
    if not v.size:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p25": None,
                "p75": None, "p90": None, "min": None, "max": None}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "p10": float(np.percentile(v, 10)),
        "p25": float(np.percentile(v, 25)),
        "p75": float(np.percentile(v, 75)),
        "p90": float(np.percentile(v, 90)),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def load_subject_index() -> tuple[dict[str, dict], dict[str, dict]]:
    """``asset_id -> row`` and ``source_path -> row`` from the published cache.

    The subject cache publishes one ``.vrmeta.json`` member per subject, and
    ``metadata.jsonl`` carries that member's payload inline -- so the whole map
    is one 150k-line scan with no tar reads.  ``construct.sources`` sets
    ``SourceRecord.source_id = asset_id or stable_id("source", realpath(path))``,
    and every eligible row here has an ``asset_id``, so the asset map is the
    primary key and ``source_path`` (the build's ``i_in_path``) is the fallback.
    """
    by_asset: dict[str, dict] = {}
    by_src: dict[str, dict] = {}
    with (SUBJECT_CACHE / "metadata.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            if ".vrmeta.json" not in line:
                continue
            row = json.loads(line)
            if not str(row.get("member", "")).endswith(".vrmeta.json"):
                continue
            if row.get("asset_id"):
                by_asset[str(row["asset_id"])] = row
            if row.get("source_path"):
                by_src[str(row["source_path"])] = row
    return by_asset, by_src


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    # Queue/env quirk: jobs inherit a soft NOFILE of 1024 while the hard limit is
    # far higher, and the shard/catalog readers below open one descriptor per
    # shard per thread.  Raised unconditionally, as the task card requires.
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f"RLIMIT_NOFILE {soft} -> {resource.getrlimit(resource.RLIMIT_NOFILE)[0]}",
              flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"RLIMIT_NOFILE raise failed: {type(exc).__name__}: {exc}", flush=True)

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="deliverable directory (d0_7/)")
    ap.add_argument("--split", default="V_where")
    ap.add_argument("--k", type=int, default=8, help="number of generator re-rolls")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap on the number of LOCAL samples (validation runs)")
    ap.add_argument("--workers", type=int, default=16,
                    help="replay processes; the box is shared, keep it modest")
    ap.add_argument("--io-workers", type=int, default=24)
    args = ap.parse_args(argv)

    for entry in (str(REPO), str(REPO / "dataset_build" / "src")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    import torch

    torch.set_num_threads(1)
    from q3vl.where.fpre import grid_from_geometry
    from q3vl.where.maskdata import MaskResolver
    from q3vl.where.upsample import area_resize
    from q3vl.whereb.data import open_dataset
    from q3vl.whereb.metrics import gt_area_k, hard_iou, topk_mask
    from q3vl.whereb.stores import PublishedStore

    out_dir = Path(args.out)
    (out_dir / "config").mkdir(parents=True, exist_ok=True)
    scratch = Path(os.environ.get(
        "D0_7_SCRATCH",
        "/tmp/claude-1001/-home-bc-VeraRetouch/f25f01e7-86bf-444e-91bf-bd8160450b32/"
        "scratchpad/d0_7_subject",
    ))
    scratch.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    skips: dict[str, int] = {}
    skip_rows: list[dict[str, Any]] = []

    def skip(sample_id: str, reason: str, **extra) -> None:
        skips[reason] = skips.get(reason, 0) + 1
        skip_rows.append({"sample_id": sample_id, "reason": reason, **extra})

    # -- 1. the split ------------------------------------------------------
    ds, ds_info = open_dataset(args.split, need_mask=True)
    rows = ds.meta_rows()
    local = [i for i, r in enumerate(rows) if r.get("render_mode") == "local"]
    n_local_total = len(local)
    if args.limit is not None:
        local = local[: args.limit]
    print(f"{args.split}: {len(local)} local samples "
          f"(of {n_local_total} local / {len(rows)} total)", flush=True)

    # -- 2. slot_id + source_id from the construction-side .vrmeta.json ----
    #    The family label survives ONLY here (amort/data.family_labels), and the
    #    slot_id is what makes the replay compare like with like.
    def _vrmeta(i: int) -> tuple[int, dict[str, Any] | str]:
        try:
            r = _vr_local.res  # type: ignore[attr-defined]
        except AttributeError:
            r = _vr_local.res = MaskResolver(verify="none", suffix=".vrmeta.json")
        try:
            rec = ds.record(i)
            vm = json.loads(r.read_bytes(r.resolve(rec)).decode("utf-8"))
            return i, {"rec": rec, "vm": vm}
        except Exception as exc:  # noqa: BLE001
            return i, f"{type(exc).__name__}: {str(exc)[:80]}"

    import threading

    # one MaskResolver per thread: it holds sqlite connections, which may not
    # cross threads (amort/data.py documents the 72% silent-failure mode)
    _vr_local = threading.local()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.io_workers) as ex:
        vr = dict(ex.map(_vrmeta, local))
    print(f"  .vrmeta.json for {len(vr)} samples in {time.time() - t0:.1f}s", flush=True)

    # -- 3. subject cache ---------------------------------------------------
    by_asset, by_src = load_subject_index()
    store = PublishedStore(SUBJECT_CACHE, verify=False)
    print(f"  subject cache: {len(by_asset)} eligible subjects, "
          f"{len(store)} published members", flush=True)

    jobs: list[dict[str, Any]] = []
    gt_grid: dict[str, np.ndarray] = {}
    meta: dict[str, dict[str, Any]] = {}
    subject_bytes_written = 0

    for i in local:
        sid = rows[i]["sample_id"]
        got = vr.get(i)
        if not isinstance(got, dict):
            skip(sid, "vrmeta_unreadable", detail=str(got))
            continue
        rec, vm = got["rec"], got["vm"]
        slot_id = vm.get("slot_id")
        fam = family_of(slot_id)
        if fam not in FAMILIES:
            skip(sid, "slot_id_unparseable", slot_id=slot_id)
            continue
        source_id = vm.get("source_id")
        i_in = vm.get("i_in_path")
        row = by_asset.get(str(source_id)) or by_src.get(str(i_in))
        if row is None:
            skip(sid, "subject_not_in_cache", source_id=source_id, i_in_path=i_in)
            continue

        # subject.png bytes -> a local file, because load_subject_alpha() opens a
        # PATH (archive_reader.open_image, which is local-first).
        png_path = scratch / f"{row['sample_id']}.subject.png"
        if not png_path.exists():
            try:
                png_path.write_bytes(store.read(row["sample_id"], ".subject.png"))
                subject_bytes_written += 1
            except Exception as exc:  # noqa: BLE001
                skip(sid, "subject_png_unreadable",
                     detail=f"{type(exc).__name__}: {str(exc)[:80]}")
                continue

        # the production raster size: EXIF-oriented image scaled to short edge
        # 1024 (rendering.preprocess_source).  Cross-checked against the
        # published .cgt.png below.
        img = rec.get("image") or {}
        try:
            ow, oh = int(img["oriented_w"]), int(img["oriented_h"])
            out_h, out_w = int(img["out_h"]), int(img["out_w"])
        except (KeyError, TypeError, ValueError):
            skip(sid, "record_geometry_missing")
            continue
        scale = RENDER_SHORT_EDGE / min(ow, oh)
        width = max(1, int(round(ow * scale)))
        height = max(1, int(round(oh * scale)))
        gh, gw = grid_from_geometry(out_h, out_w)

        # published GT at the same grid, same convention as every Where-B criterion
        try:
            hi = ds[i].mask_target_hi()
            g = area_resize(hi[None, None].float(), (gh, gw))[0, 0].clamp(0, 1)
            gt_grid[sid] = g.numpy()
        except Exception as exc:  # noqa: BLE001
            skip(sid, "published_gt_unreadable",
                 detail=f"{type(exc).__name__}: {str(exc)[:80]}")
            continue

        meta[sid] = {"family": fam, "slot_id": slot_id, "source_id": source_id,
                     "width": width, "height": height, "gh": gh, "gw": gw,
                     "subject_sample_id": row["sample_id"]}
        jobs.append({
            "sample_id": sid, "source_id": source_id, "source_path": i_in,
            "cache_dir": row["cache_dir"], "subject_png": str(png_path),
            "scene": row.get("scene"), "subject": row.get("subject"),
            "mask_area": row.get("mask_area"),
            "slot_id": slot_id, "mode": fam,
            "mode_index": int(str(slot_id).rsplit("-", 1)[1]),
            "width": width, "height": height, "gh": gh, "gw": gw, "k": args.k,
        })

    print(f"  {len(jobs)} jobs ready, {subject_bytes_written} subject.png fetched, "
          f"{sum(skips.values())} skipped so far {skips}", flush=True)

    # one-off provenance check: does the computed raster size equal the size of
    # the published .cgt.png the generator actually wrote?
    cgt_check: dict[str, Any] = {"checked": 0, "match": 0, "mismatch": []}
    idx_of = {rows[i]["sample_id"]: i for i in local}
    try:
        cres = MaskResolver(verify="none", suffix=".cgt.png")
        for j in jobs[:20]:
            raw = cres.load(cres.resolve(ds.record(idx_of[j["sample_id"]])))
            cgt_check["checked"] += 1
            if tuple(raw.shape) == (j["height"], j["width"]):
                cgt_check["match"] += 1
            else:
                cgt_check["mismatch"].append(
                    {"sample_id": j["sample_id"], "cgt": list(raw.shape),
                     "computed": [j["height"], j["width"]]})
        cres.close()
    except Exception as exc:  # noqa: BLE001
        cgt_check["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    print(f"  raster-size cross-check vs published .cgt.png: {cgt_check['match']}"
          f"/{cgt_check['checked']} exact", flush=True)

    # -- 4. replay ----------------------------------------------------------
    t0 = time.time()
    results: dict[str, dict[str, Any]] = {}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as ex:
        for r in ex.map(replay_one, jobs, chunksize=1):
            results[r["sample_id"]] = r
            done += 1
            if done % 25 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"  replay {done}/{len(jobs)}  {el:.0f}s  "
                      f"({el / max(done, 1):.2f} s/sample)", flush=True)

    # -- 5. per-sample U_replay --------------------------------------------
    per_sample: list[dict[str, Any]] = []
    n_k_zero = 0
    for j in jobs:
        sid = j["sample_id"]
        r = results.get(sid)
        m = meta[sid]
        if r is None:
            skip(sid, "worker_no_result")
            continue
        grids = [np.asarray(g, dtype=np.float32) for g in r["grids"]]
        if len(grids) < 2:
            skip(sid, "fewer_than_2_seeds_ok",
                 n_ok=len(grids), errors=r["errors"][:4])
            continue

        ts = [torch.from_numpy(g) for g in grids]
        ks = [gt_area_k(t) for t in ts]                    # each mask's OWN area
        n_k_zero += sum(1 for k in ks if k == 0)
        bins = [topk_mask(t, k) for t, k in zip(ts, ks)]
        pair = [hard_iou(bins[a], bins[b]) for a, b in combinations(range(len(bins)), 2)]
        u = float(np.mean(pair))

        g_gt = torch.from_numpy(gt_grid[sid])
        k_gt = gt_area_k(g_gt)
        b_gt = topk_mask(g_gt, k_gt)
        vs_gt = [hard_iou(b, b_gt) for b in bins]

        per_sample.append({
            "sample_id": sid,
            "family": m["family"],
            "slot_id": m["slot_id"],
            "source_id": m["source_id"],
            "U_replay": u,
            "iou_vs_published_gt_mean": float(np.mean(vs_gt)),
            "iou_vs_published_gt_max": float(np.max(vs_gt)),
            "n_seeds_ok": len(grids),
            "grid": [m["gh"], m["gw"]],
            "k_replay_mean": float(np.mean(ks)),
            "k_gt": int(k_gt),
            "pairwise_min": float(np.min(pair)),
            "pairwise_max": float(np.max(pair)),
            "seed_errors": r["errors"][:4],
        })

    per_sample.sort(key=lambda d: d["sample_id"])
    with (out_dir / "per_sample.jsonl").open("w", encoding="utf-8") as fh:
        for row in per_sample:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # -- 6. aggregate -------------------------------------------------------
    def agg(sel: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "U_replay": quantiles([d["U_replay"] for d in sel]),
            "iou_vs_published_gt": quantiles(
                [d["iou_vs_published_gt_mean"] for d in sel]),
            "n_samples": len(sel),
        }

    overall = agg(per_sample)
    geometric = agg([d for d in per_sample if d["family"] in GEOMETRIC_FAMILIES])
    by_family = {f: agg([d for d in per_sample if d["family"] == f]) for f in FAMILIES}

    u_overall = overall["U_replay"]["mean"]
    u_geom = geometric["U_replay"]["mean"]

    # -- 6b. robustness -----------------------------------------------------
    # V_where's 400 local samples come from far fewer source images (a group
    # renders eight candidates off one image), so two samples can share the same
    # (source_id, slot_id) and therefore replay to *identical* mask sets. The
    # de-duplicated aggregate says whether that reuse moved the headline number.
    groups: dict[tuple[str, str], list[float]] = {}
    for d in per_sample:
        groups.setdefault((str(d["source_id"]), str(d["slot_id"])), []).append(
            d["U_replay"])
    geo_groups = {k: v for k, v in groups.items()
                  if family_of(k[1]) in GEOMETRIC_FAMILIES}
    u_arr = np.asarray([d["U_replay"] for d in per_sample], dtype=np.float64)
    g_arr = np.asarray([d["iou_vs_published_gt_mean"] for d in per_sample],
                       dtype=np.float64)
    u_geo_arr = np.asarray([d["U_replay"] for d in per_sample
                            if d["family"] in GEOMETRIC_FAMILIES], dtype=np.float64)
    robustness = {
        "n_distinct_source_images": len({d["source_id"] for d in per_sample}),
        "n_distinct_source_slot_pairs": len(groups),
        "max_samples_per_source_slot": max((len(v) for v in groups.values()),
                                           default=0),
        "dedup_by_source_slot_all": quantiles(
            [float(np.mean(v)) for v in groups.values()]),
        "dedup_by_source_slot_geometric": quantiles(
            [float(np.mean(v)) for v in geo_groups.values()]),
        "corr_U_vs_publishedGT": (float(np.corrcoef(u_arr, g_arr)[0, 1])
                                  if u_arr.size > 1 else None),
        "mean_gap_publishedGT_minus_U": (float((g_arr - u_arr).mean())
                                         if u_arr.size else None),
        "median_k_replay_over_k_gt": float(np.median(
            [d["k_replay_mean"] / max(d["k_gt"], 1) for d in per_sample]))
        if per_sample else None,
        "frac_below_0.85_all": float((u_arr < 0.85).mean()) if u_arr.size else None,
        "frac_below_0.85_geometric": (float((u_geo_arr < 0.85).mean())
                                      if u_geo_arr.size else None),
        "frac_below_0.70_geometric": (float((u_geo_arr < 0.70).mean())
                                      if u_geo_arr.size else None),
        "frac_below_0.50_geometric": (float((u_geo_arr < 0.50).mean())
                                      if u_geo_arr.size else None),
    }

    metrics: dict[str, Any] = {
        "experiment": "D0-7 generator-randomness replay upper bound",
        "question": ("how much of the local GT mask geometry is the generator's own "
                     "coin flip, i.e. unpredictable from (image, instruction)?"),
        "split": args.split,
        "k_seeds": args.k,
        "n_local_in_split": n_local_total,
        "n_attempted": len(local),
        "n_scored": len(per_sample),
        "n_skipped": sum(skips.values()),
        "skip_reasons": skips,
        "method": {
            "replay": ("build_mask_plan(source, build_id=f'd0_7_replay_s{seed}', "
                       "seed=seed, width, height, linear_target=0.5) for seed in "
                       "0..k-1; the slot whose slot_id equals the sample's real one "
                       "is extracted, so image / subject instance / family / index "
                       "are all held fixed and only the generator rng moves"),
            "fresh_build_id": ("a fresh build_id per replay, per the main agent's "
                               "ruling: mask_id omits the seed and the production "
                               "writer short-circuits on an existing .cgt.png. This "
                               "script never writes a .cgt.png, so it is "
                               "belt-and-braces"),
            "grid": "grid_from_geometry(out_h, out_w) = (H/16, W/16), area_resize",
            "threshold": "matched-area top-k of each mask's OWN grid area (no tuning)",
            "metric": "mean of all C(k,2) pairwise hard-IoU",
            "raster_size": (f"EXIF-oriented image scaled to short edge "
                            f"{RENDER_SHORT_EDGE} (rendering.preprocess_source), "
                            f"cross-checked against the published .cgt.png"),
            "no_auc": "AUC is banned as a spatial criterion (CLAUDE.md 2026-08-05)",
        },
        "raster_size_cross_check": cgt_check,
        "thresholds_preregistered": THRESHOLDS,
        "overall": overall,
        "verdict_overall": verdict_for(u_overall),
        "geometric_only": geometric,
        "verdict_geometric_only": verdict_for(u_geom),
        "by_family": by_family,
        "robustness": robustness,
        "semantic_caveat": (
            "The semantic family's re-roll is COSMETIC: _semantic_alpha blurs the "
            "hard SAM3 instance mask and re-multiplies by it, and the only rng use "
            "is `radius *= rng.uniform(0.7, 1.4)` -- a blur-radius jitter. The "
            "silhouette is deterministic given the instance, so a semantic "
            "U_replay near 1.0 says the GENERATOR does not randomise semantic "
            "geometry. It is NOT evidence that semantic targets are unambiguous, "
            "and it must not be read as such."
        ),
        "notes": {
            "k_zero_grid_masks": n_k_zero,
            "k_zero_meaning": ("a replayed mask whose grid area top-k is 0 cells; "
                               "hard_iou of two empty masks is 1.0 by definition, "
                               "which would inflate U_replay"),
            "iou_vs_published_gt": ("each replay is an INDEPENDENT draw (fresh "
                                    "build_id), so this column should land near "
                                    "U_replay if the replay reproduces the real "
                                    "generator distribution; a large gap would mean "
                                    "the replay is not the production process"),
        },
        "dataset_info": ds_info,
        "runtime_s": round(time.time() - t_start, 1),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    if skip_rows:
        with (out_dir / "skipped.jsonl").open("w", encoding="utf-8") as fh:
            for row in skip_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    (out_dir / "config" / "run_config.json").write_text(json.dumps({
        "argv": sys.argv,
        "args": vars(args),
        "git_commit": git_commit(),
        "python": sys.version,
        "interpreter": sys.executable,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "subject_cache": str(SUBJECT_CACHE),
        "render_short_edge": RENDER_SHORT_EDGE,
        "linear_target": LINEAR_TARGET,
        "thresholds": THRESHOLDS,
    }, indent=2), encoding="utf-8")

    # -- 7. summary table ---------------------------------------------------
    def fmt(x: float | None, w: int = 6) -> str:
        return f"{x:{w}.4f}" if isinstance(x, float) else " " * (w - 1) + "-"

    print()
    print("=" * 92)
    print(f"D0-7  generator-randomness replay upper bound   split={args.split}  "
          f"k={args.k}  n={len(per_sample)}")
    print("=" * 92)
    hdr = (f"{'group':<12}{'n':>5}{'U_mean':>9}{'U_med':>9}{'P10':>9}{'P25':>9}"
           f"{'P75':>9}{'vsGT_mean':>11}")
    print(hdr)
    print("-" * len(hdr))
    for name, a in [("ALL", overall), ("geometric", geometric)] + \
                   [(f, by_family[f]) for f in FAMILIES]:
        q = a["U_replay"]
        g = a["iou_vs_published_gt"]
        print(f"{name:<12}{a['n_samples']:>5}{fmt(q['mean'], 9)}{fmt(q['median'], 9)}"
              f"{fmt(q['p10'], 9)}{fmt(q['p25'], 9)}{fmt(q['p75'], 9)}"
              f"{fmt(g['mean'], 11)}")
    print("-" * len(hdr))
    print(f"pre-registered: U<=0.85 dominates | 0.85-0.93 real | >=0.95 excluded")
    print(f"VERDICT (all families)   U_replay mean = {fmt(u_overall)} -> "
          f"{metrics['verdict_overall']}")
    print(f"VERDICT (geometric only) U_replay mean = {fmt(u_geom)} -> "
          f"{metrics['verdict_geometric_only']}")
    print(f"semantic is COSMETIC re-roll (blur-radius jitter only) -- "
          f"NOT evidence about semantic ambiguity")
    print(f"skipped {sum(skips.values())}: {skips}")
    print(f"k==0 grid masks: {n_k_zero}")
    print(f"source-image reuse: {robustness['n_distinct_source_images']} images, "
          f"{robustness['n_distinct_source_slot_pairs']} distinct (source,slot); "
          f"dedup U_mean ALL={fmt(robustness['dedup_by_source_slot_all']['mean'])} "
          f"geom={fmt(robustness['dedup_by_source_slot_geometric']['mean'])}")
    print(f"replay-vs-published-GT agreement: corr="
          f"{fmt(robustness['corr_U_vs_publishedGT'])}, mean gap="
          f"{fmt(robustness['mean_gap_publishedGT_minus_U'])}, "
          f"median k_replay/k_gt={fmt(robustness['median_k_replay_over_k_gt'])}")
    print(f"wrote {out_dir}/metrics.json, per_sample.jsonl  "
          f"({time.time() - t_start:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
