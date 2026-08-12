"""WEVAL long-tail analysis for the two PR-AMORT arms (P1 vs P3').

Reuses the WEVAL-1 taxonomy (`q3vl.whereb.analysis.taxonomy`) for the
six-dimension geometry classification -- geometry only, never a function of the
prediction being scored -- and adds an attribution layer built for *these* two
arms, because the stock `attribution.py` keys off Where-B fields
(`oracle_soft_iou`, `s_std_ratio`, `hi_lo_soft_iou_drop`) that the amort arms do
not publish.

Two ceilings, declared separately (this is the point the task card singles out)
--------------------------------------------------------------------------
`oracle_ceiling` in the stock module means "the target is not reachable through
the Phi-71 basis".  That question **only exists for P1**: P3' never touches Phi,
so a Phi reachability bound is not an upper bound on anything it does.  So:

* **basis ceiling (P1 only)** -- the published Where-A oracle latent decoded on
  the H/16 grid, top-k IoU against the same GT.  Directly comparable to the
  board, because the board is also scored on the H/16 grid.  A tail sample under
  this bound is *unreachable through the basis*, not badly fitted.
* **chain ceiling (both arms, identical)** -- the GT itself carried through the
  arms' own output chain: GT -> area_resize to H/16 -> guided upsample -> top-k
  at delivery resolution -> IoU against the delivery-resolution GT.  This is the
  resolution/upsampling bound and it binds both arms equally.

They are **not** interchangeable and are never summed: the basis ceiling is
measured at H/16 (same scale as the board), the chain ceiling at delivery
resolution (a bound on the delivered mask, not on the board number).  Each table
says which one it is using.

Tail buckets are organised as the task card asks -- recoverable vs not:

  NOT RECOVERABLE   chain_ceiling      the H/16 + guided-upsample chain cannot
                                       express this mask at delivery resolution
                    basis_ceiling      (P1 only) Phi-71 cannot express it
  RECOVERABLE       context_missing    GT-context scores well, generated does
                                       not => the information was lost in the
                                       `<where>` text, not in the head
                    coverage_bias      area ratio far from 1 in a consistent
                                       direction => loss weighting
                    underfit           every bound is high and the context is
                                       fine => the head simply has not learned it
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _median(xs) -> float | None:
    v = [float(x) for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def load_rows(per_sample: Path, mode: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with per_sample.open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("mode") == mode and not r.get("uncovered") and not r.get("is_fake"):
                out[r["sample_id"]] = r
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p1", default="/home/bc/data/runs/where_b/amort_P1_20260810/eval_final")
    ap.add_argument("--p3", default="/home/bc/data/runs/where_b/amort_P3prime_20260810/eval_final")
    ap.add_argument("--geometry", default="/home/bc/VeraRetouch/experiments/"
                    "Q3VL_metacanvas_where_what_20260804/where_b/geometry_V_where.json")
    ap.add_argument("--oracle", default="/mnt/nfs-ro/bc/data/datasets/where_a-20260805/"
                    "oracle/BA-3-Joint/s5/V_where")
    ap.add_argument("--split", default="V_where")
    ap.add_argument("--readout", default="band")
    ap.add_argument("--main", default="generated")
    ap.add_argument("--tail-frac", type=float, default=0.10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-ceilings", action="store_true")
    args = ap.parse_args(argv)

    try:
        import resource

        s_, h_ = resource.getrlimit(resource.RLIMIT_NOFILE)
        if s_ < h_:
            resource.setrlimit(resource.RLIMIT_NOFILE, (h_, h_))
    except Exception:
        pass

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from q3vl.whereb.analysis.taxonomy import DIMENSIONS, classify_geometry

    geo_blob = json.loads(Path(args.geometry).read_text())
    geometry = geo_blob["geometry"]
    labels = {sid: classify_geometry(g) for sid, g in geometry.items()}

    arms = {"P1": Path(args.p1), "P3prime": Path(args.p3)}
    rows_main = {k: load_rows(v / "per_sample.jsonl", args.main) for k, v in arms.items()}
    rows_gt = {k: load_rows(v / "per_sample.jsonl", "gt") for k, v in arms.items()}
    common = sorted(set(rows_main["P1"]) & set(rows_main["P3prime"]))
    print(f"common samples: {len(common)}", flush=True)

    # ---------------- ceilings -------------------------------------------
    ceil: dict[str, dict[str, float]] = {"chain": {}, "basis": {}}
    if not args.skip_ceilings:
        import torch

        from q3vl.where.config import UpsampleConfig
        from q3vl.where.fpre import grid_from_geometry
        from q3vl.where.readout import apply_readout
        from q3vl.where.upsample import area_resize, guided_upsample, luma_guide
        from q3vl.whereb.data import open_dataset
        from q3vl.whereb.metrics import gt_area_k, hard_iou, topk_mask
        from q3vl.whereb.stores import OracleStore

        ds, _ = open_dataset(args.split, need_mask=True)
        by_id = {ds.record(i)["sample_id"]: i for i in range(len(ds))}
        ucfg = UpsampleConfig()
        try:
            ostore = OracleStore(Path(args.oracle))
        except Exception as exc:  # noqa: BLE001
            print(f"WARN no oracle store: {exc}", flush=True)
            ostore = None

        for n, sid in enumerate(common):
            if sid not in by_id:
                continue
            s = ds[by_id[sid]]
            gh, gw = grid_from_geometry(s.geometry.out_h, s.geometry.out_w)
            gt_hi = s.mask_target_hi().float()
            # --- chain ceiling: GT through the arms' own output chain ---
            gt_low = area_resize(gt_hi[None, None], (gh, gw))
            guide = luma_guide(s.image_tensor().unsqueeze(0))
            up = guided_upsample(gt_low * 2 * 3.0 - 3.0, guide, ucfg)   # into (-3,3)
            up = torch.sigmoid((up + 3.0) / 6.0 * 12.0 - 6.0)           # back to [0,1]
            k_hi = gt_area_k(gt_hi)
            ceil["chain"][sid] = hard_iou(topk_mask(up.reshape(gt_hi.shape), k_hi),
                                          topk_mask(gt_hi, k_hi))
            # --- basis ceiling (P1 only): published oracle latent at H/16 ---
            if ostore is not None:
                lat = None
                try:
                    lat = ostore.latent(sid, args.readout)
                except Exception:
                    lat = None
                if lat is not None:
                    from q3vl.whereb.fields import phi_dir_fast

                    img_low = area_resize(s.image_tensor().unsqueeze(0), (gh, gw))[0]
                    npz = None
                    cache = Path("/home/bc/data/runs/where_b/amort_cache_20260810/cache") / f"{sid}.npz"
                    if cache.exists():
                        npz = np.load(cache)
                    if npz is not None:
                        sem = torch.from_numpy(npz["semantic_low"]).float()
                        phi = phi_dir_fast(sem, img_low, gh, gw)
                        from q3vl.where.basis import alpha_of, w_dir_of
                        from q3vl.whereb.fields import s_from_params

                        s_low = s_from_params(phi, lat.w0.float(),
                                              alpha_of(lat.alpha_raw.float()),
                                              w_dir_of(lat.w_raw.float()))
                        m = apply_readout(args.readout, s_low,
                                          {k2: v.float() for k2, v in lat.rho.items()})
                        m = m.reshape(gh, gw)
                        g16 = area_resize(gt_hi[None, None], (gh, gw))[0, 0]
                        k16 = gt_area_k(g16)
                        ceil["basis"][sid] = hard_iou(topk_mask(m, k16),
                                                      topk_mask(g16, k16))
            if (n + 1) % 100 == 0:
                print(f"  ceilings [{n+1}/{len(common)}]", flush=True)
        print(f"chain ceiling n={len(ceil['chain'])} median="
              f"{_median(ceil['chain'].values())}", flush=True)
        print(f"basis ceiling n={len(ceil['basis'])} median="
              f"{_median(ceil['basis'].values())}", flush=True)

    # ---------------- six-dimension tables --------------------------------
    def strata(arm: str, dim: str, normal_only: bool) -> dict[str, dict]:
        buckets: dict[str, list] = defaultdict(list)
        for sid in common:
            r = rows_main[arm].get(sid)
            if r is None or sid not in labels:
                continue
            if normal_only and r.get("winner_confidence") != "normal":
                continue
            buckets[labels[sid][dim]].append(r)
        out = {}
        for lab, rs in sorted(buckets.items()):
            out[lab] = {
                "n": len(rs),
                "topk_iou_median": _median(x["hard_iou"] for x in rs),
                "center_prior_median": _median(x["center_prior_hard_iou"] for x in rs),
                "random_floor_median": _median(x["random_floor"] for x in rs),
                "boundary_f1_median": _median(x["grid_boundary_f1"] for x in rs),
            }
        return out

    tables = {arm: {dim: {"normal_only": strata(arm, dim, True),
                          "pooled": strata(arm, dim, False)}
                    for dim in DIMENSIONS} for arm in arms}

    # ---------------- family x area cross table ---------------------------
    cross = {}
    for arm in arms:
        c: dict[str, dict[str, Any]] = {}
        for sid in common:
            r = rows_main[arm].get(sid)
            if r is None or sid not in labels:
                continue
            key = f"{r.get('family')}|{labels[sid]['area']}"
            c.setdefault(key, []).append(r)
        cross[arm] = {k: {"n": len(v), "topk_iou_median": _median(x["hard_iou"] for x in v),
                          "center_prior_median": _median(x["center_prior_hard_iou"] for x in v)}
                      for k, v in sorted(c.items())}

    # ---------------- deep tail attribution --------------------------------
    CHAIN_LO, BASIS_LO, CTX_GAP, AREA_GAP = 0.75, 0.70, 0.10, 0.50
    tail: dict[str, list[dict]] = {}
    mech: dict[str, Counter] = {}
    for arm in arms:
        scored = [(rows_main[arm][s]["hard_iou"], s) for s in common if s in rows_main[arm]]
        scored.sort()
        n_tail = max(1, int(round(args.tail_frac * len(scored))))
        rows = []
        cnt = Counter()
        for iou, sid in scored[:n_tail]:
            r = rows_main[arm][sid]
            gr = rows_gt[arm].get(sid)
            ch = ceil["chain"].get(sid)
            ba = ceil["basis"].get(sid) if arm == "P1" else None
            ctx_gap = (gr["hard_iou"] - iou) if gr else None
            ar = r["pred_area_frac"] / max(r["gt_mean"], 1e-6)
            buckets = []
            if ch is not None and ch < CHAIN_LO:
                buckets.append("chain_ceiling")
            if ba is not None and ba < BASIS_LO:
                buckets.append("basis_ceiling")
            if ctx_gap is not None and ctx_gap >= CTX_GAP:
                buckets.append("context_missing")
            if abs(ar - 1.0) >= AREA_GAP:
                buckets.append("coverage_bias")
            if not buckets:
                buckets.append("underfit")
            primary = buckets[0]
            cnt[primary] += 1
            rows.append({
                "sample_id": sid, "topk_iou": iou, "gt_context_iou": gr and gr["hard_iou"],
                "context_gap": ctx_gap, "chain_ceiling": ch, "basis_ceiling": ba,
                "area_ratio": ar, "area_direction": "over" if ar > 1 else "under",
                "center_prior": r["center_prior_hard_iou"],
                "random_floor": r["random_floor"],
                "family": r.get("family"), "head": r.get("head"),
                "winner_confidence": r.get("winner_confidence"),
                "labels": labels.get(sid), "buckets": buckets, "primary": primary,
            })
        tail[arm] = rows
        mech[arm] = cnt
        print(f"{arm} tail n={len(rows)} primary={dict(cnt)}", flush=True)

    payload = {
        "split": args.split, "main_context": args.main, "n_common": len(common),
        "convention": "matched-area top-k IoU; normal_only tables are the "
                      "reporting convention (winner_confidence=='normal')",
        "ceilings": {
            "chain": {"applies_to": ["P1", "P3prime"],
                      "scale": "delivery resolution",
                      "definition": "GT -> area_resize(H/16) -> guided_upsample -> "
                                    "top-k -> IoU vs delivery GT",
                      "n": len(ceil["chain"]), "median": _median(ceil["chain"].values())},
            "basis": {"applies_to": ["P1"],
                      "scale": "H/16 grid (same as the board)",
                      "definition": "published Where-A oracle latent decoded at H/16",
                      "note": "P3' never touches Phi-71, so this bound does not "
                              "apply to it and is not reported for it",
                      "n": len(ceil["basis"]), "median": _median(ceil["basis"].values())},
        },
        "six_dim_tables": tables,
        "family_x_area": cross,
        "tail_mechanism_counts": {a: dict(c) for a, c in mech.items()},
        "tail_frac": args.tail_frac,
    }
    (out_dir / "per_class_metrics.json").write_text(json.dumps(payload, indent=2),
                                                    encoding="utf-8")
    with (out_dir / "tail_samples.jsonl").open("w") as fh:
        for arm, rs in tail.items():
            for r in rs:
                fh.write(json.dumps({"arm": arm, **r}) + "\n")
    print(json.dumps({"out": str(out_dir),
                      "tail_mechanism_counts": {a: dict(c) for a, c in mech.items()}},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
