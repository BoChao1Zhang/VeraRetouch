#!/usr/bin/env python
"""A1 pilot gate: how much of the GT geometry code survives the text path.

Two stages, deliberately separable because only the second needs a GPU:

**stage 1 (CPU, no generation)** -- the model's own ``<where>`` spans are
already cached for every training sample (``GenContextStore``, 159,215 rows), so
the deployable parse can be scored without generating anything: decode the
cached ids, run the 21-d parser over them, and score the result against the
construction-side GT code.  This is the ``F1_A1`` reference line that D-10
requires every "we beat 82%" claim to be stated against, and it is the
denominator of the C-arm rule (§3.2: parsed capture >= 0.7 x GT capture => no C
arm).

**stage 2 (GPU)** -- the D-7 two-pass form: keep pass-1's free reasoning, then
ask the model to restate the geometry behind a forced ``<geom>`` prefix and
parse *that*.  The question it answers is narrow: does a canonical restatement
recover code the free-text parse drops?

A caveat that governed the first board (``--gt vrmeta``): **that GT code is not a
superset of the parsed code**.  ``.vrmeta.json`` carries ``slot_id`` (shape) and
``region`` and *no extent*, so the seven extent slots are identically zero in the
GT vector, and ``region`` is a 3x3 centroid bucket that is 82% the single value
"center" -- a direction column scored against a near-constant.  Extent was
excluded from that score and reported as "not measurable against this GT".

``--gt construction`` (AMD-8) replaces it: the same geometry parameters the v4a
reasoning template read, so direction comes from the axis/compass the annotator
was shown and extent from the gauge buckets, and both are measurable.  The
parameters are not published in ``.vrmeta.json``; they come from the sqlite
sidecar built by ``export_construct_geometry.py``.  ``--gt both`` scores the two
side by side on one sample draw, which is the only way the columns are
comparable.

Usage (stage 1, CPU):
    python -m q3vl.whereb.scripts.run_a1_pilot --stage 1 --n 500 \
        --out experiments/.../where_b/a1_pilot_20260812
    python -m q3vl.whereb.scripts.run_a1_pilot --stage 1 --n 500 --gt both \
        --geom-db /home/bc/data/runs/where_b/construct_geometry.sqlite3 \
        --out experiments/.../where_b/a1_recompute_20260812
"""

from __future__ import annotations

import sqlite3  # import order: sqlite3 before torch -- campaign bug R6

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def _macro_f1(pred: np.ndarray, gt: np.ndarray, cols: list[int]) -> dict[str, Any]:
    """Per-slot F1 at threshold 0.5, macro-averaged over the given columns.

    Slots the GT never lights are dropped from the macro average rather than
    scored as perfect: a slot that is always 0 in the reference is not evidence
    about the extractor, and averaging it in would inflate the score toward 1
    exactly where the code carries no information.
    """
    per_slot, skipped = {}, []
    for c in cols:
        p, g = pred[:, c] >= 0.5, gt[:, c] >= 0.5
        if not g.any():
            skipped.append(c)
            continue
        tp = float((p & g).sum())
        fp = float((p & ~g).sum())
        fn = float((~p & g).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)
        per_slot[c] = {"f1": f1, "support": int(g.sum()),
                       "precision": tp / max(tp + fp, 1e-9),
                       "recall": tp / max(tp + fn, 1e-9)}
    macro = (float(np.mean([v["f1"] for v in per_slot.values()]))
             if per_slot else None)
    return {"macro_f1": macro, "n_slots_scored": len(per_slot),
            "slots_unsupported_in_gt": skipped, "per_slot": per_slot}


class _GeomDB:
    """The construction-side geometry sidecar, keyed by ``candidate_id``."""

    def __init__(self, path: str):
        self.db = sqlite3.connect(f"file:{path}?mode=ro", uri=True,
                                  check_same_thread=False)
        n = self.db.execute("select count(*) from candidate_geometry").fetchone()[0]
        print(f"geom sidecar: {n} candidates ({path})", flush=True)

    def get(self, candidate_id: str) -> dict[str, Any] | None:
        row = self.db.execute(
            "select slot_id, slot_mode, geometry, effective_alpha_mean "
            "from candidate_geometry where candidate_id=?", (candidate_id,)
        ).fetchone()
        if row is None:
            return None
        return {"slot_id": row[0], "slot_mode": row[1],
                "geometry": json.loads(row[2]) if row[2] else None,
                "effective_alpha_mean": row[3]}


def _template_slots(slot_mode: str, geometry: dict[str, Any],
                    size: tuple[float, float] | None) -> np.ndarray:
    """The v4a hint text this candidate produced, run through the text parser.

    The consistency gate: the GT code and the annotator's edit-region hint are
    two renderings of one bucket decision, so parsing the hint back with the
    project's own ``geom_features`` must land on the same slots.  Anything else
    means the code drifted from ``responses.py`` -- which would show up
    downstream as "the extractor got worse", not as a bug.
    """
    from dataset_build.src.construct.responses import _geometry_words

    from q3vl.whereb.amort.geomparse import geom_features
    return geom_features(_geometry_words(slot_mode, geometry, size))


def stage1(args) -> dict[str, Any]:
    from transformers import AutoProcessor

    from q3vl.whereb.amort.geomparse import (GEOM_SLOTS, geom_features,
                                             geom_features_from_construction,
                                             geom_features_from_vrmeta)
    from q3vl.whereb.config import GENCTX_DIR
    from q3vl.whereb.data import open_dataset
    from q3vl.whereb.stores import GenContextStore
    from q3vl.where.maskdata import MaskResolver

    names = [n for n, _ in GEOM_SLOTS]
    grp = {"shape": [i for i, n in enumerate(names) if n.startswith("shape_")],
           "dir": [i for i, n in enumerate(names) if n.startswith("dir_")],
           "ext": [i for i, n in enumerate(names) if n.startswith("ext_")]}
    #: slots ``geom_features_from_construction`` claims to fill; the consistency
    #: gate is scoped to these.  ext_soft/ext_hard/ext_partial and shape_oval are
    #: reachable from the template vocabulary but deliberately not derived (see
    #: the EPR NOTES), and dir_edge is boilerplate inside band/linear.
    claimed = [names.index(n) for n in (
        "dir_left", "dir_right", "dir_top", "dir_bottom", "dir_center",
        "dir_horizontal", "dir_vertical", "dir_diagonal",
        "ext_large", "ext_small", "ext_whole", "ext_moderate")]

    want_vr = args.gt in ("vrmeta", "both")
    want_cs = args.gt in ("construction", "both")
    gdb = _GeomDB(args.geom_db) if want_cs else None

    proc = AutoProcessor.from_pretrained(args.checkpoint)
    tok = proc.tokenizer
    ds, info = open_dataset(args.split, need_mask=True, exclude_low=True)
    rows = ds.meta_rows()
    idx = [i for i, r in enumerate(rows) if r.get("render_mode") == "local"]
    print(f"{args.split}: {len(idx)} local samples", flush=True)

    store = GenContextStore(Path(GENCTX_DIR) / args.split)
    res = MaskResolver(verify="none", suffix=".vrmeta.json")

    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(idx), size=min(args.n, len(idx)), replace=False)

    P, Q, G, C, per_sample = [], [], [], [], []
    n_no_genctx = n_no_vrmeta = n_empty_text = 0
    n_no_cand = n_no_geom_fam = 0
    n_consistent = n_consistency_checked = 0
    t0 = time.time()
    for j, jj in enumerate(pick):
        i = idx[int(jj)]
        rec = ds.record(i)
        sid = rec.get("sample_id") or rows[i].get("sample_id")
        try:
            ids = store.where_ids(sid)
        except Exception:
            n_no_genctx += 1
            continue
        text = tok.decode(ids, skip_special_tokens=True)
        if not text.strip():
            n_empty_text += 1
        try:
            vm = json.loads(res.read_bytes(res.resolve(rec)).decode())
            g = geom_features_from_vrmeta(vm.get("slot_id"), vm.get("region"))
        except Exception:
            n_no_vrmeta += 1
            continue
        if g is None:
            n_no_vrmeta += 1
            continue

        row: dict[str, Any] = {
            "sample_id": sid,
            "family": str(vm.get("slot_id") or "").rsplit("-", 1)[0] or None,
            "text_len": len(text),
        }
        c = cont = conf = None
        if gdb is not None:
            got = gdb.get(str(rec.get("candidate_id") or ""))
            if got is None:
                n_no_cand += 1
            else:
                img = rec.get("image") or {}
                size = None
                if img.get("oriented_w") and img.get("oriented_h"):
                    size = (float(img["oriented_w"]), float(img["oriented_h"]))
                c, cont, conf = geom_features_from_construction(
                    got["slot_id"], got["geometry"], size,
                    alpha_mean=got["effective_alpha_mean"])
                if got["geometry"] is None and got["slot_mode"] != "semantic":
                    n_no_geom_fam += 1
                if got["geometry"] is not None:
                    t = _template_slots(str(got["slot_mode"]), got["geometry"], size)
                    ok = bool(np.array_equal(c[claimed] >= 0.5, t[claimed] >= 0.5))
                    n_consistency_checked += 1
                    n_consistent += int(ok)
                    row["template_consistent"] = ok
                row["cont"] = [round(float(x), 5) for x in cont]
                row["conf"] = [float(x) for x in conf]
                row["gt_construction"] = [int(x) for x in (c >= 0.5)]
        if want_cs and c is None:
            continue

        p = geom_features(text)
        P.append(p)
        # The annotator's own span, parsed the same way.  Not a second arm: it
        # splits the capture loss into "the template word never reached the GT
        # text" and "the model dropped a word the GT text had", which the
        # generated-text column alone cannot separate.
        Q.append(geom_features(rec.get("where") or ""))
        G.append(g)
        if c is not None:
            C.append(c)
        row.update({
            "n_active_parsed": int((p >= 0.5).sum()),
            "n_active_gt": int((g >= 0.5).sum()),
            "abstain": bool((p >= 0.5).sum() == 0),
        })
        if c is not None:
            row["n_active_gt_construction"] = int((c >= 0.5).sum())
        per_sample.append(row)
        if args.progress and j % 100 == 0:
            print(f"  {j}/{len(pick)}  ({time.time()-t0:.0f}s)", flush=True)

    P, Q, G = np.asarray(P), np.asarray(Q), np.asarray(G)
    C = np.asarray(C) if C else None
    out: dict[str, Any] = {
        "stage": 1,
        "what": "parsed code from the model's OWN cached <where> vs GT code(s)",
        "gt": args.gt,
        "n_requested": int(args.n), "n_scored": int(len(P)),
        "n_missing_genctx": n_no_genctx, "n_missing_vrmeta": n_no_vrmeta,
        "n_missing_candidate_row": n_no_cand,
        "n_geometry_family_without_geometry": n_no_geom_fam,
        "n_empty_text": n_empty_text,
        "split": args.split, "seed": args.seed,
    }
    if want_vr:
        for name, cols in grp.items():
            out[f"capture_{name}"] = _macro_f1(P, G, cols)
        out["capture_overall_shape_dir"] = _macro_f1(P, G, grp["shape"] + grp["dir"])
        out["extent_note"] = (
            "extent slots are identically zero in the GT (.vrmeta carries slot_id "
            "and region only), so capture_ext is NOT a measure of extent quality; "
            "the parser's extent output is reported as an activation rate instead")
    if C is not None:
        for name, cols in grp.items():
            out[f"capture_{name}_construction"] = _macro_f1(P, C, cols)
        out["capture_overall_shape_dir_construction"] = _macro_f1(
            P, C, grp["shape"] + grp["dir"])
        out["capture_overall_all_construction"] = _macro_f1(
            P, C, grp["shape"] + grp["dir"] + grp["ext"])
        for name, cols in grp.items():
            out[f"capture_{name}_construction_gt_text"] = _macro_f1(Q, C, cols)
        out["gt_text_note"] = (
            "capture_*_construction_gt_text scores the ANNOTATOR's own <where> "
            "span against the construction code -- the ceiling the extraction "
            "arms inherit, not a result of this run")
        out["gt_construction_activation_rate"] = {
            n: float((C[:, k] >= 0.5).mean()) for k, n in enumerate(names)}
        out["gt_construction_extent_nonzero_rate"] = float(
            (C[:, grp["ext"]] >= 0.5).any(axis=1).mean())
        out["gt_construction_dir_nonzero_rate"] = float(
            (C[:, grp["dir"]] >= 0.5).any(axis=1).mean())
        out["template_consistency"] = {
            "n_checked": n_consistency_checked,
            "n_consistent": n_consistent,
            "rate": (n_consistent / n_consistency_checked
                     if n_consistency_checked else None),
            "slots": [names[k] for k in claimed],
        }
    if want_vr:
        out["gt_vrmeta_activation_rate"] = {
            n: float((G[:, k] >= 0.5).mean()) for k, n in enumerate(names)}
    out["extent_activation_rate_parsed"] = float(
        (P[:, grp["ext"]] >= 0.5).any(axis=1).mean()) if len(P) else None
    out["abstain_rate"] = float(np.mean([r["abstain"] for r in per_sample])) \
        if per_sample else None
    out["mean_active_parsed"] = float(np.mean([r["n_active_parsed"]
                                               for r in per_sample])) \
        if per_sample else None
    out["mean_active_gt"] = float(np.mean([r["n_active_gt"]
                                           for r in per_sample])) \
        if per_sample else None
    if C is not None:
        out["mean_active_gt_construction"] = float(np.mean(
            [r["n_active_gt_construction"] for r in per_sample
             if "n_active_gt_construction" in r]))
    return out, per_sample


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=[1])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--checkpoint",
                    default="/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976")
    ap.add_argument("--gt", default="vrmeta",
                    choices=["vrmeta", "construction", "both"],
                    help="which GT code to score against; 'vrmeta' reproduces "
                         "EPR-004, 'construction' is the AMD-8 code")
    ap.add_argument("--geom-db",
                    default="/home/bc/data/runs/where_b/construct_geometry.sqlite3",
                    help="sidecar from export_construct_geometry.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args(argv)
    if args.gt != "vrmeta" and not Path(args.geom_db).is_file():
        ap.error(f"--geom-db not found: {args.geom_db} "
                 "(build it with export_construct_geometry.py)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    board, per_sample = stage1(args)
    (out_dir / "a1_pilot_stage1.json").write_text(
        json.dumps(board, indent=2, default=str), encoding="utf-8")
    with (out_dir / "a1_pilot_stage1_per_sample.jsonl").open("w") as fh:
        for r in per_sample:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps({k: board[k] for k in
                      ("n_scored", "abstain_rate", "mean_active_parsed",
                       "mean_active_gt")}, indent=2))
    for k in ("capture_shape", "capture_dir", "capture_ext",
              "capture_overall_shape_dir",
              "capture_shape_construction", "capture_dir_construction",
              "capture_ext_construction",
              "capture_overall_shape_dir_construction",
              "capture_overall_all_construction",
              "capture_shape_construction_gt_text",
              "capture_dir_construction_gt_text",
              "capture_ext_construction_gt_text"):
        if k in board and board[k] is not None:
            print(f"{k}: macro_f1={board[k]['macro_f1']} "
                  f"(slots scored {board[k]['n_slots_scored']})")
    if board.get("template_consistency"):
        print("template_consistency: " + json.dumps(
            {k: v for k, v in board["template_consistency"].items()
             if k != "slots"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
