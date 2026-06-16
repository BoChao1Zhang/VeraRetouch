"""Auto-gate: combine NR-IQA + questionnaire A/B + cheap deterministic detectors
into a non-destructive `auto_verdict` (keep | drop | review) suggestion per image.
Never writes a final human decision — humans confirm via the UI.

A shared PREAMBLE runs first in BOTH modes and is ABSOLUTE (never relativized):
  * resolution floor (megapixels / longedge, portrait-aware)   -> drop   [QA-1]
  * portrait pool needs a usable face                          -> drop   [QA-6]
  * missing validity/suitability signal (pass_a/pass_b NULL)   -> review [QA-2 fail-closed]
Then per-mode logic decides keep/drop/review from the (now non-NULL) signals.

Run: python -m dataset_build.source_qa.gate [--corpus C] [--mode relative|absolute] [--apply-auto-decisions]
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional

from . import config, db

G = config.GATE


# --------------------------------------------------------------------------- #
# shared absolute preamble
# --------------------------------------------------------------------------- #
def _preamble(a) -> Optional[tuple]:
    is_portrait = bool(a["is_portrait_pool"])
    min_mp = G["min_megapixels_portrait"] if is_portrait else G["min_megapixels"]
    min_le = G["min_longedge_portrait"] if is_portrait else G["min_longedge"]
    if a["megapixels"] is not None and a["megapixels"] < min_mp:
        return "drop", f"megapixels<{min_mp}"
    if a["width"] is not None and a["height"] is not None and max(a["width"], a["height"]) < min_le:
        return "drop", f"longedge<{min_le}"
    if is_portrait and a["max_face_frac"] is not None and a["max_face_frac"] < G["min_face_frac"]:
        return "drop", f"portrait: face<{G['min_face_frac']}"
    # a definitive invalid-photo verdict drops even if suitability is incomplete
    if a["pass_a"] == 0:
        return "drop", "PASS_A=0 (invalid photo)"
    # fail-CLOSED: never let a missing validity/suitability signal slip through as keep
    if a["pass_a"] is None or a["pass_b"] is None:
        return "review", "incomplete LLM QA (NULL pass_a/pass_b)"
    return None


# --------------------------------------------------------------------------- #
# absolute mode
# --------------------------------------------------------------------------- #
def _hard_iqa_fail(a) -> Optional[str]:
    if a["musiq"] is not None and a["musiq"] < G["musiq_drop_below"]:
        return f"musiq<{G['musiq_drop_below']}"
    if a["niqe"] is not None and a["niqe"] > G["niqe_drop_above"]:
        return f"niqe>{G['niqe_drop_above']}"
    if a["brisque"] is not None and a["brisque"] > G["brisque_drop_above"]:
        return f"brisque>{G['brisque_drop_above']}"
    if a["sharpness"] is not None and a["sharpness"] < G["laplacian_drop_below"]:
        return f"sharpness<{G['laplacian_drop_below']}"
    if a["noise_sigma"] is not None and a["noise_sigma"] > G["noise_sigma_drop_above"]:
        return f"noise>{G['noise_sigma_drop_above']}"
    return None


def _iqa_keep_band(a) -> bool:
    if a["musiq"] is not None and a["musiq"] >= G["musiq_keep_above"]:
        return True
    if a["clipiqa"] is not None and a["clipiqa"] >= G["clipiqa_keep_above"]:
        return True
    if a["aesthetic_vlm"] is not None and a["aesthetic_vlm"] >= G["aesthetic_vlm_keep_above"]:
        return True
    return False


def verdict_for(a) -> tuple:
    """ABSOLUTE mode: fixed config.GATE thresholds. Returns (verdict, reason)."""
    pre = _preamble(a)
    if pre:
        return pre
    if a["pass_a"] == 0:
        return "drop", "PASS_A=0 (invalid photo)"
    hard = _hard_iqa_fail(a)
    if hard:
        return "drop", f"hard IQA fail: {hard}"
    if a["pass_b"] == 0:
        return "drop", "PASS_B=0 (low quality / unsuitable source)"
    if a["pass_a"] == 1 and a["pass_b"] == 1 and _iqa_keep_band(a):
        return "keep", "PASS_A&B + IQA keep-band"
    return "review", "borderline / incomplete signals"


# --------------------------------------------------------------------------- #
# relative mode (per-corpus calibrated tails + multi-metric voting)
# --------------------------------------------------------------------------- #
_MCOL = {"musiq": "musiq", "clipiqa+": "clipiqa", "niqe": "niqe",
         "brisque": "brisque", "laplacian": "sharpness"}


def verdict_relative(a, thr: dict, drop_min_votes: int = 2, keep_min_good: int = 2) -> tuple:
    pre = _preamble(a)
    if pre:
        return pre
    if a["pass_a"] == 0:
        return "drop", "PASS_A=0 (invalid photo)"
    table = thr.get(a["corpus"]) or thr.get("*") or {}
    bad, good, bad_metrics = 0, 0, []
    for metric, col in _MCOL.items():
        t = table.get(metric)
        v = a[col]
        if t is None or v is None:
            continue
        higher = t["direction"] == 1
        is_bad = (v < t["drop_value"]) if higher else (v > t["drop_value"])
        is_good = (v >= t["keep_value"]) if higher else (v <= t["keep_value"])
        if is_bad:
            bad += 1; bad_metrics.append(metric)
        elif is_good:
            good += 1
    # soft keep votes from aesthetic + graded quality (previously zero-weight)
    if a["aesthetic_vlm"] is not None and a["aesthetic_vlm"] >= G["aesthetic_vlm_keep_above"]:
        good += 1
    if a["b_quality"] is not None and a["b_quality"] >= 3:
        good += 1
    if bad >= drop_min_votes:
        return "drop", f"{bad} metrics in bad tail: {','.join(bad_metrics)}"
    if a["pass_b"] == 0:
        return "drop", "PASS_B=0 (unsuitable source)"
    if a["pass_a"] == 1 and a["pass_b"] == 1 and bad == 0 and good >= keep_min_good:
        return "keep", f"PASS_A&B + {good} good votes"
    return "review", f"borderline (bad={bad}, good={good})"


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run(corpus: Optional[str] = None, apply_auto_decisions: bool = False,
        mode: str = "relative", drop_min_votes: int = 2, keep_min_good: int = 2) -> dict:
    conn = db.connect()
    run_id = db.start_run(conn, "gate", {"corpus": corpus, "apply_auto": apply_auto_decisions, "mode": mode})
    thr = {}
    if mode == "relative":
        from .calibrate import load_thresholds
        thr = load_thresholds(conn)
        if not thr:
            print("[gate] no calibrated thresholds -> run `calibrate` first; falling back to absolute",
                  file=sys.stderr)
            mode = "absolute"
    # widen vs the old query: also surface images that have IQA (megapixels) but
    # no LLM yet, so the resolution floor + fail-closed review reach them (FLOW-6).
    where = ["asset_type='image'", "dup_of IS NULL",   # heads only; siblings inherit at apply
             "(musiq IS NOT NULL OR pass_a IS NOT NULL OR megapixels IS NOT NULL)"]
    params: list = []
    if corpus:
        where.append("corpus=?"); params.append(corpus)
    rows = conn.execute(f"SELECT * FROM assets WHERE {' AND '.join(where)}", params).fetchall()
    print(f"[gate] {len(rows)} images to gate (mode={mode})", file=sys.stderr)

    counts = {"keep": 0, "drop": 0, "review": 0}
    for i, a in enumerate(rows):
        if mode == "relative":
            v, reason = verdict_relative(a, thr, drop_min_votes, keep_min_good)
        else:
            v, reason = verdict_for(a)
        counts[v] += 1
        status = {"keep": "auto_pass", "drop": "auto_fail", "review": "needs_review"}[v]
        db.update_asset_fields(conn, a["asset_id"], auto_verdict=v, status=status)
        db.log_event(conn, a["asset_id"], "gate", "ok", {"verdict": v, "reason": reason}, run_id)
        if apply_auto_decisions and v in ("keep", "drop"):
            db.add_decision(conn, a["asset_id"], v, "auto:gate", reason)
        if (i + 1) % 2000 == 0:
            conn.commit()
    conn.commit()
    db.finish_run(conn, run_id, counts)
    conn.close()
    print({"counts": counts, "run_id": run_id})
    return {"counts": counts, "run_id": run_id}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--mode", choices=["relative", "absolute"], default="relative",
                    help="relative = per-corpus calibrated tails + voting (needs `calibrate` first)")
    ap.add_argument("--drop-min-votes", type=int, default=2)
    ap.add_argument("--keep-min-good", type=int, default=2)
    ap.add_argument("--apply-auto-decisions", action="store_true",
                    help="also write auto:gate keep/drop decisions (still overridable by humans)")
    args = ap.parse_args()
    run(corpus=args.corpus, apply_auto_decisions=args.apply_auto_decisions, mode=args.mode,
        drop_min_votes=args.drop_min_votes, keep_min_good=args.keep_min_good)


if __name__ == "__main__":
    main()
