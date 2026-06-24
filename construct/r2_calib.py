"""R2 step 1: QA-oracle calibration + mini-relevance (the critic's highest-value run).

Per source, 4 arms (deduped, each preset rendered once, QA-scored pairwise vs source):
  reranker : LAB-recall(k=POOL) -> VL-rerank top-3            [candidate architecture]
  mismatch : aggressive OPPOSITE-temperature preset           [QA-oracle negative control]
  random   : 3 random from the recall pool                    [relevance floor]
  vlemb    : centered cross-modal (source-img . preset-text) top-3  [VL-embedding ABLATION]

Gates:
  (a) QA-ORACLE VALID  : mismatch.q < min(reranker.q) in >= 80% sources (else QA metric is noise).
  (b) reranker vs random : mean best-q(reranker) - mean best-q(random)  (early relevance read).
  (c) reranker vs vlemb  : confirms VL-embedding recall is not better than the reranker pipeline.
  (d) pool width        : how often a reranker pick has recall-rank > POOL//2 (reachability).

CLI:  python -m construct.r2_calib [--n 20] [--pool 80] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from skimage.color import rgb2lab

from dataset_build.source_qa import db
from . import sf_client, qa, render
from .bank import PresetBank, load_captions, doc_text
from .recall import recall
from .rerank import rerank_candidates

FULL = "/home/bc/data/datasets/vera_directionA_1M/preset_bank_full"
_SAT_MIN = 0.0  # set by --sat-min; >0 forces a colorful-only source sample
RERANK_INSTR = ("Rank these editing presets by how well their color grading aesthetically "
                "enhances the given photo.")
# R2 iterates this (goal). v2 targets the observed BW/over-aggressive bias.
RERANK_INSTR_V2 = ("Rank these color-grading presets by how well each ENHANCES this photo while "
                   "PRESERVING and enriching its existing colors and subject. Strongly prefer "
                   "natural, flattering grades that suit the scene; AVOID black-and-white or heavy "
                   "desaturation unless the photo is already monochrome, and avoid harsh, "
                   "over-processed or muddy looks.")
_RERANK_INSTR = RERANK_INSTR


def _img_warmth(path: str) -> float:
    im = Image.open(path).convert("RGB"); im.thumbnail((256, 256))
    return float(rgb2lab(np.asarray(im, "float32") / 255.0)[..., 2].mean())  # mean b* (warm=high)


def _pick_sources(n: int, sat_min: float = 0.0) -> list:
    """Stratified: portrait/non-portrait x saturation band, plus a few sat<0.05 grayscale.
    sat_min>0 forces a colorful-only sample (closes the high-sat gap)."""
    conn = db.connect()
    rows = [dict(r) for r in conn.execute(
        "SELECT asset_id, path, is_portrait_pool, saturation_mean, mean_luma FROM assets "
        "WHERE asset_type='image' AND b_quality=3 AND dup_of IS NULL AND saturation_mean IS NOT NULL "
        "ORDER BY asset_id").fetchall()]
    conn.close()
    rows = [r for r in rows if os.path.exists(r["path"]) and r["saturation_mean"] >= sat_min]
    rng = random.Random(0); rng.shuffle(rows)
    if sat_min > 0:  # colorful-only: just take a portrait/non-portrait mix
        port = [r for r in rows if r["is_portrait_pool"]]
        land = [r for r in rows if not r["is_portrait_pool"]]
        out = (port[:n // 2] + land)[:n]
        return out
    gray = [r for r in rows if r["saturation_mean"] < 0.05][:max(2, n // 10)]
    muted = [r for r in rows if 0.05 <= r["saturation_mean"] < 0.20]
    normal = [r for r in rows if r["saturation_mean"] >= 0.20]
    port = [r for r in normal if r["is_portrait_pool"]]
    land = [r for r in normal if not r["is_portrait_pool"]]
    out, seen = [], set()
    for bucket, k in [(gray, len(gray)), (muted, max(3, n // 4)), (port, n // 3), (land, n)]:
        for r in bucket:
            if r["asset_id"] not in seen:
                out.append(r); seen.add(r["asset_id"])
            if len(out) >= n:
                break
        if len(out) >= n:
            break
    return out[:n]


def _mismatch(bank: PresetBank, pool: list, warmth: float) -> dict:
    """Aggressive wrong-direction preset: opposite temperature to the photo's warmth, among
    the highest color-movers in the recall pool (so the error is conspicuous)."""
    want = "cool" if warmth > 5 else "warm" if warmth < -2 else "cool"
    cands = [c for c in pool if c["axes"].get("temperature") == want] or pool
    return max(cands, key=lambda c: c["score"])


def run(n: int, pool_k: int, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    bank = PresetBank.load(FULL)                      # lab + feats (no emb needed for recall)
    caps = load_captions()
    # VL-embedding ablation: centered cross-modal
    z = np.load(os.path.join(FULL, "text_emb.vlm_plain.npz"), allow_pickle=True)
    pemb = z["emb"].astype("float32"); pmu = pemb.mean(0)
    pemb_c = pemb - pmu; pemb_c /= np.linalg.norm(pemb_c, axis=1, keepdims=True) + 1e-9
    pid2idx = {pid: i for i, pid in enumerate(bank.ids)}

    srcs = _pick_sources(n, sat_min=_SAT_MIN)
    iemb = sf_client.embed_images([s["path"] for s in srcs])
    imu = iemb.mean(0)
    iemb_c = iemb - imu; iemb_c /= np.linalg.norm(iemb_c, axis=1, keepdims=True) + 1e-9

    results = []
    for si, s in enumerate(srcs):
        src = s["path"]; warmth = _img_warmth(src)
        cand = recall(bank, src, k=pool_k)
        rank_of = {c["preset_id"]: r for r, c in enumerate(cand)}
        rr1 = rerank_candidates(src, cand, bank, captions=caps, top=3, instruction=RERANK_INSTR)
        rr2 = rerank_candidates(src, cand, bank, captions=caps, top=3, instruction=RERANK_INSTR_V2)
        rnd = random.Random(si).sample(cand, min(3, len(cand)))
        # vlemb ablation top-3 over full bank
        sims = pemb_c @ iemb_c[si]
        vtop = [str(bank.ids[j]) for j in np.argsort(-sims)[:3]]
        arms = {
            "reranker_v1": [c["preset_id"] for c in rr1],
            "reranker_v2": [c["preset_id"] for c in rr2],   # anti-BW instruction iteration
            "random": [c["preset_id"] for c in rnd],
            "vlemb": vtop,
        }
        # dedup presets to render
        uniq = {}
        for pid in {p for a in arms.values() for p in a}:
            i = pid2idx.get(pid)
            if i is not None:
                uniq[pid] = bank.feats[i]
        def _r(pid_feat):
            pid, f = pid_feat
            res = render.render_preset(f["path"], f["kind"], f.get("fmt"), src)
            return pid, (res["after_path"] if res.get("ok") else None)
        with ThreadPoolExecutor(max_workers=6) as ex:
            rendered = dict(ex.map(_r, uniq.items()))
        variants = [(pid, p) for pid, p in rendered.items() if p]
        qres = qa.qa_rank(src, variants, is_portrait=bool(s.get("is_portrait_pool"))) if variants else {"ranking": [], "scores": {}}
        sc = qres["scores"]
        def arm_q(pids):  # best q among an arm's rendered, vetoed -> -99
            qs = [sc[p]["q"] if (p in sc and not sc[p]["veto"]) else -99 for p in pids if p in rendered and rendered[p]]
            return max(qs) if qs else -99
        rec = {
            "source": src, "scene": "portrait" if s["is_portrait_pool"] else "other",
            "sat": round(s["saturation_mean"], 3), "warmth": round(warmth, 1),
            "arms": arms,
            "best_q": {a: arm_q(p) for a, p in arms.items()},
            "recall_rank": {a: [rank_of.get(p) for p in arms[a]] for a in arms},
            "scores": {p: sc.get(p) for p in sc},
            "captions": {p: (caps.get(p, {}).get("vlm_name")) for a in arms.values() for p in a},
        }
        results.append(rec)
        bq = rec["best_q"]
        print(f"[{si+1}/{len(srcs)}] {os.path.basename(src)[:26]:26s} sat={rec['sat']:.2f} "
              f"| rr1={bq['reranker_v1']:+.0f} rr2={bq['reranker_v2']:+.0f} "
              f"rand={bq['random']:+.0f} vlemb={bq['vlemb']:+.0f}")

    arm_names = list(results[0]["arms"]) if results else []
    def mean(xs): return round(float(np.mean(xs)), 3) if xs else None
    summary = {
        "n": len(results), "pool_k": pool_k,
        "mean_best_q": {a: mean([r["best_q"][a] for r in results if r["best_q"][a] > -90]) for a in arm_names},
        "vs_random": {a: mean([r["best_q"][a] - r["best_q"]["random"] for r in results
                               if r["best_q"][a] > -90 and r["best_q"]["random"] > -90]) for a in arm_names},
        "by_band": {band: {a: mean([r["best_q"][a] for r in results if lo <= r["sat"] < hi and r["best_q"][a] > -90]) for a in arm_names}
                    for band, lo, hi in [("low", 0, 0.15), ("mid", 0.15, 0.30), ("hi", 0.30, 9)]},
    }
    json.dump({"summary": summary, "results": results}, open(os.path.join(out_dir, "r2_calib.json"), "w"),
              ensure_ascii=False, indent=1)
    print("\n=== R2 CALIB SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"-> {out_dir}/r2_calib.json")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--pool", type=int, default=80)
    ap.add_argument("--out", default=FULL)
    ap.add_argument("--sat-min", type=float, default=0.0)
    ap.add_argument("--rerank-instr", choices=["v1", "v2"], default="v1")
    a = ap.parse_args()
    global _SAT_MIN, _RERANK_INSTR
    _SAT_MIN = a.sat_min
    _RERANK_INSTR = RERANK_INSTR_V2 if a.rerank_instr == "v2" else RERANK_INSTR
    run(a.n, a.pool, a.out)


if __name__ == "__main__":
    main()
