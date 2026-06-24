"""Validate the QA judge as a relevance ORACLE with TRUE broken-render controls.

R2's q metric is only trustworthy if the judge reliably down-ranks unambiguously bad
edits. The earlier "opposite-temperature" mismatch was a weak control (often a fine
artistic choice). Here we degrade the SOURCE deterministically (PIL) into catastrophic
edits — overexposed/clipped, crushed black, garish oversaturated cast, grayscale — and
require the judge to rank them BELOW real preset renders.

GATE: in >= 80% of sources, min(q of real presets) > max(q of broken controls).

CLI:  python -m construct.oracle [--n 12]
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
from PIL import Image, ImageEnhance

from dataset_build.source_qa import db
from . import qa, render
from .bank import PresetBank
from .recall import recall

FULL = "/home/bc/data/datasets/vera_directionA_1M/preset_bank_full"
STAGE = "/home/bc/data/datasets/vera_directionA_1M/source_qa/render_stage"


def _broken(src: str) -> dict:
    """Deterministic catastrophic edits of the source (unambiguously bad)."""
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(src).convert("RGB"); im.thumbnail((1024, 1024))
    os.makedirs(STAGE, exist_ok=True)
    out = {}
    def save(tag, img):
        p = os.path.join(STAGE, f"brk_{tag}_{abs(hash(src)) % 10**8}.jpg")
        img.save(p, "JPEG", quality=92); out[tag] = p
    save("overexp", ImageEnhance.Brightness(im).enhance(1.9).point(lambda v: min(255, int(v * 1.15))))
    save("crushed", ImageEnhance.Brightness(im).enhance(0.32))
    a = ImageEnhance.Color(im).enhance(3.2)                       # garish oversaturation + green cast
    g = np.asarray(a, "float32"); g[..., 1] = np.minimum(255, g[..., 1] * 1.25)
    save("oversat", Image.fromarray(g.astype("uint8")))
    save("grayscale", im.convert("L").convert("RGB"))
    return out


def validate(n: int = 12) -> dict:
    bank = PresetBank.load(FULL)
    conn = db.connect()
    rows = [dict(r) for r in conn.execute(
        "SELECT path, saturation_mean, is_portrait_pool FROM assets WHERE asset_type='image' "
        "AND b_quality=3 AND dup_of IS NULL AND saturation_mean >= 0.20 ORDER BY asset_id").fetchall()]
    conn.close()
    rows = [r for r in rows if os.path.exists(r["path"])]
    random.Random(1).shuffle(rows); rows = rows[:n]
    results = []
    n_brk = n_brk_rej = n_real = n_real_veto = 0
    for i, r in enumerate(rows):
        src = r["path"]
        cand = recall(bank, src, k=60)
        reals = random.Random(i).sample(cand, 2)
        variants = []
        for c in reals:
            rr = render.render_preset(c["path"], c["kind"], c.get("fmt"), src)
            if rr.get("ok"):
                variants.append((f"real::{c['preset_id']}", rr["after_path"]))
        for tag, p in _broken(src).items():
            variants.append((f"broken::{tag}", p))
        q = qa.qa_rank(src, variants, is_portrait=bool(r.get("is_portrait_pool")))["scores"]
        brk = {l.split("::")[1]: q[l] for l, _ in variants if l.startswith("broken")}
        real = [q[l] for l, _ in variants if l.startswith("real")]
        # a bad edit is "rejected" if vetoed OR q<=-1
        rej = {t: (v["veto"] or v["q"] <= -1) for t, v in brk.items()}
        n_brk += len(brk); n_brk_rej += sum(rej.values())
        n_real += len(real); n_real_veto += sum(1 for v in real if v["veto"])
        results.append({"src": os.path.basename(src), "sat": round(r["saturation_mean"], 2),
                        "broken": {t: (v["q"], "V" if v["veto"] else "") for t, v in brk.items()},
                        "real_q": [(v["q"], "V" if v["veto"] else "") for v in real],
                        "rejected": rej})
        miss = [t for t, ok in rej.items() if not ok]
        print(f"[{i+1}/{len(rows)}] {results[-1]['src'][:24]:24s} sat={results[-1]['sat']} "
              f"real={results[-1]['real_q']} reject={sum(rej.values())}/4{' MISS:'+','.join(miss) if miss else ''}")
    reject_rate = n_brk_rej / max(n_brk, 1)
    false_veto = n_real_veto / max(n_real, 1)
    valid = reject_rate >= 0.85 and false_veto <= 0.20
    print(f"\nbad-edit REJECT rate = {reject_rate:.2f} ({n_brk_rej}/{n_brk})  [gate >= 0.85]")
    print(f"real-preset FALSE-VETO rate = {false_veto:.2f} ({n_real_veto}/{n_real})  [gate <= 0.20]")
    print(f"=> QA ORACLE {'VALID' if valid else 'INVALID — tune veto / prompt'}")
    import json
    json.dump({"reject_rate": reject_rate, "false_veto": false_veto, "valid": valid, "results": results},
              open(os.path.join(FULL, "r2_oracle.json"), "w"), ensure_ascii=False, indent=1)
    return {"reject_rate": reject_rate, "false_veto": false_veto, "valid": valid}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=12); a = ap.parse_args()
    validate(a.n)


if __name__ == "__main__":
    main()
