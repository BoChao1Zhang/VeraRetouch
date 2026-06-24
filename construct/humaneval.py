"""Human-eval sheets for the binary F⊕R QA judge (design 人核 — the only real validation).

Synthetic oracles are confounded (grayscale/crush = style not breakage; random presets are
not a reliable 'good' reference). So instead: render varied presets on a few sources, run the
binary judge, and lay out a labeled montage per source with the judge's KEEP/VETO + merit +
one-line reason on each variant. The user eyeballs whether keep/veto matches taste.

CLI:  python -m construct.humaneval [--n 10]  -> writes review/qa_eval_*.png + qa_eval.md
"""
from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw

from dataset_build.source_qa import db
from . import qa, render
from .bank import PresetBank, load_captions
from .recall import recall

FULL = "/home/bc/data/datasets/vera_directionA_1M/preset_bank_full"
REVIEW = os.path.join(FULL, "review")


def _label(im: Image.Image, text: str, color) -> Image.Image:
    bar = 34
    c = Image.new("RGB", (im.width, im.height + bar), (15, 15, 15))
    c.paste(im, (0, bar))
    d = ImageDraw.Draw(c)
    d.text((4, 3), text[:60], fill=color)
    if len(text) > 60:
        d.text((4, 17), text[60:120], fill=color)
    return c


def run(n: int, per: int = 5) -> None:
    os.makedirs(REVIEW, exist_ok=True)
    bank = PresetBank.load(FULL); caps = load_captions()
    conn = db.connect()
    rows = [dict(r) for r in conn.execute(
        "SELECT path, saturation_mean, is_portrait_pool FROM assets WHERE asset_type='image' "
        "AND b_quality=3 AND dup_of IS NULL AND saturation_mean IS NOT NULL ORDER BY asset_id").fetchall()]
    conn.close()
    rows = [r for r in rows if os.path.exists(r["path"])]
    random.Random(7).shuffle(rows)
    # mixed: some colorful, some muted, some portrait
    pick = (rows[:n])
    md = ["# QA judge human-eval (binary F⊕R) — does KEEP/VETO match your eye?\n"]
    for i, r in enumerate(pick):
        src = r["path"]; isp = bool(r["is_portrait_pool"])
        cand = recall(bank, src, k=60)
        chosen = random.Random(i).sample(cand, min(per, len(cand)))
        def _r(c):
            res = render.render_preset(c["path"], c["kind"], c.get("fmt"), src)
            return (c["preset_id"], res["after_path"]) if res.get("ok") else None
        with ThreadPoolExecutor(max_workers=6) as ex:
            variants = [v for v in ex.map(_r, chosen) if v]
        if not variants:
            continue
        qres = qa.qa_rank(src, variants, is_portrait=isp)
        sc = qres["scores"]
        # montage
        tiles = [_label(_open(src), "ORIG", (255, 255, 255))]
        for pid, ap in variants:
            s = sc[pid]
            verdict = "VETO" if s["veto"] else f"KEEP m={s.get('merit','?')}"
            col = (255, 90, 90) if s["veto"] else (120, 255, 120)
            nm = caps.get(pid, {}).get("vlm_name") or ""
            tiles.append(_label(_open(ap), f"{verdict} | {nm} | {s.get('why','')}", col))
        _grid(tiles).save(os.path.join(REVIEW, f"qa_eval_{i:02d}.png"))
        md.append(f"- qa_eval_{i:02d}.png  sat={r['saturation_mean']:.2f} portrait={isp}  "
                  f"ranking: {qres['ranking']}")
        print(f"[{i+1}/{len(pick)}] {os.path.basename(src)[:30]:30s} -> qa_eval_{i:02d}.png "
              f"keep={sum(1 for v in sc.values() if not v['veto'])}/{len(sc)}")
    open(os.path.join(REVIEW, "qa_eval.md"), "w").write("\n".join(md))
    print(f"\n-> {REVIEW}/qa_eval_*.png  (open these and judge KEEP/VETO)")


def _open(p):
    im = Image.open(p).convert("RGB"); im.thumbnail((480, 480)); return im


def _grid(tiles):
    cols = min(3, len(tiles)); rows = (len(tiles) + cols - 1) // cols
    cw = max(t.width for t in tiles); ch = max(t.height for t in tiles)
    g = Image.new("RGB", (cols * cw, rows * ch), (0, 0, 0))
    for k, t in enumerate(tiles):
        g.paste(t, ((k % cols) * cw, (k // cols) * ch))
    return g


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--per", type=int, default=5); a = ap.parse_args()
    run(a.n, a.per)


if __name__ == "__main__":
    main()
