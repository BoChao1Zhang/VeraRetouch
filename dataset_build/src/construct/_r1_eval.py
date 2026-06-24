"""R1 re-eval with REAL VLM captions (scratch). Two things viz.py doesn't cover:
  1. fusion sweep  vlm-caption-emb (+) LAB  — does the caption ADD to LAB, or is
     LAB still the ceiling? (the design's "VL文本嵌入 + 24维LAB" claim, now with
     real high-cardinality captions instead of templated axes)
  2. qualitative cross-modal: source image -> top-8 preset captions by cosine,
     dumped with the actual captions for human/judge eyeball (leakage-free signal).
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from dataset_build.source_qa import db
from . import sf_client, bank

OUT = "/home/bc/data/datasets/vera_directionA_1M/preset_bank_v2"


def unit(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def knn_purity(emb, lab, k=10):
    s = emb @ emb.T; np.fill_diagonal(s, -np.inf)
    nn = np.argpartition(-s, k, axis=1)[:, :k]
    return float((lab[nn] == lab[:, None]).mean())


def main():
    feats = [json.loads(l) for l in open(f"{OUT}/features.jsonl") if l.strip()]
    axby = {f["preset_id"]: f["axes"] for f in feats}
    labz = np.load(f"{OUT}/lab.npz", allow_pickle=True)
    labpos = {i: k for k, i in enumerate(list(labz["ids"]))}
    L = labz["lab"].astype("float32")
    Ln = unit((L - L.mean(0)) / (L.std(0) + 1e-6))

    print("== separability: vlm-caption embedding vs LAB, + fusion sweep ==")
    for tag in ["vlm_plain", "vlm_instr"]:
        z = np.load(f"{OUT}/text_emb.{tag}.npz", allow_pickle=True)
        ids = list(z["ids"]); T = unit(z["emb"].astype("float32"))
        order = [labpos[i] for i in ids]; Lc = Ln[order]
        fam = np.array([axby[i]["grade_family"] for i in ids])
        temp = np.array([axby[i]["temperature"] for i in ids])
        m = fam != "stylized"
        print(f"\n  [{tag}]  (maj fam .771 / noStyl .471 / temp .488)")
        print(f"  {'a=LAB':>6} {'fam':>7} {'fam_noStyl':>11} {'temp':>7}")
        for a in [0.0, 0.3, 0.5, 0.7, 1.0]:
            Fz = unit(np.concatenate([(1 - a) * T, a * Lc], axis=1))
            print(f"  {a:>6.1f} {knn_purity(Fz, fam):>7.3f} {knn_purity(Fz[m], fam[m]):>11.3f} {knn_purity(Fz, temp):>7.3f}")

    # qualitative cross-modal on vlm_plain
    z = np.load(f"{OUT}/text_emb.vlm_plain.npz", allow_pickle=True)
    pids = list(z["ids"]); pemb = unit(z["emb"].astype("float32"))
    caps = bank.load_captions()
    conn = db.connect()
    rows = conn.execute("SELECT path FROM assets WHERE asset_type='image' AND b_quality=3 "
                        "AND dup_of IS NULL ORDER BY asset_id LIMIT 400").fetchall()
    conn.close()
    paths = [r["path"] for r in rows if os.path.exists(r["path"])][::50][:8]
    iemb = sf_client.embed_images(paths)
    sims = iemb @ pemb.T
    out = []
    for i, p in enumerate(paths):
        top = np.argsort(-sims[i])[:8]
        ex = {"source": p, "cos_range": [round(float(sims[i].min()), 3), round(float(sims[i].max()), 3)],
              "top8": [{"preset_id": pids[j], "cos": round(float(sims[i][j]), 3),
                        "family": axby[pids[j]]["grade_family"],
                        "name": caps.get(pids[j], {}).get("vlm_name"),
                        "caption": (caps.get(pids[j], {}).get("vlm_caption") or "")[:120]} for j in top]}
        out.append(ex)
        print(f"\nIMG {os.path.basename(p)[:45]}  cos[{ex['cos_range'][0]}..{ex['cos_range'][1]}]")
        for t in ex["top8"][:4]:
            print(f"   {t['cos']:.3f} {t['family']:13s} {t['name']}")
    json.dump(out, open(f"{OUT}/r1_crossmodal_examples.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n-> {OUT}/r1_crossmodal_examples.json")


if __name__ == "__main__":
    main()
