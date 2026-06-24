"""R1 gate: is the VL text embedding of preset docs SEPARABLE by grade_family?

For each text_emb.<tag>.npz we report, on the FULL 4096-d embedding:
  - silhouette (cosine) by grade_family and by temperature
  - kNN purity@k  vs the majority-class baseline  (the honest "did we beat
    guessing the biggest family?" number — separability only counts if purity
    >> majority frac)
and render a UMAP scatter coloured by grade_family.

Gate (design §4 R1): families must visibly cluster AND kNN purity must clear the
majority baseline by a real margin. Numbers are reported; the go/kill call is the
human eyeball on the UMAP + the purity lift. ponytail: no auto-threshold pretending
to be judgment.

CLI:  python -m construct.viz [--out DIR] [--tags rich_plain,rich_instr,axes_plain] [--k 10]
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
from sklearn.metrics import silhouette_score


def _labels(out_dir: str) -> Dict[str, dict]:
    feats = [json.loads(l) for l in open(os.path.join(out_dir, "features.jsonl")) if l.strip()]
    return {f["preset_id"]: f["axes"] for f in feats}


def _knn_purity(emb: np.ndarray, lab: np.ndarray, k: int) -> float:
    """Mean fraction of each point's k nearest cosine neighbours sharing its label."""
    sims = emb @ emb.T          # emb is L2-normalized -> cosine
    np.fill_diagonal(sims, -np.inf)
    nn = np.argpartition(-sims, kth=k, axis=1)[:, :k]
    same = (lab[nn] == lab[:, None])
    return float(same.mean())


def _metrics(emb: np.ndarray, axes: List[dict], k: int) -> dict:
    out: Dict[str, object] = {"n": len(emb)}
    for axis in ("grade_family", "temperature"):
        lab = np.array([a.get(axis) or "unknown" for a in axes])
        uniq, counts = np.unique(lab, return_counts=True)
        maj = float(counts.max() / counts.sum())
        sil = float(silhouette_score(emb, lab, metric="cosine")) if len(uniq) > 1 else 0.0
        out[axis] = {"silhouette": round(sil, 4),
                     "knn_purity": round(_knn_purity(emb, lab, k), 4),
                     "majority_frac": round(maj, 4),
                     "lift": round(_knn_purity(emb, lab, k) - maj, 4),
                     "classes": dict(zip(uniq.tolist(), counts.tolist()))}
    return out


def _umap_png(emb: np.ndarray, lab: np.ndarray, path: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap
    xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0).fit_transform(emb)
    plt.figure(figsize=(8, 7))
    for c in sorted(set(lab.tolist())):
        s = lab == c
        plt.scatter(xy[s, 0], xy[s, 1], s=6, alpha=0.6, label=f"{c} ({s.sum()})")
    plt.legend(markerscale=2, fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def crossmodal(out_dir: str, emb_tag: str, n_images: int = 12, k: int = 8) -> dict:
    """The real go/kill probe: does the SOURCE-IMAGE embedding land meaningfully
    against the PRESET-TEXT embeddings in the unified VL space?

    Family/temperature silhouette is partly TRIVIAL — the doc text literally
    contains the family/temperature word, so the text side self-separates by token
    leakage regardless of whether the VL space is useful cross-modally. This probe
    sidesteps that. We check, label-free:
      - non-degeneracy: per-image cosine spread (std≈0 => collapsed => KILL)
      - discrimination: do different images retrieve different presets? mean
        Jaccard overlap of top-k across image pairs (≈1.0 => image ignored => KILL)
    Plus per-image top-5 docs for the eyeball.
    """
    from dataset_build.source_qa import db
    from . import bank, sf_client
    z = np.load(os.path.join(out_dir, f"text_emb.{emb_tag}.npz"), allow_pickle=True)
    pids, pemb = z["ids"], z["emb"].astype("float32")
    feats = {f["preset_id"]: f for f in (json.loads(l) for l in
             open(os.path.join(out_dir, "features.jsonl")) if l.strip())}
    docs = [bank.doc_text(feats[p], "rich") for p in pids]
    conn = db.connect()
    rows = conn.execute(
        "SELECT path FROM assets WHERE asset_type='image' AND b_quality=3 "
        "AND dup_of IS NULL ORDER BY asset_id LIMIT %s", (n_images * 40,)).fetchall()
    conn.close()
    paths = [r["path"] for r in rows if os.path.exists(r["path"])][::40][:n_images]
    iemb = sf_client.embed_images(paths)
    sims = iemb @ pemb.T                         # [n_img, n_preset], both L2-normed
    topk = np.argsort(-sims, axis=1)[:, :k]
    # discrimination: mean pairwise Jaccard of top-k sets
    jac = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            a, b = set(topk[i].tolist()), set(topk[j].tolist())
            jac.append(len(a & b) / len(a | b))
    out = {"n_images": len(paths), "emb_tag": emb_tag,
           "cos_mean": round(float(sims.mean()), 4), "cos_std": round(float(sims.std()), 4),
           "per_image_std_mean": round(float(sims.std(axis=1).mean()), 4),
           "topk_jaccard_mean": round(float(np.mean(jac)) if jac else 0.0, 4)}
    print(f"[crossmodal {emb_tag}] cos={out['cos_mean']:+.3f}±{out['cos_std']:.3f} "
          f"per-img spread={out['per_image_std_mean']:.3f} top{k}-overlap={out['topk_jaccard_mean']:.3f}")
    for i, p in enumerate(paths[:4]):
        print(f"  IMG {os.path.basename(p)[:40]}: top-3 -> " +
              " | ".join(f"{feats[pids[j]]['axes']['grade_family']}/{feats[pids[j]]['axes']['temperature']}"
                         f"({sims[i,j]:.2f})" for j in topk[i][:3]))
    return out


def run(out_dir: str, tags: List[str], k: int = 10) -> dict:
    axes_by = _labels(out_dir)
    report = {}
    for tag in tags:
        p = os.path.join(out_dir, f"text_emb.{tag}.npz")
        if not os.path.exists(p):
            print(f"[skip] {p} missing"); continue
        z = np.load(p, allow_pickle=True)
        ids, emb = z["ids"], z["emb"].astype("float32")
        axes = [axes_by[i] for i in ids]
        m = _metrics(emb, axes, k)
        report[tag] = m
        fam = np.array([a.get("grade_family") or "unknown" for a in axes])
        _umap_png(emb, fam, os.path.join(out_dir, f"r1_umap_{tag}.png"), f"grade_family — {tag}")
        gf = m["grade_family"]
        print(f"[{tag:18s}] family: sil={gf['silhouette']:+.3f} purity@{k}={gf['knn_purity']:.3f} "
              f"(majority={gf['majority_frac']:.3f}, lift={gf['lift']:+.3f})  [NOTE: family/temp "
              "words leak into doc text -> separability partly trivial]")
        try:
            m["crossmodal"] = crossmodal(out_dir, tag)
        except Exception as e:  # noqa: BLE001 - crossmodal needs the API/DB; don't lose the table report
            print(f"  crossmodal skipped: {e}")
    with open(os.path.join(out_dir, "r1_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"-> {out_dir}/r1_report.json  + r1_umap_<tag>.png")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/bc/data/datasets/vera_directionA_1M/preset_bank_v2")
    ap.add_argument("--tags", default="rich_plain,rich_instr,axes_plain")
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    run(a.out, [t for t in a.tags.split(",") if t], a.k)


if __name__ == "__main__":
    main()
