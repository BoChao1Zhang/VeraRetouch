"""VL rerank: query = source image, documents = candidate preset captions -> top-N.

The reranker (Qwen3-VL-Reranker-8B) is a SEPARATE model from the embedding one and
is the design's real cross-modal judge. R1 showed the embedding cosine is a weak
cross-modal signal; whether the *reranker* is sharper is an R2 question. Documents
are the real VLM captions (high-cardinality, semantic).

CLI:  python -m construct.rerank <source.jpg> [--k 50] [--top 8]
"""
from __future__ import annotations

import argparse
from typing import List, Optional

from . import sf_client
from .bank import PresetBank, doc_text, load_captions
from .recall import recall


def rerank_candidates(source_path: str, candidates: List[dict], bank: PresetBank,
                      captions: Optional[dict] = None, top: int = 8,
                      instruction: Optional[str] = None) -> List[dict]:
    """candidates: recall() output (each has preset_id + idx). Returns top-N reordered,
    each with added `rerank_score` and `doc`."""
    caps = captions if captions is not None else load_captions()
    docs = []
    for c in candidates:
        f = dict(bank.feats[c["idx"]]); f.update(caps.get(c["preset_id"], {}))
        docs.append(doc_text(f, "vlm"))
    order = sf_client.rerank(source_path, docs, instruction=instruction)
    out = []
    for orig_idx, score in order[:top]:
        c = dict(candidates[orig_idx]); c["rerank_score"] = round(score, 4); c["doc"] = docs[orig_idx]
        out.append(c)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--out", default="/home/bc/data/datasets/vera_directionA_1M/preset_bank_v2")
    ap.add_argument("--emb-tag", default=None)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--instruction", default="Rank these editing presets by how well their color "
                    "grading aesthetically enhances the given photo.")
    a = ap.parse_args()
    bank = PresetBank.load(a.out, a.emb_tag)
    cands = recall(bank, a.source, a.scene, a.k)
    top = rerank_candidates(a.source, cands, bank, top=a.top, instruction=a.instruction)
    print(f"rerank: {len(cands)} -> top {len(top)}")
    for t in top:
        print(f"  {t['rerank_score']:.4f} {t['grade_family']:13s} {t['doc'][:70]}")


if __name__ == "__main__":
    main()
