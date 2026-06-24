"""LAB diverse pre-filter recall (R1 verdict: the 24-d LAB response is the signal;
VL-text embedding is dead weight for recall — see preset_bank_v2/R1_FINDINGS.md).

Ported from the original dataset_build.preset_bank.PresetBank.retrieve, adapted to
the v2 bank (construct.bank.PresetBank: self.lab[N,24], self.feats, self.ids).

The retriever is a DIVERSE PRE-FILTER, not a predictor — it biases toward presets
that actually MOVE the source's dominant hues and spreads across grade_family;
real fitness is decided by VL-rerank then post-render aesthetic QA (design §1).

CLI:  python -m construct.recall <source.jpg> [--k 50] [--scene portrait]
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import numpy as np

from dataset_build.source_qa import pilot_preset as PP
from .bank import PresetBank, _PROBE_ORDER, _SIG_KEYS


def _allowed_scene(scene: Optional[str]) -> Optional[set]:
    s = (scene or "").lower()
    if s in ("portrait", "wedding"):
        return {"portrait", "any"}
    if s in ("landscape", "architecture", "street", "night"):
        return {"landscape", "any"}
    return None  # still_life/product/food/any/unknown -> no scene gate


def _probe_part(lab: np.ndarray, i: int, name: str) -> np.ndarray:
    j = _PROBE_ORDER.index(name) * len(_SIG_KEYS)
    return lab[i, j:j + len(_SIG_KEYS)]  # [dL, da, db, dC]


def recall(bank: PresetBank, source_path: str, scene: Optional[str] = None,
           k: int = 50, seed: int = 0) -> List[dict]:
    stats = PP._img_color_stats(source_path)
    mass = stats.get("mass", {})
    hues = [h for h, _ in sorted(mass.items(), key=lambda kv: -kv[1])[:2] if mass.get(h, 0) > 0.01]
    probes = (hues or ["neutral"]) + ["skin", "neutral"]
    allowed = _allowed_scene(scene)
    pool = [i for i in range(len(bank.ids))
            if allowed is None or (bank.feats[i].get("scene_affinity") in allowed)]
    if not pool:
        pool = list(range(len(bank.ids)))

    def score(i: int) -> float:  # how much the preset moves colors present in the image
        return float(sum(np.abs(_probe_part(bank.lab, i, p)[1:]).sum() for p in probes))  # |da|+|db|+|dC|

    rng = np.random.default_rng(seed)
    buckets: Dict[str, List[int]] = {}
    for i in pool:
        buckets.setdefault(bank.feats[i]["axes"].get("grade_family") or "other", []).append(i)
    for fam in buckets:
        buckets[fam].sort(key=score, reverse=True)
    fams = list(buckets)
    rng.shuffle(fams)
    out: List[int] = []
    ptr = {f: 0 for f in fams}
    while len(out) < k and any(ptr[f] < len(buckets[f]) for f in fams):
        for f in fams:
            if ptr[f] < len(buckets[f]):
                out.append(buckets[f][ptr[f]]); ptr[f] += 1
                if len(out) >= k:
                    break
    return [{"preset_id": str(bank.ids[i]), "score": round(score(i), 2),
             "grade_family": bank.feats[i]["axes"].get("grade_family"),
             "scene_affinity": bank.feats[i].get("scene_affinity"),
             "kind": bank.feats[i].get("kind"), "axes": bank.feats[i]["axes"],
             "path": bank.feats[i].get("path"), "idx": i} for i in out]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--out", default="/home/bc/data/datasets/vera_directionA_1M/preset_bank_v2")
    ap.add_argument("--emb-tag", default=None)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--scene", default=None)
    a = ap.parse_args()
    bank = PresetBank.load(a.out, a.emb_tag)
    cands = recall(bank, a.source, a.scene, a.k)
    from collections import Counter
    fam = Counter(c["grade_family"] for c in cands)
    print(f"recall: {len(cands)} candidates, families={dict(fam)}")
    for c in cands[:8]:
        print(f"  {c['grade_family']:14s} score={c['score']:6.1f} {c['kind']:5s} {c['preset_id']}")


if __name__ == "__main__":
    main()
