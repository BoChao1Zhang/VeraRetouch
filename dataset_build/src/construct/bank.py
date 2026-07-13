"""Preset bank for the 50k construct agent — VL-text + 24-d LAB, rebuilt from the
LIVE DB (assets.pass_c=1, >=6 canonical probes), NOT the stale recipe_index.

Single stage（embed/doc_text 已随 vlemb 召回退役，2026-07-13）:

  extract  DB + 6 probe before/after JPGs
             -> lab_vec[24] = [dL,da,db,dC] x (red,yellow,green,blue,skin,neutral)
             -> axes = tag_preset_function(sigs)  (temperature/tint/.../grade_family)
             -> features.jsonl (+ lab.npz aligned ids/lab_vec). SLOW, run once.

Reuse (load-bearing): source_qa.pilot_preset {_lab_sig, _PROBE_ORDER, MANIFEST},
source_qa.qa_clean.tag_preset_function, source_qa.db. The probe RESPONSE — never
the probe image — is what characterizes a preset (design §1).

CLI:
  python -m construct.bank extract [--out DIR] [--kind param|lut|all] [--limit-per-kind N]
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np

from dataset_build.source_qa import db
from dataset_build.source_qa import pilot_preset as PP
from dataset_build.source_qa import qa_clean as QC

_PROBE_ORDER = PP._PROBE_ORDER          # (red, yellow, green, blue, skin, neutral)
_SIG_KEYS = ("dL", "da", "db", "dC")
VEC_DIM = len(_PROBE_ORDER) * len(_SIG_KEYS)  # 24
_DEFAULT_OUT = "/home/bc/data/datasets/vera_directionA_1M/preset_bank_v2"

# --------------------------------------------------------------------------- #
# extract
# --------------------------------------------------------------------------- #
def _probe_name_map() -> Dict[str, str]:
    """canonical probe asset_id -> slot name, restricted to the 6 _PROBE_ORDER slots."""
    m = json.load(open(PP.MANIFEST))
    return {p["asset_id"]: p["name"] for p in (m.get("probes") or [])
            if p.get("name") in _PROBE_ORDER}


def _fetch(kind: str, limit_per_kind: Optional[int]) -> Tuple[Dict[str, dict], Dict[str, List[dict]], Dict[str, str]]:
    pname = _probe_name_map()
    probe_ids = list(pname)
    conn = db.connect()
    where_kind = "" if kind == "all" else " AND a.kind = %s"
    params: list = [] if kind == "all" else [kind]
    rows = conn.execute(
        "SELECT a.asset_id, a.scene_affinity, a.has_local_mask, a.has_ai_mask, a.kind, "
        "       a.fmt, a.pack_id, a.path, a.preset_content_hash, q.coherence_score "
        "FROM assets a LEFT JOIN preset_qa_runs q ON q.asset_id = a.asset_id "
        "WHERE a.asset_type='preset' AND a.pass_c=1" + where_kind, tuple(params)).fetchall()
    meta = {r["asset_id"]: dict(r) for r in rows}
    if limit_per_kind:  # deterministic subset for the pilot
        keep, seen = set(), {}
        for aid in sorted(meta):
            k = meta[aid]["kind"]
            if seen.get(k, 0) < limit_per_kind:
                keep.add(aid); seen[k] = seen.get(k, 0) + 1
        meta = {a: meta[a] for a in keep}
    ids = list(meta)
    prevs: Dict[str, List[dict]] = {}
    for r in conn.execute(
        "SELECT DISTINCT ON (asset_id, probe_image) asset_id, probe_image, before_path, after_path "
        "FROM preset_previews WHERE asset_id = ANY(%s) AND probe_image = ANY(%s) "
        "AND before_path IS NOT NULL AND after_path IS NOT NULL "
        "ORDER BY asset_id, probe_image, id DESC", (ids, probe_ids)).fetchall():
        prevs.setdefault(r["asset_id"], []).append(dict(r))
    conn.close()
    return meta, prevs, pname


def _build_one(aid: str, prev_rows: List[dict], meta: dict, pname: Dict[str, str]) -> Optional[dict]:
    sigs_by: Dict[str, dict] = {}
    for pr in prev_rows:
        nm = pname.get(pr["probe_image"])
        if not nm or not (os.path.exists(pr["before_path"]) and os.path.exists(pr["after_path"])):
            continue
        try:
            sig = PP._lab_sig(pr["before_path"], pr["after_path"])
        except Exception:  # noqa: BLE001 - one unreadable probe shouldn't kill the bank
            continue
        sig["name"] = nm
        sigs_by[nm] = sig
    if any(nm not in sigs_by for nm in _PROBE_ORDER):
        return None  # require all 6 canonical probes
    vec = [float(sigs_by[nm][k]) for nm in _PROBE_ORDER for k in _SIG_KEYS]
    det = QC.tag_preset_function([sigs_by[nm] for nm in _PROBE_ORDER])
    axes = {k: det.get(k) for k in
            ("temperature", "tint", "saturation", "contrast", "tone", "exposure", "grade_family")}
    return {
        "preset_id": aid, "kind": meta.get("kind"), "fmt": meta.get("fmt"),
        "pack_id": meta.get("pack_id"), "path": meta.get("path"),
        "preset_content_hash": meta.get("preset_content_hash"),
        "scene_affinity": meta.get("scene_affinity"),
        "has_local_mask": int(meta.get("has_local_mask") or 0),
        "has_ai_mask": int(meta.get("has_ai_mask") or 0),
        "coherence": meta.get("coherence_score"),
        "lab_vec": vec, "axes": axes, "metrics": det.get("metrics", {}),
    }


def extract(out_dir: str = _DEFAULT_OUT, kind: str = "all",
            limit_per_kind: Optional[int] = None, workers: int = 12) -> dict:
    meta, prevs, pname = _fetch(kind, limit_per_kind)
    items = [(a, prevs[a]) for a in meta if a in prevs]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        built = list(ex.map(lambda kv: _build_one(kv[0], kv[1], meta[kv[0]], pname), items))
    built = [b for b in built if b is not None]
    os.makedirs(out_dir, exist_ok=True)
    ids = np.array([b["preset_id"] for b in built])
    lab = np.array([b["lab_vec"] for b in built], dtype="float32")
    np.savez(os.path.join(out_dir, "lab.npz"), ids=ids, lab=lab)
    with open(os.path.join(out_dir, "features.jsonl"), "w") as f:
        for b in built:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    from collections import Counter
    fam = Counter(b["axes"]["grade_family"] for b in built)
    temp = Counter(b["axes"]["temperature"] for b in built)
    summary = {"n_candidates": len(items), "n_bank": len(built), "vec_dim": VEC_DIM,
               "grade_family": dict(fam.most_common()), "temperature": dict(temp.most_common())}
    print(f"bank.extract: {summary} -> {out_dir}")
    return summary


_CAPTIONS = "/home/bc/VeraRetouch/dataset_build/source_qa/pilot/round_10/preset_tags.jsonl"  # 绝对路径：进程 cwd 不可假设（2026-07-12 pilot tier-skip 事故）


def load_captions(path: str = _CAPTIONS) -> Dict[str, dict]:
    """asset_id -> {vlm_name, vlm_caption, vlm_function}（8027 条 VLM 标注；
    tier/annotate 的 style 任务点名风格用）。"""
    out = {}
    for l in open(path):
        if not l.strip():
            continue
        c = json.loads(l)
        out[c["asset_id"]] = {k: c.get(k) for k in ("vlm_name", "vlm_caption", "vlm_function")}
    return out


# --------------------------------------------------------------------------- #
class PresetBank:
    """Loaded bank: aligned ids + lab[N,24] + feature dicts（emb 已随召回退役）。"""

    def __init__(self, ids, lab, feats):
        self.ids, self.lab = ids, lab
        self.feats = feats
        self.by_id = {f["preset_id"]: i for i, f in enumerate(feats)}

    @classmethod
    def load(cls, out_dir: str = _DEFAULT_OUT) -> "PresetBank":
        feats = [json.loads(l) for l in open(os.path.join(out_dir, "features.jsonl")) if l.strip()]
        labz = np.load(os.path.join(out_dir, "lab.npz"), allow_pickle=True)
        return cls(labz["ids"], labz["lab"], feats)

    def __len__(self) -> int:
        return len(self.ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--out", default=_DEFAULT_OUT)
    e.add_argument("--kind", default="all", choices=["param", "lut", "all"])
    e.add_argument("--limit-per-kind", type=int, default=None)
    e.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.out, a.kind, a.limit_per_kind, a.workers)


if __name__ == "__main__":
    main()
