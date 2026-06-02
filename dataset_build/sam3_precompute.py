"""Precompute SAM3 concept C_GT masks for the region-local (S2/S3) source pools.

RUN IN THE monetgpt_sam3 ENV ON A GPU (cannot co-exist with the renderer):
    SAM3_PY=/home/bc/miniconda3/envs/monetgpt_sam3/bin/python
    CUDA_VISIBLE_DEVICES=0 $SAM3_PY -m dataset_build.sam3_precompute --shard 0/2 &
    CUDA_VISIBLE_DEVICES=1 $SAM3_PY -m dataset_build.sam3_precompute --shard 1/2 &
    wait

Writes <out_root>/sam3_cache/<path_key>/<concept_slug>.png (8-bit L, [0,1]).
Resumable (skips sources whose concepts are all already cached). The base-env
orchestrator then reads these via mask_cache.CachedMasker (set models.sam3.use_cache:true).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Iterator, List, Optional, Set

import numpy as np

from dataset_build.mask_cache import concept_slug, path_key

# SC-local defaults (probe_sam3 §6: most-local, highest C_GT signal).
DEFAULT_CONCEPTS = ["skin", "face", "foliage", "grass", "water", "sky", "flowers"]
DEFAULT_CORPORA = {"tad66k", "awards", "korean", "quandian", "unsplash", "mmart"}


def _load_yaml(p: str) -> dict:
    import yaml

    with open(p) as f:
        return yaml.safe_load(f)


def _concepts_from_config(cfg: dict) -> List[str]:
    vocab = (cfg.get("cgt", {}) or {}).get("concept_vocab")
    if isinstance(vocab, dict):
        seen: List[str] = []
        for v in vocab.values():
            for c in (v if isinstance(v, list) else [v]):
                if c not in seen:
                    seen.append(c)
        return seen or DEFAULT_CONCEPTS
    if isinstance(vocab, list) and vocab:
        return list(vocab)
    return DEFAULT_CONCEPTS


def _parse_corpora_arg(raw: str) -> Set[str]:
    vals = {c.strip() for c in str(raw or "").split(",") if c.strip()}
    if any(v.lower() in {"all", "*"} for v in vals):
        return set()
    return vals or set(DEFAULT_CORPORA)


def _iter_sources(index_path: str, corpora: Set[str], shard_i: int, shard_n: int, limit: int) -> Iterator[dict]:
    kept = 0
    seen: Set[str] = set()
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if corpora and r.get("corpus") not in corpora:
                continue
            path = r.get("path")
            if not path:
                continue
            key = path_key(path)
            if key in seen:
                continue
            seen.add(key)
            if shard_n > 1 and (int(key, 16) % shard_n) != shard_i:
                continue
            yield r
            kept += 1
            if limit and kept >= limit:
                return


def _save_png(mask: np.ndarray, out_path: str) -> None:
    from PIL import Image

    a = (np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0) * 255.0).round().astype("uint8")
    tmp = out_path + ".tmp"
    Image.fromarray(a, mode="L").save(tmp, format="PNG", compress_level=1)
    os.replace(tmp, out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute SAM3 concept C_GT masks into a PNG cache.")
    ap.add_argument("--config", default="dataset_build/config.yaml")
    ap.add_argument("--shard", default="0/1", help="i/n stable path-hash partition")
    ap.add_argument("--limit", type=int, default=0, help="cap sources this shard (smoke)")
    ap.add_argument("--corpora", default="", help="comma list to restrict corpora; default = S2/S3 SAM3 pools; use ALL for no filter")
    ap.add_argument("--concepts", default="", help="comma list; empty = config cgt.concept_vocab or SC defaults")
    ap.add_argument("--cache-dir", default="", help="override cache dir")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    out_root = cfg["out_root"]
    index_path = os.path.join(out_root, "source_index.jsonl")
    if not os.path.exists(index_path):
        sys.exit(f"source_index.jsonl not found at {index_path} — run registry.py first")

    sam = (cfg.get("models", {}) or {}).get("sam3", {}) or {}
    cache_dir = args.cache_dir or os.path.join(out_root, sam.get("cache_subdir", "sam3_cache"))
    os.makedirs(cache_dir, exist_ok=True)

    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    concepts = [c.strip() for c in args.concepts.split(",") if c.strip()] or _concepts_from_config(cfg)
    corpora = _parse_corpora_arg(args.corpora)

    print(f"[sam3] shard {shard_i}/{shard_n} cache={cache_dir} concepts={concepts} corpora={corpora or 'ALL'}", flush=True)

    # Lazy SAM3 load (only here, only in monetgpt_sam3).
    from dataset_build.masking import Sam3Masker

    masker = Sam3Masker(
        model_dir=sam.get("model_dir", "/home/bc/data/models"),
        score_threshold=float(sam.get("score_threshold", 0.3)),
        mask_threshold=float(sam.get("mask_threshold", 0.5)),
        concept_map=(cfg.get("cgt", {}) or {}).get("concept_map"),
    )

    done = skipped = failed = 0
    for r in _iter_sources(index_path, corpora, shard_i, shard_n, args.limit):
        path = r["path"]
        d = os.path.join(cache_dir, path_key(path))
        if all(os.path.exists(os.path.join(d, concept_slug(c) + ".png")) for c in concepts):
            skipped += 1
            continue
        os.makedirs(d, exist_ok=True)
        try:
            cmaps: Dict[str, np.ndarray] = masker.masks(path, concepts)
        except Exception as e:  # missing file / decode / OOM on one image
            failed += 1
            print(f"[sam3] skip {path}: {e}", file=sys.stderr)
            continue
        for c in concepts:
            m = cmaps.get(c)
            if m is None:
                continue
            try:
                _save_png(m, os.path.join(d, concept_slug(c) + ".png"))
            except Exception as e:
                print(f"[sam3] save fail {path}/{c}: {e}", file=sys.stderr)
        done += 1
        if done % 200 == 0:
            print(f"[sam3] shard {shard_i}/{shard_n}: cached={done} skipped={skipped} failed={failed}", flush=True)

    print(f"[sam3] DONE shard {shard_i}/{shard_n}: cached={done} skipped={skipped} failed={failed} -> {cache_dir}", flush=True)


if __name__ == "__main__":
    main()
