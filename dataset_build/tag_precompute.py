"""Precompute SOURCE-IMAGE TAGS + AESTHETIC scores per UNIQUE source image.

Mirrors ``sam3_precompute.py`` (same argparse / source_index iteration /
``--shard i/N`` path-hash stride / ``--limit`` / ``--corpora`` / resumable
skip-if-exists / atomic tmp+replace write), but writes ONE ``tags.json`` per
unique source path instead of per-concept PNGs.

Per source it computes (DUAL-CARD: two VLM servers, one --base-url each):
  (a) VLM tags + aesthetic via ``vlm_clean.QwenVLCleaner.tag_full`` — the same
      scene/style/region_local/sam3_concepts/groundingdino_prompt/
      masksubtype_hint as the inline ``tag_scene_region``, PLUS an integer
      ``aesthetic_vlm`` (1-10). The existing inline ``tag_scene_region`` callers
      in the build are UNCHANGED (``tag_full`` is an additive method).
  (b) ``aesthetic_model`` via the dedicated CLIP+MLP ``aesthetic.AestheticScorer``
      (None if the weights are absent — graceful).

Cache layout (path_key REUSED from mask_cache so this cache aligns with sam3):
    <out_root>/<tag_cache.subdir>/<path_key(source_path)>/tags.json

DUAL-CARD launch (two processes, disjoint shards, one server each):
    QWEN_PY=<python with openai+torch+transformers>
    CUDA_VISIBLE_DEVICES=0 $QWEN_PY -m dataset_build.tag_precompute --shard 0/2 \
        --base-url http://localhost:8001/v1 &
    CUDA_VISIBLE_DEVICES=1 $QWEN_PY -m dataset_build.tag_precompute --shard 1/2 \
        --base-url http://localhost:8002/v1 &
    wait

Resumable: skips a source whose tags.json already exists (unless --overwrite).
A per-image failure logs and continues (never crashes the run). The base-env
orchestrator then reads these via ``tag_cache.CachedTagger`` (set
``tag_cache.use_cache: true`` in config.yaml).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator, List, Optional, Set

from dataset_build.mask_cache import path_key

DEFAULT_CORPORA: Set[str] = set()  # default = ALL corpora (tags are universal)


def _load_yaml(p: str) -> dict:
    import yaml

    with open(p) as f:
        return yaml.safe_load(f)


def _parse_corpora_arg(raw: str) -> Set[str]:
    vals = {c.strip() for c in str(raw or "").split(",") if c.strip()}
    if any(v.lower() in {"all", "*"} for v in vals):
        return set()
    return vals or set(DEFAULT_CORPORA)


def _iter_sources(
    index_path: str, corpora: Set[str], shard_i: int, shard_n: int, limit: int
) -> Iterator[dict]:
    """Identical sharding to sam3_precompute: unique path, same path_key stride."""
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


def _atomic_write_json(obj: dict, out_path: str) -> None:
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, out_path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Precompute per-source VLM tags + aesthetic into a JSON cache."
    )
    ap.add_argument("--config", default="dataset_build/config.yaml")
    ap.add_argument("--shard", default="0/1", help="i/n stable path-hash partition")
    ap.add_argument("--limit", type=int, default=0, help="cap sources this shard (smoke)")
    ap.add_argument("--corpora", default="", help="comma list to restrict corpora; ALL/empty = no filter")
    ap.add_argument("--base-url", default="", help="vLLM OpenAI endpoint (overrides config vllm.base_url)")
    ap.add_argument("--concurrency", type=int, default=64, help="ThreadPool fan-out for VLM HTTP work")
    ap.add_argument("--cache-dir", default="", help="override cache dir")
    ap.add_argument("--overwrite", action="store_true", help="recompute even if tags.json exists")
    ap.add_argument("--no-aesthetic-model", action="store_true", help="skip the CLIP+MLP aesthetic scorer")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    out_root = cfg["out_root"]
    index_path = os.path.join(out_root, "source_index.jsonl")
    if not os.path.exists(index_path):
        sys.exit(f"source_index.jsonl not found at {index_path} — run registry.py first")

    tc = (cfg.get("tag_cache", {}) or {})
    cache_dir = args.cache_dir or tc.get("cache_dir") or os.path.join(
        out_root, tc.get("subdir", "tag_cache")
    )
    os.makedirs(cache_dir, exist_ok=True)

    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    corpora = _parse_corpora_arg(args.corpora)

    # Cleaner over the (per-process) --base-url. Override base_url so the dual-card
    # launch points each process at its own server.
    from dataset_build.vlm_clean import QwenVLCleaner

    if args.base_url:
        cfg.setdefault("vllm", {})["base_url"] = args.base_url
    cleaner = QwenVLCleaner.from_config(cfg)

    # Dedicated objective aesthetic scorer (graceful: None if weights absent).
    scorer = None
    if not args.no_aesthetic_model:
        from dataset_build.aesthetic import AestheticScorer

        ae_cfg = (cfg.get("aesthetic", {}) or {})
        scorer = AestheticScorer(
            clip_dir=ae_cfg.get("clip_dir", "/home/bc/data/models/clip-vit-large-patch14"),
            mlp_path=ae_cfg.get(
                "mlp_path",
                "/home/bc/data/models/laion_aesthetic_sac_logos_ava1_l14_linearMSE.pth",
            ),
        )

    print(
        f"[tag] shard {shard_i}/{shard_n} cache={cache_dir} base_url={cleaner.base_url} "
        f"corpora={corpora or 'ALL'} concurrency={args.concurrency} "
        f"aesthetic_model={'on' if (scorer and scorer.enabled) else 'off'}",
        flush=True,
    )

    counters = {"done": 0, "skipped": 0, "failed": 0}
    clock = {"t0": time.time()}
    lock = threading.Lock()

    def _process(r: dict) -> None:
        path = r["path"]
        d = os.path.join(cache_dir, path_key(path))
        out_path = os.path.join(d, "tags.json")
        if (not args.overwrite) and os.path.exists(out_path):
            with lock:
                counters["skipped"] += 1
            return
        try:
            tags = cleaner.tag_full(path, "")  # VLM tags + aesthetic_vlm
            aesthetic_model: Optional[float] = None
            if scorer is not None and scorer.enabled:
                aesthetic_model = scorer.score(path)
            blob = dict(tags)
            blob["aesthetic_model"] = aesthetic_model
            # Combined: prefer the model score, else the VLM score.
            blob["aesthetic"] = (
                aesthetic_model if aesthetic_model is not None else tags.get("aesthetic_vlm")
            )
            blob["source_id"] = r.get("source_id")
            blob["corpus"] = r.get("corpus")
            os.makedirs(d, exist_ok=True)
            _atomic_write_json(blob, out_path)
            with lock:
                counters["done"] += 1
                done = counters["done"]
            if done % 200 == 0:
                with lock:
                    elapsed = max(1e-6, time.time() - clock["t0"])
                    rate = counters["done"] / elapsed
                print(
                    f"[tag] shard {shard_i}/{shard_n}: cached={counters['done']} "
                    f"skipped={counters['skipped']} failed={counters['failed']} "
                    f"({rate:.1f}/s)",
                    flush=True,
                )
        except Exception as e:  # missing file / decode / HTTP flake on one image
            with lock:
                counters["failed"] += 1
            print(f"[tag] skip {path}: {type(e).__name__}: {e}", file=sys.stderr)

    sources = _iter_sources(index_path, corpora, shard_i, shard_n, args.limit)
    with ThreadPoolExecutor(max_workers=max(1, int(args.concurrency))) as ex:
        futures = [ex.submit(_process, r) for r in sources]
        for fut in as_completed(futures):
            # _process never raises; this is just to surface any unexpected escape.
            try:
                fut.result()
            except Exception as e:  # pragma: no cover
                print(f"[tag] worker error: {e}", file=sys.stderr)

    elapsed = max(1e-6, time.time() - clock["t0"])
    print(
        f"[tag] DONE shard {shard_i}/{shard_n}: cached={counters['done']} "
        f"skipped={counters['skipped']} failed={counters['failed']} "
        f"in {elapsed:.0f}s -> {cache_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
