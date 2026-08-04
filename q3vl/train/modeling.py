"""Model / processor construction shared by the entrypoint and the preflight."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

from .freeze import FreezeReport, apply_arm_b_freeze
from .tokens import prepare_embeddings, register_special_tokens

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

# Upstream digests fetched from the Hugging Face model API on 2026-08-04 for
# repo revision ebb281ec70b05090aa6165b016eac8ec08e71b17 (see NOTES.md).
UPSTREAM_SHARD_SHA256 = {
    "model-00001-of-00002.safetensors":
        "30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9",
    "model-00002-of-00002.safetensors":
        "046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6",
}
UPSTREAM_SHARD_SIZE = {
    "model-00001-of-00002.safetensors": 4967229296,
    "model-00002-of-00002.safetensors": 3908490048,
}
UPSTREAM_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"

PARTIAL_SUFFIXES = (".aria2", ".part", ".incomplete", ".tmp", ".download")


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def verify_model_shards(
    model_path: str | os.PathLike,
    expected_sha256: dict[str, str] | None = None,
    compute_hashes: bool = True,
) -> dict[str, Any]:
    """Preflight item 1: every shard in the index is present, sized and hashed.

    ``.aria2``/``.part`` leftovers are a hard failure: spec 2.1 warns that the
    presence of the final filename does not prove a complete download.
    """
    root = Path(model_path)
    expected_sha256 = UPSTREAM_SHARD_SHA256 if expected_sha256 is None else expected_sha256
    result: dict[str, Any] = {
        "model_path": str(root),
        "upstream_revision": UPSTREAM_REVISION,
        "shards": [],
        "partial_files": [],
        "errors": [],
    }

    partials = [str(p) for p in root.rglob("*") if p.name.endswith(PARTIAL_SUFFIXES)]
    result["partial_files"] = partials
    if partials:
        result["errors"].append(f"partial download artefacts present: {partials}")

    index_path = root / "model.safetensors.index.json"
    if not index_path.exists():
        result["errors"].append("model.safetensors.index.json missing")
        result["passed"] = False
        return result
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    declared_total = index.get("metadata", {}).get("total_size")
    shard_names = sorted(set(weight_map.values()))
    result["n_tensors_in_index"] = len(weight_map)
    result["declared_total_size"] = declared_total

    total_bytes = 0
    for name in shard_names:
        p = root / name
        entry: dict[str, Any] = {"file": name, "present": p.exists()}
        if not p.exists():
            result["errors"].append(f"shard listed in index is missing: {name}")
            result["shards"].append(entry)
            continue
        size = p.stat().st_size
        entry["size"] = size
        total_bytes += size
        want_size = UPSTREAM_SHARD_SIZE.get(name)
        if want_size is not None:
            entry["expected_size"] = want_size
            entry["size_ok"] = size == want_size
            if size != want_size:
                result["errors"].append(f"{name}: size {size} != upstream {want_size}")
        if compute_hashes:
            got = sha256_file(p)
            entry["sha256"] = got
            want = expected_sha256.get(name)
            entry["expected_sha256"] = want
            entry["sha256_ok"] = (want is None) or (got == want)
            if want is not None and got != want:
                result["errors"].append(f"{name}: sha256 {got} != upstream {want}")
        result["shards"].append(entry)

    result["total_shard_bytes"] = total_bytes
    if declared_total is not None:
        # safetensors headers add a few hundred bytes per shard, so the index's
        # total_size is a tensor-bytes figure and must not exceed the files.
        result["total_size_le_bytes_on_disk"] = declared_total <= total_bytes
        if declared_total > total_bytes:
            result["errors"].append(
                f"index total_size {declared_total} exceeds bytes on disk {total_bytes}"
            )

    # every tensor named in the index must actually be readable from its shard
    try:
        from safetensors import safe_open

        missing: list[str] = []
        by_shard: dict[str, list[str]] = {}
        for tensor, shard in weight_map.items():
            by_shard.setdefault(shard, []).append(tensor)
        for shard, tensors in by_shard.items():
            with safe_open(str(root / shard), framework="pt") as fh:
                keys = set(fh.keys())
            missing.extend(t for t in tensors if t not in keys)
        result["missing_tensors"] = missing
        if missing:
            result["errors"].append(f"{len(missing)} tensors in the index are absent from shards")
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"safetensors header scan failed: {type(exc).__name__}: {exc}")

    result["passed"] = not result["errors"]
    return result


def load_processor(model_path: str, max_length: int | None = None):
    processor = AutoProcessor.from_pretrained(model_path)
    special_ids = register_special_tokens(processor.tokenizer)
    if max_length is not None:
        processor.tokenizer.model_max_length = max_length
    return processor, special_ids


def load_model(
    model_path: str,
    attn_implementation: str = "flash_attention_2",
    dtype: str = "bfloat16",
    device_map: str | None = None,
) -> Qwen3VLForConditionalGeneration:
    torch_dtype = DTYPES[dtype]
    kwargs: dict[str, Any] = {
        "dtype": torch_dtype,
        "attn_implementation": attn_implementation,
    }
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    model.config.use_cache = False
    return model


def setup_model(
    model_path: str,
    processor,
    special_ids: dict[str, int],
    attn_implementation: str = "flash_attention_2",
    dtype: str = "bfloat16",
    device_map: str | None = None,
    reinit_special_token_rows: bool = True,
    seed: int = 0,
) -> tuple[Qwen3VLForConditionalGeneration, FreezeReport, dict[str, Any]]:
    model = load_model(model_path, attn_implementation, dtype, device_map)
    emb_info = prepare_embeddings(
        model, processor.tokenizer, special_ids, seed=seed,
        reinit_new_rows=reinit_special_token_rows,
    )
    freeze_report = apply_arm_b_freeze(model)
    return model, freeze_report, emb_info


def model_architecture_facts(model_path: str) -> dict[str, Any]:
    cfg = AutoConfig.from_pretrained(model_path)
    v = cfg.vision_config
    t = cfg.text_config
    return {
        "architectures": cfg.architectures,
        "vision": {
            "depth": v.depth,
            "hidden_size": v.hidden_size,
            "patch_size": v.patch_size,
            "spatial_merge_size": v.spatial_merge_size,
            "deepstack_visual_indexes": list(v.deepstack_visual_indexes),
            "out_hidden_size": v.out_hidden_size,
        },
        "text": {
            "num_hidden_layers": t.num_hidden_layers,
            "hidden_size": t.hidden_size,
            "vocab_size": t.vocab_size,
            "tie_word_embeddings": t.tie_word_embeddings,
        },
        "tie_word_embeddings": cfg.tie_word_embeddings,
        "config_transformers_version": getattr(cfg, "transformers_version", None),
    }
