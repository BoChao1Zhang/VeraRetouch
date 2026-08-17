"""Qwen3-VL-4B-Instruct Arm B base SFT entrypoint (spec 2/4/5/6/7/8).

    torchrun --nproc_per_node=2 -m q3vl.train.train_sft --config <yaml|json>

Refuses to start unless:
  * the weight shards pass the integrity check (spec 2.1 / 9.1);
  * the frozen/trainable boundary matches Arm B (spec 2.2 / 9.2);
  * the special tokens are single tokens with distinct ids (spec 4.3 / 9.4) --
    six of them since the v2seg re-train (2026-08-14);
  * the spec-frozen hyperparameters are unchanged and global batch == 32 (spec 7);
  * a terminal manifest supplies ``N_effective`` (spec 8.1) -- 2645/5290 are
    estimates and are never used as checkpoint authority.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import HfArgumentParser, set_seed

from .args import DataArguments, ModelArguments, SFTTrainingArguments, validate_frozen_hyperparameters
from .collator import Sft2SegCollator
from .constants import (
    GLOBAL_BATCH_SIZE, SEG_COLOR, SEG_COLOR_TOK, SEG_SEGCOLOR, SEG_SEGWHERE, SEG_WHERE_TOK,
)
from .dataset import Sft2SegDataset
from .freeze import format_freeze_table
from .modeling import load_processor, model_architecture_facts, setup_model, verify_model_shards
from .shards import ShardIndex, ShardStore, TerminalManifest
from .trainer import Qwen3VLSFTTrainer

logger = logging.getLogger(__name__)


def _read_config_file(path: str) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith((".yaml", ".yml")):
        import yaml

        return yaml.safe_load(text) or {}
    return json.loads(text)


def parse_args(argv: list[str] | None = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = HfArgumentParser((ModelArguments, DataArguments, SFTTrainingArguments))
    if "--config" in argv:
        i = argv.index("--config")
        cfg_path = argv[i + 1]
        del argv[i : i + 2]
        cfg = _read_config_file(cfg_path)
        flat: dict[str, Any] = {}
        for section in ("model", "data", "training"):
            flat.update(cfg.pop(section, {}) or {})
        flat.update(cfg)
        for key, value in _parse_overrides(argv).items():  # CLI wins over the file
            field = _FIELD_TYPES.get(key)
            if field is None:
                raise SystemExit(f"unknown override --{key}")
            flat[key] = _coerce(field, value)
        return parser.parse_dict(flat, allow_extra_keys=True)
    return parser.parse_args_into_dataclasses()


def _field_types() -> dict[str, Any]:
    import dataclasses

    out: dict[str, Any] = {}
    for dc in (ModelArguments, DataArguments, SFTTrainingArguments):
        for f in dataclasses.fields(dc):
            out[f.name] = f.type
    return out


_FIELD_TYPES = _field_types()


def _coerce(field_type: Any, value: str) -> Any:
    text = str(field_type)
    if "bool" in text:
        return str(value).lower() in ("1", "true", "yes")
    if "int" in text and "float" not in text:
        return int(value)
    if "float" in text:
        return float(value)
    return value


def _parse_overrides(argv: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--"):
            raise SystemExit(f"unexpected argument {tok!r}")
        key = tok[2:]
        if "=" in key:
            key, _, val = key.partition("=")
            out[key.replace("-", "_")] = val
            i += 1
        else:
            out[key.replace("-", "_")] = argv[i + 1]
            i += 2
    return out


def build_dataset(index_path: str, data_args: DataArguments, split: str | None, split_ids: str | None):
    """``split`` filters rows inside a *combined* index; leave it None (the
    default) when the producer writes one index file per split, which is what
    ``q3vl.data.pipeline`` does (``splits/<name>.index.jsonl``)."""
    index = ShardIndex.load(index_path, split=split)
    if split_ids:
        keep = {line.strip() for line in Path(split_ids).read_text().splitlines() if line.strip()}
        before = len(index)
        index = index.filter_ids(keep)
        logger.info("split %s: %d -> %d samples after applying %s", split, before, len(index), split_ids)
    store = ShardStore(data_args.shard_root or Path(index_path).parent, verify=data_args.verify)
    ds = Sft2SegDataset(
        index, store,
        image_root=data_args.image_root,
        allow_local_assembly=data_args.allow_local_assembly,
    )
    return ds


def assert_v2seg_template(collator, sample, special_ids: dict[str, int]) -> dict[str, Any]:
    """Fail at startup if the v2seg readout tokens are not in the real batch.

    CLAUDE.md: a pre-registered criterion needs a *runtime* assertion -- a
    template that lives in collator.py but never reaches the tensors is the
    "defined but not wired" failure this project has already paid for. Costs one
    sample encode on every rank; every rank computes the same answer, so a
    failure cannot leave one rank hanging on a collective.
    """
    enc = collator.encode_one(sample)
    ids, labels, segs = enc["input_ids"], enc["labels"], enc["segment_ids"]
    sw = [i for i, s in enumerate(segs) if s == SEG_SEGWHERE]
    sc = [i for i, s in enumerate(segs) if s == SEG_SEGCOLOR]
    last_color = max((i for i, s in enumerate(segs) if s == SEG_COLOR), default=-1)
    ok = (
        len(sw) == 1 and len(sc) == 1 and last_color >= 0 and last_color < sw[0] < sc[0]
        and ids[sw[0]] == labels[sw[0]] == special_ids[SEG_WHERE_TOK]
        and ids[sc[0]] == labels[sc[0]] == special_ids[SEG_COLOR_TOK]
    )
    info = {
        "sample_id": sample.sample_id,
        "seg_where_pos": sw,
        "seg_color_pos": sc,
        "last_color_pos": last_color,
        "seq_len": len(ids),
        "tail_ids": ids[-4:],
    }
    if not ok:
        raise SystemExit(f"v2seg template assertion FAILED: {info}")
    return info


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
    )
    model_args, data_args, train_args = parse_args(argv)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = int(os.environ.get("RANK", "0")) == 0

    allow_nonspec = os.environ.get("Q3VL_ALLOW_NONSPEC") == "1"
    batch_info = validate_frozen_hyperparameters(train_args, world_size, allow_nonspec)
    set_seed(train_args.seed)

    if is_main:
        logger.info("transformers %s | torch %s", transformers.__version__, torch.__version__)
        logger.info("batch plan: %s", batch_info)

    # -- preflight item 1 ---------------------------------------------------
    if model_args.verify_shards:
        expected = None
        if model_args.shard_checksums_json:
            expected = json.loads(Path(model_args.shard_checksums_json).read_text())
        shard_report = verify_model_shards(model_args.model_name_or_path, expected)
        if not shard_report["passed"]:
            raise SystemExit(f"model shard verification FAILED: {shard_report['errors']}")
        if is_main:
            logger.info("shard verification PASSED (%d shards)", len(shard_report["shards"]))

    # -- tokenizer / model --------------------------------------------------
    processor, special_ids = load_processor(model_args.model_name_or_path, data_args.max_length)
    if is_main:
        logger.info("special token ids: %s", special_ids)

    model, freeze_report, emb_info = setup_model(
        model_args.model_name_or_path,
        processor,
        special_ids,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
        reinit_special_token_rows=model_args.reinit_special_token_rows,
        seed=model_args.special_token_init_seed,
    )
    if is_main:
        logger.info("embeddings: %s", json.dumps(emb_info, default=str))
        logger.info("\n%s", format_freeze_table(freeze_report))

    # -- data ---------------------------------------------------------------
    if not data_args.train_index or not data_args.terminal_manifest:
        raise SystemExit(
            "train_index and terminal_manifest are required; spec 8.1 forbids deriving "
            "checkpoint steps from anything but the terminal manifest."
        )
    train_ds = build_dataset(
        data_args.train_index, data_args, data_args.train_split_name, data_args.split_ids_train
    )
    eval_ds = (
        build_dataset(
            data_args.eval_index, data_args, data_args.eval_split_name, data_args.split_ids_eval
        )
        if data_args.eval_index else None
    )

    manifest = TerminalManifest.load(data_args.terminal_manifest)
    n_effective = manifest.n_effective(data_args.manifest_split_name)
    if n_effective != len(train_ds):
        raise SystemExit(
            f"terminal manifest N_effective={n_effective} but the train index yields "
            f"{len(train_ds)} samples; refusing to train on a disagreeing manifest."
        )
    expected_steps = math.ceil(n_effective / GLOBAL_BATCH_SIZE)
    manifest_info = manifest.summary() | {
        "n_effective_train": n_effective,
        "n_eval": len(eval_ds) if eval_ds else 0,
        "expected_steps_per_epoch": expected_steps,
        "train_index": data_args.train_index,
        "index_layout": train_ds.index.layout,
    }
    if is_main:
        logger.info("data manifest: %s", json.dumps(manifest_info, default=str))

    collator = Sft2SegCollator(
        processor,
        max_length=data_args.max_length,
        system_prompt=data_args.system_prompt,
        supervise_eos=data_args.supervise_eos,
    )
    v2seg_probe = assert_v2seg_template(collator, train_ds[0], special_ids)
    if is_main:
        logger.info("v2seg template assertion PASSED: %s", v2seg_probe)

    trainer = Qwen3VLSFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=processor,
        data_manifest_info=manifest_info,
        special_token_ids=special_ids,
        freeze_report=freeze_report.to_dict(),
        diag_every_n_steps=train_args.diag_every_n_steps,
        gen_diag_samples=train_args.gen_diag_samples,
        gen_diag_max_new_tokens=train_args.gen_diag_max_new_tokens,
        expected_steps_per_epoch=expected_steps,
    )

    if is_main:
        Path(train_args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(train_args.output_dir) / "run_setup.json").write_text(
            json.dumps(
                {
                    "batch_plan": batch_info,
                    "v2seg_template": v2seg_probe,
                    "special_token_ids": special_ids,
                    "embeddings": emb_info,
                    "freeze": freeze_report.to_dict(),
                    "manifest": manifest_info,
                    "architecture": model_architecture_facts(model_args.model_name_or_path),
                    "versions": {
                        "transformers": transformers.__version__,
                        "torch": torch.__version__,
                    },
                },
                indent=2, default=str, ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    result = trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
    trainer.save_model()
    trainer.save_state()
    if is_main:
        logger.info("training finished: %s", result.metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
