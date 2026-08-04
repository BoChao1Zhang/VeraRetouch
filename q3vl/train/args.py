"""Argument dataclasses for the base SFT entrypoint.

Values that spec 7/8 freeze are the *defaults* here; a config file may restate
them but :func:`validate_frozen_hyperparameters` refuses to run if any is
changed, except for the one calibration knob the spec explicitly allows
(micro-batch 4xGAS4 -> 2xGAS8 on OOM, with global batch pinned to 32).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from transformers import TrainingArguments

from .constants import GLOBAL_BATCH_SIZE, MODEL_MAX_LENGTH

DEFAULT_MODEL_PATH = "/home/bc/data/models/Qwen3-VL-4B-Instruct"


@dataclass
class ModelArguments:
    model_name_or_path: str = DEFAULT_MODEL_PATH
    attn_implementation: str = "flash_attention_2"
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    reinit_special_token_rows: bool = True
    special_token_init_seed: int = 0
    verify_shards: bool = True
    shard_checksums_json: str | None = field(
        default=None,
        metadata={"help": "JSON {filename: sha256} of upstream shard digests (preflight item 1)."},
    )


@dataclass
class DataArguments:
    train_index: str = field(default="", metadata={"help": "JSONL/JSON shard index for train."})
    eval_index: str = field(default="", metadata={"help": "JSONL/JSON shard index for eval."})
    shard_root: str = field(default="", metadata={"help": "Root the index 'shard' field resolves against."})
    terminal_manifest: str = field(
        default="", metadata={"help": "Terminal manifest; the only authority for N_effective (spec 8.1)."}
    )
    image_root: str | None = None
    split_ids_train: str | None = None
    split_ids_eval: str | None = None
    train_split_name: str | None = field(
        default=None,
        metadata={"help": "Only needed for a COMBINED index; per-split index files need no filter."},
    )
    eval_split_name: str | None = None
    manifest_split_name: str = field(
        default="train", metadata={"help": "Split key to read N_effective from (spec 8.1)."}
    )
    verify: str = field(default="checksum", metadata={"help": "checksum|length|none"})
    max_length: int = MODEL_MAX_LENGTH
    system_prompt: str | None = None
    supervise_eos: bool = True
    allow_local_assembly: bool = False
    dataloader_prefetch: int = 4


@dataclass
class SFTTrainingArguments(TrainingArguments):
    # spec 7.1 -- official Arm B optimiser
    optim: str = "adamw_torch"
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    bf16: bool = True
    fp16: bool = False
    # spec 7.2 -- two GPUs, ZeRO-3, FA2, gradient checkpointing, global batch 32
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    # spec 8
    num_train_epochs: float = 1.0
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    logging_steps: int = 10
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    load_best_model_at_end: bool = False
    # plumbing
    remove_unused_columns: bool = False
    dataloader_num_workers: int = 4
    report_to: list[str] = field(default_factory=list)
    seed: int = 42
    data_seed: int | None = 42
    save_safetensors: bool = True
    # custom
    diag_every_n_steps: int = 50
    gen_diag_samples: int = 0
    gen_diag_max_new_tokens: int = 512

    def __post_init__(self):
        if self.gradient_checkpointing and not self.gradient_checkpointing_kwargs:
            # non-reentrant keeps grads flowing when part of a checkpointed
            # subtree has requires_grad=False (the frozen vision blocks)
            self.gradient_checkpointing_kwargs = {"use_reentrant": False}
        super().__post_init__()


FROZEN_HPARAMS: dict[str, Any] = {
    "optim": "adamw_torch",
    "learning_rate": 1e-5,
    "weight_decay": 0.0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 1.0,
    "bf16": True,
    "num_train_epochs": 1.0,
    "eval_steps": 500,
    "save_steps": 500,
    "save_total_limit": 3,
    "gradient_checkpointing": True,
}

ALLOWED_BATCH_COMBOS = ((4, 4), (2, 8))  # (per_device_train_batch_size, GAS) at world_size 2


def validate_frozen_hyperparameters(
    args: SFTTrainingArguments, world_size: int, allow_nonspec: bool = False
) -> dict[str, Any]:
    """Refuse to start if a spec-frozen hyperparameter drifted.

    ``allow_nonspec`` (env ``Q3VL_ALLOW_NONSPEC=1``) downgrades the failure to a
    warning. It exists solely so the single-GPU integration smoke can exercise
    the real entrypoint; the real run must never set it.
    """
    violations = []
    for key, want in FROZEN_HPARAMS.items():
        got = getattr(args, key)
        if isinstance(want, float):
            ok = abs(float(got) - want) < 1e-12
        elif isinstance(want, str):
            ok = str(got).lower().endswith(want.lower()) or str(got) == want
        else:
            ok = got == want
        if not ok:
            violations.append(f"{key}: spec {want!r}, config {got!r}")

    combo = (args.per_device_train_batch_size, args.gradient_accumulation_steps)
    if combo not in ALLOWED_BATCH_COMBOS:
        violations.append(
            f"(per_device_train_batch_size, gradient_accumulation_steps)={combo} is not one of "
            f"{ALLOWED_BATCH_COMBOS} (spec 7.2)"
        )
    effective = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
    if effective != GLOBAL_BATCH_SIZE:
        violations.append(
            f"effective global batch {effective} != {GLOBAL_BATCH_SIZE} "
            f"(micro {args.per_device_train_batch_size} x GAS {args.gradient_accumulation_steps} "
            f"x world_size {world_size})"
        )
    if world_size != 2:
        violations.append(f"world_size {world_size} != 2 (spec 7.2)")
    if violations:
        message = "spec-frozen hyperparameters violated:\n  - " + "\n  - ".join(violations)
        if not allow_nonspec:
            raise ValueError(message)
        import warnings

        warnings.warn("Q3VL_ALLOW_NONSPEC is set -- " + message, stacklevel=2)
    return {
        "spec_violations": violations,
        "effective_global_batch": effective,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "world_size": world_size,
    }
