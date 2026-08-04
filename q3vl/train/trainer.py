"""Trainer subclass: protected milestone checkpoints + segment diagnostics.

Two behaviours the stock ``Trainer`` does not provide:

1. **Protected checkpoints (spec 8.1/8.2).** The 0.5- and 1.0-epoch steps are
   derived from ``state.max_steps`` at ``on_train_begin`` -- i.e. from the
   terminal manifest's ``N_effective`` via the dataset length -- never from the
   2645/5290 estimates in the spec. Those two checkpoints are force-saved and
   exempted from ``save_total_limit`` rolling deletion.
2. **Where/Color token loss (spec 4.4, 8.3)**, recorded as diagnostics only.

Checkpoint selection uses assistant-only ``eval_loss`` (spec 8.3), which is
what the masked labels already make ``eval_loss`` mean.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
from transformers import Trainer, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from .constants import SEG_IGNORE
from .diagnostics import (
    SegmentAccumulator,
    aggregate_generation_diagnostics,
    parse_two_segment,
    segment_token_stats,
)

logger = logging.getLogger(__name__)

NON_MODEL_KEYS = ("segment_ids",)
PROTECTED_MARKER = "protected_checkpoint.json"


class ProtectedStepCallback(TrainerCallback):
    """Derives and enforces the 0.5/1.0 epoch milestones."""

    def __init__(self, trainer_ref):
        self._trainer = trainer_ref

    def on_train_begin(self, args, state, control, **kwargs):
        trainer = self._trainer
        trainer.resolve_protected_steps(state.max_steps)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self._trainer.protected_steps:
            control.should_save = True
            control.should_evaluate = True
        return control


class Qwen3VLSFTTrainer(Trainer):
    def __init__(
        self,
        *args,
        data_manifest_info: dict[str, Any] | None = None,
        special_token_ids: dict[str, int] | None = None,
        freeze_report: dict[str, Any] | None = None,
        diag_every_n_steps: int = 50,
        gen_diag_samples: int = 0,
        gen_diag_max_new_tokens: int = 512,
        expected_steps_per_epoch: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data_manifest_info = data_manifest_info or {}
        self.special_token_ids = special_token_ids or {}
        self.freeze_report = freeze_report or {}
        self.diag_every_n_steps = max(0, int(diag_every_n_steps))
        self.gen_diag_samples = max(0, int(gen_diag_samples))
        self.gen_diag_max_new_tokens = int(gen_diag_max_new_tokens)
        self.expected_steps_per_epoch = expected_steps_per_epoch
        self.protected_steps: set[int] = set()
        self._train_acc = SegmentAccumulator()
        self._eval_acc = SegmentAccumulator()
        self.add_callback(ProtectedStepCallback(self))

    # -- protected checkpoints ---------------------------------------------
    def resolve_protected_steps(self, max_steps: int) -> set[int]:
        if max_steps <= 0:
            raise RuntimeError("max_steps must be positive to derive protected checkpoints")
        half = int(round(0.5 * max_steps))
        self.protected_steps = {s for s in (half, max_steps) if s > 0}
        msg = (
            f"[protected-checkpoints] steps_per_epoch(from trainer state)={max_steps} "
            f"-> half_epoch_step={half}, full_epoch_step={max_steps}"
        )
        if self.expected_steps_per_epoch is not None and abs(
            self.expected_steps_per_epoch - max_steps
        ) > 1:
            msg += (
                f"  WARNING: manifest-derived ceil(N_effective/global_batch)="
                f"{self.expected_steps_per_epoch} differs from trainer max_steps={max_steps}"
            )
        logger.warning(msg)
        return self.protected_steps

    def _sorted_checkpoints(self, output_dir=None, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False):
        """Hide the protected milestones from every deletion path.

        ``Trainer`` deletes checkpoints in two places, and only one of them goes
        through ``_rotate_checkpoints``: at the end of ``_inner_training_loop``
        transformers 4.57.1 additionally runs

            if should_save and best_model_checkpoint is not None and save_total_limit == 1:
                for checkpoint in checkpoints_sorted:
                    if not samefile(checkpoint, best_model_checkpoint): rmtree(checkpoint)

        which would silently delete the 0.5-epoch checkpoint. Both paths source
        their candidate list from ``_sorted_checkpoints``, so filtering here --
        after the base implementation has done its best-checkpoint index
        bookkeeping -- is the single point that closes both.
        """
        checkpoints_sorted = super()._sorted_checkpoints(
            output_dir=output_dir, checkpoint_prefix=checkpoint_prefix, use_mtime=use_mtime
        )
        if not self.protected_steps:
            return checkpoints_sorted
        protected_dirs = {f"{checkpoint_prefix}-{s}" for s in self.protected_steps}
        kept = [c for c in checkpoints_sorted if Path(c).name not in protected_dirs]
        if len(kept) != len(checkpoints_sorted):
            logger.debug(
                "protecting milestone checkpoints from deletion: %s",
                sorted(protected_dirs & {Path(c).name for c in checkpoints_sorted}),
            )
        return kept

    def _save_checkpoint(self, model, trial=None, **kwargs):
        out = super()._save_checkpoint(model, trial, **kwargs)
        try:
            ckpt_dir = os.path.join(
                self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            )
            if self.args.should_save and os.path.isdir(ckpt_dir):
                self._write_checkpoint_sidecar(ckpt_dir)
        except Exception:  # noqa: BLE001 - never lose a checkpoint over a sidecar
            logger.exception("failed to write checkpoint sidecar metadata")
        return out

    def _write_checkpoint_sidecar(self, ckpt_dir: str) -> None:
        step = self.state.global_step
        protected = step in self.protected_steps
        payload = {
            "global_step": step,
            "epoch": self.state.epoch,
            "protected": protected,
            "milestone": (
                "1.0_epoch" if step == max(self.protected_steps or {0})
                else "0.5_epoch" if protected else None
            ),
            "special_token_ids": self.special_token_ids,
            "data_manifest": self.data_manifest_info,
            "freeze_report": self.freeze_report,
            "protected_steps": sorted(self.protected_steps),
            "best_metric": self.state.best_metric,
            "best_model_checkpoint": self.state.best_model_checkpoint,
        }
        with open(os.path.join(ckpt_dir, PROTECTED_MARKER), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

    # -- loss ---------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        segment_ids = inputs.pop("segment_ids", None)
        want_diag = self._should_diagnose()
        outputs = model(**inputs)
        loss = outputs.loss
        if want_diag and segment_ids is not None and getattr(outputs, "logits", None) is not None:
            acc = self._train_acc if model.training else self._eval_acc
            per_token, correct, segs = segment_token_stats(
                outputs.logits.detach(), inputs["labels"], segment_ids
            )
            if segs.numel():
                acc.update(per_token, correct, segs)
        return (loss, outputs) if return_outputs else loss

    def _should_diagnose(self) -> bool:
        if not self.model.training:
            return True  # eval batches are cheap and the metric is required
        if self.diag_every_n_steps == 0:
            return False
        return (self.state.global_step % self.diag_every_n_steps) == 0

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if self._train_acc.n_batches and "loss" in logs:
            logs.update(self._train_acc.metrics(prefix="train_seg_"))
            self._train_acc.reset()
        try:
            super().log(logs, start_time)
        except TypeError:  # older signature
            super().log(logs)

    # -- eval ---------------------------------------------------------------
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self._eval_acc.reset()
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        extra = self._eval_acc.metrics(prefix=f"{metric_key_prefix}_seg_")
        if self.gen_diag_samples > 0:
            try:
                extra.update(
                    self.generation_diagnostics(
                        eval_dataset if eval_dataset is not None else self.eval_dataset,
                        prefix=f"{metric_key_prefix}_gen_",
                    )
                )
            except Exception as exc:  # noqa: BLE001 - diagnostics must not kill a run
                logger.warning("generation diagnostics failed: %r", exc)
                extra[f"{metric_key_prefix}_gen_failed"] = 1.0
        if extra:
            metrics.update(extra)
            self.log(extra)
        return metrics

    @torch.no_grad()
    def generation_diagnostics(self, dataset, prefix: str = "eval_gen_") -> dict[str, float]:
        """Spec 8.3 structural rates on a small fixed generated sample."""
        if dataset is None or len(dataset) == 0:
            return {}
        collator = self.data_collator
        tokenizer = collator.tokenizer
        n = min(self.gen_diag_samples, len(dataset))
        model = self.model
        was_training = model.training
        model.eval()
        parsed: list[dict[str, Any]] = []
        for i in range(n):
            sample = dataset[i]
            enc = collator.encode_one(sample)
            n_prompt = enc["n_prompt_tokens"]
            ids = torch.tensor([enc["input_ids"][:n_prompt]], device=self.args.device)
            image_inputs = collator.processor.image_processor(
                images=[sample.image], do_resize=False, return_tensors="pt"
            )
            gen = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                pixel_values=image_inputs["pixel_values"].to(self.args.device, model.dtype),
                image_grid_thw=image_inputs["image_grid_thw"].to(self.args.device),
                max_new_tokens=self.gen_diag_max_new_tokens,
                do_sample=False,
            )
            text = tokenizer.decode(gen[0][n_prompt:], skip_special_tokens=False)
            parsed.append(parse_two_segment(text))
        if was_training:
            model.train()
        return aggregate_generation_diagnostics(parsed, prefix=prefix)
