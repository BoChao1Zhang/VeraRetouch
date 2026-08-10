"""Protocol 10.3 -- the Where-B training loop.

```yaml
optimizer: AdamW
learning_rate: 2.0e-4
weight_decay: 0.01
warmup_ratio: 0.03
scheduler: cosine
max_grad_norm: 1.0
precision: bf16
effective_batch_per_arm: 32
epochs: 1.0
eval_steps: 500
save_steps: 500
```

    "One arm per GPU.  Probe the micro-batch for memory first, then set gradient
     accumulation so the effective batch stays 32.  The frozen VLM builds no
     parameter gradients but its vision and language forwards are the real
     model, not fake features."

Red line: **checkpoint selection may not use val loss.**  This loop therefore
records eval loss as a diagnostic only; the selection metric is the
generated-context ``V_where`` board of :mod:`q3vl.whereb.metrics`, and
:meth:`WhereBTrainer.best` ranks checkpoints by that.

Numerics: the connector runs under bf16 autocast, but everything from
``phi_dir`` onwards is float32.  The guided filter divides by ``var + 1e-3`` and
the readouts exponentiate; both lose too much in bf16, and the resulting mask is
the quantity the loss and every gate are defined on.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from q3vl.where.calibrate import make_scheduler

from .config import ArmConfig, TrainConfig
from .context import BalancedContextSampler
from .data import Batch, BatchBuilder, WhereBDataset
from .fields import predict_fields
from .losses import SampleLoss, aggregate, sample_loss, schedule_weights
from .model import WhereBModel

__all__ = ["compute_batch", "build_optimizer", "WhereBTrainer", "TrainState"]


def compute_batch(
    model: WhereBModel,
    batch: Batch,
    cfg: ArmConfig,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, Any], list[SampleLoss]]:
    """One micro-batch: model forward, then the protocol 5.5 loss per sample.

    The connector forward may run under bf16 autocast; **everything from
    ``phi_dir`` onwards runs with autocast explicitly disabled** and is asserted
    to be float32 (review blocker B4).  ``phi_dir @ w_dir`` is a matmul and would
    otherwise be demoted to bf16 by an outer autocast -- and since
    ``evaluate_context`` never used autocast, training and the gate would have
    been optimising and measuring two different functions, with the damage
    landing on the four CBand12 arms and not on the four Band arms.
    """
    out = model(**batch.inputs)
    device_type = batch.inputs["f_pre"].device.type
    losses: list[SampleLoss] = []
    with torch.autocast(device_type=device_type, enabled=False):
        for i, tgt in enumerate(batch.targets):
            params = {k: v.float() for k, v in out.select(i).items()}
            fields = predict_fields(
                tgt["phi_dir"].float(), params, cfg.readout,
                tgt["grid_h"], tgt["grid_w"],
                guide_hi=tgt["guide_hi"].float() if cfg.mask_loss_space == "hi" else None,
                up_cfg=cfg.upsample,
                require_dtype=torch.float32,
            )
            if fields["s_low"].dtype != torch.float32:
                raise AssertionError(
                    f"s_low is {fields['s_low'].dtype}, must be float32 "
                    "(review blocker B4)"
                )
            rho_pred = {k: v for k, v in params.items()
                        if k not in ("w0", "w_raw", "alpha_raw")}
            losses.append(sample_loss(
                readout=cfg.readout, fields=fields, rho_pred=rho_pred,
                mask_target=tgt["mask_hi"].float() if cfg.mask_loss_space == "hi"
                else tgt["mask_low"].float(),
                weights=weights,
                s_star=tgt.get("s_star"), r_star=tgt.get("r_star"),
                w_dir_star=tgt.get("w_dir_star"),
                mask_space=cfg.mask_loss_space,
            ))
        total, stats = aggregate(losses, [c.mode for c in batch.contexts])
    stats["s_dtype"] = str(losses[0].mask_term.dtype) if losses else None
    return total, stats, losses


def build_optimizer(model: WhereBModel, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW with the usual no-decay set for bias / LayerNorm / gates / positions."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        bare = (p.dim() <= 1) or name.endswith(("gate_text", "gate_vis"))
        bare = bare or name.endswith(("positions", "probe"))
        (no_decay if (cfg.no_decay_on_bias_norm and bare) else decay).append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=cfg.learning_rate)


@dataclass
class TrainState:
    step: int = 0
    micro_step: int = 0
    epoch_float: float = 0.0
    total_steps: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    #: evals that raised.  Non-empty => some checkpoints have no eval record and
    #: cannot be selected; the run is still valid, just partially unmeasured.
    eval_failures: list[dict[str, Any]] = field(default_factory=list)


class WhereBTrainer:
    def __init__(
        self,
        model: WhereBModel,
        builder: BatchBuilder,
        dataset: WhereBDataset,
        arm_cfg: ArmConfig,
        train_cfg: TrainConfig,
        *,
        run_dir: Path,
        device: str | torch.device = "cuda",
        eval_fn: Callable[[int], dict[str, Any]] | None = None,
        log_every: int = 10,
    ):
        self.model = model.to(device)
        self.builder = builder
        self.dataset = dataset
        self.arm_cfg = arm_cfg
        self.cfg = train_cfg
        self.device = torch.device(device)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.eval_fn = eval_fn
        self.log_every = log_every
        self.state = TrainState()

        torch.manual_seed(train_cfg.seed)
        self.sampler = BalancedContextSampler(
            len(dataset), train_cfg.micro_batch, seed=train_cfg.seed,
            teacher_fraction=train_cfg.teacher_fraction,
        )
        self.gas = train_cfg.grad_accum()
        self.state.total_steps = max(1, len(self.sampler) // self.gas)
        self.optimizer = build_optimizer(self.model, train_cfg)
        self.scheduler = make_scheduler(
            self.optimizer, self.state.total_steps, train_cfg.warmup_ratio,
            train_cfg.scheduler,
        )
        self.autocast = (
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            if train_cfg.precision == "bf16" and self.device.type == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )

    # -- setup record -------------------------------------------------------
    def setup(self) -> dict[str, Any]:
        return {
            "arm": self.arm_cfg.arm,
            "structure": self.arm_cfg.structure,
            "readout": self.arm_cfg.readout,
            "model": self.model.facts(),
            "train": asdict(self.cfg),
            "sampler": self.sampler.facts(),
            "grad_accum": self.gas,
            "total_optimizer_steps": self.state.total_steps,
            "n_dataset": len(self.dataset),
            "basis": self.builder.basis.facts(),
            "vlm": self.builder.vlm.facts(),
            "mask_loss_space": self.arm_cfg.mask_loss_space,
            "schedule_boundary_step": schedule_weights(0, self.state.total_steps)["boundary_step"],
        }

    # -- one epoch ----------------------------------------------------------
    def train(self) -> TrainState:
        t0 = time.time()
        log = (self.run_dir / "steps.jsonl").open("a")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        accum = 0
        try:
            for micro in self.sampler:
                weights = schedule_weights(self.state.step, self.state.total_steps)
                samples = [self.dataset[i] for i, _ in micro]
                modes = [m for _, m in micro]
                batch = self.builder.build(samples, modes)
                with self.autocast:
                    total, stats, _ = compute_batch(
                        self.model, batch, self.arm_cfg, weights
                    )
                (total / self.gas).backward()
                accum += 1
                self.state.micro_step += 1
                if accum < self.gas:
                    continue

                gnorm = float(torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm))
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                accum = 0
                self.state.step += 1
                self.state.epoch_float = (
                    self.state.micro_step * self.cfg.micro_batch / max(1, len(self.dataset))
                )

                row = {
                    "step": self.state.step, "stage": weights["stage"],
                    "weights": {k: v for k, v in weights.items() if k in ("mask", "s", "curve", "dir")},
                    # review blocker B5 / ruling D-B16: the effective auxiliary
                    # weight is nominal x aux_effective_scale.  It is 1.0 under
                    # the "with_oracle" denominator; logging it makes any future
                    # dilution visible in steps.jsonl instead of invisible.
                    "effective_aux_weights": {
                        k: weights[k] * float(stats.get("aux_effective_scale", 1.0))
                        for k in ("s", "curve", "dir")
                    },
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "grad_norm": gnorm, "elapsed_s": round(time.time() - t0, 1),
                    "epoch": round(self.state.epoch_float, 4), **stats,
                }
                self.state.history.append(row)
                log.write(json.dumps(row) + "\n")
                if self.state.step % self.log_every == 0:
                    log.flush()
                    print(json.dumps(row), flush=True)
                if self.eval_fn and self.state.step % self.cfg.eval_steps == 0:
                    self._eval_and_record()
                if self.state.step % self.cfg.save_steps == 0:
                    self.save(f"step{self.state.step}")
        finally:
            log.close()
        self.save("final")
        if self.eval_fn:
            self._eval_and_record()
        return self.state

    def _eval_and_record(self) -> None:
        """Run the eval callback, but never let it kill an unattended arm.

        An arm is ~10 h of GPU time; an exception inside evaluation (a wedged
        read mount, a missing genctx record, an OOM in a seven-board pass) used
        to propagate and take the whole run with it, losing the training that had
        already completed.  Evaluation is a *reporting* step -- protocol 5.6
        selects from eval records, so a checkpoint with no record is simply not
        selectable, which is the honest outcome and not a silent one.

        So: record the failure loudly (status file with the traceback, an
        upper-case log marker) and keep training.  What is NOT done is swallowing
        it -- ``EVAL_FAILED_step*.json`` on disk and ``eval_failures`` in the
        state make a run with broken evaluation impossible to mistake for a
        healthy one.
        """
        import traceback

        self.model.eval()
        try:
            report = self.eval_fn(self.state.step)
        except BaseException as exc:                  # noqa: BLE001
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise                                 # operator intent, not a fault
            tb = traceback.format_exc()
            marker = {
                "step": self.state.step,
                "epoch": round(self.state.epoch_float, 4),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": tb,
                "arm": self.arm_cfg.arm,
                "note": ("evaluation failed; TRAINING CONTINUED. This checkpoint "
                         "has no eval record and is therefore not selectable "
                         "under protocol 5.6."),
            }
            path = self.run_dir / f"EVAL_FAILED_step{self.state.step}.json"
            path.write_text(json.dumps(marker, indent=2, ensure_ascii=False),
                            encoding="utf-8")
            print(f"!!! EVAL FAILED at step {self.state.step} -- TRAINING CONTINUES; "
                  f"see {path} !!!\n{tb}", flush=True)
            self.state.eval_failures.append(
                {"step": self.state.step, "error": marker["error"],
                 "marker": str(path)})
            return
        finally:
            self.model.train()
        report["step"] = self.state.step
        self.state.checkpoints.append(report)
        (self.run_dir / "eval.jsonl").open("a").write(json.dumps(report) + "\n")

    # -- checkpoints --------------------------------------------------------
    def save(self, tag: str) -> Path:
        path = self.run_dir / f"where_b_{tag}.pt"
        torch.save({
            "arm": self.arm_cfg.arm,
            "step": self.state.step,
            "model": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "facts": self.model.facts(),
            "basis_digest": self.builder.basis.digest(),
            "train_cfg": asdict(self.cfg),
        }, path)
        return path

    def best(self, key: str = "local_soft_iou_median") -> dict[str, Any] | None:
        """Best recorded checkpoint on the generated-context board.

        Red line: never ``eval_loss``.  ``key`` names a metric produced by
        :func:`q3vl.whereb.metrics.arm_metrics`.
        """
        scored = [c for c in self.state.checkpoints if c.get(key) is not None]
        if not scored:
            return None
        return max(scored, key=lambda c: float(c[key]))


def probe_micro_batch(
    builder: BatchBuilder, dataset: WhereBDataset, model: WhereBModel,
    arm_cfg: ArmConfig, candidates: Sequence[int] = (2, 4, 8),
    device: str = "cuda",
) -> dict[str, Any]:
    """Protocol 10.3: find the largest even micro-batch that fits, then derive GAS."""
    rows = []
    for mb in candidates:
        if mb % 2:
            continue
        try:
            torch.cuda.reset_peak_memory_stats()
            samples = [dataset[i] for i in range(mb)]
            modes = ["gt", "generated"] * (mb // 2)
            batch = builder.build(samples, modes[:mb])
            weights = schedule_weights(0, 100)
            total, _stats, _ = compute_batch(model, batch, arm_cfg, weights)
            total.backward()
            model.zero_grad(set_to_none=True)
            rows.append({"micro_batch": mb, "ok": True,
                         "peak_gib": torch.cuda.max_memory_allocated() / 2**30})
        except torch.cuda.OutOfMemoryError:
            rows.append({"micro_batch": mb, "ok": False, "peak_gib": None})
            torch.cuda.empty_cache()
            break
    ok = [r for r in rows if r["ok"]]
    return {"rows": rows, "chosen": ok[-1]["micro_batch"] if ok else None}
