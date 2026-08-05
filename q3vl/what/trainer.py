"""Protocol 10.4 -- the Stage-What training loop.

```yaml
optimizer: AdamW
query_connector_backend_lr: 1.0e-4
decoder_head_lr:            1.0e-4
shared_geometry_lr:         5.0e-5       # SB48 only
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

    "geometry, bias, LayerNorm and ModLN parameters get no weight decay.  The
     0.5 and 1.0 epoch checkpoints are protected; ordinary checkpoints are kept
     at most 3 deep, and the selected and protected ones are exempt from the
     rolling deletion."

Two things the loop is strict about:

* **Checkpoint selection may never use val loss** (campaign red line).  Eval loss
  is logged as a diagnostic; :meth:`WhatTrainer.best` ranks by the protocol 12.4
  board and nothing else.
* **The gradient-norm ratio of protocol 9.2 is logged**, on a fixed schedule that
  always includes step 0.  It costs two extra backward passes, which is cheap
  next to finding out at the end of a run that the Lab term carried 200x the
  gradient of the reconstruction term.

Numerics: the connector, backend and heads run under bf16 autocast; everything
from the renderer onwards -- the Gaussian mixture, the 33^3 bake, the tetrahedral
readback and the whole loss -- runs in float32 with autocast explicitly disabled.
The mixture normaliser exponentiates and the bake gate is stated at 1e-4 RGB,
which is below bf16's resolution at 1.0 (~4e-3): measuring it in bf16 would be
measuring the dtype.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from q3vl.where.calibrate import make_scheduler

from .config import (ArmConfig, CONTEXT_GENERATED, CONTEXT_GT,
                     PROTECTED_EPOCHS, TrainConfig)
from .context import BalancedContextSampler, context_breakdown, iter_modes
from .data import Batch
from .gaussians import render
from .losses import StyleQueue, bake_readback, compute_loss, grad_norm_ratio
from .metrics import activation_stats, bake_metrics, style_diagnostics
from .model import WhatModel

__all__ = ["compute_batch", "build_optimizer", "param_groups", "WhatTrainer",
           "TrainState"]


def _no_decay(name: str, p: torch.Tensor) -> bool:
    """Protocol 10.4: geometry, bias, LayerNorm and ModLN take no weight decay."""
    if p.dim() <= 1:                        # biases, LayerNorm weights, scalar gates
        return True
    if ".mod_" in name or name.startswith("mod_"):   # ModLN, incl. its projection
        return True
    if "geometry" in name or "norm" in name:
        return True
    return False


def _group_of(name: str) -> str:
    if "geometry.raw" in name:
        return "geometry"
    if ".heads." in name or ".provisional." in name:
        return "head"
    return "backend"


def param_groups(model: WhatModel, cfg: TrainConfig) -> list[dict[str, Any]]:
    lrs = {"backend": cfg.backend_lr, "head": cfg.head_lr, "geometry": cfg.geometry_lr}
    buckets: dict[tuple[str, bool], list[torch.nn.Parameter]] = {}
    names: dict[tuple[str, bool], list[str]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = (_group_of(name), _no_decay(name, p))
        buckets.setdefault(key, []).append(p)
        names.setdefault(key, []).append(name)
    groups = []
    for (grp, nd), ps in sorted(buckets.items()):
        groups.append({
            "params": ps, "lr": lrs[grp],
            "weight_decay": 0.0 if nd else cfg.weight_decay,
            "name": f"{grp}{'_no_decay' if nd else ''}",
            "n_params": sum(p.numel() for p in ps),
            "n_tensors": len(ps),
        })
    return groups


def build_optimizer(model: WhatModel, cfg: TrainConfig) -> torch.optim.Optimizer:
    groups = param_groups(model, cfg)
    opt = torch.optim.AdamW(
        [{k: v for k, v in g.items() if k in ("params", "lr", "weight_decay")}
         for g in groups], lr=cfg.backend_lr)
    opt._q3vl_groups = [{k: v for k, v in g.items() if k != "params"}   # type: ignore[attr-defined]
                        for g in groups]
    return opt


# --- one micro-batch --------------------------------------------------------

@torch.no_grad()
def _per_context_loss(t_pred: torch.Tensor, t_gt: torch.Tensor,
                      contexts: Sequence[Any]) -> dict[str, float]:
    """``L_func`` split by teacher / generated context (amendment A-4 item 2)."""
    from .losses import charbonnier

    per_sample = charbonnier(t_pred - t_gt).mean(dim=(1, 2))
    out: dict[str, float] = {}
    for mode in sorted({c.mode for c in contexts}):
        sel = torch.tensor([c.mode == mode for c in contexts],
                           device=per_sample.device)
        if bool(sel.any()):
            out[f"L_func_ctx_{mode}"] = float(per_sample[sel].mean())
    return out


@torch.no_grad()
def _geometry_stats(params: dict[str, torch.Tensor],
                    seed_geom: dict[str, torch.Tensor] | None) -> dict[str, float]:
    """Review nit N-6: two readings a results review cannot reconstruct later.

    ``geom_drift_*`` -- FG48 pools with the *provisional* geometry and renders
    with the *refined* one, and protocol 7.4 puts no constraint between them.  If
    they diverge, protocol 7.2's "each Gaussian's real colour distribution" stops
    being true and nobody would see it in the loss.  For SB48 the two are the same
    tensor and the drift is identically zero, which is itself the check.

    ``n_sigma_over_cube`` -- ``softplus`` has no upper bound (decision D-W2), and
    preflight already observed sigma = 1.42 on random parameters.  A Gaussian
    wider than the cube's edge covers the whole cube almost uniformly and has
    degenerated into an extra global affine branch.  How many of the 48 have done
    so is a §12.3-class reading, not a bug.
    """
    out: dict[str, float] = {
        "n_sigma_over_cube": float((params["sigma"] > 1.0).sum()),
        "sigma_max": float(params["sigma"].max()),
    }
    if seed_geom is not None:
        out["geom_drift_mu"] = float(
            (params["mu"] - seed_geom["mu"]).norm(dim=-1).mean())
        out["geom_drift_sigma"] = float(
            (params["sigma"] - seed_geom["sigma"]).abs().mean())
    return out


def compute_batch(model: WhatModel, batch: Batch, cfg: ArmConfig,
                  queue: StyleQueue | None = None, *, d_func_scale: float,
                  want_grad_ratio: bool = False) -> tuple[torch.Tensor, dict[str, Any]]:
    """Model forward + protocol 9 loss.  Everything after the model is float32.

    ``d_func_scale`` is the published train-set constant ``C`` of amendment A-2.
    It is a required keyword rather than a default so that no call site can
    quietly fall back to a per-batch scale.
    """
    out = model(**batch.inputs)
    device_type = batch.inputs["f_pre"].device.type
    with torch.autocast(device_type=device_type, enabled=False):
        params = {k: (v.float() if torch.is_tensor(v) else v)
                  for k, v in out.params.items()}
        x = torch.stack([t["x"] for t in batch.targets]).float()
        t_gt = torch.stack([t["t_gt"] for t in batch.targets]).float()
        z_gt = torch.stack([t["z_gt"] for t in batch.targets]).float()
        u_gt = torch.stack([t["u_gt"] for t in batch.targets]).float()
        kind = batch.targets[0]["query_kind"]
        z_style = out.z_style.float()

        t_pred = render(params, x, cfg.lut)
        t_read = bake_readback(params, cfg.lut, x)
        loss = compute_loss(
            t_pred=t_pred, t_gt=t_gt, x=x, params=params, z_style=z_style,
            z_gt=z_gt, u_gt=u_gt, d_func_scale=d_func_scale, lut_cfg=cfg.lut,
            query_kind=kind, t_read=t_read, queue=queue,
        )
        stats: dict[str, Any] = dict(loss.scalars)
        stats.update(bake_metrics(t_pred.detach(), t_read.detach()))
        stats.update(activation_stats(params))
        stats.update(style_diagnostics(z_style, z_gt, u_gt))
        stats.update(out.extra.get("pool", {}))
        stats.update(_geometry_stats(params, out.extra.get("seed_geometry")))
        if batch.contexts:
            # amendment A-4: the 50/50 ratio is asserted per micro-batch, and the
            # per-context loss is reported so "generated is worse by X" is a
            # number in every step row rather than a post-hoc reconstruction.
            stats.update(context_breakdown(batch.contexts))
            stats.update(_per_context_loss(t_pred, t_gt, batch.contexts))
        if want_grad_ratio:
            stats.update(grad_norm_ratio(model.parameters(), loss.parts["L_func"],
                                         loss.parts["L_hc"]))
    if queue is not None:
        queue.push(z_style)
    return loss.total, stats


@dataclass
class TrainState:
    step: int = 0
    micro_step: int = 0
    epoch_float: float = 0.0
    total_steps: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    saved: list[dict[str, Any]] = field(default_factory=list)
    #: step of the incumbent ``V_what`` best (protocol 12.4), or ``None``
    best_step: int | None = None
    #: steps an eval named as best but whose file was already rolled away.  Must
    #: stay empty; a non-empty list means protocol 10.4's "the selected and
    #: protected checkpoints are exempt from rolling deletion" was violated.
    lost_best_steps: list[int] = field(default_factory=list)


class WhatTrainer:
    def __init__(self, model: WhatModel, builder, dataset, arm_cfg: ArmConfig,
                 train_cfg: TrainConfig, *, run_dir: Path,
                 device: str | torch.device = "cuda",
                 eval_fn: Callable[[int], dict[str, Any]] | None = None,
                 log_every: int = 10, order: Sequence[int] | None = None):
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
        # amendment A-4: training is always 50/50, so a builder that cannot
        # produce a generated context cannot train.  Refused here rather than at
        # the first generated micro-batch, which would be a few minutes in.
        if getattr(builder, "color_genctx", None) is None:
            raise ValueError(
                "the batch builder has no generated <color> context store; "
                "amendment A-4 makes training 50% generated and forbids falling "
                "back to the GT span.  Run the extended make_generated_context "
                "job for this split first."
            )
        self.generated_mode = CONTEXT_GENERATED
        self.state = TrainState()
        self.queue = StyleQueue(train_cfg.style_queue_size)
        # amendment A-2: C comes from the builder, which got it from the published
        # zgt_center.npz.  Reading it off the builder rather than taking it as an
        # argument means the loss and the batch cannot disagree about it.
        self.d_func_scale = float(getattr(builder, "d_func_scale", 0.0) or 0.0)
        if self.d_func_scale <= 0.0:
            raise ValueError(
                "the batch builder has no published d_func scale C; run "
                "scripts/make_zgt_center.py and pass its constant "
                "(amendment A-2 forbids a per-batch scale)")

        torch.manual_seed(train_cfg.seed)
        # amendment A-4: every micro-batch is exactly 50% teacher / 50% generated.
        # Enforcing the ratio at the *micro* batch makes it hold for every
        # gradient-accumulated effective batch whatever the accumulation factor
        # is -- the same argument Where-B's BalancedContextSampler is built on,
        # and the same class, so the two stages cannot drift apart.
        self.sampler = BalancedContextSampler(
            len(dataset), train_cfg.micro_batch, seed=train_cfg.seed,
            teacher_fraction=train_cfg.teacher_fraction)
        self.order = list(order) if order is not None else None
        self.gas = train_cfg.grad_accum()
        n_micro = (len(self.order) // train_cfg.micro_batch if self.order is not None
                   else len(self.sampler))
        self.state.total_steps = max(1, n_micro // self.gas)
        self.optimizer = build_optimizer(self.model, train_cfg)
        self.scheduler = make_scheduler(self.optimizer, self.state.total_steps,
                                        train_cfg.warmup_ratio, train_cfg.scheduler)
        self.autocast = (
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            if train_cfg.precision == "bf16" and self.device.type == "cuda"
            else torch.autocast(device_type="cpu", enabled=False))

    # -- setup record -------------------------------------------------------
    def setup(self) -> dict[str, Any]:
        return {
            "arm": self.arm_cfg.arm, "wc": self.arm_cfg.wc,
            "generator": self.arm_cfg.generator,
            "where_source": self.arm_cfg.where_source,
            "model": self.model.facts(),
            "train": asdict(self.cfg),
            "param_groups": getattr(self.optimizer, "_q3vl_groups", []),
            "grad_accum": self.gas,
            "total_optimizer_steps": self.state.total_steps,
            "n_dataset": len(self.dataset),
            "style_queue": self.queue.facts(),
            "d_func_scale": self.d_func_scale,
            "teacher_fraction": self.cfg.teacher_fraction,
            "generated_mode": self.generated_mode,
            "sampler": self.sampler.facts(),
            "builder": self.builder.facts() if hasattr(self.builder, "facts") else {},
        }

    def micro_batches(self):
        """``[(dataset_index, context_mode), ...]`` per micro-batch.

        ``order`` is an explicit index list used by the mock loop and by resume;
        it is paired with the same 50/50 alternation so a fixed order does not
        quietly become a teacher-only run.
        """
        mb = self.cfg.micro_batch
        if self.order is None:
            yield from iter_modes(self.sampler)
            return
        half = mb // 2
        modes = [CONTEXT_GT] * half + [self.generated_mode] * (mb - half)
        for i in range(0, len(self.order) - mb + 1, mb):
            yield list(zip(self.order[i:i + mb], modes))

    # -- one epoch ----------------------------------------------------------
    def train(self, max_steps: int | None = None) -> TrainState:
        t0 = time.time()
        log = (self.run_dir / "steps.jsonl").open("a")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        accum = 0
        try:
            for micro in self.micro_batches():
                want_ratio = (self.state.step % self.cfg.grad_ratio_every == 0
                              and accum == 0)
                # N-3: the pooling diagnostics sync; only pay for them on the
                # steps whose row actually reaches steps.jsonl.
                self.model.collect_pool_stats = (
                    accum == 0 and (want_ratio
                                    or (self.state.step + 1) % self.log_every == 0))
                samples = [self.dataset[i] for i, _ in micro]
                modes = [m for _, m in micro]
                batch = self.builder.build(samples, modes)
                with self.autocast:
                    total, stats = compute_batch(
                        self.model, batch, self.arm_cfg, self.queue,
                        d_func_scale=self.d_func_scale,
                        want_grad_ratio=want_ratio)
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
                    self.state.micro_step * self.cfg.micro_batch / max(1, len(self.dataset)))

                row = {
                    "step": self.state.step,
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
                self._maybe_protect()
                if max_steps is not None and self.state.step >= max_steps:
                    break
        finally:
            log.close()
        self.save("final", protected=True)
        if self.eval_fn:
            self._eval_and_record()
        return self.state

    def _maybe_protect(self) -> None:
        for e in PROTECTED_EPOCHS:
            tag = f"epoch{e}"
            if self.state.epoch_float >= e and not any(
                    c["tag"] == tag for c in self.state.saved):
                self.save(tag, protected=True)

    def _eval_and_record(self) -> None:
        self.model.eval()
        try:
            report = self.eval_fn(self.state.step)
        finally:
            self.model.train()
        report["step"] = self.state.step
        self.state.checkpoints.append(report)
        (self.run_dir / "eval.jsonl").open("a").write(json.dumps(report) + "\n")
        self._protect_best()

    # -- checkpoints --------------------------------------------------------
    def save(self, tag: str, protected: bool = False) -> Path:
        path = self.run_dir / f"what_{tag}.pt"
        torch.save({
            "arm": self.arm_cfg.arm, "step": self.state.step,
            "epoch": self.state.epoch_float,
            "model": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "facts": self.model.facts(),
            "train_cfg": asdict(self.cfg),
        }, path)
        self.state.saved.append({"tag": tag, "path": str(path),
                                 "step": self.state.step, "protected": protected,
                                 "best_protected": False, "deleted": False})
        # order matters: mark the incumbent best *before* rolling, otherwise the
        # file that was just pushed over the keep_last edge is gone by the time
        # anything asks which checkpoint won.
        self._protect_best()
        self._roll()
        return path

    # -- protocol 10.4: "the selected and protected checkpoints are exempt from
    #    the rolling deletion" ------------------------------------------------
    def _protect_best(self) -> None:
        """Mark the incumbent ``V_what`` best as un-deletable (review blocker B-2).

        Without this, ``keep_last = 3`` at ``save_steps = 500`` over a ~5k-step
        epoch deletes the first six of nine ordinary checkpoints, and a run whose
        best ``V_what`` score lands at step 500-3000 arrives at selection time with
        no file to select.  Protocol 12.4 ("only the best checkpoint of an arm
        enters the cross-arm ranking") silently presupposes the file still exists.

        Two flags, deliberately separate:

        ``protected``       a protocol milestone -- the 0.5 / 1.0 epoch checkpoints
                            and ``final``.  Never cleared.
        ``best_protected``  the incumbent best.  Moves as the best moves, so the
                            run keeps exactly one extra file, not one per eval.

        Called from **both** paths that can create or invalidate a protection --
        after every ``save`` and after every eval.  The two are interleaved
        (``eval_steps == save_steps == 500``, eval first), so an eval can name a
        best whose file does not exist yet, and a save can create the file for a
        best that was named one line earlier.  Marking from one side only leaves
        the other side's ordering unprotected; that is the shape of the
        double-deletion-path bug S0-TRAIN hit, and it is why
        ``tests/test_e2e_mock.py`` exercises both orderings rather than one.
        """
        b = self.best()
        for c in self.state.saved:
            c["best_protected"] = False
        if b is None:
            return
        step = int(b["step"])
        self.state.best_step = step
        entries = [c for c in self.state.saved if c["step"] == step]
        alive = [c for c in entries if not c.get("deleted")]
        if any(c["protected"] for c in alive):
            # a milestone already pins this step's weights; a second flag on the
            # same content would keep an extra copy for nothing
            return
        if alive:
            alive[-1]["best_protected"] = True      # the newest surviving file
        elif entries:
            # every file for that step has been unlinked: that is exactly the
            # failure this method exists to prevent, so it is recorded loudly
            # rather than swallowed.
            if step not in self.state.lost_best_steps:
                self.state.lost_best_steps.append(step)
        # `not entries` is the benign case: eval runs *before* save at the same
        # step (eval_steps == save_steps == 500), so the incumbent's file does not
        # exist yet.  The save that follows calls this method again and protects
        # it.  Treating that as a loss would make the check cry wolf on every
        # single eval and nobody would read it afterwards.

    def _roll(self) -> None:
        """Keep at most ``keep_last`` ordinary checkpoints (protocol 10.4).

        "Ordinary" excludes both protections.  Deleted entries stay in
        ``state.saved`` with ``deleted: True`` so the checkpoint index remains a
        complete history -- dropping them would also make ``_maybe_protect``
        re-save an epoch milestone it had already written.
        """
        if self.cfg.keep_last is None:
            return
        ordinary = [c for c in self.state.saved
                    if not c["protected"] and not c.get("best_protected")
                    and not c.get("deleted")]
        excess = len(ordinary) - self.cfg.keep_last
        for c in ordinary[:max(0, excess)]:
            Path(c["path"]).unlink(missing_ok=True)
            c["deleted"] = True

    def checkpoint_index(self) -> list[dict[str, Any]]:
        """Protocol 13.1's ``checkpoint index``, including what was rolled away."""
        return [dict(c) for c in self.state.saved]

    def best(self, key: str | None = None,
             bigger_is_better: bool = False) -> dict[str, Any] | None:
        """Best recorded checkpoint on the protocol 12.4 board.

        Red line: never ``eval_loss``.  The default key is
        ``TrainConfig.selection_key`` = ``ONLINE_SELECTION_KEY``, the in-loop
        **proxy** for protocol 12.4's primary ordering: same direction (local
        CIEDE2000, generated context, smaller is better) computed on the fixed
        eval subset without rendering images.  It decides which checkpoint files
        survive ``_roll``; the offline board
        (``scripts/evaluate_what.py`` -> ``main_board``) decides which
        checkpoint wins.  A proxy pointing the other way would let the rolling
        deletion discard the file the offline board later wants, which is blocker
        B-2 one level up.

        Gate-failing checkpoints are skipped when the eval report carries
        ``gate_pass`` -- selection may not land on a checkpoint the gate rejected
        (protocol 12.4 step 1), and neither may the protection that keeps a
        checkpoint alive *for* selection.  ``gate_fallback`` in the returned dict
        (review N-18) marks the case where *every* checkpoint failed and this is
        therefore a diagnostic pick, not a selection.
        """
        key = key or self.cfg.selection_key
        scored = [c for c in self.state.checkpoints if c.get(key) is not None]
        gated = [c for c in scored if c.get("gate_pass", True)]
        n_unknown = sum(1 for c in scored if "gate_pass" not in c)
        pool = gated or scored
        if not pool:
            return None
        pick = max if bigger_is_better else min
        best = dict(pick(pool, key=lambda c: float(c[key])))
        # review N-18: same shape as main_board's ``selection_possible: false``
        best["gate_fallback"] = not gated
        # review N-19: a report with no ``gate_pass`` key counted as passing; it
        # is still treated as passing (the production arm_metrics always writes
        # it) but the count is surfaced instead of being invisible.
        best["n_gate_unknown"] = n_unknown
        return best
