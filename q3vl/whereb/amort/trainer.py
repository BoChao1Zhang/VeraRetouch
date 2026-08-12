"""The PR-AMORT training loop.

Deliberately a separate loop from :class:`q3vl.whereb.trainer.WhereBTrainer`
rather than a subclass: the model signature, the loss and the per-sample (not
per-padded-batch) forward all differ, and inheriting would have meant overriding
every method that matters while pretending the contract was shared.

Two disciplines are wired in rather than left to the report:

* **Early-warning columns every step.**  ``area_ratio_median``,
  ``std_ratio_median``, the online swap-subject delta and the antonym
  invariance are the registered stop-and-check triggers.  W01/W02's failure
  signature -- an over-covering, low-variance, centre-prior-shaped field -- is
  visible in these columns within a few hundred steps, long before any
  evaluation.  :meth:`AmortTrainer._warn` prints a loud marker when one trips.

* **Checkpoint selection never reads a val loss** (red line).  Every save is
  scored by the quick eval and selected on median local soft-IoU behind hard
  gates; a checkpoint with no eval record is simply not selectable.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from q3vl.where.calibrate import make_scheduler

from .losses import LossWeights, aggregate, amort_sample_loss, signed_distance_field

__all__ = ["AmortTrainConfig", "AmortTrainer", "compute_micro_batch"]


@dataclass(frozen=True)
class AmortTrainConfig:
    arm: str = "P1"
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    scheduler: str = "cosine"
    max_grad_norm: float = 1.0
    precision: str = "bf16"
    effective_batch: int = 32
    micro_batch: int = 8
    epochs: float = 1.0
    eval_steps: int = 500
    save_steps: int = 500
    seed: int = 20260810
    teacher_fraction: float = 0.5
    #: review U2.  "mixed" = the registered teacher/generated mix; the ceiling
    #: rows pin every non-fake sample to one context instead.
    train_context: str = "mixed"
    #: hard wall clock.  Backstop only -- see `max_steps`.
    max_hours: float = 3.6
    #: review U4.  The A/B must be matched on TRAINING STEPS, not on wall clock:
    #: P1 carries an extra float32, autocast-disabled `predict_fields` stage that
    #: P3' does not, so equal wall clock buys unequal steps AND stops the two
    #: arms at different points of the same cosine schedule -- confounding the
    #: one question the pair exists to answer.  0 = unbounded (wall clock only).
    max_steps: int = 0
    #: registered early-warning thresholds
    warn_area_ratio_hi: float = 1.5
    warn_area_ratio_lo: float = 0.8
    warn_std_ratio_lo: float = 0.5
    warn_after_frac: float = 0.20

    def grad_accum(self) -> int:
        if self.effective_batch % self.micro_batch:
            raise ValueError(
                f"effective batch {self.effective_batch} is not a multiple of "
                f"micro batch {self.micro_batch}"
            )
        return self.effective_batch // self.micro_batch


def compute_micro_batch(
    model,
    inputs: Sequence[Any],
    weights: LossWeights,
    *,
    want_hi: bool = False,
    sdf_cache: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, Any], list[dict[str, Any]]]:
    """Forward + loss for one micro-batch, sample by sample.

    Per-sample rather than a padded batch, because the H/16 grid genuinely
    varies (13 distinct shapes in 400 samples) and reflect-padding a padded
    batch would convolve real cells against another image's pad.  There are
    therefore **no pad cells at all** in this representation, which is asserted
    rather than assumed.
    """
    losses, rows = [], []
    for x in inputs:
        cond = model.cond_of(x.cond_h, x.cond_mask, x.word_ids, x.word_offsets)
        if x.route_semantic and model.sem is not None:
            out = model.forward_sem(x.feat, cond, sim=x.sim, center=x.center,
                                    geom=x.geom,
                                    guide_hi=x.guide_hi if want_hi else None)
            head = "semantic"
        else:
            out = model.forward_geo(x.feat, cond, x.phi_dir, sim=x.sim,
                                    center=x.center, geom=x.geom,
                                    guide_hi=x.guide_hi if want_hi else None,
                                    grid_h=x.grid_h, grid_w=x.grid_w)
            head = "geometry"
        m_low = out["m_low"]
        eik_term = None
        if weights.eik and "s_raw" in out:
            from .losses import eikonal

            eik_term = eikonal(out["s_raw"])
        if m_low.shape != (x.grid_h, x.grid_w):
            raise AssertionError(
                f"{x.sample_id}: m_low is {tuple(m_low.shape)}, grid is "
                f"{(x.grid_h, x.grid_w)}"
            )
        phi_sdf = None
        if weights.sdf and not x.is_fake:
            key = f"{x.sample_id}@{x.grid_h}x{x.grid_w}"
            if sdf_cache is not None and key in sdf_cache:
                phi_sdf = sdf_cache[key]
            else:
                phi_sdf = signed_distance_field(x.gt_low)
                if sdf_cache is not None:
                    sdf_cache[key] = phi_sdf
        sl = amort_sample_loss(
            m_low, x.gt_low, weights, valid=None, phi_sdf=phi_sdf,
            gt_partner=x.gt_partner_low, is_fake=x.is_fake,
            # family-gated: analytic families only
            structural=x.family in ("radial", "linear", "band"),
        )
        if eik_term is not None:
            sl.total = sl.total + weights.eik * eik_term
            sl.terms["eik"] = eik_term
        losses.append(sl)
        rows.append({"sample_id": x.sample_id, "head": head, "family": x.family,
                     "is_fake": x.is_fake, **sl.stats})
    total, stats = aggregate(losses)
    n_sem = sum(1 for r in rows if r["head"] == "semantic")
    stats["frac_semantic_head"] = n_sem / max(1, len(rows))
    return total, stats, rows


@dataclass
class AmortState:
    step: int = 0
    micro_step: int = 0
    total_steps: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    eval_failures: list[dict[str, Any]] = field(default_factory=list)
    nan_steps: int = 0
    stopped_reason: str = ""


class AmortTrainer:
    def __init__(
        self,
        model,
        builder,
        dataset,
        indices: Sequence[int],
        cfg: AmortTrainConfig,
        weights: LossWeights,
        *,
        run_dir: Path,
        device: str = "cuda",
        eval_fn: Callable[[int], dict[str, Any]] | None = None,
        log_every: int = 10,
        want_hi: bool = False,
    ):
        self.model = model.to(device)
        self.builder = builder
        self.dataset = dataset
        self.indices = list(indices)
        self.cfg = cfg
        self.weights = weights
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.eval_fn = eval_fn
        self.log_every = log_every
        self.want_hi = want_hi
        self.state = AmortState()
        self.sdf_cache: dict[str, torch.Tensor] = {}
        self.fake_mode = "foreign"
        #: falls back to teacher text when the arm has no generated-context
        #: store; recorded in setup() rather than silently substituted.
        self.gen_mode = "generated" if getattr(builder, "genctx", None) else "gt"

        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.gas = cfg.grad_accum()
        n_micro = int(len(self.indices) * cfg.epochs) // cfg.micro_batch
        self.state.total_steps = max(1, n_micro // self.gas)
        # review U4.  Clamp HERE -- after the data-derived value and *before* the
        # scheduler is constructed.  Both placements matter: an earlier clamp is
        # overwritten by the line above (measured: --max-steps 6 still ran 8
        # steps), and a later one would leave the cosine schedule stretched over
        # a horizon the run never reaches, so the two arms would stop at
        # different points of their LR curves -- exactly the confound the
        # step-matching is meant to remove.
        if cfg.max_steps:
            self.state.total_steps = min(self.state.total_steps, int(cfg.max_steps))
        self.optimizer = torch.optim.AdamW(
            [
                {"params": [p for n, p in model.named_parameters()
                            if p.requires_grad and p.dim() > 1],
                 "weight_decay": cfg.weight_decay},
                {"params": [p for n, p in model.named_parameters()
                            if p.requires_grad and p.dim() <= 1],
                 "weight_decay": 0.0},
            ],
            lr=cfg.learning_rate,
        )
        self.scheduler = make_scheduler(self.optimizer, self.state.total_steps,
                                        cfg.warmup_ratio, cfg.scheduler)
        self.autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if cfg.precision == "bf16" and self.device.type == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )

    # -- setup record -------------------------------------------------------
    def setup(self) -> dict[str, Any]:
        return {
            "arm": self.cfg.arm,
            "model": self.model.facts(),
            "train": asdict(self.cfg),
            "loss_weights": self.weights.to_dict(),
            "grad_accum": self.gas,
            "total_optimizer_steps": self.state.total_steps,
            "n_train_samples": len(self.indices),
            "builder": self.builder.facts(),
            "want_hi_in_training_loss": self.want_hi,
            "nan_steps": self.state.nan_steps,
            "train_generated_context": self.gen_mode,
        }

    # -- warnings -----------------------------------------------------------
    def _warn(self, stats: dict[str, Any]) -> None:
        if self.state.step < self.cfg.warn_after_frac * self.state.total_steps:
            return
        hits = []
        ar = stats.get("area_ratio_median")
        sr = stats.get("std_ratio_median")
        if ar is not None and ar > self.cfg.warn_area_ratio_hi:
            hits.append(f"area_ratio_median={ar:.3f} > {self.cfg.warn_area_ratio_hi}")
        if ar is not None and ar < self.cfg.warn_area_ratio_lo:
            hits.append(f"area_ratio_median={ar:.3f} < {self.cfg.warn_area_ratio_lo}")
        if sr is not None and sr < self.cfg.warn_std_ratio_lo:
            hits.append(f"std_ratio_median={sr:.3f} < {self.cfg.warn_std_ratio_lo}")
        if hits:
            rec = {"step": self.state.step, "hits": hits}
            self.state.warnings.append(rec)
            print(f"EARLY_WARNING {json.dumps(rec)}", flush=True)

    # -- loop ---------------------------------------------------------------
    def _rotate(self, name: str) -> None:
        """Move a previous run's log aside instead of appending to it.

        `steps.jsonl` / `eval.jsonl` were opened in append mode, so a re-run
        (e.g. after a blocker fix) left ONE file holding TWO runs with restarting
        step numbers.  The result reviewer reads only the delivery folder and has
        no way to see the seam -- `head` shows the dead run and `tail` the live
        one.  Rotate rather than delete: the project rule is to back up before
        clearing, because a false restart plus a delete loses the evidence too.
        """
        p = self.run_dir / name
        if p.exists() and p.stat().st_size:
            ts = time.strftime("%Y%m%dT%H%M%S")
            p.rename(self.run_dir / f"{name}.prev-{ts}")

    def train(self) -> AmortState:
        t0 = time.time()
        self._rotate("steps.jsonl")
        self._rotate("eval.jsonl")
        log = (self.run_dir / "steps.jsonl").open("w")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        # review N1: a first `permutation` here was immediately overwritten --
        # dead, but it consumed RNG state and so quietly changed the sample order
        # relative to what the seed implies.
        n_epochs = max(1, int(np.ceil(self.cfg.epochs)))
        order = np.concatenate([self.rng.permutation(len(self.indices))
                                for _ in range(n_epochs)])
        mb, accum, cursor = self.cfg.micro_batch, 0, 0
        try:
            while cursor + mb <= len(order):
                if self.state.step >= self.state.total_steps:
                    self.state.stopped_reason = (
                        "max_steps" if self.cfg.max_steps
                        and self.state.total_steps == int(self.cfg.max_steps)
                        else "total_steps")
                    break
                if (time.time() - t0) / 3600.0 > self.cfg.max_hours:
                    self.state.stopped_reason = "max_hours"
                    print(f"STOP max_hours={self.cfg.max_hours} reached at step "
                          f"{self.state.step}", flush=True)
                    break
                sel = [self.indices[int(order[cursor + k])] for k in range(mb)]
                cursor += mb
                samples = [self.dataset[i] for i in sel]
                # p=0.15 foreign (empty-mask control); the rest split
                # teacher/generated exactly as W01/W02 did, so the arms are
                # conditioned on the same text distribution they are deployed on.
                r_fake = self.rng.random(mb)
                r_ctx = self.rng.random(mb)
                if self.cfg.train_context == "mixed":
                    def _ctx(k: int) -> str:
                        return ("gt" if r_ctx[k] < self.cfg.teacher_fraction
                                else self.gen_mode)
                else:
                    # ceiling rows: every non-fake sample sees one fixed context
                    def _ctx(k: int) -> str:
                        return self.cfg.train_context
                modes = [self.fake_mode if r_fake[k] < self.weights.fake_prob
                         else _ctx(k) for k in range(mb)]
                inputs = self.builder.build(samples, modes)
                with self.autocast:
                    total, stats, _ = compute_micro_batch(
                        self.model, inputs, self.weights,
                        want_hi=self.want_hi, sdf_cache=self.sdf_cache)
                if not torch.isfinite(total):
                    # A single non-finite loss propagates into every weight and
                    # the arm silently finishes as a constant field with a
                    # plausible-looking board.  Skip the step, count it, and
                    # make it loud rather than discovering it in the metrics.
                    self.state.nan_steps += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    accum = 0
                    print(f"NON_FINITE_LOSS step={self.state.step} "
                          f"micro={self.state.micro_step} (skipped; total "
                          f"{self.state.nan_steps})", flush=True)
                    if self.state.nan_steps > 50:
                        self.state.stopped_reason = "too_many_nan"
                        break
                    continue
                if total.requires_grad:
                    (total / self.gas).backward()
                else:
                    # Possible only with a frozen base: a micro-batch routed
                    # entirely to a head the injector does not touch has no
                    # trainable tensor in its graph, so it contributes exactly
                    # zero gradient and `backward()` would raise.  Count it like
                    # any other micro-batch -- dropping it instead would shift
                    # the accumulation cadence and with it the LR schedule.
                    self.state.warnings.append(
                        {"step": self.state.step, "kind": "no_grad_microbatch"})
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

                row = {"step": self.state.step, "loss": float(total.detach()),
                       "lr": self.optimizer.param_groups[0]["lr"],
                       "grad_norm": gnorm,
                       "elapsed_s": round(time.time() - t0, 1), **stats}
                self.state.history.append(row)
                log.write(json.dumps(row) + "\n")
                self._warn(stats)
                if self.state.step % self.log_every == 0:
                    log.flush()
                    print(json.dumps(row), flush=True)
                if self.eval_fn and self.state.step % self.cfg.eval_steps == 0:
                    self._eval_and_record(log)
                if self.state.step % self.cfg.save_steps == 0:
                    self.save(f"step{self.state.step}")
        finally:
            log.close()
        self.save("final")
        if self.eval_fn:
            self._eval_and_record()
        return self.state

    def _eval_and_record(self, log=None) -> None:
        """Evaluation is reporting, so a failure must not kill the arm.

        A checkpoint whose eval raised simply has no record and is therefore not
        selectable -- the honest outcome.  The failure is written loudly rather
        than swallowed.
        """
        import traceback

        try:
            rec = self.eval_fn(self.state.step)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            self.state.eval_failures.append({"step": self.state.step,
                                             "error": f"{type(exc).__name__}: {exc}"})
            (self.run_dir / f"EVAL_FAILED_step{self.state.step}.json").write_text(
                json.dumps({"step": self.state.step, "traceback": tb}, indent=2))
            print(f"EVAL_FAILED step={self.state.step} {type(exc).__name__}: {exc}",
                  flush=True)
            return
        rec = {"step": self.state.step, **rec}
        self.state.checkpoints.append(rec)
        with (self.run_dir / "eval.jsonl").open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"EVAL {json.dumps(rec)}", flush=True)

    # -- checkpoints --------------------------------------------------------
    def save(self, tag: str) -> Path:
        p = self.run_dir / f"amort_{tag}.pt"
        torch.save({"arm": self.cfg.arm, "step": self.state.step,
                    "model": self.model.state_dict(),
                    "facts": self.model.facts(),
                    "loss_weights": self.weights.to_dict(),
                    "train_cfg": asdict(self.cfg)}, p)
        return p

    def best(self, key: str = "local_soft_iou_median") -> dict[str, Any] | None:
        """Gate-passing checkpoints only, then the criterion.  Never a val loss.

        Review U3: this used to fall back to the full checkpoint list when none
        passed (`pool = ok or self.state.checkpoints`), which contradicts the
        pre-registration ("all three gates must pass to be selectable") and this
        module's own docstring.  The honest outcome when nothing passes is **no
        selectable checkpoint** -- i.e. the arm did not clear its gates -- not
        "the best of a bad bunch" quietly promoted to the delivered board.
        """
        ok = [c for c in self.state.checkpoints
              if c.get("gate_pass") and c.get(key) is not None]
        if not ok:
            return None
        return max(ok, key=lambda c: c.get(key, float("-inf")))
