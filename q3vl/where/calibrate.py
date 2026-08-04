"""Protocol 4.4 / 10.2 -- the four Where-A basis-calibration arms.

| arm            | projector target                                   |
|----------------|----------------------------------------------------|
| `BA-0-Fixed`   | seeded orthogonal 1024->64, never trained          |
| `BA-1-Band`    | back-propagated from the `R-Band` oracle only      |
| `BA-2-CBand12` | back-propagated from the `R-CBand12` oracle only   |
| `BA-3-Joint`   | Band/CBand keep *independent* oracle latents, `B` is shared |

One optimisation step is a two-level (bilevel) update:

1. inner: with ``B`` frozen, fit the per-image oracle latent ``(w*, rho*)`` by
   multi-start L-BFGS in float64 on the detached ``phi_dir`` -- the latent is a
   free per-image variable, never an amortised prediction;
2. outer: hold the latent fixed and take one AdamW step on ``B`` through
   ``phi_dir``.

Step 2 is justified by the envelope theorem, and that argument is only valid
because ``CalibConfig.objective`` drives **both** levels (D2 as re-adjudicated on
2026-08-05).  With ``lambda*(B) = argmin_lambda f(B, lambda)`` and the same ``f``
outside, ``d/dB f(B, lambda*(B)) = partial f / partial B``; the implicit term
vanishes.  Had the two levels used different objectives ``f`` and ``g``, the
dropped term ``(dg/dlambda)(dlambda*/dB)`` would be O(1) and the gradient would
belong to no well-defined bilevel problem at all (REVIEW-impl-WhereA B-6).

A fit whose ``status`` is not ``"ok"`` never reaches step 2 and never enters a
ceiling statistic; it goes to the rejection report instead (protocol 10.2,
REVIEW-impl-WhereA B-1/B-2).

Instruction text is never read here (protocol 4.4: "the calibration target is
oracle mask expressiveness, the instruction is not used").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import torch

from .basis import mask_from_latent
from .config import (
    ARM_READOUTS, ARMS, GUIDED_PARAMS_PROVISIONAL, HEADLINE_WINNER_CONFIDENCE,
    JOINT_READOUT_WEIGHTS, CalibConfig,
)
from .oracle import FitResult, evaluate_latent, fit_latent, objective_value
from .phi import build_phi_dir
from .projector import BasisProjector

__all__ = ["WhereASample", "Calibrator", "make_scheduler", "arm_readouts",
           "percentiles"]


def percentiles(xs: Sequence[float]) -> dict[str, float]:
    """Fixed percentile convention (REVIEW-impl-WhereA N-11).

    Nearest-rank on a 0-indexed sorted array: ``q`` maps to
    ``idx = clip(ceil(q*k) - 1, 0, k-1)``, so p10/p50/p90 are symmetric and
    ``p10`` of 100 values is the 10th order statistic, ``p90`` the 90th.
    Protocol 5.6 has a p10 gate, so this has to be pinned, not incidental.
    """
    xs = sorted(float(x) for x in xs)
    if not xs:
        return {}
    k = len(xs)

    def q(p: float) -> float:
        return xs[min(k - 1, max(0, math.ceil(p * k) - 1))]

    # ``n`` travels with the percentiles on purpose: at k=2 the nearest-rank
    # p10 equals the median, which reads as "tight" unless the count is right
    # next to it (REVIEW-impl-WhereA N-23).
    return {"n": k, "mean": sum(xs) / k, "p10": q(0.10), "median": q(0.50),
            "p90": q(0.90), "min": xs[0], "max": xs[-1]}


def arm_readouts(arm: str) -> tuple[str, ...]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; protocol 4.4 defines {ARMS}")
    return ARM_READOUTS[arm]


@dataclass
class WhereASample:
    """One prepared training item.  ``F_pre`` is the frozen VLM's own output."""

    sample_id: str
    fpre: torch.Tensor       # (P, 1024)
    img_low: torch.Tensor    # (3, grid_h, grid_w) sRGB in [0,1]
    mask_low: torch.Tensor   # (P,) soft GT in [0,1]
    grid_h: int
    grid_w: int
    meta: dict[str, Any] = field(default_factory=dict)
    # the high-resolution branch: present for evaluation samples, so the
    # delivered-resolution ceiling is measured on the real path rather than
    # inferred from the low-res one (REVIEW-impl-WhereA B-4)
    mask_hi: torch.Tensor | None = None      # (out_h, out_w)
    guide_hi: torch.Tensor | None = None     # (1, 1, out_h, out_w) luma in [0,1]

    def __post_init__(self) -> None:
        p = self.grid_h * self.grid_w
        if self.fpre.shape[0] != p:
            raise ValueError(f"{self.sample_id}: fpre has {self.fpre.shape[0]} rows, grid says {p}")
        if self.mask_low.numel() != p:
            raise ValueError(f"{self.sample_id}: mask_low has {self.mask_low.numel()} points, grid says {p}")
        if (self.mask_hi is None) != (self.guide_hi is None):
            raise ValueError(
                f"{self.sample_id}: mask_hi and guide_hi must be supplied together; "
                "a guide without a target (or the reverse) cannot be evaluated"
            )
        if self.mask_hi is not None:
            if self.guide_hi.shape[-2:] != self.mask_hi.shape[-2:]:
                raise ValueError(
                    f"{self.sample_id}: guide {tuple(self.guide_hi.shape[-2:])} != "
                    f"mask_hi {tuple(self.mask_hi.shape[-2:])}"
                )

    @property
    def has_hi(self) -> bool:
        return self.mask_hi is not None and self.guide_hi is not None


def make_scheduler(optimizer, total_steps: int, warmup_ratio: float, kind: str):
    """Protocol 10.2: linear warmup over 3% of steps, then cosine to 0.

    ``total_steps`` must be the number of steps that will *actually* run, i.e.
    derived from the ``eligibility()``-filtered sample count.  Feeding it the
    unfiltered count stretches the warmup and leaves cosine unfinished, so the
    arms stop at different points on the schedule and stop being comparable
    (REVIEW-impl-WhereA B-3).
    """
    warmup = max(1, int(round(total_steps * warmup_ratio)))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        if kind != "cosine":
            return 1.0
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Calibrator:
    """Runs one arm.  Holds the only trainable tensor in Stage-Where-A."""

    def __init__(
        self,
        cfg: CalibConfig,
        projector: BasisProjector | None = None,
        total_steps: int | None = None,
        device: str = "cpu",
    ):
        # B-6 / N-18: the envelope-theorem argument for the fixed-latent
        # gradient holds only when both levels minimise the same functional.
        # Two module constants that happen to agree is not a guarantee --
        # `CalibConfig(objective=...)` and `FitConfig(objective=...)` are public
        # arguments that four scripts construct explicitly -- so it is checked
        # here, at the one place every path goes through.
        if cfg.objective != cfg.inner_fit.objective:
            raise ValueError(
                f"inner/outer objective mismatch: inner L-BFGS minimises "
                f"{cfg.inner_fit.objective!r} while the AdamW step on B uses "
                f"{cfg.objective!r}. With different objectives the dropped implicit "
                f"term (dg/dlatent)(dlatent*/dB) is O(1), so the gradient belongs to "
                f"no well-defined bilevel problem (REVIEW-impl-WhereA B-6)."
            )
        self.cfg = cfg
        self.readouts = arm_readouts(cfg.arm)
        self.device = device
        self.projector = (projector or BasisProjector(seed=cfg.seed)).to(device)
        self.trains_projector = bool(self.readouts)
        self.projector.requires_grad_(self.trains_projector)
        self.optimizer = None
        self.scheduler = None
        if self.trains_projector:
            self.optimizer = torch.optim.AdamW(
                self.projector.parameters(), lr=cfg.projector_lr,
                weight_decay=cfg.weight_decay,
            )
            if total_steps:
                self.scheduler = make_scheduler(
                    self.optimizer, total_steps, cfg.warmup_ratio, cfg.scheduler
                )
        self.step_count = 0
        self.total_steps = total_steps
        self.warmup_steps = (
            max(1, int(round(total_steps * cfg.warmup_ratio))) if total_steps else None
        )
        self.fit_report: list[dict[str, Any]] = []
        # running rejection tally over the calibration epoch (B-2)
        self.n_fits = 0
        self.n_rejected = 0
        self.reject_reasons: dict[str, int] = {}
        self.flag_counts: dict[str, int] = {}

    # -- forward ------------------------------------------------------------
    def phi_for(self, sample: WhereASample) -> Any:
        sem = self.projector(sample.fpre.to(self.device))
        return build_phi_dir(sem, sample.img_low.to(self.device),
                             sample.grid_h, sample.grid_w, self.cfg.phi)

    def fit_sample(
        self, sample: WhereASample, phi_dir: torch.Tensor, readout: str,
        fit_cfg=None, seed_offset: int = 0,
    ) -> FitResult:
        cfg = fit_cfg or self.cfg.inner_fit
        if cfg.objective != self.cfg.objective:
            raise ValueError(
                f"fit_cfg.objective={cfg.objective!r} != calibration objective "
                f"{self.cfg.objective!r}; the two levels must minimise the same "
                f"functional (REVIEW-impl-WhereA B-6/N-18)"
            )
        cfg = type(cfg)(**{**cfg.__dict__, "seed": cfg.seed + seed_offset})
        return fit_latent(
            phi_dir.detach().double(), sample.mask_low.to(self.device).double(),
            readout, cfg,
        )

    def freeze_projector(self) -> None:
        """Make "B is never touched" structural: drop the optimiser/scheduler and
        clear ``requires_grad``.  Used by the oracle-latent job, which loads an
        already-calibrated ``B`` (REVIEW-impl-WhereA N-24)."""
        self.trains_projector = False
        self.projector.requires_grad_(False)
        self.optimizer = None
        self.scheduler = None

    def _tally(self, fit: FitResult) -> None:
        self.n_fits += 1
        if not fit.usable:
            self.n_rejected += 1
            reason = fit.reject_reason or "unknown"
            self.reject_reasons[reason] = self.reject_reasons.get(reason, 0) + 1
        for f in fit.flags:
            self.flag_counts[f] = self.flag_counts.get(f, 0) + 1

    # -- one optimisation step ---------------------------------------------
    def step(
        self,
        batch: Sequence[WhereASample],
        record_fits: bool = False,
        sample_every: int = 0,
    ) -> dict[str, Any]:
        """One AdamW step on ``B``.

        Rejected fits (including ``all_starts_failed``, whose latent is ``None``)
        contribute **no** loss term and **no** gradient; they are returned in
        ``out["fit_rows"]`` so the driver can append them to
        ``fit_rejections.jsonl`` (protocol 10.2 / B-1 / B-2).  ``sample_every``
        additionally emits one accepted row every N steps, so the report has a
        healthy baseline to compare the failures against.
        """
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        totals: dict[str, float] = {r: 0.0 for r in self.readouts}
        counts: dict[str, int] = {r: 0 for r in self.readouts}
        fit_losses: dict[str, list[float]] = {r: [] for r in self.readouts}
        rejected = 0
        rows: list[dict[str, Any]] = []
        loss_sum = torch.zeros((), device=self.device)
        sampled = bool(sample_every) and (self.step_count % sample_every == 0)

        for j, sample in enumerate(batch):
            parts = self.phi_for(sample)
            target = sample.mask_low.to(self.device).to(parts.phi_dir.dtype)
            for r in self.readouts:
                fit = self.fit_sample(sample, parts.phi_dir, r, seed_offset=self.step_count * 1000 + j)
                self._tally(fit)
                if record_fits:
                    row = fit.to_dict()
                    row["sample_id"] = sample.sample_id
                    row["step"] = self.step_count
                    self.fit_report.append(row)
                if not fit.usable:
                    # never train B on a fit the protocol calls failed
                    rejected += 1
                    rows.append(fit.rejection_row(
                        sample_id=sample.sample_id, step=self.step_count,
                        build=sample.meta.get("build"),
                        winner_confidence=sample.meta.get("winner_confidence"),
                    ))
                    continue
                if sampled:
                    rows.append(fit.rejection_row(
                        sample_id=sample.sample_id, step=self.step_count,
                        build=sample.meta.get("build"),
                        winner_confidence=sample.meta.get("winner_confidence"),
                        sampled=True,
                    ))
                latent = fit.latent.to(parts.phi_dir.dtype).detach()
                m, _ = mask_from_latent(parts.phi_dir, latent)
                loss = objective_value(self.cfg.objective, m, target)
                w = JOINT_READOUT_WEIGHTS[r] if len(self.readouts) > 1 else 1.0
                loss_sum = loss_sum + w * loss
                totals[r] += float(loss.detach())
                counts[r] += 1
                fit_losses[r].append(fit.loss)

        n_used = max(1, sum(counts.values()) // max(1, len(self.readouts) or 1))
        loss_mean = loss_sum / n_used
        grad_norm = None
        if self.trains_projector and loss_mean.requires_grad:
            loss_mean.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                self.projector.parameters(), self.cfg.max_grad_norm))
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
        self.step_count += 1
        return {
            "step": self.step_count,
            "loss": float(loss_mean.detach()),
            "loss_per_readout": {
                r: (totals[r] / counts[r] if counts[r] else float("nan")) for r in self.readouts
            },
            "n_used_per_readout": dict(counts),
            "fit_loss_per_readout": {
                r: (sum(v) / len(v) if v else float("nan")) for r, v in fit_losses.items()
            },
            "n_rejected_fits": rejected,
            "fit_rows": rows,
            "grad_norm": grad_norm,
            "lr": (self.optimizer.param_groups[0]["lr"] if self.optimizer else 0.0),
            "batch_size": len(batch),
        }

    # -- evaluation ---------------------------------------------------------
    @torch.no_grad()
    def _eval_phi(self, sample: WhereASample):
        return self.phi_for(sample)

    def evaluate(
        self, samples: Iterable[WhereASample], readouts: Sequence[str] | None = None,
        fit_cfg=None, record_fits: bool = True, headline_confidence=HEADLINE_WINNER_CONFIDENCE,
    ) -> dict[str, Any]:
        """Refit oracle latents with ``B`` frozen (protocol 4.4 final step) and
        report the oracle ceiling per readout.

        Three rules the review made explicit:

        * only ``ok`` fits enter an aggregate -- a rejected fit's metrics would
          drag the ceiling down and thereby *loosen* the protocol 5.6 gate
          "soft-IoU >= 85% of the per-image oracle" (B-1);
        * when a sample carries ``mask_hi``/``guide_hi`` the high-resolution
          number is computed on the real path and reported next to the low-res
          one, together with the s-domain report (B-4);
        * the headline population is ``winner_confidence in headline_confidence``
          (D1: ``normal``); every other stratum is reported separately, never
          merged into the headline.
        """
        readouts = readouts or (self.readouts or ("band", "cband12"))
        per: dict[str, list[dict[str, Any]]] = {r: [] for r in readouts}
        rows: list[dict[str, Any]] = []
        n_rejected = {r: 0 for r in readouts}
        rejected_reasons: dict[str, dict[str, int]] = {r: {} for r in readouts}
        n = n_hi = 0
        for j, sample in enumerate(samples):
            parts = self._eval_phi(sample)
            target = sample.mask_low.to(self.device).double()
            for r in readouts:
                fit = self.fit_sample(sample, parts.phi_dir, r,
                                      fit_cfg=fit_cfg or self.cfg.inner_fit, seed_offset=j)
                row = fit.to_dict() if record_fits else {"readout": r, "status": fit.status}
                row["sample_id"] = sample.sample_id
                row["meta"] = sample.meta
                if record_fits:
                    row["phi_diag"] = parts.diag
                if not fit.usable:
                    n_rejected[r] += 1
                    reason = fit.reject_reason or "unknown"
                    rejected_reasons[r][reason] = rejected_reasons[r].get(reason, 0) + 1
                    rows.append(row)
                    continue
                with torch.no_grad():
                    ev = evaluate_latent(
                        parts.phi_dir.double(), fit.latent, target,
                        sample.grid_h, sample.grid_w,
                        guide_hi=(sample.guide_hi.to(self.device).double()
                                  if sample.has_hi else None),
                        target_hi=(sample.mask_hi.to(self.device).double()
                                   if sample.has_hi else None),
                        up_cfg=self.cfg.upsample,
                    )
                entry: dict[str, Any] = {
                    "low": ev["low"], "hi": ev.get("hi"),
                    "s_domain": ev.get("s_domain"),
                    "winner_confidence": sample.meta.get("winner_confidence"),
                }
                per[r].append(entry)
                if record_fits:
                    row["eval"] = ev
                rows.append(row)
            n += 1
            n_hi += int(sample.has_hi)

        def summarize(entries, res: str) -> dict[str, Any]:
            vals = [e[res] for e in entries if e.get(res) is not None]
            out: dict[str, Any] = {"n": len(vals)}
            for key in ("soft_iou_minmax", "soft_iou_prod", "mae", "mse"):
                out[key] = percentiles([v[key] for v in vals])
            return out

        def strata(entries, res: str) -> dict[str, Any]:
            groups: dict[str, list] = {}
            for e in entries:
                groups.setdefault(str(e.get("winner_confidence")), []).append(e)
            return {k: summarize(v, res) for k, v in groups.items()}

        result = {
            "arm": self.cfg.arm,
            "n_samples": n,
            "n_with_hi_res": n_hi,
            "objective": self.cfg.objective,
            "headline_winner_confidence": list(headline_confidence),
            "projector": self.projector.facts(),
            "upsample": {"radius_low": self.cfg.upsample.radius_low,
                         "eps": self.cfg.upsample.eps,
                         "clamp_domain": self.cfg.upsample.clamp_domain,
                         "domain": list(self.cfg.upsample.domain),
                         "provisional": GUIDED_PARAMS_PROVISIONAL},
            "per_readout": {},
            "rows": rows,
        }
        for r in readouts:
            head = [e for e in per[r]
                    if e.get("winner_confidence") in headline_confidence]
            dom = [e["s_domain"] for e in per[r] if e.get("s_domain")]
            result["per_readout"][r] = {
                # headline: ok fits, headline confidence stratum
                "headline_low": summarize(head, "low"),
                "headline_hi": summarize(head, "hi"),
                # everything that fitted, for reference
                "all_ok_low": summarize(per[r], "low"),
                "all_ok_hi": summarize(per[r], "hi"),
                "by_winner_confidence_low_res": strata(per[r], "low"),
                "by_winner_confidence_hi_res": strata(per[r], "hi"),
                "n_ok": len(per[r]),
                "n_rejected": n_rejected[r],
                "reject_reasons": rejected_reasons[r],
                "fit_success_rate": 1.0 - n_rejected[r] / max(1, n),
                "s_domain": {
                    # per-sample distribution, so the pre-registered gate can be
                    # applied to the median rather than to an outlier (N-20)
                    "frac_out_of_domain": percentiles(
                        [d["frac_out_of_domain"] for d in dom]),
                    "max_frac_out_of_domain": max((d["frac_out_of_domain"] for d in dom), default=None),
                    "mean_frac_out_of_domain": (
                        sum(d["frac_out_of_domain"] for d in dom) / len(dom) if dom else None),
                    "raw_min": min((d["raw_min"] for d in dom), default=None),
                    "raw_max": max((d["raw_max"] for d in dom), default=None),
                    "clamped": bool(self.cfg.upsample.clamp_domain),
                },
            }
        return result

    # -- reporting ----------------------------------------------------------
    def rejection_summary(self) -> dict[str, Any]:
        """Epoch-level tally for ``fit_rejections.jsonl``'s header (B-2)."""
        return {
            "arm": self.cfg.arm,
            "steps": self.step_count,
            "n_fits": self.n_fits,
            "n_rejected": self.n_rejected,
            "reject_rate": self.n_rejected / max(1, self.n_fits),
            "reject_reasons": dict(self.reject_reasons),
            "flag_counts": dict(self.flag_counts),
        }

    # -- checkpointing ------------------------------------------------------
    def state(self) -> dict[str, Any]:
        return {
            "arm": self.cfg.arm,
            "step": self.step_count,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "projector": {k: v.detach().cpu() for k, v in self.projector.state_dict().items()},
            "projector_facts": self.projector.facts(),
            "rejections": self.rejection_summary(),
        }
