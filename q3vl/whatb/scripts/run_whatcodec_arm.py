#!/usr/bin/env python
"""EPR-031 C1 -- **O0**: the direct carrier table (no generator, no VLM).

Spec: ``experiments/prs/EPR-031_whatcodec-evidence-code-slotdecoder/PROPOSAL.md``
§5 (the O0 row) and §7 (the C1 gate); the carrier's own mathematics is EPR-028
R1 and lives in :mod:`q3vl.whatb.arms.g4d`, which this runner only calls.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/bc/VeraRetouch \\
    /home/bc/envs/q3vl_sft/bin/python -m q3vl.whatb.scripts.run_whatcodec_arm \\
        --level O0 --stage S2 --carrier A3 --out /home/bc/data/runs/epr031_c1_s2_a3

What O0 is
----------
``Theta in R^{n_lut x D_theta}``: one independent learnable carrier parameter set
per ``lut_id``, indexed by ``lut_id``.  There is no conditioner of any kind, so
``E_O0`` is the **carrier floor** the other three levels of §5 are differenced
against.  It is therefore an oracle: ``metrics.json`` carries
``published = false`` and ``oracle_reference = true``, and the board is *not* a
headline board -- no images, no ``z`` cache, no predicted field is opened,
required, or reported missing.

What it reports
---------------
``E_O0`` = the ``17^3`` grid dE00 mean against the data law's own target
``(1-s) x + s L_l(x)``, in five ``s`` columns (0 / 0.25 / 0.5 / 0.75 / 1), plus
EPR-028 R1 §10's stability telemetry on every ``steps.jsonl`` row.

采信 discipline (§8, and EPR-030's own hole): **every** quick eval re-checks
``L_rec`` for NaN / Inf.  A NaN prediction scores dE00 ``0.0``, not NaN, so a
guard that only looks once publishes a perfect-looking floor.  A non-finite
value stops the run with rc 3 and writes ``void_reason`` into ``metrics.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from q3vl.whatb import queries, splits
from q3vl.whatb.arms import g4d
from q3vl.whatb.codec import tables as T
from q3vl.whatb.codec.lutcode import train_lut_ids
from q3vl.whatb.colorimetry import delta_e00_srgb
from q3vl.whatb.guards import record_step_witness
from q3vl.whatb.lutdata import BANK_DIR, LutBank, mix_alpha

__all__ = ["build_arg_parser", "main", "RC_NONFINITE", "LEVEL_CHOICES"]

#: exit code of the NaN re-check (distinct from the degeneracy guard's 2)
RC_NONFINITE: int = 3

#: §5's four levels.  C1 implements the first; O1/O2/O3 land in C2/C3/C5.
LEVEL_CHOICES: tuple[str, ...] = ("O0",)

#: the base lr this campaign froze (EPR-030 §3.1: 1.6e-2 died seven of seven;
#: {1e-3, 4e-3, 3e-4} died zero of seven)
BASE_LR: float = 1e-3


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="run_whatcodec_arm",
        description="EPR-031 C1: O0 direct carrier table (oracle floor)")
    ap.add_argument("--level", choices=LEVEL_CHOICES, default="O0",
                    help="§5's capacity level.  C1 implements O0 only")
    ap.add_argument("--stage", choices=T.STAGE_CHOICES, default="S1",
                    help="§7-C1 ladder: S1 1x2048x4x2000 / S2 32x512x4x4000 / "
                         "S3 (whole pool) 256x2048x4x18760")
    ap.add_argument("--carrier", choices=T.CARRIER_CHOICES, default="A1",
                    help="EPR-028 R1 §3: A1 = 3D explicit gate (22/prim), "
                         "A3 = 4D joint (27/prim).  N is --n-gauss")
    ap.add_argument("--n-gauss", type=int, default=48)
    ap.add_argument("--clamp", choices=("two", "one"), default="two")
    ap.add_argument("--marg-norm", choices=g4d.MARG_NORM_CHOICES, default="peak")
    ap.add_argument("--lut-id", default=None,
                    help="the single LUT of --stage S1; default = the first "
                         "train lut_id in sorted order")
    ap.add_argument("--pool-seed", type=int, default=20260810,
                    help="seed of S2's fixed 32-LUT draw from the train pool")
    # -- loss (§5 / EPR-028 R1 §4.4) ---------------------------------------- #
    ap.add_argument("--loss-level", type=int, default=1, choices=(1,),
                    help="pure L1: lambda_hc = lambda_sparse = 0 (the EPR-030 "
                         "caliber; O0 is an oracle floor, not a loss ablation)")
    ap.add_argument("--lam-line", type=float, default=g4d.LAMBDA_LINE,
                    help="R_line weight, pre-registered 0.1; 0 switches the term "
                         "off and it is then not computed at all")
    # -- optimiser (§4) ------------------------------------------------------ #
    ap.add_argument("--base-lr", "--lr", dest="base_lr", type=float,
                    default=BASE_LR)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=None,
                    help="override the stage's pre-registered step count "
                         "(smoke only; the override is recorded)")
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--mining-epochs", type=int, default=40,
                    help="the GLUT mining ramp (App A.1: epoch 5 -> 20, 10 %% -> "
                         "40 %%) is expressed over this many epoch-equivalents "
                         "of the stage's own horizon")
    # -- evaluation ---------------------------------------------------------- #
    ap.add_argument("--eval-grid", type=int, default=17,
                    help="side of the sRGB lattice E_O0 is measured on")
    ap.add_argument("--quick-eval-every", type=int, default=500)
    ap.add_argument("--quick-eval-luts", type=int, default=32,
                    help="LUTs a QUICK eval scores (0 = the whole pool); the "
                         "final eval always scores the whole pool")
    ap.add_argument("--eval-chunk", type=int, default=4,
                    help="LUTs per evaluation forward")
    ap.add_argument("--log-every", type=int, default=1)
    # -- plumbing ------------------------------------------------------------ #
    ap.add_argument("--lut-bank", default=str(BANK_DIR))
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the stage, write run_setup.json, then stop.  "
                         "Needs neither a z cache nor images nor a pred field")
    ap.add_argument("--no-train", action="store_true",
                    help="skip training and evaluate the initialisation")
    return ap


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    tmp.replace(path)


def describe(values: Sequence[float]) -> dict[str, Any]:
    """mean / median / p95 / max / n -- the only reduction this runner reports."""
    v = sorted(float(x) for x in values)
    n = len(v)
    if not n:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {"n": n, "mean": sum(v) / n, "median": v[n // 2],
            "p95": v[min(n - 1, int(round(0.95 * (n - 1))))], "max": v[-1]}


def source_freeze() -> dict[str, str]:
    """sha256 of the files this run's behaviour depends on (submit-time freeze)."""
    root = Path(__file__).resolve().parents[1]
    names = ["codec/lutcode.py", "codec/tables.py", "arms/g4d.py", "glut.py",
             "lutdata.py", "queries.py", "colorimetry.py", "splits.py",
             "scripts/run_whatcodec_arm.py"]
    out: dict[str, str] = {}
    for name in names:
        p = root / name
        if not p.exists():
            continue
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        out[name] = h.hexdigest()
    return out


class NonFiniteLoss(SystemExit):
    """``L_rec`` went non-finite; the run stops with :data:`RC_NONFINITE`."""

    def __init__(self, void_reason: Mapping[str, Any]) -> None:
        super().__init__(RC_NONFINITE)
        self.void_reason = dict(void_reason)


def assert_l_rec_finite(value: Any, *, step: int, where: str,
                        run_dir: Path | None = None,
                        extra: Mapping[str, Any] | None = None) -> float | None:
    """Re-check ``L_rec`` at EVERY quick eval, not only the first one.

    A NaN prediction scores dE00 ``0.0`` (the Lab conversion and the hue
    ``atan2`` both swallow it), so a floor built on a NaN checkpoint reads as
    perfect.  A non-finite value writes ``void_reason`` into ``metrics.json`` and
    raises :class:`NonFiniteLoss` -- a non-zero rc, never a warning.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = float("nan")
    if math.isfinite(v):
        return v
    reason = {"void_reason": "L_rec is not finite",
              "L_rec": None if math.isnan(v) else v, "raw": repr(value),
              "step": int(step), "where": where, "rc": RC_NONFINITE,
              "authority": ("EPR-031 §8 / EPR-028 R1 §10: a NaN prediction "
                            "scores dE00 = 0.0, so an unchecked board publishes "
                            "a perfect floor"),
              **dict(extra or {})}
    if run_dir is not None:
        write_json(Path(run_dir) / "metrics.json",
                   {"arm": T.ARM, "level": T.LEVEL, "published": False,
                    "oracle_reference": True, "quick": False, "void": True,
                    **reason})
    print(f"[EPR-031] NON-FINITE L_rec at {where} (step {step}): stopping with "
          f"rc={RC_NONFINITE}", file=sys.stderr, flush=True)
    raise NonFiniteLoss(reason)


# --------------------------------------------------------------------------- #
# pools
# --------------------------------------------------------------------------- #
def resolve_pool(args, all_ids: Sequence[str], spec: T.StageSpec) -> list[str]:
    """The ``lut_id`` this stage owns a table row for."""
    if spec.stage == "S1":
        lid = args.lut_id or all_ids[0]
        if lid not in set(all_ids):
            raise SystemExit(
                f"--lut-id {lid!r} is not a train lut_id ({len(all_ids)} in the "
                "pool); S1 overfits one LUT of the training pool")
        return [str(lid)]
    if spec.n_lut_pool >= len(all_ids):
        return list(all_ids)
    rng = random.Random(int(args.pool_seed))
    return sorted(rng.sample(list(all_ids), int(spec.n_lut_pool)))


def lut_values(bank: LutBank, lut_ids: Sequence[str], x: torch.Tensor
               ) -> torch.Tensor:
    """``L_l(x)`` per batch row -- the bank's own operator, never a re-write."""
    return torch.stack([bank.apply(x[i], lid) for i, lid in enumerate(lut_ids)])


# --------------------------------------------------------------------------- #
# evaluation: E_O0
# --------------------------------------------------------------------------- #
@torch.no_grad()
def grid_metrics(carrier: g4d.Glut4DCarrier, table: T.DirectCarrierTable,
                 bank: LutBank, lut_ids: Sequence[str], device: torch.device, *,
                 grid_n: int = 17, chunk: int = 4) -> dict[str, Any]:
    """``E_O0``: the ``grid_n^3`` dE00 mean per ``s``, against the data law.

    No image, no ``z``, no predicted field is touched -- the whole read-out is
    ``luts.npz`` plus the table.
    """
    grid = queries.uniform_grid(int(grid_n), device=device)          # (P, 3)
    per_s: dict[str, list[float]] = {key: [] for _, key in g4d.S_AXIS_GRID}
    every: list[float] = []
    ids = list(lut_ids)
    for start in range(0, len(ids), max(1, int(chunk))):
        block = ids[start:start + max(1, int(chunk))]
        params = table.params_for(block)
        targets = torch.stack([bank.apply(grid, lid) for lid in block])  # (b,P,3)
        for s_val, key in g4d.S_AXIS_GRID:
            f_s = carrier(grid, float(s_val), params)                # (b, P, 3)
            tgt = mix_alpha(grid.unsqueeze(0).expand_as(targets), targets,
                            torch.full_like(targets[..., :1], float(s_val)))
            de = delta_e00_srgb(f_s, tgt).mean(dim=-1)               # (b,)
            vals = [float(v) for v in de]
            per_s[key].extend(vals)
            every.extend(vals)
    out: dict[str, Any] = {key: describe(vals) for key, vals in per_s.items()}
    out["grid_de00_mean"] = describe(every)
    out["n_lut"] = len(ids)
    out["grid"] = f"{grid_n}^3 uniform sRGB lattice x 5 s values"
    out["s_values"] = [s for s, _ in g4d.S_AXIS_GRID]
    return out


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train(args, carrier: g4d.Glut4DCarrier, table: T.DirectCarrierTable,
          bank: LutBank, pool: Sequence[str], spec: T.StageSpec, run_dir: Path,
          device: torch.device, eval_hook) -> dict[str, Any]:
    """The C1 loop: paired anchors, colour-group mining, Adam + cosine."""
    total_steps = int(args.steps) if args.steps else spec.total_steps
    lam_line = float(args.lam_line)
    n_pairs_s = spec.colors_per_step

    opt = torch.optim.Adam(table.param_groups(args.base_lr),
                           weight_decay=float(args.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, total_steps))
    sampler = queries.QuerySampler(seed=int(args.seed), q=spec.q_colors)
    pool_gen = random.Random(int(args.seed) + 17)

    steps_path = run_dir / "steps.jsonl"
    steps_path.parent.mkdir(parents=True, exist_ok=True)
    fh = steps_path.open("a", encoding="utf-8")
    anchor_facts: dict[str, Any] = {}
    t0 = time.time()
    try:
        for step in range(total_steps):
            batch_ids = (list(pool) if spec.b_luts >= len(pool)
                         else pool_gen.sample(list(pool), spec.b_luts))
            # the mining ramp is GLUT App A.1's, expressed over this stage's own
            # horizon: epoch-equivalent = mining_epochs * step / total_steps
            epoch = float(args.mining_epochs) * step / max(1, total_steps)
            ratio = queries.mining_ratio(epoch) if spec.mining else 0.0

            params = table.params_for(batch_ids)
            x0, s0 = T.paired_anchor_batch(sampler, spec.b_luts, spec.q_colors,
                                           device=device)
            y0 = g4d.target_4d(x0, s0, lut_values(bank, batch_ids, x0))
            if ratio > 0:
                x1, s1 = T.paired_anchor_batch(sampler, spec.b_luts,
                                               spec.q_colors, device=device)
                y1 = g4d.target_4d(x1, s1, lut_values(bank, batch_ids, x1))
                with torch.no_grad():
                    probe = carrier(x0, s0, params.detach())
                    err = (probe - y0).abs().mean(-1)
                keep = T.select_hard_colour_groups(err, ratio, spec.b_luts,
                                                   spec.q_colors)
                x = torch.where(keep.unsqueeze(-1), x0, x1)
                s = torch.where(keep, s0, s1)
                y = torch.where(keep.unsqueeze(-1), y0, y1)
            else:
                x, s, y = x0, s0, y0
            if step == 0:
                anchor_facts = T.assert_anchor_structure(s, spec.b_luts,
                                                         spec.q_colors)

            y_hat, aux = carrier(x, s, params, return_aux=True)
            line = (T.r_line_from_anchors(y_hat, s, spec.b_luts, spec.q_colors)
                    if lam_line else None)
            loss, cols = g4d.total_loss(y_hat, y, aux.opacity, lam_hc=0.0,
                                        lam_sparse=0.0, lam_line=lam_line,
                                        line=line)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # clip_grad_norm_ returns the PRE-clip total norm: R1 §10's ``gnorm``
            gnorm = torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"]],
                float(args.max_grad_norm) if args.max_grad_norm
                else float("inf"))
            if not torch.isfinite(gnorm):
                assert_l_rec_finite(float("nan"), step=step,
                                    where=f"step{step} (gnorm)",
                                    run_dir=run_dir,
                                    extra={"gnorm": float(gnorm)})
            opt.step()
            sched.step()

            row = {"step": step, "epoch": epoch, **cols,
                   "n_colors": n_pairs_s,
                   "n_colors_distinct": spec.distinct_colors_per_step,
                   "n_luts_in_batch": len(set(batch_ids)),
                   "mining_ratio": float(ratio),
                   "lr": float(opt.param_groups[0]["lr"]),
                   **T.telemetry_row(
                       aux, weight_underflow_tau=carrier.cfg.weight_underflow_tau),
                   "gnorm": float(gnorm)}
            if int(row["n_pairs_s"]) != n_pairs_s:
                raise AssertionError(
                    f"n_pairs_s is {row['n_pairs_s']} and the resolved sampler "
                    f"structure is {spec.b_luts} x {spec.q_colors} x "
                    f"{T.S_ANCHORS} = {n_pairs_s}; the column exists to pin "
                    "exactly this")
            if step == 0:
                record_step_witness(row)
                g4d.assert_step_row(steps_row=row, loss_level=int(args.loss_level),
                                    mode=carrier.cfg.mode,
                                    r_line=bool(lam_line))
            if args.log_every and step % int(args.log_every) == 0:
                fh.write(json.dumps(row) + "\n")
                fh.flush()

            if args.quick_eval_every and (step + 1) % int(args.quick_eval_every) == 0:
                eval_hook(step + 1, cols)
    finally:
        fh.close()
    return {"total_steps": total_steps,
            "steps_override": args.steps is not None,
            "loss_level": int(args.loss_level), "lam_line": lam_line,
            "n_pairs_s": n_pairs_s,
            "batch_structure": spec.as_dict(),
            "anchor_assertion": anchor_facts,
            "optimizer": {"name": "Adam", "base_lr": float(args.base_lr),
                          "schedule": "cosine, no warmup",
                          "weight_decay": float(args.weight_decay),
                          "grad_clip": float(args.max_grad_norm)},
            "wall_time_s": time.time() - t0,
            "sampler": sampler.facts()}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.level != "O0":                                # pragma: no cover
        raise SystemExit(f"--level {args.level} is not implemented by C1")
    torch.manual_seed(int(args.seed))
    run_dir = Path(args.out)
    device = torch.device(args.device)

    all_ids = train_lut_ids(args.train_split)
    spec = T.resolve_stage(args.stage, n_train_luts=len(all_ids))
    pool = resolve_pool(args, all_ids, spec)
    cfg = g4d.G4DConfig(mode=args.carrier, n_gauss=int(args.n_gauss),
                        clamp=args.clamp, marg_norm=args.marg_norm,
                        init_seed=int(args.seed))
    theta_dim = g4d.n_params_g4d(cfg.n_gauss, cfg.mode)

    setup: dict[str, Any] = {
        "arm": T.ARM, "level": T.LEVEL, "gate": "C1",
        "published": False, "oracle_reference": True,
        "conditioner": "none (direct table; no generator, no VLM, no z cache)",
        "stage": T.stage_facts(spec, n_train_luts=len(all_ids)),
        "carrier": {**cfg.as_dict(), "theta_dim_per_lut": theta_dim,
                    "authority": "EPR-028 R1 §3/§4 via q3vl/whatb/arms/g4d.py"},
        "table": {"n_lut": len(pool), "theta_dim": theta_dim,
                  "n_params_total": len(pool) * theta_dim,
                  "lut_id_head": pool[:8]},
        "data": {"train_split": args.train_split,
                 "n_train_lut_id_measured": len(all_ids),
                 "lut_bank": str(args.lut_bank),
                 "needs_zcache": False, "needs_images": False,
                 "needs_pred_field": False},
        "optimizer": {"name": "Adam", "base_lr": float(args.base_lr),
                      "schedule": "cosine, no warmup",
                      "weight_decay": float(args.weight_decay),
                      "grad_clip": float(args.max_grad_norm),
                      "loss_level": int(args.loss_level),
                      "lam_line": float(args.lam_line)},
        "loss": g4d.loss_preregistration(
            type("_A", (), {"glut4d_mode": cfg.mode, "w_img": 0.0, "alpha_s": 0.0,
                            "lam_line": float(args.lam_line),
                            "loss_level": int(args.loss_level),
                            "lam_hc": 0.0, "lam_sparse": 0.0})()),
        "eval": {"metric": "E_O0 = grid dE00 mean vs (1-s) x + s L_l(x)",
                 "grid": f"{args.eval_grid}^3",
                 "s_values": [s for s, _ in g4d.S_AXIS_GRID],
                 "quick_eval_every": int(args.quick_eval_every),
                 "quick_eval_luts": int(args.quick_eval_luts)},
        "seed": int(args.seed), "device": str(device),
        "command": " ".join([sys.executable, "-m",
                             "q3vl.whatb.scripts.run_whatcodec_arm",
                             *(argv if argv is not None else sys.argv[1:])]),
        "source_sha256": source_freeze(),
    }
    if args.dry_run:
        write_json(run_dir / "run_setup.json", {**setup, "dry_run": True})
        print(json.dumps({"dry_run": True, "stage": setup["stage"],
                          "table": setup["table"],
                          "carrier_mode": cfg.mode,
                          "theta_dim_per_lut": theta_dim,
                          "out": str(run_dir)}, indent=2, ensure_ascii=False),
              flush=True)
        return 0

    bank = LutBank(args.lut_bank, cache_size=min(256, max(8, len(pool))))
    carrier = g4d.Glut4DCarrier(cfg).to(device)
    table = T.DirectCarrierTable(pool, cfg).to(device)
    setup["table"].update(table.config())
    write_json(run_dir / "run_setup.json", setup)

    quick_ids = (pool if not args.quick_eval_luts
                 else pool[:int(args.quick_eval_luts)])
    history: list[dict[str, Any]] = []

    def eval_hook(step: int, loss_cols: Mapping[str, Any] | None = None) -> None:
        # 采信 discipline: the TRAINING L_rec is re-checked at every quick eval,
        # before its dE00 is allowed to be believed.
        assert_l_rec_finite((loss_cols or {}).get("L_rec"), step=step,
                            where=f"quick_eval@step{step} (train L_rec)",
                            run_dir=run_dir)
        grid = grid_metrics(carrier, table, bank, quick_ids, device,
                            grid_n=int(args.eval_grid), chunk=int(args.eval_chunk))
        score = (grid.get("grid_de00_mean") or {}).get("mean")
        assert_l_rec_finite(score, step=step,
                            where=f"quick_eval@step{step} (grid_de00_mean)",
                            run_dir=run_dir)
        rec = {"step": int(step), "grid_de00_mean": score,
               "n_lut": grid["n_lut"],
               **{k: (grid[k] or {}).get("mean") for _, k in g4d.S_AXIS_GRID}}
        history.append(rec)
        write_json(run_dir / "quick_evals.json", history)
        print(f"[EPR-031] step {step}: grid_de00_mean = {score}", flush=True)

    train_facts: dict[str, Any] = {"skipped": True}
    if not args.no_train:
        train_facts = train(args, carrier, table, bank, pool, spec, run_dir,
                            device, eval_hook)

    grid = grid_metrics(carrier, table, bank, pool, device,
                        grid_n=int(args.eval_grid), chunk=int(args.eval_chunk))
    assert_l_rec_finite((grid.get("grid_de00_mean") or {}).get("mean"),
                        step=int(train_facts.get("total_steps", 0)),
                        where="final (grid_de00_mean)", run_dir=run_dir)
    metrics = {
        "arm": T.ARM, "level": T.LEVEL, "gate": "C1",
        # §5: O0 sees the GT lut_id; its number is a carrier floor and may not be
        # read next to any arm's headline.
        "published": False, "oracle_reference": True,
        "headline_board": False,
        "E_O0": grid,
        "quick_evals": history,
        "stage": setup["stage"], "carrier": setup["carrier"],
        "table": table.config(), "train": train_facts,
        "seed": int(args.seed),
        "source_sha256": setup["source_sha256"],
        "criteria_note": ("O0 is not a headline board: no images, no z cache and "
                          "no predicted field are read, so the headline / five "
                          "trivial baselines / three negative controls are not "
                          "computed and are not reported as missing"),
    }
    write_json(run_dir / "metrics.json", metrics)
    print(json.dumps({"E_O0": {k: (grid[k] or {}).get("mean")
                               for _, k in g4d.S_AXIS_GRID},
                      "grid_de00_mean": (grid["grid_de00_mean"] or {}).get("mean"),
                      "n_lut": grid["n_lut"],
                      "published": False, "oracle_reference": True},
                     indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
