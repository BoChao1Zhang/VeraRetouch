from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .action import decode_raw_action, make_raw_parameter, mask_raw_for_config
from .metrics import metric_summary
from .m1_2_common import (
    A2_THRESHOLDS,
    M12DataConfig,
    dataset_batches,
    load_clean_tier_a,
    load_lpips_model,
    now_stamp,
    select_device,
    summarize_metric_tensors,
    write_json,
)
from .render import (
    apply_lut,
    gamut_penalty,
    identity_lut,
    lut_stats,
    render_hybrid,
    smoothness2_3d,
    tv3d,
)


@dataclass(frozen=True)
class A2RenderMode:
    name: str
    render_weight: float
    final_grid_weight: float
    pre_grid_weight: float
    regularize: bool
    gt_action_weight: float = 0.0
    smoothness_weight: float = 0.005
    gamut_weight: float = 1.0
    gamut_l1_weight: float = 0.0
    bound_margin: float = 0.0
    dict_l1_weight: float = 0.002
    tail_gate_weight: float = 0.01


@dataclass(frozen=True)
class A2FitSpec:
    name: str
    dense: bool = False
    include_dictionary: bool = True
    include_free_tail: bool = True


DEFAULT_MODES = (
    A2RenderMode("A2-grid-only", render_weight=0.0, final_grid_weight=1.0, pre_grid_weight=1.0, regularize=False),
    A2RenderMode("A2-grid+render", render_weight=1.0, final_grid_weight=0.5, pre_grid_weight=0.5, regularize=False),
    A2RenderMode(
        "A2-grid+render+regularizer",
        render_weight=1.0,
        final_grid_weight=0.5,
        pre_grid_weight=0.5,
        regularize=True,
    ),
    A2RenderMode(
        "A2-grid+render+gt-action-aux",
        render_weight=1.0,
        final_grid_weight=0.5,
        pre_grid_weight=0.5,
        gt_action_weight=0.2,
        regularize=True,
    ),
)

DEFAULT_FIT_SPECS = (
    A2FitSpec("Fit-Main", include_dictionary=False, include_free_tail=False),
    A2FitSpec("Fit-Dict", include_dictionary=True, include_free_tail=False),
    A2FitSpec("Fit-Full", include_dictionary=True, include_free_tail=True),
    A2FitSpec("D4-Dense17", dense=True, include_dictionary=False, include_free_tail=False),
)


def run_a2_render_check(
    data_config: M12DataConfig,
    output: Path,
    fit_steps: int,
    lr: float,
    modes: tuple[A2RenderMode, ...] = DEFAULT_MODES,
    fit_specs: tuple[A2FitSpec, ...] = DEFAULT_FIT_SPECS,
) -> dict[str, Any]:
    started = now_stamp()
    device = select_device(data_config.device)
    lpips_model = load_lpips_model(device)
    data = load_clean_tier_a(data_config)

    gt_rows = []
    for batch in dataset_batches(data, data_config.batch_size):
        gt_rows.append(gt_replay_batch(batch, data.dictionary, data.rho, lpips_model))
    gt_metrics = summarize_metric_tensors(gt_rows)
    gt_pass = passes_a2_with_gamut(gt_metrics)

    rows: list[dict[str, Any]] = []
    for mode in modes:
        for spec in fit_specs:
            metric_rows = []
            for batch in dataset_batches(data, data_config.batch_size):
                metric_rows.append(fit_a2_batch(batch, mode, spec, data.dictionary, data.rho, lpips_model, fit_steps, lr))
            metrics = summarize_metric_tensors(metric_rows)
            rows.append(
                {
                    "mode": mode.name,
                    "fit": spec.name,
                    "mode_config": asdict(mode),
                    "fit_config": asdict(spec),
                    "metrics": metrics,
                    "image_metric_pass": passes_a2_image_metrics(metrics),
                    "bound_pass": passes_gamut_bounds(metrics),
                    "pass": passes_a2_with_gamut(metrics),
                }
            )
            _write_partial(output, started, data_config, data, gt_metrics, gt_pass, rows, complete=False)

    result = {
        "experiment": "M1.3_A2_rendered_image_closure",
        "started_at": started,
        "ended_at": now_stamp(),
        "complete": True,
        "data_config": data_config.to_dict(),
        "fit_steps": fit_steps,
        "lr": lr,
        "thresholds": thresholds_with_gamut(),
        "gt_action_stats": data.gt_stats,
        "sampling": data.sampling,
        "gt_replay": gt_replay_payload(gt_metrics, gt_pass),
        "rows": rows,
        "decision": decide_a2_render(gt_pass, rows),
    }
    write_json(output, result)
    return result


def fit_a2_batch(
    batch: dict[str, torch.Tensor],
    mode: A2RenderMode,
    spec: A2FitSpec,
    dictionary: torch.Tensor,
    rho: torch.Tensor,
    lpips_model,
    fit_steps: int,
    lr: float,
) -> dict[str, torch.Tensor]:
    if spec.dense:
        return fit_dense_a2_batch(batch, mode, lpips_model, fit_steps, lr)
    return fit_shape_a2_batch(batch, mode, spec, dictionary, rho, lpips_model, fit_steps, lr)


def fit_shape_a2_batch(
    batch: dict[str, torch.Tensor],
    mode: A2RenderMode,
    spec: A2FitSpec,
    dictionary: torch.Tensor,
    rho: torch.Tensor,
    lpips_model,
    fit_steps: int,
    lr: float,
) -> dict[str, torch.Tensor]:
    source = batch["source"]
    target = batch["target"]
    gt_action = decode_raw_action(batch["raw"])
    raw = make_raw_parameter(source.shape[0], source.device, spec.include_dictionary, spec.include_free_tail)
    opt = torch.optim.AdamW([raw], lr=lr)
    for _ in range(fit_steps):
        opt.zero_grad(set_to_none=True)
        action = decode_raw_action(mask_raw_for_config(raw, spec.include_dictionary, spec.include_free_tail))
        pred, luts = render_hybrid(source, action, dictionary, rho, spec.include_dictionary, spec.include_free_tail)
        loss = grid_render_loss(pred, target, luts, batch, mode)
        if mode.gt_action_weight:
            loss = loss + mode.gt_action_weight * action_aux_loss(action, gt_action, spec.include_dictionary, spec.include_free_tail)
        if mode.regularize:
            loss = loss + mode.smoothness_weight * smoothness2_3d(luts["final"]).mean()
            loss = loss + mode.gamut_weight * gamut_penalty(luts["pre"])
            if mode.gamut_l1_weight:
                loss = loss + mode.gamut_l1_weight * gamut_l1_penalty(luts["pre"], margin=mode.bound_margin)
            if spec.include_dictionary:
                loss = loss + mode.dict_l1_weight * action.dict_coef.abs().mean()
            if spec.include_free_tail:
                loss = loss + mode.tail_gate_weight * action.tail_gate.mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        action = decode_raw_action(mask_raw_for_config(raw, spec.include_dictionary, spec.include_free_tail))
        pred, luts = render_hybrid(source, action, dictionary, rho, spec.include_dictionary, spec.include_free_tail)
        aux = lut_stats(action, luts)
        add_grid_metrics(aux, luts, batch)
        aux["action_aux_l1"] = action_aux_per_image(action, gt_action, spec.include_dictionary, spec.include_free_tail)
        return metric_summary(pred, target, aux, lpips_model)


def fit_dense_a2_batch(
    batch: dict[str, torch.Tensor],
    mode: A2RenderMode,
    lpips_model,
    fit_steps: int,
    lr: float,
) -> dict[str, torch.Tensor]:
    source = batch["source"]
    target = batch["target"]
    grid = batch["gt_final_lut"].shape[1]
    identity = identity_lut(g=grid, device=source.device, dtype=source.dtype).unsqueeze(0).repeat(source.shape[0], 1, 1, 1, 1)
    raw_delta = nn.Parameter(torch.zeros_like(identity))
    opt = torch.optim.AdamW([raw_delta], lr=lr)
    for _ in range(fit_steps):
        opt.zero_grad(set_to_none=True)
        luts = dense_luts(identity, raw_delta)
        pred = apply_lut(luts["final"], source)
        loss = grid_render_loss(pred, target, luts, batch, mode)
        if mode.regularize:
            loss = loss + mode.smoothness_weight * smoothness2_3d(luts["final"]).mean()
            loss = loss + 0.002 * tv3d(luts["final"])
            loss = loss + mode.gamut_weight * gamut_penalty(luts["pre"])
        loss.backward()
        opt.step()

    with torch.no_grad():
        luts = dense_luts(identity, raw_delta)
        pred = apply_lut(luts["final"], source)
        energy = (luts["final"] - identity).pow(2).mean(dim=(1, 2, 3, 4)).sqrt()
        aux = {
            "clipping_ratio": ((luts["pre"] < 0.0) | (luts["pre"] > 1.0)).float().mean(dim=(1, 2, 3, 4)),
            "gamut_violation_ratio": ((luts["pre"] < -1e-4) | (luts["pre"] > 1.0001)).float().mean(dim=(1, 2, 3, 4)),
            "banding_smoothness_penalty": smoothness2_3d(luts["final"]),
            "main_energy": torch.zeros_like(energy),
            "dictionary_energy": torch.zeros_like(energy),
            "free_tail_energy": torch.zeros_like(energy),
            "residual_ratio": torch.ones_like(energy),
            "dictionary_explained_ratio": torch.zeros_like(energy),
            "active_atom_count": torch.zeros_like(energy),
            "inactive_atom_ratio": torch.ones_like(energy),
        }
        add_grid_metrics(aux, luts, batch)
        return metric_summary(pred, target, aux, lpips_model)


def gt_replay_batch(
    batch: dict[str, torch.Tensor],
    dictionary: torch.Tensor,
    rho: torch.Tensor,
    lpips_model,
) -> dict[str, torch.Tensor]:
    action = decode_raw_action(batch["raw"])
    pred, luts = render_hybrid(batch["source"], action, dictionary, rho, True, True)
    aux = lut_stats(action, luts)
    add_grid_metrics(aux, luts, batch)
    return metric_summary(pred, batch["target"], aux, lpips_model)


def dense_luts(identity: torch.Tensor, raw_delta: torch.Tensor) -> dict[str, torch.Tensor]:
    lut_pre = identity + 0.35 * torch.tanh(raw_delta)
    lut_final = lut_pre.clamp(0.0, 1.0)
    return {
        "pre": lut_pre,
        "final": lut_final,
        "identity": identity[:1],
    }


def grid_render_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    luts: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    mode: A2RenderMode,
) -> torch.Tensor:
    loss = pred.new_tensor(0.0)
    if mode.render_weight:
        loss = loss + mode.render_weight * (F.mse_loss(pred, target) + 0.15 * F.l1_loss(pred, target))
    if mode.final_grid_weight:
        loss = loss + mode.final_grid_weight * F.mse_loss(luts["final"], batch["gt_final_lut"])
    if mode.pre_grid_weight:
        loss = loss + mode.pre_grid_weight * F.mse_loss(luts["pre"], batch["gt_pre_lut"])
    return loss


def gamut_l1_penalty(lut_pre: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    lo = margin
    hi = 1.0 - margin
    return F.relu(lut_pre - hi).mean() + F.relu(lo - lut_pre).mean()


def action_aux_loss(pred, target, include_dictionary: bool, include_free_tail: bool) -> torch.Tensor:
    return action_aux_per_image(pred, target, include_dictionary, include_free_tail).mean()


def action_aux_per_image(pred, target, include_dictionary: bool, include_free_tail: bool) -> torch.Tensor:
    curve = (pred.curves - target.curves).abs().mean(dim=(1, 2))
    hsl = (pred.hsl - target.hsl).abs().mean(dim=(1, 2))
    wb = (pred.wb - target.wb).abs().mean(dim=1)
    parts = [curve, hsl, wb]
    if include_dictionary:
        parts.append((pred.dict_coef - target.dict_coef).abs().mean(dim=1))
    if include_free_tail:
        parts.append((pred.tail_gate - target.tail_gate).abs().mean(dim=1))
        parts.append((pred.tail_color - target.tail_color).abs().mean(dim=(1, 2)))
    return sum(parts) / len(parts)


def add_grid_metrics(aux: dict[str, torch.Tensor], luts: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> None:
    aux["final_grid_mse"] = (luts["final"] - batch["gt_final_lut"]).pow(2).mean(dim=(1, 2, 3, 4))
    aux["pre_grid_mse"] = (luts["pre"] - batch["gt_pre_lut"]).pow(2).mean(dim=(1, 2, 3, 4))


def thresholds_with_gamut() -> dict[str, float]:
    out = dict(A2_THRESHOLDS)
    out["gamut_violation_ratio"] = 0.01
    return out


def passes_a2_with_gamut(metrics: dict[str, float]) -> bool:
    return passes_a2_image_metrics(metrics) and passes_gamut_bounds(metrics)


def passes_a2_image_metrics(metrics: dict[str, float]) -> bool:
    thresholds = thresholds_with_gamut()
    return bool(
        metrics.get("PSNR", 0.0) >= thresholds["PSNR"]
        and metrics.get("LPIPS", float("inf")) <= thresholds["LPIPS"]
        and metrics.get("mean_deltaE2000", float("inf")) <= thresholds["mean_deltaE2000"]
        and metrics.get("p95_deltaE2000", float("inf")) <= thresholds["p95_deltaE2000"]
    )


def passes_gamut_bounds(metrics: dict[str, float]) -> bool:
    thresholds = thresholds_with_gamut()
    return bool(
        metrics.get("clipping_ratio", float("inf")) <= thresholds["clipping_ratio"]
        and metrics.get("gamut_violation_ratio", float("inf")) <= thresholds["gamut_violation_ratio"]
    )


def decide_a2_render(gt_pass: bool, rows: list[dict[str, Any]]) -> dict[str, Any]:
    full_by_mode = {row["mode"]: row for row in rows if row["fit"] == "Fit-Full"}
    any_full_pass = any(row.get("pass", False) for row in full_by_mode.values())
    any_full_image_pass = any(row.get("image_metric_pass", False) for row in full_by_mode.values())
    any_full_bound_pass = any(row.get("bound_pass", False) for row in full_by_mode.values())
    if not gt_pass:
        conclusion = "A2_GT_REPLAY_FAIL"
    elif any_full_pass:
        conclusion = "A2_RENDER_CLOSED"
    elif any_full_image_pass and not any_full_bound_pass:
        conclusion = "A2_CLIPPING_GAMUT_NOT_CLOSED"
    elif "A2-grid-only" in full_by_mode and not full_by_mode["A2-grid-only"].get("image_metric_pass", False):
        conclusion = "A2_GRID_ONLY_IMAGE_TRANSFER_FAIL"
    elif "A2-grid+render" in full_by_mode and not full_by_mode["A2-grid+render"].get("image_metric_pass", False):
        conclusion = "A2_GRID_RENDER_IMAGE_METRICS_FAIL"
    elif "A2-grid+render+regularizer" in full_by_mode and not full_by_mode["A2-grid+render+regularizer"].get("image_metric_pass", False):
        conclusion = "A2_REGULARIZER_IMAGE_METRICS_CONFLICT"
    else:
        conclusion = "A2_RENDER_NOT_CLOSED"
    return {
        "conclusion": conclusion,
        "gt_replay_pass": gt_pass,
        "fit_full_by_mode": {
            mode: {
                "image_metric_pass": row.get("image_metric_pass", False),
                "bound_pass": row.get("bound_pass", False),
                "pass": row.get("pass", False),
            }
            for mode, row in full_by_mode.items()
        },
    }


def gt_replay_payload(metrics: dict[str, float], passed: bool) -> dict[str, Any]:
    return {
        "metrics": metrics,
        "image_metric_pass": passes_a2_image_metrics(metrics),
        "bound_pass": passes_gamut_bounds(metrics),
        "pass": passed,
    }


def _write_partial(
    output: Path,
    started: str,
    data_config: M12DataConfig,
    data,
    gt_metrics: dict[str, float],
    gt_pass: bool,
    rows: list[dict[str, Any]],
    complete: bool,
) -> None:
    write_json(
        output,
        {
            "experiment": "M1.3_A2_rendered_image_closure",
            "started_at": started,
            "ended_at": now_stamp() if complete else None,
            "complete": complete,
            "data_config": data_config.to_dict(),
            "thresholds": thresholds_with_gamut(),
            "gt_action_stats": data.gt_stats,
            "sampling": data.sampling,
            "gt_replay": gt_replay_payload(gt_metrics, gt_pass),
            "rows": rows,
            "decision": decide_a2_render(gt_pass, rows),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run M1.3 A2 rendered-image closure check.")
    parser.add_argument("--data-root", default="~/retouching/monetGPT/data")
    parser.add_argument("--output", default="m1_results/m1_3_a2_render_check_20260525.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--fit-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--regularizer-gamut-sweep",
        default="",
        help="Comma-separated gamut weights for Fit-Full A2-grid+render+regularizer troubleshooting.",
    )
    parser.add_argument(
        "--bound-l1-sweep",
        default="",
        help="Comma-separated L1 bound weights for Fit-Full A2-grid+render+regularizer troubleshooting.",
    )
    parser.add_argument("--gt-action-aux", action="store_true", help="Run only the GT-action auxiliary Fit-Full closure mode.")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.count = min(args.count, 8)
        args.batch_size = min(args.batch_size, 4)
        args.fit_steps = min(args.fit_steps, 4)
    modes = DEFAULT_MODES
    fit_specs = DEFAULT_FIT_SPECS
    if args.regularizer_gamut_sweep:
        weights = [float(item) for item in args.regularizer_gamut_sweep.split(",") if item.strip()]
        modes = tuple(
            A2RenderMode(
                f"A2-grid+render+regularizer-gamut{weight:g}",
                render_weight=1.0,
                final_grid_weight=0.5,
                pre_grid_weight=0.5,
                regularize=True,
                gamut_weight=weight,
            )
            for weight in weights
        )
        fit_specs = (A2FitSpec("Fit-Full", include_dictionary=True, include_free_tail=True),)
    if args.bound_l1_sweep:
        weights = [float(item) for item in args.bound_l1_sweep.split(",") if item.strip()]
        modes = tuple(
            A2RenderMode(
                f"A2-grid+render+regularizer-boundL1{weight:g}",
                render_weight=1.0,
                final_grid_weight=0.5,
                pre_grid_weight=0.5,
                regularize=True,
                gamut_weight=4.0,
                gamut_l1_weight=weight,
            )
            for weight in weights
        )
        fit_specs = (A2FitSpec("Fit-Full", include_dictionary=True, include_free_tail=True),)
    if args.gt_action_aux:
        modes = (
            A2RenderMode(
                "A2-grid+render+gt-action-aux",
                render_weight=1.0,
                final_grid_weight=0.5,
                pre_grid_weight=0.5,
                gt_action_weight=0.2,
                regularize=True,
            ),
        )
        fit_specs = (A2FitSpec("Fit-Full", include_dictionary=True, include_free_tail=True),)
    result = run_a2_render_check(
        M12DataConfig(
            data_root=Path(args.data_root).expanduser(),
            count=args.count,
            image_size=args.image_size,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
        ),
        Path(args.output),
        args.fit_steps,
        args.lr,
        modes,
        fit_specs,
    )
    print(result["decision"]["conclusion"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
