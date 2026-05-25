from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .m1_2_common import M12DataConfig, load_clean_tier_a, load_lpips_model, now_stamp, select_device, write_json
from .m1_3_capacity import (
    DynamicFitHyper,
    DynamicShapeConfig,
    build_dynamic_dictionary,
    fit_dynamic_batch,
)
from .render import build_dictionary


@dataclass(frozen=True)
class StabilityVariant:
    name: str
    dictionary_kind: str
    dict_l1_weight: float = 0.0005
    gamut_l1_weight: float = 0.5


DEFAULT_VARIANTS = (
    StabilityVariant("semantic-l1-0.0005", "semantic", 0.0005),
    StabilityVariant("semantic-l1-0.002", "semantic", 0.002),
    StabilityVariant("semantic-l1-0.005", "semantic", 0.005),
    StabilityVariant("semantic-l1-0.01", "semantic", 0.01),
    StabilityVariant("shuffled-l1-0.002", "shuffled", 0.002),
    StabilityVariant("random-l1-0.002", "random", 0.002),
)


def run_dictionary_stability(
    data_config: M12DataConfig,
    output: Path,
    shape_config: DynamicShapeConfig,
    variants: tuple[StabilityVariant, ...],
    restarts: int,
    fit_steps: int,
    lr: float,
    top_k: int,
) -> dict[str, Any]:
    started = now_stamp()
    device = select_device(data_config.device)
    lpips_model = load_lpips_model(device)
    data = load_clean_tier_a(data_config)
    dictionary16, rho16, atom_names = build_dictionary(device=device)
    source = data.source
    target = data.target

    rows = []
    for variant in variants:
        atoms, rho = build_dynamic_dictionary(
            dictionary16,
            rho16,
            DynamicShapeConfig(
                shape_config.name,
                shape_config.m_atoms,
                shape_config.r_free,
                shape_config.k_spline,
                dictionary_kind=variant.dictionary_kind,
            ),
            data_config.seed + stable_variant_offset(variant.name),
            device,
        )
        restart_rows = []
        for restart in range(restarts):
            torch.manual_seed(data_config.seed + 10_000 * (restart + 1) + stable_variant_offset(variant.name))
            metrics = fit_dynamic_batch(
                source,
                target,
                shape_config,
                DynamicFitHyper(
                    steps=fit_steps,
                    lr=lr,
                    dict_l1_weight=variant.dict_l1_weight,
                    gamut_l1_weight=variant.gamut_l1_weight,
                ),
                atoms,
                rho,
                lpips_model,
            )
            restart_rows.append(summarize_restart(metrics, top_k))
            _write_partial(output, started, data_config, shape_config, variants, rows, complete=False)
        row = {
            "variant": asdict(variant),
            "summary": summarize_variant(restart_rows, top_k),
            "restarts": restart_rows,
        }
        rows.append(row)
        _write_partial(output, started, data_config, shape_config, variants, rows, complete=False)

    result = {
        "experiment": "M1.3_dictionary_restart_stability",
        "started_at": started,
        "ended_at": now_stamp(),
        "complete": True,
        "data_config": data_config.to_dict(),
        "shape_config": asdict(shape_config),
        "restarts": restarts,
        "fit_steps": fit_steps,
        "lr": lr,
        "top_k": top_k,
        "variants": rows,
        "decision": decide_stability(rows),
        "atom_names": list(atom_names[: shape_config.m_atoms]),
    }
    write_json(output, result)
    return result


def summarize_restart(metrics: dict[str, torch.Tensor], top_k: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        values = value.detach().float().cpu().numpy().reshape(-1)
        out[key] = float(values.mean()) if values.size else float("nan")
    coef = metrics.get("dict_coef")
    if coef is not None:
        coef_np = coef.detach().float().cpu().numpy()
        out["coef_abs_mean"] = float(np.abs(coef_np).mean())
        out["topk"] = topk_indices(coef_np, top_k)
    return out


def summarize_variant(restarts: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    numeric: dict[str, list[float]] = {}
    for row in restarts:
        for key, value in row.items():
            if isinstance(value, int | float) and not np.isnan(value):
                numeric.setdefault(key, []).append(float(value))
    summary: dict[str, Any] = {}
    for key, vals in numeric.items():
        arr = np.asarray(vals, dtype=np.float64)
        summary[key] = float(arr.mean())
        summary[f"{key}_std"] = float(arr.std())
    topks = [row.get("topk", []) for row in restarts]
    summary["topk_jaccard"] = mean_topk_jaccard(topks)
    summary["top_k"] = top_k
    return summary


def topk_indices(coef_np: np.ndarray, top_k: int) -> list[list[int]]:
    return [np.argsort(-np.abs(row))[:top_k].astype(int).tolist() for row in coef_np]


def mean_topk_jaccard(topks: list[list[list[int]]]) -> float:
    if len(topks) < 2:
        return float("nan")
    vals = []
    for left, right in combinations(topks, 2):
        for l_row, r_row in zip(left, right):
            l_set = set(l_row)
            r_set = set(r_row)
            vals.append(len(l_set & r_set) / max(1, len(l_set | r_set)))
    return float(np.mean(vals)) if vals else float("nan")


def decide_stability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    semantic = [row for row in rows if row["variant"]["dictionary_kind"] == "semantic"]
    controls = [row for row in rows if row["variant"]["dictionary_kind"] != "semantic"]
    best_semantic = min(semantic, key=lambda row: row["summary"].get("mean_deltaE2000", float("inf"))) if semantic else None
    best_control = min(controls, key=lambda row: row["summary"].get("mean_deltaE2000", float("inf"))) if controls else None
    stable = False
    semantic_gain = None
    if best_semantic:
        s = best_semantic["summary"]
        stable = (
            s.get("topk_jaccard", 0.0) >= 0.50
            and s.get("active_atom_count", float("inf")) <= 6.0
            and s.get("inactive_atom_ratio", 0.0) >= 0.40
        )
    if best_semantic and best_control:
        semantic_gain = {
            "deltaE_drop_vs_best_control": best_control["summary"].get("mean_deltaE2000", float("nan"))
            - best_semantic["summary"].get("mean_deltaE2000", float("nan")),
            "lpips_drop_vs_best_control": best_control["summary"].get("LPIPS", float("nan"))
            - best_semantic["summary"].get("LPIPS", float("nan")),
        }
    if stable:
        conclusion = "DICTIONARY_STABILITY_PASS"
    elif best_semantic:
        conclusion = "DICTIONARY_INTERPRETABILITY_PARTIAL"
    else:
        conclusion = "DICTIONARY_STABILITY_NOT_RUN"
    return {
        "conclusion": conclusion,
        "best_semantic": best_semantic["variant"]["name"] if best_semantic else None,
        "best_control": best_control["variant"]["name"] if best_control else None,
        "semantic_gain": semantic_gain,
    }


def stable_variant_offset(text: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text)) % 100_000


def _write_partial(
    output: Path,
    started: str,
    data_config: M12DataConfig,
    shape_config: DynamicShapeConfig,
    variants: tuple[StabilityVariant, ...],
    rows: list[dict[str, Any]],
    complete: bool,
) -> None:
    write_json(
        output,
        {
            "experiment": "M1.3_dictionary_restart_stability",
            "started_at": started,
            "ended_at": now_stamp() if complete else None,
            "complete": complete,
            "data_config": data_config.to_dict(),
            "shape_config": asdict(shape_config),
            "variant_plan": [asdict(v) for v in variants],
            "variants": rows,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run M1.3 dictionary restart stability controls.")
    parser.add_argument("--data-root", default="~/retouching/monetGPT/data")
    parser.add_argument("--output", default="m1_results/m1_3_dictionary_stability_20260525.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--fit-steps", type=int, default=80)
    parser.add_argument("--fit-lr", type=float, default=0.07)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--m-atoms", type=int, default=32)
    parser.add_argument("--r-free", type=int, default=8)
    parser.add_argument("--k-spline", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.count = min(args.count, 16)
        args.restarts = min(args.restarts, 2)
        args.fit_steps = min(args.fit_steps, 4)
    result = run_dictionary_stability(
        M12DataConfig(
            data_root=Path(args.data_root).expanduser(),
            count=args.count,
            image_size=args.image_size,
            batch_size=args.count,
            seed=args.seed,
            device=args.device,
        ),
        Path(args.output),
        DynamicShapeConfig(f"M{args.m_atoms}-R{args.r_free}-K{args.k_spline}", args.m_atoms, args.r_free, args.k_spline),
        DEFAULT_VARIANTS,
        args.restarts,
        args.fit_steps,
        args.fit_lr,
        args.top_k,
    )
    print(result["decision"]["conclusion"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
