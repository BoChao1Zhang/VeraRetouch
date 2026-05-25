from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .action import G_MAX_TAIL, TAIL_COLOR_SCALE, decode_raw_action, identity_raw
from .data import DataRequirements, discover_m1_data
from .image_io import load_image
from .metrics import MetricAccumulator, metric_summary
from .m1_2_common import A2_THRESHOLDS, load_clean_tier_a, load_lpips_model, now_stamp, select_device, write_json, M12DataConfig
from .render import (
    apply_lut,
    bspline_basis,
    build_dictionary,
    gamut_penalty,
    identity_lut,
    lut_stats,
    main_lut,
    normalize_atom,
    render_hybrid,
    softclip_identity,
    smooth3d,
    smoothness2_3d,
    tv3d,
)
from .synthetic import TIER_B_FAMILIES, render_dense_teacher


@dataclass(frozen=True)
class DynamicShapeConfig:
    name: str
    m_atoms: int
    r_free: int
    k_spline: int
    dictionary_kind: str = "semantic"


@dataclass(frozen=True)
class DynamicFitHyper:
    steps: int = 80
    lr: float = 0.07
    gamut_weight: float = 4.0
    smoothness_weight: float = 0.005
    dict_l1_weight: float = 0.0005
    tail_gate_weight: float = 0.01


@dataclass(frozen=True)
class DenseFitConfig:
    name: str
    grid: int
    steps: int
    lr: float
    smoothness_weight: float
    tv_weight: float
    gamut_weight: float
    delta_scale: float = 0.35


SHAPE_CONFIGS = (
    DynamicShapeConfig("C0-Main", 0, 0, 10),
    DynamicShapeConfig("C1-M16-R4-K10", 16, 4, 10),
    DynamicShapeConfig("C2-M32-R4-K10", 32, 4, 10),
    DynamicShapeConfig("C3-M16-R8-K10", 16, 8, 10),
    DynamicShapeConfig("C4-M16-R4-K12", 16, 4, 12),
    DynamicShapeConfig("C5-M32-R8-K12", 32, 8, 12),
)

D4_CONFIGS = (
    DenseFitConfig("D4-33-midreg", 33, 80, 0.05, 0.02, 0.005, 1.0),
    DenseFitConfig("D4-33-highreg", 33, 80, 0.05, 0.05, 0.01, 2.0),
)


def run_capacity(
    output: Path,
    data_root: Path,
    device_name: str,
    image_size: int,
    batch_size: int,
    tier_a_count: int,
    tier_b_count: int,
    fivek_count: int,
    ppr10k_count: int,
    seed: int,
    fit_hyper: DynamicFitHyper,
) -> dict[str, Any]:
    started = now_stamp()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = select_device(device_name)
    lpips_model = load_lpips_model(device)
    dictionary16, rho16, atom_names = build_dictionary(device=device)
    discovery = discover_m1_data(data_root, DataRequirements())
    if not discovery.ready:
        raise RuntimeError("M1 data discovery is not ready")

    tier_a = load_clean_tier_a(M12DataConfig(data_root, tier_a_count, image_size, batch_size, seed, str(device)))
    tiers = {
        "tier_a_clean": (tier_a.source, tier_a.target),
        "tier_b_dense_teacher": load_tier_b(discovery, tier_b_count, image_size, batch_size, seed + 20_000, device),
        "real_ppr10k_target_c": load_pairs(_stable_sample(list(discovery.ppr10k_pairs), ppr10k_count, seed + 4), image_size, device),
        "real_fivek_mmart_like": load_pairs(_stable_sample(list(discovery.fivek_pairs), fivek_count, seed + 3), image_size, device),
    }

    report: dict[str, Any] = {
        "experiment": "M1.3_capacity_ablation",
        "started_at": started,
        "config": {
            "data_root": str(data_root),
            "device": str(device),
            "image_size": image_size,
            "batch_size": batch_size,
            "tier_a_count": tier_a_count,
            "tier_b_count": tier_b_count,
            "fivek_count": fivek_count,
            "ppr10k_count": ppr10k_count,
            "seed": seed,
            "fit_hyper": asdict(fit_hyper),
        },
        "atom_names": list(atom_names),
        "tier_a_gt_stats": tier_a.gt_stats,
        "tiers": {},
    }

    for tier_name, (source, target) in tiers.items():
        tier_rows: dict[str, Any] = {}
        for shape_config in SHAPE_CONFIGS:
            atoms, rho = build_dynamic_dictionary(dictionary16, rho16, shape_config, seed, device)
            metrics = fit_dynamic_dataset(source, target, shape_config, fit_hyper, atoms, rho, lpips_model, batch_size)
            tier_rows[shape_config.name] = {"metrics": metrics, "config": asdict(shape_config)}
        for dense_config in D4_CONFIGS:
            metrics = fit_dense_dataset(source, target, dense_config, lpips_model, batch_size)
            tier_rows[dense_config.name] = {"metrics": metrics, "config": asdict(dense_config)}
        report["tiers"][tier_name] = {
            "configs": tier_rows,
            "derived": derive_capacity_metrics(tier_rows),
        }
        write_json(output, {**report, "complete": False, "ended_at": None})

    report["decision"] = decide_capacity(report)
    report["complete"] = True
    report["ended_at"] = now_stamp()
    write_json(output, report)
    return report


def fit_dynamic_dataset(
    source: torch.Tensor,
    target: torch.Tensor,
    config: DynamicShapeConfig,
    hyper: DynamicFitHyper,
    atoms: torch.Tensor,
    rho: torch.Tensor,
    lpips_model,
    batch_size: int,
) -> dict[str, float]:
    rows = []
    for start in range(0, source.shape[0], batch_size):
        sl = slice(start, min(start + batch_size, source.shape[0]))
        rows.append(fit_dynamic_batch(source[sl], target[sl], config, hyper, atoms, rho, lpips_model))
    return summarize(rows)


def fit_dynamic_batch(
    source: torch.Tensor,
    target: torch.Tensor,
    config: DynamicShapeConfig,
    hyper: DynamicFitHyper,
    atoms: torch.Tensor,
    rho: torch.Tensor,
    lpips_model,
) -> dict[str, torch.Tensor]:
    batch = source.shape[0]
    raw_main = nn.Parameter(identity_raw(batch, source.device))
    dict_raw = nn.Parameter(torch.zeros(batch, config.m_atoms, device=source.device))
    tail_gate_raw = nn.Parameter(torch.full((batch, config.r_free), -3.0, device=source.device))
    tail_color_raw = nn.Parameter(0.01 * torch.randn(batch, config.r_free, 3, device=source.device))
    tail_alpha = nn.Parameter(0.05 * torch.randn(batch, config.r_free, config.k_spline, device=source.device))
    tail_beta = nn.Parameter(0.05 * torch.randn(batch, config.r_free, config.k_spline, device=source.device))
    tail_gamma = nn.Parameter(0.05 * torch.randn(batch, config.r_free, config.k_spline, device=source.device))
    params = [raw_main]
    if config.m_atoms:
        params.append(dict_raw)
    if config.r_free:
        params.extend([tail_gate_raw, tail_color_raw, tail_alpha, tail_beta, tail_gamma])
    opt = torch.optim.AdamW(params, lr=hyper.lr)
    for _ in range(hyper.steps):
        opt.zero_grad(set_to_none=True)
        pred, luts, aux_action = render_dynamic(source, raw_main, dict_raw, tail_gate_raw, tail_color_raw, tail_alpha, tail_beta, tail_gamma, config, atoms, rho)
        loss = F.mse_loss(pred, target) + 0.15 * F.l1_loss(pred, target)
        loss = loss + hyper.smoothness_weight * smoothness2_3d(luts["final"]).mean()
        loss = loss + hyper.gamut_weight * gamut_penalty(luts["pre"])
        if config.m_atoms:
            loss = loss + hyper.dict_l1_weight * aux_action["dict_coef"].abs().mean()
        if config.r_free:
            loss = loss + hyper.tail_gate_weight * aux_action["tail_gate"].mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred, luts, aux_action = render_dynamic(source, raw_main, dict_raw, tail_gate_raw, tail_color_raw, tail_alpha, tail_beta, tail_gamma, config, atoms, rho)
        aux = dynamic_lut_stats(aux_action, luts)
        return metric_summary(pred, target, aux, lpips_model)


def render_dynamic(
    image: torch.Tensor,
    raw_main: torch.Tensor,
    dict_raw: torch.Tensor,
    tail_gate_raw: torch.Tensor,
    tail_color_raw: torch.Tensor,
    tail_alpha: torch.Tensor,
    tail_beta: torch.Tensor,
    tail_gamma: torch.Tensor,
    config: DynamicShapeConfig,
    atoms: torch.Tensor,
    rho: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    main_action = decode_raw_action(raw_main)
    lut_main = main_lut(main_action)
    if config.m_atoms:
        dict_coef = torch.tanh(dict_raw)
        lut_dict = torch.einsum("bm,mijkc,m->bijkc", dict_coef, atoms, rho)
    else:
        dict_coef = torch.zeros(image.shape[0], 0, device=image.device)
        lut_dict = torch.zeros_like(lut_main)
    if config.r_free:
        gate = G_MAX_TAIL * torch.sigmoid(tail_gate_raw)
        color = torch.tanh(tail_color_raw) * TAIL_COLOR_SCALE
        lut_tail = dynamic_tail_lut(gate, color, tail_alpha, tail_beta, tail_gamma, config.k_spline)
    else:
        gate = torch.zeros(image.shape[0], 0, device=image.device)
        color = torch.zeros(image.shape[0], 0, 3, device=image.device)
        lut_tail = torch.zeros_like(lut_main)
    lut_pre = lut_main + lut_dict + lut_tail
    lut_final = softclip_identity(lut_pre)
    luts = {
        "main": lut_main,
        "dict": lut_dict,
        "tail": lut_tail,
        "pre": lut_pre,
        "final": lut_final,
        "identity": identity_lut(device=image.device, dtype=image.dtype).unsqueeze(0),
    }
    return apply_lut(lut_final, image), luts, {"dict_coef": dict_coef, "tail_gate": gate, "tail_color": color}


def dynamic_tail_lut(gate: torch.Tensor, color: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, k_spline: int) -> torch.Tensor:
    basis = bspline_basis(k=k_spline, device=gate.device).to(gate.dtype)
    u = torch.einsum("gk,brk->brg", basis, alpha)
    v = torch.einsum("gk,brk->brg", basis, beta)
    w = torch.einsum("gk,brk->brg", basis, gamma)
    u = u / (u.norm(dim=-1, keepdim=True) + 1e-6)
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-6)
    w = w / (w.norm(dim=-1, keepdim=True) + 1e-6)
    return torch.einsum("br,brc,bri,brj,brk->bijkc", gate, color, u, v, w)


def dynamic_lut_stats(aux_action: dict[str, torch.Tensor], luts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    identity = luts["identity"]
    main_energy = (luts["main"] - identity).pow(2).mean(dim=(1, 2, 3, 4)).sqrt()
    dict_energy = luts["dict"].pow(2).mean(dim=(1, 2, 3, 4)).sqrt()
    tail_energy = luts["tail"].pow(2).mean(dim=(1, 2, 3, 4)).sqrt()
    residual = dict_energy + tail_energy
    dict_coef = aux_action["dict_coef"]
    if dict_coef.numel():
        active = (dict_coef.abs() >= 0.03).float().sum(dim=1)
        inactive = (dict_coef.abs() < 0.03).float().mean(dim=1)
    else:
        active = torch.zeros_like(main_energy)
        inactive = torch.ones_like(main_energy)
    return {
        "clipping_ratio": ((luts["pre"] < 0.0) | (luts["pre"] > 1.0)).float().mean(dim=(1, 2, 3, 4)),
        "gamut_violation_ratio": ((luts["pre"] < -1e-4) | (luts["pre"] > 1.0001)).float().mean(dim=(1, 2, 3, 4)),
        "banding_smoothness_penalty": smoothness2_3d(luts["final"]),
        "main_energy": main_energy,
        "dictionary_energy": dict_energy,
        "free_tail_energy": tail_energy,
        "residual_ratio": residual / (main_energy + residual + 1e-6),
        "dictionary_explained_ratio": dict_energy / (residual + 1e-6),
        "active_atom_count": active,
        "inactive_atom_ratio": inactive,
    }


def fit_dense_dataset(source: torch.Tensor, target: torch.Tensor, config: DenseFitConfig, lpips_model, batch_size: int) -> dict[str, float]:
    rows = []
    for start in range(0, source.shape[0], batch_size):
        sl = slice(start, min(start + batch_size, source.shape[0]))
        rows.append(fit_dense33_batch(source[sl], target[sl], config, lpips_model))
    return summarize(rows)


def fit_dense33_batch(source: torch.Tensor, target: torch.Tensor, config: DenseFitConfig, lpips_model) -> dict[str, torch.Tensor]:
    batch = source.shape[0]
    identity = identity_lut(g=config.grid, device=source.device, dtype=source.dtype).unsqueeze(0).repeat(batch, 1, 1, 1, 1)
    raw_delta = nn.Parameter(torch.zeros_like(identity))
    opt = torch.optim.AdamW([raw_delta], lr=config.lr)
    for _ in range(config.steps):
        opt.zero_grad(set_to_none=True)
        lut_pre = identity + config.delta_scale * torch.tanh(raw_delta)
        lut_final = torch.clamp(lut_pre, 0.0, 1.0)
        pred = apply_lut(lut_final, source)
        loss = F.mse_loss(pred, target) + 0.15 * F.l1_loss(pred, target)
        loss = loss + config.smoothness_weight * smoothness2_3d(lut_final).mean()
        loss = loss + config.tv_weight * tv3d(lut_final)
        loss = loss + config.gamut_weight * gamut_penalty(lut_pre)
        loss.backward()
        opt.step()
    with torch.no_grad():
        lut_pre = identity + config.delta_scale * torch.tanh(raw_delta)
        lut_final = torch.clamp(lut_pre, 0.0, 1.0)
        pred = apply_lut(lut_final, source)
        energy = (lut_final - identity).pow(2).mean(dim=(1, 2, 3, 4)).sqrt()
        aux = {
            "clipping_ratio": ((lut_pre < 0.0) | (lut_pre > 1.0)).float().mean(dim=(1, 2, 3, 4)),
            "gamut_violation_ratio": ((lut_pre < -1e-4) | (lut_pre > 1.0001)).float().mean(dim=(1, 2, 3, 4)),
            "banding_smoothness_penalty": smoothness2_3d(lut_final),
            "main_energy": torch.zeros_like(energy),
            "dictionary_energy": torch.zeros_like(energy),
            "free_tail_energy": torch.zeros_like(energy),
            "residual_ratio": torch.ones_like(energy),
            "dictionary_explained_ratio": torch.zeros_like(energy),
            "active_atom_count": torch.zeros_like(energy),
            "inactive_atom_ratio": torch.ones_like(energy),
        }
        return metric_summary(pred, target, aux, lpips_model)


def build_dynamic_dictionary(
    semantic_atoms: torch.Tensor,
    semantic_rho: torch.Tensor,
    config: DynamicShapeConfig,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if config.m_atoms == 0:
        return torch.empty(0, *semantic_atoms.shape[1:], device=device), torch.empty(0, device=device)
    if config.dictionary_kind == "semantic":
        atoms = semantic_atoms
        rho = semantic_rho
        if config.m_atoms > semantic_atoms.shape[0]:
            extra = random_atoms(config.m_atoms - semantic_atoms.shape[0], seed + 31, device)
            atoms = torch.cat([atoms, extra], dim=0)
            rho = torch.cat([rho, torch.full((extra.shape[0],), float(semantic_rho.mean()), device=device)], dim=0)
        return atoms[: config.m_atoms], rho[: config.m_atoms]
    if config.dictionary_kind == "random":
        atoms = random_atoms(config.m_atoms, seed + 71, device)
        rho = torch.full((config.m_atoms,), float(semantic_rho.mean()), device=device)
        return atoms, rho
    if config.dictionary_kind == "shuffled":
        gen = torch.Generator(device=device).manual_seed(seed + 101)
        atoms = semantic_atoms[torch.randperm(semantic_atoms.shape[0], generator=gen, device=device)]
        rho = semantic_rho[torch.randperm(semantic_rho.shape[0], generator=gen, device=device)]
        if config.m_atoms > atoms.shape[0]:
            extra = random_atoms(config.m_atoms - atoms.shape[0], seed + 131, device)
            atoms = torch.cat([atoms, extra], dim=0)
            rho = torch.cat([rho, torch.full((extra.shape[0],), float(semantic_rho.mean()), device=device)], dim=0)
        return atoms[: config.m_atoms], rho[: config.m_atoms]
    raise ValueError(config.dictionary_kind)


def random_atoms(count: int, seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device=device).manual_seed(seed)
    atoms = []
    for _ in range(count):
        noise = torch.randn(5, 5, 5, 3, generator=gen, device=device)
        up = F.interpolate(noise.permute(3, 0, 1, 2).unsqueeze(0), size=(17, 17, 17), mode="trilinear", align_corners=True)
        field = smooth3d(up.squeeze(0).permute(1, 2, 3, 0), passes=2)
        atoms.append(normalize_atom(field))
    return torch.stack(atoms, dim=0)


def load_tier_b(discovery, count: int, image_size: int, batch_size: int, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    paths = _stable_sample(list(discovery.base_pool), count, seed)
    sources = []
    targets = []
    for batch_idx, sl_paths in enumerate(_chunks(paths, batch_size)):
        source = torch.stack([load_image(path, image_size) for path in sl_paths], dim=0).to(device)
        family = TIER_B_FAMILIES[batch_idx % len(TIER_B_FAMILIES)]
        seeds = [seed + batch_idx * batch_size + i for i in range(source.shape[0])]
        target, _ = render_dense_teacher(source, family, seeds, g=17)
        sources.append(source)
        targets.append(target.detach())
    return torch.cat(sources, dim=0), torch.cat(targets, dim=0)


def load_pairs(pairs, image_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    source = torch.stack([load_image(pair.source, image_size) for pair in pairs], dim=0).to(device)
    target = torch.stack([load_image(pair.target, image_size) for pair in pairs], dim=0).to(device)
    return source, target


def summarize(rows: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    acc = MetricAccumulator.create()
    for row in rows:
        for key, value in row.items():
            acc.extend_tensor(key, value)
    return acc.summary()


def derive_capacity_metrics(rows: dict[str, Any]) -> dict[str, Any]:
    d4_candidates = {name: row["metrics"] for name, row in rows.items() if name.startswith("D4-33")}
    clean = {name: m for name, m in d4_candidates.items() if m["clipping_ratio"] <= 0.01 and m["gamut_violation_ratio"] <= 0.01}
    d4_name, d4 = min((clean or d4_candidates).items(), key=lambda item: item[1]["mean_deltaE2000"])
    main = rows["C0-Main"]["metrics"]
    derived: dict[str, Any] = {"d4_reference": d4_name}
    denom = max(main["mean_deltaE2000"] - d4["mean_deltaE2000"], 1e-6)
    for name, row in rows.items():
        metrics = row["metrics"]
        derived[name] = {
            "gap_to_D4_deltaE": metrics["mean_deltaE2000"] - d4["mean_deltaE2000"],
            "gap_to_D4_LPIPS": metrics["LPIPS"] - d4["LPIPS"],
            "gap_to_D4_PSNR": d4["PSNR"] - metrics["PSNR"],
            "gap_closed_deltaE": (main["mean_deltaE2000"] - metrics["mean_deltaE2000"]) / denom,
            "deltaE_drop_vs_main": (main["mean_deltaE2000"] - metrics["mean_deltaE2000"]) / max(abs(main["mean_deltaE2000"]), 1e-6),
            "lpips_drop_vs_main": (main["LPIPS"] - metrics["LPIPS"]) / max(abs(main["LPIPS"]), 1e-6),
        }
    return derived


def decide_capacity(report: dict[str, Any]) -> dict[str, Any]:
    tier_names = ["tier_b_dense_teacher", "real_ppr10k_target_c", "real_fivek_mmart_like"]
    pass_by_config = {}
    for config in [c.name for c in SHAPE_CONFIGS if c.name != "C0-Main"]:
        tier_passes = []
        for tier in tier_names:
            derived = report["tiers"][tier]["derived"][config]
            metrics = report["tiers"][tier]["configs"][config]["metrics"]
            tier_passes.append(
                derived["gap_closed_deltaE"] >= 0.6
                and derived["gap_to_D4_deltaE"] <= 1.0
                and metrics["clipping_ratio"] <= 0.01
                and metrics["gamut_violation_ratio"] <= 0.01
            )
        pass_by_config[config] = all(tier_passes)
    if pass_by_config.get("C1-M16-R4-K10"):
        conclusion = "CAPACITY_PASS_BASELINE"
    elif pass_by_config.get("C5-M32-R8-K12"):
        conclusion = "CAPACITY_PASS_HIGH_CAPACITY"
    else:
        conclusion = "CAPACITY_FAIL_PIVOT_DENSE_RESIDUAL"
    return {"capacity_conclusion": conclusion, "pass_by_config": pass_by_config}


def _stable_sample(items: list[Any], count: int, seed: int) -> list[Any]:
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    if len(items) < count:
        repeats = (count + len(items) - 1) // max(1, len(items))
        items = (items * repeats)[:count]
    return items[:count]


def _chunks(items: list[Any], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run M1.3 capacity ablation.")
    parser.add_argument("--data-root", default="~/retouching/monetGPT/data")
    parser.add_argument("--output", default="m1_results/m1_3_capacity_20260525.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tier-a-count", type=int, default=500)
    parser.add_argument("--tier-b-count", type=int, default=500)
    parser.add_argument("--fivek-count", type=int, default=100)
    parser.add_argument("--ppr10k-count", type=int, default=100)
    parser.add_argument("--fit-steps", type=int, default=80)
    parser.add_argument("--fit-lr", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.tier_a_count = 8
        args.tier_b_count = 8
        args.fivek_count = 4
        args.ppr10k_count = 4
        args.batch_size = min(args.batch_size, 4)
        args.fit_steps = min(args.fit_steps, 4)
    result = run_capacity(
        Path(args.output),
        Path(args.data_root).expanduser(),
        args.device,
        args.image_size,
        args.batch_size,
        args.tier_a_count,
        args.tier_b_count,
        args.fivek_count,
        args.ppr10k_count,
        args.seed,
        DynamicFitHyper(steps=args.fit_steps, lr=args.fit_lr),
    )
    print(result["decision"]["capacity_conclusion"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
