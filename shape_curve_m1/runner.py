from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random
import time
from typing import Any

import numpy as np
import torch

from .data import DataRequirements, PairRecord, discover_m1_data
from .fit import A1_NO_REG_FULL, ALL_FIT_CONFIGS, D4_DENSE, FIT_FULL, FIT_MAIN, fit_batch
from .image_io import load_image
from .metrics import MetricAccumulator
from .render import build_dictionary, render_hybrid, lut_stats
from .synthetic import TIER_B_FAMILIES, decode_raw_batch, render_dense_teacher, sample_filtered_raw_actions
from .tiny_sft import TinySFTConfig, run_tiny_sft


@dataclass(frozen=True)
class M1RunConfig:
    data_root: Path
    output: Path
    device: str
    image_size: int
    batch_size: int
    fit_steps: int
    fit_lr: float
    tier_a_count: int
    tier_b_count: int
    fivek_count: int
    ppr10k_count: int
    tiny_train_count: int
    tiny_val_count: int
    tiny_epochs: int
    seed: int
    smoke: bool = False


def run_m1(config: M1RunConfig) -> dict[str, Any]:
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() or not config.device.startswith("cuda") else "cpu")
    discovery = discover_m1_data(config.data_root, DataRequirements())
    if not discovery.ready:
        result = {
            "M1_CONCLUSION": "BLOCKED",
            "reason": "required M1 data is missing",
            "data": discovery.to_dict(),
            "config": _json_config(config),
            "started_at": started,
        }
        _write_result(config.output, result)
        return result

    try:
        import lpips

        lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    except Exception as exc:  # noqa: BLE001
        result = {
            "M1_CONCLUSION": "BLOCKED",
            "reason": f"LPIPS model could not be initialized: {type(exc).__name__}: {exc}",
            "data": discovery.to_dict(),
            "config": _json_config(config),
            "started_at": started,
        }
        _write_result(config.output, result)
        return result

    dictionary, rho, atom_names = build_dictionary(device=device)
    report: dict[str, Any] = {
        "M1_CONCLUSION": "BLOCKED",
        "started_at": started,
        "config": _json_config(config),
        "device": str(device),
        "data": discovery.to_dict(),
        "atom_names": list(atom_names),
        "tiers": {},
    }

    base_paths = _stable_sample(list(discovery.base_pool), config.tier_a_count + config.tier_b_count + config.tiny_train_count + config.tiny_val_count, config.seed)
    tier_a_paths = base_paths[: config.tier_a_count]
    tier_b_paths = base_paths[config.tier_a_count : config.tier_a_count + config.tier_b_count]
    tiny_train_paths = base_paths[config.tier_a_count + config.tier_b_count : config.tier_a_count + config.tier_b_count + config.tiny_train_count]
    tiny_val_paths = base_paths[-config.tiny_val_count :]

    report["tiers"]["renderer_aligned_synthetic"] = _run_renderer_aligned_tier(
        tier_a_paths, config, dictionary, rho, lpips_model, device
    )
    report["a_sanity_checks"] = evaluate_tier_a_sanity(report["tiers"]["renderer_aligned_synthetic"])
    report["thresholds"] = {
        "A_renderer_aligned": evaluate_tier_a_threshold(report["tiers"]["renderer_aligned_synthetic"])
    }
    if not report["thresholds"]["A_renderer_aligned"]["pass"]:
        report["M1_CONCLUSION"] = conclusion_from_thresholds(report["thresholds"])
        report["fallback_decision"] = fallback_decision(report)
        report.update(protocol_interpretation(report))
        report["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _write_result(config.output, report)
        return report
    _write_result(config.output, report)

    report["tiers"]["off_manifold_dense_teacher"] = _run_dense_teacher_tier(
        tier_b_paths, config, dictionary, rho, lpips_model, device
    )
    _write_result(config.output, report)

    fivek_pairs = _stable_sample(list(discovery.fivek_pairs), config.fivek_count, config.seed + 3)
    ppr_pairs = _stable_sample(list(discovery.ppr10k_pairs), config.ppr10k_count, config.seed + 4)
    report["tiers"]["real_fivek_expert_c"] = _run_real_tier(fivek_pairs, config, dictionary, rho, lpips_model, device)
    _write_result(config.output, report)
    report["tiers"]["real_ppr10k_target_c"] = _run_real_tier(ppr_pairs, config, dictionary, rho, lpips_model, device)
    _write_result(config.output, report)

    tiny_train = _load_path_batch(tiny_train_paths, config.image_size, device)
    tiny_val = _load_path_batch(tiny_val_paths, config.image_size, device)
    report["tiny_sft"] = run_tiny_sft(
        tiny_train,
        tiny_val,
        TinySFTConfig(
            train_count=config.tiny_train_count,
            val_count=config.tiny_val_count,
            epochs=config.tiny_epochs,
            batch_size=config.batch_size,
            image_size=config.image_size,
            seed=config.seed,
        ),
        lpips_model,
    )
    report["thresholds"] = evaluate_thresholds(report)
    report["M1_CONCLUSION"] = conclusion_from_thresholds(report["thresholds"])
    report["fallback_decision"] = fallback_decision(report)
    report.update(protocol_interpretation(report))
    report["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _write_result(config.output, report)
    return report


def _json_config(config: M1RunConfig) -> dict[str, Any]:
    data = asdict(config)
    data["data_root"] = str(config.data_root)
    data["output"] = str(config.output)
    return data


def _write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


def _stable_sample(items: list[Any], count: int, seed: int) -> list[Any]:
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    if len(items) < count:
        repeats = (count + len(items) - 1) // max(1, len(items))
        items = (items * repeats)[:count]
    return items[:count]


def _load_path_batch(paths: list[Path], image_size: int, device: torch.device) -> torch.Tensor:
    images = [load_image(path, image_size) for path in paths]
    return torch.stack(images, dim=0).to(device)


def _load_pair_batch(pairs: list[PairRecord], image_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    source = torch.stack([load_image(pair.source, image_size) for pair in pairs], dim=0).to(device)
    target = torch.stack([load_image(pair.target, image_size) for pair in pairs], dim=0).to(device)
    return source, target


def _run_renderer_aligned_tier(
    paths: list[Path],
    config: M1RunConfig,
    dictionary: torch.Tensor,
    rho: torch.Tensor,
    lpips_model,
    device: torch.device,
) -> dict[str, Any]:
    fit_configs = (*ALL_FIT_CONFIGS, A1_NO_REG_FULL)
    acc = {fit.name: MetricAccumulator.create() for fit in fit_configs}
    gt_acc = MetricAccumulator.create()
    a0_acc = MetricAccumulator.create()
    sampling_rows: list[dict[str, Any]] = []
    count = 0
    for batch_idx, sl_paths in enumerate(_chunks(paths, config.batch_size)):
        source = _load_path_batch(sl_paths, config.image_size, device)
        raw, sample_summary = sample_filtered_raw_actions(
            source.shape[0],
            config.seed + 10_000 + batch_idx,
            device,
            dictionary,
            rho,
        )
        sampling_rows.append(sample_summary)
        action = decode_raw_batch(raw)
        with torch.no_grad():
            target, luts = render_hybrid(source, action, dictionary, rho, True, True)
            replay, _ = render_hybrid(source, action, dictionary, rho, True, True)
            replay_error = (replay - target).abs()
            replay_mse = replay_error.pow(2).mean(dim=(1, 2, 3))
            a0_acc.extend_tensor("MSE", replay_mse)
            a0_acc.extend_tensor("PSNR", 10.0 * torch.log10(1.0 / replay_mse.clamp_min(1e-12)))
            a0_acc.extend_tensor("max_abs_error", replay_error.amax(dim=(1, 2, 3)))
            gt_stats = lut_stats(action, luts)
            for key, value in gt_stats.items():
                gt_acc.extend_tensor(key, value)
        for fit in fit_configs:
            _, metrics = fit_batch(source, target, fit, dictionary, rho, lpips_model, config.fit_steps, config.fit_lr)
            _add_metrics(acc[fit.name], metrics)
        count += source.shape[0]
    return {
        "count": count,
        "configs": {name: acc[name].summary() for name in acc},
        "gt_action_stats": gt_acc.summary(),
        "a0_gt_replay": a0_acc.summary(),
        "sampling": _summarize_sampling(sampling_rows),
    }


def _run_dense_teacher_tier(
    paths: list[Path],
    config: M1RunConfig,
    dictionary: torch.Tensor,
    rho: torch.Tensor,
    lpips_model,
    device: torch.device,
) -> dict[str, Any]:
    acc = {fit.name: MetricAccumulator.create() for fit in ALL_FIT_CONFIGS}
    count = 0
    families: dict[str, int] = {name: 0 for name in TIER_B_FAMILIES}
    for batch_idx, sl_paths in enumerate(_chunks(paths, config.batch_size)):
        source = _load_path_batch(sl_paths, config.image_size, device)
        family = TIER_B_FAMILIES[batch_idx % len(TIER_B_FAMILIES)]
        seeds = [config.seed + 20_000 + batch_idx * config.batch_size + i for i in range(source.shape[0])]
        target, _ = render_dense_teacher(source, family, seeds, g=17)
        families[family] += source.shape[0]
        for fit in ALL_FIT_CONFIGS:
            _, metrics = fit_batch(source, target, fit, dictionary, rho, lpips_model, config.fit_steps, config.fit_lr)
            _add_metrics(acc[fit.name], metrics)
        count += source.shape[0]
    return {
        "count": count,
        "teacher_family_counts": families,
        "configs": {name: acc[name].summary() for name in acc},
    }


def _run_real_tier(
    pairs: list[PairRecord],
    config: M1RunConfig,
    dictionary: torch.Tensor,
    rho: torch.Tensor,
    lpips_model,
    device: torch.device,
) -> dict[str, Any]:
    acc = {fit.name: MetricAccumulator.create() for fit in ALL_FIT_CONFIGS}
    count = 0
    for sl_pairs in _chunks(pairs, config.batch_size):
        source, target = _load_pair_batch(sl_pairs, config.image_size, device)
        for fit in ALL_FIT_CONFIGS:
            _, metrics = fit_batch(source, target, fit, dictionary, rho, lpips_model, config.fit_steps, config.fit_lr)
            _add_metrics(acc[fit.name], metrics)
        count += source.shape[0]
    return {
        "count": count,
        "configs": {name: acc[name].summary() for name in acc},
    }


def _chunks(items: list[Any], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _add_metrics(acc: MetricAccumulator, metrics: dict[str, torch.Tensor]) -> None:
    for key, value in metrics.items():
        acc.extend_tensor(key, value)


def _summarize_sampling(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    out: dict[str, Any] = {"policy": rows[0].get("policy", {})}
    numeric_keys = sorted(key for key, value in rows[0].items() if isinstance(value, int | float))
    total_requested = sum(int(row.get("requested", 0)) for row in rows)
    total_attempted = sum(int(row.get("attempted", 0)) for row in rows)
    for key in numeric_keys:
        if key in {"requested", "attempted", "accepted", "acceptance_rate"}:
            continue
        vals = [float(row[key]) for row in rows if key in row]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    out["requested"] = total_requested
    out["attempted"] = total_attempted
    out["accepted"] = sum(int(row.get("accepted", 0)) for row in rows)
    out["acceptance_rate"] = total_requested / max(1, total_attempted)
    return out


def _metric(tier: dict[str, Any], config_name: str, key: str) -> float:
    return float(tier["configs"][config_name][key])


def evaluate_thresholds(report: dict[str, Any]) -> dict[str, Any]:
    tiers = report["tiers"]
    out: dict[str, Any] = {}

    tier_a = tiers["renderer_aligned_synthetic"]
    out["A_renderer_aligned"] = evaluate_tier_a_threshold(tier_a)

    tier_b = tiers["off_manifold_dense_teacher"]
    main_b = tier_b["configs"][FIT_MAIN.name]
    full_b = tier_b["configs"][FIT_FULL.name]
    dense_b = tier_b["configs"][D4_DENSE.name]
    out["B_off_manifold"] = {
        "pass": bool(
            dense_b["PSNR"] - full_b["PSNR"] <= 1.0
            and full_b["mean_deltaE2000"] - dense_b["mean_deltaE2000"] <= 1.0
            and full_b["LPIPS"] - dense_b["LPIPS"] <= 0.03
            and full_b["PSNR"] - main_b["PSNR"] >= 0.5
            and _relative_drop(main_b["LPIPS"], full_b["LPIPS"]) >= 0.10
            and _relative_drop(main_b["mean_deltaE2000"], full_b["mean_deltaE2000"]) >= 0.15
        ),
        "full_vs_dense": {
            "psnr_gap": dense_b["PSNR"] - full_b["PSNR"],
            "deltaE_gap": full_b["mean_deltaE2000"] - dense_b["mean_deltaE2000"],
            "lpips_gap": full_b["LPIPS"] - dense_b["LPIPS"],
        },
        "full_vs_main": {
            "psnr_gain": full_b["PSNR"] - main_b["PSNR"],
            "lpips_drop": _relative_drop(main_b["LPIPS"], full_b["LPIPS"]),
            "deltaE_drop": _relative_drop(main_b["mean_deltaE2000"], full_b["mean_deltaE2000"]),
        },
    }

    real_checks = {}
    for key in ("real_fivek_expert_c", "real_ppr10k_target_c"):
        tier = tiers[key]
        main = tier["configs"][FIT_MAIN.name]
        full = tier["configs"][FIT_FULL.name]
        dense = tier["configs"][D4_DENSE.name]
        gap_closed = (main["mean_deltaE2000"] - full["mean_deltaE2000"]) / max(
            main["mean_deltaE2000"] - dense["mean_deltaE2000"], 1e-6
        )
        real_checks[key] = {
            "pass": bool(
                full["PSNR"] - main["PSNR"] >= 0.3
                and _relative_drop(main["LPIPS"], full["LPIPS"]) >= 0.05
                and _relative_drop(main["mean_deltaE2000"], full["mean_deltaE2000"]) >= 0.10
                and gap_closed >= 0.6
            ),
            "psnr_gain": full["PSNR"] - main["PSNR"],
            "lpips_drop": _relative_drop(main["LPIPS"], full["LPIPS"]),
            "deltaE_drop": _relative_drop(main["mean_deltaE2000"], full["mean_deltaE2000"]),
            "gap_closed_deltaE": gap_closed,
        }
    out["C_real_probe"] = {"pass": all(item["pass"] for item in real_checks.values()), "tiers": real_checks}

    interp_rows = [tiers[name]["configs"][FIT_FULL.name] for name in tiers]
    interp = _mean_rows(interp_rows)
    free_ratio = interp["free_tail_energy"] / max(interp["dictionary_energy"] + interp["free_tail_energy"], 1e-6)
    residual_eff = np.mean(
        [
            tiers[name]["configs"][FIT_FULL.name]["PSNR"] - tiers[name]["configs"][FIT_MAIN.name]["PSNR"]
            for name in tiers
        ]
    )
    out["D_interpretability"] = {
        "pass": bool(
            interp["dictionary_explained_ratio"] >= 0.5
            and free_ratio <= 0.4
            and 2.0 <= interp["active_atom_count"] <= 6.0
            and interp["inactive_atom_ratio"] >= 0.4
            and residual_eff > 0.0
        ),
        "dictionary_explained_ratio": interp["dictionary_explained_ratio"],
        "free_tail_energy_ratio": free_ratio,
        "active_atom_count": interp["active_atom_count"],
        "inactive_atom_ratio": interp["inactive_atom_ratio"],
        "residual_efficiency": float(residual_eff),
    }

    tiny = report.get("tiny_sft", {}).get("summary", {})
    out["E_tiny_sft"] = {
        "pass": bool(
            tiny.get("LPIPS", float("inf")) <= 0.10
            and tiny.get("mean_deltaE2000", float("inf")) <= 2.0
            and tiny.get("curve_hsl_wb_l1", float("inf")) <= 0.08
            and tiny.get("dict_coef_mae", float("inf")) <= 0.10
            and tiny.get("free_tail_grid_loss_decreased", False)
            and tiny.get("no_single_loss_dominates", False)
        ),
        "metrics": tiny,
    }
    return out


def evaluate_tier_a_threshold(tier_a: dict[str, Any]) -> dict[str, Any]:
    full_a = tier_a["configs"][FIT_FULL.name]
    a_sanity = evaluate_tier_a_sanity(tier_a)
    return {
        "pass": bool(
            a_sanity["pass"]
            and full_a["PSNR"] >= 38.0
            and full_a["LPIPS"] <= 0.05
            and full_a["mean_deltaE2000"] <= 1.5
            and full_a["p95_deltaE2000"] <= 4.0
            and full_a["clipping_ratio"] < 0.01
        ),
        "metrics": full_a,
        "sanity": a_sanity,
    }


def evaluate_tier_a_sanity(tier_a: dict[str, Any]) -> dict[str, Any]:
    gt = tier_a.get("gt_action_stats", {})
    a0 = tier_a.get("a0_gt_replay", {})
    a1 = tier_a.get("configs", {}).get(A1_NO_REG_FULL.name, {})
    a2 = tier_a.get("configs", {}).get(FIT_FULL.name, {})
    checks = {
        "gt_gate_clean": bool(
            gt.get("clipping_ratio", float("inf")) <= 0.01
            and gt.get("gamut_violation_ratio", float("inf")) <= 0.01
            and 2.0 <= gt.get("active_atom_count", float("inf")) <= 6.0
            and 1e-8 <= gt.get("free_tail_energy", float("inf")) <= 0.003
        ),
        "a0_gt_replay": bool(a0.get("max_abs_error", float("inf")) <= 1e-6),
        "a1_no_reg_inverse": bool(
            a1.get("PSNR", 0.0) >= 38.0
            and a1.get("LPIPS", float("inf")) <= 0.05
            and a1.get("mean_deltaE2000", float("inf")) <= 1.5
        ),
        "a2_regularized_inverse": bool(
            a2.get("PSNR", 0.0) >= 38.0
            and a2.get("LPIPS", float("inf")) <= 0.05
            and a2.get("mean_deltaE2000", float("inf")) <= 1.5
            and a2.get("clipping_ratio", float("inf")) < 0.01
        ),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "gt_action_stats": gt,
        "A0_GT_replay": a0,
        "A1_no_reg_inverse": a1,
        "A2_regularized_inverse": a2,
    }


def conclusion_from_thresholds(thresholds: dict[str, Any]) -> str:
    if not thresholds:
        return "BLOCKED"
    passes = {key: bool(value.get("pass")) for key, value in thresholds.items()}
    if all(passes.values()):
        return "PASS"
    if passes.get("A_renderer_aligned") and any(passes.values()):
        return "PARTIAL"
    return "FAIL"


def fallback_decision(report: dict[str, Any]) -> dict[str, str]:
    conclusion = report.get("M1_CONCLUSION", "BLOCKED")
    thresholds = report.get("thresholds", {})
    if conclusion == "PASS":
        return {
            "category": "PASS",
            "decision": "允许进入 M2 hidden-layer probe 与 Gaussian policy credit assignment test。",
        }
    if not thresholds:
        return {
            "category": "BLOCKED",
            "decision": "实验未真正运行或缺少阈值证据；不得进入 M2。",
        }
    if not thresholds.get("A_renderer_aligned", {}).get("pass", False):
        gt_issue = _tier_a_gt_gate_issue(report)
        if gt_issue:
            return {
                "category": "A_renderer_aligned_failed",
                "failure_scope": "implementation_protocol",
                "primary_protocol_issue": gt_issue,
                "architecture_claim": "not_established",
                "decision": "Tier A GT action already violates clipping/gamut gate; keep M1_CONCLUSION=FAIL, stop VLM/SFT/GRPO, fix renderer-aligned generator and A0/A1/A2 before making any ShapeCurve architecture claim.",
            }
        sanity_issue = _tier_a_sanity_issue(report)
        if sanity_issue:
            return {
                "category": "A_renderer_aligned_failed",
                "failure_scope": "implementation_protocol",
                "primary_protocol_issue": sanity_issue,
                "architecture_claim": "not_established",
                "decision": "Tier A clean GT is established, but A0/A1/A2 sanity did not close; stop VLM/SFT/GRPO and repair inverse fitting or renderer protocol before full M1.",
            }
        return {
            "category": "A_renderer_aligned_failed",
            "failure_scope": "implementation_protocol",
            "architecture_claim": "not_established",
            "decision": "判定 renderer / scale / loss / softclip / identity / gate / B-spline 实现错误，停止 VLM/SFT/GRPO，先修 unit test；当前失败不得解释为 action space 架构级失败。",
        }

    tiers = report.get("tiers", {})
    b_pass = thresholds.get("B_off_manifold", {}).get("pass", False)
    c_pass = thresholds.get("C_real_probe", {}).get("pass", False)
    e_pass = thresholds.get("E_tiny_sft", {}).get("pass", False)
    dict_gain = _mean_full_minus_dict_gain(tiers)
    full_gain = _mean_full_minus_main_gain(tiers)

    if (not b_pass or not c_pass) and _dense_upper_bound_is_stronger(tiers):
        return {
            "category": "expression_capacity_failed",
            "decision": "A 通过但 B/C 失败且 D4-Dense 明显更强，判定 action space 表达力不足；回退顺序：dictionary 16→32，R_free 4→8，K 10→12，加入 technical atoms；仍失败则 pivot 到 basis-LUT/dense-LUT residual renderer。",
        }
    if dict_gain <= 0.05 and full_gain > 0.1:
        return {
            "category": "dictionary_ineffective_free_tail_dominates",
            "decision": "Fit-Dict 无有效增益但 Fit-Full 有增益，判定 dictionary 无效、free_tail 吞噬 residual；重设计 atoms、降低 g_max、提高 free_tail efficiency penalty，或降级 interpretability claim。",
        }
    if b_pass and c_pass and not e_pass:
        return {
            "category": "tiny_sft_failed",
            "decision": "inverse fitting 通过但 Tiny-SFT 失败，判定 action space 可行但 decoder/condition/loss 不行；进入 layer probe、color_stats、deterministic SFT、分阶段训练和 gradient audit。",
        }
    return {
        "category": "partial_or_mixed_failure",
        "decision": "未满足 M1 放行条件；按最先失败阈值定位，优先检查 B/C 表达力与 dictionary/free_tail 解释性指标。",
    }


def protocol_interpretation(report: dict[str, Any]) -> dict[str, str]:
    conclusion = report.get("M1_CONCLUSION", "BLOCKED")
    if conclusion == "PASS":
        return {
            "failure_scope": "none",
            "architecture_level_conclusion": "action_space_passed_m1",
            "next_action": "enter M2 hidden-layer probe and Gaussian policy credit assignment test",
        }
    if conclusion == "BLOCKED":
        return {
            "failure_scope": "blocked",
            "architecture_level_conclusion": "not_established",
            "next_action": "resolve missing data or runtime dependency before interpreting ShapeCurve architecture",
        }

    gt_issue = _tier_a_gt_gate_issue(report)
    if gt_issue:
        return {
            "failure_scope": "implementation_protocol",
            "architecture_level_conclusion": "not_established",
            "primary_protocol_issue": gt_issue,
            "next_action": "fix Tier A rejection sampling, inverse-fit ablations, free_tail stress, and data audit before M2",
        }
    sanity_issue = _tier_a_sanity_issue(report)
    if sanity_issue:
        return {
            "failure_scope": "implementation_protocol",
            "architecture_level_conclusion": "not_established",
            "primary_protocol_issue": sanity_issue,
            "next_action": "repair A0/A1/A2 inverse-fit closure before running B/C/D/E or entering M2",
        }

    fallback = report.get("fallback_decision", {})
    if fallback.get("category") == "expression_capacity_failed":
        return {
            "failure_scope": "architecture_capacity_candidate",
            "architecture_level_conclusion": "candidate_only_after_A_passed",
            "next_action": "run dictionary/free_tail/K upgrades and compare against D4-Dense before pivoting renderer",
        }
    return {
        "failure_scope": "implementation_protocol",
        "architecture_level_conclusion": "not_established",
        "next_action": "stop VLM/SFT/GRPO and repair M1 protocol before architecture-level conclusions",
    }


def _tier_a_gt_gate_issue(report: dict[str, Any]) -> str:
    gt = report.get("tiers", {}).get("renderer_aligned_synthetic", {}).get("gt_action_stats", {})
    clipping = float(gt.get("clipping_ratio", 0.0))
    gamut = float(gt.get("gamut_violation_ratio", 0.0))
    issues = []
    if clipping > 0.01:
        issues.append(f"GT clipping_ratio={clipping:.5f} exceeds 0.01000 gate")
    if gamut > 0.01:
        issues.append(f"GT gamut_violation_ratio={gamut:.5f} exceeds 0.01000 gate")
    return "; ".join(issues)


def _tier_a_sanity_issue(report: dict[str, Any]) -> str:
    sanity = report.get("a_sanity_checks") or report.get("thresholds", {}).get("A_renderer_aligned", {}).get("sanity", {})
    checks = sanity.get("checks", {})
    failed = [key for key, value in checks.items() if not value]
    if not failed:
        return ""
    return "Tier A sanity failed: " + ", ".join(failed)


def _relative_drop(before: float, after: float) -> float:
    return (before - after) / max(abs(before), 1e-6)


def _mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _mean_full_minus_main_gain(tiers: dict[str, Any]) -> float:
    gains = []
    for tier in tiers.values():
        configs = tier.get("configs", {})
        if FIT_FULL.name in configs and FIT_MAIN.name in configs:
            gains.append(configs[FIT_FULL.name]["PSNR"] - configs[FIT_MAIN.name]["PSNR"])
    return float(np.mean(gains)) if gains else 0.0


def _mean_full_minus_dict_gain(tiers: dict[str, Any]) -> float:
    gains = []
    for tier in tiers.values():
        configs = tier.get("configs", {})
        if FIT_FULL.name in configs and "Fit-Dict" in configs:
            gains.append(configs[FIT_FULL.name]["PSNR"] - configs["Fit-Dict"]["PSNR"])
    return float(np.mean(gains)) if gains else 0.0


def _dense_upper_bound_is_stronger(tiers: dict[str, Any]) -> bool:
    stronger = []
    for tier in tiers.values():
        configs = tier.get("configs", {})
        if D4_DENSE.name in configs and FIT_FULL.name in configs:
            stronger.append(configs[D4_DENSE.name]["PSNR"] > configs[FIT_FULL.name]["PSNR"])
    return bool(stronger and all(stronger))
