from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
import yaml
from box import Box
from transformers import AutoTokenizer

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, TASK_AUTO_RETOUCH_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import process_images_, tokenizer_image_token
from llava.model.VeraRetouch import VeraRetouchForCausalLLM_Unified
from llava.utils import disable_torch_init

from .action import action_main_l1, decode_raw_action, identity_raw, raw_layout
from .data import DataRequirements, discover_m1_data
from .m1_2_common import M12DataConfig, load_clean_tier_a, load_lpips_model, now_stamp, select_device, write_json
from .m1_3_capacity import (
    DenseFitConfig,
    DynamicFitHyper,
    DynamicShapeConfig,
    build_dynamic_dictionary,
    dynamic_lut_stats,
    fit_dense_dataset,
    fit_dynamic_dataset,
    load_pairs,
    load_tier_b,
    render_dynamic,
    summarize,
)
from .metrics import MetricAccumulator, metric_summary
from .render import build_dictionary, gamut_penalty, smoothness2_3d


@dataclass(frozen=True)
class VlmProbeConfig:
    hidden_layers: tuple[int, ...] = (16, 18, 20)
    resolutions: tuple[int, ...] = (256, 512)
    feature_batch_size: int = 4
    train_batch_size: int = 64
    epochs: int = 3
    lr: float = 2e-3
    weight_decay: float = 1e-4
    image_size: int = 64
    fit_steps: int = 80


@dataclass
class TierTensors:
    name: str
    source: torch.Tensor
    target: torch.Tensor
    raw: torch.Tensor | None = None


class C5ActionHead(nn.Module):
    def __init__(self, hidden_dim: int, config: DynamicShapeConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.main_dim = raw_layout().dim
        self.dict_dim = config.m_atoms
        self.tail_gate_dim = config.r_free
        self.tail_color_dim = config.r_free * 3
        self.tail_factor_dim = config.r_free * config.k_spline
        self.out_dim = self.main_dim + self.dict_dim + self.tail_gate_dim + self.tail_color_dim + 3 * self.tail_factor_dim
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, self.out_dim),
        )
        self._reset_parameters()
        self._init_identity_bias(device)

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _init_identity_bias(self, device: torch.device) -> None:
        bias = torch.zeros(self.out_dim, device=device)
        start = 0
        bias[start : start + self.main_dim] = identity_raw(1, device).squeeze(0)
        start += self.main_dim + self.dict_dim
        bias[start : start + self.tail_gate_dim] = -3.0
        with torch.no_grad():
            self.net[-1].bias.copy_(bias)
            self.net[-1].weight.mul_(0.01)

    def split(self, out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        start = 0
        raw_main = out[:, start : start + self.main_dim]
        start += self.main_dim
        dict_raw = out[:, start : start + self.dict_dim]
        start += self.dict_dim
        tail_gate_raw = out[:, start : start + self.tail_gate_dim]
        start += self.tail_gate_dim
        tail_color_raw = out[:, start : start + self.tail_color_dim].view(out.shape[0], self.config.r_free, 3)
        start += self.tail_color_dim
        tail_alpha = out[:, start : start + self.tail_factor_dim].view(out.shape[0], self.config.r_free, self.config.k_spline)
        start += self.tail_factor_dim
        tail_beta = out[:, start : start + self.tail_factor_dim].view(out.shape[0], self.config.r_free, self.config.k_spline)
        start += self.tail_factor_dim
        tail_gamma = out[:, start : start + self.tail_factor_dim].view(out.shape[0], self.config.r_free, self.config.k_spline)
        return raw_main, dict_raw, tail_gate_raw, tail_color_raw, tail_alpha, tail_beta, tail_gamma

    def forward(self, features: torch.Tensor):
        return self.split(self.net(features))


def run_m1_4b(
    output: Path,
    data_root: Path,
    models_root: Path,
    plan_path: Path | None,
    device_name: str,
    seed: int,
    tier_a_train: int,
    tier_a_val: int,
    tier_b_train: int,
    tier_b_val: int,
    ppr10k_train: int,
    ppr10k_val: int,
    fivek_train: int,
    fivek_val: int,
    probe: VlmProbeConfig,
) -> dict[str, Any]:
    started = now_stamp()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = select_device(device_name)
    lpips_model = load_lpips_model(device)
    discovery = discover_m1_data(data_root, DataRequirements())
    if not discovery.ready:
        raise RuntimeError("M1 data discovery is not ready")

    dictionary16, rho16, _ = build_dictionary(device=device)
    c5 = DynamicShapeConfig("C5-M32-R8-K12", 32, 8, 12)
    atoms, rho = build_dynamic_dictionary(dictionary16, rho16, c5, seed, device)
    shape_hyper = DynamicFitHyper(steps=probe.fit_steps, lr=0.07, gamut_l1_weight=0.5)
    d4 = DenseFitConfig("D4-33-midreg", 33, probe.fit_steps, 0.05, 0.02, 0.005, 1.0)

    train_tiers, val_tiers, data_notes = build_tiers(
        discovery,
        data_root,
        device,
        seed,
        probe.image_size,
        probe.train_batch_size,
        tier_a_train,
        tier_a_val,
        tier_b_train,
        tier_b_val,
        ppr10k_train,
        ppr10k_val,
        fivek_train,
        fivek_val,
    )

    report: dict[str, Any] = {
        "experiment": "M1.4b_VLM_HIDDEN_SFT_SANITY",
        "started_at": started,
        "plan_status": plan_status(plan_path),
        "config": {
            "data_root": str(data_root),
            "models_root": str(models_root),
            "device": str(device),
            "seed": seed,
            "probe": asdict(probe),
            "tier_counts": {
                "tier_a_train": tier_a_train,
                "tier_a_val": tier_a_val,
                "tier_b_train": tier_b_train,
                "tier_b_val": tier_b_val,
                "ppr10k_train": ppr10k_train,
                "ppr10k_val": ppr10k_val,
                "fivek_train": fivek_train,
                "fivek_val": fivek_val,
            },
            "shape_config": asdict(c5),
            "shape_hyper": asdict(shape_hyper),
            "d4": asdict(d4),
        },
        "data_notes": data_notes,
        "method_notes": [
            "VeraRetouch backbone is frozen; only a small C5 ShapeCurve action head is trained.",
            "The VLM input is a before/after paired panel so the random synthetic action is conditionally inferable.",
            "This is a condition-representation sanity probe, not a source-only deployment claim.",
            "VeraRetouch image processor pads panels and feeds MobileCLIP at its configured 1024px input; 256 vs 512 changes pre-panel detail before that processor.",
        ],
        "oracles": {},
        "color_stats_archetype_proxy": {},
        "vlm_hidden_sft": {},
        "historical_tiny_cnn_proxy": load_historical_tiny_proxy(Path("m1_results/full_m1_20260525.json")),
    }
    write_json(output, {**report, "complete": False, "ended_at": None})

    for tier in val_tiers:
        c5_metrics = fit_dynamic_dataset(tier.source, tier.target, c5, shape_hyper, atoms, rho, lpips_model, probe.train_batch_size)
        d4_metrics = fit_dense_dataset(tier.source, tier.target, d4, lpips_model, probe.train_batch_size)
        report["oracles"][tier.name] = {
            c5.name: {"metrics": c5_metrics, "config": asdict(c5)},
            d4.name: {"metrics": d4_metrics, "config": asdict(d4)},
        }
        write_json(output, {**report, "complete": False, "ended_at": None})

    color_train_features = extract_color_stats_features(train_tiers)
    color_val_features = extract_color_stats_features(val_tiers)
    color_train = train_head_for_features(
        train_tiers,
        color_train_features,
        c5,
        atoms,
        rho,
        probe,
        device,
    )
    color_eval = evaluate_head(
        val_tiers,
        color_val_features,
        color_train["head"],
        c5,
        atoms,
        rho,
        lpips_model,
        probe.train_batch_size,
    )
    report["color_stats_archetype_proxy"] = {
        "feature": "RGB/HSV/luma summary stats on before, after, and delta plus tier/archetype one-hot",
        "train_history": color_train["history"],
        "metrics_by_tier": color_eval,
        "derived": derive_probe_metrics(color_eval, report["oracles"]),
    }
    write_json(output, {**report, "complete": False, "ended_at": None})

    vlm, tokenizer = load_vera_vlm(models_root, device)
    prompt_ids = build_prompt_ids(tokenizer, device)

    for resolution in probe.resolutions:
        train_features = extract_features_for_tiers(
            vlm,
            tokenizer,
            prompt_ids,
            train_tiers,
            resolution,
            probe.hidden_layers,
            probe.feature_batch_size,
            device,
        )
        val_features = extract_features_for_tiers(
            vlm,
            tokenizer,
            prompt_ids,
            val_tiers,
            resolution,
            probe.hidden_layers,
            probe.feature_batch_size,
            device,
        )
        for layer in probe.hidden_layers:
            key = f"res{resolution}_L{layer}"
            train_result = train_head_for_features(
                train_tiers,
                train_features[layer],
                c5,
                atoms,
                rho,
                probe,
                device,
            )
            eval_result = evaluate_head(
                val_tiers,
                val_features[layer],
                train_result["head"],
                c5,
                atoms,
                rho,
                lpips_model,
                probe.train_batch_size,
            )
            report["vlm_hidden_sft"][key] = {
                "resolution": resolution,
                "hidden_layer": layer,
                "train_history": train_result["history"],
                "metrics_by_tier": eval_result,
            }
            report["vlm_hidden_sft"][key]["derived"] = derive_probe_metrics(eval_result, report["oracles"])
            write_json(output, {**report, "complete": False, "ended_at": None})

    report["decision"] = decide(report)
    report["complete"] = True
    report["ended_at"] = now_stamp()
    write_json(output, report)
    return report


def build_tiers(
    discovery,
    data_root: Path,
    device: torch.device,
    seed: int,
    image_size: int,
    batch_size: int,
    tier_a_train: int,
    tier_a_val: int,
    tier_b_train: int,
    tier_b_val: int,
    ppr10k_train: int,
    ppr10k_val: int,
    fivek_train: int,
    fivek_val: int,
) -> tuple[list[TierTensors], list[TierTensors], list[str]]:
    tier_a = load_clean_tier_a(M12DataConfig(data_root, tier_a_train + tier_a_val, image_size, batch_size, seed, str(device)))
    tier_b_source, tier_b_target = load_tier_b(discovery, tier_b_train + tier_b_val, image_size, batch_size, seed + 20_000, device)
    ppr_pairs = _stable_sample(list(discovery.ppr10k_pairs), ppr10k_train + ppr10k_val, seed + 4)
    fivek_pairs = _stable_sample(list(discovery.fivek_pairs), fivek_train + fivek_val, seed + 3)
    ppr_source, ppr_target = load_pairs(ppr_pairs, image_size, device)
    fivek_source, fivek_target = load_pairs(fivek_pairs, image_size, device)

    train = [
        TierTensors("tier_a_clean", tier_a.source[:tier_a_train], tier_a.target[:tier_a_train], tier_a.raw[:tier_a_train]),
        TierTensors("tier_b_dense_teacher", tier_b_source[:tier_b_train], tier_b_target[:tier_b_train]),
        TierTensors("real_ppr10k_target_c", ppr_source[:ppr10k_train], ppr_target[:ppr10k_train]),
        TierTensors("real_fivek_mmart_like", fivek_source[:fivek_train], fivek_target[:fivek_train]),
    ]
    val = [
        TierTensors("tier_a_clean", tier_a.source[tier_a_train:], tier_a.target[tier_a_train:], tier_a.raw[tier_a_train:]),
        TierTensors("tier_b_dense_teacher", tier_b_source[tier_b_train:], tier_b_target[tier_b_train:]),
        TierTensors("real_ppr10k_target_c", ppr_source[ppr10k_train:], ppr_target[ppr10k_train:]),
        TierTensors("real_fivek_mmart_like", fivek_source[fivek_train:], fivek_target[fivek_train:]),
    ]
    notes = [
        "Tier A uses clean renderer-aligned synthetic actions from the corrected rejection-sampled generator.",
        "Tier B uses off-manifold dense-teacher pairs.",
        "Tier C real probes keep the existing names from M1.3; fivek_mmart_like is not treated as strict FiveK Expert C.",
    ]
    return train, val, notes


def load_vera_vlm(models_root: Path, device: torch.device):
    model_path = models_root / "VeraRetouch"
    with open("configs/infer_config.yaml", "r", encoding="utf-8") as f:
        config_add = Box(yaml.safe_load(f))
    config_add.project_name = "m1_4b_vlm_hidden_sft"
    config_add.freeze_retouch_decoder = True
    disable_torch_init()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        cache_dir="./cache",
        model_max_length=4096,
        padding_side="right",
        use_fast=False,
    )
    model = VeraRetouchForCausalLLM_Unified.from_pretrained(
        str(model_path),
        config_add=config_add,
        cache_dir="./cache",
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer


def build_prompt_ids(tokenizer, device: torch.device) -> torch.Tensor:
    question = (
        f"{DEFAULT_IMAGE_TOKEN}\n{TASK_AUTO_RETOUCH_TOKEN}\n"
        "The image is a before/after panel: left is the source photo and right is the target retouch. "
        "Infer the retouch action that transforms the left image into the right image."
    )
    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    return tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").to(device)


def extract_features_for_tiers(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    tiers: list[TierTensors],
    resolution: int,
    layers: tuple[int, ...],
    batch_size: int,
    device: torch.device,
) -> dict[int, dict[str, torch.Tensor]]:
    del tokenizer
    features: dict[int, dict[str, torch.Tensor]] = {layer: {} for layer in layers}
    image_processor = model.get_vision_tower().image_processor
    for tier in tiers:
        tier_rows = {layer: [] for layer in layers}
        for start in range(0, tier.source.shape[0], batch_size):
            sl = slice(start, min(start + batch_size, tier.source.shape[0]))
            panels = [make_panel(tier.source[i], tier.target[i], resolution) for i in range(sl.start, sl.stop)]
            images = process_images_(panels, image_processor).to(device=device, dtype=torch.bfloat16)
            input_ids = prompt_ids.unsqueeze(0).repeat(len(panels), 1)
            attention = torch.ones_like(input_ids)
            with torch.inference_mode():
                _, position_ids, attention_mask, _, inputs_embeds, _ = model.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    None,
                    attention,
                    None,
                    None,
                    images,
                    image_sizes=[panel.size for panel in panels],
                    batch_infer=True,
                )
                out = model.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )
            mask = attention_mask.to(dtype=torch.float32).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            for layer in layers:
                pooled = (out.hidden_states[layer].float() * mask).sum(dim=1) / denom
                tier_rows[layer].append(pooled.detach().cpu())
        for layer in layers:
            features[layer][tier.name] = torch.cat(tier_rows[layer], dim=0)
    return features


def extract_color_stats_features(tiers: list[TierTensors]) -> dict[str, torch.Tensor]:
    out = {}
    tier_count = len(tiers)
    for tier_idx, tier in enumerate(tiers):
        one_hot = F.one_hot(
            torch.full((tier.source.shape[0],), tier_idx, dtype=torch.long, device=tier.source.device),
            num_classes=tier_count,
        ).float()
        stats = torch.cat(
            [
                image_stats(tier.source),
                image_stats(tier.target),
                image_stats((tier.target - tier.source + 1.0) * 0.5),
                one_hot,
            ],
            dim=1,
        )
        out[tier.name] = stats.detach().cpu()
    return out


def image_stats(image: torch.Tensor) -> torch.Tensor:
    rgb = image.clamp(0, 1)
    hsv = rgb_to_hsv_image(rgb)
    luma = (0.2126 * rgb[:, 0:1] + 0.7152 * rgb[:, 1:2] + 0.0722 * rgb[:, 2:3]).clamp(0, 1)
    tensors = [rgb, hsv, luma]
    rows = []
    for item in tensors:
        flat = item.flatten(2)
        rows.extend(
            [
                flat.mean(dim=2),
                flat.std(dim=2),
                flat.amin(dim=2),
                flat.amax(dim=2),
                flat.quantile(0.1, dim=2),
                flat.quantile(0.5, dim=2),
                flat.quantile(0.9, dim=2),
            ]
        )
    return torch.cat(rows, dim=1)


def rgb_to_hsv_image(rgb: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc = torch.maximum(torch.maximum(r, g), b)
    minc = torch.minimum(torch.minimum(r, g), b)
    delta = maxc - minc
    eps = 1e-6
    hue_r = ((g - b) / (delta + eps)) % 6.0
    hue_g = ((b - r) / (delta + eps)) + 2.0
    hue_b = ((r - g) / (delta + eps)) + 4.0
    hue = torch.where(maxc == r, hue_r, torch.where(maxc == g, hue_g, hue_b))
    hue = torch.where(delta < eps, torch.zeros_like(hue), hue / 6.0)
    sat = torch.where(maxc < eps, torch.zeros_like(maxc), delta / (maxc + eps))
    return torch.stack([hue, sat, maxc], dim=1)


def make_panel(source: torch.Tensor, target: torch.Tensor, resolution: int) -> Image.Image:
    src = to_pil_image(source.detach().cpu().clamp(0, 1))
    tgt = to_pil_image(target.detach().cpu().clamp(0, 1))
    src = src.resize((resolution, resolution), Image.Resampling.BICUBIC)
    tgt = tgt.resize((resolution, resolution), Image.Resampling.BICUBIC)
    panel = Image.new("RGB", (resolution * 2, resolution))
    panel.paste(src, (0, 0))
    panel.paste(tgt, (resolution, 0))
    return panel


def train_head_for_features(
    tiers: list[TierTensors],
    features: dict[str, torch.Tensor],
    config: DynamicShapeConfig,
    atoms: torch.Tensor,
    rho: torch.Tensor,
    probe: VlmProbeConfig,
    device: torch.device,
) -> dict[str, Any]:
    x = torch.cat([features[tier.name] for tier in tiers], dim=0).to(device)
    source = torch.cat([tier.source for tier in tiers], dim=0)
    target = torch.cat([tier.target for tier in tiers], dim=0)
    tier_ids = torch.cat([
        torch.full((tier.source.shape[0],), idx, dtype=torch.long, device=device) for idx, tier in enumerate(tiers)
    ])
    raw = torch.cat([
        tier.raw if tier.raw is not None else torch.zeros(tier.source.shape[0], raw_layout().dim, device=device)
        for tier in tiers
    ], dim=0)
    head = C5ActionHead(x.shape[1], config, device).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=probe.lr, weight_decay=probe.weight_decay)
    history = []
    for epoch in range(probe.epochs):
        perm = torch.randperm(x.shape[0], device=device)
        rows = []
        for start in range(0, x.shape[0], probe.train_batch_size):
            idx = perm[start : start + probe.train_batch_size]
            raw_main, dict_raw, tail_gate_raw, tail_color_raw, tail_alpha, tail_beta, tail_gamma = head(x[idx])
            pred, luts, aux_action = render_dynamic(
                source[idx],
                raw_main,
                dict_raw,
                tail_gate_raw,
                tail_color_raw,
                tail_alpha,
                tail_beta,
                tail_gamma,
                config,
                atoms,
                rho,
            )
            render_loss = F.mse_loss(pred, target[idx]) + 0.15 * F.l1_loss(pred, target[idx])
            loss = render_loss + 0.005 * smoothness2_3d(luts["final"]).mean()
            loss = loss + 4.0 * gamut_penalty(luts["pre"]) + 0.5 * gamut_l1_penalty(luts["pre"])
            a_mask = tier_ids[idx] == 0
            action_loss = torch.zeros((), device=device)
            if bool(a_mask.any()):
                pred_action = decode_raw_action(raw_main[a_mask])
                gt_action = decode_raw_action(raw[idx][a_mask])
                action_loss = action_main_l1(pred_action, gt_action) + 0.25 * F.l1_loss(
                    torch.tanh(dict_raw[a_mask, : gt_action.dict_coef.shape[1]]),
                    gt_action.dict_coef,
                )
                loss = loss + 0.5 * action_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            opt.step()
            rows.append({
                "loss": float(loss.detach().cpu()),
                "render_loss": float(render_loss.detach().cpu()),
                "action_loss": float(action_loss.detach().cpu()),
            })
        history.append({key: float(np.mean([row[key] for row in rows])) for key in rows[0]})
    return {"head": head, "history": history}


def evaluate_head(
    tiers: list[TierTensors],
    features: dict[str, torch.Tensor],
    head: C5ActionHead,
    config: DynamicShapeConfig,
    atoms: torch.Tensor,
    rho: torch.Tensor,
    lpips_model,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    out = {}
    device = atoms.device
    head.eval()
    for tier in tiers:
        rows = []
        x = features[tier.name].to(device)
        with torch.no_grad():
            for start in range(0, tier.source.shape[0], batch_size):
                sl = slice(start, min(start + batch_size, tier.source.shape[0]))
                pred, luts, aux_action = render_dynamic(
                    tier.source[sl],
                    *head(x[sl]),
                    config,
                    atoms,
                    rho,
                )
                rows.append(metric_summary(pred, tier.target[sl], dynamic_lut_stats(aux_action, luts), lpips_model))
        out[tier.name] = summarize(rows)
    return out


def derive_probe_metrics(metrics_by_tier: dict[str, dict[str, float]], oracles: dict[str, Any]) -> dict[str, Any]:
    derived = {}
    for tier, metrics in metrics_by_tier.items():
        c5 = oracles[tier]["C5-M32-R8-K12"]["metrics"]
        d4 = oracles[tier]["D4-33-midreg"]["metrics"]
        derived[tier] = {
            "gap_to_c5_deltaE": metrics["mean_deltaE2000"] - c5["mean_deltaE2000"],
            "gap_to_c5_lpips": metrics["LPIPS"] - c5["LPIPS"],
            "gap_to_d4_deltaE": metrics["mean_deltaE2000"] - d4["mean_deltaE2000"],
            "gap_to_d4_lpips": metrics["LPIPS"] - d4["LPIPS"],
            "exceeds_c5_deltaE": metrics["mean_deltaE2000"] < c5["mean_deltaE2000"] - 0.2,
            "approaches_c5": metrics["mean_deltaE2000"] <= c5["mean_deltaE2000"] + 0.5 and metrics["LPIPS"] <= c5["LPIPS"] + 0.02,
        }
    return derived


def decide(report: dict[str, Any]) -> dict[str, Any]:
    best_key, best_score = None, float("inf")
    for key, row in report["vlm_hidden_sft"].items():
        vals = [m["mean_deltaE2000"] for m in row["metrics_by_tier"].values()]
        if not np.all(np.isfinite(vals)):
            continue
        score = float(np.mean(vals))
        if score < best_score:
            best_key, best_score = key, score
    if best_key is None:
        return {
            "conclusion": "VLM_SFT_NO_GAIN",
            "best_config": None,
            "best_mean_deltaE2000_across_tiers": float("nan"),
            "note": "No finite VLM-SFT metric row was produced; inspect training history and renderer numerics.",
        }
    best = report["vlm_hidden_sft"][best_key]
    derived = best["derived"]
    approaches = {tier: bool(row["approaches_c5"]) for tier, row in derived.items()}
    exceeds = {tier: bool(row["exceeds_c5_deltaE"]) for tier, row in derived.items()}
    tiny = report.get("historical_tiny_cnn_proxy", {})
    tier_a = best["metrics_by_tier"].get("tier_a_clean", {})
    improves_tiny = bool(
        tiny
        and tier_a
        and tier_a.get("mean_deltaE2000", float("inf")) < tiny.get("mean_deltaE2000", float("inf"))
        and tier_a.get("LPIPS", float("inf")) < tiny.get("LPIPS", float("inf"))
    )
    color_proxy = report.get("color_stats_archetype_proxy", {}).get("metrics_by_tier", {})
    improves_color_proxy = bool(
        color_proxy
        and all(
            best["metrics_by_tier"][tier]["mean_deltaE2000"] < metrics["mean_deltaE2000"]
            for tier, metrics in color_proxy.items()
        )
    )
    res_scores: dict[int, float] = {}
    for key, row in report["vlm_hidden_sft"].items():
        res_scores.setdefault(row["resolution"], []).append(
            float(np.mean([m["mean_deltaE2000"] for m in row["metrics_by_tier"].values()]))
        )
    res_summary = {str(res): float(np.min(vals)) for res, vals in res_scores.items()}
    res512_better = res_summary.get("512", float("inf")) + 0.05 < res_summary.get("256", float("inf"))
    if any(exceeds.values()):
        conclusion = "VLM_SFT_EXCEEDS_C5_ORACLE_RECHECK_FITTING"
    elif all(approaches.values()):
        conclusion = "VLM_SFT_APPROACHES_C5_ORACLE"
    elif improves_tiny:
        conclusion = "VLM_SFT_CONFIRMS_PROXY_WEAKNESS"
    else:
        conclusion = "VLM_SFT_NO_GAIN"
    return {
        "conclusion": conclusion,
        "best_config": best_key,
        "best_mean_deltaE2000_across_tiers": best_score,
        "approaches_c5_by_tier": approaches,
        "exceeds_c5_by_tier": exceeds,
        "improves_historical_tiny_on_tier_a": improves_tiny,
        "improves_color_stats_archetype_proxy_all_tiers": improves_color_proxy,
        "resolution_best_mean_deltaE2000": res_summary,
        "res512_better_than_256": res512_better,
        "interpretation": (
            "If VLM-SFT remains below or near C5, it can weaken the Tiny-CNN proxy concern but does not overturn the C5-vs-D4 capacity result. "
            "If it exceeds C5, inspect inverse fitting, renderer consistency, and leakage before making an architecture claim."
        ),
    }


def gamut_l1_penalty(lut_pre: torch.Tensor) -> torch.Tensor:
    return F.relu(lut_pre - 1.0).mean() + F.relu(-lut_pre).mean()


def plan_status(plan_path: Path | None) -> dict[str, Any]:
    candidates = []
    if plan_path is not None:
        candidates.append(plan_path)
    candidates.extend([Path("plan/m1-3b.md"), Path("/home/bc/VeraRetouch/plan/m1-3b.md")])
    for path in candidates:
        path = path.expanduser()
        if path.exists():
            text = path.read_text(encoding="utf-8")
            return {
                "plan_path": str(path),
                "found": True,
                "bytes": len(text.encode("utf-8")),
                "note": "plan/m1-3b.md is ignored by .gitignore, so this artifact records the resolved external/local plan path.",
            }
    return {"plan_path": str(plan_path) if plan_path else "plan/m1-3b.md", "found": False}


def load_historical_tiny_proxy(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("tiny_sft", {}).get("summary", {})


def _stable_sample(items: list[Any], count: int, seed: int) -> list[Any]:
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    if len(items) < count:
        repeats = (count + len(items) - 1) // max(1, len(items))
        items = (items * repeats)[:count]
    return items[:count]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run M1.4b VeraRetouch hidden SFT sanity probe.")
    parser.add_argument("--data-root", default="~/retouching/monetGPT/data")
    parser.add_argument("--models-root", default="~/data/models")
    parser.add_argument("--plan-path", default=None)
    parser.add_argument("--output", default="m1_results/m1_4b_vlm_hidden_sft_20260525.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--fit-steps", type=int, default=80)
    parser.add_argument("--hidden-layers", default="16,18,20")
    parser.add_argument("--resolutions", default="256,512")
    parser.add_argument("--tier-a-train", type=int, default=5000)
    parser.add_argument("--tier-a-val", type=int, default=500)
    parser.add_argument("--tier-b-train", type=int, default=2000)
    parser.add_argument("--tier-b-val", type=int, default=500)
    parser.add_argument("--ppr10k-train", type=int, default=500)
    parser.add_argument("--ppr10k-val", type=int, default=100)
    parser.add_argument("--fivek-train", type=int, default=500)
    parser.add_argument("--fivek-val", type=int, default=100)
    parser.add_argument("--smoke", action="store_true")
    return parser


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.tier_a_train = 16
        args.tier_a_val = 8
        args.tier_b_train = 8
        args.tier_b_val = 8
        args.ppr10k_train = 8
        args.ppr10k_val = 4
        args.fivek_train = 8
        args.fivek_val = 4
        args.epochs = 1
        args.fit_steps = min(args.fit_steps, 4)
        args.hidden_layers = "16"
        args.resolutions = "256"
    probe = VlmProbeConfig(
        hidden_layers=_ints(args.hidden_layers),
        resolutions=_ints(args.resolutions),
        feature_batch_size=args.feature_batch_size,
        train_batch_size=args.train_batch_size,
        epochs=args.epochs,
        image_size=args.image_size,
        fit_steps=args.fit_steps,
    )
    result = run_m1_4b(
        Path(args.output),
        Path(args.data_root).expanduser(),
        Path(args.models_root).expanduser(),
        Path(args.plan_path).expanduser() if args.plan_path else None,
        args.device,
        args.seed,
        args.tier_a_train,
        args.tier_a_val,
        args.tier_b_train,
        args.tier_b_val,
        args.ppr10k_train,
        args.ppr10k_val,
        args.fivek_train,
        args.fivek_val,
        probe,
    )
    print(result["decision"]["conclusion"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
