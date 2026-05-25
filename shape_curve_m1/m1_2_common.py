from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random
import time
from typing import Any, Iterator

import numpy as np
import torch

from .action import decode_raw_action
from .data import DataRequirements, discover_m1_data
from .image_io import load_image
from .metrics import MetricAccumulator
from .render import build_dictionary, render_hybrid, lut_stats
from .synthetic import sample_filtered_raw_actions


A2_THRESHOLDS = {
    "PSNR": 38.0,
    "LPIPS": 0.05,
    "mean_deltaE2000": 1.5,
    "p95_deltaE2000": 4.0,
    "clipping_ratio": 0.01,
}


@dataclass(frozen=True)
class M12DataConfig:
    data_root: Path
    count: int = 128
    image_size: int = 64
    batch_size: int = 16
    seed: int = 20260525
    device: str = "cuda:0"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["data_root"] = str(self.data_root)
        return data


@dataclass
class CleanTierAData:
    source: torch.Tensor
    target: torch.Tensor
    raw: torch.Tensor
    gt_pre_lut: torch.Tensor
    gt_final_lut: torch.Tensor
    gt_stats: dict[str, float]
    sampling: dict[str, Any]
    atom_names: tuple[str, ...]
    dictionary: torch.Tensor
    rho: torch.Tensor


def select_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def load_lpips_model(device: torch.device):
    import lpips

    return lpips.LPIPS(net="alex").to(device).eval()


def load_clean_tier_a(config: M12DataConfig) -> CleanTierAData:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = select_device(config.device)
    discovery = discover_m1_data(config.data_root, DataRequirements())
    if not discovery.ready:
        raise RuntimeError("M1 data discovery is not ready")
    paths = _stable_sample(list(discovery.base_pool), config.count, config.seed)
    dictionary, rho, atom_names = build_dictionary(device=device)

    sources: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    raws: list[torch.Tensor] = []
    gt_pre: list[torch.Tensor] = []
    gt_final: list[torch.Tensor] = []
    sampling_rows: list[dict[str, Any]] = []
    gt_acc = MetricAccumulator.create()

    for batch_idx, sl_paths in enumerate(_chunks(paths, config.batch_size)):
        source = torch.stack([load_image(path, config.image_size) for path in sl_paths], dim=0).to(device)
        raw, sample_summary = sample_filtered_raw_actions(
            source.shape[0],
            config.seed + 10_000 + batch_idx,
            device,
            dictionary,
            rho,
        )
        action = decode_raw_action(raw)
        with torch.no_grad():
            target, luts = render_hybrid(source, action, dictionary, rho, True, True)
            stats = lut_stats(action, luts)
        for key, value in stats.items():
            gt_acc.extend_tensor(key, value)
        sources.append(source.detach())
        targets.append(target.detach())
        raws.append(raw.detach())
        gt_pre.append(luts["pre"].detach())
        gt_final.append(luts["final"].detach())
        sampling_rows.append(sample_summary)

    return CleanTierAData(
        source=torch.cat(sources, dim=0),
        target=torch.cat(targets, dim=0),
        raw=torch.cat(raws, dim=0),
        gt_pre_lut=torch.cat(gt_pre, dim=0),
        gt_final_lut=torch.cat(gt_final, dim=0),
        gt_stats=gt_acc.summary(),
        sampling=summarize_sampling(sampling_rows),
        atom_names=atom_names,
        dictionary=dictionary,
        rho=rho,
    )


def dataset_batches(data: CleanTierAData, batch_size: int) -> Iterator[dict[str, torch.Tensor]]:
    count = data.source.shape[0]
    for start in range(0, count, batch_size):
        sl = slice(start, min(start + batch_size, count))
        yield {
            "source": data.source[sl],
            "target": data.target[sl],
            "raw": data.raw[sl],
            "gt_pre_lut": data.gt_pre_lut[sl],
            "gt_final_lut": data.gt_final_lut[sl],
        }


def summarize_metric_tensors(rows: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    acc = MetricAccumulator.create()
    for row in rows:
        for key, value in row.items():
            acc.extend_tensor(key, value)
    return acc.summary()


def passes_a2(metrics: dict[str, float]) -> bool:
    return bool(
        metrics.get("PSNR", 0.0) >= A2_THRESHOLDS["PSNR"]
        and metrics.get("LPIPS", float("inf")) <= A2_THRESHOLDS["LPIPS"]
        and metrics.get("mean_deltaE2000", float("inf")) <= A2_THRESHOLDS["mean_deltaE2000"]
        and metrics.get("p95_deltaE2000", float("inf")) <= A2_THRESHOLDS["p95_deltaE2000"]
        and metrics.get("clipping_ratio", float("inf")) <= A2_THRESHOLDS["clipping_ratio"]
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def now_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def summarize_sampling(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    out: dict[str, Any] = {"policy": rows[0].get("policy", {})}
    total_requested = sum(int(row.get("requested", 0)) for row in rows)
    total_attempted = sum(int(row.get("attempted", 0)) for row in rows)
    total_accepted = sum(int(row.get("accepted", 0)) for row in rows)
    numeric_keys = sorted(key for key, value in rows[0].items() if isinstance(value, int | float))
    for key in numeric_keys:
        if key in {"requested", "attempted", "accepted", "acceptance_rate"}:
            continue
        vals = [float(row[key]) for row in rows if key in row]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    out["requested"] = total_requested
    out["attempted"] = total_attempted
    out["accepted"] = total_accepted
    out["acceptance_rate"] = total_accepted / max(1, total_attempted)
    return out


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
