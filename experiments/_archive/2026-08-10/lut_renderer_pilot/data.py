from __future__ import annotations

import random
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import Dataset, Sampler

from .common import derive_seed, read_jsonl


@dataclass(frozen=True)
class LutRecord:
    style_index: int
    preset_id: str
    content_hash: str
    path: str
    grid_size: int
    domain_min: tuple[float, float, float]
    domain_max: tuple[float, float, float]
    taxonomy_major: str
    taxonomy_minor: str

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "LutRecord":
        return cls(
            style_index=int(row["style_index"]),
            preset_id=str(row["preset_id"]),
            content_hash=str(row["lut_content_hash"]),
            path=str(row["path"]),
            grid_size=int(row["grid_size"]),
            domain_min=tuple(float(value) for value in row["domain_min"]),
            domain_max=tuple(float(value) for value in row["domain_max"]),
            taxonomy_major=str(row["taxonomy_major"]),
            taxonomy_minor=str(row["taxonomy_minor"]),
        )


def load_lut_manifest(path: str | Path) -> list[LutRecord]:
    records = [LutRecord.from_dict(row) for row in read_jsonl(path)]
    if [record.style_index for record in records] != list(range(len(records))):
        raise ValueError("LUT manifest style_index must be contiguous and ordered")
    return records


class PackedLutStore:
    """Lazy process-local cache over the uncompressed canonical NPZ bank."""

    def __init__(self, npz_path: str | Path, records: Sequence[LutRecord]) -> None:
        self.path = Path(npz_path)
        self.records = records
        self._archive = np.load(self.path, allow_pickle=False)
        self._cache: dict[int, np.ndarray] = {}

    def get(self, style_index: int) -> np.ndarray:
        cached = self._cache.get(style_index)
        if cached is not None:
            return cached
        record = self.records[style_index]
        grid = np.ascontiguousarray(self._archive[record.preset_id], dtype=np.float32)
        expected = (record.grid_size, record.grid_size, record.grid_size, 3)
        if grid.shape != expected:
            raise ValueError(
                f"packed grid shape mismatch for {record.preset_id}: {grid.shape} != {expected}"
            )
        self._cache[style_index] = grid
        return grid

    def close(self) -> None:
        self._archive.close()


def resize_source(image: Image.Image, short_edge: int, long_edge: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    scale = short_edge / min(width, height)
    if max(width, height) * scale > long_edge:
        scale = long_edge / max(width, height)
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    if target != image.size:
        image = image.resize(target, Image.Resampling.LANCZOS)
    return image


def load_source_tensor(path: str | Path, short_edge: int, long_edge: int) -> torch.Tensor:
    with Image.open(path) as image:
        resized = resize_source(image, short_edge, long_edge)
        pixels = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(pixels)).permute(2, 0, 1)


class OnlinePairDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        sources: Sequence[dict[str, Any]],
        luts: Sequence[LutRecord],
        *,
        short_edge: int,
        long_edge: int,
    ) -> None:
        self.sources = sources
        self.luts = luts
        self.short_edge = short_edge
        self.long_edge = long_edge

    def __len__(self) -> int:
        return len(self.luts)

    def __getitem__(self, index: tuple[int, int]) -> dict[str, Any]:
        style_index, source_index = index
        source = self.sources[source_index]
        return {
            "image": load_source_tensor(
                source["path"], self.short_edge, self.long_edge
            ),
            "style_index": style_index,
            "source_id": source["source_id"],
            "source_cluster": source["source_cluster"],
            "source_path": source["path"],
        }


class OnlinePairSampler(Sampler[tuple[int, int]]):
    def __init__(
        self,
        num_styles: int,
        num_sources: int,
        *,
        base_seed: int,
        epoch: int,
    ) -> None:
        self.num_styles = num_styles
        self.num_sources = num_sources
        self.base_seed = base_seed
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_styles

    def __iter__(self) -> Iterator[tuple[int, int]]:
        style_rng = random.Random(derive_seed(self.base_seed, "style_order", self.epoch))
        source_rng = random.Random(derive_seed(self.base_seed, "source_order", self.epoch))
        styles = list(range(self.num_styles))
        style_rng.shuffle(styles)
        for style_index in styles:
            yield style_index, source_rng.randrange(self.num_sources)


class OnlinePairBatchSampler(Sampler[list[tuple[int, int]]]):
    """Group the online-random pairs by source aspect ratio without dropping styles."""

    def __init__(
        self,
        sources: Sequence[dict[str, Any]],
        num_styles: int,
        batch_size: int,
        *,
        base_seed: int,
        epoch: int,
    ) -> None:
        self.sources = sources
        self.num_styles = num_styles
        self.batch_size = batch_size
        self.base_seed = base_seed
        self.epoch = epoch

    def __len__(self) -> int:
        return math.ceil(self.num_styles / self.batch_size)

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        pairs = list(
            OnlinePairSampler(
                self.num_styles,
                len(self.sources),
                base_seed=self.base_seed,
                epoch=self.epoch,
            )
        )
        pairs.sort(
            key=lambda pair: (
                float(self.sources[pair[1]]["width"])
                / float(self.sources[pair[1]]["height"]),
                pair[0],
            )
        )
        batches = [
            pairs[start : start + self.batch_size]
            for start in range(0, len(pairs), self.batch_size)
        ]
        batch_rng = random.Random(
            derive_seed(self.base_seed, "aspect_bucket_order", self.epoch)
        )
        batch_rng.shuffle(batches)
        yield from batches


class FixedPairDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        pairs: Sequence[dict[str, Any]],
        *,
        short_edge: int,
        long_edge: int,
    ) -> None:
        self.pairs = pairs
        self.short_edge = short_edge
        self.long_edge = long_edge

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        return {
            "image": load_source_tensor(
                pair["source_path"], self.short_edge, self.long_edge
            ),
            "style_index": int(pair["style_index"]),
            **pair,
        }


def collate_images(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot collate an empty batch")
    max_height = max(row["image"].shape[1] for row in rows)
    max_width = max(row["image"].shape[2] for row in rows)
    images = torch.zeros((len(rows), 3, max_height, max_width), dtype=torch.float32)
    valid = torch.zeros((len(rows), 1, max_height, max_width), dtype=torch.bool)
    sizes: list[tuple[int, int]] = []
    for index, row in enumerate(rows):
        image = row["image"]
        height, width = image.shape[1:]
        images[index, :, :height, :width] = image
        valid[index, :, :height, :width] = True
        sizes.append((height, width))
    metadata = [{key: value for key, value in row.items() if key != "image"} for row in rows]
    return {
        "images": images,
        "valid": valid,
        "sizes": sizes,
        "style_indices": torch.tensor(
            [int(row["style_index"]) for row in rows], dtype=torch.long
        ),
        "metadata": metadata,
    }


def render_lut_batch(
    images: torch.Tensor,
    style_indices: torch.Tensor,
    records: Sequence[LutRecord],
    store: PackedLutStore,
) -> torch.Tensor:
    """Apply variable-resolution LUTs using x=R, y=G, z=B grid coordinates."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must be BCHW RGB")
    output = torch.empty_like(images)
    groups: dict[int, list[int]] = {}
    for batch_index, style_index in enumerate(style_indices.tolist()):
        groups.setdefault(records[style_index].grid_size, []).append(batch_index)

    for _, batch_indices in sorted(groups.items()):
        group_styles = [int(style_indices[index]) for index in batch_indices]
        volume = torch.stack(
            [
                torch.from_numpy(store.get(style_index)).permute(3, 0, 1, 2)
                for style_index in group_styles
            ]
        ).to(device=images.device, dtype=images.dtype, non_blocking=True)
        selected = images[batch_indices]
        domain_min = torch.tensor(
            [records[index].domain_min for index in group_styles],
            device=images.device,
            dtype=images.dtype,
        )[:, :, None, None]
        domain_max = torch.tensor(
            [records[index].domain_max for index in group_styles],
            device=images.device,
            dtype=images.dtype,
        )[:, :, None, None]
        span = torch.where(domain_max == domain_min, torch.ones_like(domain_max), domain_max - domain_min)
        coordinates = ((selected - domain_min) / span).clamp(0.0, 1.0)
        coordinates = coordinates.permute(0, 2, 3, 1).mul(2.0).sub(1.0).unsqueeze(1)
        rendered = F.grid_sample(
            volume,
            coordinates,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(2)
        output[batch_indices] = rendered.clamp(0.0, 1.0)
    return output


def native_lattice(
    record: LutRecord,
    grid: np.ndarray,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    size = record.grid_size
    red = torch.linspace(record.domain_min[0], record.domain_max[0], size, device=device)
    green = torch.linspace(record.domain_min[1], record.domain_max[1], size, device=device)
    blue = torch.linspace(record.domain_min[2], record.domain_max[2], size, device=device)
    b_coord, g_coord, r_coord = torch.meshgrid(blue, green, red, indexing="ij")
    inputs = torch.stack((r_coord, g_coord, b_coord), dim=-1).reshape(-1, 3)
    targets = torch.from_numpy(grid).to(device=device, dtype=torch.float32).reshape(-1, 3)
    return inputs, targets
