from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class DataRequirements:
    renderer_synthetic_min: int = 500
    off_manifold_min: int = 500
    fivek_real_min: int = 100
    ppr10k_real_min: int = 100


@dataclass(frozen=True)
class PairRecord:
    sample_id: str
    source: Path
    target: Path
    tier: str

    def to_dict(self) -> dict[str, str]:
        return {
            "sample_id": self.sample_id,
            "source": str(self.source),
            "target": str(self.target),
            "tier": self.tier,
        }


@dataclass(frozen=True)
class TierStatus:
    name: str
    required: int
    available: int
    path: Path
    ready: bool
    missing_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "required": self.required,
            "available": self.available,
            "path": str(self.path),
            "ready": self.ready,
            "missing_reason": self.missing_reason,
        }


@dataclass(frozen=True)
class DataDiscoveryReport:
    data_root: Path
    statuses: tuple[TierStatus, ...]
    fivek_pairs: tuple[PairRecord, ...]
    ppr10k_pairs: tuple[PairRecord, ...]
    base_pool: tuple[Path, ...]

    @property
    def ready(self) -> bool:
        return all(status.ready for status in self.statuses)

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_root": str(self.data_root),
            "ready": self.ready,
            "statuses": [status.to_dict() for status in self.statuses],
            "counts": {
                "fivek_pairs": len(self.fivek_pairs),
                "ppr10k_pairs": len(self.ppr10k_pairs),
                "base_pool": len(self.base_pool),
            },
            "examples": {
                "fivek_pairs": [p.to_dict() for p in self.fivek_pairs[:3]],
                "ppr10k_pairs": [p.to_dict() for p in self.ppr10k_pairs[:3]],
                "base_pool": [str(p) for p in self.base_pool[:3]],
            },
        }

    def format_text(self) -> str:
        lines = [
            f"data_root: {self.data_root}",
            f"ready: {self.ready}",
            "",
            "tiers:",
        ]
        for status in self.statuses:
            marker = "OK" if status.ready else "MISSING"
            reason = f" ({status.missing_reason})" if status.missing_reason else ""
            lines.append(
                f"- {status.name}: {marker} {status.available}/{status.required} at {status.path}{reason}"
            )
        return "\n".join(lines)


def _image_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _discover_fivek_pairs(data_root: Path) -> list[PairRecord]:
    root = data_root / "fivek_mmart_like"
    pairs: list[PairRecord] = []
    for split in ("test_global", "train_global"):
        split_root = root / split
        if not split_root.exists():
            continue
        for sample_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
            source = sample_dir / "before.jpg"
            target = sample_dir / "processed.jpg"
            if source.exists() and target.exists():
                pairs.append(
                    PairRecord(
                        sample_id=sample_dir.name,
                        source=source,
                        target=target,
                        tier="fivek_expert_c",
                    )
                )
    return pairs


def _discover_ppr10k_pairs(data_root: Path, target_name: str = "target_c") -> list[PairRecord]:
    root = data_root / "ppr10k"
    source_root = root / "source"
    target_root = root / target_name
    pairs: list[PairRecord] = []
    if not source_root.exists() or not target_root.exists():
        return pairs
    for source in _image_files(source_root):
        target = target_root / source.name
        if target.exists():
            pairs.append(
                PairRecord(
                    sample_id=source.stem,
                    source=source,
                    target=target,
                    tier=f"ppr10k_{target_name}",
                )
            )
    return pairs


def _base_pool(data_root: Path) -> list[Path]:
    ppr_sources = _image_files(data_root / "ppr10k" / "source")
    fivek_before = [
        sample_dir / "before.jpg"
        for split in ("test_global", "train_global")
        for sample_dir in sorted((data_root / "fivek_mmart_like" / split).glob("*"))
        if sample_dir.is_dir() and (sample_dir / "before.jpg").exists()
    ]
    return sorted(ppr_sources + fivek_before)


def _status(name: str, required: int, available: int, path: Path) -> TierStatus:
    ready = available >= required
    reason = "" if ready else f"requires at least {required}, found {available}"
    if not path.exists():
        reason = f"path does not exist: {path}"
    return TierStatus(name=name, required=required, available=available, path=path, ready=ready, missing_reason=reason)


def discover_m1_data(data_root: Path, requirements: DataRequirements) -> DataDiscoveryReport:
    data_root = data_root.resolve()
    fivek_pairs = _discover_fivek_pairs(data_root)
    ppr10k_pairs = _discover_ppr10k_pairs(data_root)
    base_pool = _base_pool(data_root)

    statuses = (
        _status(
            "renderer_aligned_synthetic_base_pool",
            requirements.renderer_synthetic_min,
            len(base_pool),
            data_root,
        ),
        _status(
            "off_manifold_dense_teacher_base_pool",
            requirements.off_manifold_min,
            len(base_pool),
            data_root,
        ),
        _status(
            "fivek_expert_c_real_probe",
            requirements.fivek_real_min,
            len(fivek_pairs),
            data_root / "fivek_mmart_like",
        ),
        _status(
            "ppr10k_expert_target_c_real_probe",
            requirements.ppr10k_real_min,
            len(ppr10k_pairs),
            data_root / "ppr10k",
        ),
    )
    return DataDiscoveryReport(
        data_root=data_root,
        statuses=statuses,
        fivek_pairs=tuple(fivek_pairs),
        ppr10k_pairs=tuple(ppr10k_pairs),
        base_pool=tuple(base_pool),
    )
