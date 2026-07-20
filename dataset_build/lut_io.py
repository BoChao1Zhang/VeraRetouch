"""Strict on-disk 3D LUT parsing shared by GPU renderers and tooling."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def load_lut(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(grid, domain_min, domain_max)`` with ``grid[b, g, r]``."""
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix == ".cube":
        rows, size, domain_min, domain_max = _read_cube(target)
        grid = _rows_to_grid(rows, size, red_fastest=True)
    elif suffix == ".3dl":
        rows, size = _read_3dl(target)
        maximum = max((max(row) for row in rows), default=1.0)
        divisor = 1023.0 if maximum > 255 else (255.0 if maximum > 1 else 1.0)
        grid = _rows_to_grid(
            ((r / divisor, g / divisor, b / divisor) for r, g, b in rows),
            size,
            red_fastest=False,
        )
        domain_min = (0.0, 0.0, 0.0)
        domain_max = (1.0, 1.0, 1.0)
    else:
        raise ValueError(f"unsupported LUT extension {suffix!r}: {target}")
    if not np.isfinite(grid).all():
        raise ValueError(f"non-finite LUT value: {target}")
    return (
        np.ascontiguousarray(grid, dtype=np.float32),
        np.asarray(domain_min, dtype=np.float32),
        np.asarray(domain_max, dtype=np.float32),
    )


def _read_cube(
    path: Path,
) -> tuple[list[tuple[float, float, float]], int, tuple[float, ...], tuple[float, ...]]:
    size: int | None = None
    domain_min = (0.0, 0.0, 0.0)
    domain_max = (1.0, 1.0, 1.0)
    rows: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            parts = text.split()
            keyword = parts[0].upper()
            if keyword == "TITLE":
                continue
            if keyword == "LUT_3D_SIZE":
                if len(parts) != 2:
                    raise ValueError(f"invalid LUT_3D_SIZE at {path}:{line_number}")
                size = int(parts[1])
                continue
            if keyword == "LUT_1D_SIZE":
                raise ValueError(f"1D LUT is unsupported: {path}")
            if keyword in {"DOMAIN_MIN", "DOMAIN_MAX"}:
                if len(parts) != 4:
                    raise ValueError(f"invalid {keyword} at {path}:{line_number}")
                value = tuple(float(item) for item in parts[1:4])
                if keyword == "DOMAIN_MIN":
                    domain_min = value
                else:
                    domain_max = value
                continue
            if len(parts) != 3:
                raise ValueError(f"invalid LUT row at {path}:{line_number}")
            rows.append(tuple(float(item) for item in parts))
    if size is None or size < 2:
        raise ValueError(f"missing or invalid LUT_3D_SIZE: {path}")
    _check_row_count(path, rows, size)
    return rows, size, domain_min, domain_max


def _read_3dl(path: Path) -> tuple[list[tuple[float, float, float]], int]:
    nodes: list[int] | None = None
    rows: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            if text.upper().startswith(("3DMESH", "MESH", "3DL")):
                continue
            parts = text.split()
            if nodes is None and len(parts) > 3 and all(_is_int(item) for item in parts):
                nodes = [int(item) for item in parts]
                continue
            if len(parts) == 3 and all(_is_number(item) for item in parts):
                rows.append(tuple(float(item) for item in parts))
                continue
            raise ValueError(f"invalid 3DL row at {path}:{line_number}")
    if nodes is None or len(nodes) < 2:
        raise ValueError(f"missing 3DL node row: {path}")
    _check_row_count(path, rows, len(nodes))
    return rows, len(nodes)


def _rows_to_grid(
    rows: Iterable[tuple[float, float, float]], size: int, *, red_fastest: bool
) -> np.ndarray:
    flat = np.asarray(list(rows), dtype=np.float32)
    if red_fastest:
        return flat.reshape(size, size, size, 3)
    return flat.reshape(size, size, size, 3).transpose(2, 1, 0, 3)


def _check_row_count(path: Path, rows: list[tuple[float, float, float]], size: int) -> None:
    expected = size ** 3
    if len(rows) != expected:
        raise ValueError(f"{path}: expected {expected} LUT rows, found {len(rows)}")


def _is_int(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return True


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


__all__ = ["load_lut"]
