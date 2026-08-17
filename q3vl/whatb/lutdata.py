"""The LUT bank, ``L_l(x)``, and the data generation law ``F*(x, p)``.

Three facts this module exists to hold, all of them repository facts rather than
choices:

**1. The evaluation operator is the generator's operator.**  ``L_l`` is evaluated
with exactly the call ``dataset_build/src/construct/rendering.py:390-405`` uses::

    volume = from_numpy(grid).permute(3, 0, 1, 2)[None]          # BGR storage
    coords = ((x - dmin) / span).clamp(0, 1)
    points = (coords * 2 - 1)                                     # (r, g, b)
    grid_sample(volume, points, mode="bilinear",
                padding_mode="border", align_corners=True)

The ``.cube``/``.3dl`` grid is stored ``grid[b, g, r, :]`` (the CPU oracle at
``rendering.py:77-109`` says so in its docstring), and ``grid_sample``'s last
grid axis is ``(x, y, z) = (W, H, D) = (r, g, b)`` after the permute -- so the
colour channels go in in RGB order and index the right axes.  The unit test pins
this against ``apply_lut_cpu_oracle``, which is the one place a transposed axis
order would be invisible in every downstream number.

**2. No resampling.**  Ruling 11.1-3: LUTs are evaluated on their own grid (the
bank has nine sizes: 16/17/21/25/32/33/40/64/65), because resampling to a common
33^3 would change ``y`` and the function-space target would no longer be the law
that produced the dataset's GT image.  ``--lut-resample none`` is the default and
the only implemented value; asking for anything else raises.

**3. The law.**  ``rendering.py:301-313``::

    F*(x, p) = (1 - a(p)) x + a(p) L_l(x)
    with the endpoints snapped: out[a == 0] = x, out[a == 1] = L_l(x)

``style`` samples have ``a == 1`` everywhere (no mask member exists for them).
The same mixer is what the headline image-formation uses, which is why
:func:`mix_alpha` lives here and the criteria import it instead of re-writing
``(1-a) * I + a * f``.

Bank facts (2026-08-15, ``/var/cache/veradata/preset_bank_full``): 4051 entries
(``.cube`` 4000 / ``.3dl`` 51), every one with ``dmin = (0,0,0)`` and
``dmax = (1,1,1)``; the four splits' 3149 / 531 / 577 / 259 ``lut_id`` all hit.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "BANK_DIR",
    "LutEntry",
    "LutBank",
    "apply_lut_volume",
    "mix_alpha",
    "f_star",
]

#: ``presets.bank_dir`` of the production databuild configs
BANK_DIR = Path("/var/cache/veradata/preset_bank_full")

_MAX_CACHE = 256


@dataclass(frozen=True)
class LutEntry:
    """One bank row: ``lut_id`` -> preset path + domain."""

    lut_id: str
    path: Path
    dmin: tuple[float, float, float]
    dmax: tuple[float, float, float]

    @property
    def suffix(self) -> str:
        return self.path.suffix.lower()

    def to_dict(self) -> dict[str, Any]:
        return {"lut_id": self.lut_id, "preset_path": str(self.path),
                "dmin": list(self.dmin), "dmax": list(self.dmax),
                "format": self.suffix}


def apply_lut_volume(volume: torch.Tensor, x: torch.Tensor, *,
                     dmin: torch.Tensor | None = None,
                     dmax: torch.Tensor | None = None) -> torch.Tensor:
    """``L(x)`` for a packed volume.  ``x`` is ``(..., 3)`` sRGB in [0,1].

    ``volume`` is ``(1, 3, D_b, D_g, D_r)`` -- the ``permute(3,0,1,2)[None]`` of
    the stored ``grid[b, g, r, :]``.  Output has ``x``'s shape, dtype and device.
    """
    if volume.dim() != 5 or volume.shape[0] != 1 or volume.shape[1] != 3:
        raise ValueError(
            f"volume must be (1, 3, D, D, D), got {tuple(volume.shape)}")
    if x.shape[-1] != 3:
        raise ValueError(f"x must be (..., 3), got {tuple(x.shape)}")
    if x.device != volume.device:
        raise ValueError(
            f"x is on {x.device} and the LUT volume on {volume.device}; move the "
            "volume with LutBank.volume(lut_id, device=...) instead of moving "
            "the data")
    lead = x.shape[:-1]
    flat = x.reshape(-1, 3)
    if dmin is not None:
        span = torch.where(dmax == dmin, torch.ones_like(dmax), dmax - dmin)
        coords = ((flat - dmin) / span).clamp(0.0, 1.0)
    else:
        coords = flat.clamp(0.0, 1.0)
    points = (coords * 2.0 - 1.0).view(1, 1, 1, -1, 3)
    out = F.grid_sample(volume.to(dtype=points.dtype), points, mode="bilinear",
                        padding_mode="border", align_corners=True)
    # (1, 3, 1, 1, N) -> (N, 3)
    return out.view(3, -1).transpose(0, 1).reshape(*lead, 3).to(dtype=x.dtype)


def mix_alpha(before: torch.Tensor, edited: torch.Tensor,
              alpha: torch.Tensor | float) -> torch.Tensor:
    """``rendering.py:311-313`` verbatim, endpoint snapping included.

    ``alpha`` broadcasts against ``before``: a scalar, a per-pixel field
    ``(..., 1)`` or a full ``(..., 3)``.  The two ``torch.where`` branches are not
    cosmetic -- they are why the dataset's GT is bit-exactly ``before`` outside
    the mask, and the headline is measured against that GT.
    """
    if isinstance(alpha, (int, float)):
        alpha = torch.as_tensor(float(alpha), dtype=before.dtype,
                                device=before.device)
    alpha = alpha.to(device=before.device, dtype=before.dtype)
    mixed = before * (1.0 - alpha) + edited * alpha
    return torch.where(alpha == 0, before, torch.where(alpha == 1, edited, mixed))


def f_star(x: torch.Tensor, alpha: torch.Tensor | float,
           lut_values: torch.Tensor) -> torch.Tensor:
    """``F*(x, p) = (1 - a(p)) x + a(p) L_l(x)`` given ``L_l(x)`` already evaluated."""
    return mix_alpha(x, lut_values, alpha)


class LutBank:
    """``lut_id`` -> preset path, grid, ``L_l(x)``, with an LRU of volumes.

    ``lut_id`` is the key of both ``luts_meta.json`` and ``luts.npz`` (verified:
    the split index's ``lut_id`` is the bank's ``preset_id``).  Grids are read
    from the packed ``.npz`` when present and parsed from the ``.cube``/``.3dl``
    file otherwise, which is the same order of preference as the generator's
    ``_LutLoader`` (``rendering.py:132-145``).

    The LRU is keyed by ``(lut_id, device, dtype)`` and holds
    ``(1, 3, D, D, D)`` volumes; 256 entries of the dominant 32^3 size is about
    100 MB in fp32.
    """

    def __init__(self, bank_dir: str | Path = BANK_DIR, *,
                 cache_size: int = _MAX_CACHE, resample: str = "none"):
        if resample != "none":
            raise ValueError(
                "--lut-resample only implements 'none' (ruling 11.1-3): "
                "resampling changes y and the function-space target would stop "
                "matching the dataset's own generation law")
        self.bank_dir = Path(bank_dir)
        self.resample = resample
        meta_path = self.bank_dir / "luts_meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"{meta_path} does not exist")
        with meta_path.open("r", encoding="utf-8") as fh:
            raw: dict[str, dict[str, Any]] = json.load(fh)
        self.entries: dict[str, LutEntry] = {
            k: LutEntry(lut_id=k, path=Path(str(v.get("path") or "")),
                        dmin=tuple(float(t) for t in v.get("dmin", (0, 0, 0))),
                        dmax=tuple(float(t) for t in v.get("dmax", (1, 1, 1))))
            for k, v in raw.items()}
        self._npz_path = self.bank_dir / "luts.npz"
        self._npz: Any = None
        self._cache: "OrderedDict[tuple[str, str, torch.dtype], torch.Tensor]" = \
            OrderedDict()
        self._sizes: dict[str, int] = {}
        self.cache_size = int(cache_size)
        self.n_hit = 0
        self.n_miss = 0

    # -- metadata -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, lut_id: str) -> bool:
        return lut_id in self.entries

    def lut_ids(self) -> list[str]:
        return sorted(self.entries)

    def entry(self, lut_id: str) -> LutEntry:
        try:
            return self.entries[lut_id]
        except KeyError:
            raise KeyError(
                f"{lut_id!r} is not in {self.bank_dir}/luts_meta.json "
                f"({len(self.entries)} entries)") from None

    def preset_path(self, lut_id: str) -> Path:
        return self.entry(lut_id).path

    def grid_numpy(self, lut_id: str) -> np.ndarray:
        """``(D, D, D, 3)`` float32, stored ``grid[b, g, r]`` (BGR axis order)."""
        entry = self.entry(lut_id)
        if self._npz is None and self._npz_path.is_file():
            self._npz = np.load(self._npz_path, allow_pickle=False)
        grid: np.ndarray | None = None
        if self._npz is not None and lut_id in getattr(self._npz, "files", ()):
            grid = np.asarray(self._npz[lut_id], dtype=np.float32)
        else:
            from dataset_build.lut_io import load_lut  # clean tree, read-only

            grid = np.asarray(load_lut(str(entry.path))[0], dtype=np.float32)
        if grid.ndim != 4 or grid.shape[-1] != 3 or len(set(grid.shape[:3])) != 1:
            raise ValueError(
                f"{lut_id}: expected a cubic (D,D,D,3) grid, got {grid.shape}")
        self._sizes[lut_id] = int(grid.shape[0])
        return grid

    def size(self, lut_id: str) -> int:
        """Grid size D of one LUT (logged per sample as ``lut_size``)."""
        if lut_id not in self._sizes:
            self.grid_numpy(lut_id)
        return self._sizes[lut_id]

    # -- volumes ------------------------------------------------------------
    def volume(self, lut_id: str, *, device: Any = "cpu",
               dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """``(1, 3, D, D, D)`` volume, LRU-cached per (id, device, dtype)."""
        dev = torch.device(device)
        key = (lut_id, str(dev), dtype)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            self.n_hit += 1
            return hit
        self.n_miss += 1
        grid = self.grid_numpy(lut_id)
        vol = torch.from_numpy(grid).permute(3, 0, 1, 2)[None].to(
            device=dev, dtype=dtype).contiguous()
        self._cache[key] = vol
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return vol

    def domain(self, lut_id: str, *, device: Any = "cpu",
               dtype: torch.dtype = torch.float32
               ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        e = self.entry(lut_id)
        if e.dmin == (0.0, 0.0, 0.0) and e.dmax == (1.0, 1.0, 1.0):
            return None, None            # identity normalisation, skip the maths
        return (torch.tensor(e.dmin, device=device, dtype=dtype),
                torch.tensor(e.dmax, device=device, dtype=dtype))

    # -- evaluation ---------------------------------------------------------
    def apply(self, x: torch.Tensor, lut_id: str) -> torch.Tensor:
        """``L_l(x)`` for query colours ``x`` of shape ``(..., 3)`` in [0,1]."""
        vol = self.volume(lut_id, device=x.device, dtype=torch.float32)
        dmin, dmax = self.domain(lut_id, device=x.device, dtype=x.dtype)
        return apply_lut_volume(vol, x, dmin=dmin, dmax=dmax)

    def apply_image(self, img: torch.Tensor, lut_id: str) -> torch.Tensor:
        """``L_l(I)`` for a ``(3, H, W)`` or ``(B, 3, H, W)`` image in [0,1]."""
        if img.dim() == 3:
            return self.apply(img.permute(1, 2, 0), lut_id).permute(2, 0, 1)
        if img.dim() == 4:
            return self.apply(img.permute(0, 2, 3, 1), lut_id).permute(0, 3, 1, 2)
        raise ValueError(f"img must be (3,H,W) or (B,3,H,W), got {tuple(img.shape)}")

    def f_star(self, x: torch.Tensor, alpha: torch.Tensor | float,
               lut_id: str) -> torch.Tensor:
        """The data law ``F*(x, p)`` on query colours."""
        return f_star(x, alpha, self.apply(x, lut_id))

    def f_star_image(self, img: torch.Tensor, alpha: torch.Tensor | float,
                     lut_id: str) -> torch.Tensor:
        """The data law on an image: the dataset's own GT, recomputed."""
        return mix_alpha(img, self.apply_image(img, lut_id), alpha)

    def evaluate_library(self, lut_ids: Sequence[str], x: torch.Tensor
                         ) -> torch.Tensor:
        """``(n_lut, N, 3)``: every library LUT on the same query colours.

        The B1 (library mean), B2 (library random), B3 (bucket) and B4 (oracle)
        columns all read off this one tensor, so the library is evaluated once
        per query set rather than once per baseline.
        """
        if x.dim() != 2 or x.shape[-1] != 3:
            raise ValueError(f"x must be (N, 3), got {tuple(x.shape)}")
        out = torch.empty((len(lut_ids), x.shape[0], 3), dtype=x.dtype,
                          device=x.device)
        for i, lid in enumerate(lut_ids):
            out[i] = self.apply(x, lid)
        return out

    # -- bookkeeping --------------------------------------------------------
    def size_histogram(self, lut_ids: Iterable[str] | None = None) -> dict[int, int]:
        hist: dict[int, int] = {}
        for lid in (self.lut_ids() if lut_ids is None else lut_ids):
            d = self.size(lid)
            hist[d] = hist.get(d, 0) + 1
        return dict(sorted(hist.items()))

    def facts(self) -> dict[str, Any]:
        hist: dict[int, int] = {}
        for d in self._sizes.values():
            hist[d] = hist.get(d, 0) + 1
        return {"bank_dir": str(self.bank_dir), "n_entries": len(self.entries),
                "resample": self.resample, "cache_size": self.cache_size,
                "cache_hits": self.n_hit, "cache_misses": self.n_miss,
                "n_luts_loaded": len(self._sizes),
                "grid_size_histogram": dict(sorted(hist.items())),
                "operator": ("grid_sample(bilinear, border, align_corners=True) "
                             "on permute(3,0,1,2) BGR storage -- "
                             "rendering.py:390-405")}


def lut_ids_of(records: Iterable[Mapping[str, Any]]) -> list[str]:
    """Unique ``lut_id`` of a record/index iterable, sorted."""
    return sorted({str(r["lut_id"]) for r in records if r.get("lut_id")})
