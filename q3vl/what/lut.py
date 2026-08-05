"""3-D LUT tables: the GT function ``T_gt`` and the delivery-side readback.

Two different interpolators live here and the difference between them is a
first-class number, never an implementation detail:

``trilinear``    what the dataset's ``I_tar`` was actually rendered with --
                 ``dataset_build/core/render_backend.py::_render_cube_gpu`` uses
                 ``F.grid_sample(mode="bilinear", padding_mode="border",
                 align_corners=True)`` over the ``[domain_min, domain_max]``
                 normalised coordinate, and its CPU oracle
                 (``dataset_build/src/construct/rendering.py::apply_lut_cpu_oracle``)
                 is the same arithmetic written out.  Protocol 9.1's ``T_gt`` is
                 therefore trilinear by default: otherwise ``L_func``'s target and
                 the final-image target of protocol 12.2 would be two different
                 functions of the same ``.cube`` file.
``tetrahedral``  what a ``.cube`` *host* uses, hence what protocol 7.6/12.1 mean
                 by "production readback".  Used for ``L_bake`` and for the bake
                 gate, and always reported next to the trilinear numbers.

Table convention: ``table[r, g, b, c]``, matching ``tools/cube/cubelib`` and
``model/glut_repro/model_rdg``.  ``dataset_build.lut_io.load_lut`` returns
``grid[b, g, r]``, so :func:`load_gt_table` transposes exactly once, at the
boundary, and nothing downstream has to remember.

:func:`tetra_lookup` is the CI-pinned implementation of
``model/glut_repro/model_rdg.py`` (pinned there against colour-science's
``table_interpolation_tetrahedral``); it is re-derived here rather than imported
so that Stage-What does not depend on the previous campaign's package, and
``tests/test_lut.py`` re-pins it against that implementation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import BAKE_SIZE, GT_LUT_INTERP

__all__ = ["lattice_points", "GtLutTable", "load_gt_table", "tetra_lookup",
           "trilinear_lookup", "apply_table", "LutBank", "table_digest"]


def lattice_points(size: int = BAKE_SIZE, device=None,
                   dtype=torch.float32) -> torch.Tensor:
    """``(size^3, 3)`` uniform RGB lattice, R slowest -- ``table.reshape(-1, 3)``
    order for a ``table[r, g, b]`` array."""
    ax = torch.linspace(0.0, 1.0, size, device=device, dtype=dtype)
    r, g, b = torch.meshgrid(ax, ax, ax, indexing="ij")
    return torch.stack([r, g, b], dim=-1).reshape(-1, 3)


def table_digest(table: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(table, dtype=np.float32).tobytes()).hexdigest()


@dataclass(frozen=True)
class GtLutTable:
    """One parsed GT LUT.  ``table[r, g, b, c]`` on ``[domain_min, domain_max]``."""

    lut_id: str
    table: torch.Tensor            # (S, S, S, 3) float32
    domain_min: torch.Tensor       # (3,)
    domain_max: torch.Tensor       # (3,)
    source: str = ""
    size: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "size", int(self.table.shape[0]))
        if self.table.dim() != 4 or self.table.shape[-1] != 3:
            raise ValueError(f"table must be (S,S,S,3), got {tuple(self.table.shape)}")
        if len(set(self.table.shape[:3])) != 1:
            raise ValueError(f"table must be cubic, got {tuple(self.table.shape[:3])}")
        if not torch.isfinite(self.table).all():
            raise ValueError(f"{self.lut_id}: non-finite LUT value")

    def to(self, device=None, dtype=None) -> "GtLutTable":
        return GtLutTable(
            self.lut_id, self.table.to(device=device, dtype=dtype),
            self.domain_min.to(device=device, dtype=dtype),
            self.domain_max.to(device=device, dtype=dtype), self.source,
        )

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``x`` from the LUT's domain to ``[0, 1]``, as the renderer does."""
        lo = self.domain_min.to(x.device, x.dtype)
        hi = self.domain_max.to(x.device, x.dtype)
        span = torch.where(hi == lo, torch.ones_like(hi), hi - lo)
        return ((x - lo) / span).clamp(0.0, 1.0)

    def apply(self, x: torch.Tensor, mode: str = GT_LUT_INTERP) -> torch.Tensor:
        """``T_gt(x)`` for ``x`` of shape ``(P, 3)`` or ``(B, P, 3)``."""
        squeeze = x.dim() == 2
        xb = x.unsqueeze(0) if squeeze else x
        t = self.table.to(x.device, x.dtype).unsqueeze(0).expand(xb.shape[0], -1, -1, -1, -1)
        out = apply_table(t, self.normalise(xb), mode)
        return out.squeeze(0) if squeeze else out

    def facts(self) -> dict[str, Any]:
        return {
            "lut_id": self.lut_id, "size": self.size, "source": self.source,
            "domain_min": [float(v) for v in self.domain_min],
            "domain_max": [float(v) for v in self.domain_max],
            "digest": table_digest(self.table.detach().cpu().numpy()),
        }


def load_gt_table(path: str | Path, lut_id: str = "") -> GtLutTable:
    """Parse a ``.cube`` / ``.3dl`` into the canonical ``table[r, g, b]`` layout.

    The parser itself is ``dataset_build.lut_io.load_lut`` -- the same strict
    reader the dataset build used, so a file that produced an ``I_tar`` cannot
    silently parse differently here.
    """
    from dataset_build.lut_io import load_lut

    grid_bgr, dmin, dmax = load_lut(path)
    table = np.ascontiguousarray(np.transpose(grid_bgr, (2, 1, 0, 3)))
    return GtLutTable(
        lut_id=lut_id or Path(path).stem,
        table=torch.from_numpy(table).float(),
        domain_min=torch.from_numpy(np.asarray(dmin, dtype=np.float32)),
        domain_max=torch.from_numpy(np.asarray(dmax, dtype=np.float32)),
        source=str(path),
    )


# --- interpolators ----------------------------------------------------------

def _gather(flat: torch.Tensor, i0: torch.Tensor, K: int,
            dr: int, dg: int, db: int) -> torch.Tensor:
    idx = ((i0[..., 0] + dr) * K + (i0[..., 1] + dg)) * K + (i0[..., 2] + db)
    return torch.gather(flat, 1, idx.unsqueeze(-1).expand(-1, -1, 3))


def _corner_setup(table: torch.Tensor, x: torch.Tensor):
    B, K = table.shape[0], table.shape[1]
    xc = x.clamp(0.0, 1.0) * (K - 1)
    i0 = xc.floor().clamp(0, K - 2).long()
    f = xc - i0.to(xc.dtype)
    return table.reshape(B, K * K * K, 3), i0, f, K


def trilinear_lookup(table: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``table (B,K,K,K,3)``, ``x (B,P,3)`` in ``[0,1]`` -> ``(B,P,3)``.

    Byte-for-byte the arithmetic of ``apply_lut_cpu_oracle``: clamp to the cube,
    scale by ``K-1``, blend R then G then B.  ``align_corners=True`` with border
    padding is exactly this once the clamp is applied.
    """
    flat, i0, f, K = _corner_setup(table, x)
    fr, fg, fb = (f[..., i:i + 1] for i in range(3))
    c000 = _gather(flat, i0, K, 0, 0, 0)
    c100 = _gather(flat, i0, K, 1, 0, 0)
    c010 = _gather(flat, i0, K, 0, 1, 0)
    c110 = _gather(flat, i0, K, 1, 1, 0)
    c001 = _gather(flat, i0, K, 0, 0, 1)
    c101 = _gather(flat, i0, K, 1, 0, 1)
    c011 = _gather(flat, i0, K, 0, 1, 1)
    c111 = _gather(flat, i0, K, 1, 1, 1)
    c00 = c000 * (1 - fr) + c100 * fr
    c10 = c010 * (1 - fr) + c110 * fr
    c01 = c001 * (1 - fr) + c101 * fr
    c11 = c011 * (1 - fr) + c111 * fr
    c0 = c00 * (1 - fg) + c10 * fg
    c1 = c01 * (1 - fg) + c11 * fg
    return c0 * (1 - fb) + c1 * fb


def tetra_lookup(table: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Tetrahedral interpolation -- the host's interpolator (protocol 7.6/12.1).

    ``table (B,K,K,K,3)`` indexed ``[r, g, b]``, ``x (B,P,3)`` in ``[0,1]``.
    Standard 6-simplex decomposition of the unit cube (Kasson et al.); this is
    the implementation of ``model/glut_repro/model_rdg.py::tetra_lookup``, which
    ``ci_checks_rdg`` pinned against colour-science's
    ``table_interpolation_tetrahedral``.
    """
    flat, i0, f, K = _corner_setup(table, x)
    fr, fg, fb = f.unbind(-1)
    c000 = _gather(flat, i0, K, 0, 0, 0)
    c111 = _gather(flat, i0, K, 1, 1, 1)
    out = torch.zeros_like(c000)
    cases = [
        (fr >= fg) & (fg >= fb), (fr >= fb) & (fb > fg), (fb > fr) & (fr >= fg),
        (fg > fr) & (fr >= fb), (fb > fg) & (fg > fr), (fg >= fb) & (fb > fr),
    ]
    verts = [
        ((1, 0, 0), (1, 1, 0), fr, fg, fb),
        ((1, 0, 0), (1, 0, 1), fr, fb, fg),
        ((0, 0, 1), (1, 0, 1), fb, fr, fg),
        ((0, 1, 0), (1, 1, 0), fg, fr, fb),
        ((0, 0, 1), (0, 1, 1), fb, fg, fr),
        ((0, 1, 0), (0, 1, 1), fg, fb, fr),
    ]
    for cond, (v1, v2, a, b_, c_) in zip(cases, verts):
        val = ((1 - a).unsqueeze(-1) * c000
               + (a - b_).unsqueeze(-1) * _gather(flat, i0, K, *v1)
               + (b_ - c_).unsqueeze(-1) * _gather(flat, i0, K, *v2)
               + c_.unsqueeze(-1) * c111)
        out = torch.where(cond.unsqueeze(-1), val, out)
    return out


def apply_table(table: torch.Tensor, x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "trilinear":
        return trilinear_lookup(table, x)
    if mode == "tetrahedral":
        return tetra_lookup(table, x)
    raise ValueError(f"unknown LUT interpolation {mode!r}")


# --- the published GT bank --------------------------------------------------

class LutBank:
    """``lut_id -> GtLutTable`` with a bounded LRU, over a published shard set.

    Two sources, one interface:

    * a published indexed-tar shard set written by
      ``scripts/pack_gt_luts.py`` (protocol 2.3) -- the production path;
    * a plain ``{lut_id: preset_path}`` map -- what the packing job itself and
      the CPU tests use, and the fallback if the shards do not exist yet.

    The LRU matters: the corpus is ~3.4k distinct tables at 32^3-65^3, i.e.
    0.4-3 MiB each in float32.  Holding all of them is several GiB for no reason
    when a micro-batch touches at most ``micro_batch`` of them.
    """

    def __init__(self, store=None, path_map: dict[str, str] | None = None,
                 capacity: int = 256):
        if store is None and not path_map:
            raise ValueError("LutBank needs a published store or a path map")
        self.store = store
        self.path_map = dict(path_map or {})
        self.capacity = int(capacity)
        self._cache: dict[str, GtLutTable] = {}
        self._order: list[str] = []
        self.n_hits = 0
        self.n_misses = 0

    def __contains__(self, lut_id: str) -> bool:
        if lut_id in self._cache or lut_id in self.path_map:
            return True
        return bool(self.store is not None and self.store.has(lut_id, ".lut.npy"))

    def get(self, lut_id: str) -> GtLutTable:
        hit = self._cache.get(lut_id)
        if hit is not None:
            self.n_hits += 1
            self._order.remove(lut_id)
            self._order.append(lut_id)
            return hit
        self.n_misses += 1
        tbl = self._load(lut_id)
        self._cache[lut_id] = tbl
        self._order.append(lut_id)
        while len(self._order) > self.capacity:
            self._cache.pop(self._order.pop(0), None)
        return tbl

    def _load(self, lut_id: str) -> GtLutTable:
        if self.store is not None and self.store.has(lut_id, ".lut.npy"):
            import io

            arr = np.load(io.BytesIO(self.store.read(lut_id, ".lut.npy")),
                          allow_pickle=False)
            meta = json.loads(self.store.read(lut_id, ".lutmeta.json").decode("utf-8"))
            return GtLutTable(
                lut_id=lut_id, table=torch.from_numpy(arr.astype(np.float32)),
                domain_min=torch.tensor(meta["domain_min"], dtype=torch.float32),
                domain_max=torch.tensor(meta["domain_max"], dtype=torch.float32),
                source=meta.get("source", ""),
            )
        path = self.path_map.get(lut_id)
        if path is None:
            raise KeyError(
                f"{lut_id} is in neither the published GT-LUT shards nor the path "
                "map; a sample whose T_gt cannot be resolved must be rejected, "
                "not trained with a fabricated target"
            )
        return load_gt_table(path, lut_id)

    def facts(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity, "n_cached": len(self._cache),
            "n_hits": self.n_hits, "n_misses": self.n_misses,
            "source": "shards" if self.store is not None else "path_map",
            "n_path_map": len(self.path_map),
        }
