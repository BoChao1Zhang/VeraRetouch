"""Pixel-resolution GT for the EPR-018..023 batch: one dispatcher, five sources.

Every arm in that batch supervises **above** the ``(gh, gw)`` grid the live arms
use, and every one of them needs the same four things:

(a) the ``.cgt.png`` soft alpha at its native short-side-1024 resolution;
(b) the published ``.maskhi.png`` view at short side 512 (the spec-5 image grid);
(c) an **analytic re-render** of the three geometric families at an arbitrary
    resolution, from the construction-side ``geometry`` dict;
(d) an ``area_resize`` projection to whatever grid the head outputs -- the *same*
    operator ``gt_low`` uses (``q3vl/whereb/amort/data.py:685-686``), so the
    headline criterion keeps measuring with one ruler.

plus, for the point-sampled arms (PRND / LIIF),

(e) point evaluation at arbitrary normalised coordinates: bilinear on a raster,
    closed form on an analytic family.

The ``semantic`` family declares no geometry by construction
(``q3vl/whereb/amort/geomparse.py:351-353``), so it can only be served from
(a)/(b).  Every such fallback is **counted**, never silent.

Determinism of the ``linear`` family
------------------------------------
``raster_geometry`` renders the *raw* field; what the build published is
``clip(raw * amount)`` with ``amount = 1.0 if raw_mean <= target else
target / raw_mean`` (``dataset_build/src/construct/canonical_masks.py:145-153``
called from ``_asset`` L168-171, ``linear_target`` default 0.5 at L227).  The
sqlite sidecar stores only ``effective_alpha_mean``, so ``amount`` has to be
recomputed.  It is recomputed **deterministically** from the closed form -- see
:func:`linear_amount` -- on a declared reference grid whose resolution is
recorded, because ``raw_mean`` is a mean over the grid and therefore very
slightly resolution-dependent.

Nothing in this module touches the live arms: it is import-only new code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from q3vl.where.upsample import area_resize

__all__ = [
    "ANALYTIC_FAMILIES",
    "MASK_TYPE_OF_FAMILY",
    "LINEAR_TARGET",
    "mask_type_of",
    "render_analytic",
    "eval_analytic",
    "linear_amount",
    "load_cgt_1024",
    "load_maskhi_512",
    "area_project",
    "sample_points",
    "make_coord",
    "PixGT",
    "PixGTProvider",
    "audit_analytic",
]

#: the three families that carry an analytic ``geometry`` dict
ANALYTIC_FAMILIES: tuple[str, ...] = ("radial", "band", "linear")

#: slot family -> ``raster_geometry`` branch
#: (``dataset_build/src/construct/subject_geom.py:106, 142, 199``)
MASK_TYPE_OF_FAMILY: dict[str, str] = {
    "radial": "circulargradient",
    "band": "circulargradient",
    "linear": "gradient",
}

#: ``build_mask_plan(..., linear_target=0.5)`` -- the published default
#: (``dataset_build/src/construct/canonical_masks.py:227``)
LINEAR_TARGET = 0.5


def mask_type_of(family_or_slot: str) -> str | None:
    """``"band"`` / ``"band-1"`` -> ``"circulargradient"``; semantic -> ``None``."""
    fam = str(family_or_slot or "").split("-")[0].strip().lower()
    return MASK_TYPE_OF_FAMILY.get(fam)


# --------------------------------------------------------------------------- #
# (c) analytic re-render
# --------------------------------------------------------------------------- #
def _raster(mask_type: str, geometry: Mapping[str, Any], h: int, w: int) -> np.ndarray:
    """``raster_geometry`` itself -- imported, never re-implemented."""
    from dataset_build.src.construct.canonical_masks import raster_geometry

    return raster_geometry(str(mask_type), dict(geometry), int(h), int(w))


def linear_amount(
    geometry: Mapping[str, Any],
    ref_hw: tuple[int, int],
    *,
    target: float = LINEAR_TARGET,
) -> float:
    """The published ``amount`` of a ``linear`` mask, recomputed from closed form.

    ``amount = 1.0 if raw_mean <= target else target / raw_mean``
    (``canonical_masks.py:145-153``).  ``raw_mean`` is the mean of the *raw*
    rendered field, so the reference grid it is measured on is part of the
    definition and is recorded by the caller (``PixGTProvider.facts()``).

    MEASURED 2026-08-14 on one linear geometry: 0.887121 at 32x48 -> 0.870340 at
    128x192 -> 0.866248 at 512x768 -> 0.865569 at 1024x1536.  The construction
    side rendered at the build's own resolution (``agent.py:2927, 2945-2951``
    passes ``render.short_edge``), i.e. the ``.cgt`` grid; recomputing on the
    4x decoder grid instead moves the resulting field by 0.0027 mean-abs, an
    order of magnitude inside the 0.02 pre-flight tolerance.  Both conventions
    are therefore usable; which one a run used must be in its ``facts()``.
    """
    raw = _raster("gradient", geometry, ref_hw[0], ref_hw[1])
    raw_mean = float(np.asarray(raw, dtype=np.float64).mean())
    if not math.isfinite(raw_mean) or raw_mean <= 0:
        raise ValueError("linear mask has no alpha mass; amount is undefined")
    return 1.0 if raw_mean <= target else target / raw_mean


def render_analytic(
    mask_type: str,
    geometry: Mapping[str, Any],
    h: int,
    w: int,
    *,
    amount: float | None = None,
    amount_ref_hw: tuple[int, int] | None = None,
    target: float = LINEAR_TARGET,
    device=None,
) -> torch.Tensor:
    """``(h, w)`` float32 alpha, analytically re-rendered at any resolution.

    ``mask_type == "gradient"`` (the ``linear`` family) additionally applies the
    published ``amount``: pass it, or let it be recomputed on ``amount_ref_hw``
    (default: the requested resolution, i.e. "raw_mean from the re-rendered field
    itself", EPR-022 §3 ④).
    """
    a = _raster(mask_type, geometry, h, w)
    if mask_type == "gradient":
        if amount is None:
            amount = linear_amount(geometry, amount_ref_hw or (h, w), target=target)
        a = np.clip(a * float(amount), 0.0, 1.0)
    t = torch.from_numpy(np.ascontiguousarray(a)).to(torch.float32)
    return t.to(device) if device is not None else t


def eval_analytic(
    mask_type: str,
    geometry: Mapping[str, Any],
    coords: torch.Tensor,
    *,
    amount: float | None = None,
    amount_ref_hw: tuple[int, int] | None = None,
    target: float = LINEAR_TARGET,
) -> torch.Tensor:
    """Closed-form alpha at arbitrary normalised points.  ``coords`` is ``(N, 2)``
    in ``(x, y)``, both in ``[0, 1]``, with the **same normalisation
    ``raster_geometry`` uses**: ``x = col / width``, ``y = row / height``
    (``canonical_masks.py:99-100``) -- i.e. the top-left corner convention, NOT
    pixel centres.  ``render_analytic(h, w)`` is exactly this function evaluated
    on ``x = j/w, y = i/h``; ``tests/test_pixgt.py`` pins that identity.

    A torch re-implementation is unavoidable here (the numpy original renders a
    full grid), so the test compares it against ``raster_geometry`` itself
    rather than against a second copy of the formula.
    """
    if coords.dim() != 2 or coords.shape[-1] != 2:
        raise ValueError(f"expected (N, 2) coords, got {tuple(coords.shape)}")
    g = dict(geometry)

    def value(name: str, default: float = 0.0) -> float:
        try:
            return float(str(g.get(name, default)).lstrip("+"))
        except (TypeError, ValueError):
            return default

    x = coords[:, 0].to(torch.float32)
    y = coords[:, 1].to(torch.float32)
    if mask_type == "circulargradient":
        cx = (value("Left") + value("Right")) / 2.0
        cy = (value("Top") + value("Bottom")) / 2.0
        rx = max(abs(value("Right") - value("Left")) / 2.0, 1e-3)
        ry = max(abs(value("Bottom") - value("Top")) / 2.0, 1e-3)
        angle = math.radians(value("Angle"))
        xr = (x - cx) * math.cos(angle) + (y - cy) * math.sin(angle)
        yr = -(x - cx) * math.sin(angle) + (y - cy) * math.cos(angle)
        distance = torch.sqrt((xr / rx) ** 2 + (yr / ry) ** 2)
        feather = max(value("Feather", 50.0) / 100.0, 0.05)
        alpha = torch.clamp((distance - 1.0) / feather + 0.5, 0.0, 1.0)
    elif mask_type == "gradient":
        zx, zy = value("ZeroX"), value("ZeroY")
        fx, fy = value("FullX", 1.0), value("FullY")
        dx, dy = fx - zx, fy - zy
        length_sq = dx * dx + dy * dy + 1e-6
        alpha = torch.clamp(((x - zx) * dx + (y - zy) * dy) / length_sq, 0.0, 1.0)
    else:
        raise ValueError(f"unsupported mask_type {mask_type!r}")
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    if str(g.get("Flipped", "false")).lower().lstrip("+") == "true":
        alpha = 1.0 - alpha
    if mask_type == "gradient":
        if amount is None:
            if amount_ref_hw is None:
                raise ValueError(
                    "the linear family needs an amount: pass amount= or "
                    "amount_ref_hw= (the reference grid raw_mean is measured on)")
            amount = linear_amount(g, amount_ref_hw, target=target)
        alpha = torch.clamp(alpha * float(amount), 0.0, 1.0)
    return alpha


# --------------------------------------------------------------------------- #
# (a) / (b) raster sources
# --------------------------------------------------------------------------- #
def load_cgt_1024(resolver, record: Mapping[str, Any]) -> torch.Tensor:
    """``(h, w)`` float32 in [0,1] -- the ``.cgt.png`` at its native resolution.

    ``resolver`` is a :class:`q3vl.where.maskdata.MaskResolver` (its ``suffix``
    default is ``.cgt.png``, ``q3vl/where/config.py:197``); ``record`` is the raw
    dataset record.  Short side 1024 (``q3vl/where/maskdata.py:180-190``).
    """
    arr = resolver.load(resolver.resolve(dict(record)))
    return torch.from_numpy(np.ascontiguousarray(arr)).to(torch.float32)


def load_maskhi_512(maskviews, sample_id: str) -> torch.Tensor:
    """``(out_h, out_w)`` float32 -- the published ``.maskhi.png`` (short side 512).

    ``maskviews`` is a :class:`q3vl.whereb.stores.MaskViewStore`
    (``stores.py:189-207``); the read root is ``/mnt/nfs-ro`` by config
    (``q3vl/where/config.py:225``).
    """
    if not maskviews.has(sample_id, maskviews.HI):
        raise KeyError(f"{sample_id}: no published .maskhi.png")
    return maskviews.mask_hi(sample_id).to(torch.float32)


# --------------------------------------------------------------------------- #
# (d) projection + (e) point sampling
# --------------------------------------------------------------------------- #
def area_project(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """``(H, W)`` -> ``(h, w)`` with the **same operator ``gt_low`` uses**.

    ``area_resize`` (``q3vl/where/upsample.py:54-62``): ``mode="area"`` going
    down, bilinear going up.  Call site mirrors
    ``q3vl/whereb/amort/data.py:685-686`` exactly -- that identity is what makes
    a new arm's ``m_low`` comparable with the frozen boards.
    """
    if x.dim() != 2:
        raise ValueError(f"expected (H, W), got {tuple(x.shape)}")
    return area_resize(x[None, None], (int(size[0]), int(size[1])))[0, 0]


def make_coord(h: int, w: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """``(h*w, 2)`` cell-centre coordinates in ``[0,1]`` as ``(x, y)``.

    Cell centres -- ``((j + 0.5)/w, (i + 0.5)/h)`` -- because that is the grid a
    field of shape ``(h, w)`` actually samples, and it is what
    :func:`sample_points` inverts exactly (``align_corners=False`` maps the
    normalised centre of cell ``j`` to ``(j + 0.5)/w``).  Note this is a
    *different* convention from :func:`eval_analytic`'s ``j/w``: the analytic
    formula's normalisation is fixed by ``raster_geometry`` and is not ours to
    change.  Any code mixing the two must say which it means; the tests pin both.
    """
    ys = (torch.arange(h, dtype=dtype, device=device) + 0.5) / float(h)
    xs = (torch.arange(w, dtype=dtype, device=device) + 0.5) / float(w)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)


def sample_points(field: torch.Tensor, coords: torch.Tensor,
                  *, mode: str = "bilinear") -> torch.Tensor:
    """Bilinear read of a raster at normalised ``(x, y)`` cell-centre coords.

    ``field`` is ``(H, W)`` or ``(C, H, W)``; ``coords`` is ``(N, 2)`` in [0,1].
    Uses ``grid_sample(2*coords - 1, align_corners=False)`` -- the PointRend
    convention (``point_features.py:39``, every call site passes
    ``align_corners=False``).  Returns ``(N,)`` or ``(C, N)``.

    Half a cell of disagreement between this and the GT renderer is 8 px on a
    32x48 grid, which is why ``tests/test_pixgt.py`` asserts that sampling a
    field at :func:`make_coord` reproduces the field exactly.
    """
    squeeze = field.dim() == 2
    f = field[None, None] if squeeze else field[None]
    grid = (2.0 * coords.to(f.dtype) - 1.0).reshape(1, 1, -1, 2)
    out = F.grid_sample(f, grid, mode=mode, padding_mode="border",
                        align_corners=False)
    out = out[0, :, 0, :]
    return out[0] if squeeze else out


# --------------------------------------------------------------------------- #
# the dispatcher
# --------------------------------------------------------------------------- #
@dataclass
class PixGT:
    """One sample's pixel GT plus the provenance of where it came from."""

    alpha: torch.Tensor            # (H, W) float32 in [0,1]
    source: str                    # render | cgt1024 | maskhi512
    family: str
    mask_type: str | None = None
    geometry: dict[str, Any] | None = None
    amount: float | None = None
    reason: str = ""               # why the fallback, when source != "render"

    @property
    def analytic(self) -> bool:
        return self.source == "render"

    def at(self, size: tuple[int, int]) -> torch.Tensor:
        """Project to ``size`` with the ``gt_low`` operator."""
        return area_project(self.alpha, size)

    def points(self, coords: torch.Tensor) -> torch.Tensor:
        """Values at normalised ``(x, y)``: closed form when analytic, bilinear
        otherwise.  ``coords`` follow :func:`make_coord`'s cell-centre
        convention; the analytic branch consumes them directly because
        ``raster_geometry``'s ``[0,1]^2`` domain is the same image domain."""
        if self.analytic and self.mask_type and self.geometry is not None:
            return eval_analytic(self.mask_type, self.geometry, coords,
                                 amount=self.amount)
        return sample_points(self.alpha, coords)


class PixGTProvider:
    """Per-run object: sample -> :class:`PixGT`, with counted fallbacks.

    ``prefer`` picks the analytic path where it exists::

        prefer="render"   analytic three families re-rendered, semantic -> raster
        prefer="cgt1024"  everything from .cgt.png at short side 1024
        prefer="maskhi"   everything from the published .maskhi.png (short 512)

    which is exactly the ``--*-gt {raster,png}`` / ``--prnd-gt`` /
    ``--matte-gt-source`` flag each arm exposes.  ``raster_fallback`` picks which
    raster source the semantic family (and any store miss) falls back to.

    Every arm must publish ``facts()`` into its ``run_setup.json``: the counts
    are the pre-registered guard column for "how much of this run's supervision
    was actually the analytic target it claims".
    """

    RASTER_SOURCES = ("cgt1024", "maskhi512")

    def __init__(
        self,
        *,
        geom_store=None,          # q3vl.whereb.amort.data.ConstructGeomStore
        mask_resolver=None,       # q3vl.where.maskdata.MaskResolver
        maskviews=None,           # q3vl.whereb.stores.MaskViewStore
        families: Mapping[str, str] | None = None,
        prefer: str = "render",
        raster_fallback: str = "maskhi512",
        amount_ref_hw: tuple[int, int] | None = None,
        linear_target: float = LINEAR_TARGET,
    ):
        if prefer not in ("render", "cgt1024", "maskhi"):
            raise ValueError(f"prefer must be render|cgt1024|maskhi, got {prefer!r}")
        if raster_fallback not in self.RASTER_SOURCES:
            raise ValueError(
                f"raster_fallback must be one of {self.RASTER_SOURCES}, got "
                f"{raster_fallback!r}")
        self.geom_store = geom_store
        self.mask_resolver = mask_resolver
        self.maskviews = maskviews
        self.families = dict(families or {})
        self.prefer = prefer
        self.raster_fallback = raster_fallback
        #: None = "raw_mean from the re-rendered field itself" (EPR-022 §3 ④);
        #: a tuple pins one reference grid for every sample instead.
        self.amount_ref_hw = amount_ref_hw
        self.linear_target = float(linear_target)
        self.counts: dict[str, int] = {}

    def needs_record(self, family: str) -> bool:
        """True only when this family could end up on the ``.cgt.png`` branch.

        The caller uses it to skip a second shard read per sample: the analytic
        branch reads ``candidate_id`` from the sample meta and the ``.maskhi``
        branch needs only the sample id.
        """
        # conservative: any configuration in which the .cgt branch is reachable
        # (a geometry miss under prefer="render" still falls back) asks for it.
        return self.prefer == "cgt1024" or self.raster_fallback == "cgt1024"

    # -- lookup ------------------------------------------------------------
    def _count(self, key: str) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1

    def _raster_of(self, sample_id: str, record: Mapping[str, Any] | None,
                   sample: Any = None) -> tuple[torch.Tensor, str]:
        # `prefer` wins when it names a raster source; otherwise (prefer=render)
        # the choice belongs to `raster_fallback`.  Getting this backwards would
        # serve .maskhi under `--pixgt-source cgt1024`, i.e. a run whose config
        # says 1024 and whose supervision came from 512.
        first = {"cgt1024": "cgt1024", "maskhi": "maskhi512"}.get(
            self.prefer, self.raster_fallback)
        order = [first] + [s for s in self.RASTER_SOURCES if s != first]
        errs = []
        for src in order:
            try:
                if src == "maskhi512":
                    if self.maskviews is None:
                        raise KeyError("no MaskViewStore attached")
                    got = load_maskhi_512(self.maskviews, sample_id), src
                else:
                    if self.mask_resolver is None or record is None:
                        raise KeyError("no MaskResolver / record attached")
                    got = load_cgt_1024(self.mask_resolver, record), src
            except Exception as exc:  # noqa: BLE001 -- counted, then re-raised below
                errs.append(f"{src}: {exc}")
                continue
            if src != first:
                # the declared raster was unavailable for this sample.  One
                # missing member must not kill a 42k-sample run, but "the config
                # says 1024 and the supervision came from 512" must be visible
                # in facts() rather than inferred later from a puzzling board.
                self._count(f"downgraded_{first}_to_{src}")
            return got
        if sample is not None and getattr(sample, "mask_hi", None) is not None:
            # last resort: the mask the dataset already resolved (spec-5 grid)
            self._count("raster_from_sample_mask_hi")
            return sample.mask_hi.to(torch.float32), "maskhi512"
        raise KeyError(f"{sample_id}: no pixel GT raster available ({'; '.join(errs)})")

    def geometry_of(self, sample_id: str, candidate_id: str | None,
                    family: str) -> tuple[str | None, dict[str, Any] | None, str]:
        """``(mask_type, geometry, reason)``; ``(None, None, reason)`` when the
        analytic path is not available for this sample."""
        mt = mask_type_of(family)
        if mt is None:
            return None, None, "semantic_no_geometry"
        if self.geom_store is None:
            return None, None, "no_geom_store"
        row = self.geom_store.row(str(candidate_id or ""))
        if row is None:
            return None, None, "geom_store_miss"
        geom = row.get("geometry")
        if not geom:
            return None, None, "geom_row_empty"
        # the store's own slot_mode wins over a possibly stale family label
        mt2 = mask_type_of(str(row.get("slot_mode") or family))
        return (mt2 or mt), dict(geom), ""

    # -- the dispatcher ----------------------------------------------------
    def get(
        self,
        sample: Any,
        *,
        record: Mapping[str, Any] | None = None,
        family: str | None = None,
        size: tuple[int, int] | None = None,
    ) -> PixGT:
        """One sample's pixel GT.  ``size`` (h, w), when given, is the resolution
        the analytic render is produced at and the raster sources are projected
        to -- with :func:`area_project`, the ``gt_low`` operator."""
        sid = getattr(sample, "sample_id", str(sample))
        fam = family or self.families.get(sid, "unknown")
        cand = None
        if record is not None:
            cand = record.get("candidate_id")
        if cand is None:
            cand = (getattr(sample, "meta", {}) or {}).get("candidate_id")

        if self.prefer == "render":
            mt, geom, reason = self.geometry_of(sid, cand, fam)
            if mt is not None and geom is not None:
                if size is None:
                    raise ValueError(
                        "the analytic path renders at an explicit resolution; "
                        "pass size=(h, w)")
                amount = None
                if mt == "gradient":
                    amount = linear_amount(geom, self.amount_ref_hw or size,
                                           target=self.linear_target)
                a = render_analytic(mt, geom, size[0], size[1], amount=amount,
                                    target=self.linear_target)
                self._count("render")
                self._count(f"render_{fam}")
                return PixGT(alpha=a, source="render", family=fam, mask_type=mt,
                             geometry=geom, amount=amount)
            self._count(f"fallback_{reason}")
        else:
            reason = f"prefer_{self.prefer}"

        raw, src = self._raster_of(sid, record, sample)
        if size is not None:
            raw = area_project(raw, size)
        self._count(src)
        self._count(f"{src}_{fam}")
        return PixGT(alpha=raw, source=src, family=fam, reason=reason)

    def facts(self) -> dict[str, Any]:
        n = sum(v for k, v in self.counts.items() if k in ("render",) or k in self.RASTER_SOURCES)
        return {
            "prefer": self.prefer,
            "raster_fallback": self.raster_fallback,
            "amount_ref_hw": list(self.amount_ref_hw) if self.amount_ref_hw else None,
            "linear_target": self.linear_target,
            "n": n,
            "n_render": self.counts.get("render", 0),
            "frac_render": (self.counts.get("render", 0) / n) if n else None,
            "counts": dict(sorted(self.counts.items())),
        }


# --------------------------------------------------------------------------- #
# the pre-flight audit (EPR-021 ③ / EPR-022 ④: n=200, mean-abs <= tol)
# --------------------------------------------------------------------------- #
@dataclass
class AuditReport:
    n: int = 0
    n_pass: int = 0
    n_fail: int = 0
    n_skipped: int = 0
    tol: float = 0.02
    worst: float = 0.0
    failures: list[dict[str, Any]] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"n": self.n, "n_pass": self.n_pass, "n_fail": self.n_fail,
                "n_skipped": self.n_skipped, "tol": self.tol,
                "worst_mean_abs": self.worst,
                "skipped": dict(sorted(self.skipped.items())),
                "failures": self.failures[:32]}


def audit_analytic(
    provider: PixGTProvider,
    items: Sequence[tuple[Any, Mapping[str, Any] | None]],
    *,
    tol: float = 0.02,
    n: int = 200,
) -> AuditReport:
    """Spot-check the analytic re-render against the published raster.

    For each analytic-family sample: render at the raster's own resolution and
    compare mean-abs.  Samples over ``tol`` are reported so the caller can route
    them back to the raster path and **count** it (the pre-registered guard both
    EPR-021 and EPR-022 ask for).  Runs on CPU; no GPU, no training state.
    """
    rep = AuditReport(tol=float(tol))
    for sample, record in list(items)[: int(n)]:
        sid = getattr(sample, "sample_id", str(sample))
        fam = provider.families.get(sid, "unknown")
        cand = (record or {}).get("candidate_id") or \
            (getattr(sample, "meta", {}) or {}).get("candidate_id")
        mt, geom, reason = provider.geometry_of(sid, cand, fam)
        if mt is None or geom is None:
            rep.n_skipped += 1
            rep.skipped[reason] = rep.skipped.get(reason, 0) + 1
            continue
        try:
            raster, src = provider._raster_of(sid, record, sample)
        except KeyError as exc:
            rep.n_skipped += 1
            rep.skipped[f"no_raster:{exc}"] = rep.skipped.get(f"no_raster:{exc}", 0) + 1
            continue
        h, w = int(raster.shape[-2]), int(raster.shape[-1])
        amount = None
        if mt == "gradient":
            amount = linear_amount(geom, provider.amount_ref_hw or (h, w),
                                   target=provider.linear_target)
        rendered = render_analytic(mt, geom, h, w, amount=amount,
                                   target=provider.linear_target)
        d = float((rendered - raster.to(rendered.dtype)).abs().mean())
        rep.n += 1
        rep.worst = max(rep.worst, d)
        if d <= tol:
            rep.n_pass += 1
        else:
            rep.n_fail += 1
            rep.failures.append({"sample_id": sid, "family": fam,
                                 "mask_type": mt, "mean_abs": d, "raster": src})
    return rep
