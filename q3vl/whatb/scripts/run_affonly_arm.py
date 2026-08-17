"""EPR-025 AFFONLY entry point: train the affine-only conditional head, publish a board.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/bc/VeraRetouch \\
    /home/bc/envs/q3vl_sft/bin/python -m q3vl.whatb.scripts.run_affonly_arm \\
        --run-dir /home/bc/data/runs/whatb_AFFONLY_a \\
        --z-cache /home/bc/data/caches/whatb_z_20260815/generated

Spec: ``experiments/prs/EPR-025_affine-only-conditional-head/PROPOSAL.md``.
The model, the loss, the optimiser and the three arm assertions live in
:mod:`q3vl.whatb.arms.affonly`; this file is the plumbing -- flags, data, the
loop, the evaluation board -- and nothing about the experiment is decided here.

Frozen (never a flag): train = ``split == "train" and winner_confidence ==
"normal"``, n = 93934; ``B = 32 x Q = 256 = 8192`` colours/step; 2936 steps per
epoch; 117,440 steps; ``Î = (1-a) ⊙ I + a ⊙ f̂(I)``; the twelve pre-registered
criteria keys; headline = ``.contexts.all.headline_normal_only`` and never a val
loss.

Two environment notes, both of them load-bearing
------------------------------------------------
1. ``import sqlite3`` is the FIRST import below, before ``torch``.  In this
   environment ``import torch`` first and ``import sqlite3`` afterwards raises
   ``CXXABI_1.3.15 not found`` (the conda ``libicui18n`` against the system
   ``libstdc++``); the queue's launcher works around it with
   ``LD_LIBRARY_PATH=/home/bc/miniconda3/envs/llm_factory/lib``
   (``waves/epr018_023_arm.sh:60``), and importing sqlite3 first makes this entry
   correct with or without that variable.  ``MaskViewStore`` reaches sqlite3
   through ``dataset_build.tools.indexed_tar``.
2. Reads go to ``/mnt/nfs-ro`` (soft); this script writes **only** to
   ``--run-dir``, which it refuses to place under ``/mnt/nfs`` (gate artifacts on
   a hard mount go into D state and cannot be killed --
   ``waves/epr018_023_wave.sh:62``).

The ``z`` cache contract (consumed, not produced)
-------------------------------------------------
The reader is :mod:`q3vl.whatb.zcache` -- one implementation for all six arms.
Pass ``--z-cache <root>`` and the leaves are derived::

    <root>/<split>__<tag>/{z.npy, index.jsonl, meta.json}    tag in
    none | shuffle | irrelevant | const                      fp32 (ruling 11.1-5)

``meta.checkpoint`` / ``meta.readout_kind`` / ``meta.control_tag`` are asserted
against this run's flags, and 1% of the rows are replayed through
``readout.verify_plan``, before the dataloader exists.  ``--z-cache-<tag>`` still
takes an explicit leaf directory when the four caches are not under one root.

``--z-source synthetic`` replaces the cache with a deterministic hash of the
sample id.  It exists so the whole path (loop, guards, board, publication gate)
can be exercised on CPU while both GPUs are busy; it stamps
``z_source = synthetic`` into ``run_setup.json`` and marks the board
``published = false`` so such a run can never be mistaken for a result.
"""

from __future__ import annotations

try:  # sqlite3 BEFORE torch where the import order is still ours to pick
    import sqlite3  # noqa: F401  -- see the module docstring
except ImportError:  # pragma: no cover - the parent package already pulled torch in
    pass             # :class:`AlphaReader` re-raises this with the actionable message

import argparse
import hashlib
import io
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from q3vl.whatb import caliber as K
from q3vl.whatb.arms import affonly as A
from q3vl.whatb.colorspan import assert_color_span_encoding
from q3vl.whatb.criteria import (
    LibraryValues,
    assert_criteria_ran,
    bucket_draw,
    compose_hat,
    function_distance,
    image_delta_e00,
    library_random_draw,
    locality_errors,
    oracle_lut_ids,
)
from q3vl.whatb.glut import CLAMP_FLAG_CHOICES
from q3vl.whatb.guards import record_step_witness
from q3vl.whatb.lutdata import BANK_DIR, LutBank, apply_lut_volume
from q3vl.whatb.queries import QuerySampler, image_histogram_colors, uniform_grid
from q3vl.whatb.readout import WHATB_READOUT_KINDS
from q3vl.whatb.zcache import (
    CONTROL_TAGS,
    SyntheticZCache,
    ZCache,
    resolve_leaf,
)
from q3vl.whatb.splits import (
    DATASET_ROOT,
    IndexRow,
    color_texts_of,
    load_index,
    normal_only,
    read_record,
    ro_path,
    split_facts,
    train_source_facts,
)

#: proposal section 3.4-(10): the frozen base whose ``<seg_color>`` hidden is the condition
V2SEG_CHECKPOINT = "/home/bc/data/runs/q3vl_base_sft_v2seg_20260814/checkpoint-4976"
#: Where-A's published GT alpha views (short side 512), one dataset per split
MASKVIEW_ROOT = "/mnt/nfs-ro/bc/data/datasets/where_a-20260805/maskviews"
CONTROL_ROW_KEYS = {"shuffle": "N1_shuffle", "irrelevant": "N2_irrelevant", "const": "N3_const"}

_SOURCE_FILES = (
    Path(__file__).resolve(),
    Path(A.__file__).resolve(),
    # shared layer this arm consumes rather than copies (submit-time freeze)
    Path(A.__file__).resolve().parents[1] / "zcache.py",
    Path(A.__file__).resolve().parents[1] / "colorimetry.py",
)


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


class JsonlWriter:
    """Append-only jsonl, flushed per line (a killed job keeps its rows)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("a", encoding="utf-8")

    def write(self, row: Mapping[str, Any]) -> None:
        self.fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


# --------------------------------------------------------------------------- #
# conditions
# --------------------------------------------------------------------------- #
def open_z(path: str | Path | None, *, tag: str, checkpoint: str, readout: str,
           split: str, root: str | Path | None = None, synthetic: bool = False,
           seed: int = 20260810) -> ZCache | SyntheticZCache:
    """One condition source, from the ONE shared cache (:mod:`q3vl.whatb.zcache`).

    ``path`` is a leaf directory (``<split>__<tag>``); when it is omitted and
    ``root`` is given the leaf is derived from ``(split, tag)``.  The three
    start-up assertions of HANDOFF section 4.H -- ``checkpoint``,
    ``readout_kind`` and a 1% ``verify_plan`` replay -- run here, before the
    dataloader is built.  ``synthetic`` is the smoke stand-in and never
    publishable.

    This arm used to carry its own ``ZSource`` (a ``torch.load``-ed ``.pt`` with
    its own assertion set); the frozen block says 共同依赖只写一份.
    """
    if synthetic:
        return SyntheticZCache(tag=tag, split=split, checkpoint=checkpoint,
                               readout_kind=readout,
                               context_source="generated" if tag != "none" else "generated")
    ctx = "generated"          # frozen block 6; --context teacher uses its own root
    leaf = (Path(path) if path else
            (resolve_leaf(root, split, tag, ctx) if root else None))
    if leaf is None:
        raise ValueError(
            f"no z cache for control {tag!r}.  Pass --z-cache <root> (or "
            f"--z-cache-{tag} <leaf dir>); the four caches "
            "none/shuffle/irrelevant/const are what the N1..N3 columns are "
            "computed from.  --z-source synthetic is smoke only, never publishable.")
    cache = ZCache(leaf)
    cache.assert_belongs_to(checkpoint=checkpoint, readout_kind=readout,
                            control_tag=tag, seed=seed)
    return cache


# --------------------------------------------------------------------------- #
# images and GT alpha
# --------------------------------------------------------------------------- #
class ImageReader:
    """``IndexRow -> (3, H, W)`` float32 in [0,1], read straight from the shard."""

    def __init__(self) -> None:
        self._fh: dict[str, Any] = {}
        self.n_read = 0

    def read(self, row: IndexRow) -> torch.Tensor:
        m = row.raw["members"]["image"]
        key = str(m["shard"])
        fh = self._fh.get(key)
        if fh is None:
            fh = self._fh[key] = ro_path(key).open("rb")
        fh.seek(int(m["offset"]))
        blob = fh.read(int(m["length"]))
        if len(blob) != int(m["length"]):
            raise IOError(f"short read of {key}:{m['offset']}")
        with Image.open(io.BytesIO(blob)) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        self.n_read += 1
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def close(self) -> None:
        for fh in self._fh.values():
            fh.close()
        self._fh.clear()


class AlphaReader:
    """GT alpha at short side 512: Where-A's ``.maskhi.png``; ``style`` -> 1.0.

    The frozen block pins section 4.E's strata to **GT alpha at short side 512**,
    which is what ``.maskhi.png`` is.  A shape mismatch against the image is
    resampled with the campaign's one operator (``q3vl/where/upsample.py``'s
    ``area_resize``) and counted, never resized silently by another rule.
    """

    def __init__(self, split: str, root: str = MASKVIEW_ROOT, *, verify: bool = False) -> None:
        try:
            from q3vl.whereb.stores import MaskViewStore
        except ImportError as exc:  # pragma: no cover - environment, not logic
            raise SystemExit(
                f"cannot import the mask-view store ({exc}).  In this environment "
                "`import torch` before `import sqlite3` breaks the conda libstdc++ "
                "chain; export LD_LIBRARY_PATH=/home/bc/miniconda3/envs/llm_factory/lib "
                "(what waves/epr018_023_arm.sh:60 does) and re-run.") from exc

        self.split = split
        self.store = MaskViewStore(Path(root) / split, verify=verify)
        self.n_resized = 0
        self.n_style = 0
        self.n_local = 0

    def alpha(self, row: IndexRow, img: torch.Tensor) -> torch.Tensor | float:
        if row.is_style:
            self.n_style += 1
            return 1.0
        a = self.store.mask_hi(row.sample_id)
        if tuple(a.shape) != tuple(img.shape[-2:]):
            from q3vl.where.upsample import area_resize

            a = area_resize(a[None, None], (int(img.shape[-2]), int(img.shape[-1])))[0, 0]
            self.n_resized += 1
        self.n_local += 1
        return a.unsqueeze(0)

    def facts(self) -> dict[str, Any]:
        return {"split": self.split, "source": "where_a maskhi.png (short side 512)",
                "n_style_alpha1": self.n_style, "n_local": self.n_local,
                "n_area_resized": self.n_resized}


# --------------------------------------------------------------------------- #
# the library baselines
# --------------------------------------------------------------------------- #
def library_ids(train_rows: Sequence[IndexRow], *, n_rows: int, seed: int,
                full: bool = False) -> list[str]:
    """``Lib_tr``.  Default = section 4.C's own protocol (2500 index rows -> ~1137 ids).

    The pre-registered floors (B0 32.79 / B1 25.33 / B2 35.37 / B4 9.93 / B6
    10.17) were measured on that subset; ``--lib-full`` uses all 3149 train ids
    instead and the choice is recorded in ``run_setup``.
    """
    if full or n_rows <= 0 or n_rows >= len(train_rows):
        return sorted({r.lut_id for r in train_rows})
    rng = random.Random(seed)
    picked = rng.sample(list(train_rows), int(n_rows))
    return sorted({r.lut_id for r in picked})


def mean_lut_volume(bank: LutBank, lut_ids: Sequence[str], *, grid_n: int,
                    device: Any = "cpu") -> torch.Tensor:
    """B1's ``L̄(x) = mean_l L_l(x)`` materialised as one ``grid_n^3`` LUT volume.

    ``L̄`` is a point-wise mean, hence a valid mapping (section 4.C); evaluating
    every library LUT at every image pixel is not affordable, so it is evaluated
    once on a uniform grid and applied through **the same operator** the data law
    uses.  ``grid_n`` goes into ``run_setup`` (``B1_libmean_grid``): it is a
    property of the baseline, not of the target, so ruling 11.1-3 (no resampling
    of the GT LUT) is untouched.
    """
    x = uniform_grid(grid_n, device=device)                       # (n^3, 3), index (r,g,b)
    acc = torch.zeros_like(x)
    for lid in lut_ids:
        acc += bank.apply(x, lid)
    acc /= max(1, len(lut_ids))
    vals = acc.reshape(grid_n, grid_n, grid_n, 3)                 # [i_r, i_g, i_b, c]
    grid_bgr = vals.permute(2, 1, 0, 3).contiguous()              # [i_b, i_g, i_r, c]
    return grid_bgr.permute(3, 0, 1, 2)[None].contiguous()        # (1, 3, D_b, D_g, D_r)


# --------------------------------------------------------------------------- #
# evaluation sample
# --------------------------------------------------------------------------- #
class EvalSample:
    """One evaluation row with everything the board needs, loaded once."""

    __slots__ = ("row", "img", "alpha", "i_star", "z", "lut_id", "minor")

    def __init__(self, row: IndexRow, img: torch.Tensor, alpha: Any, i_star: torch.Tensor,
                 z: torch.Tensor, lut_id: str, minor: str | None) -> None:
        self.row, self.img, self.alpha, self.i_star = row, img, alpha, i_star
        self.z, self.lut_id, self.minor = z, lut_id, minor


@torch.no_grad()
def load_eval_samples(rows: Sequence[IndexRow], *, bank: LutBank, images: ImageReader,
                      alphas: AlphaReader, z_true: ZCache | SyntheticZCache, records: Mapping[str, Any],
                      device: Any = "cpu") -> list[EvalSample]:
    out: list[EvalSample] = []
    for row in rows:
        img = images.read(row).to(device)
        a = alphas.alpha(row, img)
        if isinstance(a, torch.Tensor):
            a = a.to(device)
        i_star = bank.f_star_image(img, a, row.lut_id)
        rec = records.get(row.sample_id) or {}
        out.append(EvalSample(row, img, a, i_star, z_true.get(row.sample_id).to(device),
                              row.lut_id, rec.get("minor")))
    return out


@torch.no_grad()
def headline_error(head: A.AffineOnlyHead, s: EvalSample, z: torch.Tensor | None = None,
                   *, point_chunk: int | None = None) -> tuple[float, torch.Tensor]:
    """``E_i = mean_p dE00(Î_i(p), I*_i(p))`` with the frozen image formation."""
    f_img = head.transform_image(s.z if z is None else z, s.img, point_chunk=point_chunk)
    i_hat = compose_hat(s.img, s.alpha, f_img)
    return float(image_delta_e00(i_hat, s.i_star)), i_hat


@torch.no_grad()
def lut_headline_error(s: EvalSample, values: torch.Tensor) -> float:
    """Headline for a baseline whose ``f̂(I)`` is already evaluated."""
    return float(image_delta_e00(compose_hat(s.img, s.alpha, values), s.i_star))


# --------------------------------------------------------------------------- #
# the board
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(
    head: A.AffineOnlyHead,
    samples: Sequence[EvalSample],
    *,
    bank: LutBank,
    lib: LibraryValues | None,
    lib_mean_volume: torch.Tensor | None,
    bucket_pools: Mapping[str, Sequence[str]] | None,
    controls: Mapping[str, ZCache | SyntheticZCache],
    args: argparse.Namespace,
    quick: bool = False,
    device: Any = "cpu",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-sample rows + the columns ``build_board`` does not own.

    Order matters for one thing only: the result is written before any optional
    stage, so a failure in the interpolation protocol cannot destroy the
    headline that was already computed (campaign rule "结果落盘先于可选阶段").
    """
    cfg = head.cfg
    x_grid = uniform_grid(args.grid_n, device=device)
    sampler = QuerySampler(seed=cfg.seed, q=args.unseen_q)
    x_unseen = sampler.sample_heldout(1, args.unseen_q, device=device)[0]

    rows: list[dict[str, Any]] = []
    for s in samples:
        # one f̂(I) per sample: the headline, the strata and the diagnostics all
        # read the same tensor, so no two columns can disagree about the arm.
        f_img = head.transform_image(s.z, s.img, point_chunk=args.point_chunk)
        i_hat = compose_hat(s.img, s.alpha, f_img)
        e_arm = float(image_delta_e00(i_hat, s.i_star))
        row: dict[str, Any] = {
            "sample_id": s.row.sample_id,
            "winner_confidence": s.row.winner_confidence,
            "task_type": s.row.task_type,
            "lut_id": s.lut_id,
            "source_image_id": s.row.source_image_id,
            "E_arm": e_arm,
            "E_B0_identity": float(image_delta_e00(s.img, s.i_star)),
        }
        with torch.no_grad():
            f_grid = head.transform(s.z.reshape(1, -1), x_grid)[0]
            l_grid = bank.apply(x_grid, s.lut_id)
            row["grid_error"] = float(function_distance(f_grid, l_grid))
            colors, weights = image_histogram_colors(s.img, bits=5, top_k=args.hist_top_k)
            row["img_error"] = float(function_distance(
                head.transform(s.z.reshape(1, -1), colors)[0], bank.apply(colors, s.lut_id),
                weights))
            row["unseen_color_error"] = float(function_distance(
                head.transform(s.z.reshape(1, -1), x_unseen)[0], bank.apply(x_unseen, s.lut_id)))
        if isinstance(s.alpha, torch.Tensor):
            row.update(locality_errors(i_hat, s.i_star, s.img, s.alpha))
        rows.append(row)

    # -- trivial baselines (section 4.C) ------------------------------------
    if lib is not None and not quick:
        _add_library_baselines(rows, samples, bank=bank, lib=lib,
                               lib_mean_volume=lib_mean_volume,
                               bucket_pools=bucket_pools, args=args, device=device)

    # -- negative controls (section 4.D) ------------------------------------
    if not quick:
        _add_controls(rows, samples, head, controls, x_grid, args=args, device=device)

    extra: dict[str, Any] = {}
    if not quick:
        extra = _p1_and_arm_columns(head, samples, bank=bank, args=args, device=device)
    return rows, extra


@torch.no_grad()
def _add_library_baselines(rows: list[dict[str, Any]], samples: Sequence[EvalSample], *,
                           bank: LutBank, lib: LibraryValues,
                           lib_mean_volume: torch.Tensor | None,
                           bucket_pools: Mapping[str, Sequence[str]] | None,
                           args: argparse.Namespace, device: Any) -> None:
    x9 = lib.x
    # B4 / B6 pick their LUT on the 9^3 grid in dE76 (section 4.C's own protocol);
    # the column that reaches the board is then re-measured in the headline
    # quantity (dE00 on the image with GT alpha).  The two are not interchangeable.
    targets = {s.row.sample_id: bank.apply(x9, s.lut_id) for s in samples}
    b4 = oracle_lut_ids(lib, targets, metric=args.select_metric)
    by_lut = {s.lut_id: bank.apply(x9, s.lut_id) for s in samples if s.lut_id in lib.index}
    b6 = oracle_lut_ids(lib, by_lut, metric=args.select_metric, exclude_self=True)

    draws_b2 = library_random_draw(lib.lut_ids, len(samples), repeats=args.repeats,
                                   seed=args.seed)
    draws_b3 = None
    if bucket_pools is not None:
        draws_b3 = bucket_draw([s.minor or "" for s in samples], bucket_pools,
                               repeats=args.repeats, seed=args.seed)

    for i, (s, row) in enumerate(zip(samples, rows)):
        if lib_mean_volume is not None:
            row["E_B1_libmean"] = lut_headline_error(
                s, _apply_volume_image(lib_mean_volume, s.img))
        row["E_B2_librandom_repeats"] = [
            lut_headline_error(s, bank.apply_image(s.img, d[i])) for d in draws_b2]
        if draws_b3 is not None:
            vals = [lut_headline_error(s, bank.apply_image(s.img, d[i]))
                    for d in draws_b3 if d[i] is not None]
            row["E_B3_bucket_retrieval_repeats"] = vals
            row["n_b3_missing_bucket"] = args.repeats - len(vals)
        lid4 = b4[s.row.sample_id][0]
        row["E_B4_oracle"] = lut_headline_error(s, bank.apply_image(s.img, lid4))
        row["B4_lut_id"] = lid4
        if s.lut_id in b6:
            row["E_B6_libfill"] = lut_headline_error(s, bank.apply_image(s.img, b6[s.lut_id][0]))


def _apply_volume_image(volume: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
    return apply_lut_volume(volume, img.permute(1, 2, 0)).permute(2, 0, 1)


@torch.no_grad()
def _add_controls(rows: list[dict[str, Any]], samples: Sequence[EvalSample],
                  head: A.AffineOnlyHead, controls: Mapping[str, ZCache | SyntheticZCache],
                  x_grid: torch.Tensor, *, args: argparse.Namespace, device: Any) -> None:
    """N1 / N2 / N3, both columns each (Δ and M) -- section 4.D says both or neither."""
    for tag, key in CONTROL_ROW_KEYS.items():
        src = controls.get(tag)
        if src is None:
            continue
        for s, row in zip(samples, rows):
            if s.row.sample_id not in src:
                continue
            z_ctrl = src.get(s.row.sample_id).to(device)
            row[f"E_{key}"] = headline_error(head, s, z_ctrl, point_chunk=args.point_chunk)[0]
            with torch.no_grad():
                f_true = head.transform(s.z.reshape(1, -1), x_grid)[0]
                f_ctrl = head.transform(z_ctrl.reshape(1, -1), x_grid)[0]
            row[f"M_{key}"] = float(function_distance(f_true, f_ctrl))


def _p1_and_arm_columns(head: A.AffineOnlyHead, samples: Sequence[EvalSample], *,
                        bank: LutBank, args: argparse.Namespace, device: Any
                        ) -> dict[str, Any]:
    """The P1 four + this arm's three, plus the IP-A/IP-B detail columns."""
    z_all = torch.stack([s.z for s in samples], dim=0).to(device)
    shared = A.assert_shared_geometry_identical(
        head, z_all, n_probe=min(head.cfg.shared_geom_probes, z_all.shape[0]),
        raise_on_fail=not args.allow_assertion_failure)
    linear = A.assert_affine_linearity(
        head, z_all, n_pairs=min(head.cfg.linearity_pairs, z_all.shape[0] // 2),
        grid_n=args.grid_n, raise_on_fail=not args.allow_assertion_failure)
    degen = A.degenerate_weight_rate(head, z_all, uniform_grid(args.grid_n, device=device))

    pairs_a = _same_source_pairs(samples, limit=args.interp_pairs)
    ip_a = A.interp_ip_a(head, [(a.z, a.lut_id, b.z, b.lut_id) for a, b in pairs_a],
                         bank, grid_n=args.grid_n)
    pairs_b = _random_pairs(samples, n=args.ipb_paths, seed=args.seed)
    ip_b = A.interp_ip_b(head, [(a.z, b.z) for a, b in pairs_b], k_steps=args.ipb_k,
                         grid_n=args.grid_n)
    return A.arm_criteria_columns(interp_a=ip_a, interp_b=ip_b, shared_geom=shared,
                                  linearity=linear, degenerate=degen)


def _same_source_pairs(samples: Sequence[EvalSample], *, limit: int
                       ) -> list[tuple[EvalSample, EvalSample]]:
    """IP-A pairs: one pair per source image, two different LUTs on the same photo.

    Section 4.F: V_what normal-only has 120 sources with >= 2 samples, which is
    where the protocol's 120 pairs come from.
    """
    by_source: dict[str, list[EvalSample]] = {}
    for s in samples:
        by_source.setdefault(s.row.source_image_id, []).append(s)
    out: list[tuple[EvalSample, EvalSample]] = []
    for group in by_source.values():
        pair = next(((a, b) for i, a in enumerate(group) for b in group[i + 1:]
                     if a.lut_id != b.lut_id), None)
        if pair is not None:
            out.append(pair)
        if len(out) >= limit:
            break
    return out[:limit]


def _random_pairs(samples: Sequence[EvalSample], *, n: int, seed: int
                  ) -> list[tuple[EvalSample, EvalSample]]:
    rng = random.Random(seed)
    out: list[tuple[EvalSample, EvalSample]] = []
    if len(samples) < 2:
        return out
    for _ in range(int(n)):
        a, b = rng.sample(list(samples), 2)
        out.append((a, b))
    return out


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def make_target_fn(bank: LutBank, lut_ids: Sequence[str]) -> Callable[[torch.Tensor], torch.Tensor]:
    """``x (B, Q, 3) -> y = L_{l_i}(x_i)``, the same operator the dataset used."""

    def target(x: torch.Tensor) -> torch.Tensor:
        return torch.stack([bank.apply(x[i], lut_ids[i]) for i in range(x.shape[0])], dim=0)

    return target


def train(head: A.AffineOnlyHead, *, args: argparse.Namespace, bank: LutBank,
          z_train: ZCache | SyntheticZCache, train_rows: Sequence[IndexRow], run_dir: Path,
          quick_eval_fn: Callable[[int], dict[str, Any]] | None,
          device: Any) -> dict[str, Any]:
    cfg = head.cfg
    opt = A.build_optimizer(head, cfg)
    # --total-steps 0 means "the horizon the population implies"; the config
    # already resolved it (epochs * ceil(n / B)).
    total_steps = int(args.total_steps) or int(cfg.total_steps)
    sched = A.build_scheduler(opt, cfg, total_steps)
    sampler = QuerySampler(seed=cfg.seed, q=cfg.queries)
    steps = JsonlWriter(run_dir / "steps.jsonl")
    quick = JsonlWriter(run_dir / "quick_eval.jsonl")

    n = len(train_rows)
    # predeclared criterion with a run-time assertion: the epoch length and the
    # population it is counted over may not disagree.
    per_epoch = K.assert_steps_per_epoch(cfg.steps_per_epoch, n_train=n,
                                         batch_samples=cfg.batch_samples,
                                         where="affonly steps_per_epoch")
    order_rng = random.Random(cfg.seed)
    order = list(range(n))
    best: dict[str, Any] = {"headline": None, "step": None}
    first_row: dict[str, Any] | None = None
    t0 = time.time()

    for step in range(total_steps):
        pos = step % per_epoch
        if pos == 0:
            order_rng.shuffle(order)
        epoch = step / per_epoch
        idx = order[pos * cfg.batch_samples : (pos + 1) * cfg.batch_samples]
        if len(idx) < 2:                      # a 1-sample tail breaks cross-sample std
            continue
        batch = [train_rows[i] for i in idx]
        z = z_train.batch([r.sample_id for r in batch]).to(device)
        row = A.train_step(head, opt, z, sampler, make_target_fn(bank, [r.lut_id for r in batch]),
                           epoch=epoch, lut_ids=[r.lut_id for r in batch], scheduler=sched,
                           cfg=cfg, device=device)
        row["step"] = step
        row["wall_s"] = round(time.time() - t0, 3)
        if first_row is None:
            first_row = dict(row)
            record_step_witness(row)          # tier 3 of the board-time resolution
        if step % args.log_every == 0 or step == total_steps - 1:
            steps.write(row)

        if quick_eval_fn is not None and args.eval_every > 0 and (
                (step + 1) % args.eval_every == 0 or step == total_steps - 1):
            qe = quick_eval_fn(step)
            qe["step"] = step
            quick.write(qe)
            h = ((qe.get("contexts") or {}).get("all") or {}).get(
                "headline_normal_only", {}).get("mean")
            # checkpoint selection: the quick-eval hard gate + headline, never val loss
            if h is not None and (best["headline"] is None or h < best["headline"]):
                best = {"headline": float(h), "step": int(step)}
                torch.save({"state_dict": head.state_dict(), "step": step,
                            "headline_normal_only": float(h), "config": head.config},
                           run_dir / "best.pt")

    steps.close()
    quick.close()
    torch.save({"state_dict": head.state_dict(), "step": total_steps,
                "config": head.config}, run_dir / "last.pt")
    return {"first_step_row": first_row, "best": best, "steps_per_epoch": per_epoch,
            "n_train_rows": n, "wall_s": round(time.time() - t0, 3)}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="run_affonly_arm",
        description="EPR-025 AFFONLY: affine-only conditional generation (12N+12)")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--arm", default=A.ARM, choices=[A.ARM])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=20260810)

    # --- the experiment variable (proposal section 3.4-(12)) ----------------
    ap.add_argument("--share", default="geo_opacity", choices=list(A.SHARE_CHOICES))
    ap.add_argument("--global-affine", default="affine", choices=list(A.GLOBAL_AFFINE_CHOICES))
    # both spellings: the proposal writes --shared-lr-scale / --num-gauss, the
    # shared-layer contract writes --shared-geom-lr-scale / --n-gauss.  One dest.
    ap.add_argument("--shared-geom-lr-scale", "--shared-lr-scale", dest="shared_lr_scale",
                    type=float, default=0.1)
    ap.add_argument("--n-gauss", "--num-gauss", dest="n_gauss", type=int, default=48)
    ap.add_argument("--cond-dim", type=int, default=64)
    ap.add_argument("--gen-width", "--hidden", dest="gen_width", type=int, default=128,
                    choices=[128, 64])
    ap.add_argument("--mu-init", default="grid", choices=list(A.MU_INIT_CHOICES))
    ap.add_argument("--zero-init-heads", dest="zero_init_heads", action="store_true", default=True)
    ap.add_argument("--no-zero-init-heads", dest="zero_init_heads", action="store_false")
    ap.add_argument("--clamp", default="two", choices=list(CLAMP_FLAG_CHOICES))
    ap.add_argument("--lambda-hc", type=float, default=10.0)
    ap.add_argument("--lambda-sparse", type=float, default=0.001)
    ap.add_argument("--loss-level", type=int, default=3, choices=[1, 2, 3, 4],
                    help="1 = the single L1 term (lambda_hc = lambda_sparse = "
                         "lambda_mono = 0, asserted before step 0)")

    # --- the shared EPR-030 caliber (--data / --zcache-root-l8 / --batch-split
    # --- / --base-lr); --base-lr is registered above this arm's own block
    K.add_caliber_arguments(ap, base_lr=False)
    ap.add_argument("--base-lr", type=float, default=K.BASE_LR)

    # --- schedule (frozen; flags exist so the record carries them) ----------
    ap.add_argument("--total-steps", type=int, default=0,
                    help="0 = epochs * ceil(n_train / B) on the MEASURED "
                         "population of --data (v2seg: 117,440)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-samples", type=int, default=None,
                    help="override B; the default comes from --batch-split")
    ap.add_argument("--queries", type=int, default=None,
                    help="override Q; the default comes from --batch-split")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=2936)
    ap.add_argument("--quick-n", type=int, default=32)

    # --- data ---------------------------------------------------------------
    ap.add_argument("--dataset-root", default=str(DATASET_ROOT))
    ap.add_argument("--bank-dir", default=str(BANK_DIR))
    ap.add_argument("--maskview-root", default=MASKVIEW_ROOT)
    ap.add_argument("--base-checkpoint", default=V2SEG_CHECKPOINT)
    ap.add_argument("--tokenizer", default=None, help="default: --base-checkpoint")
    ap.add_argument("--readout", default="seg_color", choices=list(WHATB_READOUT_KINDS))
    ap.add_argument("--eval-split", default="V_what",
                    choices=["V_what", "T_final", "T_lut_unseen"])
    ap.add_argument("--eval-n", type=int, default=0, help="0 = the whole normal-only split")
    ap.add_argument("--train-n", type=int, default=0, help="0 = all 93934 normal-only rows")
    ap.add_argument("--z-source", default="cache", choices=["cache", "synthetic"])
    for tag in CONTROL_TAGS:
        ap.add_argument(f"--z-cache-{tag}", default=None,
                        help=f"packed z cache for control {tag!r} on the eval split")
    ap.add_argument("--z-cache-train", default=None,
                    help="leaf dir of the train cache; derived from --z-cache "
                         "when that is given")
    ap.add_argument("--z-cache", default=None,
                    help="root of the shared cache (q3vl/whatb/zcache.py): the "
                         "leaves are <root>/<split>__<tag>")

    # --- baselines / criteria ----------------------------------------------
    ap.add_argument("--lib-rows", type=int, default=2500,
                    help="train index rows sampled to form Lib_tr (section 4.C protocol)")
    ap.add_argument("--lib-full", action="store_true", help="use all train lut_ids instead")
    ap.add_argument("--libmean-grid", type=int, default=33)
    ap.add_argument("--select-metric", default="de76", choices=["de76", "de00"])
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--grid-n", type=int, default=17)
    ap.add_argument("--unseen-q", type=int, default=4913)
    ap.add_argument("--hist-top-k", type=int, default=4096)
    ap.add_argument("--interp-pairs", type=int, default=120)
    ap.add_argument("--ipb-paths", type=int, default=20)
    ap.add_argument("--ipb-k", type=int, default=20)
    ap.add_argument("--point-chunk", type=int, default=None)
    ap.add_argument("--bucket-pools", default=None,
                    help="cached {minor: [lut_id]} json; built from train records if absent")
    ap.add_argument("--no-bucket-pools", action="store_true",
                    help="skip B3 (the board will then refuse to publish -- smoke only)")

    # --- modes --------------------------------------------------------------
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the caliber, write run_setup.json, then stop")
    ap.add_argument("--resume", default=None, help="state_dict checkpoint to load first")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU run: never publishable, board carries published=false")
    ap.add_argument("--allow-assertion-failure", action="store_true",
                    help="record assertions 1/2 instead of raising (diagnosis only)")
    ap.add_argument("--skip-colorspan-assert", action="store_true",
                    help="smoke only: the frozen block forbids it on a real run")
    return ap


def config_from_args(args: argparse.Namespace, *, train_n: int | None = None
                     ) -> A.AffineOnlyConfig:
    """``Namespace -> AffineOnlyConfig``.  ``train_n`` is MEASURED by the caller.

    The horizon is a function of the population on disk: ``batch_samples`` /
    ``queries`` come from ``--batch-split`` through the one shared table, and
    ``total_steps`` defaults to ``epochs * ceil(n / B)``.
    """
    b, q = K.parse_batch_split(args.batch_split)
    b = int(args.batch_samples) if args.batch_samples else b
    q = int(args.queries) if args.queries else q
    # --train-n is a CAP on the rows, not the population size: the population is
    # measured by the caller and handed in.
    n = int(train_n if train_n is not None else A.FROZEN["train_n"])
    spe = K.steps_per_epoch_of(n, b)
    return A.AffineOnlyConfig(
        data=args.data,
        train_n=n,
        share=args.share,
        global_affine=args.global_affine,
        n_gauss=args.n_gauss,
        cond_dim=args.cond_dim,
        gen_width=args.gen_width,
        zero_init_heads=args.zero_init_heads,
        mu_init=args.mu_init,
        clamp=args.clamp,
        lambda_hc=args.lambda_hc,
        lambda_sparse=args.lambda_sparse,
        loss_level=args.loss_level,
        base_lr=args.base_lr,
        shared_lr_scale=args.shared_lr_scale,
        batch_samples=b,
        queries=q,
        epochs=args.epochs,
        total_steps=int(args.total_steps) or spe * int(args.epochs),
        seed=args.seed,
    )


def _colorspan_assertion(args: argparse.Namespace, rows: Sequence[IndexRow]) -> dict[str, Any]:
    """Frozen block: run it BEFORE the dataloader, on this split's colour texts."""
    path = args.tokenizer or args.base_checkpoint
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    rng = random.Random(args.seed)
    picked = rng.sample(list(rows), min(256, len(rows)))
    # ONE reader for both sources: an L8 row carries its <color> inline (the
    # manifest), an sft2seg row in its record shard -- splits.color_texts_of.
    texts = color_texts_of(picked)
    return assert_color_span_encoding(tok, texts, tokenizer_path=str(path))


def _load_records(rows: Sequence[IndexRow]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for r in rows:
        out[r.sample_id] = read_record(r)
    return out


def _bucket_pools(args: argparse.Namespace, train_rows: Sequence[IndexRow], run_dir: Path
                  ) -> dict[str, list[str]] | None:
    from q3vl.whatb.splits import bucket_pools as _pools

    if args.no_bucket_pools:
        return None
    if args.bucket_pools:
        return json.loads(Path(args.bucket_pools).read_text(encoding="utf-8"))
    cached = run_dir / "bucket_pools.json"
    if cached.is_file():
        return json.loads(cached.read_text(encoding="utf-8"))
    from q3vl.whatb.splits import iter_records

    pools = _pools(iter_records(list(train_rows)))
    _write_json(cached, pools)
    return pools


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(list(argv) if argv is not None else None)
    run_dir = Path(args.run_dir)
    if str(run_dir).startswith("/mnt/nfs"):
        raise SystemExit(
            "--run-dir must be on local disk: a gate that tests a path on the hard "
            "NFS mount goes into D state and cannot be killed "
            "(waves/epr018_023_wave.sh:62)")
    synthetic = args.z_source == "synthetic"
    if synthetic and not args.smoke:
        raise SystemExit("--z-source synthetic is smoke-only: it is not a condition")
    if args.skip_colorspan_assert and not args.smoke:
        raise SystemExit(
            "--skip-colorspan-assert is a smoke-only flag; the frozen block requires "
            "the tokenizer-vs-own-encoder check before the dataloader is built")
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    # ---- data -------------------------------------------------------------
    root = Path(args.dataset_root)
    # the training set is normal-only (frozen block item 1); ``Lib_tr`` and the B3
    # bucket pools are libraries of LUTs, not training data or GT, and the
    # section 4.C / frozen-block numbers they must reproduce (1137 ids from 2500
    # index rows; 77 buckets / 3149 ids) were measured on the whole train index.
    train_index_all = load_index("train", root)
    # --data's population is MEASURED (each source counted against its own
    # on-disk declaration); no merged n is written down anywhere.
    train_rows = (normal_only(train_index_all) if args.data == "v2seg"
                  else K.train_normal_rows(args.data, split="train", root=root))
    if args.train_n:
        train_rows = train_rows[: args.train_n]
    eval_rows_all = normal_only(load_index(args.eval_split, root))
    eval_rows = eval_rows_all[: args.eval_n] if args.eval_n else eval_rows_all

    cfg = config_from_args(args, train_n=len(train_rows))
    head = A.AffineOnlyHead(cfg).to(device)
    if args.resume:
        blob = torch.load(args.resume, map_location=device, weights_only=False)
        head.load_state_dict(blob["state_dict"] if "state_dict" in blob else blob)

    colorspan = None
    if not args.skip_colorspan_assert:
        colorspan = _colorspan_assertion(args, train_rows if not args.eval_only else eval_rows)

    train_zrec: dict[str, Any] | None = None
    if args.data != "v2seg" and not synthetic:
        if not args.z_cache:
            raise SystemExit(
                f"--data {args.data} unions a second corpus, whose z lives in its "
                "own cache root; pass --z-cache <root> (the leaves are derived) "
                "rather than a single --z-cache-train leaf")
        z_train, train_zrec = K.open_train_z(
            args.z_cache, "train", data=args.data,
            checkpoint=args.base_checkpoint, readout_kind=args.readout,
            zcache_root_l8=args.zcache_root_l8, seed=args.seed)
    else:
        z_train = open_z(args.z_cache_train, tag="none", checkpoint=args.base_checkpoint,
                         readout=args.readout, split="train", root=args.z_cache,
                         synthetic=synthetic, seed=args.seed)
    controls = {
        tag: open_z(getattr(args, f"z_cache_{tag}"), tag=tag,
                    checkpoint=args.base_checkpoint, readout=args.readout,
                    split=args.eval_split, root=args.z_cache,
                    synthetic=synthetic, seed=args.seed)
        for tag in CONTROL_TAGS
        if synthetic or args.z_cache or getattr(args, f"z_cache_{tag}")
    }
    if "none" not in controls:
        raise SystemExit(
            "the true condition on the eval split is required: pass --z-cache "
            "<root> or --z-cache-none <leaf dir>")

    # the shared caliber, asserted (steps_per_epoch == ceil(n / B)) and recorded
    caliber = K.horizon_record(
        data=args.data, n_train=len(train_rows), batch_split=cfg.batch_split,
        batch_samples=cfg.batch_samples, queries_per_sample=cfg.queries,
        steps_per_epoch=cfg.steps_per_epoch, total_steps=cfg.total_steps,
        base_lr=cfg.base_lr, epochs=cfg.epochs,
        zcache_root_l8=args.zcache_root_l8, loss_level=cfg.loss_level,
        lambda_hc=cfg.lambda_hc_effective,
        lambda_sparse=cfg.lambda_sparse_effective)
    caliber["train_source_facts"] = (
        None if args.data == "v2seg" else
        train_source_facts(args.data, split="train", root=root))
    caliber["z_cache_train"] = train_zrec
    if args.dry_run:
        _write_json(run_dir / "run_setup.json",
                    {"arm": A.ARM, "epr": A.EPR, "dry_run": True,
                     "argv": vars(args), "config": cfg.to_dict(),
                     "caliber": caliber})
        print(json.dumps({"arm": A.ARM, "dry_run": True,
                          "data": args.data, "n_train": len(train_rows),
                          "batch_split": cfg.batch_split,
                          "colours_per_step": cfg.colors_per_step,
                          "steps_per_epoch": cfg.steps_per_epoch,
                          "total_steps": cfg.total_steps,
                          "base_lr": cfg.base_lr,
                          "loss_level": cfg.loss_level}, indent=2))
        return 0

    bank = LutBank(args.bank_dir)
    images = ImageReader()
    alphas = AlphaReader(args.eval_split, args.maskview_root)
    records = _load_records(eval_rows)
    samples = load_eval_samples(eval_rows, bank=bank, images=images, alphas=alphas,
                                z_true=controls["none"], records=records, device=device)
    quick_samples = samples[: args.quick_n]
    pools = _bucket_pools(args, train_index_all, run_dir)

    lib_ids = library_ids(train_index_all, n_rows=args.lib_rows, seed=args.seed,
                          full=args.lib_full)
    x9 = uniform_grid(9, device=device)
    lib = LibraryValues.build(bank, lib_ids, x9)
    lib_mean = mean_lut_volume(bank, lib_ids, grid_n=args.libmean_grid, device=device)

    # ---- run_setup (written BEFORE the first step) ------------------------
    z_probe0 = torch.stack([s.z for s in samples[: max(2, min(8, len(samples)))]],
                           dim=0).to(device)
    setup = A.run_setup(head, extra={
        "argv": vars(args),
        "step0": A.step0_witness(head, z_probe0, grid_n=args.grid_n),
        "source_sha256": {str(p): _sha256(p) for p in _SOURCE_FILES},
        "base_checkpoint": args.base_checkpoint,
        "readout": args.readout,
        "colorspan_check": colorspan,
        "z_source": args.z_source,
        "caliber": caliber,
        "z_caches": {t: s.facts() for t, s in controls.items()} | {"train": z_train.facts()},
        "splits": {"train": split_facts(train_index_all),
                   args.eval_split: split_facts(eval_rows_all),
                   "n_train_used": len(train_rows), "n_eval_used": len(eval_rows)},
        "bank": bank.facts(),
        "library": {"n_lut": len(lib_ids),
                    "protocol": ("all train lut_ids" if args.lib_full else
                                 f"{args.lib_rows} train index rows -> Lib_tr (section 4.C)"),
                    "source": "train index, all rows (a LUT library is not GT)",
                    "B1_libmean_grid": args.libmean_grid,
                    "select_metric": args.select_metric},
        "bucket_pools": {"n_buckets": 0 if pools is None else len(pools),
                         "n_lut_ids": 0 if pools is None else
                         len({x for v in pools.values() for x in v}),
                         "source": "train records' own `minor` (never splits_presets.csv)"},
        "smoke": bool(args.smoke),
    })
    _write_json(run_dir / "run_setup.json", setup)

    # ---- the degeneracy guard: at the FIRST evaluation, whichever it is ----
    # It is deliberately not tied to --eval-every: a run with the quick eval
    # turned off would otherwise reach the board without ever being asked
    # whether its output is a constant field (PRND / CONDINST, 2.6 GPU-hours).
    state = {"guarded": False, "first_board": True}

    def guard_once(where: str) -> None:
        if state["guarded"]:
            return
        z_probe = torch.stack([s.z for s in quick_samples], dim=0).to(device)
        A.assert_not_degenerate(head, z_probe, where=where)
        state["guarded"] = True

    def quick_eval_fn(step: int) -> dict[str, Any]:
        head.eval()
        guard_once(f"quick_eval@step{step}")
        # "定义了没接线" has cost this campaign five times: the FIRST quick eval
        # runs the whole gated board (library, bucket pools, the three control
        # caches, the P1 interpolation columns) and asserts the entire
        # pre-registered table.  A missing column then fails in epoch 1 instead
        # of after the 117,440-step horizon.  Later quick evals are the cheap
        # headline-only board the checkpoint selection needs.
        full = bool(state["first_board"])
        if full:
            rows, extra = evaluate(head, quick_samples, bank=bank, lib=lib,
                                   lib_mean_volume=lib_mean, bucket_pools=pools,
                                   controls=controls, args=args, device=device)
        else:
            rows, extra = evaluate(head, quick_samples, bank=bank, lib=None,
                                   lib_mean_volume=None, bucket_pools=None,
                                   controls={}, args=args, quick=True, device=device)
        head.train()
        board = A.build_arm_board(rows, split=args.eval_split, extra_columns=extra,
                                  seed=args.seed, published=False)
        board["quick"] = True
        if full:
            # A truncated eval set (--eval-n / --quick-n on a smoke) has no
            # same-source pair, so the P1 interpolation columns are empty for a
            # reason that is not "not wired".  The skip is written onto the
            # artefact with its reason -- it is never silent, and a real run
            # (--eval-n 0, no --smoke) always asserts.
            skip = None
            if args.smoke:
                skip = "--smoke"
            elif args.eval_n or args.quick_n < 64:
                skip = f"eval_n={args.eval_n} quick_n={args.quick_n} (truncated)"
            board["first_board_assertion"] = (
                {"skipped": skip, "required": A.required_columns()} if skip else
                assert_criteria_ran(board, A.EPR, required=A.required_columns()))
            state["first_board"] = False
        return board

    train_report: dict[str, Any] = {}
    if not args.eval_only:
        train_report = train(head, args=args, bank=bank, z_train=z_train,
                             train_rows=train_rows, run_dir=run_dir,
                             quick_eval_fn=quick_eval_fn, device=device)

    # ---- the board --------------------------------------------------------
    head.eval()
    guard_once("final_eval")
    rows, extra = evaluate(head, samples, bank=bank, lib=lib, lib_mean_volume=lib_mean,
                           bucket_pools=pools, controls=controls, args=args, device=device)
    board = A.build_arm_board(rows, split=args.eval_split, extra_columns=extra, seed=args.seed,
                              published=not (args.smoke or synthetic),
                              z_source=args.z_source, train=train_report,
                              alpha=alphas.facts())
    _write_json(run_dir / "metrics.json", board)          # result before any optional stage
    report = A.publish_board(board, steps_row=train_report.get("first_step_row"),
                             steps_path=run_dir / "steps.jsonl",
                             eval_only=args.eval_only, loss_level=cfg.loss_level)
    board["publication"] = report
    _write_json(run_dir / "metrics.json", board)
    _write_json(run_dir / "publication.json", report)

    images.close()
    hn = board["contexts"]["all"]["headline_normal_only"]
    print(f"[{A.ARM}] headline_normal_only mean={hn.get('mean')} n={hn.get('n')} "
          f"-> {run_dir/'metrics.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
