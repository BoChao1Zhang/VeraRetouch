#!/usr/bin/env python
"""Runner for QDUAL (EPR-029): the Gaussian-query dual-condition decoder.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/bc/VeraRetouch \\
      python -m q3vl.whatb.scripts.run_qdual_arm --smoke --z-source synthetic \\
      --out-root /tmp/qdual --run-name smoke

What this script is responsible for (the arm's maths lives in
``q3vl/whatb/arms/qdual.py``; every shared piece is imported, never re-written):

1. **run_setup.json** -- every flag, the frozen-block numbers with the value this
   run actually used, the ``sha256`` of the arm module and of this file (the
   process-start source freeze), the degeneracy thresholds, the colour-span
   start-up assertion, the readout spec, the carrier config and the parameter
   count.
2. **training** -- Adam, cosine annealing from 1e-3, a 0.1x group for
   ``q_emb`` / colour PE / ``theta_base``; B = 32 samples x Q = 256 colours =
   8192 colours per step; 2936 steps/epoch x 40 epochs = 117,440 steps; GLUT's
   hard-example mining (epoch 5->20, 10%->40%, within-batch top-r).  Every step
   writes ``steps.jsonl``; the first row carries every pre-registered loss
   column and is also recorded as the in-process witness.
3. **the first quick eval** -- the degenerate-solution guard runs there and
   exits the process (``SystemExit(2)``) if the transform is flat across query
   colours, is the identity, or is the same for every sample.  Where-side
   PRND/CONDINST cost 2.6 GPU-hours by not having this.
4. **the board** -- per-sample rows -> ``criteria.build_board`` ->
   ``publish.assert_publishable`` (three-tier steps row, every pre-registered
   criterion with n > 0, the headline present) -> ``metrics.json``.

Data seams (documented because the shared layer does not own them yet)
----------------------------------------------------------------------
``--z-source zcache`` reads the frozen VLM's ``<seg_color>`` read-out from

    <zcache>/z_<split>_<control>.npz     arrays: sample_id (U), z (n, 2560)
                                                 [or (n, K, 2560) for qtok]
    <zcache>/z_<split>_<control>.json    {"checkpoint", "readout_kind",
                                          "readout_qtok", "split", "control"}

with ``control`` in ``none`` / ``shuffle`` / ``irrelevant`` / ``const``.  The
``checkpoint`` field is asserted equal to ``--base-checkpoint`` at start-up and
``readout_kind`` equal to ``--readout`` (HANDOFF section 4.H); the three controls
must have been produced by **re-generating the reasoning**, which is a property
of the cache, recorded in its own json and copied into ``run_setup``.

``--z-source synthetic`` fabricates deterministic conditions and fields for CPU
smoke runs.  Such a run is marked ``synthetic: true`` and its board is written
with ``published = False``, so it can never be mistaken for a result.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field as _dcfield
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from q3vl.whatb import caliber as K
from q3vl.whatb import criteria as crit
from q3vl.whatb import publish as pub
from q3vl.whatb.arms import qdual as qd
from q3vl.whatb.colorimetry import delta_e00, srgb_to_lab
from q3vl.whatb.colorspan import assert_color_span_encoding
from q3vl.whatb.guards import (
    DegeneracyThresholds,
    degeneracy_check_ran,
    record_step_witness,
)
from q3vl.whatb.lutdata import BANK_DIR, LutBank, apply_lut_volume
from q3vl.whatb.queries import (
    BATCH_SAMPLES,
    COLORS_PER_STEP,
    QUERIES_PER_SAMPLE,
    QuerySampler,
    heldout_color_levels,
    uniform_grid,
)
from q3vl.whatb.readout import WhatReadoutSpec
from q3vl.whatb.zcache import ZCacheDir
from q3vl.whatb.splits import (
    DATASET_ROOT,
    IndexRow,
    dataset_version,
    bucket_pools,
    iter_records,
    load_index,
    normal_only,
    ro_path,
    split_facts,
    train_source_facts,
)
from q3vl.where.upsample import area_resize

# --------------------------------------------------------------------------- #
# frozen numbers (the run asserts against these rather than trusting the flags)
# --------------------------------------------------------------------------- #
#: the ORIGINAL口径 (v20260804) -- what the published boards were run on.
#: The active口径 may differ; ``frozen_block_record`` reports both.
TRAIN_NORMAL_N = dataset_version("v20260804").train_normal_n
STEPS_PER_EPOCH = 2936
TOTAL_STEPS = 117_440
EPOCHS = 40
BASE_LR = 1e-3
SEED = 20260810
GEOMETRY_LR_SCALE = 0.1
BASE_CHECKPOINT = "/home/bc/data/runs/q3vl_base_sft_v2seg_20260814/checkpoint-4976"
MASKVIEWS_ROOT = Path("/mnt/nfs-ro/bc/data/datasets/where_a-20260805/maskviews")
CONTROLS: tuple[str, ...] = ("none", "shuffle", "irrelevant", "const")
CONTROL_ROW_KEYS = {"shuffle": ("E_N1_shuffle", "M_N1_shuffle"),
                    "irrelevant": ("E_N2_irrelevant", "M_N2_irrelevant"),
                    "const": ("E_N3_const", "M_N3_const")}
#: the mean-transform (B1) is materialised as a LUT volume on this grid
B1_GRID = 33


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# --------------------------------------------------------------------------- #
# samples
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    """One evaluation/training row: index fields plus the record's ``minor``."""

    sample_id: str
    split: str
    lut_id: str
    source_image_id: str
    task_type: str
    winner_confidence: str
    minor: str | None = None
    raw: Mapping[str, Any] = _dcfield(default_factory=dict, repr=False)

    @property
    def is_style(self) -> bool:
        return self.task_type == "style"


def samples_from_index(rows: Sequence[IndexRow]) -> list[Sample]:
    return [Sample(sample_id=r.sample_id, split=r.split, lut_id=r.lut_id,
                   source_image_id=r.source_image_id, task_type=r.task_type,
                   winner_confidence=r.winner_confidence, raw=r.raw)
            for r in rows]


def attach_minor(samples: Sequence[Sample], rows: Sequence[IndexRow]) -> None:
    """Read the eval split's records once and copy their ``minor`` (B3 bucket)."""
    for s, rec in zip(samples, iter_records(list(rows))):
        s.minor = rec.get("minor")


# --------------------------------------------------------------------------- #
# condition store
# --------------------------------------------------------------------------- #
class ZStore:
    """``Sample -> z`` per control tag: a thin adapter over the ONE shared cache.

    The reader, the on-disk layout and the three start-up assertions live in
    :mod:`q3vl.whatb.zcache` (HANDOFF 步骤 0-7, "共同依赖只写一份").  This class
    only maps this arm's ``Sample`` objects onto it and keeps the ``--z-expand``
    shape contract.

    ``mode == "synthetic"`` derives a deterministic vector from
    ``(lut_id, sample_id, control)`` so a CPU smoke run has a *learnable*
    condition (the LUT part is shared by every sample of that LUT) without
    touching the VLM.  It is not a read-out and no board built on it publishes.

    Ruling 11.1-5 is enforced by the shared reader by **rejection**: a cache that
    is not fp32 raises.  The private reader this class replaced did
    ``np.asarray(data["z"], dtype=np.float32)``, which silently upcast a bf16
    cache into something that looked like the fp32 the ruling asks for.
    """

    def __init__(self, *, mode: str, root: Path | None, split: str,
                 checkpoint: str, readout: WhatReadoutSpec,
                 k_rows: int = 1, expand: str = "proj",
                 tags: Sequence[str] = CONTROLS,
                 required: Sequence[str] = ("none",), data: str = "v2seg",
                 zcache_root_l8: Path | str | None = None,
                 seed: int = SEED):
        self.mode = mode
        self.root = Path(root) if root else None
        self.split = split
        self.checkpoint = checkpoint
        self.readout = readout
        self.k_rows = int(k_rows)
        self.expand = expand
        self.data = str(data)
        self._samples: dict[str, Sample] = {}
        if mode == "zcache" and self.root is None:
            raise ValueError("--z-source zcache needs --zcache-dir")
        # ``--data v2seg+l8``: the ``none`` slot is the union of two caches.
        # ZCacheDir opens nothing then, so the sft2seg member is read exactly
        # once -- by caliber.open_train_z (= run_carrier_arm.open_z_caches).
        union = (mode == "zcache" and self.data != "v2seg"
                 and "none" in tuple(tags))
        self.dir = ZCacheDir(
            self.root, split=split, checkpoint=checkpoint,
            readout_kind=readout.kind,
            tags=(() if union else tuple(tags)),
            required=(() if union else tuple(required)),
            synthetic=(mode == "synthetic"),
            k_rows=(self.k_rows if expand == "qtok" else 0),
            jitter=0.1, key_fn=self._synthetic_key)
        self.union_record: dict[str, Any] | None = None
        if union:
            cache, rec = K.open_train_z(
                self.root, split, data=self.data, checkpoint=checkpoint,
                readout_kind=readout.kind, zcache_root_l8=zcache_root_l8,
                seed=int(seed))
            self.dir.caches["none"] = cache
            self.dir.record["none"] = rec
            self.union_record = rec

    # -- synthetic keys ------------------------------------------------------
    def _synthetic_key(self, control: str, sample_id: str) -> str:
        """``none`` keys on the sample's own LUT (so a generator *can* fit it),
        ``shuffle`` on another instruction's, ``irrelevant`` / ``const`` on a
        single fixed key -- the same structure the three real controls have."""
        s = self._samples.get(sample_id)
        lut = s.lut_id if s else sample_id
        src = s.source_image_id if s else sample_id
        return {"none": f"lut:{lut}",
                "shuffle": f"shuffle:{src}",
                "irrelevant": "irrelevant:caption",
                "const": "const:Please edit this photo."}[control]

    # -- loading -------------------------------------------------------------
    def load(self, control: str) -> None:
        """No-op: every cache was opened and asserted in ``__init__``."""
        self.dir.cache(control)

    def preload(self, controls: Sequence[str] = CONTROLS) -> dict[str, Any]:
        """HANDOFF section 4.H: the caches' ``checkpoint`` field is asserted
        before the dataloader is built.  That now happens in ``__init__``; this
        stays as the explicit witness the runner writes into ``run_setup``."""
        for control in controls:
            self.load(control)
        return self.facts()

    # -- access --------------------------------------------------------------
    def get(self, sample: Sample, control: str = "none") -> torch.Tensor:
        self._samples[sample.sample_id] = sample
        z = self.dir.cache(control).vector(sample.sample_id)
        if self.expand == "qtok" and z.dim() != 2:
            raise AssertionError(
                f"--z-expand qtok needs (K, 2560) per sample, got {tuple(z.shape)}")
        if self.expand == "proj" and z.dim() != 1:
            raise AssertionError(
                f"--z-expand proj needs (2560,) per sample, got {tuple(z.shape)}")
        return z

    def batch(self, samples: Sequence[Sample], control: str = "none",
              *, device: Any = "cpu") -> torch.Tensor:
        return torch.stack([self.get(s, control) for s in samples]).to(device=device)

    @property
    def meta(self) -> dict[str, dict[str, Any]]:
        return self.dir.meta

    def facts(self) -> dict[str, Any]:
        return {"mode": self.mode, "split": self.split,
                "root": str(self.root) if self.root else None,
                "checkpoint": self.checkpoint, "data": self.data,
                "union": self.union_record,
                "expand": self.expand, "k_rows": self.k_rows,
                "controls": dict(self.dir.record),
                "synthetic": self.mode == "synthetic"}


SEG_HIDDEN = qd.SEG_COLOR_HIDDEN_DIM


def _hash_seed(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


# --------------------------------------------------------------------------- #
# alpha (GT field) and images
# --------------------------------------------------------------------------- #
class AlphaStore:
    """GT alpha at short side 512, from ``where_a-20260805/maskviews``.

    ``style`` samples have no mask member and ``alpha == 1`` everywhere
    (HANDOFF section 七).  The field the generator consumes is this alpha
    ``area_resize``-d to the arm's declared resolution -- the frozen block's
    operator (``q3vl/where/upsample.py:54-62``), and the resolution is written
    into ``run_setup`` and printed under the section-E table.
    """

    def __init__(self, split: str, *, mode: str = "real",
                 root: Path = MASKVIEWS_ROOT):
        self.split = split
        self.mode = mode
        self.root = Path(root) / split
        self.index: dict[str, dict[str, Any]] = {}
        self.n_missing = 0
        if mode == "real":
            idx_dir = self.root / "indexes"
            files = sorted(idx_dir.glob("*.idx.jsonl")) if idx_dir.is_dir() else []
            if not files:
                raise FileNotFoundError(f"no mask index under {idx_dir}")
            for path in files:
                with path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        if rec.get("suffix") == ".maskhi.png":
                            self.index[str(rec["sample_id"])] = rec

    def alpha(self, sample: Sample, hw: tuple[int, int] | None = None
              ) -> torch.Tensor:
        """``(H, W)`` alpha in [0,1]; ``hw = None`` keeps the mask's own size.

        Training only ever feeds the field path, which resamples immediately, so
        it asks for the native size and pays one ``area_resize`` instead of two.
        """
        if sample.is_style:
            # style samples carry no mask member: alpha == 1 everywhere, and a
            # 1x1 constant upsamples to any grid as the same constant.
            h, w = hw if hw is not None else (1, 1)
            return torch.ones(h, w, dtype=torch.float32)
        if self.mode == "synthetic":
            g = torch.Generator().manual_seed(_hash_seed("alpha:" + sample.sample_id))
            small = torch.rand(4, 6, generator=g)
            if hw is None:
                return small.clamp(0, 1)
            return area_resize(small[None, None], hw)[0, 0].clamp(0, 1)
        rec = self.index.get(sample.sample_id)
        if rec is None:
            self.n_missing += 1
            raise KeyError(
                f"{sample.sample_id} is a local sample with no .maskhi.png in "
                f"{self.root}; alpha is not optional for the headline")
        blob = _read_member(self.root / "shards" / f"{rec['shard']}.tar",
                            int(rec["offset"]), int(rec["length"]))
        from PIL import Image

        a = torch.from_numpy(
            np.asarray(Image.open(io.BytesIO(blob)).convert("L"),
                       dtype=np.float32) / 255.0)
        if hw is not None and tuple(a.shape) != tuple(hw):
            a = area_resize(a[None, None], hw)[0, 0]
        return a.clamp(0.0, 1.0)

    def field(self, alpha: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        """``(fh, fw)`` -- the alpha resampled onto the carrier's own grid."""
        return area_resize(alpha[None, None].float(), hw)[0, 0]

    def field_for(self, sample: Sample, hw: tuple[int, int]) -> torch.Tensor:
        """``(fh, fw)`` straight from the sample -- one resample, frozen operator."""
        return self.field(self.alpha(sample, None), hw)


class ImageStore:
    """Input images ``(3, H, W)`` in [0,1], read from the split's tar shards."""

    def __init__(self, *, mode: str = "real", synth_hw: tuple[int, int] = (64, 80)):
        self.mode = mode
        self.synth_hw = synth_hw

    def image(self, sample: Sample) -> torch.Tensor:
        if self.mode == "synthetic":
            g = torch.Generator().manual_seed(_hash_seed("img:" + sample.sample_id))
            h, w = self.synth_hw
            small = torch.rand(3, 8, 10, generator=g)
            return area_resize(small[None], (h, w))[0].clamp(0, 1)
        member = sample.raw["members"]["image"]
        blob = _read_member(ro_path(member["shard"]), int(member["offset"]),
                            int(member["length"]))
        from PIL import Image

        arr = np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"),
                         dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _read_member(path: Path, offset: int, length: int) -> bytes:
    with Path(path).open("rb") as fh:
        fh.seek(offset)
        blob = fh.read(length)
    if len(blob) != length:
        raise IOError(f"short read of {path}:{offset}")
    return blob


# --------------------------------------------------------------------------- #
# library (B1 / B2 / B4 / B6) and buckets (B3)
# --------------------------------------------------------------------------- #
def library_ids(train_rows: Sequence[IndexRow], *, n_rows: int, seed: int
                ) -> list[str]:
    """``Lib_tr``: the unique lut ids of ``n_rows`` random train index rows.

    Section 4.C's measured protocol: 2500 rows -> 1137 ids (not all 3149).  The
    draw uses its own :class:`random.Random`, so it cannot move any other stream.
    """
    rng = random.Random(seed)
    pool = list(train_rows)
    picked = rng.sample(pool, min(int(n_rows), len(pool)))
    return sorted({r.lut_id for r in picked if r.lut_id})


def mean_transform_volume(bank: LutBank, lut_ids: Sequence[str], *,
                          grid_n: int = B1_GRID, device: Any = "cpu"
                          ) -> torch.Tensor:
    """B1 as a LUT volume: ``Lbar(x) = mean_l L_l(x)`` sampled on ``grid_n^3``.

    The point-wise mean of the library is still a valid mapping (section 4.C);
    materialising it on a grid is what makes it applicable to an image at the
    same cost as any other LUT.  ``grid_n = 33`` is recorded on the board.
    """
    x = uniform_grid(grid_n, device=device)                     # (n^3, 3), (r,g,b)
    acc = torch.zeros_like(x)
    for lid in lut_ids:
        acc += bank.apply(x, lid)
    vals = acc / max(1, len(lut_ids))
    t = vals.reshape(grid_n, grid_n, grid_n, 3).permute(2, 1, 0, 3)  # (b, g, r, 3)
    return t.permute(3, 0, 1, 2)[None].contiguous()                  # (1,3,Db,Dg,Dr)


# --------------------------------------------------------------------------- #
# evaluation helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def headline_for_transform(img: torch.Tensor, alpha: torch.Tensor,
                           f_img: torch.Tensor, i_star: torch.Tensor,
                           lab_star: torch.Tensor | None = None) -> float:
    """``mean dE00(Î, I*)`` with the frozen image formation.

    ``lab_star`` is the CIELab of ``I*``; it is the same tensor for the arm and
    for every baseline of a sample, so it is converted once and passed in.  The
    quantity is identical either way -- ``criteria.image_delta_e00`` would redo
    exactly this conversion.
    """
    i_hat = crit.compose_hat(img, alpha.unsqueeze(0), f_img)
    if lab_star is None:
        return float(crit.image_delta_e00(i_hat, i_star))
    lab_hat = srgb_to_lab(i_hat.permute(1, 2, 0))
    return float(delta_e00(lab_hat, lab_star).mean())


@torch.no_grad()
def transform_values(model: qd.QDualArm, grid: torch.Tensor, z: torch.Tensor,
                     field: torch.Tensor | None) -> torch.Tensor:
    """``(P, 3)`` -- the arm's transform on a shared query grid for ONE sample."""
    return model(grid, z.unsqueeze(0), None if field is None else field.unsqueeze(0))[0]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
class _Parser(argparse.ArgumentParser):
    """``--batch-split`` resolves B and Q, unless both were given explicitly.

    The resolution happens here rather than in ``main`` so that every consumer
    of the namespace (``frozen_block_record``, the tests) sees the same pair.
    """

    def parse_args(self, args=None, namespace=None):        # type: ignore[override]
        ns = super().parse_args(args, namespace)
        b, q = K.parse_batch_split(ns.batch_split)
        ns.batch_samples = int(ns.batch_samples) if ns.batch_samples else b
        ns.queries = int(ns.queries) if ns.queries else q
        return ns


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(
        prog="run_qdual_arm",
        description="EPR-029 QDUAL: Gaussian queries + cross-attention decoding")
    # --- the arm's own flags (EPR-029:640) ---
    p.add_argument("--rung", choices=list(qd.LADDER_ROWS), default="c")
    p.add_argument("--n-gauss", type=int, default=48)
    p.add_argument("--decoder-layers", type=int, choices=(2, 4), default=4)
    p.add_argument("--decoder-width", type=int, choices=(128, 256), default=256)
    p.add_argument("--decoder-heads", type=int, default=8)
    p.add_argument("--z-expand", choices=("proj", "qtok"), default="proj")
    p.add_argument("--z-expand-k", type=int, choices=(1, 4, 8), default=4)
    p.add_argument("--field-source", choices=("m_low", "m_pix"), default="m_low")
    p.add_argument("--field-kind", choices=("gt", "pred", "const", "shuffle"),
                   default="gt")
    p.add_argument("--field-grid", default="32x48",
                   help="the (gh, gw) the field is resampled to (short side 512 "
                        "-> 32x48 from fpre.grid_from_geometry); m_pix uses 4x")
    p.add_argument("--attn-temperature", action="store_true",
                   help="SA-LUT model.py:128 learnable attention temperature "
                        "(ablation row; off in the main arm)")
    p.add_argument("--zero-init-head", dest="zero_init_head",
                   action="store_true", default=True)
    p.add_argument("--no-zero-init-head", dest="zero_init_head",
                   action="store_false",
                   help="ablation row ⑦: also drops the step-0 identity assertion "
                        "and the zeroinit_step0_maxabs required key")
    p.add_argument("--no-color-pe", dest="color_pe", action="store_false",
                   default=True, help="ablation row ⑪")
    p.add_argument("--no-e-type", dest="e_type", action="store_false",
                   default=True, help="ablation row ⑫")
    p.add_argument("--color-sampling", choices=("uniform", "alpha_hist"),
                   default="uniform")
    p.add_argument("--readout", choices=("seg_color", "color_close", "im_end",
                                         "qtok"), default="seg_color")
    p.add_argument("--readout-qtok", type=int, default=0)
    # --- frozen-block flags ---
    p.add_argument("--clamp", choices=("two", "one"), default="two")
    p.add_argument("--batch-samples", type=int, default=None,
                   help="override B; the default comes from --batch-split")
    p.add_argument("--queries", type=int, default=None,
                   help="override Q; the default comes from --batch-split")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--total-steps", type=int, default=0,
                   help="0 = epochs * ceil(n_train / B) (frozen: 117440)")
    # one spelling across the six arms; --lr stays as this arm's old name
    p.add_argument("--base-lr", "--lr", dest="lr", type=float, default=BASE_LR)
    p.add_argument("--geometry-lr-scale", type=float, default=GEOMETRY_LR_SCALE)
    p.add_argument("--grad-clip", type=float, default=0.0,
                   help="0 = off (GLUT/CGLUT give no value); the measured grad "
                        "norm is logged every step either way")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--loss-level", type=int, default=3, choices=(1, 3),
                   help="3 = L_rec + 10 L_hc + 0.001 R_sparse (this arm's own "
                        "recipe); 1 = the single L1 term (lambda_hc = "
                        "lambda_sparse = lambda_mono = 0, asserted before step 0)")
    # --- data ---
    # RENAMED 2026-08-16: this flag chose the z SOURCE, and the shared EPR-030
    # caliber needs --data for the training corpora (--data v2seg / v2seg+l8).
    p.add_argument("--z-source", dest="z_source",
                   choices=("zcache", "synthetic"), default="zcache",
                   help="where z comes from (was --data before 2026-08-16)")
    p.add_argument("--zcache-dir", default=None)
    p.add_argument("--pred-field-dir", default=None,
                   help="<sample_id>.npy predicted fields from a where arm; "
                        "required for the field_pred column")
    p.add_argument("--base-checkpoint", default=BASE_CHECKPOINT)
    p.add_argument("--tokenizer", default=None,
                   help="tokenizer for the colour-span start-up assertion "
                        "(default: --base-checkpoint)")
    p.add_argument("--dataset-root", default=None,
                   help="index root; default = the --dataset-version口径's root")
    p.add_argument("--bank-dir", default=str(BANK_DIR))
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="V_what")
    p.add_argument("--lib-sample", type=int, default=2500)
    p.add_argument("--baseline-repeats", type=int, default=8)
    p.add_argument("--bucket-pool-cache", default=None)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--eval-max-samples", type=int, default=0)
    p.add_argument("--quick-eval-n", type=int, default=32)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--quick-eval-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=50)
    # --- plumbing ---
    p.add_argument("--out-root", default="/home/bc/data/runs/whatb")
    p.add_argument("--run-name", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--smoke", action="store_true",
                   help="tiny budget for a CPU wiring check; the board is written "
                        "with published=False")
    p.add_argument("--smoke-steps", type=int, default=24)
    p.add_argument("--skip-colorspan-assert", action="store_true",
                   help="only legal with --z-source synthetic (no records to draw "
                        "colour texts from); recorded in run_setup")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve the caliber, write run_setup.json, then stop")
    # --data / --zcache-root-l8 / --batch-split; --base-lr is declared above
    K.add_caliber_arguments(p, base_lr=False)
    return p


# --------------------------------------------------------------------------- #
# setup
# --------------------------------------------------------------------------- #
def config_from_args(args: argparse.Namespace) -> qd.QDualConfig:
    gh, gw = (int(v) for v in str(args.field_grid).lower().split("x"))
    return qd.QDualConfig(
        rung=args.rung, n_gauss=args.n_gauss, decoder_layers=args.decoder_layers,
        decoder_width=args.decoder_width, decoder_heads=args.decoder_heads,
        z_expand=args.z_expand, z_expand_k=args.z_expand_k,
        field_source=args.field_source, field_kind=args.field_kind,
        attn_temperature=bool(args.attn_temperature),
        zero_init_head=bool(args.zero_init_head),
        color_sampling=args.color_sampling, readout=args.readout,
        readout_qtok=int(args.readout_qtok), clamp=args.clamp,
        color_pe=bool(args.color_pe), e_type=bool(args.e_type),
        field_grid_h=gh, field_grid_w=gw)


def resolve_precision(args: argparse.Namespace) -> dict[str, Any]:
    """bf16 is the repository default; on CPU it degrades to fp32 **loudly**."""
    device = torch.device(args.device)
    want = args.precision
    used = want
    note = None
    if want == "bf16" and device.type != "cuda":
        used = "fp32"
        note = (f"--precision bf16 requested on device {device}; autocast bf16 is "
                "a CUDA path here, so this run computed in fp32.  Recorded rather "
                "than silently applied.")
    return {"requested": want, "used": used, "device": str(device), "note": note}


def frozen_block_record(args: argparse.Namespace, n_train: int,
                        steps_per_epoch: int, total_steps: int) -> dict[str, Any]:
    """The eight frozen items with the value this run actually used."""
    return {
        "train_normal_n": {"frozen": TRAIN_NORMAL_N, "used": int(n_train),
                           "matches": int(n_train) == TRAIN_NORMAL_N},
        "batch_split": {"frozen": "B=32 x Q=256 = 8192 colours/step",
                        "used": f"B={args.batch_samples} x Q={args.queries} = "
                                f"{args.batch_samples * args.queries}",
                        "matches": args.batch_samples * args.queries == COLORS_PER_STEP},
        "steps_per_epoch": {"frozen": STEPS_PER_EPOCH, "used": int(steps_per_epoch),
                            "matches": int(steps_per_epoch) == STEPS_PER_EPOCH},
        "total_steps": {"frozen": TOTAL_STEPS, "used": int(total_steps),
                        "matches": int(total_steps) == TOTAL_STEPS},
        "clamp": {"frozen": "two", "used": args.clamp,
                  "matches": args.clamp == "two"},
        "headline_formation": "Î = (1 - a) ⊙ I + a ⊙ f̂(I)",
        "preregistered_keys": list(crit.PREREGISTERED_KEYS),
        "colorspan": "q3vl.whatb.colorspan (own implementation + tokenizer assertion)",
        "package": "q3vl/whatb/",
    }


def colorspan_assertion(args: argparse.Namespace, rows: Sequence[IndexRow]
                        ) -> dict[str, Any]:
    """Frozen block item 7, run **before** any data loader is built."""
    if args.skip_colorspan_assert:
        if args.z_source != "synthetic":
            raise SystemExit(
                "--skip-colorspan-assert is only legal with --z-source synthetic; on "
                "real data the token ids must be pinned before training starts")
        return {"skipped": True, "reason": "synthetic data, no records"}
    from transformers import AutoTokenizer

    tok_path = args.tokenizer or args.base_checkpoint
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    rng = random.Random(SEED)
    picked = rng.sample(list(rows), min(256, len(rows)))
    texts = [rec.get("color") for rec in iter_records(picked)]
    return assert_color_span_encoding(tok, [t for t in texts if t],
                                      tokenizer_path=str(tok_path))


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def lut_targets(bank: LutBank, samples: Sequence[Sample], colors: torch.Tensor
                ) -> torch.Tensor:
    """``(B, Q, 3)`` ``L_l(x)`` -- each sample's own LUT on its own colours."""
    return torch.stack([bank.apply(colors[i], s.lut_id)
                        for i, s in enumerate(samples)])


def train(model: qd.QDualArm, *, args: argparse.Namespace, cfg: qd.QDualConfig,
          samples: Sequence[Sample], zstore: ZStore, alphas: AlphaStore,
          bank: LutBank, run_dir: Path, total_steps: int, steps_per_epoch: int,
          device: torch.device, quick_eval_fn, images: ImageStore | None = None,
          pred_field_dir: Path | None = None,
          lambda_hc: float = qd.LAMBDA_HC,
          lambda_sparse: float = qd.LAMBDA_SPARSE,
          select_fn=None) -> dict[str, Any]:
    """Adam + cosine, GLUT's mining, ``steps.jsonl`` every step."""
    groups = model.param_groups(args.lr, geometry_lr_scale=args.geometry_lr_scale)
    opt = torch.optim.Adam(groups, lr=args.lr)          # no weight decay: GLUT
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps))
    sampler = QuerySampler(seed=args.seed, q=int(args.queries))
    order_rng = random.Random(args.seed)
    steps_path = run_dir / "steps.jsonl"
    if steps_path.is_file() and steps_path.stat().st_size:
        raise SystemExit(
            f"{steps_path} already has content.  The first line of this file is "
            "one of the three tiers the publication gate reads; appending a "
            "second run to it would make the gate read the previous run's "
            "losses.  Back it up and use a fresh --run-name.")
    fh = steps_path.open("w", encoding="utf-8")
    amp = _amp_context(args, device)
    idx_pool = list(range(len(samples)))
    cursor = len(idx_pool)
    first_row: dict[str, Any] | None = None
    t0 = time.time()
    try:
        for step in range(1, int(total_steps) + 1):
            if cursor + args.batch_samples > len(idx_pool):
                order_rng.shuffle(idx_pool)
                cursor = 0
            batch = [samples[i] for i in idx_pool[cursor: cursor + args.batch_samples]]
            cursor += args.batch_samples

            z = zstore.batch(batch, "none", device=device)
            field = _field_batch(batch, alphas, cfg, device,
                                 pred_field_dir=pred_field_dir,
                                 synthetic=zstore.mode == "synthetic")
            if step == 1:
                # BEFORE the first update: the zero-init witness is a property of
                # step 0, so it is measured on untouched parameters.
                witness = qd.zero_init_witness(
                    model, z, field, uniform_grid(9, device=device))
                qd.assert_zero_init(witness, enabled=cfg.zero_init_head)
                # ``zeroinit_step0_maxabs`` is a *training* quantity, so
                # ``build_board`` can never see it: hand it to the selector here,
                # strictly before the first ``select_fn`` call below, or the
                # first selection board fails ``assert_criteria_ran`` on a key
                # every other tier already carries (QDUAL_LR3E4, 2026-08-20).
                if select_fn is not None and hasattr(select_fn, "state"):
                    select_fn.state["zero_init_witness"] = dict(witness)

            colors = _color_batch(batch, sampler, cfg, device, images, alphas)
            ratio = qd.mining_ratio_for_step(step, steps_per_epoch)
            mine_stats = {"mining_ratio": float(ratio), "n_mined": 0,
                          "n_fresh": int(colors.shape[1])}
            if ratio > 0:
                probe_y = lut_targets(bank, batch, colors)
                fresh = _color_batch(batch, sampler, cfg, device, images, alphas)
                with amp():
                    colors, mine_stats = qd.mine_hard_colors(
                        model, z, field, colors, fresh, probe_y, ratio)
            y = lut_targets(bank, batch, colors)

            with amp():
                y_hat, aux = model(colors, z, field, return_aux=True)
            loss = qd.qdual_losses(y_hat.float(), y, aux,
                                   lambda_hc=lambda_hc,
                                   lambda_sparse=lambda_sparse)
            opt.zero_grad(set_to_none=True)
            loss.total.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip if args.grad_clip > 0 else float("inf"))
            opt.step()
            sched.step()

            row = {
                "step": step, "epoch": step / max(1, steps_per_epoch),
                **loss.columns,
                "n_luts_in_batch": len({s.lut_id for s in batch}),
                "n_samples": len(batch),
                "mining_ratio": float(mine_stats["mining_ratio"]),
                "n_mined": int(mine_stats["n_mined"]),
                "lr": float(opt.param_groups[0]["lr"]),
                "lr_slow": float(opt.param_groups[-1]["lr"]),
                "grad_norm": float(gnorm),
                "ladder_row": cfg.rung,
                "degenerate_precision": int(aux.degenerate_precision.sum()),
                "wall_ms_per_step": (time.time() - t0) * 1000.0 / step,
            }
            if step == 1:
                row.update(witness)
                first_row = dict(row)
                record_step_witness(row)     # tier 3 of the steps-row contract
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if step % max(1, args.log_every) == 0 or step == 1:
                fh.flush()
                print(f"[{_now()}] step {step}/{total_steps} "
                      f"L_total={row['L_total']:.5f} L_rec={row['L_rec']:.5f} "
                      f"L_hc={row['L_hc']:.5f} r={row['mining_ratio']:.2f}",
                      flush=True)
            # The degenerate-solution guard runs at EVERY quick eval, the first
            # one included -- that first call is the one the campaign requires
            # (PRND/CONDINST spent 2.6 GPU-hours on a constant field), and the
            # later ones cost one forward each.
            if step % max(1, args.quick_eval_every) == 0 or step == int(total_steps):
                quick_eval_fn(model, step)
            # checkpoint selection reads the headline, never the loss (redline)
            if select_fn is not None and args.eval_every and (
                    step % int(args.eval_every) == 0):
                select_fn(model, step)
    finally:
        fh.flush()
        fh.close()
    return {"first_step_row": first_row, "steps_path": str(steps_path),
            "wall_s": time.time() - t0}


def _field_batch(batch: Sequence[Sample], alphas: AlphaStore,
                 cfg: qd.QDualConfig, device: torch.device, *,
                 pred_field_dir: Path | None = None,
                 synthetic: bool = False) -> torch.Tensor | None:
    """``(B, fh, fw)`` field of the kind ``--field-kind`` asks for, or ``None``.

    Every sample is resampled onto the arm's **declared** grid with the frozen
    ``area_resize`` operator, so a 512x768 and a 512x640 image give the same
    token count; the declared resolution is what ``run_setup`` records and what
    the section-E table prints.

    ``--field-kind`` is honoured **here**, at the one place the field is built,
    so the flag cannot be a silent no-op: ``gt`` (default, NOTES 12), ``const``
    (this sample's mean alpha), ``shuffle`` (the next sample's field) and
    ``pred`` (a where arm's field from ``--pred-field-dir``).
    """
    if not cfg.uses_field:
        return None
    gt = [alphas.field_for(s, cfg.field_hw) for s in batch]
    if cfg.field_kind == "gt":
        out = gt
    elif cfg.field_kind == "const":
        out = [torch.full_like(f, float(f.mean())) for f in gt]
    elif cfg.field_kind == "shuffle":
        out = [gt[(i + 1) % len(gt)] for i in range(len(gt))]
    else:                                        # pred
        out = []
        for s, f in zip(batch, gt):
            p = _pred_field(pred_field_dir, s, cfg, f, synthetic=synthetic)
            if p is None:
                raise SystemExit(
                    f"--field-kind pred but {s.sample_id} has no predicted field "
                    f"in {pred_field_dir}; refusing to fall back to GT alpha "
                    "silently (that would publish a GT-alpha run under the "
                    "pred-field name)")
            out.append(p)
    return torch.stack(out).to(device=device)


def _color_batch(batch: Sequence[Sample], sampler: QuerySampler,
                 cfg: qd.QDualConfig, device: torch.device,
                 images: ImageStore | None, alphas: AlphaStore
                 ) -> torch.Tensor:
    """``(B, Q, 3)`` query colours -- ``--color-sampling`` honoured here.

    ``uniform`` is GLUT App A.1's own 128^3 draw (the main arm).  ``alpha_hist``
    is ablation row ⑬: colours drawn from the image's alpha-weighted 5-bit
    histogram, which needs the image, so it is only available when the runner
    has an :class:`ImageStore`.
    """
    if cfg.color_sampling == "uniform":
        return sampler.sample(len(batch), device=device)
    if images is None:
        raise SystemExit("--color-sampling alpha_hist needs the image store")
    from q3vl.whatb.queries import image_histogram_colors

    rows = []
    for s in batch:
        img = images.image(s)
        alpha = alphas.alpha(s, (int(img.shape[1]), int(img.shape[2])))
        pool, w = image_histogram_colors(img, bits=5, top_k=4096, alpha=alpha)
        rows.append(sampler.sample_from_pool(pool, weights=w))
    return torch.stack(rows).to(device=device)


def _amp_context(args: argparse.Namespace, device: torch.device):
    """bf16 autocast on CUDA, a no-op elsewhere (the carrier disables it anyway)."""
    import contextlib

    if args.precision == "bf16" and device.type == "cuda":
        return lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext


# --------------------------------------------------------------------------- #
# quick eval: the degenerate-solution guard
# --------------------------------------------------------------------------- #
def make_quick_eval(*, cfg: qd.QDualConfig, samples: Sequence[Sample],
                    zstore: ZStore, alphas: AlphaStore, device: torch.device,
                    thresholds: DegeneracyThresholds, run_dir: Path,
                    n_samples: int = 32, exit_process: bool = True,
                    pred_field_dir: Path | None = None):
    """Returns the callable the trainer fires at the first quick eval.

    Runs the three-condition degeneracy assertion on ``n_samples`` **different**
    conditions over the 9^3 query grid, then records the report.  Any of the
    three conditions failing leaves through ``SystemExit(2)``.
    """
    picked = [s for s in samples if s.winner_confidence == "normal"][:n_samples]
    if not picked:
        picked = list(samples[:n_samples])

    def _run(model: qd.QDualArm, step: int) -> dict[str, Any]:
        grid = uniform_grid(9, device=device)
        z = zstore.batch(picked, "none", device=device)
        field = _field_batch(picked, alphas, cfg, device,
                             pred_field_dir=pred_field_dir,
                             synthetic=zstore.mode == "synthetic")
        with torch.no_grad():
            y = model(grid, z, field)
        report = qd.assert_not_degenerate(
            y, grid, thresholds=thresholds, where=f"quick_eval@step{step}",
            exit_process=exit_process,
            extra={"arm": qd.ARM, "rung": cfg.rung, "n_conditions": len(picked)})
        out = {"step": int(step), **report.as_dict()}
        with (run_dir / "quick_eval.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(out) + "\n")
        print(f"[{_now()}] quick-eval degeneracy guard passed at step {step}: "
              f"point_std={report.point_std:.3e} iddev={report.identity_dev:.3e} "
              f"cross_std={report.cross_std:.3e}", flush=True)
        return out

    return _run


@torch.no_grad()
def step0_headline_witness(model: qd.QDualArm, *, cfg: qd.QDualConfig,
                           samples: Sequence[Sample], zstore: ZStore,
                           alphas: AlphaStore, images: ImageStore, bank: LutBank,
                           device: torch.device, n: int = 4) -> dict[str, Any]:
    """``step0_headline`` vs the B0 identity column, per sample (proposal :639).

    The proposal asks for equality "with tolerance 0".  That is unattainable and
    the reason is proposition 2, not an implementation slip: at ``eps = 1e-6``
    the base parameters give ``f(x) = (1 - delta(x)) x`` with ``delta`` up to
    ~1.1e-7, so the step-0 image is not bit-identical to the input.  The check is
    therefore *measured* -- the maximum per-sample ``|E_step0 - E_B0|`` is
    recorded here and lands in ``run_setup`` and on the board -- rather than
    asserted against a floor no correct implementation can meet.
    """
    picked = [s for s in samples if s.winner_confidence == "normal"][:int(n)]
    diffs: list[float] = []
    for s in picked:
        img = images.image(s).to(device=device)
        alpha = alphas.alpha(s, (int(img.shape[1]), int(img.shape[2]))).to(device)
        field = alphas.field(alpha, cfg.field_hw).to(device) if cfg.uses_field else None
        i_star = bank.f_star_image(img, alpha.unsqueeze(0), s.lut_id)
        lab_star = srgb_to_lab(i_star.permute(1, 2, 0))
        z = zstore.get(s, "none").to(device=device)
        f_img = model.apply_image(img, z.unsqueeze(0),
                                  None if field is None else field.unsqueeze(0))
        e0 = headline_for_transform(img, alpha, f_img, i_star, lab_star)
        e_b0 = float(delta_e00(srgb_to_lab(img.permute(1, 2, 0)), lab_star).mean())
        diffs.append(abs(e0 - e_b0))
    return {"n": len(diffs),
            "step0_headline_vs_B0_maxabs": max(diffs) if diffs else None,
            "note": ("proposition 2 with eps = 1e-6 makes step 0 (1 - delta) x, "
                     "delta <= ~1.1e-7, so this is small but not exactly 0")}


def attach_zeroinit_column(board: dict[str, Any],
                           witness: Mapping[str, Any] | None, *,
                           cfg: qd.QDualConfig, source: str) -> dict[str, Any]:
    """Put the step-0 zero-init witness on ``board`` as its own criteria column.

    ``zeroinit_step0_maxabs`` is pre-registered in :data:`qdual.REQUIRED_QDUAL`
    but it is measured in the training loop, not in :func:`evaluate`, so
    ``criteria.build_board`` cannot produce it: **every** board that goes through
    ``assert_criteria_ran`` has to be handed it through this one function.  The
    selection board used to skip that step, which is how QDUAL_LR3E4 reached its
    first selection point after ~2h with the other 24 columns green and died on
    this one.  ``witness`` is either the step-1 witness dict or the first
    ``steps.jsonl`` row (both carry the key); ``None`` leaves ``n = 0``, which is
    exactly what the assertion is there to catch.
    """
    step0 = (witness or {}).get("zeroinit_step0_maxabs")
    col: dict[str, Any] = {
        "n": 1 if step0 is not None else 0, "value": step0, "source": source,
        "quantity": "max |dtheta| at step 0; must be exactly 0 with zero-init heads"}
    if not cfg.zero_init_head:
        col["note"] = (
            "--no-zero-init-head: this key is dropped from the required table "
            "(EPR-029:772-773) and the step-0 identity assertion is off")
    board.setdefault("criteria_columns", {})["zeroinit_step0_maxabs"] = col
    return col


def make_selector(*, args: argparse.Namespace, cfg: qd.QDualConfig,
                  samples: Sequence[Sample], zstore: ZStore, alphas: AlphaStore,
                  images: ImageStore, bank: LutBank, lib_ids: Sequence[str],
                  pools: Mapping[str, Sequence[str]], device: torch.device,
                  run_dir: Path, pred_field_dir: Path | None):
    """Checkpoint selection on ``.contexts.all.headline_normal_only`` -- never loss.

    Campaign redline: the selection metric is the headline, the quick-eval guard
    is a hard gate in front of it, and ``val loss`` may not be read.  Fires every
    ``--eval-every`` steps (0 = off, one final evaluation only) on the first
    ``--eval-max-samples`` rows and writes ``selection.jsonl`` + ``best.pt``.
    """
    state: dict[str, Any] = {"best": None, "best_step": None,
                             "first_board": True, "zero_init_witness": None}

    def _run(model: qd.QDualArm, step: int) -> dict[str, Any]:
        rows, aux = evaluate(model, args=args, cfg=cfg, samples=samples,
                             zstore=zstore, alphas=alphas, images=images,
                             bank=bank, lib_ids=lib_ids, pools=pools,
                             device=device, pred_field_dir=pred_field_dir)
        board = crit.build_board(rows, arm=qd.ARM, split=args.eval_split,
                                 extra_columns=aux["extra_columns"],
                                 seed=args.seed)
        # the one pre-registered column build_board cannot produce (training-side)
        zcol = attach_zeroinit_column(board, state.get("zero_init_witness"),
                                      cfg=cfg, source="train_step1_witness")
        hn = board["contexts"]["all"]["headline_normal_only"]
        rec = {"step": int(step), "headline_normal_only": hn.get("mean"),
               "n": hn.get("n"), "metric": ".contexts.all.headline_normal_only",
               "selection_rule": "min headline; val loss is never read",
               # on the artefact so the wiring is visible even on the runs whose
               # first-board assertion is skipped (synthetic / --smoke)
               "zeroinit_step0_maxabs": zcol["value"]}
        if state["first_board"]:
            # "定义了没接线" has cost this campaign five times: assert the WHOLE
            # pre-registered table at the FIRST selection point, so a column
            # that was defined but never wired fails in epoch 1 instead of after
            # the 117,440-step horizon (the shape run_g4d_arm.py uses).  A
            # truncated / synthetic run records the skip and its reason on the
            # artefact rather than passing silently.
            skip = ("--z-source synthetic" if args.z_source == "synthetic" else
                    "--smoke" if args.smoke else None)
            rec["first_board_assertion"] = (
                {"skipped": skip} if skip else
                crit.assert_criteria_ran(board, qd.ARM,
                                         required=qd.required_criteria_table(cfg)))
            state["first_board"] = False
        if hn.get("mean") is not None and (
                state["best"] is None or hn["mean"] < state["best"]):
            state["best"], state["best_step"] = float(hn["mean"]), int(step)
            torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                        "step": int(step), "headline": state["best"]},
                       run_dir / "best.pt")
            rec["new_best"] = True
        with (run_dir / "selection.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        model.train()
        return rec

    _run.state = state          # type: ignore[attr-defined]
    return _run


# --------------------------------------------------------------------------- #
# full evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model: qd.QDualArm, *, args: argparse.Namespace, cfg: qd.QDualConfig,
             samples: Sequence[Sample], zstore: ZStore, alphas: AlphaStore,
             images: ImageStore, bank: LutBank, lib_ids: Sequence[str],
             pools: Mapping[str, Sequence[str]], device: torch.device,
             pred_field_dir: Path | None) -> tuple[list[dict[str, Any]],
                                                   dict[str, Any]]:
    """Per-sample rows for the board, plus the arm's own diagnostic columns."""
    model.eval()
    grid17 = uniform_grid(17, device=device)
    grid9 = uniform_grid(9, device=device)
    lib = crit.LibraryValues.build(bank, list(lib_ids), grid9)
    b1_vol = mean_transform_volume(bank, list(lib_ids), device=device)
    b2_draws = crit.library_random_draw(list(lib_ids), len(samples),
                                        repeats=int(args.baseline_repeats),
                                        seed=args.seed)
    b3_draws = crit.bucket_draw([str(s.minor) for s in samples], pools,
                                repeats=int(args.baseline_repeats), seed=args.seed)
    # unseen colours: the odd 8-bit levels, i.e. the complement of the 128^3
    # training set (GLUT App A.1 "reserving the remaining colours for evaluation")
    gen = torch.Generator().manual_seed(args.seed)
    lv = heldout_color_levels()
    unseen = lv[torch.randint(0, lv.numel(), (2048, 3), generator=gen)].to(device)

    rows: list[dict[str, Any]] = []
    diag: dict[str, list[float]] = {k: [] for k in (
        "query_match_drift_repeat", "query_match_drift_path",
        "query_match_mu_shift_indexed", "query_match_mu_shift_hungarian")}
    field_pred_vals: list[float] = []
    field_pred_by_source: dict[str, list[float]] = {}
    pred_sources = _pred_field_sources(pred_field_dir)
    n_pred_missing = 0

    by_source: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        by_source.setdefault(s.source_image_id, []).append(i)

    for i, s in enumerate(samples):
        img = images.image(s).to(device=device)
        hw = (int(img.shape[1]), int(img.shape[2]))
        alpha = alphas.alpha(s, hw).to(device=device)
        field_gt = alphas.field(alpha, cfg.field_hw).to(device=device)
        i_star = bank.f_star_image(img, alpha.unsqueeze(0), s.lut_id)
        lab_star = srgb_to_lab(i_star.permute(1, 2, 0))
        z = zstore.get(s, "none").to(device=device)

        f_img = model.apply_image(img, z.unsqueeze(0), field_gt.unsqueeze(0))
        e_arm = headline_for_transform(img, alpha, f_img, i_star, lab_star)
        f_grid = transform_values(model, grid17, z, field_gt)
        gt_grid = bank.apply(grid17, s.lut_id)

        row: dict[str, Any] = {
            "sample_id": s.sample_id, "winner_confidence": s.winner_confidence,
            "task_type": s.task_type, "lut_id": s.lut_id,
            "source_image_id": s.source_image_id, "minor": s.minor,
            "alpha_mean": float(alpha.mean()),
            "E_arm": e_arm,
            "grid_error": float(crit.function_distance(f_grid, gt_grid)),
            "unseen_color_error": float(crit.function_distance(
                transform_values(model, unseen, z, field_gt),
                bank.apply(unseen, s.lut_id))),
            "lut_size": bank.size(s.lut_id),
        }
        # X_img: the 5-bit histogram measure of section 4.B
        hist_c, hist_w = _image_hist(img)
        row["img_error"] = float(crit.function_distance(
            transform_values(model, hist_c, z, field_gt),
            bank.apply(hist_c, s.lut_id), hist_w))

        # -- trivial baselines (section 4.C) --
        row["E_B0_identity"] = float(delta_e00(srgb_to_lab(img.permute(1, 2, 0)),
                                             lab_star).mean())
        row["E_B1_libmean"] = headline_for_transform(
            img, alpha, _apply_volume_image(b1_vol, img), i_star, lab_star)
        row["E_B2_librandom_repeats"] = [
            headline_for_transform(img, alpha,
                                   bank.apply_image(img, b2_draws[r][i]), i_star,
                                   lab_star)
            for r in range(len(b2_draws))]
        b3 = [b3_draws[r][i] for r in range(len(b3_draws))]
        row["E_B3_bucket_retrieval_repeats"] = [
            headline_for_transform(img, alpha, bank.apply_image(img, lid),
                                   i_star, lab_star)
            for lid in b3 if lid is not None]
        row["B3_pool_missing"] = int(sum(1 for lid in b3 if lid is None))

        gt_grid9 = bank.apply(grid9, s.lut_id)
        oracle = crit.oracle_lut_ids(lib, {s.sample_id: gt_grid9}, metric="de76")
        row["B4_lut_id"], row["B4_select_de76"] = oracle[s.sample_id]
        row["E_B4_oracle"] = headline_for_transform(
            img, alpha, bank.apply_image(img, row["B4_lut_id"]), i_star, lab_star)
        fill = crit.oracle_lut_ids(lib, {s.lut_id: gt_grid9}, metric="de76",
                                   exclude_self=True)
        row["B6_lut_id"], row["B6_select_de76"] = fill[s.lut_id]
        row["E_B6_libfill"] = headline_for_transform(
            img, alpha, bank.apply_image(img, row["B6_lut_id"]), i_star, lab_star)

        # -- negative controls (section 4.D): E and M, both or neither --
        for control, (e_key, m_key) in CONTROL_ROW_KEYS.items():
            try:
                z_c = zstore.get(s, control).to(device=device)
            except (KeyError, FileNotFoundError):
                continue
            f_c = model.apply_image(img, z_c.unsqueeze(0), field_gt.unsqueeze(0))
            row[e_key] = headline_for_transform(img, alpha, f_c, i_star, lab_star)
            row[m_key] = float(crit.function_distance(
                f_grid, transform_values(model, grid17, z_c, field_gt)))

        # -- locality (section 4.E) --
        i_hat = crit.compose_hat(img, alpha.unsqueeze(0), f_img)
        row.update(crit.locality_errors(i_hat, i_star, img, alpha))

        # -- field consumption, four rows (section 4.E) --
        row["field_gt"] = e_arm
        const_field = torch.full_like(field_gt, float(alpha.mean()))
        row["field_const"] = headline_for_transform(
            img, alpha,
            model.apply_image(img, z.unsqueeze(0), const_field.unsqueeze(0)),
            i_star, lab_star)
        other = samples[(i + 1) % len(samples)]
        shuf_alpha = alphas.alpha(other, hw).to(device=device)
        shuf_field = alphas.field(shuf_alpha, cfg.field_hw).to(device=device)
        row["field_shuffle"] = headline_for_transform(
            img, alpha,
            model.apply_image(img, z.unsqueeze(0), shuf_field.unsqueeze(0)),
            i_star, lab_star)
        pred_field = _pred_field(pred_field_dir, s, cfg, field_gt,
                                 synthetic=zstore.mode == "synthetic")
        if pred_field is None:
            n_pred_missing += 1
        else:
            row["field_pred"] = headline_for_transform(
                img, alpha,
                model.apply_image(img, z.unsqueeze(0), pred_field.unsqueeze(0)),
                i_star, lab_star)
            field_pred_vals.append(row["field_pred"])
            # which product wrote that field: the where arm's prediction on the
            # local rows, or the style rows' constant-one field, which is the
            # data law (`mask is None -> out = edited`) and not a prediction.
            tag = (pred_sources.get(s.sample_id, "unlabelled")
                   if pred_field_dir is not None else "synthetic")
            row["field_pred_source"] = tag
            field_pred_by_source.setdefault(tag, []).append(row["field_pred"])

        # -- this arm's diagnostics (section 3.6) --
        rep = qd.query_match_repeat(model, z.unsqueeze(0),
                                    field_gt.unsqueeze(0), repeats=8)
        diag["query_match_drift_repeat"].append(rep["query_match_drift_repeat"])
        partner = _path_partner(samples, by_source, i)
        z_b = zstore.get(samples[partner], "none").to(device=device)
        path = qd.query_match_path(model, z.unsqueeze(0), z_b.unsqueeze(0),
                                   field_gt.unsqueeze(0), k_steps=20)
        diag["query_match_drift_path"].append(path["query_match_drift_path"])
        diag["query_match_mu_shift_indexed"].append(
            path["query_match_mu_shift_indexed"])
        diag["query_match_mu_shift_hungarian"].append(
            path["query_match_mu_shift_hungarian"])
        row.update({k: v[-1] for k, v in diag.items()})
        rows.append(row)

    extra = {k: crit.describe(v) for k, v in diag.items()}
    extra["field_pred"] = {**crit.describe(field_pred_vals),
                           "n_missing": int(n_pred_missing),
                           "by_source": {k: len(v) for k, v
                                         in sorted(field_pred_by_source.items())}}
    # the pooled column above is the pre-registered one; these split it so a
    # board can never present the style rows' definitional field as a where-arm
    # prediction (main-agent ruling 2026-08-15)
    for tag, vals in sorted(field_pred_by_source.items()):
        extra[f"field_pred__{tag}"] = {
            **crit.describe(vals), "field_pred_source": tag,
            "quantity": "the field_pred row restricted to one product source"}
    extra["ladder_row"] = {"n": len(rows), "value": cfg.rung,
                           "quantity": "the fusion-ladder row this run is"}
    meta = {"lib_n": len(lib_ids), "b1_grid": B1_GRID,
            "baseline_repeats": int(args.baseline_repeats),
            "field_hw": list(cfg.field_hw),
            "field_resample": "q3vl.where.upsample.area_resize (frozen block)",
            "unseen_colors_n": int(unseen.shape[0]),
            "n_pred_field_missing": int(n_pred_missing)}
    return rows, {"extra_columns": extra, "meta": meta}


def _image_hist(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from q3vl.whatb.queries import image_histogram_colors

    return image_histogram_colors(img, bits=5, top_k=4096)


def _apply_volume_image(volume: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
    return apply_lut_volume(volume, img.permute(1, 2, 0)).permute(2, 0, 1)


def _path_partner(samples: Sequence[Sample], by_source: Mapping[str, Sequence[int]],
                  i: int) -> int:
    """A second condition for the interpolation path: same source if possible."""
    same = [j for j in by_source.get(samples[i].source_image_id, ()) if j != i]
    return same[0] if same else (i + 1) % len(samples)


def _pred_field_sources(pred_dir: Path | None) -> dict[str, str]:
    """``sample_id -> which product wrote that ``.npy``, per the product itself.

    The ``field_pred`` directory holds two kinds of field: the where arm
    predicts only ``render_mode == "local"`` rows, and V_what's ``style`` rows
    carry a constant-one field that is the data law (``mask is None -> out =
    edited``), not a prediction.  The tag is read from the product's own
    ``per_sample.jsonl`` rather than re-derived from ``task_type``, so the board
    reports what the artefact claims and an untagged file reads ``unlabelled``.
    """
    if pred_dir is None:
        return {}
    path = Path(pred_dir) / "per_sample.jsonl"
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec.get("sample_id")
            if sid:
                out[str(sid)] = str(rec.get("source") or "unlabelled")
    return out


def _pred_field(pred_dir: Path | None, sample: Sample, cfg: qd.QDualConfig,
                gt_field: torch.Tensor, *, synthetic: bool) -> torch.Tensor | None:
    """The where-arm predicted field for the ``field_pred`` row."""
    if pred_dir is not None:
        path = Path(pred_dir) / f"{sample.sample_id}.npy"
        if path.is_file():
            arr = torch.from_numpy(np.load(path).astype(np.float32))
            if arr.dim() == 3:
                arr = arr[0]
            return area_resize(arr[None, None], cfg.field_hw)[0, 0].clamp(0, 1)
        return None
    if synthetic:
        g = torch.Generator().manual_seed(_hash_seed("pred:" + sample.sample_id))
        noise = 0.1 * torch.randn(gt_field.shape, generator=g).to(gt_field.device)
        return (gt_field + noise).clamp(0, 1)
    return None


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    dataset_ver = K.apply_dataset_version(args)   # fills in --dataset-root
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = config_from_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but no CUDA device is visible")
    # --loss-level 4 stays refused (this arm writes no L_img column, EPR-029
    # :577); level 1 is EPR-030's pure-L1 caliber and zeroes both optional
    # weights through the shared ladder rule (carrier.py:347/:351).
    if int(args.loss_level) not in (1, 3):
        raise SystemExit(
            f"--loss-level {args.loss_level}: this arm's loss is exactly "
            "L_rec + lambda_hc L_hc + lambda_sparse R_sparse and it adds no "
            "image-space term (EPR-029 :577), so level 4 would pre-register an "
            "L_img column that is never written")
    lam_hc = K.effective_lambda_hc(qd.LAMBDA_HC, args.loss_level)
    lam_sparse = K.effective_lambda_sparse(qd.LAMBDA_SPARSE, args.loss_level)
    if cfg.field_kind == "pred" and not args.pred_field_dir and \
            args.z_source != "synthetic":
        raise SystemExit("--field-kind pred needs --pred-field-dir")

    run_name = args.run_name or f"qdual_{cfg.rung}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.out_root) / run_name
    (run_dir / "config").mkdir(parents=True, exist_ok=True)

    # ---- data ----
    train_rows = load_index(args.train_split, args.dataset_root)
    # --data's population is MEASURED (each source counted against its own
    # on-disk declaration, q3vl/whatb/splits.py); the merged n is never a literal.
    train_normal = (normal_only(train_rows) if args.data == "v2seg"
                    else K.train_normal_rows(args.data, split=args.train_split,
                                             root=args.dataset_root))
    eval_rows_all = normal_only(load_index(args.eval_split, args.dataset_root))
    if args.max_train_samples:
        train_normal = train_normal[: args.max_train_samples]
    eval_rows = (eval_rows_all[: args.eval_max_samples] if args.eval_max_samples
                 else eval_rows_all)
    train_samples = samples_from_index(train_normal)
    eval_samples = samples_from_index(eval_rows)

    # the 256 colour texts come from the WHOLE split, never from a --eval-max
    # truncation: the frozen block asks for 256 samples of this split's texts
    colorspan = colorspan_assertion(args, eval_rows_all)  # BEFORE any data loader
    if args.z_source != "synthetic":
        attach_minor(eval_samples, eval_rows)
    else:
        for s in eval_samples:
            s.minor = "synthetic_bucket"

    bank = LutBank(args.bank_dir)
    zstore = ZStore(mode=args.z_source, root=args.zcache_dir, split=args.eval_split,
                    checkpoint=args.base_checkpoint,
                    readout=WhatReadoutSpec(kind=args.readout,
                                            qtok=int(args.readout_qtok)),
                    k_rows=int(args.z_expand_k), expand=args.z_expand,
                    tags=CONTROLS, required=CONTROLS)
    ztrain = ZStore(mode=args.z_source, root=args.zcache_dir, split=args.train_split,
                    checkpoint=args.base_checkpoint,
                    readout=WhatReadoutSpec(kind=args.readout,
                                            qtok=int(args.readout_qtok)),
                    k_rows=int(args.z_expand_k), expand=args.z_expand,
                    tags=("none",),
                    # --data v2seg+l8: one condition over two caches, through
                    # the shared MultiZCache (caliber.open_train_z)
                    data=args.data, zcache_root_l8=args.zcache_root_l8,
                    seed=args.seed)
    # the checkpoint / readout_kind assertions happen HERE, before any loader
    zstore.preload(CONTROLS)
    ztrain.preload(("none",))
    amode = "synthetic" if args.z_source == "synthetic" else "real"
    alphas_train = AlphaStore(args.train_split, mode=amode)
    alphas_eval = AlphaStore(args.eval_split, mode=amode)
    images = ImageStore(mode=amode)

    # ---- model ----
    model = qd.QDualArm(cfg).to(device=device)
    steps_per_epoch = K.assert_steps_per_epoch(
        math.ceil(len(train_samples) / max(1, args.batch_samples)),
        n_train=len(train_samples), batch_samples=args.batch_samples,
        where="qdual steps_per_epoch")
    total_steps = int(args.total_steps or steps_per_epoch * args.epochs)
    if args.smoke:
        total_steps = min(total_steps, int(args.smoke_steps))
    args.quick_eval_every = min(int(args.quick_eval_every), max(1, total_steps))

    thresholds = DegeneracyThresholds()
    setup = {
        "arm": qd.ARM, "epr": qd.ARM_EPR, "axes": list(qd.ARM_AXES),
        "run_name": run_name, "run_dir": str(run_dir), "created_at": _now(),
        "argv": sys.argv[1:] if argv is None else list(argv),
        "flags": vars(args),
        "config": model.config,
        "readout": WhatReadoutSpec(kind=args.readout,
                                   qtok=int(args.readout_qtok)).to_dict(),
        "colorspan_check": colorspan,
        "frozen_block": frozen_block_record(args, len(train_samples),
                                            steps_per_epoch, total_steps),
        "caliber": {
            **K.horizon_record(
                data=args.data, n_train=len(train_samples),
                batch_split=f"{args.batch_samples}x{args.queries}",
                batch_samples=args.batch_samples,
                queries_per_sample=args.queries,
                steps_per_epoch=steps_per_epoch, total_steps=total_steps,
                base_lr=args.lr, epochs=int(args.epochs),
                zcache_root_l8=args.zcache_root_l8,
                loss_level=int(args.loss_level), lambda_hc=lam_hc,
                lambda_sparse=lam_sparse),
            "batch_split_flag": args.batch_split,
            "train_source_facts": (
                None if args.data == "v2seg" else
                train_source_facts(args.data, split=args.train_split,
                                   root=args.dataset_root)),
            "z_cache_train": ztrain.union_record,
        },
        "optimiser": {"name": "Adam", "lr": args.lr, "weight_decay": 0.0,
                      "warmup": None, "schedule": "CosineAnnealingLR",
                      "t_max": total_steps,
                      "geometry_lr_scale": args.geometry_lr_scale,
                      "slow_group": ["q_emb", "pe_r", "pe_g", "pe_b", "theta_base"],
                      "grad_clip": args.grad_clip,
                      "source": "GLUT section 4.1 + App A.1 (0.1x group is a "
                                "NOVEL mapping, EPR-029 NOTES 4)"},
        "precision": resolve_precision(args),
        "seed": args.seed,
        "degeneracy_thresholds": thresholds.as_dict(),
        "collapse_guard": {"m_threshold": qd.COLLAPSE_M_THRESHOLD,
                           "delta_threshold": qd.COLLAPSE_DELTA_THRESHOLD,
                           "novel": True, "source": "EPR-029 NOTES 7"},
        "source_sha256": {
            "arm": sha256_file(qd.__file__),
            "runner": sha256_file(__file__),
            # the shared z cache / CIELab this arm consumes rather than copies
            **{n: sha256_file(Path(qd.__file__).resolve().parents[1] / n)
               for n in ("zcache.py", "colorimetry.py")},
        },
        "data": {"dataset_version": args.dataset_version,
                 "dataset_version_facts": dataset_ver.facts(),
                 "dataset_root": args.dataset_root,
                 "train_split": args.train_split, "eval_split": args.eval_split,
                 "train_facts": split_facts(train_rows),
                 "n_train_used": len(train_samples),
                 "n_eval_used": len(eval_samples),
                 "bank": bank.facts(),
                 "z_eval": zstore.facts(), "z_train": ztrain.facts(),
                 "field_source": cfg.field_source,
                 "field_hw": list(cfg.field_hw),
                 "field_kind": cfg.field_kind,
                 "field_note": ("training and the headline consume GT alpha "
                                "(EPR-029 NOTES 12); the where-arm predicted "
                                "field appears only in the field_pred row")},
        "synthetic": args.z_source == "synthetic",
    }
    (run_dir / "config" / "run_setup.json").write_text(
        json.dumps(setup, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    (run_dir / "config" / "loss_preregistration.json").write_text(
        json.dumps(qd.loss_preregistration(cfg, loss_level=args.loss_level,
                                           lambda_hc=lam_hc,
                                           lambda_sparse=lam_sparse),
                   ensure_ascii=False, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"arm": qd.ARM, "dry_run": True, "data": args.data,
                          "n_train": len(train_samples),
                          "batch_split": f"{args.batch_samples}x{args.queries}",
                          "colours_per_step": args.batch_samples * args.queries,
                          "steps_per_epoch": steps_per_epoch,
                          "total_steps": total_steps, "base_lr": args.lr,
                          "loss_level": int(args.loss_level)}, indent=2),
              flush=True)
        return 0

    pools = _bucket_pools(args, train_rows)
    lib_ids = library_ids(train_rows, n_rows=args.lib_sample, seed=args.seed)
    pred_dir = Path(args.pred_field_dir) if args.pred_field_dir else None

    # ---- train ----
    train_report: dict[str, Any] = {"skipped": bool(args.eval_only)}
    step0_headline: dict[str, Any] = {"skipped": bool(args.eval_only)}
    if not args.eval_only:
        # measured on untouched parameters, before the optimiser exists
        step0_headline = step0_headline_witness(
            model, cfg=cfg, samples=eval_samples, zstore=zstore,
            alphas=alphas_eval, images=images, bank=bank, device=device,
            n=min(4, len(eval_samples)))
        setup["step0_headline_vs_B0"] = step0_headline
        (run_dir / "config" / "run_setup.json").write_text(
            json.dumps(setup, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
        quick = make_quick_eval(cfg=cfg, samples=eval_samples, zstore=zstore,
                                alphas=alphas_eval, device=device,
                                thresholds=thresholds, run_dir=run_dir,
                                n_samples=int(args.quick_eval_n),
                                pred_field_dir=pred_dir)
        select = make_selector(args=args, cfg=cfg, samples=eval_samples,
                               zstore=zstore, alphas=alphas_eval, images=images,
                               bank=bank, lib_ids=lib_ids, pools=pools,
                               device=device, run_dir=run_dir,
                               pred_field_dir=pred_dir) if args.eval_every else None
        train_report = train(model, args=args, cfg=cfg, samples=train_samples,
                             zstore=ztrain, alphas=alphas_train, bank=bank,
                             run_dir=run_dir, total_steps=total_steps,
                             steps_per_epoch=steps_per_epoch, device=device,
                             quick_eval_fn=quick, images=images,
                             pred_field_dir=pred_dir, lambda_hc=lam_hc,
                             lambda_sparse=lam_sparse, select_fn=select)
        train_report["step0_headline_vs_B0"] = step0_headline
        if select is not None:
            train_report["selection"] = dict(select.state)
        torch.save({"model": model.state_dict(), "config": cfg.to_dict()},
                   run_dir / "last.pt")

    # the guard is not tied to the training loop: --eval-only would otherwise
    # publish a board from a process that never ran it (W4)
    if degeneracy_check_ran() is None:
        make_quick_eval(cfg=cfg, samples=eval_samples, zstore=zstore,
                        alphas=alphas_eval, device=device, thresholds=thresholds,
                        run_dir=run_dir, n_samples=int(args.quick_eval_n),
                        pred_field_dir=pred_dir)(model, -1)

    # ---- evaluate ----
    rows, aux = evaluate(model, args=args, cfg=cfg, samples=eval_samples,
                         zstore=zstore, alphas=alphas_eval, images=images,
                         bank=bank, lib_ids=lib_ids, pools=pools, device=device,
                         pred_field_dir=pred_dir)
    board = crit.build_board(rows, arm=qd.ARM, split=args.eval_split,
                             extra_columns=aux["extra_columns"], seed=args.seed)
    board["published"] = not (args.smoke or args.z_source == "synthetic")
    board["eval_meta"] = {**aux["meta"], "step0_headline_vs_B0": step0_headline,
                          "selection": train_report.get("selection")}
    board["run_setup"] = {"source_sha256": setup["source_sha256"],
                          "config": model.config,
                          "frozen_block": setup["frozen_block"],
                          "synthetic": setup["synthetic"]}
    n3 = board["criteria_columns"].get("N3_const_delta", {})
    m3 = board["criteria_columns"].get("N3_const_M", {})
    board["collapse_guard"] = qd.collapse_guard(n3.get("delta"), m3.get("mean"))

    # the step-0 witness comes back through the same three tiers the publication
    # gate uses, so an --eval-only board reads it off disk instead of losing it
    first_row, row_source = pub.resolve_first_step_row(
        train_report.get("first_step_row"), steps_path=run_dir / "steps.jsonl")
    attach_zeroinit_column(board, first_row, cfg=cfg, source=row_source)

    extra_steps = ["L_total", "R_sparse", "ladder_row"]
    if cfg.zero_init_head:
        extra_steps.append("zeroinit_step0_maxabs")
    report = pub.assert_publishable(
        board, qd.ARM, steps_row=first_row,
        steps_path=run_dir / "steps.jsonl", eval_only=bool(args.eval_only),
        loss_level=int(args.loss_level),
        extra_step_columns=tuple(extra_steps),
        required=qd.required_criteria_table(cfg))
    board["publication_report"] = report
    (run_dir / "metrics.json").write_text(
        json.dumps(board, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    headline = board["contexts"]["all"]["headline_normal_only"]
    print(f"[{_now()}] board written: {run_dir/'metrics.json'} "
          f"headline_normal_only mean={headline.get('mean')} n={headline.get('n')} "
          f"published={board['published']}", flush=True)
    return 0


def _bucket_pools(args: argparse.Namespace, train_rows: Sequence[IndexRow]
                  ) -> dict[str, list[str]]:
    """B3's ``minor -> train lut_id`` pools, cached because it reads 159k records."""
    cache = Path(args.bucket_pool_cache) if args.bucket_pool_cache else None
    if cache and cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    if args.z_source == "synthetic":
        pools: dict[str, list[str]] = {}
        for r in train_rows:
            pools.setdefault("synthetic_bucket", []).append(r.lut_id)
        return {k: sorted(set(v)) for k, v in pools.items()}
    pools = bucket_pools(iter_records(list(train_rows)))
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(pools), encoding="utf-8")
    return pools


if __name__ == "__main__":     # pragma: no cover
    raise SystemExit(main())
