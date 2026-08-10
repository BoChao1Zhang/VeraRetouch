"""Stage-What dataset, frozen-Where runner and batch assembly.

Protocol 14.9 -- "prove that no main arm's input contains ``I_tar``, a GT mask, a
GT LUT or an oracle latent" -- is enforced here, at the only place that touches
the raw records:

* :class:`WhatSample` keeps a whitelist of record fields (:data:`META_KEYS`).
  The record's ``image.baked`` locator *is* ``I_tar`` and is dropped on load, so
  no sample object can reach a training batch carrying it.  The evaluation path
  asks for it explicitly through :meth:`WhatDataset.load_target_image`, which
  exists only in :mod:`q3vl.what.evaluate`;
* :class:`Batch` separates ``inputs`` (the six entries
  :data:`q3vl.what.model.MODEL_INPUT_KEYS` allows) from ``targets``.  The model is
  called as ``model(**batch.inputs)`` and physically cannot see a target;
* the GT LUT reaches the batch **only** as ``T_gt(x)`` evaluated at the query
  points -- a target tensor, never an input -- and ``z_gt``, its frozen SRHT code.

The GT LUT function itself is the record's own ``preset_path``: a real ``.cube``
(98.5% of the corpus) or ``.3dl`` (1.5%) file, parsed with the same strict reader
the dataset build used.  There is nothing to re-render.  ``lut_id ->
preset_path`` was verified 1:1 on 481 sampled ids across train / V_what /
T_lut_unseen with zero conflicts and 481/481 files present (NOTES V-W1).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from q3vl.train.collator import Sft2SegCollator
from q3vl.train.imageproc import ImageGeometry, prepare_image
from q3vl.train.shards import ShardIndex, ShardStore
from q3vl.where.fpre import grid_from_geometry
from q3vl.where.upsample import area_resize, luma_guide

from .config import (
    COLOR_CONTEXT_MAX_TOKENS,
    CONTEXT_GENERATED,
    CONTEXT_GT,
    GLOBAL_BUILDS,
    GT_LUT_INTERP,
    LOCAL_BUILDS,
    N_QUERY_NATURAL,
    SPLIT_DIR,
    ArmConfig,
)
from .context import (
    ColorContext,
    FormatStats,
    generated_color_context,
    gt_color_context,
)
from .hiddens import ColorEncodeItem, WhatVLM
from .lut import LutBank
from .model import MODEL_INPUT_KEYS
from .queries import natural_query_points, query_kind_index, sample_seed, uniform_query_points
from .srht import encode_z_gt, u_of_table
from .wc import W_VECTOR_DIM, WhereSignals

__all__ = ["META_KEYS", "WhatSample", "WhatDataset", "open_dataset", "Batch",
           "WhatBatchBuilder", "WhereRunner", "split_index_path",
           "NATURAL_MASK_SOURCES", "ORACLE_MISSING_POLICIES"]

# ===========================================================================
# deviation D-EXEC4 (2026-08-10, main-agent ruling on task card EXEC-4)
# ===========================================================================
# Amendment A-3 makes the frozen Where checkpoint's ``m_pred`` the natural-half
# query weighting of **all twelve** arms.  The C wave is being run *before* a
# Where checkpoint exists (the Where-B main wave was halted; a new Where design
# is pending), so for the four control arms the main agent ruled:
#
#   C03/C04 (oracle ceiling)  -> weight by the GT mask they are already given;
#   C01/C02 (strict no-where) -> whole-image sampling, the natural reading of an
#                                arm that has no where information at all.
#
# This is a *declared* deviation, not a fallback: ``frozen_m_pred`` still
# refuses to run without a checkpoint, the chosen source is written into
# ``run_setup.json`` and into every per-sample row, and D-W10's twelve-arm
# unification is re-calibrated once the new Where stage is frozen (re-running
# the C wave at that point is a known, accepted cost).
NATURAL_MASK_FROZEN = "frozen_m_pred"
NATURAL_MASK_ORACLE_GT = "oracle_gt_mask"
NATURAL_MASK_GLOBAL = "global_uniform"
NATURAL_MASK_SOURCES = (NATURAL_MASK_FROZEN, NATURAL_MASK_ORACLE_GT,
                        NATURAL_MASK_GLOBAL)

# What an oracle arm does with a sample that has no Where-A oracle fit.  By
# construction that is exactly the global samples (verified 2026-08-10: the S5
# oracle covers 75,544/75,544 local train samples and 0 global ones) -- a global
# edit has no ROI, so there is no latent to fit.  ``reject`` is the original
# behaviour (raise); ``null_global`` keeps the twelve arms on one population by
# giving those samples the all-ones GT mask they already deserve plus a constant
# zero latent, and counts them.
ORACLE_MISSING_REJECT = "reject"
ORACLE_MISSING_NULL_GLOBAL = "null_global"
ORACLE_MISSING_POLICIES = (ORACLE_MISSING_REJECT, ORACLE_MISSING_NULL_GLOBAL)

#: the only record fields a sample object keeps.  ``image`` is deliberately
#: absent: the record's ``image.baked`` entry is the ``I_tar`` locator.
META_KEYS = (
    "sample_id", "build", "build_id", "batch", "split", "task_type", "render_mode",
    "region", "source_image_id", "source_sample_id", "lut_id", "winner_confidence",
    "winner_rank", "candidate_id", "mask_id", "group", "major", "minor",
)
FORBIDDEN_INPUT_SUBSTRINGS = ("baked", "i_tar", "target_image", "gt_lut", "t_gt",
                              "preset", "z_gt")


def split_index_path(split: str) -> Path:
    return SPLIT_DIR / f"{split}.index.jsonl"


@dataclass
class WhatSample:
    sample_id: str
    image: Any                         # PIL image, spec-5 sized
    geometry: ImageGeometry
    instruction: str
    where_text: str
    color_text: str
    lut_id: str
    preset_path: str
    meta: dict[str, Any]
    mask_hi: torch.Tensor | None = None   # GT mask: evaluation + oracle arms only
    grid_h: int = 0
    grid_w: int = 0
    #: locator of ``I_tar``; kept as a *string* so it can never be mistaken for a
    #: tensor input, and only resolved by the evaluation path.
    target_locator: dict[str, Any] | None = None

    @property
    def is_global(self) -> bool:
        return self.meta.get("render_mode") == "global"

    def image_tensor(self) -> torch.Tensor:
        a = np.array(self.image, dtype=np.uint8, copy=True)
        return torch.from_numpy(a).permute(2, 0, 1).float() / 255.0


class WhatDataset:
    """A frozen split index -> :class:`WhatSample`.  No ad-hoc filtering."""

    def __init__(self, split: str, *, index_path: Path | None = None,
                 store: ShardStore | None = None, maskviews=None,
                 mask_resolver=None, need_mask: bool = False,
                 include_global: bool = True, include_local: bool = True,
                 exclude_low: bool = False, limit: int | None = None,
                 verify: str = "checksum"):
        if need_mask and maskviews is None and mask_resolver is None and include_local:
            raise ValueError(
                f"{split}: need_mask=True but neither a published MaskViewStore nor "
                "a live MaskResolver was given.  Use q3vl.what.data.open_dataset()."
            )
        self.split = split
        self.index = ShardIndex.load(index_path or split_index_path(split))
        self.store = store or ShardStore("/", verify=verify)
        self.maskviews = maskviews
        self.mask_resolver = mask_resolver
        self.need_mask = need_mask
        self.rejections: list[dict[str, Any]] = []
        keep = []
        for ref in self.index.samples:
            b = ref.meta.get("build")
            if b in GLOBAL_BUILDS and not include_global:
                continue
            if b in LOCAL_BUILDS and not include_local:
                continue
            if b not in GLOBAL_BUILDS and b not in LOCAL_BUILDS:
                self.rejections.append({"sample_id": ref.sample_id,
                                        "reason": f"build_{b}"})
                continue
            if exclude_low and ref.meta.get("winner_confidence") == "low":
                continue
            keep.append(ref)
            if limit is not None and len(keep) >= limit:
                break
        self.refs = keep

    def __len__(self) -> int:
        return len(self.refs)

    def record(self, i: int) -> dict[str, Any]:
        return json.loads(self.store.read(self.refs[i].members["record"]).decode("utf-8"))

    def lut_path_map(self) -> dict[str, str]:
        """``lut_id -> preset_path`` over this split (input to :class:`LutBank`)."""
        out: dict[str, str] = {}
        for i in range(len(self)):
            rec = self.record(i)
            out.setdefault(rec["lut_id"], rec["preset_path"])
        return out

    def __getitem__(self, i: int) -> WhatSample:
        ref = self.refs[i]
        rec = self.record(i)
        meta = {k: rec.get(k) for k in META_KEYS if k in rec}
        meta.setdefault("sample_id", ref.sample_id)
        meta["upscaled"] = bool((rec.get("image") or {}).get("upscaled"))
        image, geom = prepare_image(self.store.read(ref.members["image"]))
        gh, gw = grid_from_geometry(geom.out_h, geom.out_w)
        mask = self._mask(ref.sample_id, rec, geom) if self.need_mask else None
        return WhatSample(
            sample_id=ref.sample_id, image=image, geometry=geom,
            instruction=rec["instruction"], where_text=rec["where"],
            color_text=rec["color"], lut_id=rec["lut_id"],
            preset_path=rec["preset_path"], meta=meta, mask_hi=mask,
            grid_h=gh, grid_w=gw,
            target_locator=(rec.get("image") or {}).get("baked"),
        )

    def _mask(self, sample_id: str, rec: dict[str, Any], geom) -> torch.Tensor | None:
        if rec.get("render_mode") == "global":
            return None                                   # all-ones, built lazily
        if self.maskviews is not None and self.maskviews.has(sample_id, self.maskviews.HI):
            return self.maskviews.mask_hi(sample_id)
        if self.mask_resolver is not None:
            from q3vl.where.maskdata import mask_views

            raw = self.mask_resolver.load(self.mask_resolver.resolve(rec))
            hi, _low = mask_views(raw, geom.out_h, geom.out_w,
                                  *grid_from_geometry(geom.out_h, geom.out_w))
            return hi
        raise RuntimeError(f"{sample_id}: local sample with no mask source")

    def load_target_image(self, sample: WhatSample) -> torch.Tensor:
        """``I_tar`` -- **evaluation only** (protocol 12.2).

        Deliberately not reachable from :class:`WhatBatchBuilder`: protocol 9.5
        says ``I_tar`` does not enter ``L_what``, and the way to keep that true is
        for the training path to have no call site for this method.
        """
        if sample.target_locator is None:
            raise KeyError(f"{sample.sample_id} has no baked-image locator")
        image, _geom = prepare_image(self.store.read(sample.target_locator))
        a = np.array(image, dtype=np.uint8, copy=True)
        return torch.from_numpy(a).permute(2, 0, 1).float() / 255.0


def open_dataset(split: str, *, need_mask: bool = False,
                 maskview_root: Path | None = None, limit: int | None = None,
                 verify: str = "checksum", **kwargs) -> tuple[WhatDataset, dict[str, Any]]:
    """The single sanctioned way to build a :class:`WhatDataset` for a job."""
    from q3vl.whereb.config import WHERE_A_MASKVIEW_DIR

    info: dict[str, Any] = {"split": split, "need_mask": need_mask,
                            "mask_source": None}
    maskviews = resolver = None
    if need_mask:
        root = Path(maskview_root or WHERE_A_MASKVIEW_DIR) / split
        try:
            from q3vl.whereb.stores import MaskViewStore

            maskviews = MaskViewStore(root)
            info["mask_source"] = "published_maskviews"
            info["maskview_root"] = str(root)
        except (FileNotFoundError, RuntimeError) as exc:
            info["maskview_unavailable"] = f"{type(exc).__name__}: {exc}"
            from q3vl.where.maskdata import MaskResolver

            resolver = MaskResolver(verify=verify)
            info["mask_source"] = "live_mask_resolver"
    ds = WhatDataset(split, maskviews=maskviews, mask_resolver=resolver,
                     need_mask=need_mask, limit=limit, verify=verify, **kwargs)
    info["n_samples"] = len(ds)
    return ds, info


# --- the frozen Where checkpoint -------------------------------------------

class WhereRunner:
    """Runs the one frozen Where-B checkpoint and packages its outputs.

    Protocol 6: "the Where checkpoint is exactly the same and frozen for every
    What arm".  One instance is shared by the whole run; it never builds
    gradients, and what it exposes is a :class:`~q3vl.what.wc.WhereSignals`, i.e.
    a structure with no field for anything protocol 14.9 forbids.
    """

    def __init__(self, model, basis, phi_cfg, upsample_cfg, readout: str,
                 device="cpu"):
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.basis = basis
        self.phi_cfg = phi_cfg
        self.upsample_cfg = upsample_cfg
        self.readout = readout
        self.device = torch.device(device)

    @classmethod
    def from_checkpoint(cls, path: Path, device="cpu") -> "WhereRunner":
        from q3vl.whereb.config import arm_config as where_arm_config
        from q3vl.whereb.fields import load_basis
        from q3vl.whereb.model import WhereBModel

        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = where_arm_config(ck["arm"])
        model = WhereBModel(cfg)
        model.load_state_dict(ck["model"])
        basis = load_basis(cfg.basis_arm)
        if ck.get("basis_digest") and basis.digest() != ck["basis_digest"]:
            raise RuntimeError(
                f"{path}: the checkpoint was trained on basis {ck['basis_digest']} "
                f"but {basis.digest()} is on disk; refusing to mix them"
            )
        return cls(model.to(device), basis.to(device), cfg.phi, cfg.upsample,
                   cfg.readout, device)

    @torch.no_grad()
    def signals(self, f_pre: torch.Tensor, f_pre_flat_list, h_where, h_where_mask,
                f_pos, f_mask, guides, grids) -> WhereSignals:
        """Predicted mask + latents for one batch (all tensors, no gradients)."""
        from q3vl.whereb.fields import phi_dir_fast, predict_fields

        out = self.model(f_pre, f_pos, f_mask, h_where, h_where_mask)
        m_low, m_hi = [], []
        for i, (gh, gw) in enumerate(grids):
            phi = phi_dir_fast(self.basis(f_pre_flat_list[i]), guides[i]["img_low"],
                               gh, gw, self.phi_cfg)
            params = {k: v.float() for k, v in out.select(i).items()}
            fields = predict_fields(phi.float(), params, self.readout, gh, gw,
                                    guide_hi=guides[i]["guide_hi"].float(),
                                    up_cfg=self.upsample_cfg,
                                    require_dtype=torch.float32)
            m_low.append(fields["m_low"])
            m_hi.append(fields["m_hi"].reshape(fields["m_hi"].shape[-2:]))
        rho_vec = torch.cat([v.reshape(v.shape[0], -1) for v in out.rho.values()], dim=-1)
        w_vec = torch.cat([out.w0.unsqueeze(-1), out.w_dir,
                           out.alpha.unsqueeze(-1)], dim=-1)
        return WhereSignals(
            m_low=torch.nn.utils.rnn.pad_sequence(m_low, batch_first=True),
            m_hi=m_hi, canvas_axis=out.canvas_axis, canvas_rho=out.canvas_rho,
            w_vec=w_vec, rho_vec=rho_vec, source="predicted",
            meta={"readout": self.readout},
        )


# --- batching ---------------------------------------------------------------

@dataclass
class Batch:
    inputs: dict[str, Any]
    targets: list[dict[str, Any]]
    sample_ids: list[str]
    meta: list[dict[str, Any]] = field(default_factory=list)
    #: amendment A-4: one :class:`~q3vl.what.context.ColorContext` per sample
    contexts: list[Any] = field(default_factory=list)

    def check_inputs(self, expect_source: str | None = None) -> None:
        extra = set(self.inputs) - set(MODEL_INPUT_KEYS)
        if extra:
            raise AssertionError(
                f"batch inputs carry non-whitelisted keys {sorted(extra)}; protocol 14.9")
        for k in self.inputs:
            if any(bad in k.lower() for bad in FORBIDDEN_INPUT_SUBSTRINGS):
                raise AssertionError(f"input key {k!r} looks like a target (protocol 14.9)")
        w = self.inputs.get("where")
        if w is not None and not isinstance(w, WhereSignals):
            raise AssertionError("the 'where' input must be a WhereSignals")
        # review nit N-14: a main arm handed an oracle WhereSignals would not be
        # caught by a key-name check.  Protocol 8.2 confines oracle input to
        # C03/C04, so the assertion is one line and belongs here.
        if expect_source is not None and w is not None and w.source != expect_source:
            raise AssertionError(
                f"arm expects where_source={expect_source!r} but the batch carries "
                f"{w.source!r} (protocol 8.2: oracle input only reaches C03/C04)")


class WhatBatchBuilder:
    """Samples -> one model-ready :class:`Batch` plus its protocol 9 targets.

    The frozen VLM runs once per batch and yields ``F_pre`` and ``H_color`` (and
    ``H_where`` when the arm keeps the prefix).  ``T_gt`` is evaluated per sample
    on that sample's own table -- the tables have different grid sizes across the
    corpus (16/25/32/33/64/65), so this cannot be batched, and it does not need
    to be: it is a target with no gradient.
    """

    def __init__(self, collator: Sft2SegCollator, vlm: WhatVLM, cfg: ArmConfig,
                 lut_bank: LutBank, zgt_center: torch.Tensor,
                 d_func_scale: float, *,
                 where_runner: WhereRunner | None = None,
                 oracle_store=None, color_genctx=None,
                 device: str | torch.device = "cpu",
                 gt_interp: str = GT_LUT_INTERP, seed: int = 0,
                 natural_mask_source: str = NATURAL_MASK_FROZEN,
                 oracle_missing_latent: str = ORACLE_MISSING_REJECT):
        self.collator = collator
        self.tokenizer = collator.tokenizer
        self.vlm = vlm
        self.cfg = cfg
        self.bank = lut_bank
        if zgt_center is None:
            raise ValueError(
                "z_gt needs mean_train_u; run scripts/make_zgt_center.py first. "
                "Training against an uncentred SRHT code would silently change "
                "the target every time the corpus changes."
            )
        if not d_func_scale or d_func_scale <= 0:
            raise ValueError(
                "L_style_dist needs the published train-set constant C "
                "(amendment A-2); run scripts/make_zgt_center.py first.  C must "
                "never be derived from the current batch."
            )
        self.zgt_center = zgt_center
        self.d_func_scale = float(d_func_scale)
        self.where_runner = where_runner
        self.oracle_store = oracle_store
        # --- deviation D-EXEC4 (see below and the run's NOTES) ---------------
        if natural_mask_source not in NATURAL_MASK_SOURCES:
            raise ValueError(
                f"unknown natural_mask_source {natural_mask_source!r}; "
                f"have {NATURAL_MASK_SOURCES}")
        if oracle_missing_latent not in ORACLE_MISSING_POLICIES:
            raise ValueError(
                f"unknown oracle_missing_latent {oracle_missing_latent!r}; "
                f"have {ORACLE_MISSING_POLICIES}")
        if natural_mask_source == NATURAL_MASK_FROZEN and where_runner is None:
            raise RuntimeError(
                f"{cfg.arm}: natural_mask_source='frozen_m_pred' (amendment A-3) "
                "needs the frozen Where checkpoint, but no WhereRunner was "
                "attached.  Either supply one, or declare the deviation "
                "explicitly with natural_mask_source in "
                f"{NATURAL_MASK_SOURCES[1:]} -- silently changing the query "
                "colour distribution is what A-3 exists to prevent.")
        if natural_mask_source == NATURAL_MASK_ORACLE_GT and cfg.where_source != "oracle":
            raise ValueError(
                f"{cfg.arm}: natural_mask_source='oracle_gt_mask' is only "
                "defined for the oracle ceiling arms (C03/C04), whose model "
                "input already carries the GT mask")
        self.natural_mask_source = natural_mask_source
        self.oracle_missing_latent = oracle_missing_latent
        #: how many samples got the null (all-zero) oracle latent, and why
        self.oracle_latent_stats: dict[str, int] = {"fitted": 0, "null_global": 0}
        #: amendment A-4: the published generated ``<color>`` context.  ``None``
        #: means this builder can only serve teacher batches, which is legal for
        #: a GT-context evaluation pass and illegal for training.
        self.color_genctx = color_genctx
        self.device = torch.device(device)
        self.gt_interp = gt_interp
        self.seed = seed
        self._x_uniform = uniform_query_points()
        self._kind = query_kind_index()
        self.format_stats: dict[str, FormatStats] = {}
        self.close_color_id = int(self.tokenizer(
            "</color>", add_special_tokens=False)["input_ids"][0])
        self.eos_id = self.tokenizer.eos_token_id

    # -- amendment A-4: the <color> context ---------------------------------
    def color_context(self, sample: WhatSample, mode: str) -> ColorContext:
        """GT or generated ``<color>`` span for one sample.

        The generated branch never sees ``sample.color_text``, so there is no
        value it could fall back to -- the no-GT-fallback rule of amendment A-4
        is a property of this function's inputs, not a check inside it.
        """
        if mode == CONTEXT_GT:
            ctx = gt_color_context(self.tokenizer, sample.sample_id,
                                   sample.color_text)
        elif mode == CONTEXT_GENERATED:
            if self.color_genctx is None:
                raise RuntimeError(
                    f"{sample.sample_id}: generated <color> context requested but "
                    "no ColorGenContextStore is attached.  Run the extended "
                    "make_generated_context job first (amendment A-4); never fall "
                    "back to the GT span."
                )
            rec = self.color_genctx.record(sample.sample_id)
            ctx = generated_color_context(
                sample.sample_id, rec["color_ids"], self.close_color_id,
                text=rec.get("color_text", ""), eos_id=self.eos_id,
                genctx_mode=rec["mode"],
                max_tokens=COLOR_CONTEXT_MAX_TOKENS,
            )
            if ctx.genctx_mode != self.cfg.genctx_mode:
                raise AssertionError(
                    f"{sample.sample_id}: context generated in {ctx.genctx_mode!r} "
                    f"but {self.cfg.arm} needs {self.cfg.genctx_mode!r}"
                )
        else:
            raise ValueError(f"unknown context mode {mode!r}")
        self.format_stats.setdefault(mode, FormatStats()).update(ctx)
        return ctx

    # -- query points and their GT -----------------------------------------
    def query_points(self, sample: WhatSample, m_hi: torch.Tensor | None
                     ) -> tuple[torch.Tensor, str]:
        """Protocol 9.1 / amendment A-3.

        The mask that weights the natural half is the **frozen Where
        checkpoint's** ``m_pred``, for all twelve arms.  It is supervision-side,
        the same kind of object as ``T_gt``, and ``where_source`` (predicted /
        none / oracle) changes only what the *model* is given.  Letting the
        no-where controls fall back to whole-image sampling would give them a
        different target colour distribution and make them "a lower bound plus one
        target-distribution ablation" instead of a clean lower bound
        (review blocker B-5).
        """
        img = sample.image_tensor()
        if sample.is_global:
            weighting = "global_uniform"           # protocol 9.1, global samples
            w = None
        elif self.natural_mask_source == NATURAL_MASK_GLOBAL:
            # declared deviation D-EXEC4 (C01/C02, no Where checkpoint exists)
            weighting, w = "global_uniform_declared", None
        elif m_hi is None:
            raise RuntimeError(
                f"{sample.sample_id}: local sample has no {self.natural_mask_source} "
                "mask for the natural query half.  Amendment A-3 makes this mask "
                "mandatory for every arm including C01-C04; falling back to "
                "whole-image sampling would silently change the loss for two of "
                "twelve arms."
            )
        elif self.natural_mask_source == NATURAL_MASK_ORACLE_GT:
            weighting, w = "gt_mask", m_hi
        else:
            weighting, w = "frozen_m_pred", m_hi
        nat = natural_query_points(
            img, N_QUERY_NATURAL, weights=w,
            seed=sample_seed(self.seed, sample.sample_id))
        return torch.cat([self._x_uniform, nat], dim=0), weighting

    def targets_for(self, sample: WhatSample, x: torch.Tensor,
                    natural_weighting: str) -> dict[str, Any]:
        table = self.bank.get(sample.lut_id)
        with torch.no_grad():
            t_gt = table.apply(x, self.gt_interp)
            grid_vals = table.apply(_zgt_grid(x.device, x.dtype), self.gt_interp)
            # u(T) raw: L_style_dist and the protocol 12.3 Spearman both need the
            # magnitude that z_gt's L2 normalisation throws away (amendment A-2)
            u_gt = u_of_table(grid_vals)
            z_gt = encode_z_gt(grid_vals, self.zgt_center)
        return {
            "sample_id": sample.sample_id, "lut_id": sample.lut_id,
            "x": x.to(self.device), "t_gt": t_gt.to(self.device),
            "z_gt": z_gt.to(self.device), "u_gt": u_gt.to(self.device),
            "query_kind": self._kind.to(self.device),
            "is_global": sample.is_global, "meta": dict(sample.meta),
            "natural_weighting": natural_weighting,
        }

    # -- one batch ----------------------------------------------------------
    def build(self, samples: Sequence[WhatSample],
              modes: Sequence[str] | None = None) -> Batch:
        modes = [CONTEXT_GT] * len(samples) if modes is None else list(modes)
        if len(modes) != len(samples):
            raise ValueError("samples and context modes must align")
        contexts = [self.color_context(s, m) for s, m in zip(samples, modes)]
        items, sup_items = [], []
        for s, ctx in zip(samples, contexts):
            enc = self.collator.encode_one(s)
            n_p, n_w = enc["n_prompt_tokens"], enc["n_where_tokens"]
            prompt_ids = enc["input_ids"][:n_p]
            gt_where_ids = enc["input_ids"][n_p:n_p + n_w]
            # amendment A-4: the <color> ids are the context's, teacher or
            # generated.  The GT slice of `enc` is deliberately not used here.
            color_ids = ctx.token_ids
            items.append(ColorEncodeItem(
                sample_id=s.sample_id, image=s.image, prompt_ids=prompt_ids,
                where_ids=gt_where_ids if self.cfg.where_prefix else [],
                color_ids=color_ids))
            # amendment A-3: the frozen Where checkpoint runs for *every* arm,
            # because its m_pred weights the natural query half of the loss.
            # C01/C02 drop the <where> prefix from the model's sequence, so their
            # supervision forward is a second, shorter one (prompt + <where>, no
            # colour body).  Two of twelve arms pay one extra forward; the
            # alternative is twelve arms optimising two different losses.
            # Under deviation D-EXEC4 there is no frozen checkpoint to feed, so
            # the extra forward has no consumer and is skipped.
            if not self.cfg.where_prefix and self.where_runner is not None:
                sup_items.append(ColorEncodeItem(
                    sample_id=s.sample_id, image=s.image, prompt_ids=prompt_ids,
                    where_ids=gt_where_ids, color_ids=[]))
        encoded = self.vlm.encode(items)
        sup_encoded = self.vlm.encode(sup_items) if sup_items else encoded

        f_flat, f_pos, f_mask, rgb_low, guides, grids = [], [], [], [], [], []
        for s, e in zip(samples, encoded):
            if (e.grid_h, e.grid_w) != (s.grid_h, s.grid_w):
                raise AssertionError(
                    f"{s.sample_id}: F_pre grid {(e.grid_h, e.grid_w)} != planned "
                    f"{(s.grid_h, s.grid_w)}")
            from q3vl.whereb.qwhere import fpre_grid_positions

            flat = e.f_pre.reshape(e.grid_h * e.grid_w, -1)
            img = s.image_tensor()
            low = area_resize(img.unsqueeze(0), (e.grid_h, e.grid_w))[0]
            f_flat.append(flat)
            f_pos.append(fpre_grid_positions(e.grid_h, e.grid_w))
            f_mask.append(torch.ones(flat.shape[0], dtype=torch.bool))
            rgb_low.append(low.reshape(3, -1).t())
            guides.append({"img_low": low, "guide_hi": luma_guide(img.unsqueeze(0))})
            grids.append((e.grid_h, e.grid_w))

        f_pre = _pad_stack(f_flat).to(self.device)
        inputs: dict[str, Any] = {
            "f_pre": f_pre,
            "f_pre_mask": _pad_mask(f_mask).to(self.device),
            "rgb_low": _pad_stack(rgb_low).to(self.device),
            "h_color": _pad_stack([e.h_color for e in encoded], min_len=1).to(self.device),
            "h_color_mask": _pad_mask(
                [torch.ones(e.h_color.shape[0], dtype=torch.bool) for e in encoded],
                min_len=1).to(self.device),
        }
        # amendment A-3: run the frozen Where checkpoint for every arm.  Its
        # output is the *supervision* mask; whether it also reaches the model is
        # decided one line below, by the arm's interface.  Under the declared
        # deviation D-EXEC4 there is no checkpoint and the supervision mask comes
        # from the arm's own declared source instead.
        frozen = (self._frozen_where(sup_encoded, f_pre, f_flat, f_pos,
                                     inputs["f_pre_mask"], guides, grids)
                  if self.natural_mask_source == NATURAL_MASK_FROZEN
                  else WhereSignals(source="none"))
        inputs["where"] = self._model_signals(samples, frozen, grids)

        # which signal set carries the supervision mask for the natural half
        sup_signals = (inputs["where"]
                       if self.natural_mask_source == NATURAL_MASK_ORACLE_GT
                       else frozen)
        targets = []
        for i, s in enumerate(samples):
            m_hi = None if sup_signals.m_hi is None else sup_signals.m_hi[i]
            x, weighting = self.query_points(s, m_hi)
            targets.append(self.targets_for(s, x, weighting))
        batch = Batch(inputs=inputs, targets=targets,
                      sample_ids=[s.sample_id for s in samples],
                      meta=[dict(s.meta) for s in samples],
                      contexts=contexts)
        batch.check_inputs(expect_source=self.cfg.where_source)
        return batch

    def _frozen_where(self, encoded, f_pre, f_flat, f_pos, f_mask, guides, grids
                      ) -> WhereSignals:
        """The frozen Where checkpoint's outputs -- always computed (A-3)."""
        if self.where_runner is None:
            raise RuntimeError(
                f"{self.cfg.arm} needs the frozen Where checkpoint but none was "
                "attached.  Protocol 6 requires the same frozen checkpoint for "
                "every arm, and amendment A-3 makes its m_pred the supervision "
                "mask of all twelve arms including C01-C04; there is no fallback."
            )
        h_where = _pad_stack([e.h_where for e in encoded], min_len=1).to(self.device)
        h_mask = _pad_mask(
            [torch.ones(e.h_where.shape[0], dtype=torch.bool) for e in encoded],
            min_len=1).to(self.device)
        return self.where_runner.signals(
            f_pre, [t.to(self.device) for t in f_flat], h_where, h_mask,
            _pad_stack(f_pos).to(self.device), f_mask, guides, grids)

    def _model_signals(self, samples, frozen: WhereSignals, grids) -> WhereSignals:
        """What the *model* is given, which is where the arms actually differ."""
        src = self.cfg.where_source
        if src == "none":
            return WhereSignals(source="none")
        if src == "oracle":
            return self._oracle_signals(samples, grids)
        return frozen

    def _oracle_signals(self, samples, grids) -> WhereSignals:
        """``C03``/``C04``: GT mask + Where-A oracle ``w*, rho*`` (ceiling only)."""
        if self.oracle_store is None:
            raise RuntimeError("the oracle arms need the Where-A OracleStore")
        from q3vl.where.upsample import area_resize as _ar

        from q3vl.whereb.heads import rho_numel

        n_rho = rho_numel(self.cfg.where_readout)
        m_low, m_hi, w_vecs, rho_vecs = [], [], [], []
        for s, (gh, gw) in zip(samples, grids):
            mask = (torch.ones(s.geometry.out_h, s.geometry.out_w)
                    if s.mask_hi is None else s.mask_hi)
            m_hi.append(mask)
            m_low.append(_ar(mask[None, None], (gh, gw))[0, 0].reshape(-1))
            lat = self._oracle_latent(s)
            if lat is None:
                # policy ``null_global`` only; ``reject`` raised inside the helper.
                # A global edit has no ROI, so the Where-A fit has nothing to say
                # about it: the mask above is honestly all-ones and the latent is
                # a constant, recorded per sample rather than imputed silently.
                self.oracle_latent_stats["null_global"] += 1
                w_vecs.append(torch.zeros(W_VECTOR_DIM))
                rho_vecs.append(torch.zeros(n_rho))
                continue
            from q3vl.where.basis import alpha_of, w_dir_of

            self.oracle_latent_stats["fitted"] += 1
            w_vecs.append(torch.cat([lat.w0.reshape(1), w_dir_of(lat.w_raw),
                                     alpha_of(lat.alpha_raw).reshape(1)]))
            rho_vecs.append(torch.cat([v.reshape(-1) for v in lat.rho.values()]))
        return WhereSignals(
            m_low=torch.nn.utils.rnn.pad_sequence(m_low, batch_first=True).to(self.device),
            m_hi=m_hi, w_vec=torch.stack(w_vecs).to(self.device),
            rho_vec=torch.stack(rho_vecs).to(self.device), source="oracle",
            meta={"mask": "gt", "latent": "where_a_oracle"},
        )

    def _oracle_latent(self, s: WhatSample):
        """The Where-A oracle latent, or ``None`` under the ``null_global`` policy.

        A sample with no published oracle payload at all (every global sample)
        raises out of the store, so both "no record" and "record with no usable
        fit" have to funnel through one place.
        """
        try:
            lat = self.oracle_store.latent(s.sample_id, self.cfg.where_readout)
        except (KeyError, FileNotFoundError):
            lat = None
        if lat is not None:
            return lat
        if self.oracle_missing_latent == ORACLE_MISSING_NULL_GLOBAL and s.is_global:
            return None
        raise KeyError(
            f"{s.sample_id}: no usable Where-A oracle latent (readout "
            f"{self.cfg.where_readout!r}, policy {self.oracle_missing_latent!r}); "
            "an oracle ceiling arm must reject the sample, not fabricate a latent")

    def facts(self) -> dict[str, Any]:
        return {
            "gt_interp": self.gt_interp, "lut_bank": self.bank.facts(),
            "vlm": self.vlm.facts(), "where_source": self.cfg.where_source,
            "d_func_scale": self.d_func_scale,
            "zgt_center_sha256": __import__("hashlib").sha256(
                self.zgt_center.detach().cpu().numpy().tobytes()).hexdigest(),
            # amendment A-3: true for all twelve arms, including C01-C04 --
            # unless deviation D-EXEC4 is declared (no frozen Where checkpoint)
            "natural_weighting": (
                "frozen_m_pred (global samples: global_uniform)"
                if self.natural_mask_source == NATURAL_MASK_FROZEN
                else f"{self.natural_mask_source} (deviation D-EXEC4; "
                     "global samples: global_uniform)"),
            "natural_mask_source": self.natural_mask_source,
            "a3_unified_natural_sampling": (
                self.natural_mask_source == NATURAL_MASK_FROZEN),
            "oracle_missing_latent": (self.oracle_missing_latent
                                      if self.cfg.where_source == "oracle" else None),
            "oracle_latent_stats": (dict(self.oracle_latent_stats)
                                    if self.cfg.where_source == "oracle" else None),
            "supervision_forward": (
                "not run (deviation D-EXEC4: no frozen Where checkpoint)"
                if self.natural_mask_source != NATURAL_MASK_FROZEN
                else ("shared with the model forward" if self.cfg.where_prefix
                      else "separate prompt+<where> forward")),
            # amendment A-4
            "genctx_mode": self.cfg.genctx_mode,
            "color_genctx": (self.color_genctx.summary()
                             if self.color_genctx is not None else None),
            "color_context_max_tokens": COLOR_CONTEXT_MAX_TOKENS,
            "format_stats": {m: st.to_dict()
                             for m, st in sorted(self.format_stats.items())},
        }


_ZGT_GRID_CACHE: torch.Tensor | None = None


def _zgt_grid(device, dtype) -> torch.Tensor:
    global _ZGT_GRID_CACHE
    if _ZGT_GRID_CACHE is None:
        from .srht import identity_grid

        _ZGT_GRID_CACHE = identity_grid()
    return _ZGT_GRID_CACHE.to(device=device, dtype=dtype)


def _pad_stack(xs: Sequence[torch.Tensor], min_len: int = 0) -> torch.Tensor:
    n = max([x.shape[0] for x in xs] + [min_len])
    dim = xs[0].shape[1] if xs[0].dim() > 1 else 1
    out = torch.zeros(len(xs), n, dim, dtype=xs[0].dtype)
    for i, x in enumerate(xs):
        if x.shape[0]:
            out[i, : x.shape[0]] = x if x.dim() > 1 else x.unsqueeze(1)
    return out


def _pad_mask(ms: Sequence[torch.Tensor], min_len: int = 0) -> torch.Tensor:
    n = max([m.shape[0] for m in ms] + [min_len])
    out = torch.zeros(len(ms), n, dtype=torch.bool)
    for i, m in enumerate(ms):
        if m.shape[0]:
            out[i, : m.shape[0]] = m
    return out
