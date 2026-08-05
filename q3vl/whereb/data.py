"""Where-B dataset and batch assembly.

Protocol 14.9 -- "prove that no main arm's input contains ``I_tar``, GT mask,
GT LUT or oracle latent" -- is enforced here, at the only place that touches the
raw records:

* :class:`WhereBSample` keeps a **whitelist** of record fields
  (:data:`META_KEYS`); the record's ``image.baked`` locator (which is exactly
  ``I_tar``) is dropped on load and never reaches a sample object;
* :class:`Batch` separates ``inputs`` (the five tensors
  :data:`q3vl.whereb.model.MODEL_INPUT_KEYS` allows) from ``targets``; the model
  is called as ``model(**batch.inputs)`` and physically cannot see a target.

Global samples (``render_mode == "global"``, builds g1-g4) have no ``.cgt`` mask
and no Where-A oracle latent.  Their GT mask is all ones and their oracle
auxiliaries are masked off; ``L_mask`` still applies, which is what makes the
protocol 5.6 gate "global mask soft-IoU >= 0.98" reachable at all.
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
    GLOBAL_BUILDS,
    LOCAL_BUILDS,
    ArmConfig,
    SPLIT_DIR,
)
from .context import (
    FIXED_PHRASE,
    GENERATED,
    GT,
    IRRELEVANT_WORDS,
    NULL,
    SHUFFLED,
    FormatStats,
    ShuffleIndex,
    WhereContext,
    fixed_phrase_context,
    generated_context,
    gt_context,
    irrelevant_words_context,
    null_context,
    shuffled_context,
)
from .fields import FrozenBasis, oracle_fields, phi_dir_fast
from .hiddens import EncodeItem, FrozenVLM
from .losses import curve_grid
from .model import MODEL_INPUT_KEYS

__all__ = ["META_KEYS", "WhereBSample", "WhereBDataset", "Batch", "BatchBuilder",
           "split_index_path"]

#: the only record fields a sample object keeps.  ``image`` is deliberately
#: absent: the record's ``image.baked`` entry is the ``I_tar`` locator.
META_KEYS = (
    "sample_id", "build", "build_id", "batch", "split", "task_type", "render_mode",
    "region", "source_image_id", "source_sample_id", "lut_id", "winner_confidence",
    "winner_rank", "candidate_id", "mask_id", "group", "major", "minor",
)
FORBIDDEN_INPUT_SUBSTRINGS = ("baked", "i_tar", "target_image", "gt_lut", "preset")


def split_index_path(split: str) -> Path:
    return SPLIT_DIR / f"{split}.index.jsonl"


@dataclass
class WhereBSample:
    sample_id: str
    image: Any                       # PIL image, spec-5 sized
    geometry: ImageGeometry
    instruction: str
    where_text: str
    meta: dict[str, Any]
    mask_hi: torch.Tensor | None = None      # (out_h, out_w) in [0,1]
    grid_h: int = 0
    grid_w: int = 0
    #: False when the dataset was opened with ``need_mask=False`` (the generation
    #: job never looks at a mask and should not pay for one)
    mask_loaded: bool = True

    @property
    def is_global(self) -> bool:
        return self.meta.get("render_mode") == "global"

    def mask_target_hi(self) -> torch.Tensor:
        """All-ones for a global sample; the ``.cgt`` projection for a local one.

        Raises rather than inventing an all-ones target for a local sample whose
        mask was never loaded -- that would be a silently wrong label.
        """
        if self.mask_hi is not None:
            return self.mask_hi
        if not self.is_global:
            if not self.mask_loaded:
                raise RuntimeError(
                    f"{self.sample_id}: local sample opened with need_mask=False; "
                    "its GT mask was never loaded and must not be faked as all-ones"
                )
            raise RuntimeError(f"{self.sample_id}: local sample has no GT mask")
        return torch.ones(self.geometry.out_h, self.geometry.out_w)

    def image_tensor(self) -> torch.Tensor:
        # np.array(..., copy=True): PIL hands back a read-only buffer and
        # torch.from_numpy on it produces a tensor torch calls undefined behaviour
        # to write to.
        a = np.array(self.image, dtype=np.uint8, copy=True)
        return torch.from_numpy(a).permute(2, 0, 1).float() / 255.0


class WhereBDataset:
    """A frozen split index -> :class:`WhereBSample`.  No ad-hoc filtering."""

    def __init__(
        self,
        split: str,
        *,
        index_path: Path | None = None,
        store: ShardStore | None = None,
        maskviews=None,                       # MaskViewStore | None
        mask_resolver=None,                   # q3vl.where.maskdata.MaskResolver | None
        need_mask: bool = True,
        include_global: bool = True,
        include_local: bool = True,
        exclude_low: bool = False,
        limit: int | None = None,
        verify: str = "checksum",
    ):
        # checked before any IO: a misconfigured dataset must fail here, not on
        # its first local sample forty minutes into a job (review blocker B2)
        if need_mask and maskviews is None and mask_resolver is None and include_local:
            raise ValueError(
                f"{split}: need_mask=True and include_local=True but neither a "
                "published MaskViewStore nor a live MaskResolver was given. Use "
                "q3vl.whereb.data.open_dataset(), which wires both up, or pass "
                "need_mask=False if this job does not read masks."
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
                self.rejections.append({"sample_id": ref.sample_id, "reason": f"build_{b}"})
                continue
            if exclude_low and ref.meta.get("winner_confidence") == "low":
                continue
            keep.append(ref)
            if limit is not None and len(keep) >= limit:
                break
        self.refs = keep

    def __len__(self) -> int:
        return len(self.refs)

    def meta_rows(self) -> list[dict[str, Any]]:
        """Light rows (index meta only) -- enough to build the shuffle index."""
        rows = []
        for ref in self.refs:
            m = dict(ref.meta)
            m["sample_id"] = ref.sample_id
            m.setdefault(
                "render_mode",
                "local" if m.get("build") in LOCAL_BUILDS else "global",
            )
            rows.append(m)
        return rows

    def record(self, i: int) -> dict[str, Any]:
        return json.loads(self.store.read(self.refs[i].members["record"]).decode("utf-8"))

    def shuffle_records(self) -> list[dict[str, Any]]:
        """Rows for :class:`~q3vl.whereb.context.ShuffleIndex`.

        Carries **both** halves of the swap -- protocol 5.4 exchanges the
        instruction and the where context together (review blocker B3) -- so the
        index can refuse a partner that is missing either.
        """
        rows = []
        for i, ref in enumerate(self.refs):
            rec = self.record(i)
            rows.append({
                "sample_id": ref.sample_id,
                "source_image_id": rec.get("source_image_id"),
                "render_mode": rec.get("render_mode"),
                "build": rec.get("build"),
                "where": rec.get("where", ""),
                "instruction": rec.get("instruction", ""),
            })
        return rows

    def __getitem__(self, i: int) -> WhereBSample:
        ref = self.refs[i]
        rec = self.record(i)
        meta = {k: rec.get(k) for k in META_KEYS if k in rec}
        meta.setdefault("sample_id", ref.sample_id)
        meta["upscaled"] = bool((rec.get("image") or {}).get("upscaled"))
        image, geom = prepare_image(self.store.read(ref.members["image"]))
        gh, gw = grid_from_geometry(geom.out_h, geom.out_w)
        mask = self._mask(ref.sample_id, rec, geom) if self.need_mask else None
        return WhereBSample(
            sample_id=ref.sample_id, image=image, geometry=geom,
            instruction=rec["instruction"], where_text=rec["where"],
            meta=meta, mask_hi=mask, grid_h=gh, grid_w=gw,
            mask_loaded=self.need_mask,
        )

    def _mask(self, sample_id: str, rec: dict[str, Any], geom) -> torch.Tensor | None:
        if rec.get("render_mode") == "global":
            return None                                    # all-ones, built lazily
        if self.maskviews is not None and self.maskviews.has(sample_id, self.maskviews.HI):
            m = self.maskviews.mask_hi(sample_id)
            if tuple(m.shape) != (geom.out_h, geom.out_w):
                raise ValueError(
                    f"{sample_id}: published mask_hi {tuple(m.shape)} != geometry "
                    f"{(geom.out_h, geom.out_w)}"
                )
            return m
        if self.mask_resolver is not None:
            from q3vl.where.maskdata import mask_views

            raw = self.mask_resolver.load(self.mask_resolver.resolve(rec))
            hi, _low = mask_views(raw, geom.out_h, geom.out_w, *grid_from_geometry(
                geom.out_h, geom.out_w))
            return hi
        raise RuntimeError(
            f"{sample_id} is a local sample but neither a published maskview store "
            "nor a live MaskResolver was provided"
        )


def open_dataset(
    split: str,
    *,
    need_mask: bool = True,
    maskview_root: Path | None = None,
    limit: int | None = None,
    verify: str = "checksum",
    **kwargs,
) -> tuple[WhereBDataset, dict[str, Any]]:
    """The single sanctioned way to build a :class:`WhereBDataset` for a job.

    Review blocker B2: ``make_generated_context.py`` and
    ``make_oracle_latents.py`` each constructed ``WhereBDataset(split)`` with
    neither a maskview store nor a mask resolver, so both died on their first
    local sample.  Wiring the mask sources up is exactly the kind of thing that
    belongs in one factory rather than in every call site:

    * prefer Where-A's published ``MaskViewStore`` (indexed shards, no sqlite);
    * fall back to a live ``MaskResolver`` when the shards do not exist yet
      (they are produced by Where-A's own S1, which may not have run);
    * ``need_mask=False`` skips both -- the generation job never reads a mask,
      and resolving 159,215 ``.cgt.png`` members for nothing is pure waste.

    Returns the dataset and a provenance dict for ``run_setup.json``.
    """
    from .config import WHERE_A_MASKVIEW_DIR

    info: dict[str, Any] = {"split": split, "need_mask": need_mask,
                            "mask_source": None}
    maskviews = resolver = None
    if need_mask:
        root = Path(maskview_root or WHERE_A_MASKVIEW_DIR) / split
        try:
            from .stores import MaskViewStore

            maskviews = MaskViewStore(root)
            info["mask_source"] = "published_maskviews"
            info["maskview_root"] = str(root)
            info["maskview_facts"] = maskviews.facts()
        except (FileNotFoundError, RuntimeError) as exc:
            info["maskview_unavailable"] = f"{type(exc).__name__}: {exc}"
            from q3vl.where.maskdata import MaskResolver

            resolver = MaskResolver(verify=verify)
            info["mask_source"] = "live_mask_resolver"
    ds = WhereBDataset(split, maskviews=maskviews, mask_resolver=resolver,
                       need_mask=need_mask, limit=limit, verify=verify, **kwargs)
    info["n_samples"] = len(ds)
    return ds, info


# --- batching ---------------------------------------------------------------

@dataclass
class Batch:
    inputs: dict[str, torch.Tensor]
    targets: list[dict[str, Any]]
    contexts: list[WhereContext]
    sample_ids: list[str]
    meta: list[dict[str, Any]] = field(default_factory=list)

    def check_inputs(self) -> None:
        extra = set(self.inputs) - set(MODEL_INPUT_KEYS)
        if extra:
            raise AssertionError(
                f"batch inputs carry non-whitelisted keys {sorted(extra)}; protocol 14.9"
            )
        for k in self.inputs:
            low = k.lower()
            if any(bad in low for bad in FORBIDDEN_INPUT_SUBSTRINGS):
                raise AssertionError(f"input key {k!r} looks like a target (protocol 14.9)")


class BatchBuilder:
    """Samples + context modes -> one model-ready :class:`Batch`.

    The frozen VLM is called exactly once per batch; the same forward yields
    ``F_pre`` (which also feeds ``phi_dir`` through the frozen basis) and
    ``H_where`` (which feeds the connector).  Protocol 2.3: no second visual
    forward, no persisted feature cache.
    """

    def __init__(
        self,
        collator: Sft2SegCollator,
        vlm: FrozenVLM,
        basis: FrozenBasis,
        cfg: ArmConfig,
        *,
        oracle=None,                    # OracleStore | None
        genctx=None,                    # GenContextStore | None
        shuffle_index: ShuffleIndex | None = None,
        device: str | torch.device = "cpu",
        control_seed: int = 0,
    ):
        self.collator = collator
        self.tokenizer = collator.tokenizer
        self.vlm = vlm
        self.basis = basis
        self.cfg = cfg
        self.oracle = oracle
        self.genctx = genctx
        self.shuffle_index = shuffle_index
        self.control_seed = control_seed
        self.device = torch.device(device)
        self.format_stats: dict[str, FormatStats] = {}
        self.close_id = int(self.tokenizer("</where>", add_special_tokens=False)["input_ids"][0])
        self.eos_id = self.tokenizer.eos_token_id
        self._z = curve_grid()

    # -- contexts -----------------------------------------------------------
    def context_for(self, sample: WhereBSample, mode: str) -> WhereContext:
        if mode == GT:
            ctx = gt_context(self.tokenizer, sample.sample_id, sample.where_text)
        elif mode == NULL:
            ctx = null_context()
        elif mode == GENERATED:
            if self.genctx is None:
                raise RuntimeError(
                    "generated context requested but no GenContextStore is attached; "
                    "run scripts/make_generated_context.py first (never fall back to GT)"
                )
            rec = self.genctx.record(sample.sample_id)
            ctx = generated_context(
                sample.sample_id, rec["generated_ids"], self.close_id,
                text=rec.get("generated_text", ""), eos_id=self.eos_id,
            )
        elif mode == IRRELEVANT_WORDS:
            ctx = irrelevant_words_context(self.tokenizer, sample.sample_id,
                                           seed=self.control_seed)
        elif mode == FIXED_PHRASE:
            ctx = fixed_phrase_context(self.tokenizer, sample.sample_id)
        elif mode == SHUFFLED:
            if self.shuffle_index is None:
                raise RuntimeError("shuffled context requested but no ShuffleIndex")
            partner = self.shuffle_index.partner_of(sample.sample_id)
            if partner is None:
                raise KeyError(
                    f"{sample.sample_id} has no shuffle partner in its "
                    f"{self.shuffle_index.group_keys} group; it must be reported as "
                    "uncovered, not paired across images"
                )
            row = self.shuffle_index.by_id[partner]
            # protocol 5.4 swaps instruction *and* where context, as a pair from
            # the same partner (review blocker B3 / ruling D-B15)
            ctx = shuffled_context(self.tokenizer, partner, row["where"],
                                   row["instruction"])
        else:
            raise ValueError(f"unknown context mode {mode!r}")
        self.format_stats.setdefault(mode, FormatStats()).update(ctx)
        return ctx

    # -- one batch ----------------------------------------------------------
    def build(self, samples: Sequence[WhereBSample], modes: Sequence[str]) -> Batch:
        if len(samples) != len(modes):
            raise ValueError("samples and modes must align")
        contexts = [self.context_for(s, m) for s, m in zip(samples, modes)]
        items = []
        for s, ctx in zip(samples, contexts):
            # ctx.instruction is set only by the shuffled context; every other
            # mode keeps the sample's own instruction (review blocker B3)
            enc = self.collator.encode_one(_PromptShim(s, instruction=ctx.instruction))
            n_p = enc["n_prompt_tokens"]
            items.append(EncodeItem(
                sample_id=s.sample_id, image=s.image,
                prompt_ids=enc["input_ids"][:n_p], where_ids=ctx.token_ids,
            ))
        encoded = self.vlm.encode(items)

        f_pre, f_pos, f_mask, targets = [], [], [], []
        for s, e in zip(samples, encoded):
            if (e.grid_h, e.grid_w) != (s.grid_h, s.grid_w):
                raise AssertionError(
                    f"{s.sample_id}: F_pre grid {(e.grid_h, e.grid_w)} != planned "
                    f"{(s.grid_h, s.grid_w)}"
                )
            flat = e.f_pre.reshape(e.grid_h * e.grid_w, -1)
            f_pre.append(flat)
            from .qwhere import fpre_grid_positions

            f_pos.append(fpre_grid_positions(e.grid_h, e.grid_w))
            f_mask.append(torch.ones(flat.shape[0], dtype=torch.bool))
            targets.append(self.targets_for(s, flat))

        inputs = {
            "f_pre": _pad_stack(f_pre).to(self.device),
            "f_pre_pos": _pad_stack(f_pos).to(self.device),
            "f_pre_mask": _pad_mask(f_mask).to(self.device),
            "h_where": _pad_stack([e.h_where for e in encoded], min_len=1).to(self.device),
            "h_where_mask": _pad_mask(
                [torch.ones(e.h_where.shape[0], dtype=torch.bool) for e in encoded],
                min_len=1,
            ).to(self.device),
        }
        batch = Batch(
            inputs=inputs, targets=targets, contexts=contexts,
            sample_ids=[s.sample_id for s in samples],
            meta=[dict(s.meta) for s in samples],
        )
        batch.check_inputs()
        return batch

    # -- per-sample targets -------------------------------------------------
    def targets_for(self, sample: WhereBSample, fpre_flat: torch.Tensor) -> dict[str, Any]:
        img = sample.image_tensor()
        img_low = area_resize(img.unsqueeze(0), (sample.grid_h, sample.grid_w))[0]
        guide = luma_guide(img.unsqueeze(0))
        with torch.no_grad():
            sem = self.basis(fpre_flat.to(self.basis.weight.device))
            phi = phi_dir_fast(sem, img_low.to(sem.device), sample.grid_h, sample.grid_w,
                               self.cfg.phi)
        tgt: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "phi_dir": phi.to(self.device),
            "guide_hi": guide.to(self.device),
            "grid_h": sample.grid_h, "grid_w": sample.grid_w,
            "mask_hi": sample.mask_target_hi().to(self.device),
            # amendment A-5: the grid-level / centre-prior criteria live on the
            # F_pre grid, so the GT is projected there once, here.
            "mask_low": area_resize(
                sample.mask_target_hi()[None, None],
                (sample.grid_h, sample.grid_w))[0, 0].to(self.device),
            "is_global": sample.is_global,
            "meta": {**sample.meta, "instruction": sample.instruction},
        }
        lat = None
        if self.oracle is not None and not sample.is_global:
            try:
                lat = self.oracle.latent(sample.sample_id, self.cfg.readout)
            except KeyError:
                lat = None
        if lat is not None:
            o = oracle_fields(phi.to(self.device), lat.to(self.device),
                              self.cfg.readout, self._z.to(self.device))
            tgt.update({"s_star": o["s_star"], "r_star": o["r_star"],
                        "w_dir_star": o["w_dir_star"], "has_oracle": True})
        else:
            tgt["has_oracle"] = False
        return tgt

    def stats(self) -> dict[str, Any]:
        return {m: s.to_dict() for m, s in sorted(self.format_stats.items())}


class _PromptShim:
    """Adapts a :class:`WhereBSample` to what ``Sft2SegCollator.encode_one`` reads.

    The colour body is a single placeholder token's worth of text: only
    ``n_prompt_tokens`` is used from the result, and the prompt ends before the
    first ``<where>``, so nothing downstream of it can influence the prompt ids.

    ``instruction`` overrides the sample's own instruction; the shuffled context
    is the only caller that passes it (protocol 5.4 / review blocker B3).
    """

    __slots__ = ("sample_id", "image", "geometry", "instruction", "where_text", "color_text")

    def __init__(self, s: WhereBSample, instruction: str | None = None):
        self.sample_id = s.sample_id
        self.image = s.image
        self.geometry = s.geometry
        self.instruction = s.instruction if instruction is None else instruction
        self.where_text = s.where_text
        self.color_text = "."


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
