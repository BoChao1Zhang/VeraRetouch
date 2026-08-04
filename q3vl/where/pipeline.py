"""Turn a frozen split index into ``WhereASample``s: image -> F_pre -> phi target.

This is the only place that touches the real VLM and the real shards.  It keeps
the spec-5 image contract (short side 512, aspect preserved, 32-aligned) and
cross-checks the HF processor's grid against our own plan, so a silent geometry
drift cannot reach the basis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch

from q3vl.train.imageproc import assert_grid_matches, prepare_image
from q3vl.train.shards import ShardIndex, ShardStore

from .calibrate import WhereASample
from .config import EXCLUDE_WINNER_CONFIDENCE_LOW, LOCAL_BUILDS, PhiConfig
from .fpre import FPreHook, grid_from_geometry
from .maskdata import (
    MaskResolver, MaskViewStore, aspect_check, eligibility, mask_stats, mask_views,
    split_index_path,
)
from .upsample import area_resize, luma_guide

__all__ = ["PreparedSample", "WhereADataSource"]


@dataclass
class PreparedSample:
    sample: WhereASample
    mask_hi: torch.Tensor          # (out_h, out_w)
    image_hi: torch.Tensor         # (3, out_h, out_w) sRGB in [0,1]
    geometry: Any
    record: dict[str, Any]
    mask_ref: Any
    diag: dict[str, Any]

    def guide(self) -> torch.Tensor:
        return luma_guide(self.image_hi.unsqueeze(0))


def _to_tensor(img) -> torch.Tensor:
    a = np.asarray(img, dtype=np.uint8)
    return torch.from_numpy(a).permute(2, 0, 1).float() / 255.0


class WhereADataSource:
    """Reads a split, extracts ``F_pre`` and attaches the GT mask views."""

    def __init__(
        self,
        visual: torch.nn.Module | None,
        processor,
        *,
        device: str = "cpu",
        verify: str = "checksum",
        phi_cfg: PhiConfig | None = None,
        exclude_low: bool = EXCLUDE_WINNER_CONFIDENCE_LOW,
        maskview_root: str | None = None,
        attach_hi: bool = False,
    ):
        self.visual = visual
        self.processor = processor
        self.device = device
        self.store = ShardStore("/", verify=verify)
        self.resolver = MaskResolver(verify=verify)
        self.phi_cfg = phi_cfg or PhiConfig()
        self.exclude_low = exclude_low
        # ``attach_hi`` fills WhereASample.mask_hi/guide_hi so the evaluation
        # path can measure the ceiling at delivery resolution (B-4).  Training
        # does not need it and pays ~1.5 MB/sample for it, so it is off there.
        self.attach_hi = attach_hi
        self.maskviews = MaskViewStore(maskview_root, verify=verify) if maskview_root else None
        self.rejections: list[dict[str, Any]] = []

    # -- one sample ---------------------------------------------------------
    def prepare(self, ref, record: dict[str, Any]) -> PreparedSample | None:
        ok, reason = eligibility(record, exclude_low=self.exclude_low)
        if not ok:
            self.rejections.append({"sample_id": record.get("sample_id"), "reason": reason})
            return None
        img, geom = prepare_image(self.store.read(ref.members["image"]))
        grid_h, grid_w = grid_from_geometry(geom.out_h, geom.out_w)
        if (grid_h, grid_w) != (geom.grid_h, geom.grid_w):
            raise AssertionError(
                f"{record['sample_id']}: F_pre grid {(grid_h, grid_w)} != planned "
                f"{(geom.grid_h, geom.grid_w)}"
            )

        # published mask views first (one decode ever), build fallback second
        mask_ref = None
        cached = self.maskviews.get(record["sample_id"]) if self.maskviews else None
        if cached is not None:
            mask_hi, mask_low, view_meta = cached
            ac = view_meta.get("aspect", {"ok": True, "source": "maskview_shard"})
            raw_shape = view_meta.get("mask_raw_shape")
            if tuple(mask_low.shape) != (grid_h, grid_w):
                raise AssertionError(
                    f"{record['sample_id']}: published mask_low {tuple(mask_low.shape)} "
                    f"!= F_pre grid {(grid_h, grid_w)}"
                )
        else:
            # N-19: a single unreadable .cgt.png used to raise straight through
            # and kill the epoch.  One bad sample out of 75,544 must cost one
            # sample, not a GPU-day -- and it must be *recorded*, so the
            # end-of-epoch count can be reconciled instead of merely failing.
            try:
                mask_ref = self.resolver.resolve(record)
                mask = self.resolver.load(mask_ref)
            except Exception as exc:                   # noqa: BLE001
                self.rejections.append({
                    "sample_id": record.get("sample_id"), "reason": "mask_io",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                return None
            ac = aspect_check(record, mask)
            if not ac["ok"]:
                # same rule as the packing job: an aspect mismatch is a dropped
                # sample, not a diagnostic footnote, so both paths see the same
                # population (REVIEW-impl-WhereA N-4)
                self.rejections.append({"sample_id": record["sample_id"],
                                        "reason": "aspect_mismatch", "detail": ac})
                return None
            mask_hi, mask_low = mask_views(mask, geom.out_h, geom.out_w, grid_h, grid_w)
            raw_shape = list(mask.shape)

        image_hi = _to_tensor(img)
        img_low = area_resize(image_hi.unsqueeze(0), (grid_h, grid_w))[0]

        fpre = self.extract_fpre(img, geom)
        guide_hi = luma_guide(image_hi.unsqueeze(0)) if self.attach_hi else None
        sample = WhereASample(
            sample_id=record["sample_id"],
            fpre=fpre.reshape(grid_h * grid_w, -1),
            img_low=img_low,
            mask_low=mask_low.reshape(-1),
            grid_h=grid_h, grid_w=grid_w,
            mask_hi=mask_hi if self.attach_hi else None,
            guide_hi=guide_hi,
            meta={
                "build": record.get("build"),
                "winner_confidence": record.get("winner_confidence"),
                "upscaled": (record.get("image") or {}).get("upscaled"),
                "region": record.get("region"),
                "lut_id": record.get("lut_id"),
                "source_image_id": record.get("source_image_id"),
                "out_h": geom.out_h, "out_w": geom.out_w,
                "mask_source": "maskview_shard" if cached is not None else "build_cgt_png",
            },
        )
        return PreparedSample(
            sample=sample, mask_hi=mask_hi, image_hi=image_hi, geometry=geom,
            record=record, mask_ref=mask_ref,
            diag={"aspect": ac, "mask": mask_stats(mask_low),
                  "mask_raw_shape": raw_shape},
        )

    def extract_fpre(self, img, geom) -> torch.Tensor:
        """``(grid_h, grid_w, 1024)`` from the frozen vision tower."""
        if self.visual is None:
            raise RuntimeError("no vision tower attached; F_pre needs the real model")
        inputs = self.processor.image_processor(
            images=[img], do_resize=False, return_tensors="pt"
        )
        assert_grid_matches(geom, inputs["image_grid_thw"][0])
        hook = FPreHook(self.visual)
        dtype = next(self.visual.parameters()).dtype
        with torch.no_grad(), hook.attached():
            self.visual(
                inputs["pixel_values"].to(self.device, dtype),
                grid_thw=inputs["image_grid_thw"].to(self.device),
            )
        return hook.split(inputs["image_grid_thw"])[0].float()

    # -- iteration ----------------------------------------------------------
    def iter_split(
        self, split: str, limit: int | None = None, local_only: bool = True
    ) -> Iterator[PreparedSample]:
        index = ShardIndex.load(split_index_path(split))
        n = 0
        for ref in index.samples:
            if local_only and ref.meta.get("build") not in LOCAL_BUILDS:
                continue
            record = json.loads(self.store.read(ref.members["record"]).decode("utf-8"))
            prepared = self.prepare(ref, record)
            if prepared is None:
                continue
            yield prepared
            n += 1
            if limit is not None and n >= limit:
                break

    def count_eligible(self, split: str, local_only: bool = True) -> dict[str, Any]:
        """``eligibility()``-filtered sample count for a split -- an **upper
        bound** on the samples the epoch will actually see.

        Reads records only (no images, no model) and is what the LR schedule is
        built from (REVIEW-impl-WhereA B-3).  ``prepare()`` can still drop a
        sample afterwards for a reason only visible once the mask is read
        (``aspect_mismatch``, ``mask_io``); those land in ``self.rejections`` and
        the driver reconciles the gap against them rather than failing blind
        (N-19).  The reason histogram is returned so a surprising count can be
        explained instead of guessed at.
        """
        index = ShardIndex.load(split_index_path(split))
        n_eligible = 0
        reasons: dict[str, int] = {}
        by_conf: dict[str, int] = {}
        for ref in index.samples:
            if local_only and ref.meta.get("build") not in LOCAL_BUILDS:
                continue
            record = json.loads(self.store.read(ref.members["record"]).decode("utf-8"))
            ok, reason = eligibility(record, exclude_low=self.exclude_low)
            if ok:
                n_eligible += 1
                conf = str(record.get("winner_confidence"))
                by_conf[conf] = by_conf.get(conf, 0) + 1
            else:
                reasons[reason] = reasons.get(reason, 0) + 1
        return {"split": split, "n_eligible": n_eligible,
                "is_upper_bound": True,
                "late_drop_reasons": ["aspect_mismatch", "mask_io"],
                "exclude_low": self.exclude_low,
                "skipped_reasons": reasons, "by_winner_confidence": by_conf}

    def close(self) -> None:
        self.store.close()
        self.resolver.close()
        if self.maskviews is not None:
            self.maskviews.close()
