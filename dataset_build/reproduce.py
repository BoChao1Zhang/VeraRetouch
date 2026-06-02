"""
dataset_build/reproduce.py
==========================
Reference reproduction helpers for DATAGEN v2 samples.

This module makes the plan's train-time contract executable without owning the
full trainer Stage-0 implementation. It reconstructs the before/target image
pair from one JSONL Sample and records what remains for Stage-0:

- PARAM samples: target = VeraRetouch teacher render(source, params).
- LUT samples: target = apply_lut(source, LUT).
- S5 real JPG: target = stored expert JPEG path.
- S1/S7 provenance=DEGRADE: input = teacher render(source, neg_p), target =
  clean source, and z* search is reported as required but not performed here.

Top-level imports stay light; heavy image/model deps are lazy inside methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from dataset_build.contracts import AfterSource, MaskSource, Provenance, RecipeKind, Sample
from dataset_build.pack import jsonl_to_sample


@dataclass
class ReproducedPair:
    """Executable representation of the DATAGEN pair contract."""

    sample_id: str
    input_rgb: Optional[Any]
    target_rgb: Optional[Any]
    cgt01: Optional[Any] = None
    raw_mask01: Optional[Any] = None
    z_star: Optional[Dict[str, Dict[str, float]]] = None
    needs_z_search: bool = False
    er_recon_psnr: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArrayRenderAdapter:
    """Adapt a path-based renderer to `render_fn(input_rgb, params)`.

    VeraRetouchRenderer currently renders from image paths. Stage-0 z* search,
    however, needs to repeatedly render from the teacher-degraded input array.
    This adapter bridges that gap by writing a transient PNG under scratch_dir,
    calling `renderer.render([tmp], [params])`, and deleting the file.
    """

    renderer: Any
    scratch_dir: Optional[str] = None
    prefix: str = "zsearch"
    keep_tmp: bool = False

    def __call__(self, input_rgb: Any, params: Dict[str, Dict[str, float]]) -> Any:
        path = self._write_tmp(input_rgb, params)
        try:
            return _render_one(self.renderer, path, params)
        finally:
            if not self.keep_tmp:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def _write_tmp(self, input_rgb: Any, params: Dict[str, Dict[str, float]]) -> str:
        import cv2
        import numpy as np

        root = Path(self.scratch_dir or tempfile.gettempdir()) / "datagen_zsearch"
        root.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(input_rgb)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"input_rgb must be HxWx3, got {arr.shape}")
        digest = hashlib.sha1(
            arr.tobytes() + repr(sorted((params or {}).items())).encode("utf-8")
        ).hexdigest()[:16]
        path = root / f"{self.prefix}_{digest}.png"
        bgr = cv2.cvtColor(arr.astype("uint8"), cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(path), bgr):
            raise OSError(f"failed to write z-search temp image {path}")
        return str(path)


def reproduce_pair(
    sample: Sample,
    *,
    renderer: Optional[Any] = None,
    parser: Optional[Any] = None,
    lut_applier: Optional[Any] = None,
    lut_resolver: Optional[Callable[[Sample], Optional[str]]] = None,
    run_z_search: bool = False,
    z_search_render_fn: Optional[Callable[[Any, Dict[str, Dict[str, float]]], Any]] = None,
    z_search_scratch_dir: Optional[str] = None,
    z_search_keys: Optional[Sequence[str]] = None,
    z_search_max_iters: int = 2,
    er_recon_psnr_min: float = 25.0,
) -> ReproducedPair:
    """Reproduce a DATAGEN v2 sample's input/target pixels.

    `run_z_search=True` performs a small derivative-free coordinate search when
    `z_search_render_fn` is supplied. The function must render from the already
    materialized input RGB array with candidate params:
    `render_fn(input_rgb, params) -> RGB uint8`.
    """
    if sample.recipe.provenance == Provenance.DEGRADE:
        if renderer is None:
            raise ValueError("DEGRADE reproduction requires a renderer for render(source, neg_p)")
        input_rgb = _render_one(renderer, sample.source_path, sample.recipe.params or {})
        target_rgb = _load_rgb(sample.source_path)
        pair = ReproducedPair(
            sample_id=sample.sample_id,
            input_rgb=input_rgb,
            target_rgb=target_rgb,
            cgt01=_load_mask(sample.c_gt.cgt_path),
            raw_mask01=_load_mask(sample.c_gt.raw_mask_path),
            needs_z_search=True,
            meta={
                "contract": "degrade_teacher_input",
                "warm_start": _warm_start_from_degrade(sample),
            },
        )
        if run_z_search:
            if z_search_render_fn is None:
                if z_search_scratch_dir:
                    z_search_render_fn = ArrayRenderAdapter(
                        renderer, scratch_dir=z_search_scratch_dir, prefix=sample.sample_id
                    )
                else:
                    raise ValueError(
                        "run_z_search=True requires z_search_render_fn(input_rgb, params) "
                        "or z_search_scratch_dir for the built-in path-renderer adapter."
                    )
            z_star, psnr = inverse_search_z_star(
                input_rgb,
                target_rgb,
                pair.meta["warm_start"],
                z_search_render_fn,
                keys=z_search_keys,
                max_iters=z_search_max_iters,
            )
            pair.z_star = z_star
            pair.er_recon_psnr = psnr
            pair.needs_z_search = psnr < er_recon_psnr_min
            pair.meta["er_recon_psnr_min"] = er_recon_psnr_min
        return pair

    if sample.after_source == AfterSource.REAL_JPG:
        after_path = sample.meta.get("expert_after_path")
        if not after_path:
            raise ValueError(f"{sample.sample_id}: after_source=real_jpg but no expert_after_path")
        return ReproducedPair(
            sample_id=sample.sample_id,
            input_rgb=_load_rgb(sample.source_path),
            target_rgb=_load_rgb(after_path),
            cgt01=_load_mask(sample.c_gt.cgt_path),
            raw_mask01=_load_mask(sample.c_gt.raw_mask_path),
            needs_z_search=False,
            meta={"contract": "real_jpg"},
        )

    if sample.recipe.kind == RecipeKind.PARAM:
        if renderer is None:
            raise ValueError("PARAM reproduction requires a renderer")
        target_rgb = _render_one(renderer, sample.source_path, sample.recipe.params or {})
        return ReproducedPair(
            sample_id=sample.sample_id,
            input_rgb=_load_rgb(sample.source_path),
            target_rgb=target_rgb,
            cgt01=_load_mask(sample.c_gt.cgt_path),
            raw_mask01=_load_mask(sample.c_gt.raw_mask_path),
            needs_z_search=False,
            meta={"contract": "teacher_param"},
        )

    if sample.recipe.kind == RecipeKind.LUT:
        if parser is None or lut_applier is None:
            raise ValueError("LUT reproduction requires parser and lut_applier")
        lut_path = _resolve_lut_path(sample, lut_resolver)
        lut, dmin, dmax = parser.load_cube(lut_path)
        src = _load_rgb(sample.source_path)
        target = _apply_lut_rgb(src, lut, dmin, dmax, lut_applier)
        return ReproducedPair(
            sample_id=sample.sample_id,
            input_rgb=src,
            target_rgb=target,
            cgt01=_load_mask(sample.c_gt.cgt_path),
            raw_mask01=_load_mask(sample.c_gt.raw_mask_path),
            needs_z_search=False,
            meta={
                "contract": "lut",
                "lut_path": lut_path,
                "lut_sha256": sample.recipe.meta.get("lut_sha256"),
            },
        )

    raise ValueError(f"{sample.sample_id}: unsupported recipe kind {sample.recipe.kind}")


def region_composite_from_pair(pair: ReproducedPair) -> Optional[Any]:
    """Composite target into input using the raw pre-blur mask when available.

    D7 anti-double-count contract (doc §4): the Stage-0 supervision target is the
    BAKED cgt PNG (already includes g.soft_blur, i.e. C_GT = M_r * g soft-blurred);
    aspect_magnitude / soft_blur_sigma_px / magnitude_tau are METADATA for analysis
    only -- the loss MUST NOT re-apply aspect_magnitude (would double-count g).
    """
    if pair.input_rgb is None or pair.target_rgb is None or pair.raw_mask01 is None:
        return None
    from dataset_build.render import region_composite

    return region_composite(pair.target_rgb, pair.input_rgb, pair.raw_mask01)


def inverse_search_z_star(
    input_rgb: Any,
    target_rgb: Any,
    warm_start: Dict[str, Dict[str, float]],
    render_fn: Callable[[Any, Dict[str, Dict[str, float]]], Any],
    *,
    keys: Optional[Sequence[str]] = None,
    max_iters: int = 2,
    step_schedule: Sequence[float] = (20.0, 10.0, 5.0, 2.0, 1.0),
) -> Tuple[Dict[str, Dict[str, float]], float]:
    """Small coordinate search for the DATAGEN S1/S7 z* target.

    This is intentionally simple and deterministic. It exists as a reference
    contract check, not as the final high-throughput trainer implementation.
    """
    import copy

    active = list(keys) if keys is not None else sorted(warm_start)
    params = _normalize_param_values(warm_start)
    best = copy.deepcopy(params)
    best_psnr = _psnr(render_fn(input_rgb, best), target_rgb)

    for _ in range(max(0, int(max_iters))):
        improved = False
        for key in active:
            if key not in best:
                continue
            base = float(best[key]["value"])
            local_best_val = base
            local_best_psnr = best_psnr
            for step in step_schedule:
                for cand_val in (base - step, base + step):
                    cand = copy.deepcopy(best)
                    cand[key]["value"] = float(cand_val)
                    score = _psnr(render_fn(input_rgb, cand), target_rgb)
                    if score > local_best_psnr:
                        local_best_psnr = score
                        local_best_val = float(cand_val)
            if local_best_psnr > best_psnr:
                best[key]["value"] = local_best_val
                best_psnr = local_best_psnr
                improved = True
        if not improved:
            break
    return best, best_psnr


def _render_one(renderer: Any, image_path: str, params: Dict[str, Dict[str, float]]) -> Any:
    outs = renderer.render([image_path], [params])
    if not outs:
        raise RuntimeError(f"renderer returned no output for {image_path}")
    return outs[0]


def _load_rgb(path: Optional[str]) -> Optional[Any]:
    if not path:
        return None
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(str(path))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_mask(path: Optional[str]) -> Optional[Any]:
    if not path:
        return None
    if Path(path).name == "_global_ones.png":
        return None
    import numpy as np
    from PIL import Image

    im = Image.open(path).convert("L")
    return np.asarray(im, dtype="float32") / 255.0


def _warm_start_from_degrade(sample: Sample) -> Dict[str, Dict[str, float]]:
    spec = sample.recipe.degrade
    if spec is None:
        return {}
    return {
        k: {"value": float(v)}
        for k, v in (spec.op_params or {}).items()
        if not str(k).startswith("__")
    }


def _normalize_param_values(
    params: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key, value in (params or {}).items():
        if isinstance(value, dict):
            out[key] = {"value": float(value.get("value", 0.0))}
        else:
            out[key] = {"value": float(value)}
    return out


def _psnr(pred_rgb: Any, target_rgb: Any) -> float:
    import math
    import numpy as np

    pred = np.asarray(pred_rgb, dtype="float32")
    target = np.asarray(target_rgb, dtype="float32")
    if pred.shape != target.shape:
        raise ValueError(f"PSNR shape mismatch: {pred.shape} vs {target.shape}")
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def _resolve_lut_path(
    sample: Sample,
    lut_resolver: Optional[Callable[[Sample], Optional[str]]],
) -> str:
    if lut_resolver is not None:
        resolved = lut_resolver(sample)
        if resolved:
            return resolved
    path = sample.recipe.meta.get("path")
    if path:
        return str(path)
    raise ValueError(
        f"{sample.sample_id}: LUT path missing; pass lut_resolver or store recipe.meta['path']"
    )


def _apply_lut_rgb(src_rgb: Any, lut: Any, dmin: Tuple[float, float, float],
                   dmax: Tuple[float, float, float], lut_applier: Any) -> Any:
    import numpy as np
    import torch

    x = (
        torch.from_numpy(src_rgb.astype("float32") / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    y = lut_applier.apply_lut(x, lut, dmin, dmax)
    out = y.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return (np.clip(out, 0.0, 1.0) * 255.0).round().astype("uint8")


def _main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Inspect DATAGEN v2 pair reproduction contract.")
    ap.add_argument("jsonl", help="Path to a shard JSONL file")
    ap.add_argument("--index", type=int, default=0, help="Line index to inspect")
    args = ap.parse_args()

    lines = Path(args.jsonl).read_text(encoding="utf-8").splitlines()
    sample = jsonl_to_sample(lines[args.index])
    print(
        {
            "sample_id": sample.sample_id,
            "kind": sample.recipe.kind.value,
            "provenance": sample.recipe.provenance.value,
            "after_source": sample.after_source.value,
            "needs_z_search": sample.recipe.provenance == Provenance.DEGRADE,
            "z_search_adapter": "ArrayRenderAdapter available with z_search_scratch_dir",
            "er_recon_psnr_min": None,
            "mask_source": sample.c_gt.mask_source.value if sample.c_gt else MaskSource.GLOBAL.value,
            "raw_mask_path": sample.c_gt.raw_mask_path if sample.c_gt else None,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
