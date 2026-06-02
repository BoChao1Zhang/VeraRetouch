"""
dataset_build/stage0_probe.py
=============================
Stage-0 probe for DATAGEN v2 S1/S7 degrade samples.

The build stores S1/S7 rows as recipe-only records:

    kind=PARAM, provenance=DEGRADE, params=neg_p

Stage-0 must materialize:

    I_in = render(source, neg_p)
    I_tar = source
    z* = argmin_alpha PSNR(render(I_in, alpha), I_tar)

and drop rows below the configured er_recon_psnr floor. This module runs that
contract on committed shard JSONL rows and emits a JSONL report. It stays
import-light: model weights are only touched by the CLI when not using
``--dry-run``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, TextIO, Union

from dataset_build.contracts import Provenance
from dataset_build.pack import jsonl_to_sample
from dataset_build.reproduce import reproduce_pair


JsonlPath = Union[str, Path]


def run_stage0_probe(
    jsonl_paths: Union[JsonlPath, Sequence[JsonlPath]],
    *,
    renderer: Any,
    report_path: Optional[JsonlPath] = None,
    report_stream: Optional[TextIO] = None,
    scratch_dir: Optional[str] = None,
    limit: Optional[int] = None,
    z_search_keys: Optional[Sequence[str]] = None,
    z_search_max_iters: int = 2,
    er_recon_psnr_min: float = 25.0,
    max_stage0_pixels: Optional[int] = None,
    collect: bool = False,
) -> Dict[str, Any]:
    """Run the Stage-0 degrade gate over one or more shard JSONL files.

    ``limit`` counts degrade rows, not all rows in the shard. Non-degrade rows
    are parsed and skipped so mixed smoke shards can be probed directly.
    """
    if renderer is None:
        raise ValueError("run_stage0_probe requires a renderer")
    if report_path is not None and report_stream is not None:
        raise ValueError("pass only one of report_path or report_stream")

    paths = _coerce_paths(jsonl_paths)
    max_degrade = None if limit is None else max(0, int(limit))
    records: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "files": [str(p) for p in paths],
        "lines": 0,
        "skipped_non_degrade": 0,
        "degrade": 0,
        "accepted": 0,
        "dropped": 0,
        "errors": 0,
        "er_recon_psnr_min": float(er_recon_psnr_min),
        "z_search_max_iters": int(z_search_max_iters),
        "z_search_keys": list(z_search_keys) if z_search_keys is not None else None,
        "max_stage0_pixels": int(max_stage0_pixels) if max_stage0_pixels else None,
        "scratch_dir": str(scratch_dir) if scratch_dir else None,
    }

    close_stream = False
    stream = report_stream
    if report_path is not None:
        out = Path(report_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        stream = out.open("w", encoding="utf-8")
        close_stream = True

    try:
        for path in paths:
            stop = _probe_one_path(
                path,
                renderer=renderer,
                summary=summary,
                out_stream=stream,
                records=records if collect else None,
                scratch_dir=scratch_dir,
                max_degrade=max_degrade,
                z_search_keys=z_search_keys,
                z_search_max_iters=z_search_max_iters,
                er_recon_psnr_min=float(er_recon_psnr_min),
                max_stage0_pixels=max_stage0_pixels,
            )
            if stop:
                break
    finally:
        if close_stream and stream is not None:
            stream.close()

    if collect:
        summary["records"] = records
    return summary


class DryRunParamRenderer:
    """Tiny CPU path renderer for ``--dry-run`` Stage-0 smoke checks.

    It reads the input image and applies a simple additive shift equal to the
    sum of candidate parameter values. This is not a VeraRetouch approximation;
    it only exercises shard parsing, z-search control flow, and reporting.
    """

    def render(self, image_paths, param_dicts, **kwargs):
        import cv2
        import numpy as np

        outs = []
        for path, params in zip(image_paths, param_dicts):
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(str(path))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype("float32")
            shift = 0.0
            for value in (params or {}).values():
                if isinstance(value, dict):
                    shift += float(value.get("value", 0.0))
                else:
                    shift += float(value)
            outs.append(np.clip(rgb + shift, 0, 255).round().astype("uint8"))
        return outs


def build_renderer_from_config(config: Dict[str, Any], *, dry_run: bool = False) -> Any:
    """Construct the renderer for the CLI. Real weights load only here."""
    if dry_run:
        return DryRunParamRenderer()

    vr = (config.get("models", {}) or {}).get("veraretouch", {}) or {}
    from dataset_build.render import VeraRetouchRenderer  # type: ignore

    kw: Dict[str, Any] = {
        "dtype": vr.get("dtype", "bfloat16"),
        "max_new_tokens": int(vr.get("max_new_tokens", 256)),
        "greedy": bool(vr.get("greedy", True)),
        "num_workers": int(vr.get("num_workers", 0)),
    }
    if vr.get("model_path"):
        kw["model_path"] = vr["model_path"]
    if vr.get("config_add_path"):
        kw["config_add_path"] = vr["config_add_path"]
    return VeraRetouchRenderer(**kw)


def load_config(path: JsonlPath) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to read config.yaml") from exc

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _probe_one_path(
    path: Path,
    *,
    renderer: Any,
    summary: Dict[str, Any],
    out_stream: Optional[TextIO],
    records: Optional[List[Dict[str, Any]]],
    scratch_dir: Optional[str],
    max_degrade: Optional[int],
    z_search_keys: Optional[Sequence[str]],
    z_search_max_iters: int,
    er_recon_psnr_min: float,
    max_stage0_pixels: Optional[int],
) -> bool:
    with path.open("r", encoding="utf-8") as f:
        for line_index, line in enumerate(f):
            if max_degrade is not None and summary["degrade"] >= max_degrade:
                return True
            summary["lines"] += 1
            line = line.strip()
            if not line:
                continue

            try:
                sample = jsonl_to_sample(line)
            except Exception as exc:
                summary["errors"] += 1
                record = {
                    "file": str(path),
                    "line_index": line_index,
                    "sample_id": None,
                    "accepted": False,
                    "dropped": True,
                    "decision": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                _emit(record, out_stream, records)
                continue

            if sample.recipe.provenance != Provenance.DEGRADE:
                summary["skipped_non_degrade"] += 1
                continue

            summary["degrade"] += 1
            record = _probe_sample(
                sample,
                renderer=renderer,
                file_path=path,
                line_index=line_index,
                scratch_dir=scratch_dir,
                z_search_keys=z_search_keys,
                z_search_max_iters=z_search_max_iters,
                er_recon_psnr_min=er_recon_psnr_min,
                max_stage0_pixels=max_stage0_pixels,
            )
            if record.get("error"):
                summary["errors"] += 1
            elif record["accepted"]:
                summary["accepted"] += 1
            else:
                summary["dropped"] += 1
            _emit(record, out_stream, records)
    return False


def _probe_sample(
    sample,
    *,
    renderer: Any,
    file_path: Path,
    line_index: int,
    scratch_dir: Optional[str],
    z_search_keys: Optional[Sequence[str]],
    z_search_max_iters: int,
    er_recon_psnr_min: float,
    max_stage0_pixels: Optional[int],
) -> Dict[str, Any]:
    native_h, native_w = _sample_native_hw(sample)
    native_pixels = (native_h * native_w) if native_h and native_w else None
    composite_ready = (not sample.region_local) or bool(
        getattr(getattr(sample, "c_gt", None), "raw_mask_path", None)
    )
    base: Dict[str, Any] = {
        "file": str(file_path),
        "line_index": line_index,
        "sample_id": sample.sample_id,
        "stream": sample.stream.value,
        "schema_version": sample.schema_version,
        "kind": sample.recipe.kind.value,
        "provenance": sample.recipe.provenance.value,
        "source_path": sample.source_path,
        "er_recon_psnr_min": float(er_recon_psnr_min),
        "z_search_max_iters": int(z_search_max_iters),
        "native_size": [native_h, native_w] if native_h and native_w else None,
        "native_pixels": native_pixels,
        "max_stage0_pixels": int(max_stage0_pixels) if max_stage0_pixels else None,
        "raw_mask_path": sample.c_gt.raw_mask_path if sample.c_gt else None,
        "region_composite_ready": composite_ready,
        "scratch_dir": str(scratch_dir) if scratch_dir else None,
    }
    if sample.region_local and not composite_ready:
        base.update(
            {
                "accepted": False,
                "dropped": True,
                "decision": "error",
                "drop_reason": "missing_raw_mask",
                "error": "S1 region-local degrade sample lacks raw pre-blur mask for composite replay",
            }
        )
        return base
    if max_stage0_pixels and native_pixels and native_pixels > int(max_stage0_pixels):
        base.update(
            {
                "accepted": False,
                "dropped": True,
                "decision": "dropped",
                "drop_reason": f"stage0_pixels>{int(max_stage0_pixels)}({native_pixels})",
            }
        )
        return base
    try:
        pair = reproduce_pair(
            sample,
            renderer=renderer,
            run_z_search=True,
            z_search_scratch_dir=scratch_dir,
            z_search_keys=z_search_keys,
            z_search_max_iters=z_search_max_iters,
            er_recon_psnr_min=er_recon_psnr_min,
        )
        psnr = pair.er_recon_psnr
        accepted = bool(psnr is not None and psnr >= er_recon_psnr_min)
        active_keys = (
            list(z_search_keys)
            if z_search_keys is not None
            else sorted((pair.meta.get("warm_start") or {}).keys())
        )
        base.update(
            {
                "accepted": accepted,
                "dropped": not accepted,
                "decision": "accepted" if accepted else "dropped",
                "drop_reason": None
                if accepted
                else f"er_recon_psnr<{er_recon_psnr_min:g}",
                "er_recon_psnr": _json_float(psnr),
                "z_star": pair.z_star,
                "warm_start": pair.meta.get("warm_start"),
                "z_search_keys": active_keys,
                "composite_contract": "raw_mask"
                if sample.region_local
                else "global",
            }
        )
    except Exception as exc:
        base.update(
            {
                "accepted": False,
                "dropped": True,
                "decision": "error",
                "drop_reason": "stage0_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    return base


def _emit(
    record: Dict[str, Any],
    stream: Optional[TextIO],
    records: Optional[List[Dict[str, Any]]],
) -> None:
    if records is not None:
        records.append(record)
    if stream is not None:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False))
        stream.write("\n")
        stream.flush()


def _json_float(value: Optional[float]) -> Optional[Union[float, str]]:
    if value is None:
        return None
    value = float(value)
    if math.isfinite(value):
        return value
    return "inf" if value > 0 else "-inf"


def _coerce_paths(paths: Union[JsonlPath, Sequence[JsonlPath]]) -> List[Path]:
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(p) for p in paths]


def _parse_keys(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    keys = [p.strip() for p in value.split(",") if p.strip()]
    return keys or None


def _sample_native_hw(sample) -> tuple[Optional[int], Optional[int]]:
    ns = getattr(sample, "native_size", None)
    if ns and len(ns) >= 2:
        try:
            return int(ns[0]), int(ns[1])
        except (TypeError, ValueError):
            pass
    try:
        from PIL import Image

        with Image.open(sample.source_path) as im:
            w, h = im.size
        return int(h), int(w)
    except Exception:
        return None, None


def _max_stage0_pixels_from_config(config: Dict[str, Any]) -> Optional[int]:
    stage0 = config.get("stage0", {}) or {}
    value = stage0.get("max_pixels")
    if value is None:
        value = stage0.get("max_stage0_pixels")
    if value is None:
        value = (config.get("sources", {}) or {}).get("max_source_pixels")
    if value is None:
        value = (config.get("vllm", {}) or {}).get("max_image_pixels")
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _choose_scratch_dir(requested: Optional[str], *, explicit: bool = False) -> str:
    """Return a scratch dir that can be created.

    Explicit ``--scratch-dir`` remains strict. Config defaults are allowed to
    fall back to /tmp so Stage-0 dry-run/reporting does not fail solely because a
    worker cannot write the configured dataset scratch root.
    """
    candidates = [requested] if explicit else [requested, "/tmp/datagen_stage0_probe"]
    last_error: Optional[BaseException] = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            root = Path(candidate)
            root.mkdir(parents=True, exist_ok=True)
            probe = root / ".stage0_probe_write_test"
            with open(probe, "w", encoding="utf-8") as f:
                f.write("ok")
            try:
                os.unlink(probe)
            except OSError:
                pass
            return str(root)
        except OSError as exc:
            last_error = exc
    if last_error is not None:
        raise OSError(f"no writable Stage-0 scratch dir from {candidates}: {last_error}")
    fallback = Path("/tmp/datagen_stage0_probe")
    fallback.mkdir(parents=True, exist_ok=True)
    return str(fallback)


def _summary_for_stderr(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in summary.items() if k != "records"}


def _main(argv: Optional[Sequence[str]] = None) -> int:
    default_config = str(Path(__file__).with_name("config.yaml"))
    ap = argparse.ArgumentParser(description="Run DATAGEN Stage-0 degrade z* probe.")
    ap.add_argument("jsonl", nargs="+", help="Shard JSONL file(s) to probe")
    ap.add_argument("--config", default=default_config, help="dataset_build config.yaml")
    ap.add_argument("--out", help="Write JSONL report to this path; default prints records to stdout")
    ap.add_argument("--limit", type=int, help="Max degrade rows to probe")
    ap.add_argument("--scratch-dir", help="Directory for transient z-search PNGs")
    ap.add_argument("--max-iters", type=int, default=2, help="Coordinate-search passes")
    ap.add_argument("--keys", help="Comma-separated param keys; default uses each row's warm-start keys")
    ap.add_argument("--max-stage0-pixels", type=int,
                    help="Drop rows whose native source pixels exceed this Stage-0 budget")
    ap.add_argument("--dry-run", action="store_true", help="Use CPU synthetic renderer, no weights")
    ap.add_argument("--strict", action="store_true", help="Exit nonzero if any row drops or errors")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    qa = cfg.get("qa", {}) or {}
    threshold = float(qa.get("er_recon_psnr_min", 25.0))
    max_stage0_pixels = (
        int(args.max_stage0_pixels)
        if args.max_stage0_pixels is not None
        else _max_stage0_pixels_from_config(cfg)
    )
    scratch_dir = _choose_scratch_dir(
        args.scratch_dir or cfg.get("scratch_dir") or "/tmp/datagen_stage0_probe",
        explicit=args.scratch_dir is not None,
    )
    renderer = build_renderer_from_config(cfg, dry_run=bool(args.dry_run))

    stream = None if args.out else sys.stdout
    summary = run_stage0_probe(
        args.jsonl,
        renderer=renderer,
        report_path=args.out,
        report_stream=stream,
        scratch_dir=str(scratch_dir),
        limit=args.limit,
        z_search_keys=_parse_keys(args.keys),
        z_search_max_iters=args.max_iters,
        er_recon_psnr_min=threshold,
        max_stage0_pixels=max_stage0_pixels,
    )
    print(json.dumps(_summary_for_stderr(summary), ensure_ascii=False, sort_keys=True), file=sys.stderr)
    if args.strict and (summary["dropped"] or summary["errors"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
