"""Export one inspectable source -> global -> local chain per scene."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from dataset_build.agent_loop.artifacts import ArtifactStore
from dataset_build.agent_loop.candidates import LutCatalog
from dataset_build.agent_loop.config import CatalogConfig
from dataset_build.agent_loop.models import LOCAL_DELTA_E_LADDERS, target_center
from dataset_build.agent_loop.render import (
    CanonicalCpuLutRenderer,
    CanonicalGpuRenderer,
    FullPresetRenderer,
    apply_local_strength,
    delta_e_map,
    load_alpha,
    load_rgb,
)


SCENE_ORDER = ("portrait", "food", "animal", "landscape")
SCENE_LABELS = {
    "portrait": "Portrait",
    "food": "Food",
    "animal": "Animal",
    "landscape": "Landscape",
}


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list)):
        return value
    return json.loads(str(value))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def _fit(image: Image.Image, size: tuple[int, int], background: str = "#151719") -> Image.Image:
    result = Image.new("RGB", size, background)
    copy = image.convert("RGB")
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    left = (size[0] - copy.width) // 2
    top = (size[1] - copy.height) // 2
    result.paste(copy, (left, top))
    return result


def _source_image(source: dict[str, Any], tree: dict[str, Any],
                  artifacts: ArtifactStore) -> Image.Image:
    try:
        from dataset_build.tools.archive_reader import open_image

        with open_image(source["source_path"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (FileNotFoundError, OSError, ValueError):
        with Image.open(artifacts.path_for(tree["source_artifact"])) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    scale = 1024 / min(image.size)
    size = tuple(max(1, round(value * scale)) for value in image.size)
    return image.resize(size, Image.Resampling.LANCZOS)


def _leaf_score(branch: dict[str, Any], leaf: dict[str, Any]) -> tuple[float, str]:
    global_metrics = branch["global_render"]["metrics"]
    local_metrics = leaf["render"]["metrics"]
    strength = str(branch["proposal"]["strength_bin"])
    strength_penalty = {"natural": 0.0, "medium": 0.1, "bold": 0.6}[strength]
    score = (
        strength_penalty
        + abs(float(global_metrics["delta_e"]) - target_center(strength)) * 0.15
        + abs(float(local_metrics["delta_e"]) - 4.0)
        + 100.0 * float(global_metrics.get("clip_fraction_new", 0.0))
        + 100.0 * float(local_metrics.get("clip_fraction_new", 0.0))
    )
    return score, str(leaf["branch_id"])


def _select_chain(
    tree: dict[str, Any], selection: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    committed = set(tree["committed_leaf_ids"])
    candidates = [
        (branch, leaf)
        for branch in tree["branches"]
        for leaf in branch.get("leaves", [])
        if leaf.get("branch_id") in committed
    ]
    if not candidates:
        raise ValueError(f"accepted tree has no committed leaves: {tree['source_id']}")
    if selection == "strongest":
        return max(candidates, key=lambda pair: (
            float(pair[0]["global_render"]["metrics"]["delta_e"]),
            float(pair[1]["render"]["metrics"]["delta_e"]),
            str(pair[1]["branch_id"]),
        ))
    return min(candidates, key=lambda pair: _leaf_score(*pair))


def _preview_renderer(audit: dict[str, Any]) -> tuple[FullPresetRenderer, LutCatalog]:
    catalog_row = audit["config"]["catalog"]
    catalog_config = CatalogConfig(
        annotations=Path(catalog_row["annotations"]),
        global_major_limit=int(catalog_row["global_major_limit"]),
        global_per_major_limit=int(catalog_row["global_per_major_limit"]),
        local_limit=int(catalog_row["local_limit"]),
    )
    databuild_config = Path(audit["config"]["databuild_config"])
    catalog = LutCatalog.load(catalog_config, databuild_config)
    backend = str((audit["config"].get("render") or {}).get("backend") or "gpu_lut")
    renderer = CanonicalCpuLutRenderer(databuild_config, catalog) \
        if backend == "cpu_lut" else CanonicalGpuRenderer(databuild_config, catalog)
    return renderer, catalog


def _new_clip_fraction(before: np.ndarray, after: np.ndarray) -> float:
    threshold = 1.0 / 255.0
    before_clip = (before <= threshold) | (before >= 1.0 - threshold)
    after_clip = (after <= threshold) | (after >= 1.0 - threshold)
    return float(np.logical_and(after_clip, ~before_clip).mean())


def _render_local_preview(
    *, renderer: FullPresetRenderer, catalog: LutCatalog, artifacts: ArtifactStore,
    global_artifact: dict[str, Any], mask: dict[str, Any], preset_id: str,
    output_path: Path, committed_render: dict[str, Any], mode: str,
) -> dict[str, Any]:
    input_path = artifacts.path_for(global_artifact)
    before = load_rgb(input_path)
    full = renderer.render_full(input_path, catalog.get(preset_id))
    if full.shape != before.shape:
        image = Image.fromarray(np.clip(full * 255 + 0.5, 0, 255).astype(np.uint8), "RGB")
        image = image.resize(
            (before.shape[1], before.shape[0]), Image.Resampling.LANCZOS,
        )
        full = np.asarray(image, dtype=np.float32) / 255.0
    alpha = load_alpha(
        artifacts.path_for(mask["alpha_artifact"]), (before.shape[1], before.shape[0]),
    )
    candidates = []
    for strength in np.linspace(1.0 / 17.0, 1.0, 17):
        candidate = apply_local_strength(before, full, alpha, float(strength))
        delta = delta_e_map(before, candidate)
        local_delta_e = float((delta * alpha).sum() / max(float(alpha.sum()), 1e-8))
        clip = _new_clip_fraction(before, candidate)
        candidates.append((float(strength), candidate, local_delta_e, clip))
    if mode == "full":
        strength, after, local_delta_e, clip = candidates[-1]
    else:
        # B7: this exporter is intent-free, so it reads the `high` ladder, which
        # is the band the majority of the active intents still use.
        target_low, target_high, _inclusive = LOCAL_DELTA_E_LADDERS["high"]["strong"]
        target_center_value = (target_low + target_high) / 2.0
        in_target = [
            row for row in candidates
            if target_low <= row[2] <= target_high and row[3] <= 0.01
        ]
        eligible = [row for row in candidates if row[3] <= 0.01]
        if in_target:
            strength, after, local_delta_e, clip = min(
                in_target, key=lambda row: (abs(row[2] - target_center_value), row[0]),
            )
        elif eligible:
            strength, after, local_delta_e, clip = max(
                eligible, key=lambda row: (row[2], row[0]),
            )
        else:
            strength, after, local_delta_e, clip = min(
                candidates, key=lambda row: (row[3], -row[2]),
            )
    image = Image.fromarray(np.clip(after * 255 + 0.5, 0, 255).astype(np.uint8), "RGB")
    image.save(output_path, "JPEG", quality=95, subsampling=0)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return {
        "render_hash": digest,
        "artifact": {
            "path": output_path.name,
            "sha256": digest,
            "size": output_path.stat().st_size,
            "media_type": "image/jpeg",
        },
        "metrics": {
            "delta_e": local_delta_e,
            "clip_fraction_new": clip,
            "finite": float(np.isfinite(after).all()),
        },
        "parameters": {
            "preset_id": preset_id,
            "global_strength": None,
            "local_strength": strength,
            "mask_id": mask["mask_id"],
            "strength_bin": mode,
            "terra_strength_bin": committed_render["parameters"]["strength_bin"],
        },
        "cache_hit": False,
        "preview_override": {
            "mode": f"offline_{mode}_strength",
            "target_delta_e": [target_low, target_high] if mode == "strong" else None,
            "committed_render_hash": committed_render["render_hash"],
            "committed_local_strength": committed_render["parameters"]["local_strength"],
            "committed_delta_e": committed_render["metrics"]["delta_e"],
        },
    }


def _timing_report(
    audit: dict[str, Any], accepted: dict[str, dict[str, Any]], campaign: str,
) -> dict[str, Any]:
    tables = audit["tables"]
    requests = {row["request_hash"]: row for row in tables["api_request"]}
    rows = []
    for source_id, run in sorted(accepted.items()):
        contexts = [
            row for row in tables["api_request_context"]
            if row["campaign_id"] == campaign
            and row["source_sha256"] == run["source_sha256"]
            and float(run["started_at"]) <= float(row["created_at"])
            <= float(run["updated_at"])
        ]
        request_rows = [
            requests[row["request_hash"]] for row in contexts
            if row["request_hash"] in requests
            and requests[row["request_hash"]]["status"] == "resolved"
        ]
        counts = _json_value(run["counts_json"])
        wall_seconds = float(run["updated_at"]) - float(run["started_at"])
        committed = int(counts["committed_leaves"])
        rows.append({
            "source_id": source_id,
            "wall_seconds": wall_seconds,
            "committed_samples": committed,
            "amortized_seconds_per_sample": wall_seconds / committed,
            "terra_requests": len(request_rows),
            "terra_request_seconds_sum": sum(
                float(row["updated_at"]) - float(row["created_at"])
                for row in request_rows
            ),
        })
    total_wall = sum(row["wall_seconds"] for row in rows)
    total_samples = sum(row["committed_samples"] for row in rows)
    total_requests = sum(row["terra_requests"] for row in rows)
    total_request_seconds = sum(row["terra_request_seconds_sum"] for row in rows)
    mean_source_seconds = total_wall / len(rows)
    samples_per_source = total_samples / len(rows)
    ideal_sources_per_hour_16 = 16.0 * 3600.0 / mean_source_seconds
    request_seconds = total_request_seconds / total_requests
    requests_per_source = total_requests / len(rows)
    limits = audit["config"]["limits"]
    terra_limit = int(limits["terra_concurrency_target"])
    renderer_limit = int(limits["renderer_concurrency"])
    non_terra_seconds_per_source = max(
        (total_wall - total_request_seconds) / len(rows), 1e-8,
    )
    current_terra_bound = (
        terra_limit * 3600.0 * samples_per_source
        / (request_seconds * requests_per_source)
    )
    terra_16_bound = (
        16.0 * 3600.0 * samples_per_source / (request_seconds * requests_per_source)
    )
    current_renderer_bound = (
        renderer_limit * 3600.0 * samples_per_source / non_terra_seconds_per_source
    )
    return {
        "schema": "local-retouch-timing-v1",
        "scope": "accepted source runs; offline source diagnosis excluded",
        "campaign_id": campaign,
        "sources": rows,
        "aggregate": {
            "accepted_sources": len(rows),
            "committed_samples": total_samples,
            "mean_seconds_per_source": mean_source_seconds,
            "amortized_seconds_per_committed_sample": total_wall / total_samples,
            "mean_terra_request_seconds": request_seconds,
            "terra_requests_per_source": requests_per_source,
            "mean_non_terra_seconds_per_source": non_terra_seconds_per_source,
        },
        "estimate_16_source_concurrency": {
            "ideal_sources_per_hour": ideal_sources_per_hour_16,
            "ideal_committed_samples_per_hour": ideal_sources_per_hour_16
            * samples_per_source,
            "current_terra_limit": terra_limit,
            "current_renderer_limit": renderer_limit,
            "current_terra_work_bound_samples_per_hour": current_terra_bound,
            "current_renderer_work_bound_samples_per_hour": current_renderer_bound,
            "current_caps_effective_work_bound_samples_per_hour": min(
                current_terra_bound, current_renderer_bound,
            ),
            "terra_16_renderer_current_effective_work_bound_samples_per_hour": min(
                terra_16_bound, current_renderer_bound,
            ),
            "assumptions": [
                "The ideal estimate scales accepted-run wall time linearly.",
                "Work bounds separate measured Terra time from all other measured wall time.",
                "Provider throttling, retries, and renderer contention are excluded.",
            ],
        },
    }


def _usage(audit: dict[str, Any], campaign: str, source_sha256: str) -> dict[str, Any]:
    contexts = {
        row["request_hash"]: row
        for row in audit["tables"]["api_request_context"]
        if row["campaign_id"] == campaign and row["source_sha256"] == source_sha256
    }
    requests = {
        row["request_hash"]: row for row in audit["tables"]["api_request"]
        if row["request_hash"] in contexts and row["status"] == "resolved"
    }
    stages: dict[str, dict[str, int]] = defaultdict(
        lambda: {"requests": 0, "input_tokens": 0, "output_tokens": 0,
                 "cached_tokens": 0}
    )
    for request_hash, request in requests.items():
        usage = _json_value(request.get("usage_json")) or {}
        stage = str(contexts[request_hash]["stage"])
        stages[stage]["requests"] += 1
        for key in ("input_tokens", "output_tokens", "cached_tokens"):
            stages[stage][key] += int(usage.get(key, 0))
    total = {
        key: sum(row[key] for row in stages.values())
        for key in ("requests", "input_tokens", "output_tokens", "cached_tokens")
    }
    return {"scope": "resolved requests for this source and campaign",
            "by_stage": dict(sorted(stages.items())), "total": total}


def _comparison(scene: str, directory: Path, chain: dict[str, Any]) -> Image.Image:
    tile = (520, 390)
    header, footer = 48, 58
    canvas = Image.new("RGB", (tile[0] * 3, header + tile[1] + footer), "#f3f4f4")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(25, bold=True)
    info_font = _font(17)
    names = (("Source", "source.jpg"), ("Global", "global.jpg"), ("Local", "local.jpg"))
    for index, (label, filename) in enumerate(names):
        with Image.open(directory / filename) as image:
            canvas.paste(_fit(image, tile), (index * tile[0], header))
        draw.text((index * tile[0] + 14, 10), label, fill="#17191b", font=title_font)
    global_de = chain["global"]["render"]["metrics"]["delta_e"]
    local_de = chain["local"]["render"]["metrics"]["delta_e"]
    summary = (
        f"{SCENE_LABELS[scene]}  |  global ΔE {global_de:.2f}  |  "
        f"local ΔE {local_de:.2f}  |  validator disabled"
    )
    draw.text((14, header + tile[1] + 16), summary, fill="#34383c", font=info_font)
    return canvas


def export(args: argparse.Namespace) -> None:
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    sources = {row["source_id"]: row for row in _read_jsonl(args.manifest)}
    landing = audit["config"].get("artifact_landing") or {}
    artifacts = ArtifactStore(args.artifact_root, catalog_db=landing.get("catalog_db"))
    accepted: dict[str, dict[str, Any]] = {}
    for row in audit["tables"]["agent_source_run"]:
        if row["campaign_id"] != args.campaign or row["status"] != "accepted":
            continue
        source_id = str(row["source_id"])
        if source_id not in sources:
            continue
        previous = accepted.get(source_id)
        if previous is None or float(row["updated_at"]) > float(previous["updated_at"]):
            accepted[source_id] = row
    missing = sorted(set(sources) - set(accepted))
    if missing:
        raise ValueError(f"sources lack accepted manifests: {missing}")

    args.out.mkdir(parents=True, exist_ok=True)
    preview_renderer = None
    preview_catalog = None
    if args.local_render != "committed":
        preview_renderer, preview_catalog = _preview_renderer(audit)
    comparisons: list[tuple[str, Image.Image]] = []
    summary_rows = []
    for scene in SCENE_ORDER:
        source = next(row for row in sources.values() if row["scene"] == scene)
        run = accepted[source["source_id"]]
        manifest_ref = _json_value(run["manifest_json"])
        tree = json.loads(artifacts.read_bytes(manifest_ref))
        branch, leaf = _select_chain(tree, args.selection)
        mask = next(row for row in tree["mask_bank"]
                    if row["mask_id"] == leaf["proposal"]["mask_id"])
        directory = args.out / scene
        directory.mkdir(parents=True, exist_ok=True)

        _source_image(source, tree, artifacts).save(
            directory / "source.jpg", "JPEG", quality=95, subsampling=0
        )
        shutil.copyfile(artifacts.path_for(branch["global_render"]["artifact"]),
                        directory / "global.jpg")
        if args.local_render != "committed":
            assert preview_renderer is not None and preview_catalog is not None
            local_render = _render_local_preview(
                renderer=preview_renderer, catalog=preview_catalog, artifacts=artifacts,
                global_artifact=branch["global_render"]["artifact"], mask=mask,
                preset_id=leaf["proposal"]["preset_id"],
                output_path=directory / "local.jpg", committed_render=leaf["render"],
                mode=args.local_render,
            )
        else:
            local_render = leaf["render"]
            shutil.copyfile(artifacts.path_for(local_render["artifact"]),
                            directory / "local.jpg")
        applied_alpha = local_render.get("applied_alpha_artifact") \
            or local_render.get("parameters", {}).get("applied_alpha_artifact")
        shutil.copyfile(artifacts.path_for(applied_alpha or mask["alpha_artifact"]),
                        directory / "local_mask.png")
        shutil.copyfile(source["subject_path"], directory / "subject.png")
        shutil.copyfile(source["source_annotation_path"],
                        directory / "source_annotation.json")
        _write_json(directory / "edit_tree.json", tree)
        shortlist = json.loads(artifacts.read_bytes(tree["global_shortlist_artifact"]))
        _write_json(directory / "global_shortlist.json", shortlist)

        chain = {
            "schema": "local-retouch-scene-preview-v1",
            "scene": scene,
            "source_id": tree["source_id"],
            "source_sha256": tree["source_sha256"],
            "run": {
                "campaign_id": tree["campaign_id"],
                "thread_id": tree["thread_id"],
                "prompt_revision": tree["prompt_revision"],
                "terminal_status": "accepted",
                "validator_enabled": tree["config"]["validator"]["enabled"],
                "terra_model": tree["config"]["terra"]["model"],
                "reasoning_effort": tree["config"]["terra"]["reasoning_effort"],
                "api_image_longest_edge": 512,
                "preview_mode": {
                    "selection": args.selection,
                    "local_render": args.local_render,
                },
            },
            "files": {
                "source": "source.jpg", "subject_mask": "subject.png",
                "source_annotation": "source_annotation.json",
                "global": "global.jpg", "local": "local.jpg",
                "local_mask": "local_mask.png", "edit_tree": "edit_tree.json",
                "global_shortlist": "global_shortlist.json",
            },
            "selection": {
                "method": (
                    "highest committed global delta-e"
                    if args.selection == "strongest"
                    else "lowest target deviation, clip, and strength penalty"
                ),
                "committed_leaf_id": leaf["branch_id"],
                "available_committed_leaf_ids": tree["committed_leaf_ids"],
            },
            "diagnosis": tree["diagnosis"],
            "global": {
                "branch_id": branch["branch_id"], "proposal": branch["proposal"],
                "render": branch["global_render"],
            },
            "local": {
                "branch_id": leaf["branch_id"], "proposal": leaf["proposal"],
                "mask_family": leaf["mask_family"], "render": local_render,
                "committed_render": leaf["render"],
                "validation": leaf["validation"],
            },
            "usage": _usage(audit, args.campaign, tree["source_sha256"]),
        }
        _write_json(directory / "chain.json", chain)
        comparison = _comparison(scene, directory, chain)
        comparison.save(directory / "comparison.jpg", "JPEG", quality=92, subsampling=0)
        comparisons.append((scene, comparison))
        summary_rows.append({
            "scene": scene, "source_id": tree["source_id"],
            "committed_leaf_id": leaf["branch_id"],
            "global_delta_e": branch["global_render"]["metrics"]["delta_e"],
            "local_delta_e": local_render["metrics"]["delta_e"],
            "global_preset_id": branch["proposal"]["preset_id"],
            "local_preset_id": leaf["proposal"]["preset_id"],
            "global_strength_bin": branch["proposal"]["strength_bin"],
            "local_strength": local_render["parameters"]["local_strength"],
        })

    width = max(image.width for _, image in comparisons)
    label_height = 52
    height = sum(image.height + label_height for _, image in comparisons)
    sheet = Image.new("RGB", (width, height), "#e5e7e8")
    draw = ImageDraw.Draw(sheet)
    y = 0
    for scene, image in comparisons:
        draw.text((16, y + 10), SCENE_LABELS[scene], fill="#141618",
                  font=_font(27, bold=True))
        y += label_height
        sheet.paste(image, (0, y))
        y += image.height
    sheet.save(args.out / "contact_sheet.jpg", "JPEG", quality=92, subsampling=0)
    _write_json(args.out / "summary.json", {
        "schema": "local-retouch-scene-samples-v1",
        "campaign_id": args.campaign,
        "preview_mode": {
            "selection": args.selection,
            "local_render": args.local_render,
        },
        "scenes": summary_rows,
    })
    _write_json(args.out / "timing.json", _timing_report(audit, accepted, args.campaign))
    mode_title = "Strong offline preview" if args.local_render != "committed" else \
        "Low-token Terra scene samples"
    mode_note = (
        "The strongest committed global branch is paired with the same Terra-selected local "
        "LUT and mask rendered offline at a stronger target. This is an amplitude preview, not "
        "a replacement committed render."
        if args.local_render != "committed" else
        "One accepted `source -> global -> local` LUT chain per scene."
    )
    readme = [
        f"# {mode_title}", "", mode_note, "",
        "Open `contact_sheet.jpg` for the four-scene overview. Each scene directory contains ",
        "`comparison.jpg`, the three images, masks, offline source annotation, LUT shortlist, ",
        "full accepted edit tree, and compact `chain.json`.", "", "| Scene | Source | Global ΔE | Local ΔE |", "|---|---|---:|---:|",
    ]
    readme.extend(
        f"| {row['scene']} | `{row['source_id']}` | {row['global_delta_e']:.2f} | {row['local_delta_e']:.2f} |"
        for row in summary_rows
    )
    (args.out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--audit", type=Path, required=True)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--artifact-root", type=Path, required=True)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--campaign", required=True)
    result.add_argument(
        "--selection", choices=("balanced", "strongest"), default="balanced",
    )
    result.add_argument(
        "--local-render", choices=("committed", "strong", "full"), default="committed",
    )
    return result


if __name__ == "__main__":
    export(parser().parse_args())
