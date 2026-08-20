"""Offline LUT effect clustering (EPR/A1, DECISIONS_agent_loop_annotation_optA_20260819 §2.5).

Builds a 39-dim effect feature per annotated LUT, normalises magnitude by ``de_med``,
runs average-linkage hierarchical clustering inside every ``style_major`` with a single
global distance threshold, and emits one JSONL artifact per threshold tier plus a
manifest. Optionally renders a static visual pilot for threshold acceptance.

Usage:
    python -m dataset_build.tools.cluster_lut_effects \
        --config configs/agent_loop.terra-smoke.toml \
        --out-dir /home/bc/data/scratch/lut_clusters \
        --pilot-dir docs/assets/lut_cluster_pilot_20260819 \
        --probe-jsonl configs/agent_loop.smoke5.jsonl
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import html
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_build.agent_loop.candidates import LutCatalog  # noqa: E402
from dataset_build.agent_loop.config import CatalogConfig  # noqa: E402

FEATURE_SPEC = "lut-effect-39d-v1"
RAMP_POINTS = 5
BAND_COUNT = 8
FEATURE_DIM = RAMP_POINTS * 3 + BAND_COUNT * 3
DEFAULT_TARGETS = (0.70, 0.50, 0.30)
DEFAULT_TAGS = ("t70", "t50", "t30")


# --------------------------------------------------------------------------- features


class FeatureError(ValueError):
    pass


def band_order(records: Sequence[Any]) -> tuple[str, ...]:
    names: set[str] = set()
    for row in records:
        bands = (row.hsl_features or {}).get("bands") or {}
        names.update(str(key) for key in bands)
    ordered = tuple(sorted(names))
    if len(ordered) != BAND_COUNT:
        raise FeatureError(f"expected {BAND_COUNT} hue bands, found {len(ordered)}: {ordered}")
    return ordered


def feature_vector(row: Any, bands: Sequence[str]) -> np.ndarray:
    hsl = row.hsl_features or {}
    ramp = list(hsl.get("neutral_ramp") or [])
    if len(ramp) != RAMP_POINTS:
        raise FeatureError(f"{row.preset_id}: neutral_ramp has {len(ramp)} points")
    ramp = sorted(ramp, key=lambda point: float(point["in"]))
    values: list[float] = []
    for point in ramp:
        values.extend((
            float(point["L_out"]) - float(point["L_in"]),
            float(point["a_out"]),
            float(point["b_out"]),
        ))
    table = hsl.get("bands") or {}
    for name in bands:
        band = table.get(name)
        if band is None:
            raise FeatureError(f"{row.preset_id}: missing hue band {name}")
        values.extend((
            float(band["d_hue_deg"]),
            float(band["d_sat_pct"]),
            float(band["d_lum_pct"]),
        ))
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape[0] != FEATURE_DIM:
        raise FeatureError(f"{row.preset_id}: feature dim {vector.shape[0]} != {FEATURE_DIM}")
    if not np.all(np.isfinite(vector)):
        raise FeatureError(f"{row.preset_id}: non-finite feature value")
    return vector


def build_features(records: Sequence[Any]) -> tuple[np.ndarray, tuple[str, ...], list[str]]:
    """Return (normalised matrix, band order, preset ids without de_med normalisation)."""
    bands = band_order(records)
    matrix = np.zeros((len(records), FEATURE_DIM), dtype=np.float64)
    unnormalised: list[str] = []
    for index, row in enumerate(records):
        vector = feature_vector(row, bands)
        de_med = float(row.de_med) if row.de_med is not None else 0.0
        if de_med > 0.0:
            vector = vector / de_med
        else:
            unnormalised.append(row.preset_id)
        matrix[index] = vector
    return matrix, bands, unnormalised


# --------------------------------------------------------------------------- clustering


def _linkage(matrix: np.ndarray) -> np.ndarray:
    from scipy.cluster.hierarchy import linkage

    return linkage(matrix, method="average", metric="euclidean")


def _flat_clusters(link: np.ndarray, threshold: float, size: int) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster

    if size == 1:
        return np.zeros(1, dtype=np.int64)
    return fcluster(link, t=threshold, criterion="distance").astype(np.int64)


class MajorGroup:
    __slots__ = ("major", "indices", "preset_ids", "matrix", "link")

    def __init__(self, major: str, indices: Sequence[int], preset_ids: Sequence[str],
                 matrix: np.ndarray) -> None:
        self.major = major
        self.indices = list(indices)
        self.preset_ids = list(preset_ids)
        self.matrix = matrix
        self.link = _linkage(matrix) if len(indices) > 1 else np.zeros((0, 4))

    def labels(self, threshold: float) -> np.ndarray:
        return _flat_clusters(self.link, threshold, len(self.indices))

    def cluster_count(self, threshold: float) -> int:
        if len(self.indices) <= 1:
            return len(self.indices)
        return int(np.unique(self.labels(threshold)).size)

    def clusters(self, threshold: float) -> dict[str, list[str]]:
        """cluster_id (= min preset_id in cluster) -> sorted member preset ids."""
        labels = self.labels(threshold)
        buckets: dict[int, list[str]] = {}
        for label, preset_id in zip(labels.tolist(), self.preset_ids):
            buckets.setdefault(label, []).append(preset_id)
        result: dict[str, list[str]] = {}
        for members in buckets.values():
            members = sorted(members)
            result[members[0]] = members
        return result

    def nearest_cluster_pair(self, threshold: float) -> tuple[list[str], list[str]] | None:
        """Two clusters joined by the smallest above-threshold average-linkage merge."""
        if self.link.shape[0] == 0:
            return None
        leaves = len(self.preset_ids)
        for row in self.link:
            if float(row[2]) > threshold:
                left = self._leaves(int(row[0]), leaves)
                right = self._leaves(int(row[1]), leaves)
                return (sorted(left), sorted(right))
        return None

    def _leaves(self, node: int, leaves: int) -> list[str]:
        stack = [node]
        found: list[str] = []
        while stack:
            current = stack.pop()
            if current < leaves:
                found.append(self.preset_ids[current])
                continue
            row = self.link[current - leaves]
            stack.append(int(row[0]))
            stack.append(int(row[1]))
        return found


def group_by_major(records: Sequence[Any], matrix: np.ndarray) -> list[MajorGroup]:
    order: dict[str, list[int]] = {}
    for index, row in enumerate(records):
        order.setdefault(row.style_major, []).append(index)
    groups = []
    for major in sorted(order):
        indices = order[major]
        groups.append(MajorGroup(
            major, indices, [records[i].preset_id for i in indices], matrix[indices]
        ))
    return groups


def total_clusters(groups: Sequence[MajorGroup], threshold: float) -> int:
    return sum(group.cluster_count(threshold) for group in groups)


def search_threshold(groups: Sequence[MajorGroup], target: int, upper: float,
                     iterations: int = 80) -> float:
    """Smallest threshold (to 6 decimals) whose total cluster count is <= target."""
    lo, hi = 0.0, max(upper, 1e-6)
    if total_clusters(groups, hi) > target:
        return round(hi, 6)
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        if total_clusters(groups, mid) <= target:
            hi = mid
        else:
            lo = mid
    value = float(np.ceil(hi * 1e6) / 1e6)
    while total_clusters(groups, value) > target:
        value = float(np.ceil((value + 1e-6) * 1e6) / 1e6)
    return value


def tier_stats(groups: Sequence[MajorGroup], threshold: float) -> dict[str, Any]:
    sizes: list[int] = []
    per_major: dict[str, dict[str, int]] = {}
    for group in groups:
        clusters = group.clusters(threshold)
        group_sizes = [len(members) for members in clusters.values()]
        sizes.extend(group_sizes)
        per_major[group.major] = {
            "records": len(group.preset_ids),
            "clusters": len(clusters),
            "singletons": sum(1 for size in group_sizes if size == 1),
            "max_cluster_size": max(group_sizes) if group_sizes else 0,
        }
    return {
        "threshold": threshold,
        "clusters": len(sizes),
        "singletons": sum(1 for size in sizes if size == 1),
        "max_cluster_size": max(sizes) if sizes else 0,
        "mean_cluster_size": round(float(np.mean(sizes)), 4) if sizes else 0.0,
        "per_major": per_major,
    }


# --------------------------------------------------------------------------- artifacts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_clusters(path: Path, groups: Sequence[MajorGroup], threshold: float) -> None:
    rows: list[tuple[str, str, str]] = []
    for group in groups:
        for cluster_id, members in group.clusters(threshold).items():
            for preset_id in members:
                rows.append((preset_id, group.major, cluster_id))
    rows.sort()
    payload = "".join(
        json.dumps(
            {"preset_id": preset_id, "style_major": major, "cluster_id": cluster_id},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ) + "\n"
        for preset_id, major, cluster_id in rows
    )
    path.write_text(payload, encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- rendering


_PROBE: np.ndarray | None = None
_LOADER = None
_DATABUILD: Path | None = None


def _init_worker(probe: np.ndarray, databuild: str) -> None:
    global _PROBE, _LOADER, _DATABUILD
    from dataset_build.agent_loop.source_reach import configured_lut_loader

    _PROBE = probe
    _DATABUILD = Path(databuild)
    _LOADER = configured_lut_loader(_DATABUILD)


def _render_one(job: tuple[str, str, str, int]) -> tuple[str, bool, str]:
    from PIL import Image

    from dataset_build.src.construct.rendering import apply_lut_cpu_oracle

    preset_id, lut_path, out_path, short_edge = job
    try:
        grid, dmin, dmax = _LOADER.load(Path(lut_path))
        rendered = apply_lut_cpu_oracle(_PROBE, grid, domain_min=dmin, domain_max=dmax)
        image = Image.fromarray(np.clip(rendered * 255.0 + 0.5, 0, 255).astype(np.uint8))
        scale = short_edge / min(image.size)
        if scale < 1.0:
            size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            image = image.resize(size, Image.LANCZOS)
        image.save(out_path, format="PNG", optimize=True)
        return (preset_id, True, "")
    except Exception as exc:  # pragma: no cover - reported, never silent
        return (preset_id, False, f"{type(exc).__name__}: {exc}")


def load_probe(jsonl: Path, short_edge: int) -> tuple[np.ndarray, str]:
    from PIL import Image

    with jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                break
        else:
            raise ValueError(f"no probe row in {jsonl}")
    source = Path(str(row["source_path"]))
    with Image.open(source) as image:
        image = image.convert("RGB")
        scale = short_edge / min(image.size)
        if scale < 1.0:
            size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            image = image.resize(size, Image.LANCZOS)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return array, str(source)


# --------------------------------------------------------------------------- pilot


def pick_clusters(clusters: Mapping[str, Sequence[str]], count: int) -> list[str]:
    """Deterministic pick: 3 largest multi-member clusters + evenly spaced remainder."""
    multi = sorted(
        (cid for cid, members in clusters.items() if len(members) > 1),
        key=lambda cid: (-len(clusters[cid]), cid),
    )
    if len(multi) <= count:
        return multi
    head = multi[:3]
    rest = multi[3:]
    want = count - len(head)
    step = len(rest) / float(want)
    picked = [rest[min(len(rest) - 1, int(round(i * step)))] for i in range(want)]
    seen: list[str] = []
    for cid in head + picked:
        if cid not in seen:
            seen.append(cid)
    for cid in rest:
        if len(seen) >= count:
            break
        if cid not in seen:
            seen.append(cid)
    return seen[:count]


def pick_members(members: Sequence[str], matrix_by_id: Mapping[str, np.ndarray],
                 limit: int) -> list[str]:
    """Representative (min preset_id) + members farthest from it: worst-case intra spread."""
    members = sorted(members)
    if len(members) <= limit:
        return members
    anchor = matrix_by_id[members[0]]
    scored = sorted(
        members[1:],
        key=lambda pid: (-float(np.linalg.norm(matrix_by_id[pid] - anchor)), pid),
    )
    return [members[0]] + scored[:limit - 1]


def build_pilot(pilot_dir: Path, tiers: Sequence[dict[str, Any]], groups: Sequence[MajorGroup],
                records_by_id: Mapping[str, Any], matrix_by_id: Mapping[str, np.ndarray],
                probe: np.ndarray, probe_source: str, databuild: Path, *,
                majors: Sequence[str], clusters_per_major: int, members_per_cluster: int,
                workers: int, member_edge: int, pair_edge: int,
                render_budget: int) -> dict[str, Any]:
    from PIL import Image

    renders = pilot_dir / "renders"
    renders.mkdir(parents=True, exist_ok=True)
    group_by_name = {group.major: group for group in groups}

    plan: dict[str, dict[str, Any]] = {}
    for tier in tiers:
        threshold = tier["threshold"]
        tier_plan: dict[str, Any] = {"majors": {}}
        for major in majors:
            group = group_by_name[major]
            clusters = group.clusters(threshold)
            chosen = pick_clusters(clusters, clusters_per_major)
            shown = {
                cid: pick_members(clusters[cid], matrix_by_id, members_per_cluster)
                for cid in chosen
            }
            pair = group.nearest_cluster_pair(threshold)
            pair_reps = None
            if pair is not None:
                pair_reps = (pair[0][0], pair[1][0], len(pair[0]), len(pair[1]))
            tier_plan["majors"][major] = {
                "clusters": shown,
                "cluster_sizes": {cid: len(clusters[cid]) for cid in chosen},
                "total_clusters": len(clusters),
                "pair": pair_reps,
            }
        plan[tier["tag"]] = tier_plan

    member_ids: set[str] = set()
    pair_ids: set[str] = set()
    for tier_plan in plan.values():
        for block in tier_plan["majors"].values():
            for members in block["clusters"].values():
                member_ids.update(members)
            if block["pair"]:
                pair_ids.update(block["pair"][:2])
    member_ids -= pair_ids

    jobs: list[tuple[str, str, str, int]] = []
    for preset_id in sorted(pair_ids):
        jobs.append((preset_id, records_by_id[preset_id].path,
                     str(renders / f"{preset_id}.png"), pair_edge))
    for preset_id in sorted(member_ids):
        jobs.append((preset_id, records_by_id[preset_id].path,
                     str(renders / f"{preset_id}.png"), member_edge))
    if len(jobs) > render_budget:
        raise RuntimeError(f"render budget exceeded: {len(jobs)} > {render_budget}")

    started = time.time()
    failures: list[tuple[str, str]] = []
    with futures.ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(probe, str(databuild))
    ) as pool:
        for preset_id, ok, detail in pool.map(_render_one, jobs, chunksize=4):
            if not ok:
                failures.append((preset_id, detail))
    elapsed = time.time() - started

    probe_png = pilot_dir / "renders" / "_probe.png"
    image = Image.fromarray(np.clip(probe * 255.0 + 0.5, 0, 255).astype(np.uint8))
    scale = pair_edge / min(image.size)
    if scale < 1.0:
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.LANCZOS
        )
    image.save(probe_png, format="PNG", optimize=True)

    write_index_html(pilot_dir, tiers, plan, records_by_id, probe_source, majors, failures)
    return {
        "rendered_images": len(jobs),
        "render_seconds": round(elapsed, 2),
        "render_failures": failures,
        "probe_source": probe_source,
    }


def _thumb(preset_id: str, record: Any, extra: str = "") -> str:
    label = html.escape(preset_id)
    caption = html.escape((record.caption or "")[:60])
    return (
        f'<figure class="thumb"><a href="renders/{label}.png" target="_blank">'
        f'<img src="renders/{label}.png" alt="{label}" loading="lazy"></a>'
        f'<figcaption>{label}{extra}<br><span class="cap">{caption}'
        f"</span></figcaption></figure>"
    )


def write_index_html(pilot_dir: Path, tiers: Sequence[dict[str, Any]], plan: Mapping[str, Any],
                     records_by_id: Mapping[str, Any], probe_source: str,
                     majors: Sequence[str], failures: Sequence[tuple[str, str]]) -> None:
    parts: list[str] = [
        "<!doctype html><html lang=\"zh\"><head><meta charset=\"utf-8\">",
        "<title>LUT 效果聚类 pilot 2026-08-19</title>",
        "<style>",
        "body{font-family:system-ui,'Noto Sans CJK SC',sans-serif;margin:24px;"
        "background:#fff;color:#111}",
        "h1{font-size:20px}h2{font-size:18px;margin-top:36px;border-top:2px solid #333;"
        "padding-top:12px}",
        "h3{font-size:15px;margin:18px 0 6px}",
        "table{border-collapse:collapse;font-size:13px;margin:8px 0}",
        "th,td{border:1px solid #bbb;padding:3px 8px;text-align:right}th{background:#f2f2f2}",
        "td.l,th.l{text-align:left}",
        ".row{display:flex;gap:6px;flex-wrap:wrap;align-items:flex-start;"
        "margin:4px 0 14px}",
        ".thumb{margin:0;font-size:10px;text-align:center;max-width:220px}",
        ".thumb img{display:block;width:100%;height:auto;border:1px solid #ccc}",
        ".pair .thumb{max-width:420px}",
        ".cap{color:#666}",
        ".clabel{font-size:12px;color:#333;margin:8px 0 2px;font-family:monospace}",
        ".pair{background:#fafaf0;padding:8px;border:1px dashed #999}",
        "</style></head><body>",
        "<h1>LUT 效果聚类 pilot（3 档阈值）</h1>",
        f"<p>探针图：<code>{html.escape(probe_source)}</code>；"
        "特征 39 维（neutral_ramp 5×(L_out−L_in, a_out, b_out) ⊕ "
        "8 bands×(d_hue_deg, d_sat_pct, d_lum_pct)），按 de_med 归一化；"
        "major 内 average-linkage，全局统一距离阈值。"
        f"特征 spec：<code>{FEATURE_SPEC}</code></p>",
        '<p>原图（未套 LUT）：</p><div class="row">'
        '<figure class="thumb"><img src="renders/_probe.png" alt="probe">'
        "<figcaption>source</figcaption></figure></div>",
        "<h2>三档总览</h2>",
        "<table><tr><th class=\"l\">tag</th><th>阈值</th><th>总簇数</th><th>簇数/记录数</th>"
        "<th>singleton</th><th>最大簇 size</th><th>平均簇 size</th></tr>",
    ]
    for tier in tiers:
        parts.append(
            f"<tr><td class=\"l\"><a href=\"#{tier['tag']}\">{tier['tag']}</a></td>"
            f"<td>{tier['threshold']:.6f}</td><td>{tier['clusters']}</td>"
            f"<td>{tier['clusters'] / tier['records']:.4f}</td>"
            f"<td>{tier['singletons']}</td><td>{tier['max_cluster_size']}</td>"
            f"<td>{tier['mean_cluster_size']:.2f}</td></tr>"
        )
    parts.append("</table>")

    for tier in tiers:
        tag = tier["tag"]
        parts.append(f"<h2 id=\"{tag}\">档 {tag}：阈值 {tier['threshold']:.6f}</h2>")
        parts.append(
            "<table><tr><th class=\"l\">style_major</th><th>记录数</th><th>簇数</th>"
            "<th>singleton</th><th>最大簇 size</th></tr>"
        )
        for major, stats in sorted(tier["per_major"].items()):
            parts.append(
                f"<tr><td class=\"l\">{html.escape(major)}</td><td>{stats['records']}</td>"
                f"<td>{stats['clusters']}</td><td>{stats['singletons']}</td>"
                f"<td>{stats['max_cluster_size']}</td></tr>"
            )
        parts.append(
            f"<tr><td class=\"l\"><b>全库</b></td><td>{tier['records']}</td>"
            f"<td>{tier['clusters']}</td><td>{tier['singletons']}</td>"
            f"<td>{tier['max_cluster_size']}</td></tr></table>"
        )
        for major in majors:
            block = plan[tag]["majors"][major]
            parts.append(
                f"<h3>{html.escape(major)}（该 major 共 {block['total_clusters']} 簇，"
                f"下列展示 {len(block['clusters'])} 个多成员簇）</h3>"
            )
            for cluster_id, members in block["clusters"].items():
                size = block["cluster_sizes"][cluster_id]
                parts.append(
                    f'<div class="clabel">cluster {html.escape(cluster_id)} · size={size}'
                    f" · 展示 {len(members)}</div><div class=\"row\">"
                )
                for preset_id in members:
                    parts.append(_thumb(preset_id, records_by_id[preset_id]))
                parts.append("</div>")
            if block["pair"]:
                left, right, lsize, rsize = block["pair"]
                parts.append(
                    '<div class="pair"><div class="clabel">最近簇间对（average-linkage 最小的'
                    f" 超阈值合并）：{html.escape(left)}(size={lsize}) vs "
                    f'{html.escape(right)}(size={rsize})</div><div class="row">'
                )
                parts.append(_thumb(left, records_by_id[left], " · 簇 A 代表"))
                parts.append(_thumb(right, records_by_id[right], " · 簇 B 代表"))
                parts.append("</div></div>")
    if failures:
        parts.append("<h2>渲染失败</h2><ul>")
        for preset_id, detail in failures:
            parts.append(f"<li>{html.escape(preset_id)}: {html.escape(detail)}</li>")
        parts.append("</ul>")
    parts.append("</body></html>")
    (pilot_dir / "index.html").write_text("\n".join(parts), encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- main


def load_catalog(config_path: Path) -> tuple[LutCatalog, Path, Path]:
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    databuild = (config_path.parent / str(data["agent_loop"]["databuild_config"])).resolve()
    catalog_table = data.get("catalog") or {}
    annotations = Path(str(catalog_table["annotations"])).expanduser()
    config = CatalogConfig(
        annotations=annotations,
        global_major_limit=int(catalog_table.get("global_major_limit", 8)),
        global_per_major_limit=int(catalog_table.get("global_per_major_limit", 4)),
        local_limit=int(catalog_table.get("local_limit", 12)),
        reach_limit=int(catalog_table.get("reach_limit", 300)),
    )
    return LutCatalog.load(config, databuild), databuild, annotations


def features_inputs(databuild: Path) -> dict[str, Any]:
    """C1b item 11: the preset bank's content fingerprint, for the manifest.

    `LutCatalog.load` keeps only presets that appear in `<bank_dir>/features.jsonl` AND
    resolve to an existing renderable path, so the bank decides which LUTs a clustering
    run even saw; `annotations_sha256` alone covers only half of the catalog identity.
    Kept local (not imported from `lut_render_distance`) so this module has no new
    import edge.
    """
    with Path(databuild).open("rb") as handle:
        build = tomllib.load(handle)
    path = Path(str((build.get("presets") or {}).get("bank_dir") or "")) / "features.jsonl"
    if not path.is_file():
        return {"features_jsonl": str(path), "features_jsonl_sha256": None,
                "features_jsonl_rows": None}
    with path.open("r", encoding="utf-8") as handle:
        rows = sum(1 for line in handle if line.strip())
    return {"features_jsonl": str(path), "features_jsonl_sha256": sha256_file(path),
            "features_jsonl_rows": rows}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs/agent_loop.terra-smoke.toml")
    parser.add_argument("--out-dir", type=Path, default=Path("/home/bc/data/scratch/lut_clusters"))
    parser.add_argument("--pilot-dir", type=Path,
                        default=REPO_ROOT / "docs/assets/lut_cluster_pilot_20260819")
    parser.add_argument("--probe-jsonl", type=Path,
                        default=REPO_ROOT / "configs/agent_loop.smoke5.jsonl")
    parser.add_argument("--targets", type=str, default=",".join(str(t) for t in DEFAULT_TARGETS))
    parser.add_argument("--tags", type=str, default=",".join(DEFAULT_TAGS))
    parser.add_argument("--pilot-majors", type=int, default=5)
    parser.add_argument("--clusters-per-major", type=int, default=6)
    parser.add_argument("--members-per-cluster", type=int, default=6)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--probe-short-edge", type=int, default=512)
    parser.add_argument("--member-edge", type=int, default=320)
    parser.add_argument("--pair-edge", type=int, default=448)
    parser.add_argument("--render-budget", type=int, default=600)
    parser.add_argument("--artifacts-only", action="store_true",
                        help="skip the visual pilot (used for the determinism re-run)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    targets = [float(value) for value in args.targets.split(",")]
    tags = [value.strip() for value in args.tags.split(",")]
    if len(targets) != len(tags):
        raise SystemExit("--targets and --tags length mismatch")

    catalog, databuild, annotations = load_catalog(args.config)
    records = list(catalog.records)
    matrix, bands, unnormalised = build_features(records)
    groups = group_by_major(records, matrix)
    upper = max(
        (float(group.link[:, 2].max()) for group in groups if group.link.shape[0]), default=1.0
    )

    tiers: list[dict[str, Any]] = []
    for tag, target in zip(tags, targets):
        want = int(round(len(records) * target))
        threshold = search_threshold(groups, want, upper)
        stats = tier_stats(groups, threshold)
        stats.update({
            "tag": tag, "target_ratio": target, "target_clusters": want,
            "records": len(records),
        })
        tiers.append(stats)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, Any]] = {}
    for tier in tiers:
        path = args.out_dir / f"clusters.{tier['tag']}.jsonl"
        write_clusters(path, groups, tier["threshold"])
        artifacts[tier["tag"]] = {
            "path": str(path), "sha256": sha256_file(path),
            "lines": sum(1 for _ in path.open("r", encoding="utf-8")),
        }

    manifest = {
        "schema": "lut-effect-clusters-v1",
        "feature_spec": FEATURE_SPEC,
        "feature_dim": FEATURE_DIM,
        "band_order": list(bands),
        "linkage": {"method": "average", "metric": "euclidean",
                    "normalisation": "vector / de_med", "scope": "per style_major",
                    "threshold_scope": "global"},
        "inputs": {
            "agent_loop_config": str(args.config),
            "databuild_config": str(databuild),
            "annotations": str(annotations),
            "annotations_sha256": sha256_file(annotations),
            # C1b item 11: the preset bank decides which LUTs the catalog contains.
            **features_inputs(databuild),
        },
        "records": {
            "clustered": len(records),
            "style_majors": len(groups),
            "unnormalised_preset_ids": unnormalised,
        },
        "tiers": [
            {key: tier[key] for key in (
                "tag", "target_ratio", "target_clusters", "threshold", "clusters",
                "singletons", "max_cluster_size", "mean_cluster_size", "per_major",
            )}
            for tier in tiers
        ],
        "artifacts": artifacts,
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )

    summary = {"manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path)}
    if not args.artifacts_only:
        counts: dict[str, int] = {}
        for row in records:
            counts[row.style_major] = counts.get(row.style_major, 0) + 1
        majors = sorted(counts, key=lambda name: (-counts[name], name))[:args.pilot_majors]
        if "warm__desaturated__neutral" not in majors:
            majors = ["warm__desaturated__neutral"] + majors[:-1]
        probe, probe_source = load_probe(args.probe_jsonl, args.probe_short_edge)
        args.pilot_dir.mkdir(parents=True, exist_ok=True)
        pilot = build_pilot(
            args.pilot_dir, tiers, groups, catalog.by_id,
            {row.preset_id: matrix[index] for index, row in enumerate(records)},
            probe, probe_source, databuild,
            majors=majors, clusters_per_major=args.clusters_per_major,
            members_per_cluster=args.members_per_cluster, workers=args.workers,
            member_edge=args.member_edge, pair_edge=args.pair_edge,
            render_budget=args.render_budget,
        )
        pilot["index"] = str(args.pilot_dir / "index.html")
        pilot["majors"] = majors
        summary["pilot"] = pilot
    print(json.dumps({"tiers": [
        {key: tier[key] for key in (
            "tag", "threshold", "clusters", "singletons", "max_cluster_size", "target_clusters",
        )} for tier in tiers
    ], **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
