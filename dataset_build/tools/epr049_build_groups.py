"""EPR-049 data production: 16-LUT global groups, OneAlign scores, gemini tournament.

One group = one eligible source + one taxonomy ``major`` + 16 distinct LUT presets of
that major rendered whole-image.  Every group is scored twice and independently:

* **OneAlign** (``construct.canonical_qa.OneAlignScorer``) scores the source and all 16
  candidates on 0..100.  ``canonical_qa.rank_candidates`` is not used — it asserts
  ``len(candidates) == 8`` and that main-chain file must not change — so the ordering is
  done here from the raw scores.
* **gemini** (relay, Responses API) runs a two-round tournament: round 1 splits the 16
  slots into four deterministic quadruples and ranks each quadruple in one call; the four
  rank-1 winners meet in one round-2 call whose full ranking yields top2.  Five calls per
  group.  Every call carries **exactly two images**: the source photo, then one 2x2
  contact sheet of the four candidates badged 1..4 (row-major) and nothing else.  Each
  montage cell is asserted to keep a short edge of >= 512px both as composed and as the
  encoder ships it.

Everything is deterministic given ``build_id`` + ``seed``: the source draw, the major
assignment, the preset draw, the quadruple split and the in-call presentation order all
come from sha1 over explicit tuples.  ``--smoke N`` is the first ``N`` groups of the very
same list, so a smoke run and the full run share one output directory and the smoke rows
are re-used rather than recomputed.

Journals (append-only, both under ``output_root``):

* ``stage_render_qa.jsonl`` — source, major, 16 candidates, OneAlign scores.  Written the
  moment the GPU work of a group finishes, before any relay call, so a relay outage can
  never destroy finished render/scoring work.
* ``groups.jsonl`` — the merged row including the gemini tournament.
* ``failures.jsonl`` — one row per failed relay attempt, sanitized.

Usage::

    python -m dataset_build.tools.epr049_build_groups \
        --config experiments/prs/EPR-049_gemini-vs-onealign-aesthetic/epr049.toml --smoke 2
    python -m dataset_build.tools.epr049_build_groups --config <cfg>          # full run

The config carries a relay ``api_key``; it is never printed, never journaled and never
placed in an asset.  Every message that leaves this module goes through
``construct.config.redact_text`` with the config's own secret list.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
import tomllib
from collections import Counter as collections_Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "epr049-groups-v3"
CANDIDATES_PER_GROUP = 16
QUAD_SIZE = 4
# v3: four prelim calls, plus a final only when >= 2 finalists survive, so a group
# costs 4 or 5 calls.  The guard no longer asserts a constant 5.
PRELIM_CALLS_PER_GROUP = 4
MAX_CALLS_PER_GROUP = 5
# v3: each prelim compares 4 candidates + the source image itself, shuffled together.
# Picking the source = "no candidate beats leaving it alone".
PRELIM_OPTIONS = 5
SOURCE_ITEM = "src"
MIN_FINAL_OPTIONS = 2
# Task card: "联图单格分辨率不得低于短边 512".  Asserted twice per call — once on the
# composed montage and once on the montage as the encoder will actually ship it.
MONTAGE_CELL_MIN_SHORT_EDGE = 512
# The arm whose journals keep the unsuffixed legacy filenames (groups.jsonl, ...).
LEGACY_ARM = "gemini"
# Journal key holding one arm's tournament.  Rows written before the multi-arm
# split carry it under "gemini"; ``tournament_of`` reads either.
TOURNAMENT_KEY = "tournament"
# S-split (tools/data_splits/vr_common.py): bucket 0..89 train, 90..94 val
# (V_where/V_what), 95..99 test (T_final).  Only train sources are drawn.
SPLIT_SEED = "verasplit-v1"
TRAIN_BUCKET_MAX = 89


# --------------------------------------------------------------------------- determinism


def sha1_hex(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def sha1_int(*parts: object) -> int:
    return int(sha1_hex(*parts)[:16], 16)


def seeded_order(items, *seed_parts: object) -> list[str]:
    """``presets._seeded_order`` verbatim: sorted unique items, sha256-seeded shuffle."""
    rows = sorted(set(str(item) for item in items))
    digest = hashlib.sha256("\x1f".join(map(str, seed_parts)).encode()).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(rows)
    return rows


def split_bucket(source_id: str) -> int:
    return int(hashlib.sha1(f"{SPLIT_SEED}:{source_id}".encode("utf-8")).hexdigest()[:8], 16) % 100


# ------------------------------------------------------------------------------- config


@dataclass(frozen=True, slots=True)
class Endpoint:
    id: str
    base_url: str
    api_key: str
    concurrency: int


@dataclass(frozen=True, slots=True)
class Epr049Config:
    path: Path
    arm: str
    build_id: str
    seed: int
    target_groups: int
    candidates_per_group: int
    output_root: Path
    preset_filter: str
    subject_cache: str
    postgres_dsn: str
    bank_dir: Path
    taxonomy: Path
    short_edge: int
    jpeg_quality: int
    iaa_batch: int
    external_model: str
    image_long_edge: int
    image_jpeg_quality: int
    montage_cell_short_edge: int
    montage_long_edge: int
    montage_jpeg_quality: int
    reasoning_effort: str
    max_output_tokens: int
    transport_attempts: int
    backoff_cap_s: float
    backoff_base_s: float
    endpoints: tuple[Endpoint, ...]

    def journal_v3(self, base: str, ext: str) -> Path:
        """v3 output path, always arm-qualified: ``<base>.v3.<arm>.<ext>``.

        v2's journals keep their own names and are never touched by a v3 run.
        """
        return self.output_root / f"{base}.v3.{self.arm}.{ext}"

    def journal(self, base: str, ext: str) -> Path:
        """Per-arm output path.

        The first arm keeps the unsuffixed legacy names so the 100 gemini groups already
        on disk stay exactly where they are; every other arm gets its own namespace and
        can never overwrite or interleave with them.
        """
        name = f"{base}.{ext}" if self.arm == LEGACY_ARM else f"{base}.{self.arm}.{ext}"
        return self.output_root / name

    @property
    def secrets(self) -> tuple[str, ...]:
        from dataset_build.src.construct.config import uri_secrets

        values: list[str] = []
        for endpoint in self.endpoints:
            values.append(endpoint.api_key)
            values.append(endpoint.base_url)
            values.extend(uri_secrets(endpoint.base_url))
        values.append(self.postgres_dsn)
        values.extend(uri_secrets(self.postgres_dsn))
        return tuple(value for value in values if value)


def load_config(path: Path, arm: str | None = None) -> Epr049Config:
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise SystemExit(
            f"{path} carries a relay api_key and must be chmod 0600 (is {mode:04o})"
        )
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    annotation = raw["annotation"]
    arms = annotation.get("arms") or {}
    if not arms:
        raise SystemExit("epr049 config has no [annotation.arms.<name>] tables")
    arm = str(arm or annotation.get("default_arm") or LEGACY_ARM)
    if arm not in arms:
        raise SystemExit(f"unknown --model-arm {arm!r}; config defines {sorted(arms)}")
    arm_cfg = arms[arm]
    endpoints = tuple(
        Endpoint(
            id=str(row["id"]),
            base_url=str(row["base_url"]),
            api_key=str(row["api_key"]),
            concurrency=int(row["concurrency"]),
        )
        for row in arm_cfg["external_endpoints"]
    )
    if not endpoints:
        raise SystemExit(
            f"arm {arm!r} has no [[annotation.arms.{arm}.external_endpoints]]")
    # Routing is ``endpoints[sha1_int(...) % len(endpoints)]``.  Sharing one list between
    # arms would change that modulus and silently re-route already-finished groups, and
    # would also put both arms in one rate-limit bucket.
    shared = {e["id"] for name, cfg in arms.items() if name != arm
              for e in cfg["external_endpoints"]} & {e.id for e in endpoints}
    if shared:
        raise SystemExit(
            f"arm {arm!r} shares endpoint id(s) {sorted(shared)} with another arm; "
            "each arm needs its own lane")
    if raw["sources"].get("split_seed", SPLIT_SEED) != SPLIT_SEED:
        raise SystemExit("split_seed must stay 'verasplit-v1' (frozen S-split rule)")
    if raw["sources"].get("allowed_split", "train") != "train":
        raise SystemExit("allowed_split must stay 'train': val/test are V_*/T_final")
    if str(raw.get("preset_filter", "lut")) != "lut":
        raise SystemExit("EPR-049 draws LUT presets only")
    montage_cell_short_edge = int(annotation["montage_cell_short_edge"])
    if montage_cell_short_edge < MONTAGE_CELL_MIN_SHORT_EDGE:
        raise SystemExit(
            f"montage_cell_short_edge = {montage_cell_short_edge} is below the "
            f"task-card floor {MONTAGE_CELL_MIN_SHORT_EDGE}"
        )
    return Epr049Config(
        path=path,
        arm=arm,
        build_id=str(raw["build_id"]),
        seed=int(raw["seed"]),
        target_groups=int(raw["target_groups"]),
        candidates_per_group=int(raw.get("candidates_per_group", CANDIDATES_PER_GROUP)),
        output_root=Path(str(raw["output_root"])),
        preset_filter="lut",
        subject_cache=str(raw["sources"]["subject_cache"]),
        postgres_dsn=str(raw["sources"]["postgres_dsn"]),
        bank_dir=Path(str(raw["presets"]["bank_dir"])),
        taxonomy=Path(str(raw["presets"]["taxonomy"])),
        short_edge=int(raw["render"]["short_edge"]),
        jpeg_quality=int(raw["render"]["jpeg_quality"]),
        iaa_batch=int(raw["render"]["iaa_batch"]),
        external_model=str(arm_cfg["external_model"]),
        image_long_edge=int(annotation["image_long_edge"]),
        image_jpeg_quality=int(annotation["image_jpeg_quality"]),
        montage_cell_short_edge=montage_cell_short_edge,
        montage_long_edge=int(annotation["montage_long_edge"]),
        montage_jpeg_quality=int(annotation["montage_jpeg_quality"]),
        reasoning_effort=str(annotation["external_reasoning_effort"]),
        max_output_tokens=int(annotation["external_max_output_tokens"]),
        transport_attempts=int(annotation["transport_attempts"]),
        backoff_cap_s=float(arm_cfg.get("backoff_cap_s", BACKOFF_CAP_S)),
        backoff_base_s=float(arm_cfg.get("backoff_base_s", BACKOFF_BASE_S)),
        endpoints=endpoints,
    )


def databuild_config(config: Epr049Config):
    """A ``DatabuildConfig`` built in process, so the main-chain renderer runs unchanged.

    The canonical loader validates a full production TOML (masks, viewer, local vGate,
    relay pool); EPR-049 needs only the render + preset half of it, and the alternative —
    a second copy of the LUT render math — is exactly the divergence this avoids.
    """
    from dataset_build.src.construct.config import (
        AnnotationConfig,
        DatabuildConfig,
        LocalAnnotationConfig,
        MasksConfig,
        MixConfig,
        PresetsConfig,
        RenderConfig,
        SourcesConfig,
        ViewerConfig,
    )

    return DatabuildConfig(
        schema_version=1,
        build_id=config.build_id,
        seed=config.seed,
        target_groups=config.target_groups,
        output_root=config.output_root,
        preset_filter="lut",
        mix=MixConfig(local=0.0, global_=1.0),
        sources=SourcesConfig(
            subject_cache=Path(config.subject_cache),
            postgres_dsn=config.postgres_dsn,
            max_source_uses=1,
        ),
        presets=PresetsConfig(
            bank_dir=config.bank_dir,
            taxonomy=config.taxonomy,
            fidelity_de_max=6.0,
            disabled_formats=(),
        ),
        render=RenderConfig(
            short_edge=config.short_edge,
            jpeg_quality=config.jpeg_quality,
            gpu_concurrency=1,
            diff_short_edge=512,
            visible_de_min=2.5,
            visible_fraction_de=2.3,
            visible_fraction_min=0.50,
            iaa_batch=config.iaa_batch,
            qa_preflight_forward=False,
        ),
        masks=MasksConfig(linear_target_alpha_mass=0.50, sam3_relabel_attempts=2),
        annotation=AnnotationConfig(
            external_model=config.external_model,
            image_long_edge=config.image_long_edge,
            image_jpeg_quality=config.image_jpeg_quality,
            external_reasoning_effort=config.reasoning_effort,
            external_max_output_tokens=config.max_output_tokens,
            transport_attempts_per_round=config.transport_attempts,
            queue_rounds=1,
            external_endpoints=(),
            local=LocalAnnotationConfig(
                base_url="http://127.0.0.1:8003/v1", api_key="EMPTY",
                model="unused", temperature=0.2, enable_thinking=False,
                max_output_tokens=2048,
            ),
            local_fallback=False,
        ),
        viewer=ViewerConfig(postgres_dsn=config.postgres_dsn),
    )


# ------------------------------------------------------------------------------ journals


class Journal:
    """Append-only JSONL with an fsync per row and a process-wide write lock."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, row: Mapping[str, Any]) -> None:
        line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def read(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows


# ------------------------------------------------------------------------------ sampling


def qualified_majors(catalog, minimum: int) -> dict[str, dict[str, list[str]]]:
    """``major -> minor -> [preset_id]`` for majors holding ``minimum`` distinct presets."""
    tree: dict[str, dict[str, list[str]]] = {}
    for major, minors in catalog.tree.items():
        by_minor = {
            minor: sorted({link.preset.preset_id for link in links})
            for minor, links in minors.items()
        }
        distinct = {pid for ids in by_minor.values() for pid in ids}
        if len(distinct) >= minimum:
            tree[major] = by_minor
    return tree


def draw_presets(tree_major: Mapping[str, Sequence[str]], *, k: int,
                 build_id: str, seed: int, group_id: str, major: str) -> list[tuple[str, str]]:
    """Round-robin over minors, seeded preset order inside each: ``[(preset_id, minor)]``.

    A simplified but deterministic stand-in for ``presets.CoverageSelector``'s three-level
    draw: the major is already fixed by the caller, minors rotate in a per-cycle seeded
    bag so a group spreads over the major's minors, and inside a minor the preset order is
    a seeded shuffle.  Distinctness is enforced by ``used``.
    """
    minors = sorted(tree_major)
    orders = {
        minor: seeded_order(tree_major[minor], build_id, seed, group_id, major, minor, "preset")
        for minor in minors
    }
    cursor = {minor: 0 for minor in minors}
    used: set[str] = set()
    picked: list[tuple[str, str]] = []
    cycle = 0
    while len(picked) < k:
        bag = seeded_order(minors, build_id, seed, group_id, major, "minor", cycle)
        progressed = False
        for minor in bag:
            if len(picked) >= k:
                break
            order = orders[minor]
            while cursor[minor] < len(order) and order[cursor[minor]] in used:
                cursor[minor] += 1
            if cursor[minor] < len(order):
                preset_id = order[cursor[minor]]
                cursor[minor] += 1
                used.add(preset_id)
                picked.append((preset_id, minor))
                progressed = True
        if not progressed:
            raise RuntimeError(f"major {major!r} cannot supply {k} distinct presets")
        cycle += 1
    return picked


def plan_groups(config: Epr049Config, sources, majors: Sequence[str],
                tree: Mapping[str, Mapping[str, Sequence[str]]]) -> list[dict[str, Any]]:
    """The full deterministic group list; ``--smoke N`` is its first ``N`` entries."""
    train = [row for row in sources if split_bucket(row.source_id) <= TRAIN_BUCKET_MAX]
    ordered = sorted(
        train, key=lambda row: (sha1_hex(config.build_id, config.seed, "source", row.source_id),
                                row.source_id)
    )
    if len(ordered) < config.target_groups:
        raise SystemExit(
            f"train-split source pool has {len(ordered)} rows, need {config.target_groups}"
        )
    majors = list(majors)
    plans: list[dict[str, Any]] = []
    for index, source in enumerate(ordered[: config.target_groups]):
        cycle, position = divmod(index, len(majors))
        bag = seeded_order(majors, config.build_id, config.seed, "major", cycle)
        major = bag[position]
        group_id = f"g{index:04d}"
        picked = draw_presets(
            tree[major], k=config.candidates_per_group, build_id=config.build_id,
            seed=config.seed, group_id=group_id, major=major,
        )
        plans.append({
            "group_id": group_id,
            "group_index": index,
            "source_id": source.source_id,
            "source_path": str(source.source_path),
            "scene": source.scene,
            "split_bucket": split_bucket(source.source_id),
            "major": major,
            "candidates": [
                {
                    "slot_id": f"s{slot:02d}",
                    "candidate_id": sha1_hex(
                        "epr049-candidate", config.build_id, group_id, f"s{slot:02d}", preset_id
                    )[:16],
                    "preset_id": preset_id,
                    "minor": minor,
                }
                for slot, (preset_id, minor) in enumerate(picked)
            ],
        })
    return plans


# ------------------------------------------------------------------------------ GPU work


class GroupRenderer:
    """Main-chain LUT renderer + OneAlign, both pinned to the single visible CUDA device."""

    def __init__(self, config: Epr049Config):
        from dataset_build.src.construct.presets import PresetCatalog
        from dataset_build.src.construct.rendering import LocalGpuOnlyRenderer

        self.config = config
        self.db_config = databuild_config(config)
        self.catalog = PresetCatalog.load(self.db_config)
        self.renderer = LocalGpuOnlyRenderer.create(self.db_config)
        self.renderer.bind_catalog(self.catalog)
        self.scorer = None

    def load_scorer(self) -> None:
        from dataset_build.src.construct.canonical_qa import OneAlignScorer

        if self.scorer is None:
            self.scorer = OneAlignScorer.create(device="cuda:0")

    def render_group(self, plan: Mapping[str, Any], out_root: Path) -> dict[str, Any]:
        from dataset_build.src.construct.rendering import (
            preprocess_source,
            save_candidate_jpeg,
        )

        source_asset = out_root / "assets" / "sources" / f"{plan['group_id']}.jpg"
        candidate_dir = out_root / "assets" / "candidates" / plan["group_id"]
        prepared = preprocess_source(plan["source_path"], short_edge=self.config.short_edge)
        if not _usable(source_asset):
            save_candidate_jpeg(prepared.pixels, source_asset, quality=self.config.jpeg_quality)
        presets = [self.catalog.by_id[row["preset_id"]] for row in plan["candidates"]]
        results = self.renderer.render_many(prepared, [(preset, None) for preset in presets])
        rendered: list[dict[str, Any]] = []
        for row, preset, result in zip(plan["candidates"], presets, results):
            if isinstance(result, Exception):
                raise result
            asset = candidate_dir / f"{row['slot_id']}.jpg"
            save_candidate_jpeg(result.pixels, asset, quality=self.config.jpeg_quality)
            rendered.append({
                **row,
                "preset_path": str(preset.path),
                "style_name": preset.style_name,
                "render_engine": result.engine,
                "lut_size": result.diagnostics.get("lut_size"),
                "asset": str(asset.relative_to(out_root)),
            })
        return {
            "source_asset": str(source_asset.relative_to(out_root)),
            "source_size": [prepared.width, prepared.height],
            "candidates": rendered,
        }

    def score_group(self, source_asset: Path, candidates: Sequence[Mapping[str, Any]],
                    out_root: Path) -> dict[str, Any]:
        self.load_scorer()
        paths = [str(source_asset)] + [str(out_root / row["asset"]) for row in candidates]
        # ``score_batch`` yields ``None`` for an image it could not score.  Feeding that
        # to ``float()`` used to raise TypeError mid-group; the null is counted, the
        # affected slot is dropped from the ranking and the count is journaled instead.
        values: list[float | None] = []
        batch = max(1, self.config.iaa_batch)
        for start in range(0, len(paths), batch):
            chunk = paths[start:start + batch]
            scored = list(self.scorer.score_batch(chunk))
            if len(scored) != len(chunk):
                raise RuntimeError("OneAlign batch returned a mismatched number of scores")
            values.extend(None if value is None else float(value) for value in scored)
        source_score = values[0]
        scores: dict[str, float] = {}
        null_slots: list[str] = []
        for index, row in enumerate(candidates):
            value = values[index + 1]
            if value is None:
                null_slots.append(row["slot_id"])
                continue
            scores[row["slot_id"]] = round(value, 4)
        # Ties break on slot_id so the order is reproducible from the journal alone.
        ranking = sorted(scores, key=lambda slot: (-scores[slot], slot))
        return {
            "source_score": None if source_score is None else round(source_score, 4),
            "source_score_null": source_score is None,
            "scores": scores,
            "ranking": ranking,
            "top2": ranking[:2],
            "null_slots": null_slots,
            "null_count": len(null_slots) + (1 if source_score is None else 0),
            "scored_candidates": len(scores),
        }


def _usable(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


# ---------------------------------------------------------------------------- tournament


def ranking_schema(n: int) -> dict[str, Any]:
    """Strict schema for a full ranking of ``n`` labelled cells."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["ranking"],
        "properties": {
            "ranking": {
                "type": "array",
                "minItems": n,
                "maxItems": n,
                "items": {"type": "integer", "minimum": 1, "maximum": n},
            }
        },
    }

def build_prompt(n: int, *, source_is_an_option: bool) -> str:
    """Prompt for ranking ``n`` labelled cells of the second image.

    When ``source_is_an_option`` the sheet also contains the unedited photo itself as one
    of the numbered cells, and the model is told so without being told which one.
    """
    labels = ", ".join(str(i) for i in range(1, n + 1))
    head = (
        "You are judging photo edits. You are given exactly TWO images.\n"
        "Image 1 is the INPUT photo (unedited), shown for reference.\n"
        f"Image 2 is a single contact sheet holding {n} numbered cells, read left to "
        "right, top to bottom. Each cell carries a number badge in its top-left corner. "
        "The numbers are arbitrary labels and carry no information.\n"
    )
    if source_is_an_option:
        body = (
            f"{n - 1} of these cells are candidate EDITED versions of the input photo, "
            "and ONE of them is the unedited input photo itself. You are not told which. "
            "Rank all cells by overall aesthetic quality of the photo as shown. "
            "If none of the edits improves on the unedited photo, the unedited one "
            "should rank first.\n"
        )
    else:
        body = (
            "Every cell is a candidate EDITED version of the input photo. "
            "Rank all cells by overall aesthetic quality of the edit.\n"
        )
    return (
        head + body
        + "Judge colour, tone, contrast and how well the result suits this photo.\n"
        + 'Answer with JSON only: {"ranking": [...]} listing the cell numbers from best '
        + f"to worst. The list must be a permutation of {labels}. "
        + "Every number must appear exactly once. Ties are not allowed."
    )

# --------------------------------------------------------------------------- montage
# 2x2 contact sheet, panel+caption convention of build_smoke_review_pack._panel /
# compose_sheet: one flat background colour, a fixed-colour badge, no per-image
# scaling of anything.  The badge vocabulary is closed to "1".."4" — no preset id, no
# model name, no score, no slot id may ever be drawn into the pixels.
MONTAGE_GAP = 12
MONTAGE_BG = (18, 18, 18)
MONTAGE_BADGE_BG = (18, 18, 18)
MONTAGE_BADGE_FG = (245, 245, 245)
MONTAGE_LABELS = ("1", "2", "3", "4", "5")
# Grid shape per option count.  2x3 (with one neutral blank cell) is chosen over 1x5
# for n=5 because a 1x5 strip of 512-short-edge cells runs 2600-3900px on the long
# edge, which the encoder would scale back under the 512 floor; 2x3 stays ~1580-2360.
MONTAGE_GRID = {2: (1, 2), 3: (1, 3), 4: (2, 2), 5: (2, 3)}


def _font(size: int):
    from PIL import ImageFont

    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()  # pragma: no cover


def assert_cell_short_edge(cell_w: int, cell_h: int, where: str) -> int:
    """Runtime assertion for the task card's 512 floor on one montage cell."""
    short = int(min(cell_w, cell_h))
    if short < MONTAGE_CELL_MIN_SHORT_EDGE:
        raise ValueError(
            f"montage cell short edge {short}px < {MONTAGE_CELL_MIN_SHORT_EDGE}px "
            f"({where}; cell {cell_w}x{cell_h})"
        )
    return short


def compose_montage(paths: Sequence[str], *, cell_short_edge: int):
    """The options as one contact sheet with neutral 1..N corner badges.

    ``MONTAGE_GRID`` fixes the shape per option count; every cell is the same size, and
    a grid slot with no image (n=5 in a 2x3) is left as flat background with no badge, so
    it carries no information.  Returns ``(image, meta)``; ``meta['label_to_position']``
    records the row-major reading order the prompt declares.
    """
    from PIL import Image, ImageDraw, ImageOps

    n = len(paths)
    if n not in MONTAGE_GRID:
        raise ValueError(f"no montage grid for {n} options (have {sorted(MONTAGE_GRID)})")
    rows, cols = MONTAGE_GRID[n]
    resampling = getattr(Image, "Resampling", Image)
    cells: list[Any] = []
    for path in paths:
        with Image.open(path) as handle:
            image = ImageOps.exif_transpose(handle).convert("RGB")
        scale = cell_short_edge / min(image.size)
        size = tuple(max(1, int(round(value * scale))) for value in image.size)
        cells.append(image if size == image.size else image.resize(size, resampling.LANCZOS))
    cell_w = max(cell.width for cell in cells)
    cell_h = max(cell.height for cell in cells)
    assert_cell_short_edge(cell_w, cell_h, "composed montage")

    sheet = Image.new(
        "RGB",
        (cols * cell_w + (cols + 1) * MONTAGE_GAP,
         rows * cell_h + (rows + 1) * MONTAGE_GAP),
        MONTAGE_BG,
    )
    draw = ImageDraw.Draw(sheet)
    badge_size = max(28, int(round(min(cell_w, cell_h) / 12)))
    font = _font(badge_size)
    pad = max(6, badge_size // 4)
    label_to_position: dict[str, list[int]] = {}
    for index, cell in enumerate(cells):
        row, col = divmod(index, cols)
        x = MONTAGE_GAP + col * (cell_w + MONTAGE_GAP)
        y = MONTAGE_GAP + row * (cell_h + MONTAGE_GAP)
        sheet.paste(cell, (x + (cell_w - cell.width) // 2, y + (cell_h - cell.height) // 2))
        label = MONTAGE_LABELS[index]
        box = draw.textbbox((0, 0), label, font=font)
        w, h = box[2] - box[0], box[3] - box[1]
        draw.rectangle([x, y, x + w + 2 * pad, y + h + 2 * pad], fill=MONTAGE_BADGE_BG)
        draw.text((x + pad - box[0], y + pad - box[1]), label,
                  font=font, fill=MONTAGE_BADGE_FG)
        label_to_position[label] = [row, col]
    meta = {
        "options": n,
        "cell_size": [cell_w, cell_h],
        "cell_short_edge": int(min(cell_w, cell_h)),
        "montage_size": list(sheet.size),
        "grid": [rows, cols],
        "blank_cells": rows * cols - n,
        "label_to_position": label_to_position,
    }
    return sheet, meta


def encode_pil_data_url(image, *, longest_edge: int, quality: int) -> tuple[str, float]:
    """``responses_vlm.encode_image_data_url`` semantics for an in-memory image.

    Returns the data URL and the scale factor the encoder applied, so the caller can
    re-assert the 512 floor on the pixels that actually leave the process.
    """
    import base64
    import io

    from PIL import Image

    scale = 1.0
    if max(image.size) > longest_edge:
        scale = longest_edge / max(image.size)
        size = tuple(max(1, int(round(value * scale))) for value in image.size)
        resampling = getattr(Image, "Resampling", Image)
        image = image.resize(size, resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + payload, scale


def quadruples(group_id: str, slots: Sequence[str]) -> list[list[str]]:
    """Deterministic 16 -> 4x4 split keyed by ``sha1(group_id, slot)``."""
    ordered = sorted(slots, key=lambda slot: (sha1_hex(group_id, slot), slot))
    return [ordered[i:i + QUAD_SIZE] for i in range(0, len(ordered), QUAD_SIZE)]


def presentation(group_id: str, round_name: str, quad_index: int,
                 item_ids: Sequence[str]) -> list[str]:
    """In-call display order = montage cells 1..N read row-major.

    v3 seeds on ``"present-v3"`` and on generic item ids (slot ids plus ``SOURCE_ITEM``),
    so the source image is shuffled in among the candidates exactly like a candidate.
    The label -> item map is journaled under ``presentation`` for bias analysis.
    """
    return sorted(
        item_ids,
        key=lambda item: (
            sha1_hex(group_id, "present-v3", round_name, quad_index, item), item),
    )


def parse_ranking(text: str, n: int) -> list[int]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("response is not a JSON object")
    ranking = payload.get("ranking")
    if not isinstance(ranking, list) or len(ranking) != n:
        raise ValueError(f"ranking is not a list of {n} positions")
    values = []
    for item in ranking:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError("ranking entries must be integers")
        values.append(int(item))
    if sorted(values) != list(range(1, n + 1)):
        raise ValueError(f"ranking is not a permutation of 1..{n} (ties or gaps)")
    return values


BACKOFF_BASE_S = 2.0
BACKOFF_CAP_S = 30.0


def backoff_delay(attempt: int, rng: random.Random,
                  cap_s: float = BACKOFF_CAP_S,
                  base_s: float = BACKOFF_BASE_S) -> float:
    """Full-jitter exponential backoff: ``U(0, min(cap, base * 2^(attempt-1)))``.

    Local to EPR-049 — ``responses_vlm`` is main-chain and stays untouched.  The jitter
    RNG is a private instance so it cannot perturb the sha-seeded draws elsewhere.
    """
    # cap_s is per-arm: at the default 30s the four attempts span only ~20-27s of
    # wall clock, measured to be far too narrow to escape this relay's degradation
    # windows.  Widening it is transport-only; the stimulus is untouched.
    # NOTE: with base=2 and 4 attempts the ceilings are 2/4/8s, so the CAP is never
    # binding -- raising it alone changes nothing.  base_s is the parameter that
    # actually widens the window.  Both are transport-only and per-arm.
    ceiling = min(cap_s, base_s * (2 ** max(0, attempt - 1)))
    return rng.uniform(0.0, ceiling)


class Relay:
    """One bounded ranking call with parse-level retries on top of transport retries."""

    def __init__(self, config: Epr049Config, failures: Journal):
        self.config = config
        self.failures = failures
        self.endpoints = config.endpoints
        self._secrets = config.secrets
        # SystemRandom: the jitter is drawn from several worker threads at once and must
        # not share (or perturb) the sha-seeded generators the sampling side relies on.
        self._rng = random.SystemRandom()

    def sanitize(self, value: object) -> str:
        from dataset_build.src.construct.config import redact_text

        return redact_text(value, self._secrets)

    def image_url(self, path: str, cache: dict[str, str]) -> str:
        """Encode once per group: the source image appears in all five calls.

        The cache is owned by ``tournament`` and dies with the group, so a hundred groups
        never accumulate a hundred groups' worth of base64 in memory.
        """
        from dataset_build.core.responses_vlm import encode_image_data_url

        cached = cache.get(path)
        if cached is None:
            cached = encode_image_data_url(
                path, longest_edge=self.config.image_long_edge,
                quality=self.config.image_jpeg_quality,
            )
            cache[path] = cached
        return cached

    def montage_url(self, option_paths: Sequence[str]) -> tuple[str, dict[str, Any]]:
        """The contact sheet, with the 512 floor asserted before and after encoding."""
        sheet, meta = compose_montage(
            option_paths, cell_short_edge=self.config.montage_cell_short_edge
        )
        url, scale = encode_pil_data_url(
            sheet, longest_edge=self.config.montage_long_edge,
            quality=self.config.montage_jpeg_quality,
        )
        cell_w, cell_h = meta["cell_size"]
        shipped_w = int(round(cell_w * scale))
        shipped_h = int(round(cell_h * scale))
        assert_cell_short_edge(shipped_w, shipped_h, "montage after encode scaling")
        meta["encode_scale"] = round(scale, 6)
        meta["encoded_montage_size"] = [
            int(round(meta["montage_size"][0] * scale)),
            int(round(meta["montage_size"][1] * scale)),
        ]
        meta["encoded_cell_size"] = [shipped_w, shipped_h]
        meta["encoded_cell_short_edge"] = min(shipped_w, shipped_h)
        meta["montage_bytes"] = len(url)
        return url, meta

    def rank(self, *, group_id: str, call_id: str, source_path: str,
             option_paths: Sequence[str], source_is_an_option: bool,
             cache: dict[str, str]) -> dict[str, Any]:
        from dataset_build.core.responses_vlm import ResponsesVlmError, request_text

        n = len(option_paths)
        endpoint = self.endpoints[sha1_int(group_id, call_id) % len(self.endpoints)]
        # Image work (open / EXIF / resize / compose / assert / base64) happens before any
        # attempt and used to escape this method: an OSError on one asset took down the
        # whole thread pool.  It is now a failure of this one call.
        try:
            source_url = self.image_url(source_path, cache)
            montage_url, montage_meta = self.montage_url(option_paths)
        except Exception as exc:  # noqa: BLE001 - one bad asset fails one call, not the run
            error = f"image:{type(exc).__name__}:{self.sanitize(exc)[:200]}"
            self.failures.append({
                "schema": SCHEMA, "group_id": group_id, "call_id": call_id,
                "attempt": 0, "endpoint_id": endpoint.id, "error": error, "ts": time.time(),
            })
            return {
                "ok": False, "attempts": 0, "errors": [error], "options": n,
                "endpoint_id": endpoint.id, "model_substituted": 0, "montage": None,
            }
        # Exactly two images: the input photo, then the one contact sheet.
        content: list[dict[str, Any]] = [
            {"type": "input_text",
             "text": build_prompt(n, source_is_an_option=source_is_an_option)},
            {"type": "input_image", "image_url": source_url},
            {"type": "input_image", "image_url": montage_url},
        ]
        errors: list[str] = []
        substituted = 0
        max_attempts = max(1, self.config.transport_attempts)
        for attempt in range(1, max_attempts + 1):
            try:
                result = request_text(
                    base_url=endpoint.base_url,
                    api_key=endpoint.api_key,
                    model=self.config.external_model,
                    content=content,
                    schema_name="epr049_aesthetic_ranking",
                    schema=ranking_schema(n),
                    timeout=180.0,
                    temperature=0.1,
                    max_output_tokens=self.config.max_output_tokens,
                    attempts=1,
                    vgate_class="qa-judge",
                    reasoning_effort=self.config.reasoning_effort,
                    expect_model=self.config.external_model,
                )
                ranking = parse_ranking(result.text, n)
                return {
                    "ok": True,
                    "attempts": attempt,
                    "options": n,
                    "ranking_positions": ranking,
                    "response_raw": result.text,
                    "returned_model": result.model,
                    "endpoint_id": endpoint.id,
                    "model_substituted": substituted,
                    "errors": errors,
                    "montage": montage_meta,
                }
            except ResponsesVlmError as exc:
                if exc.error_type == "ModelSubstituted":
                    substituted += 1
                errors.append(f"transport:{exc.error_type}")
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                errors.append(f"parse:{type(exc).__name__}:{self.sanitize(exc)[:200]}")
            delay = (backoff_delay(attempt, self._rng, self.config.backoff_cap_s,
                                   self.config.backoff_base_s)
                     if attempt < max_attempts else 0.0)
            self.failures.append({
                "schema": SCHEMA, "group_id": group_id, "call_id": call_id,
                "attempt": attempt, "endpoint_id": endpoint.id, "error": errors[-1],
                "backoff_s": round(delay, 3), "ts": time.time(),
            })
            if delay > 0.0:
                time.sleep(delay)
        return {
            "ok": False, "attempts": max_attempts, "errors": errors, "options": n,
            "endpoint_id": endpoint.id, "model_substituted": substituted,
            "montage": montage_meta,
        }


def tournament(relay: Relay, stage: Mapping[str, Any], out_root: Path) -> dict[str, Any]:
    """v3: four 5-way prelims (4 candidates + the source), then a final over survivors."""
    group_id = str(stage["group_id"])
    by_slot = {row["slot_id"]: row for row in stage["candidates"]}
    source_path = str(out_root / stage["source_asset"])
    cache: dict[str, str] = {}

    def path_of(item: str) -> str:
        return source_path if item == SOURCE_ITEM else str(out_root / by_slot[item]["asset"])

    rounds1: list[dict[str, Any]] = []
    finalists: list[str] = []
    source_wins: list[int] = []
    failed: list[str] = []
    for quad_index, quad in enumerate(quadruples(group_id, list(by_slot))):
        # The source competes inside the quadruple: 5 options, shuffled together.
        options = list(quad) + [SOURCE_ITEM]
        shown = presentation(group_id, "r1", quad_index, options)
        call_id = f"r1q{quad_index}"
        result = relay.rank(
            group_id=group_id, call_id=call_id, source_path=source_path,
            option_paths=[path_of(item) for item in shown],
            source_is_an_option=True, cache=cache,
        )
        row = {
            "call_id": call_id,
            "quad_index": quad_index,
            "quad_slots": quad,
            "options": shown,
            "presentation": {str(i + 1): item for i, item in enumerate(shown)},
            "presentation_slots": shown,
            # Cheap shuffle guard: which label the source landed on.  Pooled over the run
            # this must be uniform over 1..N; a skew means the shuffle or the label ->
            # item mapping is broken, which would masquerade as a real source preference.
            "source_label": shown.index(SOURCE_ITEM) + 1,
            **{key: value for key, value in result.items() if key != "ranking_positions"},
        }
        if result["ok"]:
            ranked = [shown[position - 1] for position in result["ranking_positions"]]
            row["ranking_positions"] = result["ranking_positions"]
            row["ranking_slots"] = ranked
            row["rank1"] = ranked[0]
            row["source_won"] = ranked[0] == SOURCE_ITEM
            row["source_rank"] = ranked.index(SOURCE_ITEM) + 1
            if row["source_won"]:
                source_wins.append(quad_index)
            else:
                finalists.append(ranked[0])
        else:
            failed.append(call_id)
        rounds1.append(row)

    prelims_ok = sum(1 for row in rounds1 if row["ok"])
    round2: dict[str, Any] | None = None
    final_note: str | None = None
    if prelims_ok != PRELIM_CALLS_PER_GROUP:
        final_note = "final_skipped:incomplete_prelims"
        failed.append("r2:skipped_incomplete_round1")
    elif len(finalists) >= MIN_FINAL_OPTIONS:
        # The source does not compete in the final; only surviving candidates do.
        shown = presentation(group_id, "r2", 0, finalists)
        result = relay.rank(
            group_id=group_id, call_id="r2", source_path=source_path,
            option_paths=[path_of(item) for item in shown],
            source_is_an_option=False, cache=cache,
        )
        round2 = {
            "call_id": "r2",
            "finalists": finalists,
            "options": shown,
            "presentation": {str(i + 1): item for i, item in enumerate(shown)},
            "presentation_slots": shown,
            **{key: value for key, value in result.items() if key != "ranking_positions"},
        }
        if result["ok"]:
            ranked = [shown[position - 1] for position in result["ranking_positions"]]
            round2["ranking_positions"] = result["ranking_positions"]
            round2["ranking_slots"] = ranked
            round2["top2"] = ranked[:2]
        else:
            failed.append("r2")
    elif len(finalists) == 1:
        final_note = "final_degenerate:single_finalist"
        round2 = {
            "call_id": None, "finalists": list(finalists), "ok": True, "attempts": 0,
            "model_substituted": 0, "ranking_slots": list(finalists),
            "top2": list(finalists), "degenerate": True,
        }
    else:
        final_note = "final_skipped:all_source_won"

    calls_expected = PRELIM_CALLS_PER_GROUP + (
        1 if (prelims_ok == PRELIM_CALLS_PER_GROUP
              and len(finalists) >= MIN_FINAL_OPTIONS) else 0)
    calls_made = prelims_ok + (
        1 if (round2 is not None and not round2.get("degenerate")
              and round2.get("ok")) else 0)
    return {
        "model": relay.config.external_model,
        "round1": rounds1,
        "round2": round2,
        "calls_expected": calls_expected,
        "calls_parsed": calls_made,
        "prelims_parsed": prelims_ok,
        "finalists": list(finalists),
        "n_finalists": len(finalists),
        "source_won_quads": source_wins,
        "n_source_won": len(source_wins),
        "final_note": final_note,
        "failed_calls": failed,
        "attempts_total": sum(row["attempts"] for row in rounds1)
        + (round2["attempts"] if round2 is not None else 0),
        "model_substituted": sum(row.get("model_substituted", 0) for row in rounds1)
        + (round2.get("model_substituted", 0) if round2 is not None else 0),
    }


def run_group(relay: Relay, stage: Mapping[str, Any], out_root: Path) -> dict[str, Any]:
    """``tournament`` with a group-wide exception boundary.

    A ``KeyError`` from ``by_slot``, an ``OSError`` from an unreadable asset or any other
    surprise used to propagate out of the worker and abort the whole pool at
    ``future.result()``.  Here it becomes one failure row plus one incomplete group.
    """
    group_id = str(stage.get("group_id"))
    try:
        return tournament(relay, stage, out_root)
    except Exception as exc:  # noqa: BLE001 - one group must never kill the round
        error = f"group:{type(exc).__name__}:{relay.sanitize(exc)[:300]}"
        relay.failures.append({
            "schema": SCHEMA, "group_id": group_id, "call_id": "group",
            "attempt": 0, "endpoint_id": None, "error": error, "ts": time.time(),
        })
        return {
            "model": relay.config.external_model,
            "round1": [],
            "round2": None,
            "calls_expected": PRELIM_CALLS_PER_GROUP,
            "calls_parsed": 0,
            "prelims_parsed": 0,
            "finalists": [],
            "n_finalists": 0,
            "source_won_quads": [],
            "n_source_won": 0,
            "final_note": "final_skipped:group_error",
            "failed_calls": [f"group_error:{type(exc).__name__}"],
            "attempts_total": 0,
            "model_substituted": 0,
            "group_error": error,
        }


# --------------------------------------------------------------------------------- main


def restrict(scores: Mapping[str, float], slots: Sequence[str]) -> list[str]:
    """Slots ordered by OneAlign score; slots whose score came back null are dropped."""
    known = [slot for slot in slots if slot in scores]
    return sorted(known, key=lambda slot: (-float(scores[slot]), slot))


def tournament_of(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """One arm's tournament block, under either the new or the pre-split key."""
    block = row.get(TOURNAMENT_KEY)
    if block is None:
        block = row.get("gemini")
    if block is None:
        raise KeyError(f"group {row.get('group_id')!r} has no tournament block")
    return block


def latest_by_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Append-only journal -> one row per group, the last line winning.

    A group re-run after an ``incomplete`` verdict appends a second line; every reader
    (resume, stats, questionnaire) must see only the newest one.
    """
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        latest[str(row["group_id"])] = dict(row)
    return latest


def merge_group(row: Mapping[str, Any], gemini: Mapping[str, Any],
                *, arm: str = LEGACY_ARM) -> dict[str, Any]:
    """One stage row + its v3 tournament -> the journal row.

    A group is ``complete`` when all four prelims parsed, the final resolved one of its
    three legitimate ways (ranked / degenerate single finalist / skipped because every
    quadruple went to the source), and OneAlign produced no nulls.
    """
    onealign = row["onealign"]
    scores = dict(onealign["scores"])
    null_count = int(onealign.get("null_count", 0))
    # v3: the source is a competitor, so OneAlign is restricted over candidates + source.
    source_score = onealign.get("source_score")
    if source_score is not None:
        scores[SOURCE_ITEM] = source_score
    finalists = list(gemini.get("finalists") or [])
    reasons = list(gemini["failed_calls"])
    prelims_ok = int(gemini.get("prelims_parsed", 0)) == PRELIM_CALLS_PER_GROUP
    final_ok = (
        gemini.get("round2") is not None
        and (gemini["round2"].get("ok") or gemini["round2"].get("degenerate"))
    ) or gemini.get("final_note") == "final_skipped:all_source_won"
    if not prelims_ok:
        reasons.append(f"prelims_parsed:{gemini.get('prelims_parsed')}")
    if not final_ok and gemini.get("final_note"):
        reasons.append(str(gemini["final_note"]))
    if null_count:
        reasons.append(f"onealign_null:{null_count}")
    complete = prelims_ok and final_ok and null_count == 0
    merged = {
        **{key: value for key, value in row.items() if key != "ts"},
        "arm": arm,
        "model": gemini.get("model"),
        TOURNAMENT_KEY: dict(gemini),
        "onealign_restricted": {
            # Each prelim's option set is the quadruple plus the source.
            "round1_options": {
                str(entry["quad_index"]): restrict(
                    scores, list(entry["quad_slots"]) + [SOURCE_ITEM])
                for entry in gemini["round1"]
            },
            "finalists": restrict(scores, finalists) if finalists else [],
            "source_score": source_score,
        },
        "status": "complete" if complete else "incomplete",
        "incomplete_reason": None if complete else reasons,
        "ts": time.time(),
    }
    if arm == LEGACY_ARM:
        merged["gemini"] = merged[TOURNAMENT_KEY]
    return merged


def load_or_plan(config: Epr049Config, majors: Sequence[str],
                 tree: Mapping[str, Mapping[str, Sequence[str]]]) -> tuple[list[dict], dict]:
    """The group plan, cached on disk.

    ``build_inventory`` walks the whole 56k-entry source pool; the plan it feeds is a pure
    function of ``build_id``/``seed``/``target_groups``/the eligible pool, so a resumed run
    reads it back instead of paying for the walk again.  The cache is invalidated by any
    change to those keys, which is what makes ``--smoke N`` and the later full run share
    one plan rather than two.
    """
    from dataset_build.src.construct.sources import build_inventory

    signature = {
        "schema": SCHEMA, "build_id": config.build_id, "seed": config.seed,
        "target_groups": config.target_groups,
        "candidates_per_group": config.candidates_per_group,
        "subject_cache": config.subject_cache, "majors": list(majors),
        "split_seed": SPLIT_SEED, "train_bucket_max": TRAIN_BUCKET_MAX,
    }
    cache = config.output_root / "plan.json"

    def semantic(sig: Mapping[str, Any]) -> dict[str, Any]:
        """Signature fields the plan actually depends on.

        ``schema`` is carried for provenance only.  It used to be compared too, so
        bumping the tournament schema (v1 -> v3) silently invalidated this cache and
        forced another full ``build_inventory`` walk of the 56k source pool -- six
        minutes per run, and a needless risk of drawing from a changed pool.  The plan
        is a function of build_id / seed / target_groups / candidates / source pool /
        split rule only.
        """
        return {key: value for key, value in sig.items() if key != "schema"}

    if cache.is_file():
        payload = json.loads(cache.read_text(encoding="utf-8"))
        if semantic(payload.get("signature") or {}) == semantic(signature):
            return payload["plans"], payload["inventory"]
    inventory = build_inventory(config.subject_cache, config.postgres_dsn)
    plans = plan_groups(config, inventory.eligible, majors, tree)
    summary = {
        "eligible_sources": len(inventory.eligible),
        "scene_metadata_status": inventory.scene_metadata_status,
        "counts": dict(inventory.counts),
        "train_split_sources": sum(
            1 for row in inventory.eligible if split_bucket(row.source_id) <= TRAIN_BUCKET_MAX
        ),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.name + ".tmp")
    tmp.write_text(
        json.dumps({"signature": signature, "inventory": summary, "plans": plans},
                   ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    os.replace(tmp, cache)
    return plans, summary


def build(config: Epr049Config, *, smoke: int | None, gemini_workers: int) -> dict[str, Any]:
    out_root = config.output_root
    out_root.mkdir(parents=True, exist_ok=True)
    # Phase A is shared across arms by design: same 16 candidates, same OneAlign
    # scores, same quadruple split.  Only the tournament journals are per-arm.
    stage_journal = Journal(out_root / "stage_render_qa.jsonl")
    groups_journal = Journal(config.journal_v3("groups", "jsonl"))
    failures = Journal(config.journal_v3("failures", "jsonl"))

    renderer = GroupRenderer(config)
    tree = qualified_majors(renderer.catalog, config.candidates_per_group)
    if len(tree) < 8:
        raise SystemExit(
            f"only {len(tree)} majors hold >= {config.candidates_per_group} distinct LUT "
            "presets; the task card forbids silently degrading to 8 candidates"
        )
    majors = sorted(tree)

    plans, inventory = load_or_plan(config, majors, tree)
    wanted = plans if smoke is None else plans[:smoke]

    done_stage = latest_by_group(stage_journal.read())
    # Resume at group granularity: only a group whose newest journal line says
    # ``complete`` is skipped.  An ``incomplete`` group is re-run and its new line
    # supersedes the old one (``latest_by_group``), so no relay call is paid for twice
    # and no failed group is frozen into the output.
    done_groups = {
        group_id for group_id, row in latest_by_group(groups_journal.read()).items()
        if row.get("status") == "complete"
    }

    # Phase A (serial, GPU): render + OneAlign. Journaled before any relay call so a
    # relay outage cannot destroy finished GPU work.
    stages: list[dict[str, Any]] = []
    for plan in wanted:
        group_id = plan["group_id"]
        if group_id in done_stage:
            stages.append(done_stage[group_id])
            continue
        rendered = renderer.render_group(plan, out_root)
        onealign = renderer.score_group(
            out_root / rendered["source_asset"], rendered["candidates"], out_root
        )
        row = {
            "schema": SCHEMA,
            "build_id": config.build_id,
            "group_id": group_id,
            "group_index": plan["group_index"],
            "major": plan["major"],
            "source": {
                "source_id": plan["source_id"],
                "source_path": plan["source_path"],
                "scene": plan["scene"],
                "split_bucket": plan["split_bucket"],
                "size": rendered["source_size"],
            },
            "source_asset": rendered["source_asset"],
            "candidates": rendered["candidates"],
            "onealign": onealign,
            "ts": time.time(),
        }
        stage_journal.append(row)
        stages.append(row)

    # Phase B (concurrent): the gemini tournament.  Each group is appended + fsynced the
    # moment it finishes (``as_completed``), at the same granularity as Phase A, so a
    # crash at group 73 keeps groups 1..72 instead of losing the whole pool's work.
    relay = Relay(config, failures)
    pending = [row for row in stages if str(row["group_id"]) not in done_groups]
    workers = max(1, min(gemini_workers, sum(e.concurrency for e in config.endpoints)))
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(run_group, relay, row, out_root): row for row in pending
            }
            for future in as_completed(futures):
                groups_journal.append(
                    merge_group(futures[future], future.result(), arm=config.arm))

    rows = list(latest_by_group(groups_journal.read()).values())
    wanted_ids = {plan["group_id"] for plan in wanted}
    subset = sorted(
        (row for row in rows if row["group_id"] in wanted_ids),
        key=lambda row: str(row["group_id"]),
    )
    complete = [row for row in subset if row["status"] == "complete"]
    all_scores = [
        value for row in subset for value in row["onealign"]["scores"].values()
    ]
    source_scores = [
        row["onealign"]["source_score"] for row in subset
        if row["onealign"].get("source_score") is not None
    ]
    return {
        "schema": SCHEMA,
        "build_id": config.build_id,
        "arm": config.arm,
        "transport_attempts": config.transport_attempts,
        "backoff_cap_s": config.backoff_cap_s,
        "backoff_base_s": config.backoff_base_s,
        "output_root": str(out_root),
        "groups_journal": str(config.journal_v3("groups", "jsonl")),
        "model": config.external_model,
        "smoke": smoke,
        "majors_with_enough_lut_presets": len(tree),
        "majors": majors,
        "inventory": inventory,
        "planned_groups": len(wanted),
        "groups_written": len(subset),
        "groups_complete": len(complete),
        "groups_incomplete": len(subset) - len(complete),
        "candidates_per_group": sorted({len(row["candidates"]) for row in subset}),
        "onealign_scores_written": len(all_scores),
        "onealign_score_min": min(all_scores) if all_scores else None,
        "onealign_score_max": max(all_scores) if all_scores else None,
        "onealign_source_score_min": min(source_scores, default=None),
        "onealign_source_score_max": max(source_scores, default=None),
        "onealign_null_scores": sum(
            int(row["onealign"].get("null_count", 0)) for row in subset),
        "onealign_null_slots": {
            str(row["group_id"]): list(row["onealign"].get("null_slots") or [])
            for row in subset if row["onealign"].get("null_slots")
        },
        "onealign_groups_with_null": sum(
            1 for row in subset if int(row["onealign"].get("null_count", 0)) > 0),
        "gemini_calls_expected": sum(
            tournament_of(row)["calls_expected"] for row in subset),
        "prelim_calls_expected": len(subset) * PRELIM_CALLS_PER_GROUP,
        "prelim_calls_parsed": sum(
            int(tournament_of(row).get("prelims_parsed", 0)) for row in subset),
        "source_won_quads_total": sum(
            int(tournament_of(row).get("n_source_won", 0)) for row in subset),
        "source_won_rate_over_quads": (
            round(sum(int(tournament_of(row).get("n_source_won", 0)) for row in subset)
                  / (len(subset) * PRELIM_CALLS_PER_GROUP), 6) if subset else None),
        "finalist_count_histogram": _finalist_histogram(subset),
        "source_label_distribution": _source_label_distribution(subset),
        "final_note_counts": _final_note_counts(subset),
        "gemini_calls_parsed": sum(tournament_of(row)["calls_parsed"] for row in subset),
        "gemini_attempts_total": sum(tournament_of(row)["attempts_total"] for row in subset),
        "gemini_attempt_histogram": _attempt_histogram(subset),
        "gemini_model_substituted": sum(tournament_of(row)["model_substituted"] for row in subset),
        "gemini_group_errors": sum(
            1 for row in subset if tournament_of(row).get("group_error")),
        "montage": _montage_stats(subset),
        "failure_rows": len(failures.read()),
    }


def _source_label_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Where the source image landed among the shuffled prelim labels.

    Must be uniform over 1..PRELIM_OPTIONS.  A skew here would look exactly like a real
    preference for (or against) the source, so it is reported next to the win rate.
    """
    counts: dict[str, int] = {}
    for row in rows:
        for entry in tournament_of(row)["round1"]:
            label = entry.get("source_label")
            if label is not None:
                counts[str(label)] = counts.get(str(label), 0) + 1
    total = sum(counts.values())
    expected = total / PRELIM_OPTIONS if total else 0.0
    chi2 = (sum((counts.get(str(k), 0) - expected) ** 2 / expected
                for k in range(1, PRELIM_OPTIONS + 1)) if expected else None)
    return {
        "counts": dict(sorted(counts.items())),
        "n": total,
        "expected_per_label": round(expected, 2),
        "chi2_df4": round(chi2, 3) if chi2 is not None else None,
        "chi2_crit_p05_df4": 9.488,
    }


def _finalist_histogram(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """How many candidates survived the prelims per group (0..4)."""
    counts: dict[str, int] = {}
    for row in rows:
        key = str(int(tournament_of(row).get("n_finalists", 0)))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _final_note_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        note = tournament_of(row).get("final_note")
        if note:
            counts[str(note)] = counts.get(str(note), 0) + 1
    return dict(sorted(counts.items()))


def _montage_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Measured montage geometry, so the 512 floor is visible in build_stats.json."""
    cells: list[int] = []
    sizes: set[tuple[int, int]] = set()
    scales: set[float] = set()
    options: collections_Counter = collections_Counter()
    for row in rows:
        block = tournament_of(row)
        calls = list(block["round1"])
        if block.get("round2"):
            calls.append(block["round2"])
        for call in calls:
            if call.get("degenerate"):
                continue
            meta = call.get("montage")
            if not meta:
                continue
            cells.append(int(meta["encoded_cell_short_edge"]))
            options[str(meta.get("options"))] += 1
            sizes.add(tuple(meta["encoded_montage_size"]))
            scales.add(float(meta["encode_scale"]))
    return {
        "calls_with_montage": len(cells),
        "cell_short_edge_min": min(cells) if cells else None,
        "cell_short_edge_max": max(cells) if cells else None,
        "floor": MONTAGE_CELL_MIN_SHORT_EDGE,
        "encoded_sizes": sorted(list(size) for size in sizes),
        "encode_scales": sorted(scales),
        "calls_by_option_count": dict(sorted(options.items())),
    }


def _attempt_histogram(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        block = tournament_of(row)
        calls = list(block["round1"])
        if block.get("round2"):
            calls.append(block["round2"])
        for call in calls:
            if call.get("degenerate"):
                continue
            key = str(call["attempts"])
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("experiments/prs/EPR-049_gemini-vs-onealign-aesthetic/epr049.toml"),
    )
    parser.add_argument("--smoke", type=int, default=None,
                        help="build only the first N groups of the same deterministic list")
    parser.add_argument("--gemini-workers", type=int, default=8)
    parser.add_argument("--model-arm", default=None,
                        help="which [annotation.arms.<name>] to run "
                             "(default: annotation.default_arm)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config, args.model_arm)
    stats = build(config, smoke=args.smoke, gemini_workers=args.gemini_workers)
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    config.journal_v3("build_stats", "json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
