"""Deterministic LUT retrieval and frozen Mask v2 candidate construction."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2lab, rgb2hsv

from .artifacts import ArtifactStore
from .config import CatalogConfig
from .direction_match import (
    FORBIDDEN_CAPTION_ALIASES, SATURATION_DROP_WORDS, SATURATION_LIFT_WORDS,
    MaskDirection, SegmentFingerprintTable, combined_direction_scores,
    direction_match_scores,
)
from .lut_annotations import file_sha256, validate_closed_annotation
from .segment_fingerprints import (
    HISTOGRAM_AGGREGATES, load_segment_fingerprints, load_segment_histograms,
)
from .source_histogram import histogram_match_bonus
from .models import (
    ACTIVE_LOCAL_INTENTS, BAND_GEOMETRY_GATE, DISABLED_LOCAL_INTENTS,
    FINGERPRINT_FIELDS,
    INTENT_BIN_QUOTA, INTENT_DIRECTION_GATE, INTENT_FINGERPRINT_GATE,
    INTENT_SEGMENT_GATE, INTENT_V1_VARIANTS,
    LOCAL_INTENTS, LOCAL_ONLINE_RETRIEVAL, LOCAL_PACKET_ROW_LIMIT, MASK_REACH_GATE,
    SUBJECT_HEADROOM_GATE, intent_packet_order, intent_serves_role, local_ladder,
)

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib


class CandidateError(RuntimeError):
    pass


GLOBAL_BIN_ORDER: tuple[str, ...] = ("natural", "medium", "bold")
LOCAL_BIN_ORDER: tuple[str, ...] = ("subtle", "natural", "strong")
GLOBAL_BIN_QUOTA = 2
LOCAL_BIN_QUOTA = 3
# `INTENT_BIN_QUOTA` (one row per local strength bin inside a packet, then score fill
# up to `LOCAL_PACKET_ROW_LIMIT` rows) moved to models.py so the prompt revision
# fingerprint can read it without importing this module. Re-exported for callers.

# Mask v2 spatial roles (DECISIONS_agent_loop_local_intent_optB_20260819 §2-C1 / §4-B1).
# `subject` keeps the frozen Mask v2 gate; `background` is its mirror image.
MASK_ROLES: tuple[str, ...] = ("subject", "background")
# B6 item 2: default sibling role mix of one local packet.
ROLE_PACKET_TARGET: dict[str, int] = {"subject": 2, "background": 1}
BACKGROUND_ROLE_GATE: dict[str, float] = {
    "subject_alpha_mean_max": 0.15,
    "subject_high_coverage_max": 0.02,
    "background_alpha_mean_min": 0.35,
    # B8 item 3 (R4.2, user-approved 2026-08-20): 0.12 -> 0.20.
    "half_area_min": 0.20,
    # B6 item 5: pre-registered visibility floor applied to the *calibrated* alpha
    # (alpha x strength) of a background local render, so a strength small enough to
    # make the background edit invisible is rejected instead of silently accepted.
    "applied_background_alpha_mean_min": 0.18,
}
# B7 item 4: subject-role band masks must also cover a minimum share of the frame
# (`half_area`, the fraction of pixels with alpha >= 0.5). Band slots that miss it are
# dropped and counted through the existing slot-drop note channel; radial, linear and
# semantic subject masks are untouched.
SUBJECT_BAND_GATE: dict[str, float] = {"half_area_min": 0.28}
BACKGROUND_CENTER_HINT = "the background around the main subject"
# `linear` was dropped from the background pool after the 100-source pilot review
# (edge ramps read as an abrupt overlay); radial and band only.
BACKGROUND_FAMILY_COUNTS: dict[str, int] = {"radial": 2, "band": 2}


def _number(summary: Mapping[str, Any], key: str) -> float:
    value = summary.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


@dataclass(frozen=True, slots=True)
class LutRecord:
    preset_id: str
    path: str
    format: str
    name: str
    style_major: str
    style_minor: str
    scene_affinity: tuple[str, ...]
    de_med: float
    caption: str
    per_probe: dict[str, str]
    hsl_features: dict[str, Any]
    # B10 (R6.1): shadows/mids/highlights x (dL, dC, d_hue, cast_a, cast_b), mounted
    # from the derived `segment_fingerprints` artifact when the catalog config points at
    # one. `None` means "not mounted"; nothing in the prompt or the shortlist reads it.
    segment_fingerprint: dict[str, dict[str, float]] | None = None
    # B12 item 2/3: the LUT's L* bin-share response, mounted from a v2 fingerprint
    # artifact. `None` means "not mounted" (a v1 artifact), and every B12 column and
    # scoring term degrades to the pre-B12 behaviour when it is missing.
    histogram_response: dict[str, Any] | None = None

    def fingerprint(self) -> dict[str, float]:
        """The frozen eight-number effect fingerprint (decisions doc section 2.1)."""
        summary = self.hsl_features.get("summary") or {}
        mid_a = _number(summary, "mid_gray_a")
        mid_b = _number(summary, "mid_gray_b")
        return {
            "dL": round(_number(summary, "mid_gray_dL"), 1),
            "contrast": round(_number(summary, "contrast_ratio"), 3),
            "shadow_dL": round(_number(summary, "shadow_dL"), 1),
            "highlight_dL": round(_number(summary, "highlight_dL"), 1),
            "cast_hue": float(round(math.degrees(math.atan2(mid_b, mid_a)))),
            "cast_mag": round(math.hypot(mid_a, mid_b), 1),
            "dSat": round(_number(summary, "sat_pct_mean"), 1),
            "hue_rot": round(_number(summary, "hue_rot_abs_max"), 1),
        }

    def direction_vector(self) -> tuple[float, float, float, float]:
        """(dL, mid_gray_a, mid_gray_b, dSat) used by the audit direction cosine."""
        summary = self.hsl_features.get("summary") or {}
        return (
            _number(summary, "mid_gray_dL"), _number(summary, "mid_gray_a"),
            _number(summary, "mid_gray_b"), _number(summary, "sat_pct_mean"),
        )

    def histogram_aggregates(self) -> dict[str, float] | None:
        """`d_shadow` / `d_mid` / `d_high`, or `None` when nothing is mounted."""
        if not self.histogram_response:
            return None
        return {
            name: float(self.histogram_response.get(name, 0.0))
            for name in HISTOGRAM_AGGREGATES
        }

    def prompt_view(self) -> dict[str, Any]:
        view = {
            "preset_id": self.preset_id,
            "style_major": self.style_major,
            "fingerprint": self.fingerprint(),
            "caption": self.caption,
        }
        # B12 item 3: the three histogram aggregates are the only part of the v2 group
        # that is serialized into the shortlist table.
        aggregates = self.histogram_aggregates()
        if aggregates is not None:
            view["histogram"] = aggregates
        return view


def _format_of(row: Mapping[str, Any]) -> str:
    fmt = str(row.get("fmt") or "").lower()
    kind = str(row.get("kind") or "").lower()
    suffix = Path(str(row.get("path") or "")).suffix.lower()
    if fmt in {"cube", "3dl"} or kind in {"lut", "cube"} or suffix in {".cube", ".3dl"}:
        return "lut"
    if fmt == "xmp" or suffix == ".xmp":
        return "xmp"
    if fmt == "lrtemplate" or suffix == ".lrtemplate":
        return "lrtemplate"
    return ""


def load_cluster_map(path: str | Path) -> dict[str, str]:
    """Read the offline effect-cluster artifact (JSONL: preset_id/style_major/cluster_id)."""
    clusters: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            preset_id = str(row.get("preset_id") or "")
            cluster_id = row.get("cluster_id")
            if not preset_id or cluster_id is None:
                raise CandidateError(f"invalid cluster artifact row: {row!r}")
            clusters[preset_id] = f"{row.get('style_major') or ''}\x1f{cluster_id}"
    if not clusters:
        raise CandidateError("cluster artifact is empty")
    return clusters


class LutCatalog:
    def __init__(
        self, records: Iterable[LutRecord], clusters: Mapping[str, str] | None = None,
        segment_fingerprints_sha256: str = "",
    ) -> None:
        self.records = tuple(sorted(records, key=lambda row: row.preset_id))
        self.by_id = {row.preset_id: row for row in self.records}
        if len(self.by_id) != len(self.records):
            raise CandidateError("duplicate LUT annotation IDs")
        self.clusters = dict(clusters or {})
        # B11 item 6: SHA-256 of the mounted segment-fingerprint artifact, empty when
        # nothing is mounted. Reported into the branch audit, never into the prompt.
        self.segment_fingerprints_sha256 = str(segment_fingerprints_sha256 or "")
        self._fingerprint_table: SegmentFingerprintTable | None = None

    def cluster_id(self, record: LutRecord) -> str:
        """Cluster key; without a wired artifact every preset is its own cluster."""
        return self.clusters.get(record.preset_id) or record.preset_id

    @property
    def segment_fingerprints_mounted(self) -> bool:
        return all(row.segment_fingerprint is not None for row in self.records)

    @property
    def histogram_responses_mounted(self) -> bool:
        """B12 item 3: true only when every record carries a v2 histogram group."""
        return all(row.histogram_response is not None for row in self.records)

    def segment_fingerprint_table(self) -> SegmentFingerprintTable:
        """The (N, 3, 5) fingerprint table of this catalog; built once, then reused."""
        if not self.segment_fingerprints_mounted:
            raise CandidateError("segment fingerprints are not mounted on this catalog")
        if self._fingerprint_table is None:
            self._fingerprint_table = SegmentFingerprintTable.from_records(self.records)
        return self._fingerprint_table

    @classmethod
    def load(cls, config: CatalogConfig, databuild_config: Path) -> "LutCatalog":
        with databuild_config.open("rb") as handle:
            build = tomllib.load(handle)
        preset_table = build.get("presets") or {}
        bank_dir = Path(str(preset_table.get("bank_dir") or ""))
        features: dict[str, tuple[str, str]] = {}
        with (bank_dir / "features.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                preset_id = str(row.get("preset_id") or "")
                fmt = _format_of(row)
                path = str(row.get("path") or "")
                if preset_id and fmt == "lut" and Path(path).is_file():
                    features[preset_id] = (path, fmt)
        fingerprint_path = getattr(config, "segment_fingerprints", None)
        fingerprints = (
            load_segment_fingerprints(fingerprint_path) if fingerprint_path else {}
        )
        # B12 item 3: a v2 artifact also carries the histogram response; a v1 artifact
        # returns `{}` here and the catalog stays exactly as it was before B12.
        histograms = (
            load_segment_histograms(fingerprint_path) if fingerprint_path else {}
        )
        records: list[LutRecord] = []
        with config.annotations.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                preset_id = str(row.get("preset_id") or row.get("key") or "")
                feature = features.get(preset_id)
                if not row.get("ok") or feature is None:
                    continue
                try:
                    validate_closed_annotation(row)
                except ValueError as exc:
                    raise CandidateError(
                        f"invalid closed LUT annotation for {preset_id or '<missing>'}: {exc}"
                    ) from exc
                major = str(row.get("style_major") or "").strip()
                minor = str(row.get("style_minor") or "").strip()
                if not major or not minor:
                    continue
                segment_fingerprint: dict[str, dict[str, float]] | None = None
                if fingerprints:
                    if preset_id not in fingerprints:
                        raise CandidateError(
                            f"segment fingerprint artifact does not cover {preset_id}"
                        )
                    segment_fingerprint = {
                        segment: dict(values)
                        for segment, values in fingerprints[preset_id].items()
                    }
                histogram_response: dict[str, Any] | None = None
                if histograms:
                    if preset_id not in histograms:
                        raise CandidateError(
                            f"histogram fingerprint artifact does not cover {preset_id}"
                        )
                    histogram_response = dict(histograms[preset_id])
                records.append(LutRecord(
                    preset_id=preset_id, path=feature[0], format=feature[1],
                    name=str(row.get("name") or preset_id), style_major=major,
                    style_minor=minor,
                    scene_affinity=tuple(str(item) for item in row["scene_affinity"]),
                    de_med=float(row["de_med"]),
                    caption=str(row.get("caption") or ""),
                    per_probe={str(k): str(v) for k, v in (row.get("per_probe") or {}).items()},
                    hsl_features=dict(row.get("hsl_features") or {}),
                    segment_fingerprint=segment_fingerprint,
                    histogram_response=histogram_response,
                ))
        if not records:
            raise CandidateError("no renderable annotated LUT records")
        cluster_path = getattr(config, "cluster_artifact", None)
        clusters = load_cluster_map(cluster_path) if cluster_path else {}
        return cls(
            records, clusters,
            segment_fingerprints_sha256=(
                file_sha256(Path(fingerprint_path)) if fingerprint_path else ""
            ),
        )

    def get(self, preset_id: str) -> LutRecord:
        try:
            return self.by_id[preset_id]
        except KeyError as exc:
            raise CandidateError(f"unknown preset ID: {preset_id}") from exc

    def restrict(self, preset_ids: Iterable[str]) -> "LutCatalog":
        allowed = set(preset_ids)
        records = [row for row in self.records if row.preset_id in allowed]
        if not records:
            raise CandidateError("no annotated LUT is renderable by the active backend")
        return LutCatalog(
            records, self.clusters,
            segment_fingerprints_sha256=self.segment_fingerprints_sha256,
        )

    def direction_scores(
        self, direction: MaskDirection
    ) -> tuple[dict[str, float], dict[str, float]]:
        """(combined prefilter score, correction-mode score) per preset, whole catalog.

        Both are one vectorized pass over the mounted (N, 3, 5) fingerprint table
        (R7.1: the prefilter reads the whole catalog, not an offline reach subset).
        """
        table = self.segment_fingerprint_table()
        combined = combined_direction_scores(direction, table)
        correction = direction_match_scores(
            direction.correction, table, direction.tonal_weights
        )
        return (
            {preset: float(combined[index])
             for index, preset in enumerate(table.preset_ids)},
            {preset: float(correction[index])
             for index, preset in enumerate(table.preset_ids)},
        )

    def global_shortlist(
        self, diagnosis: Mapping[str, Any], palette: Mapping[str, Any], scene: str,
        config: CatalogConfig, preset_reach: Mapping[str, Any] | None = None,
        source_sha256: str = "", source_histogram: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        reach = dict((preset_reach or {}).get("presets") or {})
        scored = [
            (self._score(row, diagnosis, palette, scene, source_histogram), row)
            for row in self.records
            if not reach or (row.preset_id in reach and reach[row.preset_id]["achievable_bins"])
        ]
        by_major: dict[str, list[tuple[float, LutRecord]]] = {}
        for score, row in scored:
            by_major.setdefault(row.style_major, []).append((score, row))
        deficits: list[dict[str, Any]] = []
        eligible, correction_note = _correction_majors(by_major, diagnosis, palette)
        if correction_note is not None:
            deficits.append(correction_note)
        ranked_majors = sorted(
            eligible,
            key=lambda major: (-sum(score for score, _ in sorted(
                eligible[major], reverse=True, key=lambda pair: pair[0]
            )[:3]), major),
        )[:config.global_major_limit]
        result: dict[str, list[dict[str, Any]]] = {}
        for major in ranked_majors:
            ranked = sorted(eligible[major], key=lambda pair: (-pair[0], pair[1].preset_id))
            rank_of = {row.preset_id: index for index, (_score, row) in enumerate(ranked)}
            pool = self._cluster_unique(ranked, source_sha256)

            def bins_of(row: LutRecord) -> list[str]:
                if not reach:
                    return list(GLOBAL_BIN_ORDER)
                return list(reach[row.preset_id]["achievable_bins"])

            selected, missing = _quota_select(
                pool, bins_of, GLOBAL_BIN_ORDER, GLOBAL_BIN_QUOTA,
                config.global_per_major_limit,
            )
            for row in missing:
                deficits.append({"scope": "global", "style_major": major, **row})
            result[major] = [
                self._row_view(
                    record, score, rank_of[record.preset_id], offered, reach, "global"
                )
                for offered, (score, record) in enumerate(selected)
            ]
        return {
            "by_major": result, "quota_deficits": deficits,
            "offered_majors": list(result),
        }

    def _row_view(
        self, record: LutRecord, score: float, scorer_rank_raw: int,
        scorer_rank_offered: int, reach: Mapping[str, Any], level: str,
    ) -> dict[str, Any]:
        view = record.prompt_view()
        # Two rank readings: raw = rank before cluster dedupe and quota selection;
        # offered = rank inside the shortlist the model actually sees.
        view["scorer_rank_raw"] = int(scorer_rank_raw)
        view["scorer_rank_offered"] = int(scorer_rank_offered)
        view["score"] = round(float(score), 6)
        view["cluster_id"] = self.cluster_id(record)
        if level != "global":
            raise CandidateError("only the global round reads the offline reach set")
        if reach:
            view["d_full"] = float(reach[record.preset_id]["d_full"])
            view["achievable_bins"] = list(reach[record.preset_id]["achievable_bins"])
        return view

    def _cluster_unique(
        self, ranked: Sequence[tuple[float, LutRecord]], source_sha256: str
    ) -> list[tuple[float, LutRecord]]:
        """Keep one member per effect cluster, rotating on sha256(source_sha, cluster)."""
        groups: dict[str, list[tuple[float, LutRecord]]] = {}
        for pair in ranked:
            groups.setdefault(self.cluster_id(pair[1]), []).append(pair)
        selected = []
        for cluster, members in groups.items():
            members = sorted(members, key=lambda pair: (-pair[0], pair[1].preset_id))
            digest = hashlib.sha256(
                f"{source_sha256}\x1f{cluster}".encode("utf-8")
            ).hexdigest()
            selected.append(members[int(digest[:16], 16) % len(members)])
        return sorted(selected, key=lambda pair: (-pair[0], pair[1].preset_id))

    def reach_candidates(
        self, diagnosis: Mapping[str, Any], palette: Mapping[str, Any], scene: str,
        limit: int, source_histogram: Mapping[str, Any] | None = None,
    ) -> list[LutRecord]:
        ranked = sorted(
            self.records,
            key=lambda row: (
                -self._score(row, diagnosis, palette, scene, source_histogram),
                row.preset_id,
            ),
        )
        return ranked[:limit]

    def intent_rows(
        self, intent: str, exclude: Iterable[str], diagnosis: Mapping[str, Any],
        palette: Mapping[str, Any], scene: str, limit: int, *,
        mask: Mapping[str, Any], mask_reach: Any,
        combined_scores: Mapping[str, float], correction_scores: Mapping[str, float],
        source_sha256: str = "",
        global_fingerprint: Mapping[str, float] | None = None,
        source_histogram: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """LUT subset of one intent packet, retrieved online over the whole catalog.

        B11 item 2 (R7.1) replaced the offline top-300 reach subset with three stages:

        1. the intent direction domain (`intent_admits`) over every catalog record,
        2. the segmented-fingerprint direction prefilter, keeping
           `LOCAL_ONLINE_RETRIEVAL["prefilter_top_k"]` presets by the combined
           (measured residual, diagnosis enhancement) match on this mask's tonal
           weighting,
        3. the mask-conditioned reach probe (B8) on those survivors only.

        A row's `achievable_bins` therefore comes from its mask-conditioned reach, not
        from the whole-image `d_full` of the offline artifact, which the local round no
        longer reads at all. Contract C2's global-preset exclusion and the cluster
        dedupe are unchanged.
        """
        if intent not in LOCAL_INTENTS:
            raise CandidateError(f"unknown local intent: {intent}")
        if mask_reach is None:
            raise CandidateError("local intent rows require a mask reach probe")
        blocked = set(exclude)
        domain = [
            row for row in self.records
            if row.preset_id not in blocked
            and intent_admits(
                intent, row.fingerprint(), global_fingerprint,
                segment_fingerprint=row.segment_fingerprint,
                correction_match=correction_scores.get(row.preset_id),
            )
        ]
        top_k = int(LOCAL_ONLINE_RETRIEVAL["prefilter_top_k"])
        prefiltered = sorted(
            domain,
            key=lambda row: (-float(combined_scores.get(row.preset_id, 0.0)),
                             row.preset_id),
        )[:top_k]
        floor = float(MASK_REACH_GATE["reach_de_min"])
        reach_of: dict[str, float] = {}
        below: list[dict[str, Any]] = []
        for record in prefiltered:
            value = round(float(mask_reach.measure(mask, record.preset_id)), 6)
            if value < floor:
                below.append({"preset_id": record.preset_id, "mask_reach_de": value})
                continue
            reach_of[record.preset_id] = value
        rows = [
            (self._score(record, diagnosis, palette, scene, source_histogram), record)
            for record in prefiltered if record.preset_id in reach_of
        ]
        rows.sort(key=lambda pair: (-pair[0], pair[1].preset_id))
        rank_of = {row.preset_id: index for index, (_score, row) in enumerate(rows)}
        pool = self._cluster_unique(rows, source_sha256)
        ladder = local_ladder(intent)

        def bins_of(row: LutRecord) -> list[str]:
            value = reach_of[row.preset_id]
            return [
                name for name, (low, _high, _inclusive) in ladder.items()
                if value >= low
            ]

        selected, missing = _quota_select(
            pool, bins_of, LOCAL_BIN_ORDER, INTENT_BIN_QUOTA, limit
        )
        return {
            "rows": [
                self._local_row_view(
                    record, score, rank_of[record.preset_id], offered,
                    reach_of[record.preset_id], bins_of(record),
                    float(combined_scores.get(record.preset_id, 0.0)),
                )
                for offered, (score, record) in enumerate(selected)
            ],
            "quota_deficits": [
                {"scope": "local", "intent": intent, **row} for row in missing
            ],
            "domain_size": len(domain),
            "prefiltered": len(prefiltered),
            "below_floor": below,
        }

    def _local_row_view(
        self, record: LutRecord, score: float, scorer_rank_raw: int,
        scorer_rank_offered: int, mask_reach_de: float, achievable_bins: Sequence[str],
        direction_score: float,
    ) -> dict[str, Any]:
        """R7.1 local row: the reach column is mask-conditioned, not whole-image."""
        view = record.prompt_view()
        view["scorer_rank_raw"] = int(scorer_rank_raw)
        view["scorer_rank_offered"] = int(scorer_rank_offered)
        view["score"] = round(float(score), 6)
        view["cluster_id"] = self.cluster_id(record)
        view["achievable_bins"] = [str(name) for name in achievable_bins]
        view["mask_reach_de"] = round(float(mask_reach_de), 6)
        # Audit only: never serialized into the shortlist table.
        view["direction_score"] = round(float(direction_score), 6)
        return view

    @staticmethod
    def _score(
        row: LutRecord, diagnosis: Mapping[str, Any], palette: Mapping[str, Any],
        scene: str, source_histogram: Mapping[str, Any] | None = None,
    ) -> float:
        summary = row.hsl_features.get("summary") or {}
        mid_a = float(summary.get("mid_gray_a", 0.0) or 0.0)
        mid_b = float(summary.get("mid_gray_b", 0.0) or 0.0)
        mid_l = float(summary.get("mid_gray_dL", 0.0) or 0.0)
        saturation = float(summary.get("sat_pct_mean", 0.0) or 0.0)
        source_a = float(palette.get("lab_a_mean", 0.0))
        source_b = float(palette.get("lab_b_mean", 0.0))
        source_l = float(palette.get("lab_l_mean", 50.0))
        score = -(source_a * mid_a + source_b * mid_b) / 80.0
        if source_l < 42:
            score += max(mid_l, 0.0) / 10.0
        elif source_l > 72:
            score += max(-mid_l, 0.0) / 10.0
        text = " ".join(
            str(item) for key in (
                "correction_needs", "enhancement_opportunities", "forbidden_directions"
            ) for item in diagnosis.get(key, [])
        ).lower()
        # B10: the two saturation word lists and the forbidden-direction alias table are
        # `direction_match`'s single source of truth now; the arithmetic is unchanged.
        if any(word in text for word in SATURATION_LIFT_WORDS):
            score += max(saturation, 0.0) / 20.0
        if any(word in text for word in SATURATION_DROP_WORDS):
            score += max(-saturation, 0.0) / 20.0
        if scene in row.scene_affinity:
            score += 0.15 if len(row.scene_affinity) > 1 else 0.5
        forbidden = " ".join(map(str, diagnosis.get("forbidden_directions", []))).lower()
        caption = (row.caption + " " + " ".join(row.per_probe.values())).lower()
        for needle, aliases in FORBIDDEN_CAPTION_ALIASES:
            if needle in forbidden and any(alias in caption for alias in aliases):
                score -= 3.0
        # B12 item 3: the pre-registered histogram term. It is exactly 0.0 whenever the
        # source histogram is not wired or the catalog carries no v2 histogram group,
        # so a v1 mount reproduces the pre-B12 score bit for bit.
        score += histogram_match_bonus(source_histogram, row.histogram_response)
        return float(score)


def _quota_select(
    pool: Sequence[tuple[float, LutRecord]], bins_of: Any, bin_order: Sequence[str],
    per_bin: int, limit: int,
) -> tuple[list[tuple[float, LutRecord]], list[dict[str, Any]]]:
    """Per-bin quota first, then score fill. Shortfalls are reported, never raised."""
    selected: list[tuple[float, LutRecord]] = []
    used: set[str] = set()
    deficits: list[dict[str, Any]] = []
    eligible_by_bin = {
        name: [pair for pair in pool if name in bins_of(pair[1])] for name in bin_order
    }
    # Scarcest bin first: a greedy pass in fixed bin order starves rare bins whose
    # only candidates are also the highest scoring candidates of a common bin.
    scarcity = sorted(
        bin_order, key=lambda name: (len(eligible_by_bin[name]), bin_order.index(name))
    )
    for name in scarcity:
        eligible = eligible_by_bin[name]
        covered = sum(1 for pair in selected if name in bins_of(pair[1]))
        for pair in eligible:
            if covered >= per_bin:
                break
            if pair[1].preset_id in used:
                continue
            used.add(pair[1].preset_id)
            selected.append(pair)
            covered += 1
        if covered < per_bin:
            deficits.append({
                "bin": name, "required": per_bin, "covered": covered,
                "reachable_candidates": len(eligible),
            })
    deficits.sort(key=lambda row: bin_order.index(str(row["bin"])))
    for pair in pool:
        if len(selected) >= limit:
            break
        if pair[1].preset_id not in used:
            used.add(pair[1].preset_id)
            selected.append(pair)
    selected = sorted(selected, key=lambda pair: (-pair[0], pair[1].preset_id))[:limit]
    return selected, deficits


def _correction_majors(
    by_major: Mapping[str, Sequence[tuple[float, LutRecord]]],
    diagnosis: Mapping[str, Any], palette: Mapping[str, Any],
) -> tuple[dict[str, list[tuple[float, LutRecord]]], dict[str, Any] | None]:
    """Correction-first hard filter: keep majors whose mid-gray response opposes the cast."""
    kept = {major: list(rows) for major, rows in by_major.items()}
    if not diagnosis.get("correction_needs"):
        return kept, None
    source_a = float(palette.get("lab_a_mean", 0.0) or 0.0)
    source_b = float(palette.get("lab_b_mean", 0.0) or 0.0)
    opposing_share: dict[str, tuple[int, int]] = {}
    for major, rows in by_major.items():
        opposing = 0
        for _score, record in rows:
            summary = record.hsl_features.get("summary") or {}
            dot = source_a * _number(summary, "mid_gray_a") \
                + source_b * _number(summary, "mid_gray_b")
            opposing += int(dot < 0.0)
        opposing_share[major] = (opposing, len(rows))
    filtered = {
        major: list(by_major[major]) for major, (opposing, total) in opposing_share.items()
        if total and opposing * 2 > total
    }
    if not filtered:
        return kept, {
            "scope": "global", "reason": "correction_filter_empty",
            "correction_needs": len(diagnosis.get("correction_needs") or []),
            "majors_before": len(by_major), "majors_after": 0,
        }
    note = None
    if len(filtered) < len(by_major):
        note = {
            "scope": "global", "reason": "correction_filter_applied",
            "majors_before": len(by_major), "majors_after": len(filtered),
            "dropped_majors": sorted(set(by_major) - set(filtered)),
        }
    return filtered, note


def _cast_b(fingerprint: Mapping[str, float]) -> float:
    """Lab b* component of the mid-gray cast; positive is warm, negative is cool."""
    return float(fingerprint.get("cast_mag", 0.0)) * math.sin(
        math.radians(float(fingerprint.get("cast_hue", 0.0)))
    )


def _fp(fingerprint: Mapping[str, float], key: str) -> float:
    value = fingerprint.get(key, 0.0)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) \
        else 0.0


def intent_admits(
    intent: str, fingerprint: Mapping[str, float],
    global_fingerprint: Mapping[str, float] | None = None, *,
    segment_fingerprint: Mapping[str, Mapping[str, float]] | None = None,
    correction_match: float | None = None,
) -> bool:
    """Intent fingerprint direction domain (decisions doc section 3).

    The local LUT is applied on top of global_after, so its own eight-number
    fingerprint is the direction of the added edit relative to global_after. The two
    paired intents additionally read the global preset fingerprint, which is what
    they are contrasted against.

    B11 item 3 (R7.3) adds two intents whose domain is not expressible in the eight
    numbers: `contrast_boost` reads the segmented fingerprint (`segment_fingerprint`)
    and `cast_correction_local` reads the R6.2 match against the measured mask residual
    (`correction_match`). Both raise when their input is missing, so an unwired caller
    fails loudly instead of silently admitting everything.
    """
    gate = INTENT_FINGERPRINT_GATE
    dl = _fp(fingerprint, "dL")
    dsat = _fp(fingerprint, "dSat")
    cast_mag = _fp(fingerprint, "cast_mag")
    global_fingerprint = dict(global_fingerprint or {})
    if intent == "luminance_pop":
        return dl >= gate["dL_positive_min"]
    if intent == "sat_boost":
        return dsat >= gate["dSat_positive_min"] \
            and abs(dl) <= gate["dL_small_abs_max"]
    if intent == "hue_shift":
        return gate["cast_mag_mid_min"] <= cast_mag <= gate["cast_mag_mid_max"] \
            and abs(dl) <= gate["dL_small_abs_max"] \
            and abs(dsat) <= gate["dSat_small_abs_max"]
    if intent == "highlight_rescue":
        return dl <= gate["dL_negative_max"] \
            and _fp(fingerprint, "highlight_dL") <= gate["highlight_dL_negative_max"]
    if intent == "background_control":
        return dl <= gate["dL_nonpositive_max"] \
            and dsat <= gate["dSat_nonpositive_max"] \
            and cast_mag <= gate["cast_mag_small_max"]
    if intent == "zonal_contrast":
        # v1 `background_darken`: darken the background while the global edit did not
        # darken the frame.
        return dl <= gate["dL_negative_max"] \
            and _fp(global_fingerprint, "dL") >= gate["dL_nonpositive_max"]
    if intent == "warm_cool_split":
        # v1 `background_cool`: cool the background while the global edit is not cool.
        return _cast_b(fingerprint) <= gate["cast_b_cool_max"] \
            and _cast_b(global_fingerprint) >= 0.0
    if intent == "contrast_boost":
        # R7.3: tonal contrast inside the mask. Read off the segmented fingerprint, so
        # a LUT that only shifts the mid gray cannot qualify.
        if segment_fingerprint is None:
            raise CandidateError(
                "contrast_boost requires a mounted segment fingerprint"
            )
        return (
            float(segment_fingerprint["shadows"]["dL"])
            <= INTENT_SEGMENT_GATE["contrast_shadow_dL_max"]
            and float(segment_fingerprint["highlights"]["dL"])
            >= INTENT_SEGMENT_GATE["contrast_highlight_dL_min"]
        )
    if intent == "cast_correction_local":
        # R7.3: undo the cast that survived the global edit inside this mask. The score
        # is already oriented (correction mode), so "points against the residual" is
        # simply a large positive number.
        if correction_match is None:
            raise CandidateError(
                "cast_correction_local requires a measured mask direction"
            )
        return float(correction_match) \
            >= INTENT_DIRECTION_GATE["cast_correction_match_min"]
    raise CandidateError(f"unknown local intent: {intent}")


def intent_offered(
    intent: str, role: str, headroom: Mapping[str, Any] | None = None
) -> tuple[bool, str]:
    """Role match plus the conditional triggers; returns (offered, reason)."""
    if intent not in LOCAL_INTENTS:
        raise CandidateError(f"unknown local intent: {intent}")
    if intent in DISABLED_LOCAL_INTENTS:
        return False, "intent_disabled"
    if not intent_serves_role(intent, role):
        return False, "role_mismatch"
    values = dict(headroom or {})
    pressure = bool(values.get("highlight_pressure", False))
    if intent == "luminance_pop" and pressure:
        # B2 candidate-side guard: no dL>0 intent when the subject has no headroom.
        return False, "highlight_headroom_exhausted"
    if intent == "highlight_rescue" and not pressure:
        return False, "no_highlight_pressure"
    if intent == "sat_boost" and float(
        values.get("subject_saturation_mean", 0.0)
    ) > SUBJECT_HEADROOM_GATE["sat_mean_max"]:
        return False, "subject_saturation_high"
    return True, "offered"


def build_local_packets(
    catalog: "LutCatalog", masks: Sequence[Mapping[str, Any]], *,
    exclude: Iterable[str], diagnosis: Mapping[str, Any], palette: Mapping[str, Any],
    scene: str,
    source_sha256: str = "", global_fingerprint: Mapping[str, float] | None = None,
    headroom: Mapping[str, Any] | None = None,
    row_limit: int = LOCAL_PACKET_ROW_LIMIT,
    mask_reach: Any | None = None,
    direction_probe: Any | None = None,
    source_histogram: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """(mask, intent, LUT subset) packets for one global branch.

    Rows of every packet are flattened into one pick-by-index shortlist; each packet
    carries the row indices it admits, so the model still answers with a row id.

    B8 item 4 (R1): every (mask, LUT) pair is measured on the mask itself and a LUT that
    cannot reach `reach_de_min` there is dropped. B11 item 2 (R7.1) moves the pool that
    reaches the probe from the offline top-300 reach set to an online direction
    prefilter over the whole catalog, one `MaskDirection` per mask. Both probes are
    mandatory; `mask_reach_applied` / `direction_prefilter_applied` report that they
    ran, so callers can assert it.

    Packets of one mask are emitted in `intent_packet_order` (R7.3), which is also the
    order their rows enter the flattened shortlist.
    """
    if mask_reach is None:
        raise CandidateError("local packets require the mask reach probe")
    if direction_probe is None:
        raise CandidateError("local packets require the mask direction probe")
    rows: list[dict[str, Any]] = []
    # B7 item 1: two intents on different ladders can admit the same preset and claim
    # different achievable bins for it, so the flattened shortlist is keyed by the
    # (preset, claimed bins) pair instead of the preset alone. B8 item 4 adds the
    # mask-conditioned reach to the key: the same preset measured on two masks is two
    # rows, because the number is serialized into the prompt.
    index_of: dict[tuple[str, tuple[str, ...], float | None], int] = {}
    packets: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []
    deficits: list[dict[str, Any]] = []
    retrieval: list[dict[str, Any]] = []
    floor = float(MASK_REACH_GATE["reach_de_min"])
    order = intent_packet_order(ACTIVE_LOCAL_INTENTS)
    for mask in masks:
        role = _mask_role(mask)
        direction = direction_probe.direction(mask)
        combined_scores, correction_scores = catalog.direction_scores(direction)
        for intent in order:
            if not intent_serves_role(intent, role):
                continue
            offered, reason = intent_offered(intent, role, headroom)
            if not offered:
                notes.append({"mask_id": str(mask["mask_id"]), "intent": intent,
                              "role": role, "reason": reason})
                continue
            payload = catalog.intent_rows(
                intent, exclude, diagnosis, palette, scene, row_limit,
                mask=mask, mask_reach=mask_reach, combined_scores=combined_scores,
                correction_scores=correction_scores, source_sha256=source_sha256,
                global_fingerprint=global_fingerprint,
                source_histogram=source_histogram,
            )
            retrieval.append({
                "mask_id": str(mask["mask_id"]), "intent": intent, "role": role,
                "domain_size": int(payload["domain_size"]),
                "prefiltered": int(payload["prefiltered"]),
                "reach_dropped": len(payload["below_floor"]),
                "rows": len(payload["rows"]),
            })
            if not payload["prefiltered"]:
                notes.append({"mask_id": str(mask["mask_id"]), "intent": intent,
                              "role": role, "reason": "no_row_in_intent_domain"})
                continue
            if not payload["rows"]:
                notes.append({
                    "mask_id": str(mask["mask_id"]), "intent": intent, "role": role,
                    "reason": "no_row_reaches_mask",
                    "mask_reach_de_min": floor,
                    "mask_reach_de_max": max(
                        (row["mask_reach_de"] for row in payload["below_floor"]),
                        default=0.0,
                    ),
                    "dropped_rows": len(payload["below_floor"]),
                })
                continue
            indices = []
            for row in payload["rows"]:
                key = (
                    str(row["preset_id"]),
                    tuple(str(name) for name in row.get("achievable_bins") or ()),
                    row.get("mask_reach_de"),
                )
                if key not in index_of:
                    index_of[key] = len(rows)
                    rows.append(dict(row))
                indices.append(index_of[key])
            packets.append({
                "mask_id": str(mask["mask_id"]), "role": role, "intent": intent,
                "intent_variant": INTENT_V1_VARIANTS.get(intent, intent),
                "row_indices": indices,
            })
            deficits.extend(
                {"mask_id": str(mask["mask_id"]), **row}
                for row in payload["quota_deficits"]
            )
    return {"rows": rows, "packets": packets, "notes": notes,
            "quota_deficits": deficits, "retrieval": retrieval,
            "mask_reach_applied": True, "direction_prefilter_applied": True,
            # B12 item 3 runtime assertion channel: the histogram term only entered
            # `_score` if a source reading was actually handed down here.
            "source_histogram_applied": bool(source_histogram)}


def offered_intents(packets: Sequence[Mapping[str, Any]], mask_id: str) -> list[str]:
    return [str(row["intent"]) for row in packets if str(row["mask_id"]) == mask_id]


def palette_summary(path: str | Path) -> dict[str, Any]:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((256, 256), getattr(Image, "Resampling", Image).LANCZOS)
        rgb = np.asarray(image, dtype=np.float32) / 255.0
    lab = rgb2lab(rgb)
    hsv = rgb2hsv(rgb)
    quantized = np.clip((rgb.reshape(-1, 3) * 7).astype(np.int32), 0, 7)
    keys, counts = np.unique(quantized, axis=0, return_counts=True)
    order = np.argsort(-counts)[:8]
    palette = [
        {"rgb": [round(float(v / 7), 3) for v in keys[index]],
         "fraction": round(float(counts[index] / counts.sum()), 4)}
        for index in order
    ]
    return {
        "lab_l_mean": round(float(lab[..., 0].mean()), 4),
        "lab_a_mean": round(float(lab[..., 1].mean()), 4),
        "lab_b_mean": round(float(lab[..., 2].mean()), 4),
        "saturation_mean": round(float(hsv[..., 1].mean()), 4),
        "value_mean": round(float(hsv[..., 2].mean()), 4),
        "palette": palette,
    }


def _seed(*parts: object) -> int:
    data = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def _smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _stats(alpha: np.ndarray, core: np.ndarray) -> dict[str, float]:
    subject = core > 0.5
    values = alpha[subject]
    return {
        "effective_alpha_mean": float(alpha.mean()),
        "half_area": float((alpha >= 0.5).mean()),
        "support_area": float((alpha > 0.05).mean()),
        "subject_high_coverage": float((values >= 0.5).mean()),
        "subject_support_coverage": float((values > 0.05).mean()),
    }


def _background_stats(alpha: np.ndarray, core: np.ndarray) -> dict[str, float]:
    """The two extra readings the background role is gated on."""
    subject = core > 0.5
    background = ~subject
    return {
        "subject_alpha_mean": float(alpha[subject].mean()) if subject.any() else 0.0,
        "background_alpha_mean": float(alpha[background].mean()) if background.any() else 0.0,
    }


def _avoids_subject(alpha: np.ndarray, core: np.ndarray) -> bool:
    """Mirror of the subject coverage gate: the mask must stay off the subject."""
    stats = _stats(alpha, core)
    extra = _background_stats(alpha, core)
    return (
        extra["subject_alpha_mean"] <= BACKGROUND_ROLE_GATE["subject_alpha_mean_max"]
        and stats["subject_high_coverage"]
        <= BACKGROUND_ROLE_GATE["subject_high_coverage_max"]
    )


def _covers_background(alpha: np.ndarray, core: np.ndarray) -> bool:
    stats = _stats(alpha, core)
    extra = _background_stats(alpha, core)
    return (
        extra["background_alpha_mean"]
        >= BACKGROUND_ROLE_GATE["background_alpha_mean_min"]
        and stats["half_area"] >= BACKGROUND_ROLE_GATE["half_area_min"]
    )


def _passes_background(alpha: np.ndarray, core: np.ndarray) -> bool:
    return _avoids_subject(alpha, core) and _covers_background(alpha, core)


def _subject_geometry(core: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    ys, xs = np.nonzero(core > 0.5)
    if len(xs) < 32:
        raise CandidateError("subject mask has fewer than 32 working pixels")
    points = np.stack([xs, ys], axis=1).astype(np.float64)
    center = points.mean(axis=0)
    covariance = np.cov((points - center).T)
    _, vectors = np.linalg.eigh(covariance)
    major = vectors[:, 1]
    return points, center, math.atan2(major[1], major[0])


def _ellipse(shape: tuple[int, int], center: np.ndarray, angle: float,
             axis_a: float, axis_b: float) -> np.ndarray:
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    dx, dy = xx - center[0], yy - center[1]
    along = dx * math.cos(angle) + dy * math.sin(angle)
    across = -dx * math.sin(angle) + dy * math.cos(angle)
    distance = np.sqrt((along / axis_a) ** 2 + (across / axis_b) ** 2)
    return _smoothstep((1.25 - distance) / 0.50).astype(np.float32)


def _passes(alpha: np.ndarray, core: np.ndarray, target: float) -> bool:
    stats = _stats(alpha, core)
    return stats["effective_alpha_mean"] >= target and \
        stats["subject_high_coverage"] >= 0.98 and \
        stats["subject_support_coverage"] >= 1.0


def _radials(core: np.ndarray, rng: random.Random, count: int) -> list[tuple[np.ndarray, dict[str, Any]]]:
    points, center, pca_angle = _subject_geometry(core)
    short = float(min(core.shape))
    result = []
    for index in range(count):
        angle = (
            pca_angle + math.radians(rng.uniform(-12, 12)) if index == 0 else
            rng.choice((0.0, math.pi / 2)) + math.radians(rng.uniform(-15, 15)) if index == 1 else
            pca_angle + math.pi / 2 + math.radians(rng.uniform(-15, 15))
        )
        unit = np.array([math.cos(angle), math.sin(angle)])
        normal = np.array([-math.sin(angle), math.cos(angle)])
        centered = points - center
        a = max(1.18 * float(np.quantile(np.abs(centered @ unit), 0.98)), 0.25 * short)
        b = max(1.18 * float(np.quantile(np.abs(centered @ normal), 0.98)), 0.18 * short)
        if a < b:
            a, b, angle = b, a, angle + math.pi / 2
        target = rng.uniform(0.46, 0.58)
        low = high = 1.0
        while not _passes(_ellipse(core.shape, center, angle, a * high, b * high), core, target) \
                and high < 16:
            high *= 1.5
        if high >= 16 and not _passes(
            _ellipse(core.shape, center, angle, a * high, b * high), core, target
        ):
            raise CandidateError("radial mask cannot satisfy Mask v2 gate")
        if high > 1:
            for _ in range(28):
                middle = (low + high) / 2
                if _passes(_ellipse(core.shape, center, angle, a * middle, b * middle), core, target):
                    high = middle
                else:
                    low = middle
        alpha = _ellipse(core.shape, center, angle, a * high, b * high)
        result.append((alpha, {
            "direction": "elliptical",
            "center_hint": "through the main subject and surrounding background",
            "_geometry": {
                "kind": "radial", "center_x": float(center[0] / core.shape[1]),
                "center_y": float(center[1] / core.shape[0]), "angle": float(angle),
                "axis_a_short": float(a * high / short),
                "axis_b_short": float(b * high / short),
            },
        }))
    return result


def _bands(core: np.ndarray, rng: random.Random, count: int) -> list[tuple[np.ndarray, dict[str, Any]]]:
    _points, center, _angle = _subject_geometry(core)
    height, width = core.shape
    if width > height:
        directions = [("vertical", math.pi / 2 + math.radians(value))
                      for value in np.linspace(-10, 10, count)]
    else:
        diagonal = rng.choice((45.0, 135.0))
        pool = [("horizontal", 0.0), ("vertical", 90.0), ("diagonal", diagonal)]
        chosen = rng.sample(pool, k=min(count, len(pool)))
        directions = [(name, math.radians(value + rng.uniform(-8, 8)))
                      for name, value in chosen]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    short = float(min(core.shape))
    result = []
    for direction, angle in directions:
        target = rng.uniform(0.46, 0.58)

        def at_width(half: float) -> np.ndarray:
            distance = np.abs((xx - center[0]) * -math.sin(angle) +
                              (yy - center[1]) * math.cos(angle))
            return _smoothstep((1.25 - distance / half) / 0.50).astype(np.float32)

        low, high = 0.18 * short, 2.0 * max(core.shape)
        if not _passes(at_width(low), core, target):
            if not _passes(at_width(high), core, target):
                raise CandidateError("band mask cannot satisfy Mask v2 gate")
            for _ in range(28):
                middle = (low + high) / 2
                if _passes(at_width(middle), core, target):
                    high = middle
                else:
                    low = middle
            low = high
        result.append((at_width(low), {
            "direction": direction,
            "center_hint": "through the main subject and surrounding background",
            "_geometry": {
                "kind": "band", "center_x": float(center[0] / width),
                "center_y": float(center[1] / height), "angle": float(angle),
                "half_short": float(low / short),
            },
        }))
    return result


def _linears(core: np.ndarray) -> list[tuple[np.ndarray, dict[str, Any]]]:
    ys, xs = np.nonzero(core > 0.5)
    height, width = core.shape
    bbox = (xs.min() / width, ys.min() / height, (xs.max() + 1) / width,
            (ys.max() + 1) / height)
    rooms = {"left": bbox[0], "right": 1 - bbox[2], "top": bbox[1], "bottom": 1 - bbox[3]}
    sides = [name for name, room in sorted(rooms.items(), key=lambda pair: (-pair[1], pair[0]))
             if room >= 0.20]
    if not sides:
        sides = ["left", "right"]
    elif len(sides) == 1:
        sides.append(next(name for name, _room in sorted(
            rooms.items(), key=lambda pair: (-pair[1], pair[0])
        ) if name != sides[0]))
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    x, y = xx / max(width - 1, 1), yy / max(height - 1, 1)
    result = []
    for index in range(2):
        side = sides[index % len(sides)]
        raw = {"left": 1 - x, "right": x, "top": 1 - y, "bottom": y}[side]
        raw = _smoothstep(raw)
        amount = min(1.0, 0.5 / max(float(raw.mean()), 1e-6))
        alpha = np.clip(raw * amount, 0, 1).astype(np.float32)
        result.append((alpha, {"direction": side, "center_hint": f"the {side} side around the subject"}))
    return result


def _search_core(core: np.ndarray, short_edge: int = 256) -> np.ndarray:
    if min(core.shape) <= short_edge:
        return core
    scale = short_edge / min(core.shape)
    size = (max(1, round(core.shape[1] * scale)),
            max(1, round(core.shape[0] * scale)))
    resized = Image.fromarray((core * 255).astype(np.uint8), "L").resize(
        size, getattr(Image, "Resampling", Image).NEAREST
    )
    return (np.asarray(resized, dtype=np.uint8) > 127).astype(np.float32)


def _evaluate_geometry(shape: tuple[int, int], geometry: Mapping[str, Any]) -> np.ndarray:
    height, width = shape
    short = float(min(shape))
    kind = str(geometry["kind"])
    center = np.array([
        float(geometry["center_x"]) * width,
        float(geometry["center_y"]) * height,
    ])
    angle = float(geometry["angle"])
    if kind == "radial":
        alpha = _ellipse(
            shape, center, angle, float(geometry["axis_a_short"]) * short,
            float(geometry["axis_b_short"]) * short,
        )
    elif kind == "band":
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        distance = np.abs((xx - center[0]) * -math.sin(angle)
                          + (yy - center[1]) * math.cos(angle))
        half = max(float(geometry["half_short"]) * short, 1e-6)
        alpha = _smoothstep((1.25 - distance / half) / 0.50).astype(np.float32)
    else:
        raise CandidateError(f"unknown mask geometry: {geometry['kind']}")
    if geometry.get("complement"):
        alpha = (1.0 - alpha).astype(np.float32)
    return alpha


def _band_slabs(alpha: np.ndarray, angle: float) -> list[tuple[int, int]]:
    """(slab width in px, slab pixel count) of the alpha >= 0.5 support of a band.

    Pixels are projected on the band normal and binned at one pixel; contiguous runs of
    occupied bins are the slabs. A plain band has one slab, the complement band used by
    the background role has up to two, and each is measured on its own.
    """
    binary = alpha >= 0.5
    ys, xs = np.nonzero(binary)
    if xs.size == 0:
        return []
    distance = xs * -math.sin(angle) + ys * math.cos(angle)
    bins = np.floor(distance - distance.min()).astype(np.int64)
    counts = np.bincount(bins)
    occupied = counts > 0
    slabs: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(occupied):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            slabs.append((index - start, int(counts[start:index].sum())))
            start = None
    if start is not None:
        slabs.append((len(occupied) - start, int(counts[start:].sum())))
    return slabs


def band_geometry_reading(
    alpha: np.ndarray, geometry: Mapping[str, Any]
) -> dict[str, float]:
    """B8 item 2: narrow-side width (relative to the short edge) and aspect of a band.

    Both readings are the worst slab: the thinnest short side and the longest aspect.
    Slab length is `pixels / width`, i.e. the mean extent along the band direction, so
    the frame clipping of a rotated band is accounted for without a chord formula.
    """
    slabs = _band_slabs(alpha, float(geometry["angle"]))
    short = float(min(alpha.shape))
    if not slabs:
        return {"band_min_width": 0.0, "band_aspect": float("inf")}
    return {
        "band_min_width": min(width / short for width, _count in slabs),
        "band_aspect": max((count / width) / width for width, count in slabs),
    }


def _band_geometry_rejection(reading: Mapping[str, float]) -> str | None:
    if float(reading["band_min_width"]) < BAND_GEOMETRY_GATE["min_width_short"]:
        return "band_min_width"
    if float(reading["band_aspect"]) > BAND_GEOMETRY_GATE["aspect_max"]:
        return "band_aspect"
    return None


def _expanded_geometry(geometry: Mapping[str, Any], expansion: int) -> dict[str, Any]:
    """One bounded quantization step, always toward the safer side of the gate.

    Subject masks grow to recover subject coverage lost to nearest-neighbour
    downsampling; background masks grow the excluded region, which lowers subject
    alpha for exactly the same reason.
    """
    adjusted = dict(geometry)
    kind = str(adjusted["kind"])
    if kind == "radial":
        scale = 1.01 ** expansion
        adjusted["axis_a_short"] = float(adjusted["axis_a_short"]) * scale
        adjusted["axis_b_short"] = float(adjusted["axis_b_short"]) * scale
    elif kind == "band":
        adjusted["half_short"] = float(adjusted["half_short"]) * (1.01 ** expansion)
    return adjusted


def _fit_full_resolution_geometry(
    core: np.ndarray, geometry: Mapping[str, Any], *, max_expansions: int = 4,
) -> np.ndarray:
    """Map search geometry to full resolution with a bounded quantization correction."""
    for expansion in range(max_expansions + 1):
        adjusted = _expanded_geometry(geometry, expansion)
        alpha = _evaluate_geometry(core.shape, adjusted)
        stats = _stats(alpha, core)
        if stats["effective_alpha_mean"] > 0.45 and \
                stats["subject_high_coverage"] >= 0.98 and \
                stats["subject_support_coverage"] >= 1.0:
            return alpha
    raise CandidateError(f"{geometry['kind']} violates Mask v2 gate")


def _fit_background_geometry(
    core: np.ndarray, geometry: Mapping[str, Any], *, max_expansions: int = 4,
) -> np.ndarray | None:
    """Full-resolution background fit; returns None when the mirrored gate fails.

    Both gate halves are monotone in the expansion: growing the excluded region
    lowers subject alpha and lowers background coverage. So the first expansion
    that clears subject avoidance is also the one with the most background left;
    if its coverage half fails, no further expansion can rescue it.
    """
    for expansion in range(max_expansions + 1):
        alpha = _evaluate_geometry(core.shape, _expanded_geometry(geometry, expansion))
        if _avoids_subject(alpha, core):
            return alpha if _covers_background(alpha, core) else None
    return None


def _minimal_avoiding_scale(
    core: np.ndarray, at_scale: Any, *, ceiling: float = 16.0, steps: int = 24,
) -> float | None:
    """Smallest geometry scale whose complement clears the subject-avoidance gate."""
    low = high = 1.0
    while not _avoids_subject(at_scale(high), core) and high < ceiling:
        high *= 1.5
    if not _avoids_subject(at_scale(high), core):
        return None
    if high > 1.0:
        for _ in range(steps):
            middle = (low + high) / 2.0
            if _avoids_subject(at_scale(middle), core):
                high = middle
            else:
                low = middle
    return high


def _band_direction_name(angle: float) -> str:
    degrees = math.degrees(angle) % 180.0
    if degrees < 22.5 or degrees >= 157.5:
        return "horizontal"
    if 67.5 <= degrees < 112.5:
        return "vertical"
    return "diagonal"


def _background_radials(
    core: np.ndarray, rng: random.Random, count: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Radial background: the complement of the smallest ellipse that hides the subject."""
    points, center, pca_angle = _subject_geometry(core)
    height, width = core.shape
    short = float(min(core.shape))
    centered = points - center
    geometries: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    for index in range(count):
        angle = pca_angle + math.radians(rng.uniform(-12, 12)) if index % 2 == 0 else \
            pca_angle + math.pi / 2 + math.radians(rng.uniform(-15, 15))
        unit = np.array([math.cos(angle), math.sin(angle)])
        normal = np.array([-math.sin(angle), math.cos(angle)])
        a = max(float(np.abs(centered @ unit).max()), 0.01 * short)
        b = max(float(np.abs(centered @ normal).max()), 0.01 * short)
        if a < b:
            a, b, angle = b, a, angle + math.pi / 2

        def at_scale(scale: float, angle: float = angle, a: float = a,
                     b: float = b) -> np.ndarray:
            return (1.0 - _ellipse(core.shape, center, angle, a * scale, b * scale)) \
                .astype(np.float32)

        scale = _minimal_avoiding_scale(core, at_scale)
        if scale is None:
            rejects.append({"role": "background", "family": "radial", "slot": index,
                            "reason": "subject_avoidance_unreachable"})
            continue
        geometry = {
            "kind": "radial", "complement": True,
            "center_x": float(center[0] / width), "center_y": float(center[1] / height),
            "angle": float(angle), "axis_a_short": float(a * scale / short),
            "axis_b_short": float(b * scale / short),
        }
        if not _covers_background(_evaluate_geometry(core.shape, geometry), core):
            rejects.append({"role": "background", "family": "radial", "slot": index,
                            "reason": "background_coverage_search"})
            continue
        geometries.append({
            "family": "radial", "geometry": geometry, "direction": "surrounding",
        })
    return geometries, rejects


def _background_bands(
    core: np.ndarray, rng: random.Random, count: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Band background: the complement of the narrowest stripe that hides the subject."""
    points, center, pca_angle = _subject_geometry(core)
    height, width = core.shape
    short = float(min(core.shape))
    centered = points - center
    geometries: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    for index in range(count):
        angle = pca_angle + math.radians(rng.uniform(-8, 8)) if index % 2 == 0 else \
            pca_angle + math.pi / 2 + math.radians(rng.uniform(-8, 8))
        normal = np.array([-math.sin(angle), math.cos(angle)])
        half = max(float(np.abs(centered @ normal).max()), 0.01 * short)
        base = {
            "kind": "band", "complement": True,
            "center_x": float(center[0] / width), "center_y": float(center[1] / height),
            "angle": float(angle),
        }

        def at_scale(scale: float, half: float = half,
                     base: dict[str, Any] = base) -> np.ndarray:
            return _evaluate_geometry(
                core.shape, {**base, "half_short": half * scale / short}
            )

        scale = _minimal_avoiding_scale(core, at_scale)
        if scale is None:
            rejects.append({"role": "background", "family": "band", "slot": index,
                            "reason": "subject_avoidance_unreachable"})
            continue
        geometry = {**base, "half_short": float(half * scale / short)}
        if not _covers_background(_evaluate_geometry(core.shape, geometry), core):
            rejects.append({"role": "background", "family": "band", "slot": index,
                            "reason": "background_coverage_search"})
            continue
        geometries.append({
            "family": "band", "geometry": geometry,
            "direction": _band_direction_name(angle),
        })
    return geometries, rejects


def background_geometry_bank(
    core: np.ndarray, rng: random.Random,
    counts: Mapping[str, int] = BACKGROUND_FAMILY_COUNTS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Search the background-role geometry pool on the downscaled core."""
    geometries: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    for builder, family in (
        (_background_radials, "radial"), (_background_bands, "band"),
    ):
        found, missed = builder(core, rng, int(counts.get(family, 0)))
        geometries.extend(found)
        rejects.extend(missed)
    return geometries, rejects


def build_mask_bank(
    subject_path: str | Path, *, render_size: tuple[int, int], source_id: str,
    prompt_revision: str, artifacts: ArtifactStore,
    include_background: bool = False,
    diagnostics: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    try:
        from dataset_build.tools.archive_reader import open_image
        subject_image = open_image(subject_path)
    except (ImportError, FileNotFoundError, ValueError):
        subject_image = Image.open(subject_path)
    with subject_image as image:
        core = ImageOps.exif_transpose(image).convert("L").resize(
            render_size, getattr(Image, "Resampling", Image).NEAREST
        )
        core_array = (np.asarray(core, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
    area = float(core_array.mean())
    if not 0.005 <= area <= 0.85:
        raise CandidateError(f"subject area outside Mask v2 range: {area:.6f}")
    rng = random.Random(_seed(source_id, prompt_revision, "mask-v2"))
    search_core = _search_core(core_array)
    raw: list[tuple[str, np.ndarray | None, dict[str, Any]]] = []
    if area >= 0.15:
        raw.append(("semantic", core_array, {"direction": "subject",
                                             "center_hint": "the main subject"}))
    raw.extend(("radial", alpha, meta) for alpha, meta in _radials(
        search_core, rng, 2 if area >= 0.15 else 3
    ))
    raw.extend(("band", alpha, meta) for alpha, meta in _bands(
        search_core, rng, 2 if area >= 0.15 else 3
    ))
    raw.extend(("linear", alpha, meta) for alpha, meta in _linears(core_array))
    notes = diagnostics if diagnostics is not None else []
    if include_background:
        found, missed = background_geometry_bank(search_core, rng)
        notes.extend(missed)
        raw.extend((
            str(item["family"]), None, {
                "_role": "background", "_geometry": item["geometry"],
                "direction": str(item["direction"]),
                "center_hint": BACKGROUND_CENTER_HINT,
            },
        ) for item in found)
    subject_ref = artifacts.put_alpha(core_array, retention="audit")
    records = []
    for index, (family, alpha, meta) in enumerate(raw):
        meta = dict(meta)
        role = str(meta.pop("_role", "subject"))
        geometry = meta.pop("_geometry", None)
        if geometry is not None:
            if role == "background":
                fitted = _fit_background_geometry(core_array, geometry)
                if fitted is None:
                    notes.append({
                        "role": role, "family": family, "slot": index,
                        "reason": "full_resolution_gate",
                    })
                    continue
                alpha = fitted
            else:
                alpha = _fit_full_resolution_geometry(core_array, geometry)
        stats = _stats(alpha, core_array)
        if family == "band" and geometry is not None:
            # B8 item 2: the all-role band geometry gate runs before the role split, so
            # a thin or elongated slab is dropped in either role.
            reading = band_geometry_reading(alpha, geometry)
            rejection = _band_geometry_rejection(reading)
            if rejection is not None:
                notes.append({
                    "role": role, "family": family, "slot": index,
                    "reason": rejection,
                    **{key: round(float(value), 8) for key, value in reading.items()},
                    "band_min_width_min": BAND_GEOMETRY_GATE["min_width_short"],
                    "band_aspect_max": BAND_GEOMETRY_GATE["aspect_max"],
                })
                continue
            stats = {**stats, **reading}
        if role == "background":
            stats = {**stats, **_background_stats(alpha, core_array)}
        elif family == "band" and \
                stats["half_area"] < SUBJECT_BAND_GATE["half_area_min"]:
            # B7 item 4: too thin a subject band is dropped, not raised.
            notes.append({
                "role": role, "family": family, "slot": index,
                "reason": "subject_band_min_area",
                "half_area": round(float(stats["half_area"]), 8),
                "half_area_min": SUBJECT_BAND_GATE["half_area_min"],
            })
            continue
        elif family in {"radial", "band"} and (
            stats["effective_alpha_mean"] <= 0.45 or
            stats["subject_high_coverage"] < 0.98 or
            stats["subject_support_coverage"] < 1.0
        ):
            raise CandidateError(f"{family} violates Mask v2 gate")
        ref = artifacts.put_alpha(alpha, retention="audit")
        records.append({
            "mask_id": f"mask_{ref.sha256[:20]}", "family": family, "role": role,
            "alpha_artifact": ref.to_dict(), "subject_artifact": subject_ref.to_dict(),
            "subject_area": area,
            **{key: round(value, 8) for key, value in stats.items()}, **meta,
            "alpha_projection": [
                round(float(value), 4) for value in np.asarray(Image.fromarray(
                    alpha, mode="F"
                ).resize((8, 8), getattr(Image, "Resampling", Image).BILINEAR))
                .reshape(-1)
            ],
            "stable_index": index,
        })
    validate_mask_bank(records, area)
    return records


def _validate_background_mask(mask: Mapping[str, Any]) -> None:
    """Runtime assertion of the four pre-registered background gate numbers."""
    checks = (
        ("subject_alpha_mean", "subject_alpha_mean_max", False),
        ("subject_high_coverage", "subject_high_coverage_max", False),
        ("background_alpha_mean", "background_alpha_mean_min", True),
        ("half_area", "half_area_min", True),
    )
    for column, bound, is_minimum in checks:
        if column not in mask:
            raise CandidateError(f"background mask is missing {column}")
        value = float(mask[column])
        limit = BACKGROUND_ROLE_GATE[bound]
        if (value < limit) if is_minimum else (value > limit):
            raise CandidateError(
                f"background mask {column}={value:.6f} violates {bound}={limit}"
            )


def validate_mask_bank(masks: Sequence[Mapping[str, Any]], subject_area: float) -> None:
    mask_ids = [str(mask.get("mask_id") or "") for mask in masks]
    if not all(mask_ids) or len(mask_ids) != len(set(mask_ids)):
        raise CandidateError("Mask v2 IDs must be nonempty and unique")
    counts: dict[str, int] = {}
    for mask in masks:
        family = str(mask["family"])
        role = str(mask.get("role") or "subject")
        if role not in MASK_ROLES:
            raise CandidateError(f"unknown mask role: {role}")
        # B8 item 2 runtime assertion of the pre-registered band geometry gate; it
        # covers both roles, so it runs before the background short circuit.
        if family == "band":
            for column, bound, is_minimum in (
                ("band_min_width", "min_width_short", True),
                ("band_aspect", "aspect_max", False),
            ):
                if column not in mask:
                    raise CandidateError(f"band mask is missing {column}")
                value = float(mask[column])
                limit = BAND_GEOMETRY_GATE[bound]
                if (value < limit) if is_minimum else (value > limit):
                    raise CandidateError(
                        f"band mask {column}={value:.6f} violates {bound}={limit}"
                    )
        if role == "background":
            _validate_background_mask(mask)
            continue
        counts[family] = counts.get(family, 0) + 1
        if family == "semantic" and subject_area < 0.15:
            raise CandidateError("semantic mask on a small subject")
        if family in {"radial", "band"}:
            if float(mask["effective_alpha_mean"]) <= 0.45:
                raise CandidateError("context mask alpha mean gate")
            if float(mask["subject_high_coverage"]) < 0.98:
                raise CandidateError("context mask subject high-coverage gate")
            if float(mask["subject_support_coverage"]) < 1.0:
                raise CandidateError("context mask subject support gate")
        # B7 item 4 runtime assertion of the pre-registered subject band area floor.
        if family == "band" and \
                float(mask["half_area"]) < SUBJECT_BAND_GATE["half_area_min"]:
            raise CandidateError(
                f"subject band half_area={float(mask['half_area']):.6f} violates "
                f"half_area_min={SUBJECT_BAND_GATE['half_area_min']}"
            )
    expected = {"radial": 2, "band": 2, "linear": 2, "semantic": 1} \
        if subject_area >= 0.15 else {"radial": 3, "band": 3, "linear": 2}
    # B7 item 4: band slots may be dropped by the area floor, so band is the one
    # family whose count is a ceiling; every other family stays exact.
    if counts.get("band", 0) > expected["band"]:
        raise CandidateError(f"Mask v2 composition mismatch: {counts}")
    if {name: value for name, value in counts.items() if name != "band"} != \
            {name: value for name, value in expected.items() if name != "band"}:
        raise CandidateError(f"Mask v2 composition mismatch: {counts}")


def _projection_position(mask: Mapping[str, Any]) -> str:
    """Coarse 3x3 position of the alpha mass, read off the stored 8x8 projection."""
    values = np.asarray(mask.get("alpha_projection") or (), dtype=np.float64)
    if values.size != 64 or float(values.sum()) <= 0.0:
        return "center"
    grid = values.reshape(8, 8)
    axis = np.arange(8, dtype=np.float64)
    total = float(grid.sum())
    center_x = float((grid.sum(axis=0) * axis).sum() / total)
    center_y = float((grid.sum(axis=1) * axis).sum() / total)

    def third(value: float, low: str, high: str) -> str:
        if value < 7.0 / 3.0:
            return low
        return high if value > 14.0 / 3.0 else ""

    parts = [third(center_y, "top", "bottom"), third(center_x, "left", "right")]
    return "-".join(part for part in parts if part) or "center"


def region_descriptor(mask: Mapping[str, Any]) -> str:
    """Low-granularity region key `family:direction_or_position` for commit diversity.

    The human-readable `center_hint` is shared by whole mask families (every radial
    and every band carries the identical sentence), so it cannot separate geometries.
    """
    family = str(mask.get("family") or "unknown")
    role = str(mask.get("role") or "subject")
    prefix = "" if role == "subject" else f"{role}:"
    direction = str(mask.get("direction") or "")
    if direction and direction != "elliptical":
        return f"{prefix}{family}:{direction}"
    return f"{prefix}{family}:{_projection_position(mask)}"


def mask_summary(mask: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"role": str(mask.get("role") or "subject")}
    summary.update({
        key: mask[key] for key in (
            "mask_id", "family", "direction", "effective_alpha_mean",
            "subject_high_coverage", "subject_support_coverage", "center_hint",
        )
    })
    return summary


def _mask_role(row: Mapping[str, Any]) -> str:
    return str(row.get("role") or "subject")


def _packet_candidates(
    rows: Sequence[Mapping[str, Any]], proposal_id: str, packet_size: int, accept: Any,
) -> list[tuple[int, str, tuple[Mapping[str, Any], ...]]]:
    """Enumerate admissible sibling packets; family diversity stays the tie-break."""
    candidates = []
    for combo in itertools.combinations(rows, min(packet_size, len(rows))):
        families = {str(row["family"]) for row in combo}
        if sum(row["family"] == "semantic" for row in combo) > 1 \
                or not accept(combo, families):
            continue
        projections = [np.asarray(row["alpha_projection"], dtype=np.float32) for row in combo]
        if any(float(np.abs(left - right).mean()) < 0.02
               for left, right in itertools.combinations(projections, 2)):
            continue
        digest = hashlib.sha256(
            (proposal_id + "\x1f" + "\x1f".join(sorted(str(row["mask_id"]) for row in combo))).encode()
        ).hexdigest()
        candidates.append((-len(families), digest, combo))
    return candidates


def _split_packet(
    rows: Sequence[dict[str, Any]],
    candidates: Sequence[tuple[int, str, tuple[Mapping[str, Any], ...]]],
    proposal_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    first = [dict(row) for row in min(candidates, key=lambda item: (item[0], item[1]))[2]]
    used = {row["mask_id"] for row in first}
    remaining = sorted(
        (row for row in rows if row["mask_id"] not in used),
        key=lambda row: hashlib.sha256((proposal_id + str(row["mask_id"])).encode()).hexdigest(),
    )
    return first, remaining


def allocate_mask_packets(
    masks: Sequence[Mapping[str, Any]], proposal_id: str, *, packet_size: int = 3
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [dict(row) for row in masks]

    def accept(combo: Sequence[Mapping[str, Any]], families: set[str]) -> bool:
        return len(families) >= min(2, len(combo))

    candidates = _packet_candidates(rows, proposal_id, packet_size, accept)
    if not candidates:
        raise CandidateError("cannot allocate a diverse local mask packet")
    return _split_packet(rows, candidates, proposal_id)


def allocate_role_packets(
    masks: Sequence[Mapping[str, Any]], proposal_id: str, *, packet_size: int = 3
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Role-aware sibling allocation, target mix 2 subject + 1 background.

    B6 item 2 (user-approved 2026-08-20) replaced the old ">=1 subject and >=1
    background, third free" rule with an exact 2:1 target; the >=1/>=1 rule is kept
    only as the relaxation used when the exact mix is not available. Replaces the
    >=2 family requirement of `allocate_mask_packets`; family diversity is kept only
    as the tie-break. When no background mask survived the mirrored gate the packet
    falls back to subject-only and records the reason.
    """
    rows = [dict(row) for row in masks]
    background = [row for row in rows if _mask_role(row) == "background"]
    subject = [row for row in rows if _mask_role(row) == "subject"]
    size = min(packet_size, len(rows))

    def target_accept(combo: Sequence[Mapping[str, Any]], _families: set[str]) -> bool:
        roles = [_mask_role(row) for row in combo]
        return roles.count("subject") == ROLE_PACKET_TARGET["subject"] \
            and roles.count("background") == ROLE_PACKET_TARGET["background"]

    def role_accept(combo: Sequence[Mapping[str, Any]], _families: set[str]) -> bool:
        roles = {_mask_role(row) for row in combo}
        return "subject" in roles and "background" in roles

    candidates: list[tuple[int, str, tuple[Mapping[str, Any], ...]]] = []
    target_met = False
    if background and size >= sum(ROLE_PACKET_TARGET.values()):
        candidates = _packet_candidates(rows, proposal_id, packet_size, target_accept)
        target_met = bool(candidates)
    if not candidates and background and size >= 2:
        candidates = _packet_candidates(rows, proposal_id, packet_size, role_accept)
    fallback = None
    if not candidates:
        fallback = "background_infeasible"

        def family_accept(combo: Sequence[Mapping[str, Any]], families: set[str]) -> bool:
            return len(families) >= min(2, len(combo))

        candidates = _packet_candidates(subject, proposal_id, packet_size, family_accept)
    if not candidates:
        raise CandidateError("cannot allocate a diverse local mask packet")
    first, remaining = _split_packet(rows, candidates, proposal_id)
    note = {
        "packet_size": len(first),
        "role_counts": {
            role: sum(1 for row in first if _mask_role(row) == role)
            for role in MASK_ROLES
        },
        "role_target": dict(ROLE_PACKET_TARGET),
        "role_target_met": bool(target_met),
        "bank_background_masks": len(background),
        "bank_subject_masks": len(subject),
        "fallback": fallback,
    }
    return first, remaining, note


__all__ = [
    "BACKGROUND_CENTER_HINT", "BACKGROUND_FAMILY_COUNTS", "BACKGROUND_ROLE_GATE",
    "BAND_GEOMETRY_GATE",
    "CandidateError", "FINGERPRINT_FIELDS", "GLOBAL_BIN_ORDER", "GLOBAL_BIN_QUOTA",
    "INTENT_BIN_QUOTA", "LOCAL_BIN_ORDER", "LOCAL_BIN_QUOTA", "MASK_REACH_GATE",
    "MASK_ROLES", "ROLE_PACKET_TARGET", "SUBJECT_BAND_GATE",
    "LutCatalog", "LutRecord", "allocate_mask_packets", "allocate_role_packets",
    "background_geometry_bank", "band_geometry_reading",
    "build_local_packets", "build_mask_bank",
    "intent_admits", "intent_offered", "load_cluster_map", "mask_summary",
    "offered_intents", "palette_summary", "region_descriptor", "validate_mask_bank",
]
