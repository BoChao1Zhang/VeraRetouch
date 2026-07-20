"""GPU-renderable preset inventory and transactional taxonomy coverage."""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .config import DatabuildConfig
from .state import stable_id


class PresetError(RuntimeError):
    """Raised when the configured preset inventory cannot satisfy the contract."""


@dataclass(frozen=True, slots=True)
class PresetRecord:
    preset_id: str
    path: Path
    format: str
    kind: str
    style_name: str | None
    fidelity_de: float | None
    render_engine: str


@dataclass(frozen=True, slots=True)
class TaxonomyLink:
    preset: PresetRecord
    major: str
    minor: str


@dataclass(frozen=True, slots=True)
class CapabilityResult:
    supported: bool
    engine: str
    reason: str = ""


@dataclass(frozen=True, slots=True)
class CandidateReservation:
    reservation_id: str
    slot_id: str
    attempt: int
    link: TaxonomyLink


def _format_of(feature: Mapping[str, Any]) -> str | None:
    kind = str(feature.get("kind") or "").lower()
    fmt = str(feature.get("fmt") or "").lower()
    suffix = Path(str(feature.get("path") or "")).suffix.lower()
    if kind in {"lut", "cube"} or fmt in {"cube", "3dl"} or suffix in {".cube", ".3dl"}:
        return "lut"
    if fmt == "lrtemplate" or suffix == ".lrtemplate":
        return "lrtemplate"
    if fmt == "xmp" or suffix == ".xmp":
        return "xmp"
    return None


def _has_embedded_local(feature: Mapping[str, Any]) -> bool:
    if feature.get("has_local_mask") or feature.get("has_ai_mask"):
        return True
    path = Path(str(feature.get("path") or ""))
    if path.suffix.lower() not in {".xmp", ".lrtemplate"}:
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return True
    markers = (
        "MaskGroupBasedCorrections",
        "CircularGradientBasedCorrections",
        "GradientBasedCorrections",
        "PaintBasedCorrections",
        "LocalCorrections",
    )
    return any(marker in text for marker in markers)


def default_capability(feature: Mapping[str, Any], fidelity_de: float | None,
                       fidelity_de_max: float) -> CapabilityResult:
    """Static local-GPU capability scan; runtime CUDA is checked by the renderer factory."""
    path = Path(str(feature.get("path") or ""))
    normalized = _format_of(feature)
    if normalized is None or not path.is_file():
        return CapabilityResult(False, "none", "missing or unsupported preset file")
    if _has_embedded_local(feature):
        return CapabilityResult(False, "none", "embedded local correction")
    if normalized == "lut":
        try:
            from dataset_build.lut_io import load_lut

            load_lut(path)
        except Exception as exc:  # noqa: BLE001 - malformed LUTs fail capability preflight
            return CapabilityResult(False, "none", f"LUT parse failed:{type(exc).__name__}")
        return CapabilityResult(True, "gpu_lut")
    if fidelity_de is None:
        return CapabilityResult(False, "none", "missing GPU fidelity calibration")
    if fidelity_de > fidelity_de_max:
        return CapabilityResult(False, "none", "GPU fidelity threshold")
    try:
        from gpu_render.route import route_preset

        route = route_preset(str(path), normalized, measured_de=fidelity_de,
                             threshold=fidelity_de_max)
    except Exception as exc:  # noqa: BLE001
        return CapabilityResult(False, "none", f"capability scan failed:{type(exc).__name__}")
    if route.get("route") != "local":
        return CapabilityResult(False, "none", "GPU route rejected preset")
    return CapabilityResult(True, "gpu_local_preset")


class PresetCatalog:
    def __init__(self, links: Iterable[TaxonomyLink], rejected: Mapping[str, int] | None = None):
        self.links = tuple(links)
        self.rejected = dict(rejected or {})
        tree: dict[str, dict[str, list[TaxonomyLink]]] = defaultdict(lambda: defaultdict(list))
        for link in self.links:
            tree[link.major][link.minor].append(link)
        self.tree = {
            major: {
                minor: tuple(sorted(rows, key=lambda row: row.preset.preset_id))
                for minor, rows in sorted(minors.items())
            }
            for major, minors in sorted(tree.items())
        }
        self.by_id = {link.preset.preset_id: link.preset for link in self.links}

    @classmethod
    def load(
        cls,
        config: DatabuildConfig,
        *,
        capability: Callable[[Mapping[str, Any], float | None, float], CapabilityResult]
        | None = None,
    ) -> "PresetCatalog":
        capability = capability or default_capability
        bank_dir = config.presets.bank_dir
        feature_path = bank_dir / "features.jsonl"
        if not feature_path.is_file():
            raise PresetError(f"missing preset features: {feature_path}")
        names_path = bank_dir / "vlm_names.jsonl"
        fidelity_path = bank_dir / "perceptual_de.jsonl"
        names: dict[str, str] = {}
        if names_path.is_file():
            for row in _read_jsonl(names_path):
                name = row.get("vlm_name") or row.get("style_name")
                if row.get("preset_id") and isinstance(name, str) and name.strip():
                    names[str(row["preset_id"])] = name.strip()
        fidelity: dict[str, float] = {}
        if fidelity_path.is_file():
            for row in _read_jsonl(fidelity_path):
                value = row.get("de_mean", row.get("de_med"))
                if row.get("preset_id") and isinstance(value, (int, float)):
                    fidelity[str(row["preset_id"])] = float(value)

        requested = config.effective_formats
        features: dict[str, PresetRecord] = {}
        rejected: Counter[str] = Counter()
        for feature in _read_jsonl(feature_path):
            preset_id = str(feature.get("preset_id") or "")
            normalized = _format_of(feature)
            if not preset_id or normalized is None:
                rejected["invalid_feature"] += 1
                continue
            if normalized not in requested:
                rejected["format_filtered"] += 1
                continue
            de = fidelity.get(preset_id)
            result = capability(feature, de, config.presets.fidelity_de_max)
            if not result.supported:
                rejected[result.reason or "gpu_unsupported"] += 1
                continue
            style_name = names.get(preset_id)
            if not style_name:
                candidate_name = feature.get("style_name") or feature.get("display_name")
                style_name = candidate_name.strip() if isinstance(candidate_name, str) else None
            features[preset_id] = PresetRecord(
                preset_id=preset_id,
                path=Path(str(feature["path"])),
                format=normalized,
                kind=str(feature.get("kind") or normalized),
                style_name=style_name,
                fidelity_de=de,
                render_engine=result.engine,
            )

        links: list[TaxonomyLink] = []
        for row in _read_jsonl(config.presets.taxonomy):
            preset = features.get(str(row.get("preset_id") or ""))
            major = row.get("major")
            minor = row.get("minor")
            if preset is None:
                continue
            if not isinstance(major, str) or not major.strip() or not isinstance(minor, str) \
                    or not minor.strip():
                rejected["missing_taxonomy"] += 1
                continue
            links.append(TaxonomyLink(preset=preset, major=major.strip(), minor=minor.strip()))
        catalog = cls(links, rejected)
        catalog.validate(config)
        return catalog

    @classmethod
    def from_links(cls, links: Iterable[TaxonomyLink]) -> "PresetCatalog":
        return cls(links)

    def eligible_for_mode(self, render_mode: str) -> "PresetCatalog":
        if render_mode == "local":
            return self
        if render_mode != "global":
            raise ValueError(f"invalid render mode: {render_mode}")
        return PresetCatalog((link for link in self.links if link.preset.style_name), self.rejected)

    def validate(self, config: DatabuildConfig) -> None:
        if not self.links:
            raise PresetError("no GPU-renderable taxonomy presets remain after filtering")
        present = {link.preset.format for link in self.links}
        if config.preset_filter != "all" and config.preset_filter not in present:
            raise PresetError(
                f"requested preset format {config.preset_filter!r} has no eligible inventory"
            )
        if config.preset_filter == "all" and not present:
            raise PresetError("preset_filter=all has no effective inventory")
        if config.mix.global_ > 0 and not self.eligible_for_mode("global").links:
            raise PresetError("global target requires at least one named eligible preset")

    def inventory_counts(self, render_mode: str) -> dict[str, Any]:
        selected = self.eligible_for_mode(render_mode)
        unique = {link.preset.preset_id: link.preset for link in selected.links}
        return {
            "presets": len(unique),
            "formats": dict(sorted(Counter(row.format for row in unique.values()).items())),
            "majors": len(selected.tree),
            "minors": sum(len(minors) for minors in selected.tree.values()),
            "rejected": dict(sorted(self.rejected.items())),
        }


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PresetError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise PresetError(f"non-object JSONL record at {path}:{line_number}")
            yield row


def _seeded_order(items: Iterable[str], *seed_parts: object) -> list[str]:
    items = sorted(set(items))
    digest = hashlib.sha256("\x1f".join(map(str, seed_parts)).encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    rng.shuffle(items)
    return items


class CoverageSelector:
    """Thread-safe group reservations whose counters commit with complete groups only."""

    def __init__(
        self,
        catalog: PresetCatalog,
        *,
        build_id: str,
        seed: int,
        render_mode: str,
        preset_filter: str,
        historical_groups: Iterable[Mapping[str, Any]] = (),
    ):
        self.catalog = catalog.eligible_for_mode(render_mode)
        self.build_id = build_id
        self.seed = seed
        self.render_mode = render_mode
        self.preset_filter = preset_filter
        self.namespace = (build_id, render_mode, preset_filter)
        self._lock = threading.RLock()
        self._major_success: Counter[str] = Counter()
        self._minor_success: Counter[tuple[str, str]] = Counter()
        self._preset_success: Counter[str] = Counter()
        self._active_major: Counter[str] = Counter()
        self._active_minor: Counter[tuple[str, str]] = Counter()
        self._active_preset: Counter[str] = Counter()
        self._reservations: dict[str, GroupReservation] = {}
        for group in historical_groups:
            if group.get("build_id") != build_id or group.get("render_mode") != render_mode \
                    or group.get("preset_filter") != preset_filter:
                continue
            major = str(group.get("major") or "")
            if not major:
                continue
            self._major_success[major] += 1
            for candidate in group.get("candidates") or []:
                if candidate.get("major") != major:
                    raise PresetError("historical group crosses taxonomy majors")
                minor = str(candidate.get("minor") or "")
                preset_id = str(candidate.get("preset_id") or "")
                if minor and preset_id:
                    self._minor_success[(major, minor)] += 1
                    self._preset_success[preset_id] += 1
        self.majors = tuple(
            major for major, minors in self.catalog.tree.items()
            if len({link.preset.preset_id for rows in minors.values() for link in rows}) >= 8
        )
        if not self.majors:
            raise PresetError(f"{render_mode} inventory has no major with eight distinct presets")

    def begin_group(self, source_id: str, group_attempt: int,
                    exclude_majors: Iterable[str] = ()) -> "GroupReservation":
        with self._lock:
            excluded = set(exclude_majors)
            pool = [major for major in self.majors if major not in excluded]
            if not pool:
                raise PresetError("no untried major can satisfy this group")
            usage = {
                major: self._major_success[major] + self._active_major[major]
                for major in pool
            }
            minimum = min(usage.values())
            cycle = minimum
            full_bag = _seeded_order(
                self.majors,
                *self.namespace, self.seed, "major", cycle,
            )
            major = next(item for item in full_bag if item in pool and usage[item] == minimum)
            reservation_id = stable_id(
                "reservation", *self.namespace, source_id, group_attempt, major
            )
            if reservation_id in self._reservations:
                raise PresetError(f"duplicate active reservation: {reservation_id}")
            self._active_major[major] += 1
            reservation = GroupReservation(
                selector=self,
                reservation_id=reservation_id,
                source_id=source_id,
                group_attempt=group_attempt,
                major=major,
                coverage_cycle=cycle,
                coverage_position=full_bag.index(major),
            )
            self._reservations[reservation_id] = reservation
            return reservation

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "namespace": self.namespace,
                "major": dict(sorted(self._major_success.items())),
                "minor": {
                    f"{major}/{minor}": count
                    for (major, minor), count in sorted(self._minor_success.items())
                },
                "preset": dict(sorted(self._preset_success.items())),
            }


class GroupReservation:
    def __init__(
        self,
        *,
        selector: CoverageSelector,
        reservation_id: str,
        source_id: str,
        group_attempt: int,
        major: str,
        coverage_cycle: int,
        coverage_position: int,
    ):
        self.selector = selector
        self.reservation_id = reservation_id
        self.source_id = source_id
        self.group_attempt = group_attempt
        self.major = major
        self.coverage_cycle = coverage_cycle
        self.coverage_position = coverage_position
        self._slot_minor: dict[str, str] = {}
        self._slot_tried_minors: dict[str, set[str]] = defaultdict(set)
        self._slot_attempts: Counter[str] = Counter()
        self._pending: dict[str, CandidateReservation] = {}
        self._accepted: dict[str, CandidateReservation] = {}
        self._attempted_group_ids: set[str] = set()
        self._closed = False

    @property
    def accepted(self) -> tuple[CandidateReservation, ...]:
        return tuple(self._accepted[key] for key in sorted(self._accepted))

    def reserve_candidate(self, slot_id: str) -> CandidateReservation | None:
        selector = self.selector
        with selector._lock:
            self._ensure_open()
            if slot_id in self._accepted:
                raise PresetError(f"slot already accepted: {slot_id}")
            if slot_id in self._pending:
                raise PresetError(f"slot already has a pending candidate: {slot_id}")
            while True:
                minor = self._slot_minor.get(slot_id)
                if minor is None:
                    minor = self._choose_minor(slot_id)
                    if minor is None:
                        return None
                    self._slot_minor[slot_id] = minor
                    selector._active_minor[(self.major, minor)] += 1
                candidates = self._candidate_bag(minor)
                if candidates:
                    link = candidates[0]
                    self._slot_attempts[slot_id] += 1
                    reservation = CandidateReservation(
                        reservation_id=stable_id(
                            "preset-reservation", self.reservation_id, slot_id,
                            self._slot_attempts[slot_id], link.preset.preset_id, minor,
                        ),
                        slot_id=slot_id,
                        attempt=self._slot_attempts[slot_id],
                        link=link,
                    )
                    self._pending[slot_id] = reservation
                    self._attempted_group_ids.add(link.preset.preset_id)
                    selector._active_preset[link.preset.preset_id] += 1
                    return reservation
                selector._active_minor[(self.major, minor)] -= 1
                self._slot_tried_minors[slot_id].add(minor)
                self._slot_minor.pop(slot_id, None)

    def reject(self, candidate: CandidateReservation) -> None:
        selector = self.selector
        with selector._lock:
            self._ensure_open()
            pending = self._pending.get(candidate.slot_id)
            if pending != candidate:
                raise PresetError("candidate is not pending for this reservation")
            selector._active_preset[candidate.link.preset.preset_id] -= 1
            self._pending.pop(candidate.slot_id)

    def accept(self, candidate: CandidateReservation) -> None:
        selector = self.selector
        with selector._lock:
            self._ensure_open()
            pending = self._pending.get(candidate.slot_id)
            if pending != candidate:
                raise PresetError("candidate is not pending for this reservation")
            self._pending.pop(candidate.slot_id)
            self._accepted[candidate.slot_id] = candidate

    def commit(self) -> dict[str, Any]:
        selector = self.selector
        with selector._lock:
            self._ensure_open()
            if self._pending:
                raise PresetError("cannot commit with pending candidates")
            if len(self._accepted) != 8:
                raise PresetError("coverage commits require exactly eight accepted candidates")
            preset_ids = [candidate.link.preset.preset_id for candidate in self._accepted.values()]
            if len(set(preset_ids)) != 8:
                raise PresetError("coverage commit contains duplicate preset IDs")
            selector._active_major[self.major] -= 1
            selector._major_success[self.major] += 1
            for candidate in self._accepted.values():
                minor = candidate.link.minor
                preset_id = candidate.link.preset.preset_id
                selector._active_minor[(self.major, minor)] -= 1
                selector._active_preset[preset_id] -= 1
                selector._minor_success[(self.major, minor)] += 1
                selector._preset_success[preset_id] += 1
            selector._reservations.pop(self.reservation_id, None)
            self._closed = True
            return {
                "major": self.major,
                "coverage_cycle": self.coverage_cycle,
                "coverage_position": self.coverage_position,
                "reservation_id": self.reservation_id,
            }

    def abandon(self) -> None:
        selector = self.selector
        with selector._lock:
            if self._closed:
                return
            selector._active_major[self.major] -= 1
            for minor in self._slot_minor.values():
                selector._active_minor[(self.major, minor)] -= 1
            for candidate in self._pending.values():
                selector._active_preset[candidate.link.preset.preset_id] -= 1
            for candidate in self._accepted.values():
                selector._active_preset[candidate.link.preset.preset_id] -= 1
            selector._reservations.pop(self.reservation_id, None)
            self._closed = True

    def _choose_minor(self, slot_id: str) -> str | None:
        selector = self.selector
        tried = self._slot_tried_minors[slot_id]
        minors = [minor for minor in selector.catalog.tree[self.major] if minor not in tried]
        if not minors:
            return None
        viable = []
        for minor in minors:
            if any(link.preset.preset_id not in self._attempted_group_ids
                   for link in selector.catalog.tree[self.major][minor]):
                viable.append(minor)
        if not viable:
            return None
        usage = {
            minor: selector._minor_success[(self.major, minor)]
            + selector._active_minor[(self.major, minor)]
            for minor in viable
        }
        minimum = min(usage.values())
        bag = _seeded_order(
            selector.catalog.tree[self.major],
            *selector.namespace, selector.seed, self.major, "minor", minimum,
        )
        return next(minor for minor in bag if minor in usage and usage[minor] == minimum)

    def _candidate_bag(self, minor: str) -> list[TaxonomyLink]:
        selector = self.selector
        links = [
            link for link in selector.catalog.tree[self.major][minor]
            if link.preset.preset_id not in self._attempted_group_ids
        ]
        if not links:
            return []
        minimum = min(
            selector._preset_success[link.preset.preset_id]
            + selector._active_preset[link.preset.preset_id]
            for link in links
        )
        ids = _seeded_order(
            (link.preset.preset_id for link in selector.catalog.tree[self.major][minor]),
            *selector.namespace, selector.seed, self.major, minor, "preset", minimum,
        )
        by_id = {link.preset.preset_id: link for link in links}
        return [
            by_id[preset_id] for preset_id in ids
            if preset_id in by_id
            and selector._preset_success[preset_id] + selector._active_preset[preset_id] == minimum
        ]

    def _ensure_open(self) -> None:
        if self._closed:
            raise PresetError("group reservation is closed")
