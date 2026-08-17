"""Price the redundancy cost of ``[sources] max_source_uses`` before a build runs.

Loads the production preset bank and drives ONLY the allocation + coverage
selector — no rendering, no QA, no annotation, no writes anywhere — to answer the
one question the unit tests' toy catalog cannot: on the real bank, how often does
a reused source draw a preset it has already been rendered with?  Nothing in the
selector prevents that (its coverage counters are per-preset and global, with no
memory of which source used what), so the rate is an emergent property of bank
size against ``8 x max_source_uses`` and has to be measured per bank and budget.

The same three numbers land in every build's manifest under
``sources.source_reuse`` (``duplicate_source_preset_pairs`` /
``duplicate_source_preset_rate`` / ``max_pair_overlap``); this script is how you
see them *before* committing weeks of GPU time.

Measured 2026-08-12 on /var/cache/veradata/preset_bank_full (3522 presets,
10 majors, 85 minors) at the L8 shape ``--sources 26460 --uses 12``:
5.25% of draws repeat, worst pair overlap 7 of 8, zero whole-group repeats.

    PYTHONPATH=<repo>:<repo>/dataset_build:<repo>/dataset_build/src \
        python -m dataset_build.tools.source_reuse_dryrun \
        --sources 26460 --uses 12 --mode local

Caveat when reading the rate: this drives ``accept`` for every slot, while a real
build also calls ``reservation.reject`` when a candidate fails the visibility
gate and redraws.  A real build therefore consumes somewhat more presets per
group; treat the number as the order of magnitude for a given draw algorithm, not
as an exact prediction.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from construct.config import (
    AnnotationConfig,
    DatabuildConfig,
    ExternalEndpointConfig,
    LocalAnnotationConfig,
    MasksConfig,
    MixConfig,
    PresetsConfig,
    RenderConfig,
    SourcesConfig,
    ViewerConfig,
)
from construct.presets import CoverageSelector, PresetCatalog, PresetError
from construct.sources import SourceRecord, allocate_sources

BANK = Path("/var/cache/veradata/preset_bank_full")
# Same scene mix the stratified order uses; spread the fake pool over all of it
# so the allocation behaves as it would on a real inventory.
SCENES = ["portrait", "landscape", "food", "unknown", "still_life",
          "architecture", "night", "street", "wedding", "product"]


def make_config(*, uses: int, target: int, local: float) -> DatabuildConfig:
    return DatabuildConfig(
        schema_version=1,
        build_id="dryrun-source-reuse",
        seed=0,
        target_groups=target,
        output_root=Path("/tmp/dryrun-source-reuse-never-written"),
        preset_filter="all",
        mix=MixConfig(local=local, global_=1.0 - local),
        sources=SourcesConfig(Path("/tmp/none"), "postgresql://u:p@127.0.0.1:5432/x",
                              max_source_uses=uses),
        presets=PresetsConfig(BANK, BANK / "taxonomy.jsonl", 6.0, ()),
        render=RenderConfig(1024, 95, 2, 512, 2.5, 2.3, 0.5),
        masks=MasksConfig(0.5, 2),
        annotation=AnnotationConfig(
            "m", 768, 90, "low", 6000, 4, 3,
            (ExternalEndpointConfig("a", "https://a.example/v1", "k", 1),),
            LocalAnnotationConfig("http://127.0.0.1:8003/v1", "EMPTY",
                                  "qwen3_5-35b-a3b", 0.2, False, 2048),
        ),
        viewer=ViewerConfig("postgresql://u:p@127.0.0.1:5432/x"),
    )


def records(n: int) -> list[SourceRecord]:
    return [
        SourceRecord(
            source_id=f"dry{index:07d}",
            source_path=Path(f"/dev/null/{index}.jpg"),
            cache_dir=Path(f"/dev/null/{index}"),
            subject_path=Path(f"/dev/null/{index}/subject.png"),
            subject_meta_path=Path(f"/dev/null/{index}/subject.json"),
            scene=SCENES[index % len(SCENES)],
            subject={"name": "subject"},
            mask_area=0.2,
        )
        for index in range(n)
    ]


def slot_ids(mode: str) -> list[str]:
    if mode == "global":
        return [f"global-{i}" for i in range(8)]
    return ["radial-0", "radial-1", "semantic-0", "semantic-1",
            "band-0", "band-1", "linear-0", "linear-1"]


def draw_group(selector: CoverageSelector, source_id: str, use_index: int,
               slots: list[str]) -> list[str] | None:
    """Exactly what ``_render_source`` asks the selector for, minus the pixels."""
    excluded: list[str] = []
    for attempt in range(len(selector.majors)):
        try:
            reservation = selector.begin_group(
                source_id, attempt, exclude_majors=excluded, use_index=use_index
            )
        except PresetError:
            return None
        picked: list[str] = []
        try:
            for slot in slots:
                candidate = reservation.reserve_candidate(slot)
                if candidate is None:
                    raise PresetError("slot exhausted")
                reservation.accept(candidate)
                picked.append(candidate.link.preset.preset_id)
            reservation.commit()
            return picked
        except PresetError:
            excluded.append(reservation.major)
            reservation.abandon()
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=int, default=2000)
    parser.add_argument("--uses", type=int, default=12)
    parser.add_argument("--mode", default="local", choices=("local", "global"))
    args = parser.parse_args()

    local = 1.0 if args.mode == "local" else 0.0
    target = args.sources * args.uses
    config = make_config(uses=args.uses, target=target, local=local)

    started = time.perf_counter()
    catalog = PresetCatalog.load(config)
    counts = catalog.inventory_counts(args.mode)
    print(f"bank                : {BANK}")
    print(f"inventory ({args.mode:6s}): presets={counts['presets']} "
          f"majors={counts['majors']} minors={counts['minors']} "
          f"formats={counts['formats']}")
    print(f"catalog load        : {time.perf_counter() - started:.1f}s")

    allocation = allocate_sources(
        records(args.sources), build_id=config.build_id, seed=config.seed,
        target_groups=target, mix=config.mix, max_source_uses=args.uses,
    )
    pool = allocation.local if args.mode == "local" else allocation.global_
    print(f"allocation          : pool={len(pool)} target={target} uses={args.uses}")

    selector = CoverageSelector(
        catalog, build_id=config.build_id, seed=config.seed,
        render_mode=args.mode, preset_filter="all",
    )
    print(f"selector majors     : {len(selector.majors)} "
          f"(a major needs >=8 distinct presets)")

    slots = slot_ids(args.mode)
    drawn: dict[str, list[list[str]]] = {}
    majors: dict[str, list[str]] = {}
    groups = failed = 0
    started = time.perf_counter()
    for use_index in range(args.uses):
        for source in pool:
            if groups >= target:
                break
            picked = draw_group(selector, source.source_id, use_index, slots)
            if picked is None:
                failed += 1
                continue
            drawn.setdefault(source.source_id, []).append(picked)
            groups += 1
        if groups >= target:
            break
    elapsed = time.perf_counter() - started
    print(f"drew                : {groups} groups ({failed} refused) in {elapsed:.1f}s")

    draws = duplicate_pairs = max_overlap = exact_sets = same_major = 0
    overlaps: list[int] = []
    for source_id, items in drawn.items():
        draws += sum(len(item) for item in items)
        seen: dict[str, int] = {}
        for item in items:
            for preset_id in item:
                seen[preset_id] = seen.get(preset_id, 0) + 1
        duplicate_pairs += sum(n - 1 for n in seen.values() if n > 1)
        sets = [frozenset(item) for item in items]
        exact_sets += len(sets) - len(set(sets))
        for first in range(len(sets)):
            for second in range(first + 1, len(sets)):
                overlap = len(sets[first] & sets[second])
                overlaps.append(overlap)
                max_overlap = max(max_overlap, overlap)
    print()
    print("=== B2 numbers (real bank) ===")
    print(f"groups                          : {groups}")
    print(f"(source, preset) draws          : {draws}")
    print(f"duplicate_source_preset_pairs   : {duplicate_pairs}")
    print(f"duplicate_source_preset_rate    : "
          f"{duplicate_pairs / draws if draws else 0:.4%}")
    print(f"max_pair_overlap (of 8)         : {max_overlap}")
    print(f"exact identical 8-preset sets   : {exact_sets}")
    if overlaps:
        print(f"pairwise overlap mean           : {sum(overlaps) / len(overlaps):.3f}")
        print(f"pairs with any overlap          : "
              f"{sum(1 for o in overlaps if o)}/{len(overlaps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
