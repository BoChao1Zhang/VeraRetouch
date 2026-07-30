from pathlib import Path

from construct.presets import (
    CoverageSelector,
    PresetCatalog,
    PresetRecord,
    TaxonomyLink,
)
from construct.state import stable_id


BUILD_ID = "selector-parity"
SOURCE_ID = "source-0"
SLOTS = tuple(f"global-{index}" for index in range(8))


def _catalog() -> PresetCatalog:
    return PresetCatalog.from_links(
        TaxonomyLink(
            preset=PresetRecord(
                preset_id=f"preset-{index:02d}",
                path=Path(f"/unused/preset-{index:02d}.cube"),
                format="lut",
                kind="lut",
                style_name=f"Style {index:02d}",
                fidelity_de=None,
                render_engine="gpu_lut",
            ),
            major="major",
            minor="minor",
        )
        for index in range(12)
    )


def _selector() -> CoverageSelector:
    return CoverageSelector(
        _catalog(),
        build_id=BUILD_ID,
        seed=17,
        render_mode="global",
        preset_filter="all",
        historical_groups=(),
    )


def _failure(slot_id, candidate):
    group_id = stable_id("group", BUILD_ID, SOURCE_ID, "global", 0)
    task_id = stable_id(
        "render-attempt",
        group_id,
        slot_id,
        candidate.attempt,
        candidate.link.preset.preset_id,
    )
    return {
        "event_id": stable_id(
            "failure",
            BUILD_ID,
            "attempt",
            "rendering",
            task_id,
            "visibility_rejected",
            0,
            candidate.attempt,
            group_id,
            None,
            None,
        ),
        "event_type": "attempt",
        "stage": "rendering",
        "task_id": task_id,
        "group_id": group_id,
        "error_code": "visibility_rejected",
        "round": 0,
        "slot_id": slot_id,
        "attempt": candidate.attempt,
        "retryable": True,
        "terminal": False,
        "preset_id": candidate.link.preset.preset_id,
    }


def _row(candidate):
    return {
        "slot_id": candidate.slot_id,
        "preset_id": candidate.link.preset.preset_id,
        "preset_attempt": candidate.attempt,
        "preset_reservation_id": candidate.reservation_id,
    }


def _should_reject(candidate) -> bool:
    return candidate.slot_id in {"global-0", "global-3"} and candidate.attempt == 1


def _active_state(selector):
    return {
        "major": dict(selector._active_major),
        "minor": dict(selector._active_minor),
        "preset": dict(selector._active_preset),
        "reservations": dict(selector._reservations),
    }


def _run_serial():
    selector = _selector()
    reservation = selector.begin_group(SOURCE_ID, 0)
    accepted = []
    failures = []
    for slot_id in SLOTS:
        while True:
            candidate = reservation.reserve_candidate(slot_id)
            assert candidate is not None
            if _should_reject(candidate):
                failures.append(_failure(slot_id, candidate))
                reservation.reject(candidate)
                continue
            reservation.accept(candidate)
            accepted.append(_row(candidate))
            break
    coverage = reservation.commit()
    return accepted, failures, coverage, selector.snapshot(), _active_state(selector)


def _run_speculative():
    selector = _selector()
    reservation = selector.begin_group(SOURCE_ID, 0)
    pending = [reservation.reserve_candidate(slot_id) for slot_id in SLOTS]
    assert all(candidate is not None for candidate in pending)
    accepted = []
    failures = []
    index = 0
    while index < len(SLOTS):
        candidate = pending[index]
        assert candidate is not None
        while _should_reject(candidate):
            failures.append(_failure(candidate.slot_id, candidate))
            for later in reversed(pending[index + 1:]):
                assert later is not None
                reservation.cancel_speculative(later)
            del pending[index + 1:]
            reservation.reject(candidate)
            candidate = reservation.reserve_candidate(SLOTS[index])
            assert candidate is not None
        reservation.accept(candidate)
        accepted.append(_row(candidate))
        while len(pending) < len(SLOTS):
            replacement = reservation.reserve_candidate(SLOTS[len(pending)])
            assert replacement is not None
            pending.append(replacement)
        index += 1
    coverage = reservation.commit()
    return accepted, failures, coverage, selector.snapshot(), _active_state(selector)


def test_speculative_rollback_matches_serial_selector_and_failure_lineage():
    serial = _run_serial()
    speculative = _run_speculative()

    assert speculative == serial
    accepted, failures, _, _, active = speculative
    assert [row["slot_id"] for row in accepted] == list(SLOTS)
    assert [row["preset_attempt"] for row in accepted] == [2, 1, 1, 2, 1, 1, 1, 1]
    assert [(row["slot_id"], row["attempt"]) for row in failures] == [
        ("global-0", 1),
        ("global-3", 1),
    ]
    assert not active["reservations"]
    assert all(value == 0 for counter in active.values() if counter for value in counter.values())
