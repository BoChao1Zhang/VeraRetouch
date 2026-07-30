"""WP15a: per-row Direction List relabelling of the fresh200 panel.

One ``gpt-5.6-sol@xhigh`` call per SFT row produces two things:

1. ``observed`` -- what the judge itself sees between BEFORE and AFTER on the
   five WP14 psychophysics axes (brightness, warmth, green/magenta, saturation,
   contrast), each ``increased`` / ``decreased`` / ``not_visible`` with one
   sentence of evidence, plus up to three object-local observations.  These are
   the *labels* WP15b scores candidate metrics against.
2. ``claims`` -- every directional assertion extracted from that row's
   instruction + reasoning, each adjudicated
   ``correct`` / ``flipped`` / ``overstated`` / ``not_visible_in_image``.

The axis wording is copied from ``jnd_calib_20260729/tools/wp14-ask.py`` so the
labels and the WP14 JND thresholds describe the same perceptual questions.

Arms
  main      160 rows, images + annotation text, both sections.
  repeat    a seeded 20-row subset of ``main`` re-asked with the identical
            prompt -- test-retest noise.
  blindobs  a caller-named subset asked for section 1 only, with the annotation
            text withheld -- measures how far the text anchors the observations.

Transport rules follow WP14's lesson: ``per_lane=1``.  Three-in-flight per lane
produced 39 task failures there; one-in-flight produced 0/167.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/bc/VeraRetouch")
from dataset_build.tools.reeval_relay import (  # noqa: E402
    JsonlStore, Task, data_url, load_lanes, run_tasks,
)

REVIEW = Path("/var/cache/veradata/annot_review/fresh200-v4a1-20260729")
OUT = Path("/var/cache/veradata/annot_review/wp15_metric_roc/labels")
SEED = 20260730

AXES = {
    "brightness": (
        "overall lightness",
        '"increased" = AFTER is lighter / brighter overall; '
        '"decreased" = AFTER is darker overall'),
    "warmth": (
        "warm-cool colour balance (the yellow-blue axis)",
        '"increased" = AFTER is warmer, shifted toward yellow/amber; '
        '"decreased" = AFTER is cooler, shifted toward blue'),
    "green_magenta": (
        "green-magenta colour balance (the second hue axis, independent of warmth)",
        '"increased" = AFTER is shifted toward magenta / red / pink; '
        '"decreased" = AFTER is shifted toward green / cyan'),
    "saturation": (
        "colour saturation",
        '"increased" = colours in AFTER are more saturated / more vivid; '
        '"decreased" = colours in AFTER are less saturated / more muted'),
    "contrast": (
        "tonal contrast",
        '"increased" = AFTER has higher contrast, darks darker and lights lighter; '
        '"decreased" = AFTER has lower contrast, flatter and more compressed'),
}

DIRECTION = ["increased", "decreased", "not_visible"]
VERDICT = ["correct", "flipped", "overstated", "not_visible_in_image"]

REGION_RULE = (
    "The third image is a REGION MAP for this edit. White marks pixels that were "
    "adjusted, black marks pixels that are unchanged, grey marks pixels adjusted "
    "partially. Report only what happened inside the marked region; ignore "
    "everything outside it."
)
GLOBAL_RULE = (
    "There is no region map: this edit was applied to the whole frame, so report "
    "what happened across the whole image."
)


def _axis_block() -> str:
    lines = []
    for key, (name, meaning) in AXES.items():
        lines.append(f'- "{key}" -- {name}. {meaning}.')
    return "\n".join(lines)


def observed_rules() -> str:
    return "\n".join([
        "SECTION 1 -- WHAT YOU SEE.",
        "",
        "Compare AFTER with BEFORE and report, for each of these five independent "
        "properties, the direction of the change you can actually see:",
        "",
        _axis_block(),
        "",
        'Use "not_visible" when you cannot see a change along that property at this '
        "resolution -- that is a real and expected answer, not a failure. warmth and "
        "green_magenta are two separate axes: an edit can move on one and not the "
        "other, and a shift toward green is NOT the same thing as a shift toward blue.",
        "",
        "For each axis give one sentence of evidence naming the concrete thing you "
        "looked at (a surface, an object, a tonal region). Do not name a direction you "
        "cannot point at.",
        "",
        "Then list up to three OBJECT-LOCAL observations: a specific named object whose "
        "change is conspicuous on its own (lipstick, sky, foliage, skin, a jacket), the "
        "axis it moved on, and the direction. These are for salient local changes that "
        "an overall reading can miss; leave the list empty if nothing stands out.",
        "",
        "Judge only what is visible in these pixels. Do not infer the answer from what "
        "an edit of this kind usually does.",
    ])


def claim_rules() -> str:
    return "\n".join([
        "SECTION 2 -- THE ANNOTATION'S DIRECTIONAL CLAIMS.",
        "",
        "The text below was written to describe this same edit. Extract every claim it "
        "makes about the DIRECTION of a visual change -- brighter/darker, warmer/cooler, "
        "greener/more magenta, more/less saturated, more/less contrast -- including "
        "claims about a specific named object. Ignore non-directional material "
        "(region/scope declarations, style names, statements about what was preserved "
        "unchanged).",
        "",
        "Judge every claim against the images, using the same eyes as section 1:",
        '- "correct" -- the change is visible and goes the way the text says.',
        '- "flipped" -- the change is visible and goes the OPPOSITE way.',
        '- "overstated" -- the direction is right but the text asserts far more of it '
        "than the images show.",
        '- "not_visible_in_image" -- you cannot see any change along that property, so '
        "the claim asserts a direction that is not there to see.",
        "",
        "Quote the claim in at most 15 words. If the text makes no directional claim at "
        "all, return an empty list.",
    ])


def header(entry: dict, with_text: bool) -> str:
    order = ["BEFORE", "AFTER"] + (["REGION MAP"] if entry["cgt"] else [])
    parts = [
        "You are reporting the visible difference between two photographs for a "
        "measurement study. Accuracy about direction is the only thing that matters.",
        "",
        "Images, in order: " + ", ".join(order) + ".",
        "",
        REGION_RULE if entry["cgt"] else GLOBAL_RULE,
        "",
        observed_rules(),
    ]
    if with_text:
        parts += [
            "",
            claim_rules(),
            "",
            "Fill in section 1 from the images alone, before you read the text below. "
            "The text is evidence about the annotation, not about the pixels: if it "
            "disagrees with what you see, section 1 must keep saying what you see.",
            "",
            "ANNOTATION",
            "instruction: " + entry["instruction"],
            "",
            "instruction_short: " + entry["instruction_short"],
            "",
            "reasoning: " + entry["reasoning"],
        ]
    parts += ["", "Return strict JSON only."]
    return "\n".join(parts)


def schema(with_claims: bool) -> dict:
    axis_obj = {
        "type": "object",
        "properties": {
            "direction": {"type": "string", "enum": DIRECTION},
            "evidence": {"type": "string"},
        },
        "required": ["direction", "evidence"],
        "additionalProperties": False,
    }
    props = {
        "observed": {
            "type": "object",
            "properties": {key: dict(axis_obj) for key in AXES},
            "required": list(AXES),
            "additionalProperties": False,
        },
        "local_observations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "object": {"type": "string"},
                    "axis": {"type": "string", "enum": list(AXES)},
                    "direction": {"type": "string", "enum": ["increased", "decreased"]},
                },
                "required": ["object", "axis", "direction"],
                "additionalProperties": False,
            },
        },
    }
    required = ["observed", "local_observations"]
    if with_claims:
        props["claims"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "quote": {"type": "string"},
                    "object": {"type": "string"},
                    "axis": {"type": "string", "enum": [*AXES, "other"]},
                    "claimed_direction": {"type": "string",
                                          "enum": ["increased", "decreased"]},
                    "verdict": {"type": "string", "enum": VERDICT},
                    "note": {"type": "string"},
                },
                "required": ["quote", "object", "axis", "claimed_direction",
                             "verdict", "note"],
                "additionalProperties": False,
            },
        }
        required.append("claims")
    return {"type": "object", "properties": props, "required": required,
            "additionalProperties": False}


def content_for(entry: dict, with_text: bool) -> tuple[list[dict], dict]:
    parts: list[dict] = [{"type": "input_text", "text": header(entry, with_text)},
                         {"type": "input_image", "image_url": data_url(entry["before"])},
                         {"type": "input_image", "image_url": data_url(entry["after"])}]
    if entry["cgt"]:
        parts.append({"type": "input_image", "image_url": data_url(entry["cgt"])})
    return parts, schema(with_text)


def validate_factory(with_claims: bool):
    def validate(result: dict) -> dict:
        observed = result.get("observed") or {}
        for key in AXES:
            cell = observed.get(key) or {}
            if cell.get("direction") not in DIRECTION:
                raise ValueError(f"bad direction on {key}")
            if len(str(cell.get("evidence") or "")) < 8:
                raise ValueError(f"empty evidence on {key}")
        local = result.get("local_observations")
        if not isinstance(local, list):
            raise ValueError("local_observations not a list")
        if len(local) > 3:
            result["local_observations"] = local[:3]
        if with_claims:
            if not isinstance(result.get("claims"), list):
                raise ValueError("claims not a list")
            for claim in result["claims"]:
                if claim.get("verdict") not in VERDICT:
                    raise ValueError("bad claim verdict")
        return result
    return validate


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def index() -> dict[str, dict]:
    """sft_id -> texts + the review-package images the blind judge was shown."""
    gdirs = {}
    for directory in sorted(REVIEW.glob("g[0-9][0-9][0-9]_*")):
        gdirs[json.loads((directory / "annot.json").read_text())["group_id"]] = directory
    modes = {r["sft_id"]: r["slot_mode"] for r in rows(REVIEW / "blind" / "sample_index.jsonl")}
    out: dict[str, dict] = {}
    for row in rows(REVIEW / "run" / "sft.jsonl"):
        directory = gdirs[row["group_id"]]
        rank = row["winner_rank"]
        cgt = directory / f"cgt_rank{rank}.png"
        out[row["sft_id"]] = {
            "sft_id": row["sft_id"], "group_id": row["group_id"],
            "candidate_id": row["candidate_id"], "task_type": row["task_type"],
            "slot_mode": modes.get(row["sft_id"]), "winner_rank": rank,
            "before": directory / "before.jpg",
            "after": directory / f"after_rank{rank}.jpg",
            "cgt": cgt if cgt.is_file() else None,
            "instruction": row["instruction"],
            "instruction_short": row["instruction_short"],
            "reasoning": row["reasoning"],
        }
    return out


def repeat_ids(entries: dict[str, dict], n: int) -> list[str]:
    """Seeded subset, stratified over slot_mode so every slot is represented."""
    by_mode: dict[str, list[str]] = {}
    for sid, entry in entries.items():
        by_mode.setdefault(entry["slot_mode"], []).append(sid)
    rng = random.Random(SEED)
    picked: list[str] = []
    modes = sorted(by_mode)
    per = max(1, n // len(modes))
    for mode in modes:
        pool = sorted(by_mode[mode])
        rng.shuffle(pool)
        picked.extend(pool[:per])
    rest = sorted(set(entries) - set(picked))
    rng.shuffle(rest)
    picked.extend(rest[: max(0, n - len(picked))])
    return picked[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=["main", "repeat", "blindobs", "probe", "plan"])
    parser.add_argument("--ids", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--repeat-n", type=int, default=20)
    parser.add_argument("--per-lane", type=int, default=1)
    parser.add_argument("--chunk", type=int, default=12)
    # 12 per-attempt errors inside the window is ~0.5/row over two chunks, well
    # above the 0.321/row measured while running strictly serial.
    parser.add_argument("--fallback-errors", type=int, default=12)
    parser.add_argument("--fallback-task-failures", type=int, default=3)
    parser.add_argument("--fallback-window", type=float, default=600.0)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=14000)
    args = parser.parse_args()

    entries = index()
    if args.arm == "plan":
        print(json.dumps({"rows": len(entries), "with_cgt": sum(
            1 for e in entries.values() if e["cgt"]), "repeat": repeat_ids(entries, args.repeat_n)},
            indent=1))
        print(header(entries[sorted(entries)[0]], True))
        return

    with_text = args.arm != "blindobs"
    if args.arm in ("main", "probe"):
        ids = sorted(entries)
    elif args.arm == "repeat":
        ids = repeat_ids(entries, args.repeat_n)
    else:
        ids = [line.strip() for line in args.ids.read_text().splitlines() if line.strip()]
    if args.arm == "probe":
        # one local row and one style row, so both prompt shapes are exercised
        ids = ([i for i in ids if entries[i]["cgt"]][:1]
               + [i for i in ids if not entries[i]["cgt"]][:1])
    if args.limit:
        ids = ids[: args.limit]

    order = random.Random(SEED + len(args.arm)).sample(range(len(ids)), len(ids))
    tasks = []
    for position in order:
        sid = ids[position]
        entry = entries[sid]
        tasks.append(Task(
            key=f"{args.arm}::{sid}",
            build=(lambda e=entry, t=with_text: content_for(e, t)),
            record={"arm": args.arm, "sft_id": sid, "group_id": entry["group_id"],
                    "task_type": entry["task_type"], "slot_mode": entry["slot_mode"],
                    "winner_rank": entry["winner_rank"], "with_text": with_text},
        ))
    OUT.mkdir(parents=True, exist_ok=True)
    store = JsonlStore(OUT / f"{args.arm}.jsonl")
    stats = drive_adaptive(tasks, store, args, with_text)
    print(json.dumps(stats, indent=1))


def drive_adaptive(tasks, store, args, with_text: bool) -> dict:
    """Run the batch in chunks so concurrency can be walked back mid-run.

    The relay's failure mode is a stream that ends without ``response.completed``
    (WP14's "status=None"), and it is concurrency-sensitive: WP14 saw 39 such
    failures at six in flight and none at two.  Four in flight is the untested
    middle, so this driver watches the lanes and drops to ``per_lane=1`` for the
    remainder when the failure signal exceeds a bar.  Chunking is what makes that
    possible: ``run_tasks`` joins its threads before returning, so the only place
    concurrency can change is between chunks.

    Two bars, because a single retried call is *not* WP14's failure.  This relay
    retries roughly a quarter of these long xhigh calls at any concurrency: the
    measured per-attempt error rate on this batch was **0.321/row at per_lane=1**
    and **0.125/row at per_lane=2**, with **zero** exhausted tasks at either.  A
    trigger on raw per-attempt errors therefore fires on the serial baseline and
    cannot discriminate.  So the primary bar is ``--fallback-task-failures``
    (tasks that burned every attempt -- exactly what WP14 counted), and the
    secondary bar on per-attempt errors sits well above the measured serial rate.
    """
    lanes = load_lanes()
    per_lane = args.per_lane
    recent: list[tuple[float, int]] = []
    totals = {"total": len(tasks), "skipped": 0, "ok": 0, "failed": 0,
              "substituted": 0, "chunks": [], "fell_back": False}
    pending = [task for task in tasks if task.key not in store.done]
    totals["skipped"] = len(tasks) - len(pending)
    size = max(1, args.chunk)
    for start in range(0, len(pending), size):
        chunk = pending[start:start + size]
        before = sum(lane.errors for lane in lanes)
        stats = run_tasks(
            chunk, lanes, store, model="gpt-5.6-sol", effort="xhigh",
            per_lane=per_lane, attempts=args.attempts, schema_name="direction_list",
            max_output_tokens=args.max_tokens, validate=validate_factory(with_text),
            progress=OUT / f"progress_{args.arm}.json")
        delta = sum(lane.errors for lane in lanes) - before
        now = time.time()
        recent.append((now, delta, stats["failed"], len(chunk)))
        recent = [row for row in recent if now - row[0] <= args.fallback_window]
        window_errors = sum(row[1] for row in recent)
        window_failures = sum(row[2] for row in recent)
        window_rows = sum(row[3] for row in recent)
        for key in ("ok", "failed", "substituted"):
            totals[key] += stats[key]
        totals["chunks"].append({
            "n": len(chunk), "per_lane": per_lane, "errors": delta,
            "window_errors": window_errors, "window_failures": window_failures,
            "ok": stats["ok"], "failed": stats["failed"], "done": totals["ok"],
            "elapsed_s": round(now - recent[0][0], 1),
        })
        summary = {"done": totals["ok"] + totals["skipped"], "of": len(tasks),
                   "per_lane": per_lane, "chunk_errors": delta,
                   "window_errors": window_errors, "window_failures": window_failures,
                   "err_per_row": round(window_errors / max(window_rows, 1), 3)}
        print(json.dumps(summary), flush=True)
        trip = (window_failures >= args.fallback_task_failures
                or window_errors >= args.fallback_errors)
        if per_lane > 1 and trip:
            per_lane = 1
            totals["fell_back"] = True
            totals["fallback_reason"] = {
                "window_failures": window_failures, "window_errors": window_errors,
                "window_rows": window_rows}
            recent = []
            print(json.dumps({"event": "fallback_to_per_lane_1",
                              "window_failures": window_failures,
                              "window_errors": window_errors,
                              "window_rows": window_rows}), flush=True)
    totals["lanes"] = {lane.lane_id: {"calls": lane.calls,
                                      "substitutions": lane.substitutions,
                                      "errors": lane.errors} for lane in lanes}
    (OUT / f"stats_{args.arm}.json").write_text(json.dumps(totals, indent=2))
    return totals


if __name__ == "__main__":
    main()
