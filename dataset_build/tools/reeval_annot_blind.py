"""Stage B of the WP5 re-evaluation: blind, metadata-free annotation review.

What makes it *blind*: the judge receives BEFORE, AFTER, the region map when the
task has one, and exactly three annotation fields.  It never sees ``region``,
``subject``, ``style_name``, ``slot_mode``, ``task_type`` or the QA scores.  The
first-round review shipped all of those in a CONTEXT block, which handed the
consistency dimension its own ground truth and made a high consistency score
unfalsifiable.

Four sub-commands:

* ``b1`` -- score every current SFT sample once.  This is the dataset's state.
* ``b1o`` -- score the *pre-reannotation* text of the 44 with the same protocol,
  so the first-round-to-blind shift can be measured on text that did not change.
  Without it, regression to the mean is indistinguishable from a prompt gain.
* ``b2`` -- re-score a stratified subsample a second time, independently.  The
  agreement between the two passes is the judge's own noise floor; no claimed
  improvement smaller than it means anything.
* ``b3`` -- for the 44 re-annotated samples, show the old and the new text for
  the *same* images in one request with the sides randomised, and ask which is
  better per dimension.  Paired and blinded, so regression to the mean (which
  produced the original 44-fail -> 0-fail headline) cannot manufacture a win.

Usage:
  python -m dataset_build.tools.reeval_annot_blind b1
  python -m dataset_build.tools.reeval_annot_blind b1o
  python -m dataset_build.tools.reeval_annot_blind b2 --n 30
  python -m dataset_build.tools.reeval_annot_blind b3
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Mapping

from dataset_build.tools.reeval_relay import (
    JsonlStore, Task, data_url, load_lanes, read_results, run_tasks,
)

BUILD_ROOT = Path("/mnt/nfs/bc/data/builds/eval100-annotqa-20260727")
REVIEW_ROOT = Path("/var/cache/veradata/annot_review/eval100-annotqa-20260727")
DIMENSIONS = ("consistency", "reasoning_alignment", "language", "usability")
# The v4 contract adds a region_scope section, and with it a fifth dimension.  It
# is requested only for text that actually has that section, so every legacy row
# -- b1o and b3 above all -- is judged with the exact four-dimension rubric the B1
# baseline used and stays comparable to it.
REGION_SCOPE_DIM = "region_scope"
ALL_DIMENSIONS = (*DIMENSIONS, REGION_SCOPE_DIM)
REGION_SCOPE_TOKEN = "<region_scope_start>"

COMMON_RULES = """You are auditing training annotations for an image-retouching SFT dataset.

You receive: a BEFORE photo, the AFTER photo (the retouched result), sometimes a
REGION MAP (white = the area that was edited, black = untouched), and annotation
text. You receive no other metadata about the sample.

Rules you must follow:
- Judge only what you can see. Do not speculate about the photographer's or the
  dataset author's intent, and do not reward text that merely sounds plausible.
- If a change is too subtle for you to verify at this resolution, say so in your
  reason and score by what you can actually verify.
- A non-English preset or style name may legitimately appear verbatim inside an
  otherwise English instruction. That is not a language error.
- The reasoning field uses fixed section tokens (<problem_*_start>, <plan_*_start>).
  Their presence and order are verified mechanically elsewhere; ignore formatting
  entirely and judge only the content of those sections.

Dimensions, each scored 1-5 (5 best) with one sentence of justification:
1. consistency - does the instruction describe the actual BEFORE->AFTER change?
   Check direction (brighter/darker, warmer/cooler, more/less saturated,
   more/less contrast), spatial extent (does the named subject or area match
   where the picture actually changed - use the REGION MAP when one is given),
   and magnitude (does "slightly"/"modestly"/"strongly" match the real size of
   the change). 5 = direction, extent and magnitude all correct. 3 = direction
   correct but extent or magnitude wrong. 1 = direction inverted, or the named
   target is not where the edit actually happened.
2. reasoning_alignment - are the problems it claims actually visible in BEFORE,
   and do the plans match the edit that was actually applied? 5 = every claim
   checks out. 3 = mostly right with an unsupported or overstated claim.
   1 = the claimed problems are not visible, or a plan contradicts the AFTER image.
3. language - natural English in a real user's voice; instruction_short is a
   faithful condensation of instruction rather than a truncation; the two do not
   contradict each other.
4. usability - would you accept this as a training sample as it stands?
   5 = pass, 3 = borderline (needs editing before use), 1 = fail (must be
   re-annotated or dropped)."""

# Appended only when the annotation carries a region_scope section.
REGION_SCOPE_RULE = """
5. region_scope - the annotation's region_scope line names the subject and states
   how far the edit reaches. Judge that statement against the REGION MAP (or, with
   no map, against BEFORE->AFTER). 5 = subject and reach both match the white area,
   including whether the edit stays on the subject or extends past it into the
   background, and in which direction it extends. 3 = subject right but the reach
   overstated or understated. 1 = the declared area is not where the picture
   changed. An edit that visibly covers the whole frame and is declared as
   "global adjustment across the entire frame" scores 5."""

CHOICE = {"type": "string", "enum": ["A", "B", "tie"]}


def score_block(dimensions: tuple[str, ...]) -> dict:
    return {
        "type": "object",
        "properties": {dimension: {"type": "integer", "minimum": 1, "maximum": 5}
                       for dimension in dimensions},
        "required": list(dimensions),
        "additionalProperties": False,
    }


def single_schema(dimensions: tuple[str, ...]) -> dict:
    return {
        "type": "object",
        "properties": {
            "scores": score_block(dimensions),
            "reasons": {
                "type": "object",
                "properties": {dimension: {"type": "string", "minLength": 8}
                               for dimension in dimensions},
                "required": list(dimensions),
                "additionalProperties": False,
            },
            "verdict": {"type": "string", "enum": ["pass", "borderline", "fail"]},
        },
        "required": ["scores", "reasons", "verdict"],
        "additionalProperties": False,
    }


def pair_schema(dimensions: tuple[str, ...]) -> dict:
    return {
        "type": "object",
        "properties": {
            "scores_A": score_block(dimensions),
            "scores_B": score_block(dimensions),
            "preference": {
                "type": "object",
                "properties": {**{dimension: CHOICE for dimension in dimensions},
                               "overall": CHOICE},
                "required": [*dimensions, "overall"],
                "additionalProperties": False,
            },
            "reason": {"type": "string", "minLength": 20},
        },
        "required": ["scores_A", "scores_B", "preference", "reason"],
        "additionalProperties": False,
    }


def dimensions_for(*texts: Mapping[str, Any]) -> tuple[str, ...]:
    """Add the region_scope dimension only when every text under review has one."""
    if texts and all(REGION_SCOPE_TOKEN in (t.get("reasoning") or "") for t in texts):
        return ALL_DIMENSIONS
    return DIMENSIONS


def rules_for(dimensions: tuple[str, ...]) -> str:
    return COMMON_RULES + (REGION_SCOPE_RULE if REGION_SCOPE_DIM in dimensions else "")


SINGLE_SCHEMA = single_schema(DIMENSIONS)
PAIR_SCHEMA = pair_schema(DIMENSIONS)


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sample_index() -> dict[str, dict]:
    """sft_id -> its texts and the review-package image paths for its group."""
    sft = rows(BUILD_ROOT / "sft.jsonl")
    gdirs = {}
    for directory in sorted(REVIEW_ROOT.glob("g[0-9][0-9][0-9]_*")):
        entry = json.loads((directory / "annot.json").read_text())
        gdirs[entry["group_id"]] = directory
    index = {}
    for row in sft:
        directory = gdirs[row["group_id"]]
        rank = row.get("winner_rank")
        cgt = directory / f"cgt_rank{rank}.png"
        index[row["sft_id"]] = {
            "sft_id": row["sft_id"],
            "group_id": row["group_id"],
            "task_type": row["task_type"],
            "dir": directory,
            "before": directory / "before.jpg",
            "after": directory / f"after_rank{rank}.jpg",
            "cgt": cgt if cgt.is_file() else None,
            "instruction": row["instruction"],
            "instruction_short": row["instruction_short"],
            "reasoning": row["reasoning"],
        }
    return index


def annotation_block(instruction: str, short: str, reasoning: str) -> str:
    return json.dumps({"instruction": instruction, "instruction_short": short,
                       "reasoning": reasoning}, ensure_ascii=False, indent=1)


def build_single(entry: Mapping[str, Any], texts: Mapping[str, str]) -> tuple[list[dict], dict]:
    dimensions = dimensions_for(texts)
    order = ["BEFORE", "AFTER"] + (["REGION MAP"] if entry["cgt"] else [])
    header = (rules_for(dimensions) + "\n\nImages, in order: " + ", ".join(order)
              + "\n\nANNOTATION:\n" + annotation_block(
                  texts["instruction"], texts["instruction_short"], texts["reasoning"])
              + "\n\nReturn strict JSON only.")
    content: list[dict] = [{"type": "input_text", "text": header},
                           {"type": "input_image", "image_url": data_url(entry["before"])},
                           {"type": "input_image", "image_url": data_url(entry["after"])}]
    if entry["cgt"]:
        content.append({"type": "input_image", "image_url": data_url(entry["cgt"])})
    return content, single_schema(dimensions)


def build_pair(entry: Mapping[str, Any], text_a: Mapping[str, str],
               text_b: Mapping[str, str]) -> tuple[list[dict], dict]:
    dimensions = dimensions_for(text_a, text_b)
    order = ["BEFORE", "AFTER"] + (["REGION MAP"] if entry["cgt"] else [])
    header = (rules_for(dimensions) + "\n\nImages, in order: " + ", ".join(order)
              + "\n\nTwo candidate annotations describe the SAME edit shown above."
                f" Score each one on all {len(dimensions)} dimensions, then say"
                " which one is better on each dimension and overall. Use \"tie\""
                " only when you genuinely cannot separate them. Judge the text"
                " against the images, not against each other's style."
                "\n\nANNOTATION A:\n"
              + annotation_block(text_a["instruction"], text_a["instruction_short"],
                                 text_a["reasoning"])
              + "\n\nANNOTATION B:\n"
              + annotation_block(text_b["instruction"], text_b["instruction_short"],
                                 text_b["reasoning"])
              + "\n\nReturn strict JSON only.")
    content: list[dict] = [{"type": "input_text", "text": header},
                           {"type": "input_image", "image_url": data_url(entry["before"])},
                           {"type": "input_image", "image_url": data_url(entry["after"])}]
    if entry["cgt"]:
        content.append({"type": "input_image", "image_url": data_url(entry["cgt"])})
    return content, pair_schema(dimensions)


def _scored(keys: Any) -> set[str]:
    """The four base dimensions are mandatory; region_scope is optional."""
    scored = set(keys)
    if not set(DIMENSIONS) <= scored <= set(ALL_DIMENSIONS):
        raise ValueError(f"unexpected score dimensions: {sorted(scored)}")
    return scored


def validate_single(result: dict) -> dict:
    _scored(result.get("scores", {}))
    if any(not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 5
           for v in result["scores"].values()):
        raise ValueError("scores out of range")
    if result.get("verdict") not in {"pass", "borderline", "fail"}:
        raise ValueError("bad verdict")
    return result


def validate_pair(result: dict) -> dict:
    scored = _scored(result.get("scores_A", {}))
    if _scored(result.get("scores_B", {})) != scored:
        raise ValueError("scores_A and scores_B disagree on dimensions")
    if set(result.get("preference", {})) != {*scored, "overall"}:
        raise ValueError("incomplete preference")
    return result


def stratified(items: list[tuple[str, str, float]], n: int, seed: int) -> list[str]:
    """Pick ``n`` ids spread over task type and score, deterministically."""
    rng = random.Random(seed)
    picked: list[str] = []
    by_type: dict[str, list[tuple[str, float]]] = {}
    for sid, task_type, score in items:
        by_type.setdefault(task_type, []).append((sid, score))
    total = sum(len(v) for v in by_type.values())
    for task_type, members in sorted(by_type.items()):
        quota = max(1, round(n * len(members) / total))
        members = sorted(members, key=lambda kv: (kv[1], kv[0]))
        # three score bands so the retest spans the range, not just the middle
        bands = [members[i::3] for i in range(3)]
        per_band = [quota // 3 + (1 if i < quota % 3 else 0) for i in range(3)]
        for band, take in zip(bands, per_band):
            rng.shuffle(band)
            picked.extend(sid for sid, _ in band[:take])
    rng.shuffle(picked)
    return picked[:n]


def cmd_b1(args: argparse.Namespace) -> None:
    index = sample_index()
    store = JsonlStore(args.out / "b1_blind.jsonl")
    tasks = [Task(key=f"b1::{sid}", build=(lambda e=entry: build_single(e, e)),
                  record={"sft_id": sid, "task_type": entry["task_type"],
                          "group_id": entry["group_id"], "pass": 1})
             for sid, entry in sorted(index.items())]
    run(tasks, store, args, validate_single, "b1")


def cmd_b1o(args: argparse.Namespace) -> None:
    """Score the *pre-reannotation* text of the 44 with the exact b1 protocol.

    Without this the only blind reading of the old text comes from the paired b3
    request, and a paired score is not comparable to a single-item score.  With
    it, the first-round-to-blind shift can be measured on frozen text in the same
    format as the control group, which is what separates regression to the mean
    from a real prompt gain.
    """
    index = sample_index()
    comparison = json.loads((REVIEW_ROOT / "reannot_ab" / "final_comparison.json").read_text())
    store = JsonlStore(args.out / "b1o_oldtext.jsonl")
    tasks = []
    for pair in comparison:
        sid = pair["sft_id"]
        entry = index[sid]
        tasks.append(Task(
            key=f"b1o::{sid}",
            build=(lambda e=entry, t=pair["before"]: build_single(e, t)),
            record={"sft_id": sid, "task_type": entry["task_type"],
                    "group_id": entry["group_id"], "text_version": "old"},
        ))
    run(tasks, store, args, validate_single, "b1o")


def cmd_b2(args: argparse.Namespace) -> None:
    index = sample_index()
    first = {row["sft_id"]: row for row in read_results(args.out / "b1_blind.jsonl")
             if row.get("ok")}
    if len(first) < args.n:
        raise SystemExit("run b1 first; not enough completed b1 rows to stratify on")
    items = [(sid, index[sid]["task_type"],
              float(row["result"]["scores"]["usability"]))
             for sid, row in first.items() if sid in index]
    chosen = stratified(items, args.n, args.seed)
    (args.out / "b2_sample.json").write_text(json.dumps(sorted(chosen), indent=2))
    store = JsonlStore(args.out / "b2_retest.jsonl")
    tasks = [Task(key=f"b2::{sid}", build=(lambda e=index[sid]: build_single(e, e)),
                  record={"sft_id": sid, "task_type": index[sid]["task_type"],
                          "group_id": index[sid]["group_id"], "pass": 2})
             for sid in sorted(chosen)]
    run(tasks, store, args, validate_single, "b2")


def cmd_b3(args: argparse.Namespace) -> None:
    index = sample_index()
    comparison = json.loads((REVIEW_ROOT / "reannot_ab" / "final_comparison.json").read_text())
    store = JsonlStore(args.out / "b3_pairs.jsonl")
    mapping = {}
    tasks = []
    for pair in comparison:
        sid = pair["sft_id"]
        entry = index[sid]
        # deterministic per-sample side assignment, recorded so it can be undone
        rng = random.Random(f"{args.seed}:{sid}")
        old_is_a = rng.random() < 0.5
        side = {"A": "old" if old_is_a else "new", "B": "new" if old_is_a else "old"}
        mapping[sid] = side
        text_old = pair["before"]
        text_new = pair["after"]
        a, b = (text_old, text_new) if old_is_a else (text_new, text_old)
        tasks.append(Task(
            key=f"b3::{sid}",
            build=(lambda e=entry, x=a, y=b: build_pair(e, x, y)),
            record={"sft_id": sid, "task_type": entry["task_type"],
                    "group_id": entry["group_id"], "side_map": side},
        ))
    (args.out / "b3_sidemap.json").write_text(json.dumps(mapping, indent=2))
    run(tasks, store, args, validate_pair, "b3")


def run(tasks: list[Task], store: JsonlStore, args: argparse.Namespace,
        validate, label: str) -> None:
    lanes = load_lanes()
    if args.lane:
        lanes = [lane for lane in lanes if lane.lane_id in set(args.lane.split(","))]
    if args.limit:
        tasks = tasks[:args.limit]
    stats = run_tasks(
        tasks, lanes, store, model=args.model, effort=args.effort,
        per_lane=args.per_lane, attempts=args.attempts, schema_name=label,
        max_output_tokens=args.max_tokens, validate=validate,
        progress=args.out / f"progress_{label}.json",
    )
    (args.out / f"stats_{label}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({"stage": label, **stats}, ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["b1", "b1o", "b2", "b3"])
    # Required, with no default: the default used to be the WP5 evidence
    # directory, and these stages write relay answers that cost money and cannot
    # be reproduced, so an accidental bare re-run must not be able to land there.
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--effort", default="xhigh")
    parser.add_argument("--per-lane", type=int, default=2)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--lane", default=None, help="comma-separated lane ids to use")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--limit", type=int, default=0, help="smoke-test: cap task count")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    {"b1": cmd_b1, "b1o": cmd_b1o, "b2": cmd_b2, "b3": cmd_b3}[args.stage](args)


if __name__ == "__main__":
    main()
