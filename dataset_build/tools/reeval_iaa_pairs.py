"""Stage C of the WP5 re-evaluation: is the OneAlign winner actually the best?

The build picks one of eight rendered candidates per group by OneAlign score and
never checked whether that ranking tracks anything a viewer would agree with.
This tool asks directly, in the only form that can answer it: forced-choice
pairwise comparison against the BEFORE photo, sides randomised, judge blind to
the scores.

For each sampled group the *reliable* candidates (a vetoed candidate has no
OneAlign score at all and is not what the ranking is being tested on) are sorted
by OneAlign and three pairs are put to the judge:

    winner vs rank2      - can it separate near-neighbours?
    winner vs mid        - can it separate a clear step?
    winner vs last       - can it separate the extremes at all?

If the winner does not beat the last-ranked candidate well above chance, the
ranking carries no signal at this resolution and the winner is effectively a
random draw.  ``gap: negligible`` on all three pairs marks a group whose eight
candidates are visually interchangeable, which is a different failure - the
ranking may be fine and the candidate set simply degenerate.

Usage:
  python -m dataset_build.tools.reeval_iaa_pairs --n-local 17 --n-global 8
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Mapping

from dataset_build.tools.archive_reader import read_bytes
from dataset_build.tools.reeval_relay import (
    JsonlStore, Task, data_url_bytes, load_lanes, run_tasks,
)

BUILD_ROOT = Path("/mnt/nfs/bc/data/builds/eval100-annotqa-20260727")
REVIEW_ROOT = Path("/var/cache/veradata/annot_review/eval100-annotqa-20260727")
DEFAULT_OUT = REVIEW_ROOT / "wp5_reeval"

RUBRIC = """You are judging retouching quality.

You receive a BEFORE photo and two retouched versions of it, LEFT and RIGHT.
Exactly one question: which version is the better retouched result for this photo?

Judge on: tonal quality (exposure, contrast, retained highlight and shadow
detail), colour quality (believable white balance, pleasing but not garish
saturation, no colour casts on skin or neutrals), naturalness and absence of
artefacts (banding, clipping, halos, muddy blacks, blown highlights), and overall
aesthetic appeal as a finished photograph.

Do not prefer a version merely because it changed more, or because it is more
saturated or more contrasty. Do not assume either side is the intended answer.
If the two are so close that you would be guessing, still name the one you
marginally prefer but set gap to "negligible".

Return strict JSON only: your choice, whether the quality gap is "negligible" or
"clear", and one sentence explaining what decided it."""

SCHEMA = {
    "type": "object",
    "properties": {
        "choice": {"type": "string", "enum": ["left", "right"]},
        "gap": {"type": "string", "enum": ["negligible", "clear"]},
        "reason": {"type": "string", "minLength": 15},
    },
    "required": ["choice", "gap", "reason"],
    "additionalProperties": False,
}


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def reliable_ranked(group: Mapping[str, Any]) -> list[dict]:
    candidates = [c for c in group.get("candidates", [])
                  if c.get("qa", {}).get("reliable") and c["qa"].get("onealign") is not None]
    return sorted(candidates, key=lambda c: -c["qa"]["onealign"])


def pick_groups(groups: list[dict], n_local: int, n_global: int, seed: int) -> list[dict]:
    eligible = [g for g in groups
                if g.get("winner_ids") and len(reliable_ranked(g)) >= 4]
    rng = random.Random(seed)
    chosen = []
    for mode, quota in (("local", n_local), ("global", n_global)):
        pool = sorted([g for g in eligible if g["render_mode"] == mode],
                      key=lambda g: g["group_id"])
        rng.shuffle(pool)
        chosen.extend(pool[:quota])
    return sorted(chosen, key=lambda g: g["group_id"])


def build_request(before_path: str, left_path: str, right_path: str) -> tuple[list[dict], dict]:
    content = [
        {"type": "input_text",
         "text": RUBRIC + "\n\nImages, in order: BEFORE, LEFT, RIGHT."},
        {"type": "input_image", "image_url": data_url_bytes(read_bytes(before_path))},
        {"type": "input_image", "image_url": data_url_bytes(read_bytes(left_path))},
        {"type": "input_image", "image_url": data_url_bytes(read_bytes(right_path))},
    ]
    return content, SCHEMA


def validate(result: dict) -> dict:
    if result.get("choice") not in {"left", "right"}:
        raise ValueError("bad choice")
    if result.get("gap") not in {"negligible", "clear"}:
        raise ValueError("bad gap")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--effort", default="xhigh")
    parser.add_argument("--per-lane", type=int, default=2)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--n-local", type=int, default=17)
    parser.add_argument("--n-global", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    groups = rows(BUILD_ROOT / "groups.jsonl")
    chosen = pick_groups(groups, args.n_local, args.n_global, args.seed)
    plan = []
    tasks = []
    for group in chosen:
        ranked = reliable_ranked(group)
        winner = ranked[0]
        opponents = {
            "rank2": ranked[1],
            "mid": ranked[len(ranked) // 2],
            "last": ranked[-1],
        }
        assert winner["candidate_id"] == group["winner_ids"][0], group["group_id"]
        for slot, opponent in opponents.items():
            if opponent["candidate_id"] == winner["candidate_id"]:
                continue
            rng = random.Random(f"{args.seed}:{group['group_id']}:{slot}")
            winner_left = rng.random() < 0.5
            left, right = (winner, opponent) if winner_left else (opponent, winner)
            record = {
                "group_id": group["group_id"],
                "render_mode": group["render_mode"],
                "opponent_slot": slot,
                "winner_side": "left" if winner_left else "right",
                "winner_id": winner["candidate_id"],
                "opponent_id": opponent["candidate_id"],
                "winner_onealign": winner["qa"]["onealign"],
                "opponent_onealign": opponent["qa"]["onealign"],
                "opponent_rank": ranked.index(opponent) + 1,
                "n_reliable": len(ranked),
            }
            plan.append(record)
            tasks.append(Task(
                key=f"c::{group['group_id']}::{slot}",
                build=(lambda s=group["source_path"], l=left["after_path"],
                       r=right["after_path"]: build_request(s, l, r)),
                record=record,
            ))
    (args.out / "c_plan.json").write_text(json.dumps(plan, indent=2))
    if args.limit:
        tasks = tasks[:args.limit]

    store = JsonlStore(args.out / "c_iaa.jsonl")
    lanes = load_lanes()
    stats = run_tasks(
        tasks, lanes, store, model=args.model, effort=args.effort,
        per_lane=args.per_lane, attempts=args.attempts, schema_name="iaa_pair",
        max_output_tokens=args.max_tokens, validate=validate,
        progress=args.out / "progress_c.json",
    )
    (args.out / "stats_c.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({"stage": "c", "groups": len(chosen), "pairs": len(plan), **stats},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
