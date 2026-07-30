"""Export eval100 winner samples into per-group review folders for annotation QA."""
import argparse
import json
import sys
from pathlib import Path

from dataset_build.tools.archive_reader import read_bytes

BUILD_ROOT = Path("/mnt/ramstage/eval100-annotqa-20260727")
MIRROR_ROOT = Path("/mnt/nfs/bc/data/builds/eval100-annotqa-20260727")
OUT = Path("/var/cache/veradata/annot_review/eval100-annotqa-20260727")


def rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def materialize(path: str, dest: Path) -> bool:
    try:
        dest.write_bytes(read_bytes(path))
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"MISS {path}: {type(exc).__name__}", file=sys.stderr)
        return False


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--sft-ids-file", type=Path)
    args = parser.parse_args(argv)

    root = BUILD_ROOT if (BUILD_ROOT / "sft.jsonl").is_file() else MIRROR_ROOT
    sft = rows(root / "sft.jsonl")
    groups = {g["group_id"]: g for g in rows(root / "groups.jsonl")}
    if args.sft_ids_file is not None:
        requested = json.loads(args.sft_ids_file.read_text())
        if not isinstance(requested, list) or len(requested) != len(set(requested)) \
                or not all(isinstance(value, str) and value for value in requested):
            raise ValueError("sft ID file must contain a unique JSON string list")
        requested_ids = set(requested)
        sft = [row for row in sft if row.get("sft_id") in requested_ids]
        found_ids = {row["sft_id"] for row in sft}
        if found_ids != requested_ids:
            raise ValueError(f"SFT ID file contains {len(requested_ids - found_ids)} missing IDs")
    else:
        requested_ids = None
    args.out.mkdir(parents=True, exist_ok=True)
    by_group: dict[str, list[dict]] = {}
    for row in sft:
        by_group.setdefault(row["group_id"], []).append(row)
    manifest = []
    export_groups = groups if requested_ids is None else {
        gid: groups[gid] for gid in by_group
    }
    for idx, (gid, group) in enumerate(sorted(export_groups.items()), start=1):
        grp_rows = by_group.get(gid, [])
        gdir = args.out / f"g{idx:03d}_{gid[:12]}"
        gdir.mkdir(exist_ok=True)
        entry = {"dir": str(gdir), "group_id": gid, "render_mode": group.get("render_mode"),
                 "scene": group.get("scene"),
                 "review_status": "pending" if grp_rows else "no_sft",
                 "samples": []}
        candidates = {
            candidate["candidate_id"]: candidate
            for candidate in group.get("candidates", [])
        }
        before_path = grp_rows[0]["I_in"] if grp_rows else group.get("source_path")
        if before_path:
            materialize(before_path, gdir / "before.jpg")
        for row in sorted(grp_rows, key=lambda r: r.get("winner_rank", 9)):
            rank = row.get("winner_rank")
            candidate = candidates.get(row.get("candidate_id"), {})
            materialize(row["I_tar"], gdir / f"after_rank{rank}.jpg")
            local = row.get("local") or {}
            if local.get("C_GT"):
                materialize(local["C_GT"], gdir / f"cgt_rank{rank}.png")
            entry["samples"].append({
                "sft_id": row["sft_id"], "rank": rank, "task_type": row["task_type"],
                "instruction": row["instruction"], "instruction_short": row["instruction_short"],
                "reasoning": row["reasoning"], "annot_src": row.get("annot_src"),
                "qa": row.get("qa"), "recipe_kind": candidate.get("kind"),
                "style_name": candidate.get("style_name"),
                "subject": local.get("subject"), "region": local.get("region"),
                "slot_mode": local.get("slot_mode"),
            })
        (gdir / "annot.json").write_text(json.dumps(entry, ensure_ascii=False, indent=2))
        manifest.append(entry)
    groups_with_sft = len(by_group)
    summary = {
        "groups": len(manifest),
        "groups_with_sft": groups_with_sft,
        "groups_without_sft": len(manifest) - groups_with_sft,
        "sft_rows": len(sft),
    }
    (args.out / "manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({**summary, "out": str(args.out)}))


if __name__ == "__main__":
    main()
