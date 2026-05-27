#!/usr/bin/env python3
"""Generate reference-teacher targets for S1 rows in the VeraRetouch 250k plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import get_model, tensor2img  # noqa: E402


DEFAULT_DATASET_ROOT = Path("/home/bc/data/datasets/VeraRetouch_250k")
DEFAULT_PRETRAINED_PATH = Path(
    "/home/bc/data/models/VeraRetouch.Encoder_Renderer/encoder_renderer.pth"
)
MODEL_NAME = "RetouchRenderer_Resnet18Encoder_InputCatMixedCBAM"
IMAGE_SIZE = 512
JPEG_QUALITY = 94
REF_PAIR_SOURCES = {"ppr10k", "fivek_mmart_like"}


def stable_int(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16], 16)


def load_json_line(line: str, line_no: int, path: Path) -> dict[str, Any]:
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def iter_plan_rows(plan_path: Path):
    with plan_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                yield load_json_line(line, line_no, plan_path)


def load_reference_pairs(plan_path: Path) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in iter_plan_rows(plan_path):
        if row.get("branch") != "S0_expert_anchor":
            continue
        operation = row.get("operation") or {}
        pair_source = operation.get("pair_source") or row.get("source_dataset")
        if pair_source not in REF_PAIR_SOURCES:
            continue
        before = row.get("input_source_path")
        after = row.get("target_source_path")
        if not before or not after:
            continue
        key = (before, after)
        if key in seen:
            continue
        if not Path(before).is_file() or not Path(after).is_file():
            continue
        seen.add(key)
        pairs.append(
            {
                "ref_before_path": before,
                "ref_after_path": after,
                "ref_pair_source": str(pair_source),
                "ref_expert": str(operation.get("expert", "")),
            }
        )
    if not pairs:
        raise RuntimeError(f"No PPR/FiveK reference pairs found in {plan_path}")
    pairs.sort(key=lambda item: (item["ref_pair_source"], item["ref_before_path"], item["ref_after_path"]))
    return pairs


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_teacher(pretrained_path: Path, device: torch.device) -> torch.nn.Module:
    model = get_model(MODEL_NAME)
    checkpoint = torch.load(pretrained_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def preprocess_ref_image(path: Path) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    preprocess = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )
    tensor = preprocess(img)
    return ((tensor - 0.5) * 2).unsqueeze(0)


def preprocess_input_image(path: Path) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transforms.ToTensor()(rgb)
    return ((tensor - 0.5) * 2).unsqueeze(0)


def write_input_jpg(source_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as img:
        img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        img.save(output_path, format="JPEG", quality=JPEG_QUALITY, optimize=True)


def write_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()


def load_manifest_ids(path: Path, dataset_root: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for row in iter_plan_rows(path):
        sample_id = row.get("id")
        input_path = row.get("teacher_input_path")
        target_path = row.get("teacher_target_path")
        if (
            sample_id
            and input_path
            and target_path
            and (dataset_root / input_path).exists()
            and (dataset_root / target_path).exists()
        ):
            ids.add(str(sample_id))
    return ids


def select_s1_rows(plan_path: Path, start_offset: int, limit: int | None) -> list[dict[str, Any]]:
    if limit == 0:
        return []

    selected: list[dict[str, Any]] = []
    seen_s1 = 0
    for row in iter_plan_rows(plan_path):
        if row.get("branch") != "S1_auto_inverse_lite":
            continue
        if seen_s1 < start_offset:
            seen_s1 += 1
            continue
        seen_s1 += 1
        selected.append(row)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def build_manifest_record(
    row: dict[str, Any],
    input_rel: Path,
    target_rel: Path,
    ref_pair: dict[str, str],
) -> dict[str, Any]:
    return {
        "id": row["id"],
        "branch": row.get("branch"),
        "index": row.get("index"),
        "split": row.get("split"),
        "source_image_id": row.get("source_image_id"),
        "input_source_path": row.get("input_source_path"),
        "teacher_input_path": input_rel.as_posix(),
        "teacher_target_path": target_rel.as_posix(),
        "teacher": "vera_ref_encoder_renderer",
        "teacher_model": MODEL_NAME,
        "ref_before_path": ref_pair["ref_before_path"],
        "ref_after_path": ref_pair["ref_after_path"],
        "ref_pair_source": ref_pair["ref_pair_source"],
        "ref_expert": ref_pair["ref_expert"],
        "instruction": row.get("instruction"),
        "task_tags": row.get("task_tags", []),
    }


def generate_one(
    model: torch.nn.Module,
    device: torch.device,
    row: dict[str, Any],
    ref_pair: dict[str, str],
    input_path: Path,
    target_path: Path,
    chunk: int,
) -> None:
    if not input_path.exists():
        write_input_jpg(Path(row["input_source_path"]), input_path)

    input_tensor = preprocess_input_image(input_path).to(device)
    ref_before = preprocess_ref_image(Path(ref_pair["ref_before_path"])).to(device)
    ref_after = preprocess_ref_image(Path(ref_pair["ref_after_path"])).to(device)
    mask = torch.tensor([1.0, 1.0, 1.0], device=device).unsqueeze(0)

    with torch.no_grad():
        pred = model(input_tensor, ref_before, ref_after, mask, chunk).clamp(0, 1)

    image = Image.fromarray(tensor2img(pred), mode="RGB")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(target_path, format="JPEG", quality=JPEG_QUALITY, optimize=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Vera reference-teacher JPG pairs for S1_auto_inverse_lite records."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_DATASET_ROOT, help="Dataset root containing plan.jsonl.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum S1 records to process.")
    parser.add_argument("--device", type=str, default="auto", help="Torch device, e.g. auto, cuda, cuda:0, cpu.")
    parser.add_argument("--start-offset", type=int, default=0, help="Number of S1 records to skip before processing.")
    parser.add_argument(
        "--chunk",
        type=int,
        default=-1,
        help="Renderer chunk size; use 262144 if GPU memory is tight.",
    )
    parser.add_argument(
        "--pretrained-path",
        type=Path,
        default=DEFAULT_PRETRAINED_PATH,
        help="Path to encoder_renderer.pth.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.start_offset < 0:
        raise ValueError("--start-offset must be non-negative")

    plan_path = args.out / "plan.jsonl"
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    if not args.pretrained_path.is_file():
        raise FileNotFoundError(args.pretrained_path)

    manifest_path = args.out / "teacher" / "vera_ref" / "manifest.jsonl"

    rows = select_s1_rows(plan_path, args.start_offset, args.limit)

    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"S1 rows selected: {len(rows)}")
    if not rows:
        print(f"Manifest: {manifest_path}")
        return

    ref_pairs = load_reference_pairs(plan_path)
    manifest_ids = load_manifest_ids(manifest_path, args.out)
    print(f"Reference pairs available: {len(ref_pairs)}")

    model = load_teacher(args.pretrained_path, device)

    generated = 0
    skipped = 0
    for row in tqdm(rows, desc="vera_ref_teacher"):
        sample_id = str(row["id"])
        input_rel = Path("teacher") / "vera_ref" / "input" / f"{sample_id}.jpg"
        target_rel = Path("teacher") / "vera_ref" / "target" / f"{sample_id}.jpg"
        input_path = args.out / input_rel
        target_path = args.out / target_rel
        ref_pair = ref_pairs[stable_int(sample_id) % len(ref_pairs)]

        if sample_id in manifest_ids and input_path.exists() and target_path.exists():
            skipped += 1
            continue

        if target_path.exists() and input_path.exists():
            manifest_record = build_manifest_record(row, input_rel, target_rel, ref_pair)
            write_jsonl_record(manifest_path, manifest_record)
            manifest_ids.add(sample_id)
            skipped += 1
            continue

        generate_one(model, device, row, ref_pair, input_path, target_path, args.chunk)
        manifest_record = build_manifest_record(row, input_rel, target_rel, ref_pair)
        write_jsonl_record(manifest_path, manifest_record)
        manifest_ids.add(sample_id)
        generated += 1

    print(f"Generated: {generated}")
    print(f"Skipped: {skipped}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
