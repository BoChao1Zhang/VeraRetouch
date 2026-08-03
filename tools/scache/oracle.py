"""oracle s 缓存构建器（INF-5 / 任务卡 T5）。

从 l 系 build 的 shards 里逐候选读出 C_GT 掩膜（.cgt.png，单通道软边、逐候选区域掩膜，
非配色真值——见 DATA_ASSIGNMENT §附录 B），面积加权下采样到 32x32（可配），写入
`<root>/oracle/<img_id>__<instr_hash>.npy` + meta；meta 里存原掩膜的 (shard, offset)
引用与输入图成员引用，便于零拷贝回读全分辨率真值。

- img_id = candidate_id（掩膜逐候选落盘；见 NOTES.md 待决策 #1）。
- instr_hash 来自 journal 归档 sft.jsonl 的 instruction（md5 前 12 位）；
  非 SFT 行的候选记 'noinstr'（--sft-only 可跳过）。
- 面积加权下采样 = PIL Image.BOX 于 float32 'F' 图（分数覆盖的真区域平均）。

CLI:
    python3 tools/scache/oracle.py \
        --build /mnt/nfs/bc/data/datasets/sft/prod-l1-local17k-20260731 \
        --journal /var/cache/veradata/annot_review/journal-archive/prod-l1-local17k-20260731 \
        --root <s_cache 根> [--size 32] [--max-groups 100] [--sft-only]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api import NO_INSTR, SCache, instr_hash  # noqa: E402

MASK_SUFFIX = ".cgt.png"
VRMETA_SUFFIX = ".vrmeta.json"
INPUT_SUFFIXES = (".in.jpg", ".in.png")
ORACLE_ARM = "oracle"
ORACLE_VERSION = "oracle-v1"


# ---------------- shard 读取 ----------------

def iter_build_samples(build_dir: Path) -> Iterator[Tuple[str, Dict[str, dict]]]:
    """迭代 build 下全部 sample：yield (sample_id, {suffix: index_row})。

    index_row 追加 'tar_path'（绝对路径）与 'batch' 字段。
    batch 目录与 shard 索引均按名字排序 → 迭代序确定。
    """
    for batch_dir in sorted(build_dir.glob("batch-*")):
        idx_dir = batch_dir / "indexes"
        if not idx_dir.is_dir():
            continue
        for idx_path in sorted(idx_dir.glob("shard-*.idx.jsonl")):
            by_sample: "defaultdict[str, Dict[str, dict]]" = defaultdict(dict)
            order: list[str] = []
            with open(idx_path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    row["tar_path"] = str(batch_dir / "shards" / f"{row['shard']}.tar")
                    row["batch"] = batch_dir.name
                    sid = row["sample_id"]
                    if sid not in by_sample:
                        order.append(sid)
                    by_sample[sid][row["suffix"]] = row
            for sid in order:
                yield sid, by_sample[sid]


def read_member(row: dict) -> bytes:
    """按 (shard, offset) 引用直读 tar 成员（无压缩 ustar，seek+read）。"""
    with open(row["tar_path"], "rb") as f:
        f.seek(row["offset_data"])
        data = f.read(row["length"])
    if len(data) != row["length"]:
        raise IOError(f"短读 {row['member']}: {len(data)} != {row['length']}")
    return data


def load_mask(row: dict) -> np.ndarray:
    """C_GT 掩膜 -> float32 (H, W)，值域 [0,1]。"""
    img = Image.open(io.BytesIO(read_member(row)))
    if img.mode != "L":
        img = img.convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def load_guide(row: dict, size_hw: Tuple[int, int]) -> np.ndarray:
    """输入图 -> float32 (H, W, 3) [0,1]，resize 到掩膜尺寸（guide 用）。"""
    img = Image.open(io.BytesIO(read_member(row))).convert("RGB")
    h, w = size_hw
    if img.size != (w, h):
        img = img.resize((w, h), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def area_downsample(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """面积加权下采样：PIL BOX 滤波（float32 'F' 模式，支持非整除分数覆盖）。"""
    h, w = size
    img = Image.fromarray(np.ascontiguousarray(mask, dtype=np.float32), mode="F")
    small = img.resize((w, h), Image.Resampling.BOX)
    return np.asarray(small, dtype=np.float32)


def _member_ref(row: dict) -> dict:
    return {
        "batch": row["batch"],
        "shard": row["shard"],
        "tar_path": row["tar_path"],
        "member": row["member"],
        "offset_data": row["offset_data"],
        "length": row["length"],
        "sha256": row.get("sha256"),
        "suffix": row["suffix"],
    }


def load_sft_instructions(journal_dir: Path) -> Dict[str, str]:
    """journal 归档 sft.jsonl -> {candidate_id: instruction}。"""
    out: Dict[str, str] = {}
    p = journal_dir / "sft.jsonl"
    if not p.exists():
        return out
    with open(p, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            cid, instr = d.get("candidate_id"), d.get("instruction")
            if cid and instr:
                out[cid] = instr
    return out


# ---------------- oracle 构建 ----------------

def build_oracle(
    build_dir: Path,
    root: Path,
    journal_dir: Optional[Path] = None,
    size: int = 32,
    max_groups: Optional[int] = None,
    sft_only: bool = False,
    overwrite: bool = True,
    progress_every: int = 200,
) -> dict:
    """构建 s_cache/oracle/。返回统计 dict。

    max_groups: 只处理 shard 迭代序下前 N 个不同 group_id（None = 全量）。
    """
    instructions = load_sft_instructions(journal_dir) if journal_dir else {}
    cache = SCache(root, ORACLE_ARM, resolution=size, arm_version=ORACLE_VERSION)

    stats = {"written": 0, "skipped_no_mask": 0, "skipped_no_sft": 0, "groups": 0}
    seen_groups: set[str] = set()

    for sid, members in iter_build_samples(build_dir):
        mask_row = members.get(MASK_SUFFIX)
        vr_row = members.get(VRMETA_SUFFIX)
        if mask_row is None or vr_row is None:
            stats["skipped_no_mask"] += 1
            continue
        vrmeta = json.loads(read_member(vr_row))
        group_id = vrmeta.get("group_id", "unknown")

        if group_id not in seen_groups:
            if max_groups is not None and len(seen_groups) >= max_groups:
                break
            seen_groups.add(group_id)

        candidate_id = vrmeta.get("candidate_id") or sid
        instr = instructions.get(candidate_id)
        if sft_only and instr is None:
            stats["skipped_no_sft"] += 1
            continue
        ihash = instr_hash(instr) if instr is not None else NO_INSTR

        mask = load_mask(mask_row)
        s = area_downsample(mask, (size, size))

        input_row = next((members[x] for x in INPUT_SUFFIXES if x in members), None)
        extra = {
            "origin": {
                "build_id": vrmeta.get("build_id"),
                "sample_id": sid,
                "candidate_id": candidate_id,
                "group_id": group_id,
                "source_id": vrmeta.get("source_id"),
                "mask_id": vrmeta.get("mask_id"),
                "mask_hw": list(mask.shape),
                "mask_member": _member_ref(mask_row),
                "input_member": _member_ref(input_row) if input_row else None,
            },
            "downsample": {"method": "pil_box_area", "from_hw": list(mask.shape)},
        }
        cache.write(
            candidate_id,
            ihash,
            s,
            layer=None,
            norm={"kind": "linear", "domain": [0.0, 1.0], "source_scale": "uint8/255"},
            extra_meta=extra,
            overwrite=overwrite,
        )
        stats["written"] += 1
        if progress_every and stats["written"] % progress_every == 0:
            print(f"[oracle] written={stats['written']} groups={len(seen_groups)}", flush=True)

    stats["groups"] = len(seen_groups)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="构建 oracle s 缓存目录（C_GT 掩膜 -> 32x32）")
    ap.add_argument("--build", required=True, type=Path, help="落盘 build 目录（含 batch-*/shards）")
    ap.add_argument("--journal", type=Path, default=None, help="journal 归档目录（取 sft.jsonl 指令）")
    ap.add_argument("--root", required=True, type=Path, help="s_cache 根目录")
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--max-groups", type=int, default=None)
    ap.add_argument("--sft-only", action="store_true", help="只为 SFT 行候选建条目")
    args = ap.parse_args()

    stats = build_oracle(
        args.build, args.root, journal_dir=args.journal, size=args.size,
        max_groups=args.max_groups, sft_only=args.sft_only,
    )
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
