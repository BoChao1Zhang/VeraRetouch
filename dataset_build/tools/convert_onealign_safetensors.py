#!/usr/bin/env python3
"""把分片 `pytorch_model*.bin` 转成 safetensors（state-dict 级，不依赖模型类）。

动机：OneAlign 15.3 GiB 权重每次冷启动要 `torch.load` 解 zip + 逐张拷贝（实测
36.5s）；safetensors 走 mmap 零拷贝，HF `from_pretrained` 在本地目录里**优先**
挑 `model.safetensors[.index.json]`（`transformers.modeling_utils.
_get_resolved_checkpoint_files`：`use_safetensors is not False` 分支排在
`pytorch_model.bin` 之前），所以只要把文件放进模型目录就自动生效，加载侧零改动。

转换严格按原 shard 划分 1:1 落盘（`pytorch_model-00001-of-00002.bin` →
`model-00001-of-00002.safetensors`），原 .bin **不动不删**（可随时回退：删掉
safetensors 文件即可）。三个必须处理的坑：

1. **共享/别名张量**：safetensors 拒绝两个 key 指向同一 storage（它只存字节，
   无法表达别名）。这里按 storage 指纹（device+data_ptr+nbytes）检测，后出现者
   `.clone()` 成独立副本并记录进摘要——值逐位不变，只是磁盘上多一份。判据比
   safetensors 自身略保守（它还会按字节区间放行同 storage 但不重叠的切片），
   代价只是极少数张量多拷一份，换来不依赖其内部实现。
2. **非连续张量**：safetensors 只接受 contiguous，先 `.contiguous()`。
3. **dtype**：全程不做任何 cast，逐位保真。

转换完立刻自校验：把写出的 safetensors 逐 key 读回，与内存里**变换前**的原张量
比 dtype/shape/原始字节（uint8 视图，NaN 也能判等，比 `torch.equal` 更严），并断
言 key 集合与原 index 完全一致。任何一项不符直接非零退出，不留半成品 index。

用法:
    python convert_onealign_safetensors.py --model-dir /path/to/OneAlign [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BIN_INDEX_NAME = "pytorch_model.bin.index.json"
SAFE_INDEX_NAME = "model.safetensors.index.json"


class ConvertError(RuntimeError):
    """转换或校验失败（调用方按非零退出处理）。"""


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def safetensors_shard_name(bin_name: str, position: int, total: int) -> str:
    """原 shard 文件名 → safetensors 文件名（尽量保留原分片编号）。"""
    match = re.fullmatch(r"pytorch_model-(\d+)-of-(\d+)\.bin", bin_name)
    if match:
        return f"model-{match.group(1)}-of-{match.group(2)}.safetensors"
    return f"model-{position:05d}-of-{total:05d}.safetensors"


def read_index(model_dir: str) -> dict[str, Any]:
    path = os.path.join(model_dir, BIN_INDEX_NAME)
    if not os.path.isfile(path):
        raise ConvertError(f"缺少分片索引 {path}（本工具只处理分片 .bin 权重）")
    with open(path, encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ConvertError(f"{path} 的 weight_map 为空或格式不对")
    return index


def _shard_order(weight_map: dict[str, str]) -> list[str]:
    """shard 文件名去重后按名字排序（同名分片编号即字典序）。"""
    return sorted(set(weight_map.values()))


def _storage_key(tensor: torch.Tensor) -> tuple[Any, ...] | None:
    """storage 指纹；空 storage 返回 None（与 safetensors 自身的别名判定同口径）。"""
    storage = tensor.untyped_storage()
    nbytes = storage.nbytes()
    pointer = storage.data_ptr()
    if nbytes == 0 or pointer == 0:
        return None
    return (str(tensor.device), pointer, nbytes)


def _raw_bytes_view(tensor: torch.Tensor) -> torch.Tensor:
    """逐位比较用的 uint8 视图（避开 torch.equal 对 NaN 判不等的坑）。"""
    return tensor.reshape(-1).contiguous().view(torch.uint8)


def bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
        return False
    if left.numel() == 0:
        return True
    return bool(torch.equal(_raw_bytes_view(left), _raw_bytes_view(right)))


def _load_shard(path: str) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (RuntimeError, TypeError, ValueError):
        # 老序列化格式（非 zipfile）不支持 mmap，退回整读。
        state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ConvertError(f"{path} 不是 state dict（拿到 {type(state).__name__}）")
    bad = [key for key, value in state.items() if not isinstance(value, torch.Tensor)]
    if bad:
        raise ConvertError(f"{path} 含非张量条目，safetensors 无法承载: {bad[:5]}")
    return state


def _prepare_shard(
    state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
    """就地整理一个 shard：非连续转连续、别名克隆。返回 (可写字典, 别名, 非连续)。"""
    out: dict[str, torch.Tensor] = {}
    aliased: list[str] = []
    non_contiguous: list[str] = []
    seen: set[tuple[Any, ...]] = set()
    for key, tensor in state.items():
        if not tensor.is_contiguous():
            non_contiguous.append(key)
            tensor = tensor.contiguous()
        fingerprint = _storage_key(tensor)
        if fingerprint is not None and fingerprint in seen:
            aliased.append(key)
            tensor = tensor.clone()
            fingerprint = _storage_key(tensor)
        if fingerprint is not None:
            seen.add(fingerprint)
        out[key] = tensor
    return out, aliased, non_contiguous


def _verify_shard(path: str, original: dict[str, torch.Tensor]) -> None:
    """把落盘文件逐 key 读回，与变换前的原张量比逐位一致。"""
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if keys != set(original):
            missing = sorted(set(original) - keys)
            extra = sorted(keys - set(original))
            raise ConvertError(
                f"{path} key 集合不一致: missing={missing[:5]} extra={extra[:5]}"
            )
        for key in original:
            if not bitwise_equal(original[key], handle.get_tensor(key)):
                raise ConvertError(f"{path} 的 {key} 读回与原 .bin 不逐位一致")


def convert(model_dir: str, dry_run: bool = False) -> dict[str, Any]:
    """转换整个模型目录，返回 JSON 摘要（dry_run 只读索引、不落盘）。"""
    model_dir = os.path.abspath(model_dir)
    index = read_index(model_dir)
    weight_map: dict[str, str] = index["weight_map"]
    shards = _shard_order(weight_map)
    keys_by_shard: dict[str, list[str]] = {name: [] for name in shards}
    for key, shard in weight_map.items():
        keys_by_shard[shard].append(key)

    summary: dict[str, Any] = {
        "model_dir": model_dir,
        "dry_run": dry_run,
        "keys": len(weight_map),
        "shards": [],
        "index": SAFE_INDEX_NAME,
        "total_size": 0,
        "index_total_size_bin": index.get("metadata", {}).get("total_size"),
        "aliased_cloned": [],
        "made_contiguous": [],
        "verified": False,
    }

    safe_weight_map: dict[str, str] = {}
    used_names: set[str] = set()
    total_size = 0
    for position, bin_name in enumerate(shards, start=1):
        safe_name = safetensors_shard_name(bin_name, position, len(shards))
        if safe_name in used_names:
            raise ConvertError(f"两个分片映射到同一输出名 {safe_name}")
        used_names.add(safe_name)
        bin_path = os.path.join(model_dir, bin_name)
        safe_path = os.path.join(model_dir, safe_name)
        entry: dict[str, Any] = {
            "source": bin_name,
            "output": safe_name,
            "tensors": len(keys_by_shard[bin_name]),
        }
        if dry_run:
            if not os.path.isfile(bin_path):
                raise ConvertError(f"索引提到的分片不存在: {bin_path}")
            summary["shards"].append(entry)
            for key in keys_by_shard[bin_name]:
                safe_weight_map[key] = safe_name
            continue

        _log(f"[{position}/{len(shards)}] load {bin_name}")
        original = _load_shard(bin_path)
        if set(original) != set(keys_by_shard[bin_name]):
            missing = sorted(set(keys_by_shard[bin_name]) - set(original))
            extra = sorted(set(original) - set(keys_by_shard[bin_name]))
            raise ConvertError(
                f"{bin_name} 与索引不符: missing={missing[:5]} extra={extra[:5]}"
            )
        prepared, aliased, non_contiguous = _prepare_shard(original)
        if aliased:
            _log(f"    别名张量克隆 {len(aliased)}: {aliased}")
        if non_contiguous:
            _log(f"    非连续张量转连续 {len(non_contiguous)}: {non_contiguous}")
        shard_bytes = sum(t.numel() * t.element_size() for t in prepared.values())

        tmp_path = safe_path + ".tmp"
        _log(f"    write {safe_name} ({shard_bytes / 2**30:.2f} GiB)")
        save_file(prepared, tmp_path, metadata={"format": "pt"})
        os.replace(tmp_path, safe_path)
        _log("    verify …")
        _verify_shard(safe_path, original)

        entry.update(
            {
                "bytes": shard_bytes,
                "aliased_cloned": aliased,
                "made_contiguous": non_contiguous,
            }
        )
        summary["shards"].append(entry)
        summary["aliased_cloned"].extend(aliased)
        summary["made_contiguous"].extend(non_contiguous)
        total_size += shard_bytes
        for key in keys_by_shard[bin_name]:
            safe_weight_map[key] = safe_name
        del prepared, original

    if set(safe_weight_map) != set(weight_map):
        raise ConvertError("写出的 weight_map 与原索引 key 集合不一致")
    summary["total_size"] = None if dry_run else total_size

    if not dry_run:
        index_path = os.path.join(model_dir, SAFE_INDEX_NAME)
        payload = {
            "metadata": {"total_size": total_size},
            "weight_map": safe_weight_map,
        }
        tmp_index = index_path + ".tmp"
        with open(tmp_index, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_index, index_path)
        summary["verified"] = True
        _log(f"done: {index_path}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-dir",
        required=True,
        help="模型目录（含 pytorch_model.bin.index.json + 分片 .bin）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只读索引、报告计划落盘的文件，不加载权重也不写盘",
    )
    args = parser.parse_args(argv)
    try:
        summary = convert(args.model_dir, dry_run=args.dry_run)
    except ConvertError as exc:
        print(f"convert failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
