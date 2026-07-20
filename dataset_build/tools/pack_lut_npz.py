#!/usr/bin/env python3
"""把全部 native LUT 预解析进单个 npz（2026-07-18 提吞吐）。

动机：load_cube 是纯 Python 解析（64ms~1.8s/次，持 GIL），lut-only 采样下它是
GPU 饥饿（util 7%）的根因。预解析成 npz 后渲染时按 preset_id O(1) 取内存网格，
彻底消除解析税。存储的 grid 已是 _cube_cached 的成品形态（float32 [n,n,n,3]、
0..1 归一化、[b][g][r] 轴序），渲染侧可直接返回。

产物（默认 bank 目录）：
  luts.npz        —— key = preset_id, value = grid float32 [n,n,n,3]
  luts_meta.json  —— {pid: {path, dmin:[3], dmax:[3]}}
用法: python pack_lut_npz.py [--bank DIR] [--out-npz P] [--out-meta P]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/bc/VeraRetouch")
sys.path.insert(0, "/home/bc/VeraRetouch/dataset_build/src")


def main() -> None:
    ap = argparse.ArgumentParser()
    B = "/home/bc/data/datasets/vera_directionA_1M/preset_bank_full"
    ap.add_argument("--bank", default=B)
    ap.add_argument("--out-npz", default=None)
    ap.add_argument("--out-meta", default=None)
    a = ap.parse_args()
    out_npz = a.out_npz or os.path.join(a.bank, "luts.npz")
    out_meta = a.out_meta or os.path.join(a.bank, "luts_meta.json")

    from dataset_build.lut_io import load_lut

    grids, meta = {}, {}
    n_ok = n_fail = 0
    for line in open(os.path.join(a.bank, "features.jsonl")):
        r = json.loads(line)
        if (r.get("kind") or "") != "lut":
            continue
        pid, path = r["preset_id"], r["path"]
        try:
            grid, dmin, dmax = load_lut(path)
            grids[pid] = np.ascontiguousarray(grid, dtype="float32")
            meta[pid] = {"path": os.path.realpath(path),
                         "dmin": [float(x) for x in dmin],
                         "dmax": [float(x) for x in dmax]}
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"[pack] FAIL {pid}: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
    tmp = out_npz + ".tmp.npz"
    np.savez(tmp, **grids)          # 不压缩：加载快，磁盘换速度（~2-3GB）
    os.replace(tmp, out_npz)
    json.dump(meta, open(out_meta, "w"))
    sz = os.path.getsize(out_npz) / 1e9
    print(f"[pack] ok={n_ok} fail={n_fail} -> {out_npz} ({sz:.2f}GB), meta {len(meta)} 条")


if __name__ == "__main__":
    main()
