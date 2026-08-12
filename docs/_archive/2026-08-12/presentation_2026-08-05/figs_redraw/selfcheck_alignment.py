"""对齐自检：`grid_to_img` 是不是 `luma_to_grid` 的严格逆映射？（红线「叠图禁直接 resize」）

方法：对每个 valid 格打单格脉冲 → `grid_to_img` 映到原图 → 再走 `luma_to_grid` 回 16×16，
看 argmax 是否回到原格。严格逆映射应当 100% 命中。同时跑 6-09 原脚本的
`Image.resize(img.size)` 作对照，量化它的系统性错位。

实测（2026-08-05，本目录三张配图用的源）：
  grid_to_img          160/160 命中（portrait 512×768 与 landscape 768×512 均 100%）
  6-09 直接 resize      48/160 命中 ⇒ **70% 的格落错位置**
"""
from __future__ import annotations

import sqlite3  # noqa: F401  isort:skip
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tools" / "readout"))

import redrawlib as RL  # noqa: E402


def roundtrip(img, valid, mode: str) -> tuple[int, int]:
    from PIL import Image

    from ro9_gl_attention import luma_to_grid

    ok = tot = 0
    for gy in range(RL.GRID):
        for gx in range(RL.GRID):
            if not valid[gy, gx]:
                continue
            tot += 1
            m = np.zeros((RL.GRID, RL.GRID), np.float32)
            m[gy, gx] = 1.0
            if mode == "strict":
                a = RL.grid_to_img(m, img)
            else:                                 # 6-09 原脚本：直接拉到未 pad 的图上
                a = np.asarray(Image.fromarray(m).resize(img.size, Image.BICUBIC),
                               dtype=np.float32)
            a = np.clip(a, 0, None)
            if a.max() <= 0:
                continue
            b, _ = luma_to_grid(a / a.max())
            ok += np.unravel_index(int(b.argmax()), b.shape) == (gy, gx)
    return ok, tot


def main() -> None:
    z = np.load(HERE / "fields_restricted_auc.npz", allow_pickle=True)
    ids = [str(x) for x in z["img_id"]]
    import picks as PK

    for iid in PK.pick_sources()[0]:
        i = ids.index(iid)
        img = RL.open_short512(str(z["img_path"][i]))
        v = z["valid16"][i]
        s_ok, s_tot = roundtrip(img, v, "strict")
        r_ok, r_tot = roundtrip(img, v, "resize")
        print(f"{iid:24s} img={img.size}  grid_to_img {s_ok}/{s_tot}  "
              f"| 6-09 直接 resize {r_ok}/{r_tot}", flush=True)
        assert s_ok == s_tot, f"grid_to_img 不是严格逆映射：{iid}"
    print("PASS：grid_to_img 对本批全部 valid 格严格可逆")


if __name__ == "__main__":
    main()
