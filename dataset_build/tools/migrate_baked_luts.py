#!/usr/bin/env python3
"""存量烘焙 .cube 轴序迁移（2026-07-17 一次性工具，幂等可重跑）。

背景：apply 端曾误按 [R,G,B] 索引 load_cube 的 [b][g][r] 网格（f(B,G,R)），
bake_luts.write_cube 当时用反标准行序（r 最外/b 最快）写出补偿。apply 端修复后，
存量烘焙文件必须迁回标准 red-fastest 行序（idx = r + g*N + b*N²），否则经修复后的
加载链会红蓝互换。数据本身正确，纯行序重排。

迁移标记：文件头写入 "# axis-standard 2026-07-17"，已含标记的文件跳过。
用法: python migrate_baked_luts.py [--dir DIR] [--dry]
"""
import argparse
import glob
import os
import sys

MARK = "# axis-standard 2026-07-17"


def migrate_one(path: str, dry: bool) -> str:
    with open(path) as f:
        lines = f.readlines()
    if any(l.strip() == MARK for l in lines[:8]):
        return "skip(已迁移)"
    header, rows = [], []
    n = None
    for l in lines:
        s = l.strip()
        if not s or s.startswith("#"):
            continue
        head = s.split()[0].upper()
        if head in ("TITLE", "DOMAIN_MIN", "DOMAIN_MAX", "LUT_3D_SIZE"):
            header.append(s)
            if head == "LUT_3D_SIZE":
                n = int(s.split()[1])
            continue
        rows.append(s)
    if n is None or len(rows) != n ** 3:
        return f"FAIL(行数 {len(rows)} != N³ 或缺 LUT_3D_SIZE)"
    if dry:
        return "would-migrate"
    # 旧行序: for r: for g: for b: → 旧 idx(r,g,b) = b + g*n + r*n²
    # 新行序(标准): for b: for g: for r: → 取 rows[b + g*n + r*n²]
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(MARK + "\n")
        for h in header:
            f.write(h + "\n")
        for b in range(n):
            for g in range(n):
                for r in range(n):
                    f.write(rows[b + g * n + r * n * n] + "\n")
    os.replace(tmp, path)
    return "migrated"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/home/bc/VeraRetouch/gpu_render/fits/baked")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    stats = {}
    for p in sorted(glob.glob(os.path.join(a.dir, "*.cube"))):
        st = migrate_one(p, a.dry)
        stats[st] = stats.get(st, 0) + 1
        if st.startswith("FAIL"):
            print(f"{os.path.basename(p)}: {st}", file=sys.stderr)
    print(stats)


if __name__ == "__main__":
    main()
