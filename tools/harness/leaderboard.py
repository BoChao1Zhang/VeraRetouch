"""INF-1 总榜聚合 — experiments/*/metrics.json -> markdown 榜单.

红线 (PLAN §3 / CLAUDE.md): 每行强制含 Δ_const / Δ_shuffle 列 — 缺失标 "N/A ⚠"
并向 stderr 发警告. 排序主键 = psnr_in (掩膜内 PSNR, 主指标), 缺失沉底.

metrics.json 最小 schema (README 有全文):
{
  "exp_id": "...",                       # 必填
  "n_images": 100,
  "metrics": {
    "psnr_in": ..., "psnr_band": ..., "psnr_out": ..., "psnr_full": ...,
    "ssim": ..., "delta_e00": ..., "lpips": ...,
    "delta_const": ..., "delta_shuffle": ...
  },
  "ci": {"delta_psnr_mean": ..., "ci95": [lo, hi], "sign_test": {"p_value": ...}}
}

CLI: python leaderboard.py "experiments/*/metrics.json" [-o leaderboard.md]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

COLUMNS = [
    ("exp_id", "exp"),
    ("n_images", "n"),
    ("psnr_in", "PSNR_in"),
    ("psnr_band", "PSNR_band"),
    ("psnr_out", "PSNR_out"),
    ("psnr_full", "PSNR_full"),
    ("ssim", "SSIM"),
    ("delta_e00", "ΔE00"),
    ("lpips", "LPIPS"),
    ("delta_const", "Δ_const"),
    ("delta_shuffle", "Δ_shuffle"),
    ("ci", "ΔPSNR [95% CI]"),
]
MANDATORY = ("delta_const", "delta_shuffle")  # 红线列


def _fmt(v, digits: int = 3) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return "N/A" if math.isnan(v) else f"{v:.{digits}f}"
    return str(v)


def _fmt_ci(ci: dict | None) -> str:
    if not ci or ci.get("delta_psnr_mean") is None:
        return "—"
    lo, hi = ci.get("ci95", [None, None])
    if lo is None:
        return _fmt(ci["delta_psnr_mean"])
    p = (ci.get("sign_test") or {}).get("p_value")
    tail = f", p={p:.3g}" if p is not None else ""
    return f"{ci['delta_psnr_mean']:+.3f} [{lo:+.3f}, {hi:+.3f}]{tail}"


def load_row(path: str | Path) -> tuple:
    """读单个 metrics.json -> (row dict, warnings list)."""
    path = Path(path)
    warnings = []
    with open(path) as f:
        data = json.load(f)
    m = data.get("metrics", {})
    row = {
        "exp_id": data.get("exp_id", path.parent.name),
        "n_images": data.get("n_images", data.get("n")),
        "ci": data.get("ci"),
        "_path": str(path),
    }
    for key, _ in COLUMNS:
        if key in ("exp_id", "n_images", "ci"):
            continue
        row[key] = m.get(key)
    for key in MANDATORY:
        if row.get(key) is None or (isinstance(row[key], float) and math.isnan(row[key])):
            warnings.append(
                f"[leaderboard] WARNING: {row['exp_id']} ({path}) 缺失红线列 {key} — 标 N/A. "
                f"每个消融行必带 Δ_const/Δ_shuffle (PLAN §3)."
            )
            row[key] = None
    return row, warnings


def build_leaderboard(paths) -> tuple:
    """paths: metrics.json 路径列表 -> (markdown str, warnings list). 按 psnr_in 降序."""
    rows, warnings = [], []
    for p in paths:
        try:
            row, w = load_row(p)
        except (json.JSONDecodeError, OSError) as e:
            warnings.append(f"[leaderboard] WARNING: 跳过 {p}: {e}")
            continue
        rows.append(row)
        warnings.extend(w)
    rows.sort(
        key=lambda r: r.get("psnr_in") if isinstance(r.get("psnr_in"), (int, float)) else -1e9,
        reverse=True,
    )
    header = "| " + " | ".join(h for _, h in COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in COLUMNS) + "|"
    lines = ["# Leaderboard", "", header, sep]
    for r in rows:
        cells = []
        for key, _ in COLUMNS:
            if key == "ci":
                cells.append(_fmt_ci(r.get("ci")))
            elif key in MANDATORY and r.get(key) is None:
                cells.append("N/A ⚠")
            else:
                cells.append(_fmt(r.get(key)))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", f"_{len(rows)} experiments; 排序: PSNR_in 降序; 生成: tools/harness/leaderboard.py_"]
    return "\n".join(lines) + "\n", warnings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="聚合 metrics.json -> markdown 总榜")
    ap.add_argument("pattern", help='glob, 如 "experiments/*/metrics.json"')
    ap.add_argument("-o", "--output", default=None, help="输出 md 路径 (缺省打印 stdout)")
    args = ap.parse_args(argv)
    paths = sorted(glob.glob(args.pattern))
    if not paths:
        print(f"[leaderboard] no files match {args.pattern}", file=sys.stderr)
        return 1
    md, warnings = build_leaderboard(paths)
    for w in warnings:
        print(w, file=sys.stderr)
    if args.output:
        Path(args.output).write_text(md)
        print(f"[leaderboard] wrote {args.output} ({len(paths)} files)", file=sys.stderr)
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
