#!/usr/bin/env python3
"""核对：主线包配置物化后与 EPR-051 现行配置逐字节相同（EPR-052 自检）。"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from veraretouch_sprf import configs as CF, _paths as _P

SRC = {
    "decoder/clut_full.toml": _P.SPRF_LEGACY / "configs/arm_clut_full.toml",
    "decoder/bkfull_adagn_ff_affhead.toml": _P.SPRF_LEGACY / "configs/bkfull_adagn_ff_affhead.toml",
    "decoder/smoke_clut_full.toml": _P.SPRF_LEGACY / "configs/smoke_clut_full.toml",
    "vlm/sft_full.toml": _P.VLMSFT_LEGACY / "configs/q3vl_sft_s1f_full.toml",
    "vlm/adapt_s2fb.toml": _P.VLMSFT_LEGACY / "configs/q3vl_adapt_s2fb.toml",
}
bad = 0
for rel, src in SRC.items():
    same = CF.render((CF.CONFIG_DIR / rel).read_text()) == src.read_text()
    print(("OK  " if same else "DIFF"), rel, "<-", src.name)
    bad += (not same)
sys.exit(1 if bad else 0)
