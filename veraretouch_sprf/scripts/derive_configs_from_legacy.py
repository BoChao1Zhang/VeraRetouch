#!/usr/bin/env python3
"""EPR-052：从 EPR-051 现行配置机械派生主线包配置（加 [paths] 头 + 路径前缀占位），并回物化逐字节核对。"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from veraretouch_sprf import configs as CF, _paths as _P

PATHS = [  # 长前缀在前
    ("repo", "/home/bc/VeraRetouch"),
    ("runs_sprf", "/home/bc/data/runs/epr051_sprf"),
    ("runs_vlm", "/home/bc/data/runs/epr051_vlmsft"),
    ("runs", "/home/bc/data/runs"),
    ("shards_local", "/home/bc/data/builds/epr051_stage0"),
    ("builds", "/home/bc/data/builds"),
    ("shards_nfs", "/mnt/nfs-ro/bc/data/datasets/epr051_stage0"),
    ("lut_bank", "/var/cache/veradata/preset_bank_full"),
    ("models", "/home/bc/data/models"),
]
SRC = {
    "decoder/clut_full.toml": _P.SPRF_LEGACY / "configs/arm_clut_full.toml",
    "decoder/bkfull_adagn_ff_affhead.toml": _P.SPRF_LEGACY / "configs/bkfull_adagn_ff_affhead.toml",
    "decoder/smoke_clut_full.toml": _P.SPRF_LEGACY / "configs/smoke_clut_full.toml",
    "vlm/sft_full.toml": _P.VLMSFT_LEGACY / "configs/q3vl_sft_s1f_full.toml",
    "vlm/adapt_s2fb.toml": _P.VLMSFT_LEGACY / "configs/q3vl_adapt_s2fb.toml",
}
for rel, src in SRC.items():
    t = src.read_text()
    used = []
    for k, v in PATHS:
        if v in t:
            t = t.replace(v, "${paths.%s}" % k)
            used.append((k, v))
    header = ("[paths]\n"
              "# veraretouch_sprf 主线配置（EPR-052）：由 %s 机械派生；去掉本 [paths] 节后与原件逐字节相同（scripts/check_configs.py 核对）。\n"
              "# 路径占位 ${paths.<key>} 由 veraretouch_sprf.configs.materialize() 解析；环境变量 VR_PATH_<KEY> 可覆盖。\n"
              % src.relative_to(_P.REPO))
    for k, v in used:
        header += '%s = "%s"\n' % (k, v)
    header += CF.END_MARK + "\n"
    dst = CF.CONFIG_DIR / rel
    dst.write_text(header + t)
    assert CF.render(dst.read_text()) == src.read_text(), rel
    print("OK", rel, "paths:", [k for k, _ in used], "占位数:", dst.read_text().count("${paths."))
