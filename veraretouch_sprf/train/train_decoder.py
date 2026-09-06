#!/usr/bin/env python3
"""解码器训练统一入口（EPR-052 新增薄包装；不改任何训练逻辑）。

  C-LUT-FULL（FiLM 主臂）   configs/decoder/clut_full.toml               -> train_sprf_xl.main()  -> train_sprf.main()
  BK-FULL（ADAGN+FF+AFFHEAD） configs/decoder/bkfull_adagn_ff_affhead.toml -> train_sprf_bk5.main() -> train_sprf_bk_core4.main()

做的事只有两件：
  1. 把 configs/decoder/ 整组 TOML 物化（解析 ${paths.*}）到 `<run.out_dir>/configs_resolved/`，文件名恢复 EPR-051
     原名（train_sprf_bk5 的 K3 用同目录 arm_clut_full.toml / smoke_clut_full.toml 作基线；train_sprf 的 A3 用 peer_configs）；
  2. 按有无 [bk] 节选入口，把 `--config <物化文件>` 与其余参数（--smoke / --limit-samples / --stop-after ...）原样透传。

用法：
  PYTHONPATH=/home/bc/VeraRetouch python -m veraretouch_sprf.train.train_decoder --arm clut_full [--smoke ...]
  PYTHONPATH=/home/bc/VeraRetouch python -m veraretouch_sprf.train.train_decoder --arm bkfull_adagn_ff_affhead
  可选 --path key=value（覆盖 [paths]，亦可用环境变量 VR_PATH_<KEY>）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from veraretouch_sprf import configs as CF


def main() -> None:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--arm", required=True, help="configs/decoder/<arm>.toml 的文件名（不含 .toml）")
    ap.add_argument("--path", action="append", default=[], help="覆盖 [paths]：key=value，可重复")
    ap.add_argument("--resolved-dir", default="", help="物化目录；默认 <run.out_dir>/configs_resolved")
    args, rest = ap.parse_known_args()
    overrides = dict(kv.split("=", 1) for kv in args.path)
    src = CF.CONFIG_DIR / "decoder" / f"{args.arm}.toml"
    if not src.is_file():
        sys.exit(f"缺配置 {src}")
    d = CF.load(src, overrides)
    out_dir = Path(d["run"]["out_dir"])
    rdir = Path(args.resolved_dir) if args.resolved_dir else out_dir / "configs_resolved"
    files = CF.materialize_group("decoder", rdir, overrides)
    cfg_path = rdir / CF.LEGACY_NAMES.get(f"decoder/{args.arm}.toml", src.name)
    print(f"[train_decoder] resolved {len(files)} configs -> {rdir}; arm config = {cfg_path}", flush=True)
    sys.argv = [sys.argv[0], "--config", str(cfg_path), *rest]
    if "bk" in d:
        from veraretouch_sprf.train import train_sprf_bk5 as ENTRY
    else:
        from veraretouch_sprf.train import train_sprf_xl as ENTRY
    print(f"[train_decoder] entry = {ENTRY.__name__}", flush=True)
    ENTRY.main()


if __name__ == "__main__":
    main()
