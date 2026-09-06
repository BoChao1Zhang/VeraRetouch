#!/usr/bin/env python3
"""VLM 端到端评测入口（EPR-052 新增薄包装）：生成+读出（dump_readout）→ 注入 BK-FULL 执行器 + 守卫（eval_vlmadapt）。

  python -m veraretouch_sprf.eval.eval_vlm_e2e dump  <dump_readout 的全部参数>
  python -m veraretouch_sprf.eval.eval_vlm_e2e eval  <eval_vlmadapt 的全部参数>（--backend bk --also-oracle-lut ...）

守卫（原件内实现，此处只转调）：
  A-inj  注入预测向量的路径与 oracle_lut 路径在同一向量下输出逐位相等（eval_vlmadapt）
  A-lat  评测端重算的向量余弦 == 读出端记录值（|Δ| ≤ 1e-6，eval_vlmadapt）
  槽序→链序：adapter 输出为 slot 序（m=1..6 恢复序），求解器按 chain 序取 edits[:, k]，注入前 `lat.flip(0)`。
held-out d6 双卡流程见 scripts/submit_heldout_headline.sh / dump_half.sh / heldout_final.sh。
"""
from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("dump", "eval"):
        sys.exit(__doc__)
    sub = sys.argv.pop(1)
    if sub == "dump":
        from veraretouch_sprf.eval import dump_readout as M
    else:
        from veraretouch_sprf.eval import eval_vlmadapt as M
    M.main()


if __name__ == "__main__":
    main()
