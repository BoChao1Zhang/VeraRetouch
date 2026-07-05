# 50k 样本整夜执行手册（2026-07-06 凌晨）

目标：**5 万张高质量 img 样本** = degraded ~20k + global ~30k。

## 正在跑的（凌晨）
| 任务 | 进程/日志 | 速率 | ETA |
|---|---|---|---|
| IAA 回填（keep∧bq3 86112 张） | pid 3453263 · scratchpad/iaa_prod.log | ~168/min | ~08:30 |
| 退化链波次生产（目标 20k） | degrade_waves.sh · scratchpad/degrade_waves.log | 冷启 ~12.5/min，16 workers 预计 20-30/min | 池子限速，随回填增长 |
| vLLM 35B（annotate/verify） | 容器 reason_g1 · GPU1 :8002（broker :8003） | conc 32 | 常驻 |

## GPU 布局（红线：卡 0 我方 ≤47GB，卡 1 全占）
- 卡 0：IAA 回填 27.8GB（唯一我方占用）
- 卡 1：vLLM 68GB（util 0.70）+ gpu_render 渲染尖峰 ~8GB（B=1 单张）

## 早晨接力：global 链（IAA 回填完成后）
回填完成信号：监控事件「IAA 生产进程退出」或 `iaa_prod.log` 出现 finish。
然后启动（**必须用 iaa437 venv**——construct QA 的 IAA 依赖 transformers 4.37）：
```bash
cd /home/bc/VeraRetouch
RENDER_BACKEND_POLICY=throughput /home/bc/.venvs/iaa437/bin/python -m construct.agent run \
    --n 16000 --render-n 4 --out /home/bc/data/datasets/vera_directionA_1M/construct_global_v3
```
- 每 source 期望 ~2 SFT → 16k source ≈ 30k+ 样本；QA-IAA 占卡 0 ~25GB（回填已退出）
- 渲染分流：本地 56%+（throughput 策略）/ 农场 ≤25% per-source cap
- annotate 1.0-1.4 次/样本 + verify ~0.15 次（q∈[0.45,0.70] 才跑）

## 决策记录
- IAA 阈值维持 GATE 55（健康分布 median 58.4，保留 ~62% ≈ 5.3 万源，量质平衡）
- scene="any"（45k 张）作为独立配额桶（12%），不做 vLLM 重分类（省 35B 容量给 annotate；
  后续可用空闲期回填细分类）
- 旧 202 行 IAA 分数确认为污染（ArtiMuse 静默 ImportError → 纯 Charm 分），已清污重打
- Track-A（E+R 专家反演）与 region-local 退化 v2 明确超出本夜范围

## 验收口径
- 数量：degrade samples.jsonl + global sft.jsonl 合计 ≥50k 行
- 质量抽检：每链随机 50 条人检 instruction 无泄露（风格名/参数/方向词按任务类型规则）、
  图像对可视合理；degrade 全过 ΔE/clip 门；global 全过 IAA q 门
