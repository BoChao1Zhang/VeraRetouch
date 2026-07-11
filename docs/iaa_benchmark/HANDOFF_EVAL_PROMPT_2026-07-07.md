# IAA Eval Handoff Prompt

Date: 2026-07-07

```text
你接手 /home/bc/VeraRetouch 的 IAA final evaluation。只做缺失实验，不要重复已经完成的预测和 metrics。

硬性要求：
- 当前主方向是 ArtiMuse + Charm 混合 IAA；QA analysis/问卷方式已经归档，只能作为历史 baseline。
- 已完成的实验一律复用现有 prediction/metric/report 文件；发现同名完整输出时不要重跑。
- 需要在 ~/.ssh/config 里的 exp-remote 机器上跑缺失实验：HostName 172.25.76.170, Port 33335, User bc。远端只确认过 /data/bc，先定位或同步 VeraRetouch repo、数据和模型，不要假设 /home/bc 路径必然存在。
- 下载缺失模型时先查远端已有缓存；脚本优先用 ~/hfd.sh，若远端实际只有 ~/.hfd.sh 则用 ~/.hfd.sh，并在报告里写清楚。
- GPU 使用卡 0。只有确实需要 OpenAI-compatible serving 的模型才启动 vLLM Docker；ArtiMuse 的官方 score head 目前以 native InternVLChatModel.score() 为可信路径，vLLM 路径未验证等价。

已有本机评测产物，作为事实基线：
- ArtiMuse-10K eval root: /home/bc/data/datasets/artimuse10k_eval
  - metadata: /home/bc/data/datasets/artimuse10k_eval/metadata.csv
  - model input: /home/bc/data/datasets/artimuse10k_eval/generic_image_score.jsonl
  - predictions: /home/bc/data/datasets/artimuse10k_eval/predictions/*.jsonl
  - metrics: /home/bc/data/datasets/artimuse10k_eval/metrics/summary.md
  - report: /home/bc/data/datasets/artimuse10k_eval/REPORT.md
  - completed metrics: ArtiMuse PLCC 0.6237 / SRCC 0.6150 / n 1002; Charm-AVA-frequency PLCC 0.2896 / SRCC 0.2951 / n 1001; OneAlign PLCC 0.3038 / SRCC 0.3172 / n 1002; OneAlign-paper-iaa-prompt PLCC 0.3051 / SRCC 0.3184 / n 1002.
- Photographer-IAA eval root: /home/bc/data/datasets/photographer_iaa_benchmark
  - benchmark root: /home/bc/data/datasets/photographer_iaa_benchmark/para_v1
  - metadata: /home/bc/data/datasets/photographer_iaa_benchmark/para_v1/metadata.csv
  - model input: /home/bc/data/datasets/photographer_iaa_benchmark/model_inputs/generic_image_score.jsonl
  - predictions: /home/bc/data/datasets/photographer_iaa_benchmark/predictions/*.jsonl
  - metrics: /home/bc/data/datasets/photographer_iaa_benchmark/metrics/summary.md
  - report: /home/bc/data/datasets/photographer_iaa_benchmark/REPORT.md
  - completed metrics: Charm-PARA-random PLCC 0.9782 / SRCC 0.9425 / n 1800; ArtiMuse PLCC 0.8555 / SRCC 0.8343 / n 1800; OneAlign PLCC 0.8491 / SRCC 0.4199 / n 1800; AesExpert-LLaVA-7B PLCC 0.8004 / SRCC 0.6038 / n 1800.

只补这些缺失项，且仅在远端确认没有完整输出时才跑：
- Photographer-IAA 的 final Charm setting: Charm-AVA-frequency。注意现有 Photographer Charm 结果是 PARA-random，不是最终 Charm-AVA-frequency。
- ArtiMuse+Charm mixed metrics on ArtiMuse-10K and Photographer-IAA。混合分用同一 benchmark_id 上的 pred_score_0_100：mixed = 0.75 * ArtiMuse + 0.25 * Charm；只在两个模型都有有效分数的样本上评测，并报告 coverage。
- QA-AES questionnaire baseline on ArtiMuse-10K and Photographer-IAA。只用 reliable=true 且 merit_frac 非空的记录，pred_score_0_100 = merit_frac * 100；映射到 benchmark_id 的规则必须写进报告。如果找不到可靠映射，不要编造结果，明确写 blocked by missing mapping。

评测工具：
- 用 dataset_build/iaa_benchmark/evaluate_predictions.py 计算单个 prediction JSONL 的 PLCC/SRCC/KRCC/RMSE/MAE/bias。
- 用 dataset_build/iaa_benchmark/summarize_metrics.py 汇总 metrics JSON。
- 输出命名建议：
  - predictions/charm_ava_frequency.jsonl
  - predictions/artimuse_charm075_025.jsonl
  - predictions/qa_aes_merit_frac.jsonl
  - metrics/charm_ava_frequency.metrics.json
  - metrics/artimuse_charm075_025.metrics.json
  - metrics/qa_aes_merit_frac.metrics.json

最终报告必须回答：
- Charm、ArtiMuse、ArtiMuse+Charm mixed、QA-AES questionnaire baseline 在两个 benchmark 上的 PLCC/SRCC/KRCC/coverage。
- ArtiMuse+Charm mixed 是否高于 QA-AES，分别按 PLCC 和 SRCC 判断；如果 QA-AES coverage 或映射不足，结论必须带 coverage 限定。
- 哪些实验是复用已有结果，哪些是本次新跑/新算，哪些因为缺模型、缺映射或 OOM 没完成。

背景结论：
- ArtiMuse-10K 上 ArtiMuse local result SRCC 0.6150 / PLCC 0.6237，接近 ArtiMuse paper Table 3 的 SRCC 0.614 / PLCC 0.627；这里没有明显复现差距。
- OneAlign 低于 target-dataset fine-tuned paper number，主要是因为本地跑的是 off-the-shelf/AVA-generalization 语境，不是 ArtiMuse-10K target fine-tuning；paper-style prompt 没显著改变结果。
- ArtiMuse vLLM 吞吐尚未被可靠验证；已有 native smoke log 是 5 images 约 3.27 img/s final tqdm rate，不要把它写成 vLLM 吞吐。
- 显存预算参考 docs/DATABUILD_VRAM_BUDGET_2026-07-05.md：ArtiMuse 约 15GB，Charm <2GB，source_qa MixedIAA 加 pyiqa 约 20GB；它们单卡错峰可跑，但不能和 full build 的 83GB vLLM replica 同卡常驻。
```
