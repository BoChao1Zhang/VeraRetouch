# veraretouch_sprf —— SPRF 局部精修主线工程包（EPR-052 扶正，2026-09-06）

从 `experiments/prs/EPR-051_masked-restore-production/stage0/sprf/` 的**现行**文件复制而来（原件不动；在跑作业引用的原件未触碰）。
每个文件 ← 原文件与 sha256 对照见 `PROVENANCE.md`；改动只有导入/路径解析四类机械改动，不改数学与默认数值。
设计与数字的权威文档仍是 EPR-051 的 `docs/REPORT_STATUS_20260906.md`、`SPRF_DERIVATION.md`、`RESULTS_v1.md`、`HANDOFF.md`。

## 目录
```
veraretouch_sprf/
  _paths.py            REPO / STAGE0 / SPRF_LEGACY 解析；ensure_sys_path()（q3vl.* 与 dataset_build/tools）；src(name) 原文件名→文件
  configs/             带 [paths] 节的 TOML（decoder/{clut_full,bkfull_adagn_ff_affhead,smoke_clut_full}、vlm/{sft_full,adapt_s2fb}）
                       + materialize()：解析 ${paths.*} 落成与 EPR-051 原配置逐字节相同的普通 TOML（scripts/check_configs.py 核对）
  data/                stage_targets（链重建/β 场/h*/A 断言）、archive_assets、train_stage0（切分/评测口径原件）、build_targets_bkfull（e*）
                       cot_text（指令模板/六段序列化/阶段 token/70-20-10 指令抽样；逐字节同原件）、q3vl_text、q3vl_data、
                       stage_assets2 / stage_assets_heldout / freeze_cot_snapshot2（VLM 快照与 y 图暂存）
  models/              stage_flow（FiLM 主臂）、stage_flow_bk/bk3/bk4（ADAGN/FF/AFFHEAD backend）、stage_backend_g4d、edit_cond、bk_load、
                       align_predictor(_time)；vlm/q3vl_common（Qwen3-VL 加载/LoRA/阶段 token）、vlm/adapter（Adapter + span-pool 读出）
  solver/              stage_solver / stage_solver_bk（逐阶段 Euler、ProductionBatch 禁入键、rollout 指标、restore 生产接口）
  train/               train_decoder（统一入口）→ train_sprf_xl→train_sprf（C-LUT-FULL）/ train_sprf_bk5→train_sprf_bk_core4（BK-FULL）；
                       train_vlm_sft（S1F-FULL）、train_vlm_adapt（S2F-B）、train_align_time（eval 复用）
  eval/                eval_decoder（batch_eval_bk）、eval_vlm_e2e（dump_readout + eval_vlmadapt 转调）、guards、probes/{probe_full_ckpt,parity_report,quick100_eval}
  rl/                  OPD/OPSD 与 GRPO 接线骨架（README、reward/、prompts/、configs/*.yaml 占位）
  scripts/             submit_decoder.sh、run_vlm_sft.sh、run_vlm_adapt.sh、eval_decoder.sh、dump_half.sh、heldout_final.sh、submit_heldout_headline.sh、
                       heldout_split.py、merge_gencache.py、smoke_a1_cpu.py、smoke_cot_roundtrip.py、check_configs.py、sync_from_legacy.py、derive_configs_from_legacy.py
```
运行环境：解码器线 `/home/bc/envs/databuild/bin/python`（py3.13/torch2.6，含 skimage）；VLM 线 `/home/bc/envs/q3vl_sft/bin/python`（py3.12/torch2.10/transformers 4.57.1/peft 0.15）。
一律 `PYTHONPATH=/home/bc/VeraRetouch`，以 `python -m veraretouch_sprf.<模块>` 运行。

## 数据律（不变，原文见 HANDOFF §8 / CLAUDE.md）
- 快照 v3：279 分片（249 在 `/mnt/nfs-ro/bc/data/datasets/epr051_stage0`，30 在 `/home/bc/data/builds/epr051_stage0`）；训练全量 655,461；held-out 4,560 逐 id 冻结
  （`stage0/snapshot_newdata_v3.heldout_ids.json`，sha `bc8b3ea7…`）；d6 层 1,464。切分 `sha1_verasplit-v1` + 冻结表，禁 ad-hoc。
- V_where/V_what/T_final 永不进训练；`winner_confidence=low` 不进主训与 GT；headline 只报 held-out normal-only。
- CoT 快照 S2：77,640（train 73,854 / val 3,786，剔 1 宽高比越界键）；held-out d6 标注 `contract=eval_only_heldout`（G4/G4b 断言禁入训练）。
- 指令：逐样本三档 `sha1("epr051-inst-mix-v1:"+key)` 70/20/10 long/medium/short；固定指令版全部作废（trash，禁读）。
- NFS 读走 `/mnt/nfs-ro`；本包只读，不写 `/mnt/nfs`。

## 条件契约（互斥，不混用；REPORT_STATUS §2.3）
| 契约 | e_m 来源 | 用途 |
|---|---|---|
| `oracle_lut` | 真值 LUT 行号 → 预计算逆表描述子(17³×4) → edit_enc(3 层 MLP) → 128 | 解码器训练；上界行 |
| `oracle_text` | GT CoT 喂 VLM → span-pool 读出 → adapter → 128 | 只测读出通路 |
| `predicted_text` | VLM 自生成 CoT → 读出 → adapter → 128（注入：edit_enc:=Identity，slot→chain `flip(0)`） | 真实推理 headline |
| null / roll / shuffle | 零向量 / 阶段错位 / 换样本 | 负控制（每行必带） |

## 守卫清单（缺一不出数）
A1 零初始化整链逐位恒等（`torch.equal`；`scripts/smoke_a1_cpu.py` 可 CPU 复现）· A2 teacher 抽查 · A3 组内逐键相同 · A6 禁入键删除后输出不变 ·
A8 源文件/配置 sha 冻结（provenance） · A10 锚点契约（rollout=uint8 y，teacher=fp32 r^m） · A12 梯度到达编辑向量路径 ·
K1–K4（BK 入口：臂/损失全开/config diff/sha 钉死；K4 钉值已按主线包文件重钉，见 PROVENANCE） ·
A-inj 注入路径==oracle_lut 路径逐位 · A-lat 评测端重算余弦==读出端记录（≤1e-6） · G4/G4b eval-only 泄漏 · X5 archive 补丁 · D-20 提交四步。

## 四个主结果如何复现（命令级；数字见 REPORT_STATUS §2.4 / §3）
前置：`export PYTHONPATH=/home/bc/VeraRetouch`；显存共存规则（已占 + 峰值 < 65 GB）；提交后 D-20 四步（`rm -f` 日志 → `ps -p` 判活 → tail → job.marker）。

1. **C-LUT-FULL**（FiLM，655k，20k 步，oracle_lut；峰值 37.5 GiB）
   ```bash
   bash veraretouch_sprf/scripts/submit_decoder.sh clut_full 0 40
   # 等价直跑：python -m veraretouch_sprf.train.train_decoder --arm clut_full   （→ train_sprf_xl.main → train_sprf.main）
   # 冒烟：python -m veraretouch_sprf.train.train_decoder --arm clut_full --smoke --limit-samples 10 --path runs_sprf=/home/bc/data/runs/epr051_sprf_smoke
   ```
2. **BK-FULL**（ADAGN+FF+AFFHEAD，同配方；峰值 16.3 GiB；VLM 线执行器）
   ```bash
   bash veraretouch_sprf/scripts/submit_decoder.sh bkfull_adagn_ff_affhead 1 20
   # 评测（B=8）：bash veraretouch_sprf/scripts/eval_decoder.sh <out_dir>/configs_resolved/bkfull_adagn_ff_affhead.toml --ckpt ckpt_last.pt
   ```
3. **S1F-FULL**（Qwen3-VL-4B 全参 SFT，视觉塔冻结；2 epoch 9,232 步；单卡 77.6 GiB，需近独占卡）
   ```bash
   q submit SPRF_SFT_S1F_FULL 0 /home/bc/data/runs/epr051_vlmsft/sft_s1f_full/job.log --mem-peak 78 -- bash veraretouch_sprf/scripts/run_vlm_sft.sh
   # 探针：python -m veraretouch_sprf.eval.probes.probe_full_ckpt --ckpt <ckpt_epochN> --keys ... --records ... --assets-index ... --out ...
   ```
4. **S2F-B**（冻结全参基座 + LoRA r16 + adapter，SmoothL1(latent)+CE；峰值 ~11 GiB）
   ```bash
   # 先建目标：python -m veraretouch_sprf.data.build_targets_bkfull --bk-run /home/bc/data/runs/epr051_sprf/bkfull_adagn_ff_affhead
   q submit SPRF_ADAPT_S2FB 1 /home/bc/data/runs/epr051_vlmsft/adapt_s2fb/job.log --mem-peak 12 -- bash veraretouch_sprf/scripts/run_vlm_adapt.sh
   # held-out d6 headline（1,464 键双卡生成 → 合并 → 读出 → BK-FULL 执行器 + oracle_lut 同表）：
   bash veraretouch_sprf/scripts/submit_heldout_headline.sh /home/bc/data/runs/epr051_vlmsft/adapt_s2fb/ckpt_epoch1 s2fb 22
   # 两半 Done 后按其打印的 heldout_final.sh 命令合并评测
   ```

## 自检（EPR-052 已跑，结果见 experiments/prs/EPR-052_rl-postraining/ENG_REPORT.md）
`scripts/smoke_a1_cpu.py`（两臂 A1，CPU 10 像素）· `scripts/smoke_cot_roundtrip.py`（一条记录序列化往返 + template_sha256 与 legacy 相等）·
`scripts/check_configs.py`（物化配置与原配置逐字节相同）· 每个模块 `python -c "import veraretouch_sprf.<mod>"`。

## 与 legacy 的关系
- provenance 里记录的源文件 sha 会与 EPR-051 已登记的 run 不同（文件改了导入）；跨 sha 比较按 HANDOFF §8 红线处理，对照表见 PROVENANCE.md。
- `_P.src(name)` 对未扶正的历史文件（bk_core/core2/core3、train_align.py 等）回落到 legacy 目录，只用于 provenance 记录与 K4 钉死。
- 未扶正的历史入口（bk/bk2/bk3/bk4/bk6/bk7、lossabl、passk_*、run_*.sh）仍在 legacy 目录。
