# 全量 global databuild（含 IAA）显存预算 — 2026-07-05

硬件基准：2 × H100 97.9GB（卡 0 / 卡 1）。盘点来源：dataset_build 全仓扫描
（config.yaml / docker/launch_reasoning.sh / core / source_qa / iaa_benchmark）。

## 一、单个消费者的显存

| 消费者 | 模型 | 精度 | 显存 | 常驻性 |
|---|---|---|---:|---|
| vLLM 在线标注（gen_instruction/reason/verify） | Qwen3.5-35B-A3B-FP8 | FP8 | **83.2GB**（=0.85×97.9 预留，权重 35GB+KV） | 常驻 docker，dp=2 每卡一副本 |
| Teacher 渲染器 | VeraRetouch llava_qwen2 ~1.1B | bf16 | 权重 2.2GB，峰值 **~5GB**（512² tiling bs8） | build 进程内常驻 |
| 本地 preset 渲染（gpu_render，本期迁入） | 无权重（torch 算子链+残差 LUT） | fp32 | **B=8: ~8.2GB / B=16: ~16.3GB / B=32: ~32.6GB**（全分辨率，~0.96GB/图 + compile 缓存 ~1GB） | 按需 |
| SAM3 主体分割（precompute） | sam3.pt | bf16 | **~5GB**（权重 3.3GB+激活） | precompute lane 常驻 |
| IAA 质检（source_qa MixedIAA） | ArtiMuse(InternVL-8B) + Charm(DINOv2-L) + pyiqa 小模型 | bf16/fp32 | **~20GB**（15 + 1.5 + ~2，懒加载共存） | 质检阶段常驻 |
| tag/aesthetic precompute | CLIP ViT-L/14 + 线性头 | fp16 | ~3.5GB | 用完即卸 |
| IAA benchmark 单 runner | OneAlign 19GB / HumanAesExpert 16GB / ArtiMuse 15GB / AesExpert 14GB / Charm <2GB | fp16/bf16 | **峰值 19GB**（一次一个模型） | 独立进程错峰 |

## 二、按阶段的每卡峰值

**阶段 1 · 离线预计算**（build 前，跑一次）：
- SAM3 lane（~5GB）+ caption/tag 走 vLLM 服务。若 vLLM 已起（83GB）：**同卡 88GB**，
  或 SAM3 放另一张卡 ~5GB。
- source_qa 质检（IQA+ArtiMuse+Charm ~20GB）单独一张卡即可。
- 阶段峰值：**每卡 ≤88GB**，宽裕。

**阶段 2 · 在线全量 build（当前设计，dp=2）**：
- 每卡 = vLLM 副本 83.2GB + teacher ~5GB ≈ **88GB / 卡**（≈90% 满载，这是设计红线）。
- SAM3/tag/aesthetic 全走缓存只读不占卡；IQA build 中不调用。
- **结论：2×H100 恰好装下当前 build，本卡再无余量给 IAA 或本地 preset 渲染。**

**阶段 3 · IAA（两种含义分开算）**：
- source_qa IAA 质检（ArtiMuse+Charm 混合分）：**~20GB**。与 build 同卡放不下
  （88+20 > 97.9）。两个选项：
  - **错峰（推荐）**：质检在 build 前/后跑，单卡 20GB，零改动；
  - **并发**：把该卡 vLLM `gpu-memory-utilization` 从 0.85 降到 **0.70**（≈68.5GB，
    KV cache 缩小、在线并发 32 可能要降到 ~20）→ 68.5+5+20 ≈ 93.5GB，可行但顶满，
    OOM 余量 <5GB，不建议长跑。
- IAA benchmark（OneAlign 等评测 harness）：**需要一张空卡错峰跑**（峰值 19GB），
  与 build 的 vLLM 副本天然互斥，本来也不是 build 的一部分。

## 三、把本地 preset 渲染（gpu_render）接进 build 的预算

农场替代渲染若与 build 同卡：B=16 要 16.3GB，装不下（88+16 > 98）。选项按优先级：
1. **B=8 + vLLM util 0.80**：78.3 + 5 + 8.2 ≈ **91.5GB**，可行（B=8 吞吐仅比 B=32 低 ~5%）；
2. 渲染挤进预计算阶段错峰跑（B=32 独占卡，1026+ img/min，100 preset 库全量图集很快清完）；
3. 若渲染量长期在线：专卡跑 gpu_render（B=32, ~33GB），另一张卡跑单副本 vLLM（dp=1，
   标注吞吐减半）。

## 四、结论（一句话版）

- **全量 build 本体**：2×H100 各 ~88GB，正好；这是 vLLM 0.85 预留决定的，不是真实活跃占用。
- **含 IAA**：错峰跑零额外显存需求（质检 20GB / benchmark 19GB 单卡即可）；
  要并发就把一张卡的 vLLM util 降到 0.70，代价是该卡在线标注并发下降 ~1/3。
- **含本地 preset 渲染**：推荐 B=8 + util 0.80（每卡 ~91.5GB），或错峰独占跑 B=32。
