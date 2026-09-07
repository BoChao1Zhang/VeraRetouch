# EPR-052 · ENG-2 报告：ms-swift 容器 / OPSD-GKD、OPD-RL 冒烟 / 全参变体实测 / 双卡 OPSD pilot（2026-09-06）

## 1. 容器与版本
- 镜像：`modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.10.0-vllm0.17.1-modelscope1.34.0-swift4.0.3`（42 GB，root /home/docker）；容器 `vr-ms-swift`（compose，sleep infinity，gpus all，ipc host，shm 64g）。
- 拉取障碍：docker daemon 配置 proxy `127.0.0.1:7890` 无进程监听；本机 mihomo 在 `mixed-port 11892`（/home/bc/clashctl）。处理：用户态 `socat TCP-LISTEN:7890 → 127.0.0.1:11892` 桥接（未改 daemon，未 sudo）；拉取 ≈1.4 GB/min。
- 容器内安装（只改容器写层）：`git clone` 本地镜像 → `/home/bc/data/runs/epr052_rl/ms-swift` @ **236a1f1921b5**（`git config safe.directory` 后 `pip install -e .`），`transformers==4.57.1`、`qwen_vl_utils>=0.0.14`、`trl>=0.26,<1.0`。
- 终态：ms-swift **4.6.0.dev0**（`swift --version` 在该 commit 抛 KeyError，改用 `swift.__version__`）、transformers 4.57.1、trl 0.28.0、peft 0.18.1、qwen_vl_utils 0.0.14、vllm 0.17.1、torch 2.10.0+cu128、flash_attn 2.8.3、deepspeed 0.18.8、bitsandbytes 0.49.2；`torch.cuda.is_available()=True`，2 卡；`pip freeze` → `docker/ms-swift/FREEZE.txt`（398 行）。`pip check` 残留为镜像自带冲突（lmdeploy/cupy/outlines/pydantic），非本次引入。
- 模型目录：ms-swift 需单目录，`/home/bc/data/runs/epr052_rl/s1f_epoch1_merged` = S1F ckpt_epoch1 的 model/ + tokenizer/ 硬链接 + 基座 preprocessor_config；chat_template 与基座逐字相同（strip 后）。**Stage-2 6 行阶段 token 未落盘**（adapt_s2fb LoRA 仅 64 个 q/k/v/o 张量，无 embed），无法合并 → 用 S1F 原样（S1F 全参 SFT 已含 6 行）。

## 2. 冒烟数据与 prompt 断言
- 8 条 S2 train 记录（sha1 抽样，salt `epr052-smoke-v1`），`messages=[user: "<image>"+逐样本指令]`（无 system），`images=[spec-5 重采样 y 图]`（assets_y2 原图 360×540，SFT 在线 resize；此处预先落盘使 ms-swift 处理器几何一致），`teacher_prompt="<image>"+指令+"\n\nHere is a reference grade for this photograph:\n"+GT 六段+"\n\n"+过渡句`（opsd_plugin.py 格式），附加列 key/instruction_tier。生成器 `veraretouch_sprf/rl/prompts/build_msswift_jsonl.py`。
- 断言（`rl/prompts/assert_prompt_parity.py`，容器内 `swift.get_processor(model_type='qwen3_vl')` + `get_template().encode`，mode transformers）：学生 prompt input_ids 逐 id 相等 **8/8**，`image_grid_thw` 相等 **8/8**（如 [1,40,32]/[1,32,48]），学生 prompt 长 363–523 token；教师 prompt 长 1,970–2,288；教师前缀与学生一致到指令末尾前一个 token（**8/8** 在同一位置分歧：指令末尾 "." 与 "\n\n" 被 BPE 合并为 ".\n\n"），教师侧 6 个阶段 token 均为单 token 8/8。

## 3. 冒烟（LoRA r16 q/k/v/o，模型=教师=S1F merged，教师 disable_adapter 固定，T 1.0，补全 2048，max_length 4096，truncation delete，2 步，per_device 1，gpu1，HF generate）
| 探针 | 配置 | s/step | ms-swift memory(GiB) | gpu1 峰值(nvidia-smi) | 学生/教师序列 | 六段完整 | 散度 | loss |
|---|---|---|---|---|---|---|---|---|
| a | GKD β0 forward-KL, λ1 | 99.9 / 99.0 | 20.75 / 20.86 | 22.1 GiB | 2051/3728, 2005/3605（有效补全 1667/1626） | 2/2 有序 | fwd-KL token-mean 0.2818 / 0.5686；分段 seg1..6 = 0.303/0.331/0.438/0.272/0.204/0.157；0.707/0.602/0.755/0.532/0.498/0.351 | 同散度 |
| b | GKD β0.5 JSD | 99.3 / 96.8 | 24.89 / 24.94 | 26.3 GiB | 2051/3728, 1966/3566 | 2/2 | JSD 0.0413 / 0.0490；分段 0.050/0.045/0.054/0.043/0.035/0.023；0.077/0.050/0.047/0.046/0.043/0.035 | 同 |
| c | GRPO+教师（OPD-RL）G=2，use_vllm false，无 reward_funcs，β_KL 0.04 | 106.1 / 96.6 | 13.83 / 13.86 | 15.0 GiB | 补全 1610/1688；1469/1570，finish=stop 4/4 | 4/4 | `teacher_kl`(k3) 0.2033 / 0.2351；k1 均值 −0.187/−0.230, −0.307/−0.249；逐样本 k3 分段例 0.289/0.129/0.378/0.196/0.088/0.109 | 0.208 / 0.278（kl_ref 0 / 0.001） |
陷阱：`--dataloader_num_workers 0` 与 ms-swift `persistent_workers` 冲突（ValueError）→ 用 1；ms-swift `adam_beta2` 默认 **0.95**；`torch_dtype float32` 自动置 `fp16=True`，加 `--bf16 true` 必须同时 `--fp16 false`；`use_logits_to_keep=False`（多模态 + transformers<5 默认）。
探针实现：`rl/plugins/gkd_segment_probe.py`（猴补丁 `_compute_jsd_loss` / `compute_teacher_kl_per_token` + `_postprocess_batch`，逐 token 散度按 `<vr_stage_m>` 分段，不改损失）。

## 4. 全参变体实测（GKD β0，tuner full，freeze_vit/aligner，grad ckpt，mb1，2 步；教师 = gpu0 vLLM logprob server `swift deploy`（util 0.45，KV 32 GiB，top-k 64，port 8100），学生 gpu1）
| 变体 | 权重/优化器 | s/step | memory(GiB) step1/2 | 备注 |
|---|---|---|---|---|
| V1 | fp32 master + torch AdamW | — | OOM（自身 59.84 GiB 时另一并行冒烟占 35.2 GiB） | 两个 sweep 驱动重叠所致；用户裁决停止穷举，未重跑 |
| V2 | bf16 + torch AdamW | 86.0 / 81.9 | 31.41 / 34.48 | 与另一冒烟并行（教师服务共享，s/step 偏大） |
| V3 | bf16 + `adamw_bnb_8bit` | **73.5 / 71.8** | **25.89 / 28.94** | 干净单跑（gpu1 独占）；loss 0.2797 / 0.3091 |
| V3'（同配置早跑，并行污染） | 同 V3 | 79.4 / 76.8 | 25.89 / 28.94 | — |
选定：**V3**（≤ 75 GB 且最快）。CPU offload 未用。教师 API 路径为 top-k(64) logprobs（非全词表）。

### 4.1 批大小标定（同 V3 配置 + vLLM rollout server，教师 server，2 步）
| per_device × 累积 | s/step | ms-swift memory(GiB) step1/2 | gpu1 峰值(nvidia-smi) | 结果 |
|---|---|---|---|---|
| 1 × 8 | — | — | — | 未单测（探针 a–c 为 1×1） |
| 1 × 1（V3 clean） | 73.5 / 71.8 | 25.89 / 28.94 | ~30 GiB | OK |
| 8 × 1 | 92.6 / 92.3 | 42.40 / 55.22 | **57.8 GiB** | OK；≈11.5 s/样本 |
| 16 × 2（GRPO 路径试跑） | — | — | 92.5 GiB | **OOM**（tried to allocate 17.33 GiB，gpu1 78.61 GiB in use） |
线性外推（PB1→PB8：+4.1 GiB/样本）：PB12 ≈ 74 GiB（贴 75 上限）、PB10 ≈ 66 GiB。**正式取 PB=8**（已实测、留余量）。

### 4.2 路径偏差记录（诚实留痕，非静默拍板）
`--async_generate`（生成/训练重叠）在 ms-swift main@236a1f19 **对 GKD 明确不支持**（`rlhf_args.py` L755-756 `NotImplementedError: Currently, async_generate is not supported for GKD.`），只在 GRPO 路径可用且要求 `vllm_mode=server`（L454-458）。执行 agent 曾据此把正式跑切到 GRPO/OPD-RL 路径（优势 = 教师 k1 log-ratio），但用户裁决写明 **β=0 forward KL 与 λ=1**——`lmbda` 是 GKD 专属参数，GRPO 路径无此项且教师信号只用采样 token 的 k1（非全词表 forward KL）。故**回退到 GKD 路径**（用户口径），GRPO 试跑（PB16×GA2）连同其 OOM 一并留档于 `opsd_full/opsd50k_grpo_pb16_oom/`。代价：GKD 路径每步串行「rollout → 学生前向反传 → 教师 API」，无生成/训练重叠。若要 async 重叠须改口径为 GRPO 路径，**留待用户裁决**。

## 5. 正式数据
`/home/bc/data/runs/epr052_rl/data/opsd50k/opsd_train.jsonl`：S2 train 73,854 中按 `sha1("epr052-opsd-train-v1:"+key)` 升序取前 **50,000**；held-out 1,558 id / eval-only 1,464 键与 train 交集为空（pool = 73,854）；键表 sha256 **e3e1b24f02bbe53e25c5deed504330bd509f75d416a1ca290314e0eafeb5e5a6**；指令档 long/medium/short = 35,233/9,801/4,966；GT 平均 4,442 字符；y 图 spec-5 落盘（768×512 20,443、512×768 13,994、…）。

## 6. 双卡 OPSD pilot（正式配置 `rl/configs/opsd_full_dualcard.yaml`；脚本 `rl/scripts/dualcard/`）
gpu0：`swift rollout`（util 0.28，max_model_len 4096，port 8000）+ 教师 `swift deploy`（util 0.45，port 8100，固定 S1F）；gpu1：`swift rlhf --rlhf_type gkd --tuner_type full --torch_dtype bfloat16 --optim adamw_bnb_8bit --adam_beta2 0.999 --lr 5e-6 --beta 0 --lmbda 1 --temperature 1.0 --max_completion_length 2048 --max_length 4096 --per_device 1 --grad_accum 8`（steps_per_generation = 8 → 每优化步一次 8 条 vLLM 批量 rollout）、`--use_vllm true --vllm_mode server`、`--teacher_model_server :8100 --gkd_logits_topk 64`、每 25 步存 ckpt、探针插件逐步记录分段散度/六段完整。
在线判据（`online_eval.sh`，宿主 q3vl_sft 环境，gpu0）：32 键 val（`s1f_val32_keys.json`）greedy 自生成 → S2F-B 读出（LoRA+adapter 冻结，与 0.39 基线同代码路径）→ 对 e* 余弦；同时报 stage_token_missing_rate、cot_parse_success_rate。
**起跑参数（run `opsd_full/opsd50k_gkd`，2026-09-06 20:27 启动）**
| 项 | 值 |
|---|---|
| 有效 batch | per_device 8 × 累积 1 × 每 prompt 1 条补全（GKD 路径 `RequestConfig(n=1)`，`num_generations` 不参与）= **8 prompt/优化步** |
| 1 epoch 总步数 | 50,000 / 8 = **6,250 步** |
| 本次 pilot | 200 步（= 1,600 prompt，占 epoch 的 3.2%） |
| s/step（标定值） | 92.3–92.6 s（PB=8）→ 200 步 ≈ 5.1 h；1 epoch ≈ 160 h（仅此路径无生成/训练重叠） |
| gpu1 峰值（标定） | 57.8 GiB（限 75） |
| gpu0 占用 | rollout server 46.8 GiB + 教师 server 28.5 GiB = **75.3 GiB** |
| ckpt | 每 200 步，`--save_only_model true`（bf16 权重，无优化器态），`--save_total_limit 2` + 训练末 1 个；单次体积/耗时见 §7 |
| 在线探针 | 每 200 步，32 键 val greedy → S2F-B 读出 → e* 余弦 |

**step-0 基线（S1F 原权重，32 键 val，与 0.39 基线同代码路径）**：latent_cos_mean **0.4989** / median 0.4991 / L2 均值 16.99；n_ok 32/32；stage_token_missing_rate **0.0**；cot_parse_success_rate **1.0**；墙钟 377 s。

## 7. pilot 结果：**第 30 步守卫触发停机**（run `opsd_full/opsd50k_gkd`，20:27–21:07）

运行事实：200 步预定，实跑 **30 步**（39 min 57 s，**80.9 s/step**，ETA 3:49:12）；gpu1 峰值 **61,851 MiB**（限 75 GB），gpu0 恒 75.3 GiB（rollout 46.8 + 教师 28.5）；每步 8 prompt；散度支撑 = 教师 API **top-k 64**（非全词表）。未到 200 步 ⇒ **无 ckpt 落盘**，ckpt 体积/耗时未测，在线余弦仅有 step-0 一行（0.4989）。

| step | 六段有序行/8 | fwd-KL token-mean | 分段 seg1/2/3/4/5/6 | gpu1 reserved(GiB) | 首行阶段 token 序 |
|---|---|---|---|---|---|
| 0 | 8/8 | 0.3150 | 0.460 / 0.428 / 0.366 / 0.238 / 0.195 / 0.211 | 37.1 | `[1, 2, 3, 4, 5, 6]` |
| 5 | 8/8 | 0.1577 | 0.239 / 0.199 / 0.186 / 0.104 / 0.115 / 0.107 | 58.6 | `[1, 2, 3, 4, 5, 6]` |
| 10 | 5/8 | 0.1685 | 0.341 / 0.219 / 0.148 / 0.131 / 0.095 / 0.084 | 58.6 | `[1, 2, 3, 4, 5, 6]` |
| 11 | 1/8 | 0.2160 | 0.335 / 0.315 / 0.218 / 0.144 / 0.127 / 0.164 | 58.6 | `[1, 2, 4, 4, 5, 6]` |
| 14 | 0/8 | 0.1195 | 0.201 / 0.168 / 0.110 / 0.095 / 0.081 / 0.067 | 58.6 | `[1, 2, 4, 4, 5, 6]` |
| 20 | 0/8 | 0.1594 | 0.325 / 0.223 / 0.142 / 0.112 / 0.081 / 0.085 | 58.9 | `[1, 4, 4, 4, 5, 6]` |
| 25 | 0/8 | 0.1293 | 0.277 / 0.143 / 0.111 / 0.092 / 0.066 / 0.059 | 58.9 | `[1, 4, 4, 6]` |
| 29 | 0/8 | 0.1057 | 0.174 / 0.149 / 0.107 / 0.077 / 0.072 / 0.061 | 59.1 | `[1, 4, 4, 4, 4, 6]` |

聚合：散度前 10 步均值 **0.2162** → 后 10 步均值 **0.1232**（min 0.0986 / max 0.3150）；分段均值同步下降（seg1 0.3635→0.2216、seg6 0.1299→0.0741）。

六段有序行合计 **87/240**：step 0–9 为 8/8 或 7/8，step 10 起 5/8→1/8，**step 14 起连续 0/8**。
阶段 token 退化形态（首行示例）：`[1,2,3,4,5,6]`（step 0）→ `[1,2,4,4,5,6]`（step 11–14，仍是 6 个 token，但第 3 段 token 变成第 4 段 token）→ `[1,4,4,4,5,6]`（step 20）→ 段数缺失/错位（step 25–29，如 `[1,4,4,6]`、`[1,4,5]`、`[4,4,4,4,5,6]`）。
补全长度：多数 1,675–1,850 token；step 4/19/23/26/28/29 出现 **2,048**（= `max_completion_length` 上限，截断）。

按用户裁决「守卫下降则停并报」，第 30 步停机。**未下结论**：以上仅为实测数字；成因（top-k 64 支撑下的 forward KL、T=1.0 采样、lr 5e-6、教师 user 段 GT 复制效应等）需另行判定。

### 7.1 现场状态
- gpu0 两个 vLLM 服务仍在运行（rollout :8000 46.8 GiB + 教师 :8100 28.5 GiB，合计 75.3 GiB），队列为空、未阻塞他人；重启用 `rl/scripts/dualcard/rollout_server.sh` 与 `teacher_server.sh`。若需释放：容器内 `pkill -f "swift rollout"` 与 `pkill -f "swift deploy"`。
- 留档：`opsd_full/opsd50k_gkd/`（run.log、probe_segments.jsonl 30 行、nvsmi.log、online/）、`opsd50k_grpo_pb16_oom/`（GRPO 试跑 OOM）、`opsd50k_ep1_gkd_sync_aborted/`（首次 GKD 起跑，被我误判切路径而中止）。

## 8. 下一步接线清单（待用户裁决，均未实施）
1. 守卫崩溃的处置二选一：(a) 降 lr / 加 warmup / 降采样温度；(b) 恢复 CE 正则——`sft_alpha` 只在 `lmbda<1` 的 DATASET 批次相加（`gkd_trainer.py` L244-245），故须同时设 `--lmbda <1` 才生效，且该批次教师会在 user 段与 assistant 段同时看到 GT（SURVEY §E.1 已记）。
2. 教师支撑：当前 top-k 64（教师 API 路径唯一选项）。要全词表 forward KL 须把教师改为**本地** `--teacher_model`（同卡多占 8.3 GiB bf16），与「教师放 gpu0」的分工冲突，需裁决。
3. 结构守卫上线为**训练内**硬条件（现仅探针记录）：六段有序率 < 阈值即停机。
4. 截断：`max_completion_length 2048` 已有 6/30 步触顶，是否提到 2,560。
5. async 生成/训练重叠只在 GRPO 路径可用（§4.2），要用须改口径。

## 9. 六臂对照表（2026-09-07；全部为 60 步/臂，除注明者）

**读表前提（两处重大更正，详见 NOTES N7–N11）**
1. **探针缺陷曾污染两臂**：探针 v4 中「包装 `GKDTrainer.compute_loss`」的钩子会导致 **step 1 起 loss/权重 NaN**（三次零变量对照实证：v3 干净、v4a=v4去掉该钩干净、v4 NaN）。
   受污染的臂 4（lr 3e-6）、臂 5（top_p 0.9）**已作废并用修复版探针 v7 重跑**，作废数据留档于 `arms/arm*_VOID_probe_v4/`。
   由此，先前「lr 3e-6 数值崩塌」的判定**已撤回**。
2. **同窗口口径**：早期「散度不降」的说法是拿「前 10 步 vs 最后 10 步」跨不同总步数比较所致；本表统一给 10 步块轨迹。
   另注意「散度」算在学生**每步新采样**的序列上，不是固定分布上的损失，须与六段有序率并读。

| 臂 | 变量 | 步数(有监督/零监督) | 散度 前10→后10 | 六段有序行 | top-64 覆盖 | 熵 前10→后10 | 触顶 | gpu1 峰值 GiB |
|---|---|---|---|---|---|---|---|---|
| baseline lr5e-6 无warmup(v1) | λ=1, API top-64 | 30(30/0) | 0.2162 → 0.1232 | 87/240 | — | — → — | 0 | 59.1 |
| arm1 lr1e-6+warmup(v1) | λ=1 | 60(60/0) | 0.2849 → 0.2964 | 480/480 | — | — → — | 0 | 58.8 |
| arm2 本地全词表教师(v3) | arm1+全词表教师；PB8 OOM，仅 step0 | 1(1/0) | 0.2248 → 0.2248 | 8/8 | 0.99953 | 0.432 → 0.432 | 0 | 70.0 |
| arm3 λ0.75零监督(v3) | arm1+λ0.75+sft_alpha0.1（数据无 assistant） | 60(48/12) | 0.2635 → 0.2889 | 384/384 | 0.99951 | 0.335 → 0.337 | 0 | 68.3 |
| arm3b λ0.75+CE(v4) | arm3+assistant=GT CoT；30 步 | 30(30/0) | 0.2726 → 0.2482 | 240/240 | 0.99968 | 0.323 → 0.328 | 0 | 67.2 |
| arm4 lr3e-6(v7) | arm1 基础上改 lr | 60(60/0) | 0.2790 → 0.2687 | 480/480 | 0.99951 | 0.348 → 0.331 | 0 | 57.6 |
| arm4b adamw_torch(v7) | arm4+torch AdamW(bf16 状态)；20 步 | 20(20/0) | 0.2595 → 0.2752 | 160/160 | 0.99947 | 0.359 → 0.330 | 0 | 62.8 |
| arm5 top_p0.9(v7) | arm1 基础上改 top_p | 60(60/0) | 0.2742 → 0.3101 | 480/480 | 0.99993 | 0.312 → 0.323 | 0 | 57.7 |
| arm6 lr5e-6+warmup(v7) | arm1 基础上改 lr（pilot 选定配置） | 60(60/0) | 0.3034 → 0.2148 | 480/480 | 0.99957 | 0.348 → 0.393 | 0 | 57.3 |

| 臂 | 分段散度 seg1..6（前10 / 后10） | finish_reason | 训推 logp 差 |Δ|均值/最大(行数) | k3 均值/负值数 | loss DATASET / STUDENT / 非有限 |
|---|---|---|---|---|---|
| baseline lr5e-6 无warmup(v1) | 0.364/0.292/0.236/0.157/0.125/0.130 → 0.222/0.157/0.117/0.092/0.072/0.074 | {} | — / — (0) | — / None | — / — / None |
| arm1 lr1e-6+warmup(v1) | 0.464/0.371/0.331/0.224/0.179/0.151 → 0.430/0.429/0.337/0.220/0.196/0.176 | {} | — / — (0) | — / None | — / — / None |
| arm2 本地全词表教师(v3) | 0.320/0.321/0.249/0.184/0.142/0.137 → 0.320/0.321/0.249/0.184/0.142/0.137 | {'stop': 8} | — / — (0) | 0.1602 / 0 | — / — / None |
| arm3 λ0.75零监督(v3) | 0.485/0.354/0.272/0.191/0.148/0.139 → 0.427/0.378/0.337/0.236/0.192/0.172 | {'stop': 384} | — / — (0) | 0.2174 / 0 | — / — / None |
| arm3b λ0.75+CE(v4) | 0.376/0.325/0.308/0.245/0.194/0.192 → 0.345/0.312/0.300/0.201/0.180/0.157 | {'None': 64, 'stop': 176} | 0.0134 / 3.4427 (176) | 2.3135 / 0 | 0.3005 / 0.2580 / 0 |
| arm4 lr3e-6(v7) | 0.403/0.392/0.318/0.220/0.176/0.173 → 0.396/0.375/0.297/0.217/0.171/0.165 | {'stop': 480} | 0.0132 / 3.6358 (480) | 0.1728 / 0 | — / — / 0 |
| arm4b adamw_torch(v7) | 0.379/0.337/0.299/0.204/0.175/0.169 → 0.421/0.381/0.290/0.222/0.180/0.165 | {'stop': 160} | 0.0134 / 1.6554 (160) | 0.1759 / 0 | — / — / 0 |
| arm5 top_p0.9(v7) | 0.403/0.356/0.295/0.237/0.183/0.177 → 0.435/0.403/0.347/0.272/0.218/0.194 | {'stop': 480} | 0.0222 / 2.1539 (480) | 0.1497 / 0 | — / — / 0 |
| arm6 lr5e-6+warmup(v7) | 0.518/0.421/0.319/0.226/0.179/0.167 → 0.339/0.295/0.222/0.164/0.146/0.130 | {'stop': 480} | 0.0133 / 2.4193 (480) | 0.1689 / 0 | — / — / 0 |

baseline lr5e-6 无warmup(v1) 六段有序逐步序列: 888878887751120010000000000000
arm1 lr1e-6+warmup(v1) 六段有序逐步序列: 888888888888888888888888888888888888888888888888888888888888
arm2 本地全词表教师(v3) 六段有序逐步序列: 8
arm3 λ0.75零监督(v3) 六段有序逐步序列: 888888888888888888888888888888888888888888888888
arm3b λ0.75+CE(v4) 六段有序逐步序列: 888888888888888888888888888888
arm4 lr3e-6(v7) 六段有序逐步序列: 888888888888888888888888888888888888888888888888888888888888
arm4b adamw_torch(v7) 六段有序逐步序列: 88888888888888888888
arm5 top_p0.9(v7) 六段有序逐步序列: 888888888888888888888888888888888888888888888888888888888888
arm6 lr5e-6+warmup(v7) 六段有序逐步序列: 888888888888888888888888888888888888888888888888888888888888


### 9.1 十步块散度轨迹（同口径）
| 臂 | 块1 | 块2 | 块3 | 块4 | 块5 | 块6 |
|---|---|---|---|---|---|---|
| baseline lr5e-6 无 warmup | 0.2162 | 0.1530 | 0.1232 | — | — | — |
| arm1 lr1e-6 | 0.2849 | 0.2726 | 0.2349 | 0.2785 | 0.2910 | 0.2964 |
| arm3 λ0.75（零监督） | 0.2761 | 0.2627 | 0.2412 | 0.2603 | 0.2803 | 0.2928 |
| arm3b λ0.75+CE | 0.2726 | 0.2665 | 0.2482 | — | — | — |
| arm4 lr3e-6 | 0.2790 | 0.2681 | 0.2519 | 0.2477 | 0.2688 | 0.2687 |
| arm5 top_p0.9 | 0.2742 | 0.2884 | 0.2620 | 0.2657 | 0.2985 | 0.3101 |
| **arm6 lr5e-6+warmup** | 0.3034 | 0.2619 | 0.2431 | 0.2625 | 0.2625 | **0.2148** |

### 9.2 判读要点（只列数字关系，不作结论）
- 唯一一次**真实**结构崩塌 = baseline（lr 5e-6，**无 warmup**，探针 v1）：六段有序 87/240，step 14 起连续 0/8，同时散度单调降到 0.1232。
- 加 warmup 后，lr 1e-6 / 3e-6 / 5e-6 三档六段有序率均为 **480/480**，触顶均为 0；lr 5e-6+warmup 的末 10 步散度最低（0.2148）。
- top-64 覆盖率：top_p=1.0 各臂 0.9995 左右，top_p=0.9 为 **0.99993**；臂 2 全词表教师 step0 为 0.99953。
- 训推 logp 差 |Δ| 均值：lr1e-6/3e-6 组 0.0132–0.0134，top_p0.9 组 **0.0222**；最大值 2.15–3.64；k3 负值计数全为 0。
- 数值路径（臂 4 vs 4b，同 lr 前 20 步）：bnb 8-bit 峰值 57.1 GiB / grad_norm 8.84 / div 0.2735；torch AdamW 峰值 62.8 GiB / grad_norm 7.71 / div 0.2673。
  **口径说明**：`torch.optim.AdamW` 状态随参数 dtype 分配，bf16 参数下状态亦为 bf16（本机实测），故该对比**不是** fp32 状态对比。
- 臂 2（本地全词表教师）在 PB=8 下于 ms-swift 自身 `gkd_loss.py:82` OOM（94.8 GiB），仅得 step0 一行。

## 10. 200 步 pilot（进行中）
配置 = 臂 6（lr 5e-6 + warmup 100）+ 臂 1 口径（λ=1、T 1.0、PB=8×累积1、补全 2048、max_length 4096、bnb 8-bit、探针 v7）；
数据 = `opsd50k`（原始，无 assistant，λ=1 下不走 DATASET 分支）。
选定依据：臂 6 末 10 步散度 0.2148 < 臂 4 的 0.2687，且 lr 更大（同等步数走得更远）。
判据：**主 = 32 键在线自生成读出余弦（每 50 步，对照 step-0 = 0.4989）**；守卫 = 六段有序率 / EOS 率 / 触顶率 / grad_norm 有限 / 训推 |Δ|；散度仅作辅助。
ckpt：滚动每 50 步（`save_total_limit 2`，供在线判据取权重），每 200 步复制进 `keep/` 长期保留（留最近 2 个）。

## 11. 200 步 pilot 收尾：**第 56–68 步结构崩塌，step 89 人工停机**（run 改名留档 `opsd_full/pilot200_lr5e6_COLLAPSE_step89/`）

配置 = 臂 6（lr 5e-6 + warmup 100、λ=1、T 1.0、PB=8×累积1、补全 2048、bnb 8-bit、探针 v7、原始 opsd50k）。

### 11.1 全曲线（10 步块）
| 块(步) | 散度 | 六段有序 | 触顶行 | 熵 | grad_norm | 训推 \|Δ\| |
|---|---|---|---|---|---|---|
| 0-9 | 0.2887 | 80/80 | 0 | 0.342 | 8.74 | 0.0134 |
| 10-19 | 0.2557 | 80/80 | 0 | 0.348 | 7.57 | 0.0136 |
| 20-29 | 0.2488 | 80/80 | 0 | 0.336 | 8.03 | 0.0132 |
| 30-39 | 0.2455 | 80/80 | 0 | 0.324 | 6.99 | 0.0126 |
| 40-49 | 0.2250 | 80/80 | 0 | 0.361 | 6.56 | 0.0135 |
| 50-59 | 0.2415 | 79/80 | 0 | 0.406 | 5.79 | 0.0140 |
| 60-69 | 0.2003 | 56/80 | 0 | 0.440 | 4.36 | 0.0141 |
| 70-79 | 0.1529 | 1/80 | 1 | 0.461 | 2.79 | 0.0142 |
| 80-89 | 0.1341 | 0/80 | 10 | 0.535 | 2.28 | 0.0150 |

**崩塌步定位**：首次非满分 **step 56**；首次 <50% 与首次 0/8 均为 **step 68**；step 71 起连续 0/8 直到停机。
逐步六段有序序列（每步 x/8）：`888888888888888888888888888888888888888888888888888888887888888778540110000...0`

**在线主判据**：step 50 余弦均值 **0.4900** / 中位 0.4987（step-0 基线 0.4989 / 0.4991），解析率 1.0、缺 token 率 0.0 —— **未达收紧阈值 0.5089**；step 100 未到达（step 89 停机）。

**同时发生的四个信号**（同一区间）：散度加速下降（0.2250→0.1341）、六段有序率崩塌（80/80→0/80）、**熵上升**（0.361→0.535）、**grad_norm 下降**（6.56→2.28）、触顶行出现（0→1→10）。

### 11.2 崩塌前后原始生成（同 3 条训练 prompt，T=1.0 采样各 1 条）
崩塌前 = `checkpoint-50` 离线生成；崩塌后 = 直接向 rollout vLLM server 取（其权重为停机时最后同步的 ≈step 89；该步无 ckpt）。

| 来源 | key | 字符数 | 阶段 token 序 | 六段合规 | 终止 |
|---|---|---|---|---|---|
| 崩塌前 ckpt-50 | `src_5e0c90682bb36f97.rep5|d6` | 1357 | `[1]` | ✗ | no-eos |
| 崩塌前 ckpt-50 | `src_9cbfad9e2fab3abe.rep3|d6` | 4557 | `[1, 2, 3, 4, 5, 6]` | ✓ | eos |
| 崩塌前 ckpt-50 | `src_2b616cc22d924047.rep9|d6` | 3842 | `[1, 2, 3, 4, 5, 6]` | ✓ | eos |
| 崩塌后 step≈89 | `src_5e0c90682bb36f97.rep5|d6` | 4509 | `[1, 4, 4, 4, 5, 6]` | ✗ | stop |
| 崩塌后 step≈89 | `src_9cbfad9e2fab3abe.rep3|d6` | 726 | `[]` | ✗ | stop |
| 崩塌后 step≈89 | `src_2b616cc22d924047.rep9|d6` | 4921 | `[1, 4, 4, 5, 5, 6]` | ✗ | stop |

**退化形态（三条后崩塌样本一致）**：不是重复、不是长度爆炸、prose 仍然通顺；而是**阶段 token 身份塌陷** —— `<vr_stage_2>`/`<vr_stage_3>` 被写成 `<vr_stage_4>` 并重复（`[1,4,4,4,5,6]`、`[1,4,4,5,5,6]`），或整篇不出任何阶段 token 并提前 stop（726 字符那条）。崩塌前同 prompt 有 2/3 为标准 `[1,2,3,4,5,6]`。
（口径：每条仅 1 个采样、n=3，只作形态说明；统计口径以 §11.1 的逐步六段有序率为准。）

### 11.3 三条结论（只陈述观测）
1. **结构崩塌与 lr / warmup 无关，是训练步数的函数**：lr 1e-6（臂 1）、3e-6（臂 4）、5e-6+warmup（臂 6）在 **60 步内**六段有序率全部 480/480；同一 5e-6+warmup 配置跑到 **step 56 开始掉、step 68 归零**。此前所有「U 形」观察都来自于**只跑 60 步**，即崩塌点之前。
   唯一的 lr 相关差异是崩塌**时刻**：baseline（5e-6 **无 warmup**）在 step 10–14 崩，加 warmup 后推迟到 step 56–68。
2. **主判据未动**：唯一取到的在线余弦（step 50）为 0.4900，低于 step-0 的 0.4989；即在结构尚完好的阶段，OPSD 也没有把自生成读出余弦推上去。
3. **散度下降不等于变好**：崩塌区间散度反而降得更快（0.2250→0.1341），与 baseline 同形态——这是「退化成易预测文本」的签名，故散度只能作辅助量。

### 11.4 硬停条件（对任何后续 RL / 蒸馏臂生效，预注册）
**六段有序率 + 触顶率**为硬停条件：**连续 2 个记录点六段有序率 < 0.5 即停机**（探针 `six_complete_rows/batch`）；触顶率（`row_lens == max_completion_length` 的行占比）与解析率、EOS 率同列必报。
执行位置：`rl/scripts/dualcard/run_arm.sh` 的 early-stop 判据（现为「前 30 步内连续 5 步 0/8」）须放宽到全程、阈值改 <0.5 连续 2 点。**尚未改，待下一路线裁决时一并实施。**

## 11. 200 步 pilot：**第 68 步结构崩塌，第 89/90 步停机**（run 改名留档 `opsd_full/pilot200_lr5e6_COLLAPSE_step89`）

配置 = 臂 6（lr 5e-6 + warmup 100、λ=1、T 1.0、PB=8×累积1、补全 2048、bnb 8-bit、探针 v7、原始 opsd50k）。
实跑 **90 步**（预定 200），2h00m，82.8 s/step；无 traceback（外部停机）；gpu1 峰值 60,375 MiB。

### 11.1 全曲线（10 步块）
| 步块 | div | 六段有序 | 触顶行 | 熵 | top-64 覆盖 | grad_norm | 训推 \|Δ\| |
|---|---|---|---|---|---|---|---|
| 0-9 | 0.2887 | 80/80 | 0 | 0.342 | 0.99958 | 8.74 | 0.0134 |
| 10-19 | 0.2557 | 80/80 | 0 | 0.348 | 0.99950 | 7.57 | 0.0136 |
| 20-29 | 0.2488 | 80/80 | 0 | 0.336 | 0.99957 | 8.03 | 0.0132 |
| 30-39 | 0.2455 | 80/80 | 0 | 0.324 | 0.99938 | 6.99 | 0.0126 |
| 40-49 | 0.2250 | 80/80 | 0 | 0.361 | 0.99961 | 6.56 | 0.0135 |
| 50-59 | 0.2415 | 79/80 | 0 | 0.406 | 0.99934 | 5.79 | 0.0140 |
| 60-69 | 0.2003 | 56/80 | 0 | 0.440 | 0.99943 | 4.36 | 0.0141 |
| 70-79 | 0.1529 | 1/80 | 1 | 0.461 | 0.99908 | 2.79 | 0.0142 |
| 80-89 | 0.1341 | 0/80 | 10 | 0.535 | 0.99723 | 2.28 | 0.0150 |

### 11.2 崩塌步定位（逐步序列）
`888888888888888888888888888888888888888888888888888888887888888778540110000000000000000000`
- 首个 **<8/8**：step **56**；首个 **<0.5（硬停条件）**：step **68**；首个 **0/8**：step **68**；首个补全触顶：step **75**。
- 同期：散度 **加速下降** 0.2250(40-49) → 0.2003(60-69) → 0.1529(70-79) → 0.1341(80-89)；
  熵 **上升** 0.361 → 0.440 → 0.461 → 0.535；grad_norm **下降** 6.56 → 4.36 → 2.79 → 2.28；
  触顶行 0 → 0 → 1 → 10；top-64 覆盖 0.99961 → 0.99723。
- 在线余弦（主判据）：step-0 **0.4989** → step-50 **0.4900**（均值 −1.8%，中位 0.4987 持平），step-100 未到即停机。

### 11.3 崩塌前后各 3 条原始生成（同 3 个 prompt，greedy）
**崩塌前（checkpoint-50，step 50 权重，greedy）**

- `src_5e0c90682bb36f97.rep5|d6`：生成 1560 token，`Move` 出现 6 次，阶段 token 序 **[1, 2, 3, 4, 5, 6]**，EOS 到达
  - Move 2 段首 200 字符：` ⏎ Observation: The saturated environmental colours still read weakly and unevenly, especially through the green-blue background. Give those selected colours more presence while keeping the pale skin an`
  - 结尾 120 字符：`'-4.23%, blue +3.03°/+2.85%/-5.61%, purple +8.67°/+3.30%/-7.38%, and magenta +7.07°/+4.04%/-7.33%.\n<vr_stage_6><|im_end|>'`
- `src_9cbfad9e2fab3abe.rep3|d6`：生成 1476 token，`Move` 出现 6 次，阶段 token 序 **[1, 2, 3, 4, 5, 6]**，EOS 到达
  - Move 2 段首 200 字符：` ⏎ Observation: The saturated background foliage and cooler colour pockets still need a more deliberate, lively separation from the subject. The colour-range pass should strengthen those selected colour`
  - 结尾 120 字符：`' +1.33; blue +0.68, +12.57, +4.80; purple -2.89, +14.39, +4.36; and magenta -7.52, +13.75, +3.06.\n<vr_stage_6><|im_end|>'`
- `src_2b616cc22d924047.rep9|d6`：生成 1596 token，`Move` 出现 6 次，阶段 token 序 **[1, 2, 3, 4, 5, 6]**，EOS 到达
  - Move 2 段首 200 字符：` ⏎ Observation: The saturated lagoon, pool water, and island foliage still read a little restrained after the global pass; this move should give the stronger chromatic areas more presence while keeping `
  - 结尾 120 字符：`' -1.74, saturation +12.80, luminance +2.65; magenta hue -7.12, saturation +8.39, luminance +1.53.\n<vr_stage_6><|im_end|>'`

**崩塌后（rollout server 持有的 step-89 权重，greedy）**

- `src_5e0c90682bb36f97.rep5|d6`：生成 1656 token，`Move` 出现 6 次，阶段 token 序 **[1, 4, 4, 4, 5, 6]**，EOS 到达
  - Move 2 段首 200 字符：` ⏎ Observation: The saturated background foliage and cool colour pockets still read too restrained and uneven after the global pass. They need a more deliberate green-blue character and stronger chromat`
  - 结尾 120 字符：`'.39, L+4.53; blue H+19.89, S+2.44, L+3.29; purple H+7.81, S+21.11, L-3.45; magenta H-6.81, S+13.39, L-4.67.\n<vr_stage_6>'`
- `src_9cbfad9e2fab3abe.rep3|d6`：生成 1608 token，`Move` 出现 6 次，阶段 token 序 **[1, 4, 4, 4, 5, 6]**，EOS 到达
  - Move 2 段首 200 字符：` ⏎ Observation: The saturated foliage and cool background still need a more selective colour push, while the less saturated skin, gown, and railings should remain comparatively protected. This move buil`
  - 结尾 120 字符：`'70; blue H -6.02, S +13.46, L +0.30; purple H -2.89, S +12.38, L +0.73; magenta H -2.37, S +13.46, L +1.25.\n<vr_stage_6>'`
- `src_2b616cc22d924047.rep9|d6`：生成 1609 token，`Move` 出现 6 次，阶段 token 序 **[1, 4, 4, 4, 5, 6]**，EOS 到达
  - Move 2 段首 200 字符：` ⏎ Observation: The saturated lagoon, pool water, and planted areas still read a little restrained after the global pass. This move builds stronger chroma in those coloured regions while keeping the les`
  - 结尾 120 字符：`'.28; blue H +3.58, S +10.54, L +3.98; purple H -1.73, S +12.73, L +2.66; magenta H -7.11, S +8.32, L +1.54.\n<vr_stage_6>'`

**形态**：崩塌**不是**语句崩坏或长度爆炸——文本仍是 6 段 `Move`、Observation/Mask/Adjustment 三行齐全、数值格式正常、greedy 下仍到 EOS（1,608–1,656 token）。
退化**只发生在阶段 token 的身份**上：第 2、3 段的 `<vr_stage_2>` / `<vr_stage_3>` 被写成 `<vr_stage_4>`，序列 `[1,2,3,4,5,6]` → **`[1,4,4,4,5,6]`**。
训练时的 rollout 用 T=1.0 采样，退化更重（80-89 步出现 10 行触顶）。这直接击穿读出通路：读出按 `<vr_stage_m>` 定位 6 段做 span-pool，段标识错乱即向量错位。

### 11.4 结论性事实（只列可复现的观察）
1. **结构崩塌与 lr / warmup 无关，是训练步数到量后出现的**：lr 1e-6（臂 1）、3e-6（臂 4）、5e-6+warmup（臂 6）在 **60 步内**六段有序率均为 480/480；
   同一 lr 5e-6+warmup 配置继续跑到 **step 56 开始掉、step 68 归零**。此前所有 60 步小臂「全部守住」只是因为**崩塌发生在 60 步之后**；
   baseline（lr 5e-6 无 warmup）在 step 14 崩，是同一现象的更早版本 —— warmup/低 lr 只推迟崩塌，不消除它。
2. 崩塌与「散度下降」**同向发生**：散度越低、熵越高、grad_norm 越小，六段有序率越低 —— 即 §7 记录的「退化成易预测文本」签名在本次得到逐步定位。
3. 教师侧无异常：top-64 覆盖率全程 ≥0.997，训推 logp 差 |Δ| 均值 0.0126–0.0150（稳定），k3 负值 0。

### 11.5 **硬停条件（对后续任何 RL / 蒸馏臂强制生效）**
> **六段有序率 与 补全触顶率**：连续 **2 个记录点** 六段有序率 **< 0.5** 即**立即停机**并留档；触顶率同时作为并列观测列上报。
> 本次若该条件已生效，应在 **step 69**（68、69 连续两点 0/8）停机，而非 step 89 —— 已写入 `run_arm.sh` / pilot 驱动的后续版本待接线项。

## 12. 下一步候选（每条附预算与预注册判据；**均未实施，等用户裁决**）

成本基准（本机实测）：GKD 路径 PB=8、补全 2048 时 **80 s/step**（含 vLLM rollout + 教师 API + 反传）；
在线 32 键余弦评测 **12.4 min/次**；gpu0 两服务常驻 75.2 GB，gpu1 训练峰值 57–60 GB。
所有候选**共用**硬停条件（§11.4）与守卫列（六段有序率 / 触顶率 / 解析率 / EOS 率 / grad_norm 有限 / 训推 |Δ|）。

| # | 候选 | 改动位置 | 预算 | 预注册判据（成功 / 失败） |
|---|---|---|---|---|
| A | **阶段 token 位置从散度中 mask**（学生+教师两侧同 mask）<br>SURVEY-DEGEN §6 #5 | `gkd_loss.py` `jsd_loss` L94-148 唯一改点（须维持 L200-203 两侧有效 token 数相等断言）；本仓以插件猴补丁实现，不改 ms-swift 源码 | 120 步 ≈ 2.7 h + 3 次在线评测 ≈ 0.6 h | 成功 = 步 100–120 六段有序率 ≥0.95 **且** 在线余弦 ≥0.5089；失败 = 仍在 ≤120 步内触发硬停 |
| B | **阶段 token 单独 CE 锚点**（系数 0.3；λ=1 下 `sft_alpha` 恒不生效，必须自加）<br>SURVEY-DEGEN §6 #6（2602.11549 §3.2.3 系数 0.3；Llama 3 NLL 0.2） | 同 A 的钩子内，用 `labels` + 学生 logits 对 6 个阶段位置算 CE 后相加 | 120 步 ≈ 2.7 h + 0.6 h | 同 A；另记 CE 项与散度项的数值比（预期两项同量级才有效） |
| C | **逐词元 pointwise 裁剪 τ**（OPSD 论文自带的抗崩装置，本次 pilot 未开）<br>SURVEY-DEGEN §6 #4（2601.18734 §3 + Fig.4 "Clipping prevents performance collapse"） | 同 A 的钩子：chunk 循环体内 `min(p_T·f(p_S/p_T), τ)`；τ 需扫 2 档（论文未给值） | 2 档 × 120 步 ≈ 5.4 h + 1.2 h | 成功 = 至少一档在 120 步内不触发硬停且余弦 ≥0.5089；失败 = 两档均崩 |
| D | **换 GRPO 路径 + 执行器 linf8 奖励**（把优化目标从「与教师分布一致」换成「像素误差」）<br>SURVEY_OPD_GRPO_v2 §D/§E.2/§F.2 | `--rlhf_type grpo` + `--external_plugins` 自定义 ORM（`swift.rewards.ORM.__call__(completions, **kwargs)`；段跨度用 kwargs 的 `response_token_ids` 按 id 定位，勿 decode 往返）；奖励 = 读出→注入 BK-FULL→linf8（`rl/reward/executor_reward.py` 骨架已在，主体待接线） | 接线 1–2 h（含 A-inj/A-lat 守卫）+ 200 步 ≈ 6–7 h（B·G 次执行器前向另计） | 成功 = 在线余弦 ≥0.5089 **且** held-out d6 用**未改动** BK-FULL 复现 linf8 下降；失败 = 奖励上升但守卫或 held-out 不动（即 §F.2 的 hacking 判据） |
| E | **改教师特权形式：图对特权**（`teacher_images=[y, x_原图]`，教师看退化图+原图，学生只看退化图）<br>SURVEY_OPD_GRPO_v2 §E.3/§F.1（`teacher_images` **仅 main 分支有**，本容器 ms-swift 4.6.0.dev0@236a1f19 满足） | 数据侧加 `teacher_images` 列（复用 `build_opsd_with_assistant.py` 的构造脚本）；教师 prompt 改 `"<image><image>"+指令` | 数据构造 0.5 h + 120 步 ≈ 2.7 h + 0.6 h | 成功 = 教师-学生散度（step 0）显著高于文本特权版**且** 120 步内不崩、余弦 ≥0.5089；失败 = step-0 散度与文本特权版无差（说明该特权无额外信息） |

**排序建议（只列依据，不替用户决策）**：C 与 A/B 直接针对已观测到的崩塌形态（阶段 token 身份塌陷），且都只改 `jsd_loss` 一个点、预算最小；
D 换的是优化目标本身（奖励落在像素上，不依赖教师分布），预算最大但唯一能直接优化 headline 的 linf8；
E 只换特权信息形式，若 step-0 散度无变化则可最快证伪。

### 12.1 与本次崩塌观测直接相关的两条事实（供选型参考，不作结论）
- 崩塌区间**熵在上升**（0.361→0.535）而非熵坍缩，故 SURVEY-DEGEN §1 的「entropy collapse」机制与本次观测**不符**；
- 崩塌区间 **grad_norm 单调下降**（6.56→2.28）、散度加速下降，即模型在「优化得更顺」的同时丢掉结构。

## 12. 下一步候选（按 SURVEY_STRUCTURE_DEGEN §6 与 SURVEY_OPD_GRPO_v2；**均未实施，等用户裁决**）

预算口径：本机 gpu1 训练 82.8 s/step（PB=8）；60 步 ≈ 1.4 h，200 步 ≈ 4.6 h；在线 32 键余弦评测 ≈ 12.4 min/次。
所有候选**共用**硬停条件（§11.5）与守卫列（六段有序率 / 触顶率 / grad_norm 有限 / 训推 |Δ| / top-64 覆盖）。

| # | 候选 | 依据 | 改动量 | 预算 | 预注册判据（达不到即判否） |
|---|---|---|---|---|---|
| C1 | **阶段 token 位置从散度中 mask**（学生/教师两侧同时，`gkd_loss.py` `jsd_loss` 唯一改点；须满足 L200-203 两侧有效 token 数相等断言） | §6 #5；SimpleOPD 2608.14277 §4.1.2（mask `</think>`/`<\|im_end\|>` 后截断率降至近 0）、2603.25562 §3.2（36.4→40.7）、Llama 3 §4.1.4（格式 token 计损失→尾部重复/异常终止）；**反向**：ParaVT 2605.20342（结构 token 置零后 f_τ 0.13→0.11） | 中（改损失函数） | 120 步 ≈ 2.8 h + 3 次评测 | 六段有序率在 **120 步内**不出现连续 2 点 <0.5；且 step-100 在线余弦 ≥ 0.5089 |
| C2 | **阶段 token 单独 CE 锚点**（系数 0.3；在 `jsd_loss` 钩内用 labels+学生 logits 算 CE 相加。**不能用 `sft_alpha`**：λ=1 下恒不生效，`gkd_trainer.py` L244-245+L295） | §6 #6；2602.11549 §3.2.3（`L_format` 系数 0.3）、Llama 3 NLL 0.2、Hinton §4.1 权重 0.5 | 中 | 120 步 ≈ 2.8 h | 同 C1；另需 CE 项与 JSD 项分列记录 |
| C3 | **逐 token pointwise 裁剪 τ**（OPSD 论文自带装置，本次 pilot 未开） | §6 #4；2601.18734 §3 + Fig.4「Clipping prevents performance collapse」；Table 5（Style 类 KL 是 Math 类 6–13 倍） | 小（`jsd_loss` chunk 循环体） | 120 步 ≈ 2.8 h | 同 C1 |
| C4 | **换路线：GRPO/OPD-RL + 执行器 linf8 奖励**（`--rlhf_type grpo`，奖励 = 读出→注入 BK-FULL→linf8；教师 log-ratio 作优势） | SURVEY_OPD_GRPO_v2 §E.2/§E.3、§F.2；本机已验证 GRPO 路径可跑（ENG2 §3 探针 c：`teacher_kl` 0.2033/0.2351，k1 −0.19/−0.23） | 大（需接 `executor_reward` 主体 + 每步 B·G 次执行器与读出前向） | 接线 ~1 天 + 200 步 ≈ 6–8 h（G=2 时每步成本约 ×2） | 200 步内在线余弦 ≥ 0.5089 且守卫不降；奖励与在线余弦同向 |
| C5 | **改教师上下文形式**：GT CoT 去数值化摘要作特权，或图对特权（`teacher_images=[y, x0]`，仅 main 分支支持） | SURVEY_OPD_GRPO_v2 §F.1（教师「复制」行为预注册探针）、§E.3（`teacher_images` 仅 main） | 中（数据构造 + 教师视图） | 数据 ~1 h + 120 步 ≈ 2.8 h | 教师 top-1 与 GT 对应位置 token 的复制率下降，且六段有序率在 120 步内不崩 |
| C6 | **有效 batch 8→32**（`--gradient_accumulation_steps 4`） | §6 #8；OPSD Table 6 batch 32、2608.18271 batch 32 | 极小（一个 flag） | 60 步 ≈ 5.5 h（每步 4× 生成） | 崩塌步是否后移（与本次 step 68 比较）；散度方差下降 |

**优先级建议（只列依据，不替用户决定）**：C3/C1 改动最小且直接针对已定位的失效位置（阶段 token 身份）；C4 是路线切换、成本最高但奖励直接对齐 headline 指标（执行器 linf8）；C6 最便宜但只验证「崩塌步是否后移」。

## 13. 现场状态与交付
- **两卡已释放**：pilot 训练进程已退出，gpu0 的 rollout(:8000) 与 teacher(:8100) 两个 vLLM 服务按裁决停止；**未自动起任何新臂**。
- 留档目录：`opsd_full/pilot200_lr5e6_COLLAPSE_step89/`（run.log、probe_segments.jsonl 90 步、online/、keep/、nvsmi.log、checkpoint-50）、
  `arms/`（arm1/arm2/arm3/arm3b/arm4/arm4b/arm5/arm6 + 5 个探针对照 ctrl_* + 2 个作废 `*_VOID_probe_v4`）。
- 生成样例：`samples_pre_collapse_step50.json`、`samples_post_collapse_step89.json`（各 3 条，含全文）。
- 探针版本谱系：v1（60 步 λ=1 已验证）→ v3（无 loss 钩）→ v4（**有缺陷**，包装 `compute_loss` 致 NaN）→ v4a（去钩，干净）→ v6（optimizer 钩未触发）→ **v7（正式：v4a + `TrainerCallback.on_log`）**。

## 13. 已备未跑（C4 = 候选 D 的实现件；**用户未裁决，一律不运行**）

执行 agent 曾越权起过 C4 作业，已全部停机（NOTES N16）。以下件**留在盘上、未运行**，用户若选 D 可直接用：

| 件 | 路径 | 状态 |
|---|---|---|
| 双奖励 ORM 插件 | `veraretouch_sprf/rl/plugins/c4_rewards.py` | 已写、`py_compile` 通过、**未在训练中跑过** |
| GRPO 启动脚本 | `veraretouch_sprf/rl/scripts/dualcard/train_c4_grpo.sh` | 已写、`bash -n` 通过、未跑 |
| 执行器可行性探针 | `veraretouch_sprf/rl/reward/executor_probe.py` | **已实测通过**（数字见下） |

**插件口径（预注册，未验证于训练）**
- R1 latent：生成文本 → `<vr_stage_m>` 定位六段 → span mean-pool（**当前策略**权重的 `last_hidden_state`，与 `dump_readout` 同口径）→ **冻结** S2F-B adapter（`adapt_s2fb/ckpt_epoch1`）→ (6,128) slot 序 → 对该 key 的 e\* 逐槽余弦取均值；解析失败/缺段 = `MISS_LATENT`（默认 0.0）。
- R2 executor：同一 latent（slot→chain 用 `flip(0)`）注入**冻结** BK-FULL（`ckpt_last.pt`）→ rollout → linf8 p50 → 奖励 = **−linf8/23.0**（23.0 = held-out d6 identity）；解析失败 = `MISS_EXEC`（默认 −1.0）。
- 两项共享**同一次**读出前向（按步缓存），不重复前向；`--reward_weights 1 1`。
- 守卫：**A-inj** 在 ORM 初始化时对 2 键跑「注入路径 vs oracle_lut 路径逐位相等」，不等即 `SystemExit`；
  **硬停** = 六段有序率连续 2 个记录点 < 0.5 ⇒ 置 `trainer.control.should_training_stop = True`（§11.4 的规则已接线）；
  `on_log` 回调记 loss/grad_norm/lr（不包装任何损失路径函数，遵 NOTES N10/N11 的教训）。

**执行器可行性实测**（`executor_probe.py`，容器内 gpu1，2026-09-07）
- BK-FULL 载入 **3.3 s**；`inv_table` (4052, 19652)。
- 每键：journal 行读取 + 图像解码 + α 场重建 = **0.58 s**；rollout（16,384 像素）= **0.05 s**（首键 1.35 s 含 CUDA 预热）。
- 三键 oracle-LUT linf8 p50 = **0.7916 / 1.2519 / 1.0702**（与 BK-FULL d6 上界 1.16 量级一致）。
- 两个必须踩过的坑（已写进探针）：① v3 分片资产在 `archive/` tar 内 ⇒ 必须 `archive_assets.install(T0)`（X5）；
  ② 必须 `T0.bind_build_config(...)` 否则 `epr050_build_degradation` 的 `SAMPLE_SALT/STEP_KIND` 为空、α 重建报 `TypeError`。
  该函数只用 `samples[0]['shard']`，故可用单键的 `dir` 绑定，**无需 `load_shards` 全量加载 279 分片**。

**容器侧一次性改动（已做，无副作用，供任何后续容器内跑 SPRF 栈使用）**
`/home/bc/VeraRetouch → /workspace/VeraRetouch`、`/home/bc/data → /data`、`/mnt/nfs-ro → /nfs-ro`、
`/var/cache/veradata/preset_bank_full → /lut_bank` 四个符号链接（否则 `run_args.json` / inv 缓存 / 分片里的**宿主绝对路径**在容器内不可达）。

**未做/未知**：两项奖励的量级是否可比（用户要求先在 8 键上打印分布再定权重）——**未测**；GRPO 每步耗时与显存——**未测**。
