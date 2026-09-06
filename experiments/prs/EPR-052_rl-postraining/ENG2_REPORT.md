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

## 5. 正式数据
`/home/bc/data/runs/epr052_rl/data/opsd50k/opsd_train.jsonl`：S2 train 73,854 中按 `sha1("epr052-opsd-train-v1:"+key)` 升序取前 **50,000**；held-out 1,558 id / eval-only 1,464 键与 train 交集为空（pool = 73,854）；键表 sha256 **e3e1b24f02bbe53e25c5deed504330bd509f75d416a1ca290314e0eafeb5e5a6**；指令档 long/medium/short = 35,233/9,801/4,966；GT 平均 4,442 字符；y 图 spec-5 落盘（768×512 20,443、512×768 13,994、…）。

## 6. 双卡 OPSD pilot（正式配置 `rl/configs/opsd_full_dualcard.yaml`；脚本 `rl/scripts/dualcard/`）
gpu0：`swift rollout`（util 0.28，max_model_len 4096，port 8000）+ 教师 `swift deploy`（util 0.45，port 8100，固定 S1F）；gpu1：`swift rlhf --rlhf_type gkd --tuner_type full --torch_dtype bfloat16 --optim adamw_bnb_8bit --adam_beta2 0.999 --lr 5e-6 --beta 0 --lmbda 1 --temperature 1.0 --max_completion_length 2048 --max_length 4096 --per_device 1 --grad_accum 8`（steps_per_generation = 8 → 每优化步一次 8 条 vLLM 批量 rollout）、`--use_vllm true --vllm_mode server`、`--teacher_model_server :8100 --gkd_logits_topk 64`、每 25 步存 ckpt、探针插件逐步记录分段散度/六段完整。
在线判据（`online_eval.sh`，宿主 q3vl_sft 环境，gpu0）：32 键 val（`s1f_val32_keys.json`）greedy 自生成 → S2F-B 读出（LoRA+adapter 冻结，与 0.39 基线同代码路径）→ 对 e* 余弦；同时报 stage_token_missing_rate、cot_parse_success_rate。
（pilot 数字见 §7，运行中逐步补。）

## 7. pilot 逐 25 步记录
（待填）
