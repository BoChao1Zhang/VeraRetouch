# EPR-052 · ENG 报告：SPRF 主线工程包 / rl 骨架 / ms-swift compose / 磁盘清理（2026-09-06）

分支 `eng/sprf-mainline`（自 `lens-exp` d6d747a 新建，未 push）。代码提交 **fc41641**（84 文件，+18,264 行）；本报告为第二次提交。
硬约束执行：EPR-051 `stage0/sprf/` 原件**只复制未改动**（在跑作业 SPRF_ADAPT_S2FB pid 567815、SPRF_VLMADAPT_PASSKF_GEN pid 1825973 引用的文件未触碰）；
`trash/` 未读取；无 sudo 动作；未拉镜像、未起容器。

## 1. 交付 1：`veraretouch_sprf/`（仓库根，Python 包）

### 1.1 文件清单（39 个复制件 + 1 节选 + 新增件；逐文件原/新 sha256 见 `veraretouch_sprf/PROVENANCE.md`）
| 子包 | 复制件（← legacy 原名） | 新增件 |
|---|---|---|
| `data/` | stage_targets, archive_assets, train_stage0(← stage0/), build_targets_bkfull, cot_text(**逐字节相同**), q3vl_text, q3vl_data, stage_assets2, stage_assets_heldout, freeze_cot_snapshot2 | — |
| `models/` | stage_flow, stage_flow_bk, stage_flow_bk3, stage_flow_bk4, stage_backend_g4d, edit_cond, bk_load, align_predictor, align_predictor_time; `vlm/q3vl_common` | `vlm/adapter.py`（train_q3vl_adapt3.py 节选：Adapter / readout_z_and_hidden / readout_z；与历史 train_q3vl_adapt.py 的 Adapter diff 相同） |
| `solver/` | stage_solver, stage_solver_bk | — |
| `train/` | train_sprf, train_sprf_xl, train_sprf_bk_core4, train_sprf_bk5, train_align_time, train_vlm_sft(← train_q3vl_sft3), train_vlm_adapt(← train_q3vl_adapt3) | `train_decoder.py`（物化配置 + 按 [bk] 节转调 xl / bk5） |
| `eval/` | eval_decoder(← batch_eval_bk), dump_readout, eval_vlmadapt, guards; `probes/` probe_full_ckpt, parity_report, quick100_eval | `eval_vlm_e2e.py`（dump / eval 子命令转调） |
| `configs/` | decoder/{clut_full ← arm_clut_full, bkfull_adagn_ff_affhead, smoke_clut_full}, vlm/{sft_full ← q3vl_sft_s1f_full, adapt_s2fb ← q3vl_adapt_s2fb} | `__init__.py`（`[paths]` 节 + `${paths.*}` 物化；env `VR_PATH_<KEY>` 覆盖） |
| `scripts/` | heldout_split(← epr051_heldout_split), merge_gencache(← epr051_merge_gencache) | submit_decoder.sh, run_vlm_sft.sh, run_vlm_adapt.sh, eval_decoder.sh, dump_half.sh, heldout_final.sh, submit_heldout_headline.sh（后三者为 vlmsft 同名脚本改入口路径版）, smoke_a1_cpu.py, smoke_cot_roundtrip.py, check_configs.py, sync_from_legacy.py, derive_configs_from_legacy.py |
| 根 | — | `__init__.py`, `_paths.py`, `README.md`, `PROVENANCE.md` |

入口对应关系（按 run 目录 provenance 核实，非按 pack README 的说法）：C-LUT-FULL = `train_sprf_xl.py`→`train_sprf.main`（`xl_provenance.json`）；
BK-FULL = `train_sprf_bk5.py`→`train_sprf_bk_core4.main`（`bk_provenance.json` recorded_by=train_sprf_bk5.py，K3 基线 arm_clut_full.toml）。故复制 bk5 而非 bk4。

### 1.2 改动类别（全部机械，不改数学与默认数值）
1. 头注释一行（cot_text.py 例外）；2. `sys.path` 注入块 → `_paths.ensure_sys_path()`，`REPO/STAGE0/SPRF` 常量改由 `_paths` 提供；
3. 同目录裸导入 → 包内绝对导入（含 `import cot_text as C, q3vl_common as Q, ...` 单行拆行）；4. provenance 记录用的 `HERE / "x.py"` → `_P.src("x.py")`
（包内文件优先，未扶正历史文件回落 legacy）；bk5 的 K3 基线目录改为与 `--config` 同目录（train_decoder 物化目录内含 arm_clut_full.toml / smoke_clut_full.toml）；
bk5 `PINS` 对包内 8 个文件重钉（原钉值表在 PROVENANCE.md），3 个未扶正文件（lossabl_core2 / bk_core / bk_core3）钉值不变；
5. `q3vl_common.MODEL_DIR`、`guards.EVAL_ONLY_DIR` 加环境变量覆盖（`VR_QWEN3VL_DIR` / `VR_EVAL_ONLY_DIR`，默认值不变）。
未扶正（留在 legacy，`_P.src` 回落）：bk/bk2/bk3/bk4/bk6/bk7 与 core/core2/core3/core6/core7、lossabl*、train_align.py、passk_*、bkfull_*ainj*、全部 run_*.sh；
`eval_decoder.py` / `bk_load.py` 对这些历史 core/模型模块的懒导入行原样保留（选到这些臂会 ImportError；四个主结果不经过它们）。

### 1.3 自检结果
| 项 | 结果 |
|---|---|
| `python -m py_compile` 全部 .py | 通过 |
| 模块 import（databuild 环境，解码器线 27 模块） | 27/27 OK |
| 模块 import（q3vl_sft 环境，VLM 线 13 模块） | 12/13 OK；`eval/probes/parity_report.py` 为模块级脚本，import 即执行并读 `/home/bc/data/runs/epr051_vlmsft/rehearse8` 输出 PARITY_STATS（n=8, latent_cos_mean 0.6428，与 REPORT_STATUS §3.5 一致）——运行成功，非导入失败 |
| rl 骨架 import + prompt 构造 | OK（student prompt 734 字符，teacher prompt 5,381 字符；latent cosine_reward 可调用） |
| `scripts/check_configs.py`：5 个配置物化后与 EPR-051 原配置逐字节相同 | 5/5 OK（decoder 三个各 289/289/14 个占位，vlm 4/7 个） |
| env 覆盖 `VR_PATH_RUNS_SPRF=/tmp/x` | out_dir / inv_cache_dir 均改写 |
| **A1 零初始化恒等 CPU 冒烟（10 像素，随机 cond/α/s，真实逆表随机行号）** | clut_full：`torch.equal(restore(nfe=1), y)=True`，abs_max=0.0，params 15,894,406；bkfull_adagn_ff_affhead：True，abs_max=0.0，params 15,957,906 |
| **cot_text 一条记录序列化/反序列化往返**（key ppr10k_0001_a.rep1\|d6，4,585 字符） | roundtrip_ok=True，n_segments=6，field_exact=True，segments 重组==target_text；`template_sha256` 包内 == legacy == `5fc89f0a973b85e3…`（与 HANDOFF 登记一致） |
| bk5 K4 钉死表经 `_P.src` 复核 | 11/11 一致（8 包内重钉 + 3 legacy 回落） |
| `docker compose config` | 渲染通过（gpus count=-1 即 all，ipc host，shm 64g） |

## 2. 交付 2：`veraretouch_sprf/rl/`
`README.md`（OPD/OPSD 与 GRPO 两条路线接线说明，全部标「待调研确认」）；`reward/latent_reward.py`（读出向量对 e* 的逐阶段余弦，可用；缺阶段惩罚参数化）；
`reward/executor_reward.py`（ExecutorContext / load_executor_context / readout_latents(转调 dump_readout.readout_from_generated) / inject_and_rollout / linf8 / executor_reward，
签名与数据流已定，主体 stub `NotImplementedError`，导入通过）；`prompts/build.py`（student_instruction / student_prompt / teacher_prompt = 学生 prompt + GT CoT 前缀上下文，格式占位）；
`configs/opd.yaml`、`configs/grpo.yaml`（模型路径、奖励入口、G、温度、KL 类型与系数、max_new_tokens 2048 等字段占位，标待调研确认）。

## 3. 交付 3：`docker/ms-swift/`
本机核实：Docker 29.1.2、Compose v5.0.0；`bc` 在 docker 组；`docker info` Runtimes 含 `nvidia`（nvidia-container-toolkit 已装）；Docker Root Dir `/home/docker`（/home 分区，可用 534 GB，**不在 / 上，无需迁 data-root**）；
GPU H100×2，driver 570.195.03 / CUDA 12.8。
镜像 tag 来源：ms-swift 仓库 `docs/source_en/GetStarted/SWIFT-installation.md`「Docker」节（打开核实）。最新 swift4.4.3 系列为 cuda13.0.3（需 driver ≥ 580，本机不满足），
`.env.example` 默认 `modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.10.0-vllm0.17.1-modelscope1.34.0-swift4.0.3`，备选 tag 注释在 .env.example。
文件：`compose.yaml`（gpus all、ipc host、shm 64g、nofile 65536、command sleep infinity、HF_HOME=/data/hf_home、MODELSCOPE_CACHE=/data/modelscope_cache、挂载 VeraRetouch→/workspace/VeraRetouch、data→/data、models→/models、nfs-ro→/nfs-ro:ro、LUT bank→/lut_bank:ro、`VR_PATH_*` 指到容器路径）、`.env.example`、`README.md`（拉取/启动/进入/验证命令，预计镜像 40–50 GB）。
**docker 可用**（免 sudo、nvidia runtime 就绪）；镜像未拉、容器未起，等裁决。

## 4. 交付 4：磁盘
### 4.1 前后数字（df，1 GB = 1e9 B）
| 分区 | 前 avail | 后 avail | 回收 |
|---|---|---|---|
| `/`（240G，docker root 不在此） | 84.86 GB | 92.65 GB | **+7.80 GB** |
| `/home`（3.8T；docker root `/home/docker` 在此） | 513.48 GB | 534.41 GB | **+20.93 GB** |
合计 **28.7 GB**。

### 4.2 已执行（均在硬约束允许类别内）
| 项 | 判断依据 | 数量 |
|---|---|---|
| `uv cache clean --force` | pip/uv 缓存可再生；两只 `uvx grok-search` MCP 进程持锁但工具环境无 symlink 指向缓存（硬链接），安全 | 649,870 文件，表观 70.6 GiB（实际释放约 18 GB，其余为硬链接共享） |
| `pip cache purge` | 同上 | 736 文件 632.0 MB |
| `conda clean --all -y` | conda pkgs 缓存可再生 | 330 tarball 507.2 MB + 38 pkgs 23.9 MB |
| `/tmp` 本人所有、mtime > 7 天的顶层条目 | 卡内允许项；lsof 无占用 | 404 条目 8.00 GB（pip-target/pip-unpack 为主，2026-08-27） |
| 仓库内 `__pycache__` | 允许项 | 164 MB |
| docker 停止容器（写层 ≤ 16 MB 的 7 个：sam3_subjects、p1-a1-recovery-v2e/v2f/v2g、docker-codexmanager-web-1/service-1、monetgpt-ms-swift-joint-dapo-v3-500-ng16） | 允许项；写层无实质状态 | 13→6 容器 |
| `docker image prune -f`（悬空镜像） | 允许项 | 0 B（无悬空） |

### 4.3 `/` 分区最大目录（`du -xsh`，排除 /home /mnt /proc）
/var 105G（/var/cache/veradata 87G：models 31G、global.sqlite3 24G、dcube 20G、annot_review 7.4G、preset_bank_full 2.8G(解码器配置引用，勿动)、rd 1.4G；/var/log 16G：syslog.1 7.6G、journal 4.1G、syslog.2.gz 2.7G、syslog.3.gz 1.4G），
/usr 21G，/tmp 11G→约 3G（剩 /tmp/claude-1001 2.2G 近期 + 其他用户 zhengh 约 0.5G），/swap.img 8.1G，/opt 3.0G，/boot 541M，/PSCC-Net 61M。

### 4.4 需用户裁决（只列不删）
| 路径 / 项 | 大小 | 最后修改 | 判断 |
|---|---|---|---|
| docker 未被任何容器引用的镜像：`verlai/verl:vllm011.latest` 48.9G（10 月前）、`monetgpt-verl:latest` 49G（3 月前）、`veraretouch-unified:dev` 35.2G（2 月前）；被停止容器引用：`monetgpt-ms-swift:latest` 45.9G、`vllm/vllm-openai:nightly` 32.8G；小件 headroom 787M、codexmanager 289M、siyuan 236M、postgres:16 642M | docker 报可回收 313.4 GB | — | 旧项目镜像；`docker rmi <名>` 或 `docker image prune -a` |
| docker build cache | 223.2 GB（可回收 160.6 GB） | — | 可再生；`docker builder prune`（或 `--filter until=720h`） |
| 停止容器 `monetgpt-ms-swift`(写层 933M)、`sar-vllm-qwen-1`(353M)、`reason_g1_util060_bak`/`070_bak`(133/134M，名含 bak)、`hungry_ishizaka`(siyuan 160M，挂 volume workspace_dir_host) | 1.7 GB | 2–4 月前 | 疑似留档，`docker rm` 前确认 |
| docker 未挂载 volumes 6 个（49M/0/70M/49M/97M/49M） | 313 MB | — | `docker volume prune` |
| `/var/cache/veradata/{models 31G, global.sqlite3 24G, dcube 20G, annot_review 7.4G}` | 83 G（在 `/`） | 7–8 月 | 构建期缓存，是否可再生需用户判断（`preset_bank_full` 2.8G 为配置引用，保留） |
| `/var/log`（syslog.1 7.6G 等） | 16 G（在 `/`） | — | root 所有，需 sudo：`sudo journalctl --vacuum-size=1G`；`sudo rm /var/log/syslog.{1,2.gz,3.gz}`（或配 logrotate 上限） |
| `/var/cache/apt` 183M + `/var/lib/apt/lists` 326M | 0.5 G | — | 需 sudo：`sudo apt-get clean` |
| `~/miniconda3/envs`：unsloth 11G(2026-01)、vllm 9.7G(01)、llm_factory 9.0G(02)、jarvisart_rl 8.6G(2025-12)、monetgpt 8.4G(02)、monetgpt_sam3 6.6G(02)、myenv 6.3G(2025-10)、其余 <1G | 54 G | 见前 | 旧环境；现役为 `/home/bc/envs/{databuild,q3vl_sft}` |
| `~/.cache/huggingface/hub`（SD1.5 4.0G、ic-light 1.7G、clip-vit-l-336 1.6G 等） | 7.7 G | — | 可重下，但可能仍在用 |
| `~/.cache/torch/hub` 3.1G；`databuild_viewer` 773M；`puppeteer` 652M；`ms-playwright` 646M | 5.2 G | — | 可再生 |
| `/home/bc/data/agent_loop` | 74 G | — | agent_loop 已退役（91453aa），产物是否留档 |
| `/home/bc/data/scratch` 39G、`shard_cache` 32G、`row_basis_20260803` 26G、`ro*_20260803` 12G | 109 G | 8 月 | 中间产物 |
| `/home/bc/retouching` 84G、`research_agent` 69G、`datasets` 54G、`code` 15G | 222 G | — | 非本战役目录 |
| `/home/bc/data/trash` | 430 M | 09-04 | 禁读；仅报大小 |
| `/tmp` 其他用户（zhengh）条目、`/swap.img` 8.1G | — | — | 非本人 / 系统 |

### 4.5 data-root 迁移方案
不适用（Docker Root Dir 已在 `/home/docker`，/home 可用 534 GB）。备用（若日后要改，需 sudo）：`sudo systemctl stop docker` → `/etc/docker/daemon.json` 写 `{"data-root": "/home/bc/docker", "runtimes": {...保留 nvidia...}}`
→ `sudo rsync -aP /home/docker/ /home/bc/docker/` → `sudo systemctl start docker` → `docker info | grep "Root Dir"` 核对后再删旧目录。

## 5. git
分支 `eng/sprf-mainline`；提交 1：**fc41641**（veraretouch_sprf/ + docker/ + .gitignore）；提交 2：本报告（`experiments/**` 被 .gitignore 覆盖，`git add -f`）。
未 push。工作区原有的未跟踪/已修改文件（dataset_build/tools/epr05*、q3vl/whatb/pubbench 等）未纳入提交。
`.gitignore` 追加：`*.pt`、`*.safetensors`、`*.tar.gz`、`pack_*/`、`pack_*.tar.gz`、`*.log`、`veraretouch_sprf/**/{runs,configs_resolved}/`、`docker/**/.env`（核对：无已跟踪文件命中）。

## 6. 待裁决项汇总
1. 4.4 表内各项是否清理（尤其 docker 未用镜像 ~210 GB、build cache 160 GB、veradata 83 GB、/var/log 16 GB）。
2. 是否拉取 ms-swift 镜像（cuda12.8.1/swift4.0.3；或先升驱动到 ≥ 580 用 swift4.4.3）并起容器。
3. rl 路线（OPD/OPSD vs GRPO）、教师上下文格式、奖励形式（−linf8 / identity 增益）、KL 形式与系数、G——全部待调研确认后再接线 `executor_reward` 主体。
4. `eval/probes/parity_report.py` 保持脚本形态还是包成函数（现为模块级执行）。
5. 主线包 provenance 的源文件 sha 与 EPR-051 已登记 run 不同（导入行改动），跨 sha 比较按 HANDOFF §8 处理；是否要在 RESULTS 登记表加一列「主线包 sha」。
