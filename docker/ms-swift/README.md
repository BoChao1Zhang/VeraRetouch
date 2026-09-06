# docker/ms-swift —— ms-swift 容器准备件（EPR-052，**未拉镜像、未启动**，等用户裁决）

## 本机核实（2026-09-06）
| 项 | 值 |
|---|---|
| docker | Docker version 29.1.2；Docker Compose v5.0.0 |
| 用户组 | `bc` 在 `docker` 组（免 sudo） |
| NVIDIA runtime | `docker info` Runtimes: `io.containerd.runc.v2 nvidia runc`（nvidia-container-toolkit 已装；`gpus: all` 可用） |
| GPU / 驱动 | H100 ×2（97,871 MiB）；driver 570.195.03，CUDA 12.8 |
| Docker Root Dir | `/home/docker`（在 /home 分区，3.8T，可用 534 GB；不在 `/` 上，无需迁 data-root） |
| 现有镜像 | 12 个共 357.6 GB（含 monetgpt-ms-swift:latest 45.9 GB —— 旧 ms-swift 自建镜像，与本准备件无关） |

## 镜像（来源核实：ms-swift 仓库 `docs/source_en/GetStarted/SWIFT-installation.md` 「Docker」节）
文档列出的官方镜像均为 `modelscope-registry.{cn-hangzhou,cn-beijing,us-west-1}.cr.aliyuncs.com/modelscope-repo/modelscope:<tag>`。
最新 `swift4.4.3` 系列是 `cuda13.0.3`（需 driver ≥ 580），本机 570 不满足；`.env.example` 默认选
`ubuntu22.04-cuda12.8.1-py311-torch2.10.0-vllm0.17.1-modelscope1.34.0-swift4.0.3`（cuda12.8.1，与本机 CUDA 12.8 匹配）。
预计镜像大小：文档未给出数字；同类 ms-swift/verl 全量镜像在本机为 45.9–49 GB（`docker images` 实测），按 **~40–50 GB** 预留（落 /home 分区）。

## 操作（用户裁决后执行）
```bash
cd /home/bc/VeraRetouch/docker/ms-swift
cp .env.example .env                      # 按需改 MS_SWIFT_IMAGE / CONTAINER_NAME
docker compose config                     # 只渲染，不拉不起
docker compose pull                       # 拉镜像（~40–50 GB，走 aliyuncs 镜像仓）
docker compose up -d                      # 起容器（command = sleep infinity，不会自动训练）
docker compose exec swift bash            # 进容器
# 容器内验证
swift --version
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
nvidia-smi
python -c "import veraretouch_sprf, veraretouch_sprf.data.cot_text as C; print(C.template_sha256()[:12])"
# 关停
docker compose down
```

## 挂载与路径
| 宿主 | 容器 | 用途 |
|---|---|---|
| /home/bc/VeraRetouch | /workspace/VeraRetouch | 代码（PYTHONPATH） |
| /home/bc/data | /data | runs / builds / hf_home / modelscope_cache |
| /home/bc/data/models | /models | Qwen3-VL-4B、SigLIP2 |
| /mnt/nfs-ro | /nfs-ro (ro) | 249/279 分片（只读；NFS 写纪律不适用于容器内） |
| /var/cache/veradata/preset_bank_full | /lut_bank (ro) | 解码器 LUT bank |
容器内的 `veraretouch_sprf` 配置通过 `VR_PATH_<KEY>` 环境变量（compose 已写）把 `[paths]` 指到容器路径。

## 注意
- 宿主 GPU 现被 q 队列作业占用（gpu0 ~68 GB）；容器内起训练前按显存共存规则（已占 + 峰值 < 65 GB）核对。
- `ipc: host` + `shm_size 64g`：dataloader worker 共享内存；`ulimits.nofile 65536`：避免 fd 1024 软限（pueued 同款陷阱）。
- 不需要 sudo：docker 组 + 已装 nvidia runtime。若日后要改 data-root（当前不需要），方案见 ENG_REPORT §4。
