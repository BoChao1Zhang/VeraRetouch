"""gpu_render —— Lightroom preset 本地渲染栈（自 monetGPT tools/lr_calib 迁移的自包含包）。

组成：
- replay.py / sweeps.py / local_apply.py：numpy 路径（preset 解析 + LR 管线顺序回放）
- ops_v2/：LR-faithful numpy 算子注册表
- gpu/：torch 算子（cuda:1）+ gpu_replay + render_batch（生产驱动）+ residual_gpu
- residual.py / route.py：每 preset 残差 LUT 拟合/应用与分流
- fits/：标定 JSON + residual/*.npz（随包携带）
- image_ops/：vendor 自 monetGPT 的算子核（non_gimp_ops 等），import 已收编到包内
- configs/hsl.yaml：HSL 带定义数据

数据根（标定 GT / 探针 / preset 库）不随包迁：CALIB_ROOT=/home/bc/data/datasets/lr_calib。

用法（cwd=/home/bc/VeraRetouch，monetgpt_sam3 env）：
  # CLI：单 preset 单图（numpy 路径）
  python -m gpu_render.replay --preset x.xmp --image p.jpg --out out.jpg
  # CLI：GPU 批渲染（残差 LUT 默认开）
  python -m gpu_render.gpu.render_batch --preset x.xmp --images dir/ --out outdir/

  # Python API（numpy）
  from gpu_render.replay import parse_preset, replay
  from gpu_render import ops_v2
  from gpu_render.image_ops.non_gimp_ops import apply_non_gimp_config
  out, leftover = replay(img_f32, parse_preset(p, "xmp"), ops_v2.REGISTRY,
                         lambda cfg, im: apply_non_gimp_config(cfg, im, 255.0))

  # Python API（GPU, cuda:1）
  from gpu_render.gpu.gpu_replay import replay_batch, to_batch, to_hwc_list
  out, info = replay_batch(to_batch([img_f32]), parse_preset(p, "xmp"))

环境变量（语义与 monetGPT 一致）：
  MONETGPT_NON_GIMP_BACKEND=numpy|torch   apply_non_gimp_config 的后端（默认 torch）
  MONETGPT_TORCH_DEVICE=cuda:1            torch 设备
  MONETGPT_HSL_CONFIG_PATH                覆盖包内 configs/hsl.yaml
"""
