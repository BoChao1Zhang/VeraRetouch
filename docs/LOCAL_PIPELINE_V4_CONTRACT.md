# Local pipeline v4 接口契约（2026-07-11 开工版）

Phase 1 改造中三方共用的接口定义。1A/1B（construct 侧）与 1C（gpu_render/core 侧）并行开发，以本文为准。

## 1. CGT spec 扩展：语义 mask 通道

`render_local_preset_variants` / `render_backend.render_local_variants` 的
`variants[i].spec` 在既有两种几何外新增语义形态：

```python
# 几何（既有，不变）
{"mask_type": "circulargradient" | "gradient", "geom": {...}, "amount": 1.0}
# 语义（新增）
{"mask_type": "semantic", "alpha": np.ndarray,  # (H,W) float32 [0,1]，可为降采样分辨率
 "amount": 1.0}
```

- `alpha` 分辨率允许 ≤ 源图（羽化在 ≤1024 短边上算，性能项6）；渲染端负责
  双线性 resize 到源图 H×W 后合成（对齐 `local_replay.corr_alpha` 的语义）。
- GPU 路（`local_preset.raster_cgt_batch`）与农场路
  （`render_backend._render_local_preset_farm`）都必须支持。
- 合成语义不变：精确 α-lerp（`composite_srgb`），α=0 处逐位等于源图。

## 2. CGT PNG 产出移交渲染后端（性能项1+3）

- 调用方在 `variants[i]` 里给出目标路径：`{"out_path": ..., "cgt_path": ..., "spec": ...}`。
- 后端把**实际用于合成的 α**（几何=GPU 光栅下载；语义=resize 后的 alpha）转
  u8 存单通道 PNG（`compress_level=1`）写到 `cgt_path`，结果 row 返回
  `"cgt_path"`。农场路在 CPU 侧同样产出。
- construct 侧（mask_synth.make_local_samples）不再自己跑 `cgt_raster` 存
  PNG——消除 p95 ~3s 的重复光栅。
- `cgt_path` 缺省（None）时后端不存（兼容非数据集调用）。

## 3. 锁粒度与编码（性能项4+5+7）

- `_gpu_lock` 只覆盖：decode/upload/replay/residual/composite/GPU 端 u8 量化
  /`.cpu()` 下载。JPEG/PNG 编码与磁盘写在锁外并行（ThreadPool）。
- `render_batch._download_u8`：量化在 GPU 上做（`(x*255).round().clamp.to(uint8)`
  再 permute/contiguous/cpu），传输量 ÷4。

## 4. 语义 α 生产端（1A，construct/subject_geom）

- 新 cache：`$CONSTRUCT_SUBJECT_CACHE`（默认
  `/home/bc/data/datasets/vera_directionA_1M/subject_cache/<path_key>/`）
  `subject.png`（清理后主体 mask）+ `subject.json`（status/scope/守卫指标）。
- 羽化：`feather_binary(hard, f_in, f_in/3)`，`f_in = 0.008+0.017*sqrt(area)`
  × U[0.7,1.4] 抖动；在短边 ≤1024 上计算。
- 每源 8 变体组成：1 径向 + 1 语义 + 2 束状 + 4 线性（四方向轮询）；
  无主体/守卫舍弃 → 8× bisect；单项几何失败 → bisect 补位。

## 5. 不变量（两侧都要保）

- α=0 像素 == 源图字节级不变（PNG 下无损意义上）。
- 结果 row 顺序 == 输入 variants 顺序。
- base preset 含内嵌 local corrections 仍拒绝（`base_preset_has_embedded_locals`）。
- 测试：`dataset_build/tools/test_local_preset_contract.py`、
  `dataset_build/core/test_render_backend_local_variants.py` 保持通过（可按新契约改）。
