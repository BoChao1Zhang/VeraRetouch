"""numpy(参考) ↔ torch(GPU) 算子对齐验证 + 计时。移植时用它保证 GPU 版复现 numpy 版。

parity(op_numpy_fn, op_gpu_fn, ctx, img) -> {de_mean, de_p95, np_ms, gpu_ms}
- op_numpy_fn(img_np_hwc01, ctx) -> img_np_hwc01
- op_gpu_fn(img_torch, ctx) -> img_torch  (设备 cuda:1, 形状见约定)
达标: de_mean < 0.5 (人眼不可辨), 视为 GPU 版忠实复现。

约定: GPU 张量 BCHW float32 [0,1] on cuda:1。parity 负责 numpy<->torch 的转换与设备搬运。
"""
from __future__ import annotations

import os
import time

import numpy as np

DEVICE = os.environ.get("MONETGPT_TORCH_DEVICE", "cuda:1")


def to_gpu(img_hwc: np.ndarray):
    import torch
    t = torch.from_numpy(np.ascontiguousarray(img_hwc.transpose(2, 0, 1)[None]))  # 1CHW
    return t.to(DEVICE, dtype=torch.float32)


def to_np(t) -> np.ndarray:
    return t.detach().float().cpu().numpy()[0].transpose(1, 2, 0)


def _de(a: np.ndarray, b: np.ndarray) -> tuple:
    from skimage.color import deltaE_ciede2000, rgb2lab
    de = deltaE_ciede2000(rgb2lab(np.clip(a, 0, 1)), rgb2lab(np.clip(b, 0, 1)))
    return float(de.mean()), float(np.percentile(de, 95))


def parity(op_numpy_fn, op_gpu_fn, ctx: dict, img_hwc: np.ndarray, iters: int = 3) -> dict:
    import torch
    # numpy 参考
    t0 = time.time()
    for _ in range(iters):
        ref = op_numpy_fn(img_hwc.copy(), ctx)
    np_ms = (time.time() - t0) / iters * 1000
    ref = np.clip(np.asarray(ref, dtype=np.float32), 0, 1)
    # GPU
    g = to_gpu(img_hwc)
    op_gpu_fn(g.clone(), ctx)  # warmup
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    for _ in range(iters):
        out = op_gpu_fn(g.clone(), ctx)
    torch.cuda.synchronize(DEVICE)
    gpu_ms = (time.time() - t0) / iters * 1000
    got = np.clip(to_np(out), 0, 1)
    if got.shape != ref.shape:
        return {"de_mean": 99.0, "de_p95": 99.0, "np_ms": np_ms, "gpu_ms": gpu_ms,
                "error": f"shape {got.shape} != {ref.shape}"}
    dm, dp = _de(ref, got)
    return {"de_mean": round(dm, 3), "de_p95": round(dp, 3),
            "np_ms": round(np_ms, 1), "gpu_ms": round(gpu_ms, 1),
            "speedup": round(np_ms / max(gpu_ms, 0.01), 1), "passed": dm < 0.5}


def batch_throughput(op_gpu_fn, ctx: dict, hw=(1067, 1600), batch=16, iters: int = 3) -> dict:
    """GPU 批处理吞吐: batch 张同时过一个 op。"""
    import torch
    x = torch.rand(batch, 3, hw[0], hw[1], device=DEVICE, dtype=torch.float32)
    op_gpu_fn(x.clone(), ctx)
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    for _ in range(iters):
        op_gpu_fn(x.clone(), ctx)
    torch.cuda.synchronize(DEVICE)
    ms = (time.time() - t0) / iters * 1000
    return {"batch": batch, "hw": hw, "ms_per_batch": round(ms, 1),
            "ms_per_image": round(ms / batch, 2), "images_per_sec": round(batch / (ms / 1000), 1)}
