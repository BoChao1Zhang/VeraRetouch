"""生产批渲染驱动：文件 → GPU preset 链(+残差 LUT) → 文件，IO 与 GPU 重叠。

流水线：解码线程池(预取下一批) → pinned uint8 H2D + GPU 归一化 → 链 → GPU 量化
uint8 D2H → 编码线程池。相比 naive 串行（float32 传输 + 同步解码/编码），
传输量降 4×、解码/编码全部藏进 GPU 时间。

单 preset 铺渲图集（吞吐主场景）：
  python -m gpu_render.gpu.render_batch --preset x.xmp --images dir/ --out outdir/ \
      [--batch 16] [--residual rcp_xxx]（有残差 LUT 的 preset 传 preset_id 启用）
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # VeraRetouch 根（gpu_render 包父目录）

import torch

from gpu_render.gpu.gpu_replay import DEVICE, replay_batch
from gpu_render.replay import parse_preset


def _decode(path: str) -> np.ndarray:
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    return np.asarray(Image.open(path).convert("RGB"))          # uint8 HWC


def _encode(arr_u8: np.ndarray, dst: str, quality: int) -> None:
    from PIL import Image
    Image.fromarray(arr_u8).save(dst, "JPEG", quality=quality)


def _upload(imgs_u8: list, device: str) -> torch.Tensor:
    """uint8 HWC 列表(同形) -> pinned -> (B,3,H,W) float01 on GPU（归一化在 GPU 上做）。"""
    arr = np.stack([np.ascontiguousarray(x.transpose(2, 0, 1)) for x in imgs_u8])
    t = torch.from_numpy(arr)
    if "cuda" in device:
        t = t.pin_memory().to(device, non_blocking=True)
    else:
        t = t.to(device)
    return t.to(torch.float32).div_(255.0)


def _download_u8(out_bchw: torch.Tensor) -> list:
    """u8 量化与 BCHW→BHWC permute 全在 GPU 上完成后一次 D2H（float 传输 ÷4；
    CPU 侧零拷贝切分，不再逐图 transpose+copy）。
    取整语义与 float 路径逐位一致：trunc(clamp(x,0,1)*255+0.5)。"""
    u8 = (out_bchw.clamp(0, 1).mul(255.0).add_(0.5).to(torch.uint8)
          .permute(0, 2, 3, 1).contiguous().cpu().numpy())
    return [u8[b] for b in range(u8.shape[0])]


def render_files(preset: dict, jobs: list, batch: int = 16, quality: int = 92,
                 residual_id: str | None = None, io_workers: int = 8) -> dict:
    """jobs: [(src_path, dst_path)]，同 preset。按 (H,W) 分桶、桶内成批流水线渲染。"""
    res = None
    if residual_id:
        from gpu_render.residual import load_residual
        res = load_residual(residual_id)
    delta_dims = None
    if res is not None:
        from gpu_render.gpu.residual_gpu import apply_residual_batch
        delta_dims = res

    stats = {"images": 0, "batches": 0, "wall_s": 0.0}
    t_all = time.perf_counter()
    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        # 流式：header 读尺寸分桶（不解码整图），批间单批预取，解码/编码藏进 GPU 时间
        from PIL import Image
        buckets: dict = {}
        for src, dst in jobs:
            with Image.open(src) as im:
                buckets.setdefault((im.size[1], im.size[0]), []).append((src, dst))
        enc_futs = []

        def _dec_chunk(chunk):
            return [pool.submit(_decode, s) for s, _ in chunk]   # 非阻塞：future 列表

        with torch.no_grad():
            for shape, items in buckets.items():
                chunks = [items[i:i + batch] for i in range(0, len(items), batch)]
                nxt = _dec_chunk(chunks[0])
                for ci, chunk in enumerate(chunks):
                    imgs = [f.result() for f in nxt]
                    if ci + 1 < len(chunks):
                        nxt = _dec_chunk(chunks[ci + 1])
                    t = _upload(imgs, DEVICE)
                    out, _ = replay_batch(t, preset)
                    if delta_dims is not None:
                        from gpu_render.gpu.residual_gpu import apply_residual_batch
                        out = apply_residual_batch(out, *delta_dims)
                    for arr, (_, dst) in zip(_download_u8(out), chunk):
                        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
                        enc_futs.append(pool.submit(_encode, arr, dst, quality))
                    stats["images"] += len(chunk)
                    stats["batches"] += 1
                    del t, out
        for f in enc_futs:
            f.result()
    torch.cuda.empty_cache()
    stats["wall_s"] = round(time.perf_counter() - t_all, 2)
    stats["images_per_min"] = round(stats["images"] / max(stats["wall_s"], 1e-6) * 60, 1)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--fmt", default="")
    ap.add_argument("--images", required=True, help="目录或单文件")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--residual", default="", help="preset_id：启用该 preset 的残差 LUT")
    a = ap.parse_args()
    fmt = a.fmt or ("xmp" if a.preset.endswith(".xmp") else "lrtemplate")
    pre = parse_preset(a.preset, fmt)
    if os.path.isdir(a.images):
        srcs = sorted(os.path.join(a.images, f) for f in os.listdir(a.images)
                      if f.lower().endswith((".jpg", ".jpeg", ".png")))
    else:
        srcs = [a.images]
    jobs = [(s, os.path.join(a.out, os.path.basename(s))) for s in srcs]
    stats = render_files(pre, jobs, a.batch, a.quality, a.residual or None)
    print(stats)


if __name__ == "__main__":
    main()
