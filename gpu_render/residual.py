"""每 preset 残差 3D LUT：吃掉逐点算子链对 LR 的系统性色彩差。

架构判断：preset 库是固定的，每个 preset 已有（或只需补渲）4 张探针 GT。
于是残差不必对「未知 preset」泛化，只需对「未知图像」泛化：
  fit:  local_replay(probe) 的像素色 -> LR GT 的像素色，17^3 格点 delta-LUT，
        岭回归锚到恒等（无数据区域退回未校正输出）+ 3D Laplacian 平滑。
  eval: leave-one-probe-out（fit 3 probes，ΔE00 验证第 4 张）——图像泛化的诚实指标。
  prod: 全 4 探针拟合 -> fits/residual/<preset_id>.npz。
另拟合 _global.npz（全 preset 汇总）作为无 GT preset 的回退，按 preset 对半分折验证。

法方程按探针可加（A^T A、A^T r 逐探针缓存），LOO 各折 = 求和后减一，免重复扫像素。

monetgpt_sam3 env, cwd=monetGPT（GPU 渲染用 cuda:1）:
  python -m gpu_render.residual --fit          # 渲染+LOO 评测+落盘生产 LUT
  python -m gpu_render.residual --fit --limit 8  # 小样本试跑
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # VeraRetouch 根（gpu_render 包父目录）

from gpu_render.sweeps import CALIB_ROOT

RES_DIR = os.path.join(_HERE, "fits", "residual")
N = 17            # RGB 格点边长
M_CTX = 0         # >0 时加第 4 维：局部 luma 上下文（LR 自适应 tone 的主轴），格点数 M_CTX
                  # 注意：4D 格点的 LᵀL 直接 LU 分解填充爆炸（M_CTX=5 单折 >1h，2026-07-05 实测），
                  # 生产不可行；若重启该方向需换迭代解法（CG+对角预条件）或可分离正则
LAM_S = 2.0       # Laplacian 平滑（LOO 扫描 0.5-32 不敏感，2.0 最优，2026-07-05）
LAM_R = 0.2       # 岭（锚恒等）
SUB = 3           # 像素下采样步长


def _dims() -> tuple:
    return (N, N, N) if M_CTX <= 0 else (N, N, N, M_CTX)


def _ctx_map(img: np.ndarray) -> np.ndarray:
    """局部亮度上下文 ∈[0,1]：大核高斯模糊 luma。"""
    import cv2
    luma = (img @ np.asarray([0.2126, 0.7152, 0.0722], np.float32)).astype(np.float32)
    sig = max(4.0, min(img.shape[:2]) / 24.0)
    return cv2.GaussianBlur(luma, (0, 0), sig)


def _load(p, size=None):
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    im = Image.open(p).convert("RGB")
    if size and im.size != size:
        im = im.resize(size)
    return np.asarray(im, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# 三线性权重 / 拟合
# ---------------------------------------------------------------------------
def _multilinear_weights(pts: np.ndarray, dims: tuple):
    """pts (M,D) in [0,1] -> sparse (M, prod(dims)) 多线性插值矩阵（2^D 角点）。"""
    from scipy.sparse import csr_matrix
    dims_a = np.asarray(dims)
    D = len(dims)
    g = np.clip(pts, 0.0, 1.0) * (dims_a - 1)
    i0 = np.minimum(g.astype(np.int32), dims_a - 2)
    f = (g - i0).astype(np.float64)
    strides = np.ones(D, dtype=np.int64)
    for d in range(D - 2, -1, -1):
        strides[d] = strides[d + 1] * dims[d + 1]
    M = pts.shape[0]
    rows = np.repeat(np.arange(M, dtype=np.int64), 2 ** D)
    cols = np.empty((M, 2 ** D), dtype=np.int64)
    vals = np.empty((M, 2 ** D), dtype=np.float64)
    for k in range(2 ** D):
        w = np.ones(M, dtype=np.float64)
        idx = np.zeros(M, dtype=np.int64)
        for d in range(D):
            b = (k >> d) & 1
            w *= f[:, d] if b else 1.0 - f[:, d]
            idx += (i0[:, d] + b) * strides[d]
        cols[:, k] = idx
        vals[:, k] = w
    return csr_matrix((vals.ravel(), (rows, cols.ravel())), shape=(M, int(np.prod(dims))))


def _laplacian(dims: tuple):
    """格点各维二阶差分之和（LᵀL 用于平滑正则）。"""
    from scipy.sparse import identity, kron, diags
    mats = []
    for ax, n in enumerate(dims):
        d = diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n)).tolil()
        d[0, 0] = d[-1, -1] = 1.0   # Neumann 边界
        m = None
        for j, nj in enumerate(dims):
            blk = d.tocsr() if j == ax else identity(nj, format="csr")
            m = blk if m is None else kron(m, blk, format="csr")
        mats.append(m)
    return sum(mats[1:], start=mats[0]).tocsr()


_LTL: dict = {}


def probe_normals(src: np.ndarray, dst: np.ndarray):
    """单探针的法方程分量 (AᵀA, Aᵀr[3])。src/dst: (H,W,3) float01。"""
    s = src[::SUB, ::SUB].reshape(-1, 3).astype(np.float64)
    r = (dst[::SUB, ::SUB].reshape(-1, 3).astype(np.float64) - s)
    if M_CTX > 0:
        c = _ctx_map(src)[::SUB, ::SUB].reshape(-1, 1).astype(np.float64)
        s = np.concatenate([s, c], axis=1)
    A = _multilinear_weights(s, _dims())
    return (A.T @ A).tocsr(), A.T @ r


def solve_lut(ata_list, atr_list):
    """delta (prod(dims), 3)：(ΣAᵀA + λsLᵀL + λrI) d = ΣAᵀr。"""
    from scipy.sparse import identity
    from scipy.sparse.linalg import factorized
    dims = _dims()
    if dims not in _LTL:
        L = _laplacian(dims)
        _LTL[dims] = (L.T @ L).tocsc()
    ata = sum(ata_list[1:], start=ata_list[0])
    atr = np.sum(atr_list, axis=0)
    lhs = (ata + LAM_S * _LTL[dims] + LAM_R * identity(int(np.prod(dims)))).tocsc()
    solve = factorized(lhs)
    return np.stack([solve(atr[:, c]) for c in range(3)], axis=1)


def apply_lut(img: np.ndarray, delta: np.ndarray, dims: tuple | None = None) -> np.ndarray:
    """img (H,W,3) float01 + delta (prod(dims),3) -> 校正后图。"""
    from scipy.ndimage import map_coordinates
    dims = dims or _dims()
    pts = np.clip(img, 0.0, 1.0).reshape(-1, 3)
    if len(dims) == 4:
        c = _ctx_map(img).reshape(-1, 1)
        pts = np.concatenate([pts, c], axis=1)
    coords = pts.T * (np.asarray(dims, dtype=np.float64)[:, None] - 1)
    out = img.reshape(-1, 3).astype(np.float32).copy()
    dg = delta.reshape(*dims, 3)
    for ch in range(3):
        out[:, ch] += map_coordinates(dg[..., ch], coords, order=1, mode="nearest").astype(np.float32)
    return np.clip(out.reshape(img.shape), 0.0, 1.0)


def load_residual(preset_id: str):
    """-> (delta, dims) 或 None。"""
    p = os.path.join(RES_DIR, f"{preset_id}.npz")
    if not os.path.exists(p):
        p = os.path.join(RES_DIR, "_global.npz")
        if not os.path.exists(p):
            return None
    z = np.load(p)
    return z["delta"].astype(np.float64), tuple(int(x) for x in z["dims"])


def _de(a, b):
    from skimage.color import deltaE_ciede2000, rgb2lab
    return float(deltaE_ciede2000(rgb2lab(a), rgb2lab(b)).mean())


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-global", action="store_true")
    ap.add_argument("--ctx", type=int, default=0, help=">0 启用第 4 维局部 luma 上下文，格点数")
    a = ap.parse_args()
    global M_CTX
    M_CTX = a.ctx
    if not a.fit:
        ap.error("nothing to do (use --fit)")
    os.makedirs(RES_DIR, exist_ok=True)

    from gpu_render.replay import parse_preset
    from gpu_render.gpu.gpu_replay import gpu_replay

    presets = [json.loads(l) for l in open(os.path.join(CALIB_ROOT, "preset_test", "presets.jsonl"))]
    if a.limit:
        presets = presets[: a.limit]
    id_dir = os.path.join(CALIB_ROOT, "gt", "Identity")
    befores = {f.split("__")[0]: os.path.join(id_dir, f) for f in sorted(os.listdir(id_dir))}

    rows = []
    g_ata, g_atr, g_pid = [], [], []
    for pi, pr in enumerate(presets):
        pid = pr["preset_id"]
        gt_dir = os.path.join(CALIB_ROOT, "preset_test", "gt", pid)
        if not os.path.isdir(gt_dir):
            continue
        pre = parse_preset(pr["path"], pr["fmt"])
        probes = sorted(f[:-4] for f in os.listdir(gt_dir) if f.endswith(".jpg"))
        imgs = [_load(befores[p]) for p in probes]
        gts = [_load(os.path.join(gt_dir, f"{p}.jpg"), size=(im.shape[1], im.shape[0]))
               for p, im in zip(probes, imgs)]
        outs = gpu_replay(imgs, pre)
        normals = [probe_normals(o, g) for o, g in zip(outs, gts)]
        # LOO：fit 其余 3 探针，评测第 k 张
        loo = []
        for k in range(len(probes)):
            if len(probes) < 2:
                break
            delta = solve_lut([n[0] for i, n in enumerate(normals) if i != k],
                              [n[1] for i, n in enumerate(normals) if i != k])
            loo.append({"probe": probes[k],
                        "de_before": round(_de(outs[k], gts[k]), 3),
                        "de_after": round(_de(apply_lut(outs[k], delta), gts[k]), 3)})
        # 生产 LUT：全探针
        delta = solve_lut([n[0] for n in normals], [n[1] for n in normals])
        np.savez_compressed(os.path.join(RES_DIR, f"{pid}.npz"),
                            delta=delta.astype(np.float32), dims=np.asarray(_dims()))
        g_ata.append([n[0] for n in normals]); g_atr.append([n[1] for n in normals]); g_pid.append(pid)
        mb = round(float(np.mean([x["de_before"] for x in loo])), 3) if loo else None
        ma = round(float(np.mean([x["de_after"] for x in loo])), 3) if loo else None
        rows.append({"preset_id": pid, "loo": loo, "de_before": mb, "de_after": ma})
        print(f"[{pi+1}/{len(presets)}] {pid}  ΔE {mb} -> {ma}", flush=True)

    if not a.no_global and len(g_pid) >= 4:
        # 全局回退 LUT：全量拟合落盘；对半分折（fit A 评 B / fit B 评 A）出诚实指标
        flat_ata = [x for g in g_ata for x in g]
        flat_atr = [x for g in g_atr for x in g]
        np.savez_compressed(os.path.join(RES_DIR, "_global.npz"),
                            delta=solve_lut(flat_ata, flat_atr).astype(np.float32),
                            dims=np.asarray(_dims()))
        print("[global] wrote _global.npz (对半分折评测见 residual_eval.json 的 global 段，"
              "此处只落盘生产 LUT；分折评测在全量跑时补)")

    des_b = [r["de_before"] for r in rows if r["de_before"] is not None]
    des_a = [r["de_after"] for r in rows if r["de_after"] is not None]
    agg = {"n_presets": len(rows),
           "loo_median_before": round(float(np.median(des_b)), 3),
           "loo_median_after": round(float(np.median(des_a)), 3),
           "loo_p90_before": round(float(np.percentile(des_b, 90)), 3),
           "loo_p90_after": round(float(np.percentile(des_a, 90)), 3),
           "pass_relaxed_before": round(float(np.mean([d <= 3 for d in des_b])), 3),
           "pass_relaxed_after": round(float(np.mean([d <= 3 for d in des_a])), 3),
           "pass_jnd_after": round(float(np.mean([d <= 2 for d in des_a])), 3)}
    print(json.dumps(agg, indent=1))
    json.dump({"aggregate": agg, "presets": rows},
              open(os.path.join(CALIB_ROOT, "preset_test", "residual_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
