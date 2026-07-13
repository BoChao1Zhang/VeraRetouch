"""烘焙 3D-LUT：纯全局色彩 param preset → 农场 HALD 采样 → .cube → ΔE 验收落盘。

原理：HALD 色卡（64³ 色点，每点 4×4 像素块抗 JPEG 色度抽样串扰）经农场 LrC 渲染后，
读回即该 preset「输入色→输出色」映射的直接采样——比参数拟合+残差更准，且渲染后
永久走本地 GPU 3D-LUT 路（render_backend.BAKED_DIR）。

选目标（全部满足）：kind=param、无内嵌 mask、无专属残差、未烘焙、空间算子键
（Clarity/Texture/Dehaze/Sharpen/Grain/Vignette/NR）幅度低于阈值——3D LUT 表达
不了空间算子，幅度大的仍走 GPU replay（_global 残差）或农场。

验收：真实照片农场渲 vs 烘焙 LUT 渲，LAB ΔE 中位 ≤ ACCEPT_DE 才落盘 .cube；
不通过的记入 rejected.jsonl（含 ΔE），preset 保持原分流。

用法：
  python -m dataset_build.tools.bake_luts --limit 50          # PoC
  python -m dataset_build.tools.bake_luts                     # 全量
农场开销 2 张/preset（HALD + 验收照片）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

from dataset_build.core import render_backend  # noqa: E402
from dataset_build.source_qa import lr_render  # noqa: E402
from gpu_render.route import _attrs, _nonzero  # noqa: E402

N = 64                    # LUT 边长（64³）
CELL = 4                  # 每色点像素块边长（抗 JPEG 4:2:0 串扰）
ACCEPT_DE = 4.0           # 验收 ΔE 中位阈值（专属残差实测中位 3.4，烘焙应≤同级）
BANK = "/home/bc/data/datasets/vera_directionA_1M/preset_bank_full"
VERIFY_PHOTO = os.environ.get(
    "BAKE_VERIFY_PHOTO",
    "/home/bc/data/datasets/lr_calib/preset_test/verify.jpg")

# 空间算子键（3D LUT 无法表达）；|值| 超过阈值的 preset 不烘焙
SPATIAL_LIMITS = {
    "Clarity2012": 10, "Texture": 10, "Dehaze": 8, "Sharpness": 40,
    "GrainAmount": 10, "PostCropVignetteAmount": 8,
    "LuminanceSmoothing": 40, "ColorNoiseReduction": 100,
}


def hald_image() -> Image.Image:
    """生成 HALD 色卡：色点按 R 最快、G 次、B 最慢排列成 (N*CELL*?) 方图。
    读回端 read_lut 与此排布严格配套（我们同时控制两端，无需遵循 HALD 标准）。"""
    idx = np.arange(N ** 3)
    r = (idx % N) * (255.0 / (N - 1))
    g = ((idx // N) % N) * (255.0 / (N - 1))
    b = (idx // (N * N)) * (255.0 / (N - 1))
    side = int(N ** 1.5)              # 64^1.5 = 512：512×512 个色点
    px = np.stack([r, g, b], -1).reshape(side, side, 3).astype(np.uint8)
    return Image.fromarray(np.kron(px, np.ones((CELL, CELL, 1), dtype=np.uint8)))


def read_lut(rendered_path: str) -> np.ndarray:
    """渲染后的 HALD → grid[R,G,B,3] float32 0..1（块中心 2×2 均值抗噪）。"""
    arr = np.asarray(Image.open(rendered_path).convert("RGB"), dtype=np.float32)
    side = int(N ** 1.5)
    if arr.shape[0] != side * CELL or arr.shape[1] != side * CELL:
        raise ValueError(f"HALD 渲染尺寸变化: {arr.shape}（农场不许缩放）")
    c0 = CELL // 2 - 1
    blocks = arr.reshape(side, CELL, side, CELL, 3)[:, c0:c0 + 2, :, c0:c0 + 2, :]
    px = blocks.mean(axis=(1, 3)) / 255.0          # (side, side, 3)
    flat = px.reshape(N ** 3, 3)
    # 排布逆变换：idx = r + g*N + b*N²  →  grid[r,g,b]
    grid = np.zeros((N, N, N, 3), dtype=np.float32)
    idx = np.arange(N ** 3)
    grid[idx % N, (idx // N) % N, idx // (N * N)] = flat
    return grid


def write_cube(grid: np.ndarray, out_path: str) -> None:
    """grid[R,G,B,3] → .cube，行序匹配库内 load_cube 的解析约定。

    实证（2026-07-13）：preset_qa.load_cube 对行做 reshape(n,n,n,3) 且 _apply_cube
    用 [R,G,B] 索引第一/二/三轴 → 文件必须 R 最外层（B 变最快），与标准 .cube
    相反；写错则烘焙 LUT 色相全乱（闭环 ΔE 19 事故）。"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"LUT_3D_SIZE {N}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n")
        for r in range(N):
            for g in range(N):
                for b in range(N):
                    v = grid[r, g, b]
                    f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")


def delta_e(img_a: str, img_b: str) -> float:
    """两图 LAB ΔE 中位（长边对齐 768）。"""
    import cv2
    ims = []
    for p in (img_a, img_b):
        im = Image.open(p).convert("RGB")
        im.thumbnail((768, 768))
        ims.append(np.asarray(im))
    if ims[0].shape != ims[1].shape:
        ims[1] = np.asarray(Image.fromarray(ims[1]).resize(
            (ims[0].shape[1], ims[0].shape[0])))
    labs = [cv2.cvtColor(im, cv2.COLOR_RGB2LAB).astype(np.float32) for im in ims]
    d = labs[0] - labs[1]
    d[..., 0] *= 100.0 / 255.0
    de = np.sqrt((d ** 2).sum(-1))
    return float(np.median(de))


def spatial_ok(preset_path: str, fmt: str) -> bool:
    attrs, prof = _attrs(preset_path, fmt)
    if prof == "mask":
        return False
    for k, lim in SPATIAL_LIMITS.items():
        v = attrs.get(k)
        if v is not None and _nonzero(v):
            try:
                if abs(float(str(v).lstrip("+"))) > lim:
                    return False
            except ValueError:
                return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只烘前 N 个（PoC）")
    ap.add_argument("--verify-photo", default=VERIFY_PHOTO)
    a = ap.parse_args()

    rejected = set()
    rej_path = os.path.join(render_backend.BAKED_DIR, "rejected.jsonl")
    if os.path.exists(rej_path):
        rejected = {json.loads(l)["preset_id"] for l in open(rej_path) if l.strip()}
    targets = []
    for l in open(f"{BANK}/features.jsonl"):
        r = json.loads(l)
        if r["kind"] != "param":
            continue
        pid = r["preset_id"]
        if pid in rejected:
            continue          # 已验收拒绝的不重烘（重测浪费农场；要重试请删 rejected.jsonl 对应行）
        if render_backend.has_residual(pid) or render_backend.baked_lut_path(pid):
            continue
        fmt = r.get("fmt") or "xmp"
        if not spatial_ok(r["path"], fmt):
            continue
        targets.append((pid, r["path"], fmt))
    if a.limit:
        targets = targets[: a.limit]
    print(f"烘焙目标 {len(targets)} 个（param、无残差、未烘焙、空间算子低）", flush=True)

    stage = os.path.join(render_backend.BAKED_DIR, "_stage")
    os.makedirs(stage, exist_ok=True)
    hald_path = os.path.join(stage, "hald_identity.png")
    if not os.path.exists(hald_path):
        hald_image().save(hald_path, "PNG")

    import threading
    from concurrent.futures import ThreadPoolExecutor

    rej = open(os.path.join(render_backend.BAKED_DIR, "rejected.jsonl"), "a")
    cnt = {"ok": 0, "rej": 0, "err": 0}
    lock = threading.Lock()

    def _bake_one(job: tuple) -> None:
        i, (pid, path, fmt) = job
        try:
            # 两次农场渲染并发提交（农场准入由 lr_render._FARM_GATE 统一管）
            with ThreadPoolExecutor(max_workers=2) as sub:
                f_hald = sub.submit(lr_render.render_via_lr, path, fmt, hald_path)
                f_ver = sub.submit(lr_render.render_via_lr, path, fmt, a.verify_photo)
                r, rv = f_hald.result(), f_ver.result()
            if not r.get("ok"):
                raise RuntimeError(f"farm hald: {r.get('error_code')}")
            if not rv.get("ok"):
                raise RuntimeError(f"farm verify: {rv.get('error_code')}")
            grid = read_lut(r["after_path"])
            cube_tmp = os.path.join(stage, f"{pid}.cube")
            write_cube(grid, cube_tmp)
            lut_out = os.path.join(stage, f"{pid}_lut.jpg")
            res = render_backend.get_backend().render_cube(
                cube_tmp, [a.verify_photo], [lut_out], long_edge=0)
            if not res.get("ok"):
                raise RuntimeError(f"lut render: {res.get('error_code')}")
            de = delta_e(rv["after_path"], lut_out)
            with lock:
                if de <= ACCEPT_DE:
                    os.replace(cube_tmp,
                               os.path.join(render_backend.BAKED_DIR, f"{pid}.cube"))
                    cnt["ok"] += 1
                    tag = "OK"
                else:
                    os.remove(cube_tmp)
                    rej.write(json.dumps({"preset_id": pid, "de": round(de, 2)}) + "\n")
                    rej.flush()
                    cnt["rej"] += 1
                    tag = "REJ"
            print(f"[{i+1}/{len(targets)}] {tag} {pid} ΔE={de:.2f}", flush=True)
        except Exception as e:  # noqa: BLE001 - 单 preset 失败不停批
            with lock:
                cnt["err"] += 1
            print(f"[{i+1}/{len(targets)}] ERR {pid}: {str(e)[:100]}", flush=True)

    # 提交端并发 6（≈2×在线 client 数）；真正的农场准入在 lr_render._FARM_GATE。
    with ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(_bake_one, enumerate(targets)))
    rej.close()
    print(f"DONE ok={cnt['ok']} rej={cnt['rej']} err={cnt['err']}", flush=True)


if __name__ == "__main__":
    main()
