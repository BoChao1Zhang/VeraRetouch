"""统一渲染后端 ``core.render_backend`` —— 本地 GPU 为核心路径，LR 农场只兜底。


分流规则（2026-07-13 重构定稿：本地优先，农场是兜底不是主路）：
  * 专属残差 LUT（``gpu_render/fits/residual/<pid>.npz``，标定 preset）→ 本地 GPU；
  * 烘焙 LUT（``BAKED_DIR/<pid>.cube``，纯全局色彩 preset 经农场 HALD 采样烘焙，
    tools/bake_luts.py 产出并 ΔE 验收）→ 本地 GPU 3D-LUT（grid_sample）；
  * 其余键覆盖良好的 param（含空间算子，gpu_render.route 判定）→ 本地 GPU
    （``_global`` 残差兜底）；
  * 农场仅三种兜底：键不覆盖/内嵌 mask/exotic profile、渲染卡显存守卫触发、
    本地渲染抛异常（逐张回退，统计 local_fallback_farm）。
  * RENDER_BACKEND_POLICY=fidelity 可切回严格模式（只放行专属残差，逐像素保真
    对齐 Adobe 时用）；默认 throughput。

约束：GPU 只用 cuda:1（MONETGPT_TORCH_DEVICE=cuda:1；卡 0 留给他人 + IAA），
batch=16；GPU 路径进程内单例 + 互斥锁（gpu_render 的 replay 链非线程安全）。

接口：
  render(preset_path, fmt, image_paths, out_paths, preset_id=None) -> dict
      同一 preset 批量渲多张；返回 {ok, route, results[...], n_local, n_farm, stats}。
  render_one(preset_path, fmt, photo_path, out_path=None, preset_id=None) -> dict
      单张便捷入口；返回与 lr_render.render_via_lr 同形的
      {ok, after_path, engine, ...} / {ok:False, error_code, ...}。
  render_local_variants(base_preset_path, fmt, source_path, variants, preset_id=None) -> dict
      把同一个全局 base preset 限定到多个几何/语义 alpha；GPU 只回放一次 base，失败时
      农场只渲一次全局 base，再在 CPU 精确 alpha 合成各 variant。variants[i] 可带
      cgt_path：后端把实际合成用的 alpha 存单通道 PNG（性能项1+3）。
  params_to_xmp(params, out_path)
      teacher 参数字典（38 键 CRS，{k:{"value":v}}）→ 临时 XMP，供旧
      RenderClient 接口代理到本后端（无残差 → 走农场）。

torch / gpu_render / lr_render 全部惰性导入：仅走农场的环境无需装 torch。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

# 硬约束：本地渲染永远在 cuda:1（gpu_render.gpu.gpu_replay 也有同样的 setdefault，
# 这里提前设保证首次 torch 初始化前生效；不覆盖显式指定）。
os.environ.setdefault("MONETGPT_TORCH_DEVICE", "cuda:1")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # /home/bc/VeraRetouch
if _REPO not in sys.path:  # gpu_render 是仓库根下的顶层包
    sys.path.insert(0, _REPO)

#: 残差 LUT 目录（有 <preset_id>.npz 才算“已标定可本地渲”；_global.npz 不算）。
RES_DIR = os.path.join(_REPO, "gpu_render", "fits", "residual")
#: 烘焙 3D-LUT 目录（纯全局色彩 param 的农场 HALD 采样产物，tools/bake_luts.py 写入
#: <preset_id>.cube，验收通过才落盘；分流优先级在专属残差之后、_global 覆盖之前）。
BAKED_DIR = os.environ.get("RENDER_BACKEND_BAKED_DIR",
                           os.path.join(_REPO, "gpu_render", "fits", "baked"))
#: 标定 preset 清单（preset_id <-> path 映射；100 条与 RES_DIR 一一对应）。
CALIB_PRESETS_JSONL = os.environ.get(
    "RENDER_BACKEND_CALIB_JSONL",
    "/home/bc/data/datasets/lr_calib/preset_test/presets.jsonl")

GPU_BATCH = int(os.environ.get("CONSTRUCT_GPU_BATCH", "16"))  # 本地 GPU 批大小（可环境覆盖，避免挤爆共享卡）
FARM_WORKERS = 6        # 农场提交端并发（真正的农场准入在 lr_render._FARM_GATE）
# 本地渲染显存守卫：渲染卡空闲低于此值整批直接走农场，不去分配（避免 OOM churn
# 且不在共卡进程如 vLLM/IAA 旁留 CUDA 上下文）；余量恢复后自动放行本地路。
LOCAL_MIN_FREE_MB = int(os.environ.get("RENDER_LOCAL_MIN_FREE_MB", "18000"))
# 守卫查的 GPU 跟随渲染设备（MONETGPT_TORCH_DEVICE，上面已 setdefault cuda:1）
_RENDER_GPU_IDX = os.environ.get("MONETGPT_TORCH_DEVICE", "cuda:1").rsplit(":", 1)[-1]

# 预解析 LUT 包（2026-07-18）：tools/pack_lut_npz.py 把全部 native LUT 解析进
# luts.npz（grid 成品形态）+ luts_meta.json（path/domain）。lut-only 采样下用它
# 免去每次 load_cube 的 Python 解析税（GIL 串行 → GPU 饥饿）。首次访问懒加载全量
# 进内存 {realpath: (grid, dmin, dmax)}（~2-3GB）；文件缺失则回退逐次 load_cube。
_LUT_PACK_DIR = os.environ.get(
    "RENDER_LUT_PACK_DIR", "/var/cache/veradata/preset_bank_full")
_lut_pack: Optional[dict] = None
_lut_pack_lock = threading.Lock()


def _get_lut_pack() -> dict:
    global _lut_pack
    if _lut_pack is None:
        with _lut_pack_lock:
            if _lut_pack is None:
                _lut_pack = _load_lut_pack()
    return _lut_pack


def _load_lut_pack() -> dict:
    import numpy as np
    npz = os.path.join(_LUT_PACK_DIR, "luts.npz")
    meta_p = os.path.join(_LUT_PACK_DIR, "luts_meta.json")
    if not (os.path.exists(npz) and os.path.exists(meta_p)):
        print(f"[render_backend] 无 LUT 预解析包（{npz}），回退逐次 load_cube", file=sys.stderr)
        return {}
    import json as _json
    meta = _json.load(open(meta_p))
    data = np.load(npz)
    pack = {}
    for pid, m in meta.items():
        try:
            pack[m["path"]] = (np.asarray(data[pid], dtype="float32"),
                               np.asarray(m["dmin"], dtype="float32"),
                               np.asarray(m["dmax"], dtype="float32"))
        except KeyError:
            continue
    print(f"[render_backend] LUT 预解析包已载入: {len(pack)} 个网格", file=sys.stderr)
    return pack


_vram_cache: tuple = (0.0, 0)   # (checked_at, free_mb)
_vram_lock = threading.Lock()


def _gpu1_free_mb() -> int:
    """nvidia-smi 查渲染卡空闲显存，30s 缓存（nvml 查询无 CUDA 上下文开销）。"""
    global _vram_cache
    with _vram_lock:
        ts, free = _vram_cache
        if time.monotonic() - ts < 30:
            return free
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free",
                 "--format=csv,noheader,nounits", "-i", _RENDER_GPU_IDX],
                capture_output=True, text=True, timeout=10).stdout
            free = int(out.strip().splitlines()[0])
        except Exception:
            free = 1 << 20  # ponytail: 查询失败按充足放行，保持旧行为（OOM 会回退农场）
        _vram_cache = (time.monotonic(), free)
        return free

_FARM_FMTS = ("xmp", "lrtemplate")   # 双路都只吃 LR develop preset；lut 由调用方自渲


def _norm_fmt(preset_path: str, fmt: Optional[str]) -> str:
    f = (fmt or "").lower()
    if f in _FARM_FMTS:
        return f
    ext = os.path.splitext(preset_path)[1].lower().lstrip(".")
    return ext if ext in _FARM_FMTS else f


def has_residual(preset_id: Optional[str]) -> bool:
    """该 preset 是否有专属残差 LUT（分流依据；不接受 _global 兜底）。"""
    return bool(preset_id) and os.path.exists(os.path.join(RES_DIR, f"{preset_id}.npz"))


def baked_lut_path(preset_id: Optional[str]) -> Optional[str]:
    """该 preset 的烘焙 3D-LUT（.cube）路径，无则 None。"""
    if not preset_id:
        return None
    p = os.path.join(BAKED_DIR, f"{preset_id}.cube")
    return p if os.path.exists(p) else None


# --- preset 路径 -> 标定 preset_id 的惰性映射（调用方没传 preset_id 时兜底） ----
_CALIB_MAP: Optional[Dict[str, str]] = None
_CALIB_LOCK = threading.Lock()


def _calib_id_for_path(preset_path: str) -> Optional[str]:
    global _CALIB_MAP
    if _CALIB_MAP is None:
        with _CALIB_LOCK:
            if _CALIB_MAP is None:
                m: Dict[str, str] = {}
                try:
                    with open(CALIB_PRESETS_JSONL, encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            r = json.loads(line)
                            m[os.path.realpath(r["path"])] = r["preset_id"]
                except OSError:
                    pass  # 清单不在（异机）→ 全部走农场，保真不降
                _CALIB_MAP = m
    return _CALIB_MAP.get(os.path.realpath(preset_path))


def resolve_preset_id(preset_path: str, preset_id: Optional[str] = None) -> Optional[str]:
    return preset_id or _calib_id_for_path(preset_path)


# --------------------------------------------------------------------------- #
# teacher 参数字典 -> 临时 XMP（旧 RenderClient 接口的适配层）
# --------------------------------------------------------------------------- #
_XMP_TMPL = """<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
    crs:PresetType="Normal"
    crs:ProcessVersion="11.0"
    crs:HasSettings="True"
{attrs}
  />
 </rdf:RDF>
</x:xmpmeta>
"""


def params_to_xmp(params: Dict[str, Any], out_path: str) -> str:
    """VeraRetouch 参数字典（{key: {"value": v}} 或 {key: v}，RAW LR 单位）→ XMP。

    只写非零键；键名本身就是 CRS 键（Exposure2012 / HueAdjustmentRed / ...），
    xmp2lua（农场）与 gpu_render.parse_preset（本地）都能直接消费。"""
    lines = []
    for k in sorted(params):
        v = params[k]
        if isinstance(v, dict):
            v = v.get("value", 0.0)
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if abs(fv) < 1e-9:
            continue
        sv = f"{fv:+.4f}".rstrip("0").rstrip(".")
        lines.append(f'    crs:{k}="{sv}"')
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(_XMP_TMPL.format(attrs="\n".join(lines)))
    return out_path


# --------------------------------------------------------------------------- #
# 后端本体
# --------------------------------------------------------------------------- #
class RenderBackend:
    """进程内单例（get_backend()）：本地 GPU 串行互斥 + 农场并发提交 + 统计。"""

    def __init__(self, gpu_batch: int = GPU_BATCH, farm_workers: int = FARM_WORKERS,
                 policy: Optional[str] = None) -> None:
        self.gpu_batch = int(gpu_batch)
        self.farm_workers = max(1, int(farm_workers))
        # 分流策略：
        #   throughput（默认，2026-07-13 起）——本地 GPU 是核心路径：专属残差 > 烘焙
        #     LUT > 键覆盖良好走 _global 残差；农场仅兜底（不可本地/显存守卫/异常）。
        #     构建的自洽性（recipe 由本管线定义与执行）优先于对 Adobe 的逐像素保真，
        #     且农场 ~71/min 撑不住 10 万级渲染。曾因默认 fidelity + env -i 启动丢
        #     环境变量导致 pilot600 农场占 88%（2026-07-13），默认值遂翻转。
        #   fidelity ——严格模式：只放行专属残差标定 preset，逐像素对齐 Adobe 时用。
        self.policy = policy or os.environ.get("RENDER_BACKEND_POLICY", "throughput")
        # gpu_render 的 replay 链（含缓存的 fits/残差）非线程安全 → 全局互斥。
        # 渲染 GPU 并发（2026-07-17）：输入 1024 化（RENDER_INPUT_SHORT_EDGE）后单流
        # 显存尖峰消失，单锁改信号量放 RENDER_GPU_CONCURRENCY 条并发流（默认 1=旧行为）。
        self._gpu_lock = threading.BoundedSemaphore(
            max(1, int(os.environ.get("RENDER_GPU_CONCURRENCY", "1"))))
        self._preset_cache: Dict[tuple, Any] = {}    # (realpath, mtime, fmt) -> parse_preset() dict 或 cube tuple
        self._route_cache: Dict[tuple, bool] = {}    # (realpath, mtime_ns, fmt) -> 可本地渲
        self._locals_cache: Dict[tuple, bool] = {}   # (realpath, mtime_ns) -> 含内嵌 local 校正
        self._stats_lock = threading.Lock()
        self.stats: Dict[str, int] = {
            "local_images": 0,        # 本地 GPU 成功张数
            "farm_images": 0,         # 农场成功张数（含回退成功）
            "local_fallback_farm": 0, # 本地失败回退农场的张数
            "failed": 0,              # 双路都失败的张数
        }

    # -- 统计 ---------------------------------------------------------------
    def _bump(self, key: str, n: int = 1) -> None:
        with self._stats_lock:
            self.stats[key] = self.stats.get(key, 0) + n

    def stats_snapshot(self) -> Dict[str, int]:
        with self._stats_lock:
            return dict(self.stats)

    # -- 健康检查 ------------------------------------------------------------
    @staticmethod
    def farm_health() -> bool:
        try:
            from dataset_build.source_qa import lr_render
            return bool(lr_render.lr_health())
        except Exception:
            return False

    # -- 本地 GPU 路 ----------------------------------------------------------
    # 解析缓存（2026-07-18 提吞吐）：旧策略「256 条一满全清」在 7778 preset 随机
    # 访问下命中率趋零，load_cube 纯 Python 解析 64ms~1.8s/次且持 GIL，实测把
    # 标定吞吐压到 ~1260 组/h。改真 LRU、容量 4096（parsed grid ~0.4-3MB/个，
    # 全量 lut 驻留 ~2-3GB RAM）。dict 单操作 GIL 原子，条目竞态最多导致重解析。
    _CACHE_CAP = int(os.environ.get("RENDER_PRESET_CACHE_CAP", "4096"))

    def _cache_get(self, key):
        hit = self._preset_cache.get(key)
        if hit is not None:
            self._preset_cache.pop(key, None)
            self._preset_cache[key] = hit          # LRU 触达移到队尾
        return hit

    def _cache_put(self, key, val) -> None:
        while len(self._preset_cache) >= self._CACHE_CAP:
            try:
                self._preset_cache.pop(next(iter(self._preset_cache)))
            except (StopIteration, KeyError):
                break
        self._preset_cache[key] = val

    def _parse_cached(self, preset_path: str, fmt: str) -> dict:
        from gpu_render.replay import parse_preset
        rp = os.path.realpath(preset_path)
        key = (rp, os.path.getmtime(rp), fmt)
        pre = self._cache_get(key)
        if pre is None:
            pre = parse_preset(rp, fmt)
            self._cache_put(key, pre)
        return pre

    def _local_capable(self, preset_path: str, fmt: str) -> bool:
        """throughput 策略：用 gpu_render.route 的启发式判 preset 是否可本地渲
        （无 local-mask / 无未覆盖键 / 非 exotic profile）。"""
        rp = os.path.realpath(preset_path)
        try:
            mtime_ns = os.stat(rp).st_mtime_ns
        except OSError:
            return False
        key = (rp, mtime_ns, fmt)
        hit = self._route_cache.get(key)
        if hit is None:
            try:
                from gpu_render.route import route_preset
                hit = route_preset(rp, fmt).get("route") == "local"
            except Exception:
                hit = False
            if len(self._route_cache) > 256:
                self._route_cache.clear()
            self._route_cache[key] = hit
        return hit

    def _base_has_embedded_locals(self, preset_path: str) -> bool:
        """Dedicated local-preset 入口只接受纯全局 base preset。

        不能依赖 ``parse_preset()['locals']``：旧式 Gradient/Paint correction 目前不会
        全部进入该字段，但同样不能安全地再包一层几何 alpha。
        按 (realpath, mtime_ns) memoize：同一 base 每源一次的全文扫描是纯浪费。
        """
        rp = os.path.realpath(preset_path)
        key = (rp, os.stat(rp).st_mtime_ns)
        hit = self._locals_cache.get(key)
        if hit is None:
            from gpu_render.route import MASK_KEYS
            with open(rp, encoding="utf-8", errors="ignore") as f:
                text = f.read()
            hit = any(k in text for k in MASK_KEYS)
            if len(self._locals_cache) > 256:
                self._locals_cache.clear()
            self._locals_cache[key] = hit
        return hit

    def _render_local(self, preset_path: str, fmt: str, jobs: List[tuple],
                      preset_id: str) -> List[bool]:
        """整批本地渲染；返回逐 job 是否产出。异常向上抛（由 render() 回退农场）。"""
        from gpu_render.gpu.render_batch import render_files
        with self._gpu_lock:                     # cuda:1 并发受 RENDER_GPU_CONCURRENCY 约束
            preset = self._parse_cached(preset_path, fmt)
            render_files(preset, jobs, batch=self.gpu_batch, residual_id=preset_id)
        return [os.path.exists(dst) and os.path.getsize(dst) > 0 for _, dst in jobs]

    # -- GPU 3D-LUT 路（烘焙 param 与原生 .cube 共用） -------------------------
    def _cube_cached(self, cube_path: str):
        """load_cube 解析缓存 (grid[n,n,n,3] float32 0..1, dmin, dmax)。

        轴序（2026-07-17 修复定案）：load_cube 对标准 .cube 返回 grid[b][g][r]
        （B 最外层、R 最快），本引擎与 canonical CPU oracle 均按该布局
        正确索引 f(R,G,B)。历史事故：2026-07-13
        起曾按 [R,G,B] 误索引（实际 f(B,G,R)，红蓝互换），致 native LUT 半库被
        误判「反转黑白」；bake_luts.write_cube 当时反向行序写出补偿，修复时已随
        存量烘焙 .cube 一并迁回标准行序。"""
        import numpy as np
        rp = os.path.realpath(cube_path)
        packed = _get_lut_pack().get(rp)
        if packed is not None:
            return packed                            # 预解析命中：免 load_cube
        key = (rp, os.path.getmtime(rp), "cube")
        hit = self._cache_get(key)
        if hit is None:
            from dataset_build.lut_io import load_lut

            hit = load_lut(rp)
            self._cache_put(key, hit)
        return hit

    def _render_cube_gpu(self, cube_path: str, image_paths: Sequence[str],
                         out_paths: Sequence[str], long_edge: int = 0) -> List[bool]:
        """GPU trilinear 3D-LUT（grid_sample），与 canonical CPU oracle
        逐像素一致（含其轴序约定，见 _cube_cached docstring）。
        grid_sample 采样坐标最后维 (x=W,y=H,z=D) 对应 grid 轴 (2,1,0)。
        异常向上抛，由调用方回退 CPU trilinear 或农场。"""
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image
        grid, dmin, dmax = self._cube_cached(cube_path)
        oks: List[bool] = []
        with self._gpu_lock:
            from gpu_render.gpu.gpu_replay import DEVICE
            vol = torch.from_numpy(grid).permute(3, 0, 1, 2)[None].to(DEVICE)  # (1,3,D=b,H=g,W=r)
            span = np.where((dmax - dmin) == 0, 1.0, dmax - dmin)
            arrays = []
            try:
                with torch.no_grad():
                    for sp in image_paths:
                        from gpu_render.gpu.render_batch import _open_input
                        im = _open_input(sp)   # RENDER_INPUT_SHORT_EDGE 输入降采样
                        if long_edge:
                            im.thumbnail((long_edge, long_edge))
                        a = torch.from_numpy(
                            np.asarray(im, dtype=np.float32) / 255.0).to(DEVICE)
                        c = ((a - torch.as_tensor(dmin, device=DEVICE))
                             / torch.as_tensor(span, device=DEVICE)).clamp(0, 1)
                        # (1,1,H,W,3) 采样点 xyz=(R,G,B)：vol 轴 (D=b,H=g,W=r)，
                        # grid_sample x→W(r) y→H(g) z→D(b)，与 DiskLutApplier 一致。
                        # 2026-07-17 修复：此前 xyz=(B,G,R) 实际执行 f(B,G,R)。
                        pts = (c * 2 - 1)[None, None]
                        out = F.grid_sample(vol, pts, mode="bilinear",
                                            padding_mode="border", align_corners=True)
                        arrays.append(out[0, :, 0].permute(1, 2, 0)
                                      .clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy())
            finally:
                del vol
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        for arr, dst in zip(arrays, out_paths):
            try:
                self._save_rgb_u8(arr, dst, quality=95)
                oks.append(True)
            except Exception:  # noqa: BLE001 - 单张编码失败不拖累整批
                oks.append(False)
        return oks

    def render_cube(self, cube_path: str, image_paths: Sequence[str],
                    out_paths: Sequence[str], long_edge: int = 0) -> dict:
        """原生 .cube/.3dl 的公共入口：GPU 优先（显存守卫），CPU trilinear 由调用方兜底。"""
        if _gpu1_free_mb() < LOCAL_MIN_FREE_MB:
            return {"ok": False, "error_code": "vram_guard", "results": []}
        try:
            oks = self._render_cube_gpu(cube_path, image_paths, out_paths, long_edge)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error_code": "gpu_lut_failed", "error": str(e)[:200],
                    "results": []}
        self._bump("local_images", sum(oks))
        return {"ok": all(oks), "engine": "gpu_lut",
                "results": [{"ok": ok, "out_path": dp, "after_path": dp,
                             "engine": "gpu_lut"} for ok, dp in zip(oks, out_paths)]}

    @staticmethod
    def _maybe_shrink(im, edge: Optional[int] = None):
        """落盘降采样（RENDER_SAVE_SHORT_EDGE>0 时短边压到该值，LANCZOS）。

        2026-07-14 r5 全量生产启用（=1024）：训练侧只用 512p（data/infer_dataset
        resize2_512p），落盘存源图原始分辨率（2048~4000px）纯属空间浪费（磁盘按
        ~20GB/h 增长撑不完 10w 目标）；recipe/源图俱全，需要高分辨率随时可复渲。
        CGT 可经 RENDER_CGT_SHORT_EDGE 单独设更小值（软羽化 α 是平滑场，512 对
        512p 训练无损，PNG 体积是磁盘大头）。默认 0=关闭，bake/校准工具不受影响。"""
        if edge is None:
            edge = int(os.environ.get("RENDER_SAVE_SHORT_EDGE", "0"))
        if edge <= 0:
            return im
        w, h = im.size
        s = min(w, h)
        if s <= edge:
            return im
        from PIL import Image
        r = edge / s
        return im.resize((max(1, round(w * r)), max(1, round(h * r))), Image.LANCZOS)

    @staticmethod
    def _save_rgb_u8(arr: Any, dst: str, quality: int = 92) -> None:
        """Encode one RGB uint8 result. PNG stays lossless; other suffixes use JPEG."""
        from PIL import Image
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        im = RenderBackend._maybe_shrink(Image.fromarray(arr, "RGB"))
        ext = os.path.splitext(dst)[1].lower()
        if ext == ".png":
            im.save(dst, "PNG")
        else:
            im.save(dst, "JPEG", quality=quality)

    @staticmethod
    def _save_alpha_png(arr_u8: Any, dst: str) -> None:
        """单通道 CGT alpha PNG。compress_level=6：编码在锁外线程池，CPU 换体积
        （level=1 的软羽化 α 几乎不压缩，实测是 r5 磁盘增长大头）。"""
        from PIL import Image
        edge = int(os.environ.get("RENDER_CGT_SHORT_EDGE", "0")) or None
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        RenderBackend._maybe_shrink(Image.fromarray(arr_u8, "L"), edge).save(
            dst, "PNG", compress_level=6)

    def _render_local_preset_gpu(self, preset_path: str, fmt: str, source_path: str,
                                 specs: Sequence[dict], out_paths: Sequence[str],
                                 residual_id: str,
                                 cgt_paths: Optional[Sequence[Optional[str]]] = None) -> dict:
        """Decode/upload the source once and render every alpha variant in one tensor call.

        锁只覆盖 GPU 计算段（parse/decode/upload/replay/composite/GPU 端 u8 量化
        /.cpu() 下载）；JPEG 与 CGT PNG 编码写盘在锁外线程池并行（性能项4+5+7）。
        """
        if cgt_paths is None:
            cgt_paths = [None] * len(out_paths)
        arrays = None
        alpha_u8 = None
        with self._gpu_lock:
            import torch
            from gpu_render.gpu.gpu_replay import DEVICE
            from gpu_render.gpu.local_preset import render_local_preset_tensor
            from gpu_render.gpu.render_batch import _decode, _download_u8, _upload
            from gpu_render.local_apply import FITS_DIR

            preset = self._parse_cached(preset_path, fmt)
            source_u8 = _decode(source_path)
            source_t = _upload([source_u8], DEVICE)
            out_t = None
            alpha_t = None
            try:
                with torch.no_grad():
                    out_t, info = render_local_preset_tensor(
                        source_t, preset, list(specs), FITS_DIR,
                        residual_id=residual_id, fallback="cpu")
                    # ``alpha`` is a potentially large CUDA diagnostic tensor. Verify its
                    # batch contract, then remove it from the public result before encoding.
                    alpha_t = info.pop("alpha", None)
                    if alpha_t is not None and int(alpha_t.shape[0]) != len(out_paths):
                        raise RuntimeError(
                            f"local preset alpha batch is {int(alpha_t.shape[0])}, "
                            f"expected {len(out_paths)}")
                    arrays = _download_u8(out_t)
                    if any(cgt_paths):
                        if alpha_t is None:
                            raise RuntimeError(
                                "cgt_path requested but renderer returned no alpha")
                        # 实际合成用的 (N,1,H,W) alpha 在 GPU 上转 u8 再下载；截断
                        # 语义 (α*255).astype(uint8) 与 construct 侧旧 CGT PNG 一致。
                        alpha_u8 = (alpha_t.clamp(0, 1).mul(255.0)
                                    .to(torch.uint8).squeeze(1).cpu().numpy())
                if len(arrays) != len(out_paths):
                    raise RuntimeError(
                        f"local preset renderer returned {len(arrays)} images for "
                        f"{len(out_paths)} variants")
            finally:
                del out_t, alpha_t
                del source_t
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()   # 批末显存礼让（共享卡），保持在锁内

        # 锁外并行编码写盘：JPEG 结果图 + 可选单通道 CGT PNG（性能项4+5）。
        n_jobs = len(out_paths) + sum(1 for c in cgt_paths if c)
        with ThreadPoolExecutor(max_workers=min(8, max(1, n_jobs))) as pool:
            futs = [pool.submit(self._save_rgb_u8, arr, dst)
                    for arr, dst in zip(arrays, out_paths)]
            if alpha_u8 is not None:
                futs += [pool.submit(self._save_alpha_png, alpha_u8[i], cgt)
                         for i, cgt in enumerate(cgt_paths) if cgt]
            for fut in futs:
                fut.result()    # 任一编码失败整批向上抛（由调用方回退农场）
        return info

    def _render_local_preset_farm(self, preset_path: str, fmt: str, source_path: str,
                                  specs: Sequence[dict], out_paths: Sequence[str],
                                  preset_id: Optional[str],
                                  cgt_paths: Optional[Sequence[Optional[str]]] = None
                                  ) -> List[dict]:
        """Farm-render global base once, then exact sRGB alpha-composite every variant."""
        import numpy as np
        from PIL import Image
        from gpu_render.local_replay import raster_alpha

        if cgt_paths is None:
            cgt_paths = [None] * len(out_paths)
        fd, farm_base_path = tempfile.mkstemp(prefix="render_base_", suffix=".jpg")
        os.close(fd)
        try:
            farm_result = self._render_farm_one(
                preset_path, fmt, source_path, farm_base_path)
            if not farm_result.get("ok"):
                return [dict(farm_result, out_path=dst) for dst in out_paths]

            from PIL import ImageOps
            # LR 农场输出是 EXIF 方向已应用的像素；源图必须同样矫正，否则带
            # Orientation 标签的源要么 shape 不匹配整批失败，要么(180°)静默错位合成。
            with Image.open(source_path) as im:
                source_u8 = np.asarray(
                    ImageOps.exif_transpose(im).convert("RGB"), dtype=np.uint8)
            with Image.open(farm_base_path) as im:
                edited_u8 = np.asarray(im.convert("RGB"), dtype=np.uint8)
            if source_u8.shape != edited_u8.shape:
                err = {
                    "ok": False,
                    "error_code": "farm_base_shape_mismatch",
                    "error": (f"source shape {source_u8.shape} != farm base "
                              f"shape {edited_u8.shape}"),
                }
                return [dict(err, out_path=dst) for dst in out_paths]

            h, w = source_u8.shape[:2]
            source_f = source_u8.astype(np.float32)
            edited_f = edited_u8.astype(np.float32)
            results: List[dict] = []
            for spec, dst, cgt in zip(specs, out_paths, cgt_paths):
                try:
                    if str(spec.get("mask_type", "")) == "semantic":
                        # 语义 alpha：低分辨率羽化 mask 双线性放大到源图分辨率
                        # （对齐 local_replay.corr_alpha / GPU 路 raster_cgt_batch）。
                        alpha = np.asarray(spec["alpha"], np.float32)
                        if alpha.shape != (h, w):
                            try:
                                import cv2
                                alpha = cv2.resize(
                                    alpha, (w, h), interpolation=cv2.INTER_LINEAR)
                            except ImportError:
                                alpha = np.asarray(
                                    Image.fromarray(alpha, mode="F").resize(
                                        (w, h), Image.BILINEAR), np.float32)
                    else:
                        alpha = raster_alpha(
                            spec["mask_type"], spec["geom"], h, w, smoothstep=True)
                    alpha = np.clip(alpha * float(spec.get("amount", 1.0)), 0.0, 1.0)
                    a3 = alpha[..., None]
                    mixed = source_f * (1.0 - a3) + edited_f * a3
                    out_u8 = np.clip(mixed + 0.5, 0, 255).astype(np.uint8)
                    # Make the alpha=0 invariant explicit before encoding (lossless for PNG).
                    zero = alpha <= 0.0
                    out_u8[zero] = source_u8[zero]
                    self._save_rgb_u8(out_u8, dst)
                    if cgt:
                        # 与 GPU 路同一 u8 截断语义：(α*255) 向零取整。
                        self._save_alpha_png((alpha * 255.0).astype(np.uint8), cgt)
                    results.append({
                        "ok": True, "out_path": dst, "after_path": dst,
                        "cgt_path": cgt,
                        "engine": "farm_local_composite", "preset_id": preset_id,
                        "farm_engine": farm_result.get("engine", "lrc"),
                    })
                except Exception as exc:  # one bad encode must not discard other variants
                    results.append({
                        "ok": False, "out_path": dst,
                        "error_code": "local_composite_failed", "error": str(exc),
                    })
            return results
        finally:
            try:
                os.remove(farm_base_path)
            except OSError:
                pass

    # -- 农场路 ----------------------------------------------------------------
    def _render_farm_one(self, preset_path: str, fmt: str, src: str, dst: Optional[str]) -> dict:
        from dataset_build.source_qa import lr_render
        r = lr_render.render_via_lr(preset_path, fmt, src)
        if r.get("ok") and r.get("after_path") and dst:
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            shutil.move(r["after_path"], dst)
            r["after_path"] = dst
            if int(os.environ.get("RENDER_SAVE_SHORT_EDGE", "0")) > 0:
                try:  # 农场回传为全分辨率 JPEG：与本地路同规则落盘降采样
                    from PIL import Image
                    im = Image.open(dst); im.load()
                    small = self._maybe_shrink(im)
                    if small is not im:
                        small.save(dst, "JPEG", quality=92)
                except Exception:  # noqa: BLE001 - 降采样失败保留原图，不丢渲染
                    pass
        return r

    # -- 统一入口 ---------------------------------------------------------------
    def render(self, preset_path: str, fmt: str, image_paths: Sequence[str],
               out_paths: Sequence[str], preset_id: Optional[str] = None) -> dict:
        """同一 preset 批量渲 N 张（本地 batch=16 / 农场并发提交），按输入序返回。

        results[i] = {ok, out_path, engine} 或 {ok:False, error_code, ...}。"""
        assert len(image_paths) == len(out_paths), "image_paths 与 out_paths 数量须一致"
        n = len(image_paths)
        fmt = _norm_fmt(preset_path, fmt)
        pid = resolve_preset_id(preset_path, preset_id)
        results: List[Optional[dict]] = [None] * n
        if fmt not in _FARM_FMTS:
            err = {"ok": False, "error_code": "unsupported_fmt",
                   "error": f"render_backend 只吃 xmp/lrtemplate，收到 {fmt!r}（lut 请走调用方本地 trilinear）"}
            self._bump("failed", n)
            return {"ok": False, "route": "none", "results": [dict(err) for _ in range(n)],
                    "n_local": 0, "n_farm": 0, "stats": self.stats_snapshot()}

        # 分流优先级：专属残差 > 烘焙 LUT（仅 throughput；fidelity 契约=只放行专属
        # 残差，与 render_local_variants 一致）> 键覆盖良好走 _global 残差 > 农场兜底。
        go_local = has_residual(pid)
        local_res_id = pid
        baked = None if (go_local or self.policy != "throughput") else baked_lut_path(pid)
        if baked:
            free = _gpu1_free_mb()
            if free < LOCAL_MIN_FREE_MB:
                print(f"[render_backend] cuda:{_RENDER_GPU_IDX} 空闲 {free}MB < "
                      f"{LOCAL_MIN_FREE_MB}MB，烘焙路({pid})跳过", file=sys.stderr)
                self._bump("local_skip_vram", n)
                baked = None
        if baked:
            try:
                oks = self._render_cube_gpu(baked, image_paths, out_paths)
            except Exception as e:  # noqa: BLE001 - 烘焙路失败 → 继续常规分流
                print(f"[render_backend] baked LUT 渲染失败({pid})，回退常规分流: {e}",
                      file=sys.stderr)
                self._bump("local_fallback_farm", n)
            else:
                for i, ok in enumerate(oks):
                    if ok:
                        results[i] = {"ok": True, "out_path": out_paths[i],
                                      "after_path": out_paths[i],
                                      "engine": "gpu_baked_lut", "preset_id": pid}
                self._bump("local_images", sum(oks))
                farm_idx = [i for i, ok in enumerate(oks) if not ok]
                if not farm_idx:
                    return {"ok": True, "route": "local", "results": results,
                            "n_local": n, "n_farm": 0, "stats": self.stats_snapshot()}
                # 个别失败张继续走农场兜底
                self._bump("local_fallback_farm", len(farm_idx))
                n_farm_ok = 0
                def _one_b(i: int) -> tuple:
                    return i, self._render_farm_one(preset_path, fmt, image_paths[i], out_paths[i])
                with ThreadPoolExecutor(max_workers=min(self.farm_workers, len(farm_idx))) as ex:
                    for i, r in ex.map(_one_b, farm_idx):
                        if r.get("ok"):
                            r.setdefault("engine", "lrc")
                            r["out_path"] = out_paths[i]
                            n_farm_ok += 1
                        results[i] = r
                self._bump("farm_images", n_farm_ok)
                self._bump("failed", len(farm_idx) - n_farm_ok)
                return {"ok": all(r and r.get("ok") for r in results),
                        "route": "local+farm_fallback", "results": results,
                        "n_local": n - len(farm_idx), "n_farm": n_farm_ok,
                        "stats": self.stats_snapshot()}
        if not go_local and self.policy == "throughput" and self._local_capable(preset_path, fmt):
            go_local = True
            local_res_id = "_global"     # 未标定 preset 用全局残差 LUT 兜底
        if go_local:
            free = _gpu1_free_mb()
            if free < LOCAL_MIN_FREE_MB:
                print(f"[render_backend] cuda:{_RENDER_GPU_IDX} 空闲 {free}MB < {LOCAL_MIN_FREE_MB}MB，"
                      f"本批({pid})直接走农场", file=sys.stderr)
                self._bump("local_skip_vram", n)
                go_local = False
        farm_idx: List[int] = list(range(n))
        n_local = 0

        if go_local:
            jobs = [(image_paths[i], out_paths[i]) for i in range(n)]
            try:
                oks = self._render_local(preset_path, fmt, jobs, local_res_id)  # type: ignore[arg-type]
            except Exception as e:  # noqa: BLE001 - 本地整批失败 → 全量回退农场
                print(f"[render_backend] 本地 GPU 渲染失败({pid})，整批回退农场: {e}",
                      file=sys.stderr)
                oks = [False] * n
            farm_idx = [i for i, ok in enumerate(oks) if not ok]
            for i, ok in enumerate(oks):
                if ok:
                    results[i] = {"ok": True, "out_path": out_paths[i],
                                  "after_path": out_paths[i], "engine": "gpu_local",
                                  "preset_id": pid}
            n_local = n - len(farm_idx)
            self._bump("local_images", n_local)
            if farm_idx:
                self._bump("local_fallback_farm", len(farm_idx))

        n_farm_ok = 0
        if farm_idx:
            def _one(i: int) -> tuple:
                r = self._render_farm_one(preset_path, fmt, image_paths[i], out_paths[i])
                return i, r
            with ThreadPoolExecutor(max_workers=min(self.farm_workers, len(farm_idx))) as ex:
                for i, r in ex.map(_one, farm_idx):
                    if r.get("ok"):
                        r.setdefault("engine", "lrc")
                        r["out_path"] = out_paths[i]
                        n_farm_ok += 1
                    results[i] = r
            self._bump("farm_images", n_farm_ok)
            self._bump("failed", len(farm_idx) - n_farm_ok)

        route = ("local" if go_local and not farm_idx else
                 "farm" if not go_local else "local+farm_fallback")
        return {"ok": all(r and r.get("ok") for r in results), "route": route,
                "results": results, "n_local": n_local, "n_farm": n_farm_ok,
                "stats": self.stats_snapshot()}

    def render_local_variants(self, base_preset_path: str, fmt: str,
                              source_path: str, variants: Sequence[dict],
                              preset_id: Optional[str] = None) -> dict:
        """Apply one *global* base preset inside N geometric or semantic masks.

        ``variants`` preserves input order and each row is::

            {"out_path": "/path/to/out.jpg",
             "cgt_path": "/path/to/cgt.png",   # 可选：后端写实际合成 alpha 的单通道 PNG
             "spec": {"mask_type": "circulargradient" | "gradient",
                      "geom": {...}, "amount": 1.0}
                  或 {"mask_type": "semantic",
                      "alpha": np.ndarray (H,W) float [0,1]（可为降采样，双线性放大），
                      "amount": 1.0}}

        A capable base preset is replayed once on GPU and broadcast across all alpha masks.
        Otherwise Lightroom renders that base globally once and CPU performs the same exact
        sRGB alpha composite for every variant. Base presets which already contain local masks
        are rejected because wrapping those corrections in another mask would double-localize
        their effect. 结果 row 带 ``cgt_path``（未请求时为 None）。
        """
        import math

        try:
            rows = list(variants)
        except TypeError:
            rows = []
        n = len(rows)
        fmt = _norm_fmt(base_preset_path, fmt)
        pid = resolve_preset_id(base_preset_path, preset_id)

        def _batch_error(code: str, message: str) -> dict:
            self._bump("failed", n)
            errs = []
            for row in rows:
                dst = row.get("out_path") if isinstance(row, dict) else None
                errs.append({"ok": False, "out_path": dst,
                             "error_code": code, "error": message})
            return {"ok": False, "route": "none", "results": errs,
                    "n_local": 0, "n_farm": 0, "stats": self.stats_snapshot()}

        if not rows:
            return {"ok": True, "route": "none", "results": [],
                    "n_local": 0, "n_farm": 0, "stats": self.stats_snapshot()}
        if fmt not in _FARM_FMTS:
            return _batch_error(
                "unsupported_fmt",
                f"render_backend only accepts xmp/lrtemplate, got {fmt!r}")
        if not os.path.isfile(base_preset_path):
            return _batch_error("base_preset_missing", base_preset_path)
        if not os.path.isfile(source_path):
            return _batch_error("source_missing", source_path)

        specs: List[dict] = []
        out_paths: List[str] = []
        cgt_paths: List[Optional[str]] = []
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                return _batch_error("invalid_variant", f"variants[{i}] must be a dict")
            dst = row.get("out_path")
            spec = row.get("spec")
            cgt = row.get("cgt_path")
            if not isinstance(dst, (str, os.PathLike)) or not os.fspath(dst):
                return _batch_error(
                    "invalid_variant", f"variants[{i}].out_path is required")
            if not isinstance(spec, dict):
                return _batch_error(
                    "invalid_variant", f"variants[{i}].spec must be a dict")
            if cgt is not None and (not isinstance(cgt, (str, os.PathLike))
                                    or not os.fspath(cgt)):
                return _batch_error(
                    "invalid_variant",
                    f"variants[{i}].cgt_path must be a non-empty path or None")
            try:
                amount = float(spec.get("amount", 1.0))
            except (TypeError, ValueError):
                return _batch_error(
                    "invalid_variant", f"variants[{i}].spec.amount must be numeric")
            if not math.isfinite(amount):
                return _batch_error(
                    "invalid_variant", f"variants[{i}].spec.amount must be finite")
            mask_type = str(spec.get("mask_type", "")).strip().lower()
            if mask_type == "semantic":
                import numpy as np
                alpha = spec.get("alpha")
                if (not isinstance(alpha, np.ndarray) or alpha.ndim != 2
                        or not np.issubdtype(alpha.dtype, np.floating)):
                    return _batch_error(
                        "invalid_variant",
                        f"variants[{i}].spec.alpha must be a 2-D float ndarray")
                # 值域裁剪 [0,1]（契约第 1 节）；float32 供两路直接消费。
                specs.append({"mask_type": "semantic",
                              "alpha": np.clip(alpha.astype(np.float32, copy=False),
                                               0.0, 1.0),
                              "amount": amount})
            elif mask_type in ("circulargradient", "gradient"):
                geom = spec.get("geom")
                if not isinstance(geom, dict) or not geom:
                    return _batch_error(
                        "invalid_variant",
                        f"variants[{i}].spec.geom must be a non-empty dict")
                specs.append({"mask_type": mask_type, "geom": dict(geom),
                              "amount": amount})
            else:
                return _batch_error(
                    "invalid_variant",
                    f"variants[{i}].spec.mask_type is unsupported: {mask_type!r}")
            out_paths.append(os.fspath(dst))
            cgt_paths.append(os.fspath(cgt) if cgt is not None else None)

        try:
            embedded_locals = self._base_has_embedded_locals(base_preset_path)
        except OSError as exc:
            return _batch_error("base_preset_unreadable", str(exc))
        if embedded_locals:
            return _batch_error(
                "base_preset_has_embedded_locals",
                "base preset contains embedded Lightroom local corrections")

        go_local = False
        residual_id: Optional[str] = None
        route_reason = ""
        if has_residual(pid):
            go_local = True
            residual_id = pid
            route_reason = "dedicated residual"
        elif self.policy == "throughput" and self._local_capable(base_preset_path, fmt):
            go_local = True
            residual_id = "_global"
            route_reason = "well-covered base with global residual"
        elif self.policy == "throughput":
            route_reason = "base preset is not locally covered"
        else:
            route_reason = "fidelity policy requires a dedicated residual"

        attempted_local = False
        if go_local:
            free = _gpu1_free_mb()
            if free < LOCAL_MIN_FREE_MB:
                print(
                    f"[render_backend] cuda:{_RENDER_GPU_IDX} free {free}MB < "
                    f"{LOCAL_MIN_FREE_MB}MB; local preset variants ({pid}) use farm",
                    file=sys.stderr)
                self._bump("local_skip_vram", n)
                route_reason = f"GPU VRAM guard ({free}MB free)"
                go_local = False

        if go_local:
            attempted_local = True
            assert residual_id is not None
            try:
                info = self._render_local_preset_gpu(
                    base_preset_path, fmt, source_path, specs, out_paths,
                    residual_id=residual_id, cgt_paths=cgt_paths)
                if isinstance(info, dict):
                    info = dict(info)
                    info.pop("alpha", None)
            except Exception as exc:  # noqa: BLE001 - entire variant set shares one GPU call
                print(
                    f"[render_backend] local preset GPU render failed ({pid}); "
                    f"falling back to one farm base render: {exc}", file=sys.stderr)
                self._bump("local_fallback_farm", n)
                route_reason = f"GPU fallback: {exc}"
            else:
                results = [
                    {"ok": True, "out_path": dst, "after_path": dst,
                     "cgt_path": cgt, "engine": "gpu_local_preset", "preset_id": pid}
                    for dst, cgt in zip(out_paths, cgt_paths)
                ]
                public_info = {
                    key: sorted(value) if isinstance(value, set) else value
                    for key, value in (info or {}).items()
                }
                self._bump("local_images", n)
                return {"ok": True, "route": "local", "route_reason": route_reason,
                        "results": results, "n_local": n, "n_farm": 0,
                        "local_info": public_info,
                        "stats": self.stats_snapshot()}

        try:
            results = self._render_local_preset_farm(
                base_preset_path, fmt, source_path, specs, out_paths, pid,
                cgt_paths=cgt_paths)
        except Exception as exc:  # noqa: BLE001 - normalize farm/network/decode failures
            results = [
                {"ok": False, "out_path": dst,
                 "error_code": "farm_local_render_failed", "error": str(exc)}
                for dst in out_paths
            ]
        n_farm = sum(bool(r.get("ok")) for r in results)
        self._bump("farm_images", n_farm)
        self._bump("failed", n - n_farm)
        route = "local+farm_fallback" if attempted_local else "farm"
        return {"ok": n_farm == n, "route": route, "route_reason": route_reason,
                "results": results, "n_local": 0, "n_farm": n_farm,
                "stats": self.stats_snapshot()}

    def render_one(self, preset_path: str, fmt: str, photo_path: str,
                   out_path: Optional[str] = None, preset_id: Optional[str] = None) -> dict:
        """单张便捷入口；返回与 render_via_lr 同形的 {ok, after_path, engine, ...}。"""
        if out_path is None:
            from dataset_build.source_qa import config as _sqa_cfg
            os.makedirs(_sqa_cfg.RENDER_STAGE, exist_ok=True)
            out_path = os.path.join(_sqa_cfg.RENDER_STAGE, f"rb_{uuid.uuid4().hex[:12]}.jpg")
        res = self.render(preset_path, fmt, [photo_path], [out_path], preset_id=preset_id)
        return res["results"][0]


# --------------------------------------------------------------------------- #
# 模块级单例 + 便捷函数
# --------------------------------------------------------------------------- #
_BACKEND: Optional[RenderBackend] = None
_BACKEND_LOCK = threading.Lock()


def get_backend() -> RenderBackend:
    global _BACKEND
    if _BACKEND is None:
        with _BACKEND_LOCK:
            if _BACKEND is None:
                _BACKEND = RenderBackend()
    return _BACKEND


def render(preset_path: str, fmt: str, image_paths: Sequence[str],
           out_paths: Sequence[str], preset_id: Optional[str] = None) -> dict:
    return get_backend().render(preset_path, fmt, image_paths, out_paths, preset_id=preset_id)


def render_one(preset_path: str, fmt: str, photo_path: str,
               out_path: Optional[str] = None, preset_id: Optional[str] = None) -> dict:
    return get_backend().render_one(preset_path, fmt, photo_path,
                                    out_path=out_path, preset_id=preset_id)


def render_local_variants(base_preset_path: str, fmt: str, source_path: str,
                          variants: Sequence[dict],
                          preset_id: Optional[str] = None) -> dict:
    return get_backend().render_local_variants(
        base_preset_path, fmt, source_path, variants, preset_id=preset_id)
