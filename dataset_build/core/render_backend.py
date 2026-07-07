"""统一渲染后端 ``core.render_backend`` —— LR 农场 + 本地 GPU 双路分流。

取代进程内常驻的 teacher renderer（llava_qwen2 教师模型，见 render_worker.py 的
历史注释）：build/QA 侧所有 preset 渲染统一走本模块，teacher 模型加载路径不再触发。

分流规则（保真第一）：
  * preset_id 在 ``gpu_render/fits/residual/`` 有 **专属** 残差 LUT（100 个标定
    preset）→ 本地 GPU 渲染（gpu_render.render_files，batch=16，cuda:1）；
  * 其余（未标定 / mask / lut 无法本地保真的）→ LR 农场
    （source_qa.lr_render.render_via_lr，提交端并发 gate 由 lr_render._FARM_GATE 统一管）。
  * 本地渲染失败 → 逐张回退农场（统计 local_fallback_farm）。

约束：GPU 只用 cuda:1（MONETGPT_TORCH_DEVICE=cuda:1；卡 0 留给他人 + IAA），
batch=16；GPU 路径进程内单例 + 互斥锁（gpu_render 的 replay 链非线程安全）。

接口：
  render(preset_path, fmt, image_paths, out_paths, preset_id=None) -> dict
      同一 preset 批量渲多张；返回 {ok, route, results[...], n_local, n_farm, stats}。
  render_one(preset_path, fmt, photo_path, out_path=None, preset_id=None) -> dict
      单张便捷入口；返回与 lr_render.render_via_lr 同形的
      {ok, after_path, engine, ...} / {ok:False, error_code, ...}。
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
#: 标定 preset 清单（preset_id <-> path 映射；100 条与 RES_DIR 一一对应）。
CALIB_PRESETS_JSONL = os.environ.get(
    "RENDER_BACKEND_CALIB_JSONL",
    "/home/bc/data/datasets/lr_calib/preset_test/presets.jsonl")

GPU_BATCH = 16          # 本地 GPU 批大小（显存 ~16.3GB @ cuda:1，实测约束）
FARM_WORKERS = 6        # 农场提交端并发（真正的农场准入在 lr_render._FARM_GATE）
# 本地渲染显存守卫：渲染卡空闲低于此值整批直接走农场，不去分配（避免 OOM churn
# 且不在共卡进程如 vLLM/IAA 旁留 CUDA 上下文）；余量恢复后自动放行本地路。
LOCAL_MIN_FREE_MB = int(os.environ.get("RENDER_LOCAL_MIN_FREE_MB", "18000"))
# 守卫查的 GPU 跟随渲染设备（MONETGPT_TORCH_DEVICE，上面已 setdefault cuda:1）
_RENDER_GPU_IDX = os.environ.get("MONETGPT_TORCH_DEVICE", "cuda:1").rsplit(":", 1)[-1]

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
        #   fidelity（默认）——只有专属残差 LUT 的标定 preset 走本地，其余农场（LR 保真优先）；
        #   throughput ——键覆盖良好的 param preset 也走本地（_global 残差兜底），
        #     仅 mask/未覆盖键/exotic profile 走农场。全量 databuild 用它：
        #     构建的自洽性（recipe 由本管线定义与执行）优先于对 Adobe 的逐像素保真，
        #     且农场 77/min 撑不住 10 万级渲染。
        self.policy = policy or os.environ.get("RENDER_BACKEND_POLICY", "fidelity")
        # gpu_render 的 replay 链（含缓存的 fits/残差）非线程安全 → 全局互斥。
        self._gpu_lock = threading.Lock()
        self._preset_cache: Dict[tuple, dict] = {}   # (realpath, mtime, fmt) -> parse_preset()
        self._route_cache: Dict[str, bool] = {}      # realpath -> throughput 模式下可本地渲
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
    def _parse_cached(self, preset_path: str, fmt: str) -> dict:
        from gpu_render.replay import parse_preset
        rp = os.path.realpath(preset_path)
        key = (rp, os.path.getmtime(rp), fmt)
        pre = self._preset_cache.get(key)
        if pre is None:
            pre = parse_preset(rp, fmt)
            if len(self._preset_cache) > 256:   # 防无界增长
                self._preset_cache.clear()
            self._preset_cache[key] = pre
        return pre

    def _local_capable(self, preset_path: str, fmt: str) -> bool:
        """throughput 策略：用 gpu_render.route 的启发式判 preset 是否可本地渲
        （无 local-mask / 无未覆盖键 / 非 exotic profile）。"""
        rp = os.path.realpath(preset_path)
        hit = self._route_cache.get(rp)
        if hit is None:
            try:
                from gpu_render.route import route_preset
                hit = route_preset(rp, fmt).get("route") == "local"
            except Exception:
                hit = False
            self._route_cache[rp] = hit
        return hit

    def _render_local(self, preset_path: str, fmt: str, jobs: List[tuple],
                      preset_id: str) -> List[bool]:
        """整批本地渲染；返回逐 job 是否产出。异常向上抛（由 render() 回退农场）。"""
        from gpu_render.gpu.render_batch import render_files
        with self._gpu_lock:                     # cuda:1 单租户串行
            preset = self._parse_cached(preset_path, fmt)
            render_files(preset, jobs, batch=self.gpu_batch, residual_id=preset_id)
        return [os.path.exists(dst) and os.path.getsize(dst) > 0 for _, dst in jobs]

    # -- 农场路 ----------------------------------------------------------------
    def _render_farm_one(self, preset_path: str, fmt: str, src: str, dst: Optional[str]) -> dict:
        from dataset_build.source_qa import lr_render
        r = lr_render.render_via_lr(preset_path, fmt, src)
        if r.get("ok") and r.get("after_path") and dst:
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            shutil.move(r["after_path"], dst)
            r["after_path"] = dst
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

        go_local = has_residual(pid)
        local_res_id = pid
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
