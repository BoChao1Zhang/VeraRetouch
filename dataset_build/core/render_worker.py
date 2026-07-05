"""core 层渲染 worker —— teacher renderer 已废弃，旧接口代理到 render_backend。

历史：``RenderClient`` 曾持有进程内常驻的 GPU teacher renderer
（llava_qwen2，configs/infer_config.yaml，bf16 ~1.1B）并用 render_lock 串行。
现在渲染统一走 ``core.render_backend``（LR 农场 + 本地 gpu_render 双路），
teacher 模型加载路径整体不再触发（显存归零）。

兼容性：``RenderClient`` 的构造与 ``render_batch`` / ``render_one`` 签名保持
不变（返回 np.uint8 HxWx3 RGB 或 None，输入序）。参数字典（38 键 CRS）由
``render_backend.params_to_xmp`` 落成临时 XMP —— 这类临时参数没有残差 LUT，
天然分流到 LR 农场（保真第一）。传入的 ``renderer`` 一律忽略并告警：调用方
不应再加载 teacher 模型。

``downscale_rgb`` 仍是唯一规范的长边降采样实现，字节级不变。
"""

from __future__ import annotations

import os
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

from . import render_backend


def downscale_rgb(arr: Any, longedge: int) -> Any:
    """Long-edge downscale of an RGB uint8 array (best-effort; returns input on
    failure or if already small). Canonical impl — keep byte-identical to the
    historical ``streams.Stream._downscale_rgb``."""
    try:
        import cv2
        import numpy as np

        a = np.asarray(arr)
        h, w = a.shape[:2]
        m = max(h, w)
        if not longedge or m <= longedge:
            return a
        s = longedge / float(m)
        return cv2.resize(a, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                          interpolation=cv2.INTER_AREA)
    except Exception:
        return arr


_WARNED_TEACHER = False


def _warn_teacher_deprecated() -> None:
    global _WARNED_TEACHER
    if not _WARNED_TEACHER:
        _WARNED_TEACHER = True
        print("[core.render] teacher renderer 已废弃且被忽略——渲染改走 "
              "render_backend（LR 农场 + 本地 gpu_render）。请不要再加载 teacher 模型。",
              file=sys.stderr)


class RenderClient:
    """旧 teacher 接口的代理：签名不变，内部路由到 ``core.render_backend``。"""

    def __init__(
        self,
        renderer: Optional[Any] = None,
        render_kw: Optional[Dict[str, int]] = None,
        render_lock: Optional[Any] = None,
        gpu: Optional[Any] = None,
        device: str = "cuda:0",
        workers: int = 6,
    ) -> None:
        if renderer is not None:
            _warn_teacher_deprecated()
        self.renderer = None                 # teacher 永久移除；保留属性名做兼容
        self.render_kw = dict(render_kw or {})   # 兼容保留（新后端不消费）
        self.render_lock = render_lock if render_lock is not None else threading.Lock()
        self.gpu = gpu                       # 兼容保留：新后端自持 cuda:1 互斥，不再用 lease
        self.device = device
        self._workers = max(1, int(workers))
        self._backend = render_backend.get_backend()

    # -- 单条参数字典 -> 临时 XMP -> 后端（无残差 → 农场） --------------------
    def _render_params_one(self, path: str, params: Dict[str, Any],
                           log_prefix: str) -> Optional[Any]:
        from dataset_build.source_qa import config as _sqa_cfg
        os.makedirs(_sqa_cfg.RENDER_STAGE, exist_ok=True)
        xmp = os.path.join(_sqa_cfg.RENDER_STAGE, f"teacherp_{uuid.uuid4().hex[:12]}.xmp")
        after: Optional[str] = None
        try:
            render_backend.params_to_xmp(params, xmp)
            r = self._backend.render_one(xmp, "xmp", path)
            if not r.get("ok") or not r.get("after_path"):
                print(f"[{log_prefix}] render failed for {path}: "
                      f"{r.get('error_code')}: {r.get('error')}", file=sys.stderr)
                return None
            after = r["after_path"]
            import numpy as np
            from PIL import Image, ImageFile
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            return np.asarray(Image.open(after).convert("RGB"))
        except Exception as e:  # noqa: BLE001 - 单样本失败不拖垮整批
            print(f"[{log_prefix}] render failed for {path}: {e}", file=sys.stderr)
            return None
        finally:
            for p in (xmp, after):           # 旧接口只返回数组，不留中间文件
                if p:
                    try:
                        os.remove(p)
                    except OSError:
                        pass

    def render_batch(
        self,
        paths: Sequence[str],
        params_list: Sequence[Optional[Dict[str, Dict[str, float]]]],
        downscale_longedge: int,
        log_prefix: str = "render",
    ) -> List[Optional[Any]]:
        """批量渲染：每条参数字典独立成 preset，经后端并发提交（农场端有全局
        gate）。返回 np.uint8 HxWx3 RGB（或 None），输入序；params=None 的槽位
        原样返回 None —— 与 teacher 时代契约一致。"""
        n = len(paths)
        results: List[Optional[Any]] = [None] * n
        idxs = [i for i, p in enumerate(params_list) if p is not None]
        if not idxs:
            return results
        cap = int(downscale_longedge)

        def _one(i: int):
            out = self._render_params_one(paths[i], params_list[i], log_prefix)  # type: ignore[arg-type]
            return i, (downscale_rgb(out, cap) if out is not None else None)

        with ThreadPoolExecutor(max_workers=min(self._workers, len(idxs))) as ex:
            for i, out in ex.map(_one, idxs):
                results[i] = out
        return results

    def render_one(
        self,
        path: str,
        params: Dict[str, Dict[str, float]],
        log_prefix: str = "render",
    ) -> Optional[Any]:
        """渲一张全局 'after'（不降采样）。失败返回 None。"""
        return self._render_params_one(path, params, log_prefix)
