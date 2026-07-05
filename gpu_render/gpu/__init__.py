"""GPU 算子注册表（torch, cuda:1）。各模块导出 OPS_GPU = {op_name: fn}；
fn(img_batch_bchw_or_bhwc, ctx) -> 同形张量。与 ops_v2 的 numpy 版一一对应、需通过 parity 验证。
"""
from __future__ import annotations
import importlib, pkgutil
REGISTRY: dict = {}
for _m in pkgutil.iter_modules(__path__):
    # [gpu_render 注] parity / render_batch 是评测与生产驱动脚本（不导出 OPS_GPU），
    # 排除出 registry 自动导入；否则 `python -m gpu_render.gpu.render_batch` 触发 runpy 警告。
    if _m.name.startswith("_") or _m.name in ("parity", "render_batch"):
        continue
    try:
        _mod = importlib.import_module(f"{__name__}.{_m.name}")
    except Exception as e:  # noqa: BLE001
        print(f"[gpu] skip {_m.name}: {type(e).__name__}: {e}")
        continue
    REGISTRY.update(getattr(_mod, "OPS_GPU", {}))
