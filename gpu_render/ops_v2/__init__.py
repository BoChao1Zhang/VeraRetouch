"""ops_v2 —— LR-faithful 算子实现注册表（标定 workflow 产物，最终合入 non_gimp_ops.py）。

每个模块文件导出 OPS_V2 = {op_name: fn}；fn(img_float01_rgb, ctx) -> img_float01_rgb，
ctx = {"label": 扫描点标签, "attrs": 该点的 crs 属性 dict, "elements": XMP 子元素串,
       "fit": fits/<op>.json 内容（可为空 dict）}。
注册同名旧算子（如 "Temperature"）即覆盖 non_gimp_ops 路径。自动发现：本包下所有模块。

约束：模块之间不得相互 import（并行开发零冲突）；共享工具从 gpu_render.* 导入。
"""
from __future__ import annotations

import importlib
import os
import pkgutil

REGISTRY: dict = {}
if os.environ.get("OPS_V2_DISABLE") == "1":
    _iter = ()   # 整合入本体后的回归验证：绕过 ops_v2，走 non_gimp_ops 路径
else:
    _iter = pkgutil.iter_modules(__path__)
for _m in _iter:
    if _m.name.startswith("_"):
        continue
    try:
        _mod = importlib.import_module(f"{__name__}.{_m.name}")
    except Exception as e:  # noqa: BLE001 - 单模块坏不拖垮全家
        print(f"[ops_v2] skip {_m.name}: {type(e).__name__}: {e}")
        continue
    REGISTRY.update(getattr(_mod, "OPS_V2", {}))
