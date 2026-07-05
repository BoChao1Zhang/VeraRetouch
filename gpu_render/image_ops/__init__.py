"""image_ops —— vendor 自 monetGPT/image_ops 的算子核（仅渲染路径实际用到的模块）。

- non_gimp_ops.py：~4300 行 numpy 算子核（标定表已烘焙为模块常量）
- non_gimp_ops_torch.py：torch 后端（apply_non_gimp_config 默认后端）
- curve.py：曲线工具（PV2012 解析 / pchip LUT 等）
- image_dehazer/：去雾实现
- operator_spec.py：vendor 自 monetGPT/shared/operator_spec.py（纯 dataclass 常量表）

vendor 裁剪说明：
- shared.repo_config / execution_core 依赖仅存在于 execute_non_gimp_pipeline
 （文件式执行入口，非渲染路径），已 stub 为 NotImplementedError。
- blending.py 未被渲染栈引用，未迁移。
- 原 image_ops/__init__.py 的 try-import 导出改为本说明（避免无谓的重模块导入）。
"""
