"""veraretouch_sprf —— SPRF（Stagewise Physical Residual Flow）局部精修主线工程包。

EPR-052 从 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/ 扶正而来：
源文件逐字复制、只改导入与路径解析（见 PROVENANCE.md 逐文件对照与 sha256）。
子包：data / models / solver / train / eval / rl / configs / scripts。
"""
from veraretouch_sprf import _paths as _P

_P.ensure_sys_path()

__all__ = ["_P"]
