"""veraretouch_sprf.configs —— 带 [paths] 节的 TOML 物化（EPR-052）。

主线包内的配置与 EPR-051 现行配置**逐键相同**，只把写死的绝对路径前缀换成 `${paths.<key>}` 占位，
并在文件头加一个 `[paths]` 节（以 `# --- end [paths] ---` 标记行结束）。
训练/评测入口不直接读这些文件：先 `materialize()` 成不含 [paths] 的普通 TOML（与原 EPR-051 配置除头注释外
逐字节相同，`scripts/check_configs.py` 会核对），再把物化文件交给原训练代码（config sha 冻结、"缺键=停机"
等行为不变）。

覆盖顺序：`[paths]` 默认值 < 环境变量 `VR_PATH_<KEY大写>` < `overrides` 参数。
"""
from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent
END_MARK = "# --- end [paths] ---"
_PH = re.compile(r"\$\{paths\.([a-z_]+)\}")

# 物化时恢复 EPR-051 原文件名（train_sprf_bk5 的 K3 用同目录下的 arm_clut_full.toml / smoke_clut_full.toml 作基线）
LEGACY_NAMES = {
    "decoder/clut_full.toml": "arm_clut_full.toml",
    "decoder/bkfull_adagn_ff_affhead.toml": "bkfull_adagn_ff_affhead.toml",
    "decoder/smoke_clut_full.toml": "smoke_clut_full.toml",
    "vlm/sft_full.toml": "q3vl_sft_s1f_full.toml",
    "vlm/adapt_s2fb.toml": "q3vl_adapt_s2fb.toml",
}


def split_paths(text: str) -> tuple[dict, str]:
    """-> ([paths] 字典, 去掉 [paths] 节后的正文)。没有 [paths] 节则原样返回。"""
    m = re.search(r"^\[paths\]\n", text, re.M)
    if m is None:
        return {}, text
    head, rest = text[: m.start()], text[m.end():]
    if END_MARK not in rest:
        raise ValueError(f"[paths] 节缺少结束标记 {END_MARK!r}")
    block, body = rest.split(END_MARK + "\n", 1)
    paths = tomllib.loads("[paths]\n" + block)["paths"]
    return paths, head + body


def resolve_paths(paths: dict, overrides: dict | None = None) -> dict:
    out = dict(paths)
    for k in list(out):
        env = os.environ.get(f"VR_PATH_{k.upper()}")
        if env:
            out[k] = env
    if overrides:
        for k, v in overrides.items():
            if k not in out:
                raise KeyError(f"覆盖了 [paths] 里不存在的键 {k!r}")
            out[k] = v
    return out


def render(text: str, overrides: dict | None = None) -> str:
    paths, body = split_paths(text)
    paths = resolve_paths(paths, overrides)

    def _sub(m):
        k = m.group(1)
        if k not in paths:
            raise KeyError(f"占位 ${{paths.{k}}} 在 [paths] 中无定义")
        return paths[k]

    body = _PH.sub(_sub, body)
    left = _PH.findall(body)
    if left:
        raise ValueError(f"仍有未解析占位: {left}")
    return body


def materialize(src: Path, out_dir: Path, overrides: dict | None = None,
                legacy_name: bool = True) -> Path:
    src = Path(src)
    rel = str(src.resolve().relative_to(CONFIG_DIR)) if CONFIG_DIR in src.resolve().parents else src.name
    name = LEGACY_NAMES.get(rel, src.name) if legacy_name else src.name
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / name
    dst.write_text(render(src.read_text(), overrides))
    return dst


def materialize_group(group: str, out_dir: Path, overrides: dict | None = None) -> dict[str, Path]:
    """把 configs/<group>/*.toml 全部物化到 out_dir（同组文件互为 K3/A3 基线，必须同时在场）。"""
    res = {}
    for f in sorted((CONFIG_DIR / group).glob("*.toml")):
        res[f.name] = materialize(f, out_dir, overrides)
    return res


def load(src: Path, overrides: dict | None = None) -> dict:
    """直接读成字典（不落盘）。"""
    return tomllib.loads(render(Path(src).read_text(), overrides))
