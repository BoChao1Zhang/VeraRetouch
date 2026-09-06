#!/usr/bin/env python3
"""EPR-052：把 EPR-051 stage0/sprf 的现行文件复制进主线包 veraretouch_sprf/（只复制，不动原件）。

对每个复制件做且只做四类机械改动（不改数学、不改默认数值）：
  1. 文件顶部加一行来源注释（cot_text.py 例外：逐字节相同，因其 template_sha256() 把自身源码哈希进口径）；
  2. 删除原 sys.path 注入块，改为 `from veraretouch_sprf import _paths as _P` + `_P.ensure_sys_path()`，
     REPO/STAGE0/SPRF 等路径常量改由 _paths 提供；
  3. 同目录裸导入（import stage_flow as SF 等）改为包内绝对导入；
  4. `HERE / "x.py"`（provenance sha 记录 / K4 钉死）改为 `_P.src("x.py")`；train_sprf_bk5.py 的 PINS 表按
     主线包文件重钉（原钉值记入 PROVENANCE.md）。
产出 veraretouch_sprf/PROVENANCE.md：新文件 <- 原文件、原 sha256、新 sha256、改动说明。
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "veraretouch_sprf"
SPRF = REPO / "experiments/prs/EPR-051_masked-restore-production/stage0/sprf"
STAGE0 = SPRF.parent
VLM = SPRF / "vlmsft"
DATE = "2026-09-06"

# (源文件, 目标相对路径, 备注)
FILES = [
    # data
    (SPRF / "stage_targets.py", "data/stage_targets.py", "数据律：链重建/β 场/h*/A 系列断言"),
    (SPRF / "archive_assets.py", "data/archive_assets.py", "v3 archive/ 资产读取补丁"),
    (STAGE0 / "train_stage0.py", "data/train_stage0.py", "数据/切分/评测口径原件（stage0）"),
    (VLM / "build_targets_bkfull.py", "data/build_targets_bkfull.py", "用 BK-FULL edit_enc 预建 e* 目标"),
    (VLM / "cot_text.py", "data/cot_text.py", "指令模板/六段序列化/阶段 token/逐样本指令抽样（逐字节不改）"),
    (VLM / "q3vl_text.py", "data/q3vl_text.py", "prompt+target 编码、span 构造、span_pool"),
    (VLM / "q3vl_data.py", "data/q3vl_data.py", "CoT 快照 Dataset / collate"),
    (VLM / "stage_assets2.py", "data/stage_assets2.py", "y 图暂存（S2 快照）"),
    (VLM / "stage_assets_heldout.py", "data/stage_assets_heldout.py", "y 图暂存（held-out d6）"),
    (VLM / "freeze_cot_snapshot2.py", "data/freeze_cot_snapshot2.py", "S2 快照冻结"),
    # models
    (SPRF / "stage_flow.py", "models/stage_flow.py", "FiLM 主臂：StageConditioner+PointwiseActionNet+SprfModel"),
    (SPRF / "stage_flow_bk.py", "models/stage_flow_bk.py", "BK 分支公共 core/backends（bk4 依赖）"),
    (SPRF / "stage_flow_bk3.py", "models/stage_flow_bk3.py", "ADAGN/CANONFILM backend（bk4 依赖）"),
    (SPRF / "stage_flow_bk4.py", "models/stage_flow_bk4.py", "ADAGN+FF / +AFFHEAD backend（BK-FULL）"),
    (SPRF / "stage_backend_g4d.py", "models/stage_backend_g4d.py", "G4D action backend（stage_flow 懒导入）"),
    (SPRF / "edit_cond.py", "models/edit_cond.py", "逆 LUT 描述子 / 参考对描述子"),
    (SPRF / "bk_load.py", "models/bk_load.py", "BK 模型装载 + 同名注入接口"),
    (SPRF / "align_predictor.py", "models/align_predictor.py", "T-ALIGN 读出预测器基类（build_targets/eval 依赖）"),
    (SPRF / "align_predictor_time.py", "models/align_predictor_time.py", "T-ALIGN 时间 FiLM 预测器（build_target_latents）"),
    (VLM / "q3vl_common.py", "models/vlm/q3vl_common.py", "Qwen3-VL 加载/LoRA/阶段 token/冻结断言"),
    # solver
    (SPRF / "stage_solver.py", "solver/stage_solver.py", "逐阶段 Euler、控制列、rollout 指标（FiLM 主臂）"),
    (SPRF / "stage_solver_bk.py", "solver/stage_solver_bk.py", "BK 分支求解器"),
    # train
    (SPRF / "train_sprf.py", "train/train_sprf.py", "解码器主训练（FiLM，C-LUT-FULL 经 train_sprf_xl 转调）"),
    (SPRF / "train_sprf_xl.py", "train/train_sprf_xl.py", "C-LUT-FULL 入口：install_subset + train_sprf.main"),
    (SPRF / "train_sprf_bk_core4.py", "train/train_sprf_bk_core4.py", "BK 分支训练核心（叠加臂）"),
    (SPRF / "train_sprf_bk5.py", "train/train_sprf_bk5.py", "BK-FULL 入口（K1-K4 断言 + core4.main）"),
    (SPRF / "train_align_time.py", "train/train_align_time.py", "T-ALIGN 训练/评测（eval_vlmadapt 复用其函数）"),
    (VLM / "train_q3vl_sft3.py", "train/train_vlm_sft.py", "Stage-1 全参 SFT（S1F-FULL）"),
    (VLM / "train_q3vl_adapt3.py", "train/train_vlm_adapt.py", "Stage-2 读出+adapter（S2F-B）"),
    # eval
    (SPRF / "batch_eval_bk.py", "eval/eval_decoder.py", "BK 解码器批量评测（B=8）"),
    (VLM / "dump_readout.py", "eval/dump_readout.py", "生成 + 读出（predicted_text/oracle_text）"),
    (VLM / "eval_vlmadapt.py", "eval/eval_vlmadapt.py", "注入执行器评测 + A-inj/A-lat 守卫"),
    (VLM / "guards.py", "eval/guards.py", "G4/G4b eval-only 泄漏断言"),
    (VLM / "probe_full_ckpt.py", "eval/probes/probe_full_ckpt.py", "逐 epoch 32 键字段一致率探针"),
    (VLM / "parity_report.py", "eval/probes/parity_report.py", "batch-1 vs batch-8 分歧统计"),
    (SPRF / "quick100_eval.py", "eval/probes/quick100_eval.py", "MiniCfg / 快评（train_align_time 依赖）"),
    # scripts
    (VLM / "epr051_heldout_split.py", "scripts/heldout_split.py", "held-out 键双卡切分"),
    (VLM / "epr051_merge_gencache.py", "scripts/merge_gencache.py", "生成缓存合并"),
]

# 裸模块名 -> (包路径, 目标 leaf 名)
MODMAP = {
    "stage_targets": ("veraretouch_sprf.data", "stage_targets"),
    "archive_assets": ("veraretouch_sprf.data", "archive_assets"),
    "train_stage0": ("veraretouch_sprf.data", "train_stage0"),
    "build_targets_bkfull": ("veraretouch_sprf.data", "build_targets_bkfull"),
    "cot_text": ("veraretouch_sprf.data", "cot_text"),
    "q3vl_text": ("veraretouch_sprf.data", "q3vl_text"),
    "q3vl_data": ("veraretouch_sprf.data", "q3vl_data"),
    "stage_flow": ("veraretouch_sprf.models", "stage_flow"),
    "stage_flow_bk": ("veraretouch_sprf.models", "stage_flow_bk"),
    "stage_flow_bk3": ("veraretouch_sprf.models", "stage_flow_bk3"),
    "stage_flow_bk4": ("veraretouch_sprf.models", "stage_flow_bk4"),
    "stage_backend_g4d": ("veraretouch_sprf.models", "stage_backend_g4d"),
    "edit_cond": ("veraretouch_sprf.models", "edit_cond"),
    "bk_load": ("veraretouch_sprf.models", "bk_load"),
    "align_predictor": ("veraretouch_sprf.models", "align_predictor"),
    "align_predictor_time": ("veraretouch_sprf.models", "align_predictor_time"),
    "q3vl_common": ("veraretouch_sprf.models.vlm", "q3vl_common"),
    "train_q3vl_adapt": ("veraretouch_sprf.models.vlm", "adapter"),   # dump_readout: from train_q3vl_adapt import Adapter
    "stage_solver": ("veraretouch_sprf.solver", "stage_solver"),
    "stage_solver_bk": ("veraretouch_sprf.solver", "stage_solver_bk"),
    "train_sprf": ("veraretouch_sprf.train", "train_sprf"),
    "train_sprf_xl": ("veraretouch_sprf.train", "train_sprf_xl"),
    "train_sprf_bk_core4": ("veraretouch_sprf.train", "train_sprf_bk_core4"),
    "train_align_time": ("veraretouch_sprf.train", "train_align_time"),
    "guards": ("veraretouch_sprf.eval", "guards"),
    "quick100_eval": ("veraretouch_sprf.eval.probes", "quick100_eval"),
}

DROP_LINES = [
    re.compile(r"^\s*sys\.path\.insert\(0, .*$"),
    re.compile(r"^for _p in \(.*\):$"),
    re.compile(r"^\s+if _p not in sys\.path:$"),
    re.compile(r"^if str\(REPO\) not in sys\.path:$"),
    re.compile(r"^if REPO_ROOT not in sys\.path:$"),
    re.compile(r"^if str\(HERE\) not in sys\.path:$"),
]
REPLACE_LINES = [
    (re.compile(r'^REPO = Path\("/home/bc/VeraRetouch"\)$'), "REPO = _P.REPO"),
    (re.compile(r'^REPO_ROOT = "/home/bc/VeraRetouch"$'), "REPO_ROOT = str(_P.REPO)"),
    (re.compile(r"^STAGE0 = HERE\.parent$"), "STAGE0 = _P.STAGE0"),
    (re.compile(r'^SPRF = Path\("/home/bc/VeraRetouch/experiments/.*/stage0/sprf"\)$'), "SPRF = _P.SPRF_LEGACY"),
    (re.compile(r'^STAGE0 = Path\("/home/bc/VeraRetouch/experiments/.*/stage0"\)$'), "STAGE0 = _P.STAGE0"),
]
IMPORT_AS = re.compile(r"^(\s*)import ([a-z0-9_]+)(?: as ([A-Za-z0-9_]+))?(\s*#.*)?$")
FROM_IMP = re.compile(r"^(\s*)from ([a-z0-9_]+) import (.+)$")
MULTI_AS = re.compile(r"^(\s*)import ((?:[a-z0-9_]+ as [A-Za-z0-9_]+, )+[a-z0-9_]+ as [A-Za-z0-9_]+)\s*$")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rewrite_imports(line: str, notes: list[str]) -> str:
    m = MULTI_AS.match(line)
    if m:
        parts = [x.strip() for x in m.group(2).split(",")]
        out = []
        for part in parts:
            mod, alias = part.split(" as ")
            if mod in MODMAP:
                pkg, leaf = MODMAP[mod]
                out.append(f"{m.group(1)}from {pkg} import {leaf} as {alias}")
            else:
                out.append(f"{m.group(1)}import {part}")
        notes.append("多模块单行 import 拆行")
        return "\n".join(out)
    m = IMPORT_AS.match(line)
    if m and m.group(2) in MODMAP:
        indent, mod, alias, cmt = m.group(1), m.group(2), m.group(3), m.group(4) or ""
        pkg, leaf = MODMAP[mod]
        alias = alias or mod
        tail = f" as {alias}" if alias != leaf else ""
        return f"{indent}from {pkg} import {leaf}{tail}{cmt}"
    m = FROM_IMP.match(line)
    if m and m.group(2) in MODMAP:
        pkg, leaf = MODMAP[m.group(2)]
        return f"{m.group(1)}from {pkg}.{leaf} import {m.group(3)}"
    return line


def transform(src: Path, rel: str, text: str) -> tuple[str, list[str]]:
    notes: list[str] = []
    if src.name == "cot_text.py":
        return text, ["逐字节相同（template_sha256 口径）"]
    lines = text.split("\n")
    out: list[str] = []
    inserted = False
    touched_path = False
    for ln in lines:
        if any(r.match(ln) for r in DROP_LINES):
            if not inserted:
                out.append("from veraretouch_sprf import _paths as _P  # EPR-052：替代原 sys.path 注入块")
                out.append("_P.ensure_sys_path()")
                inserted = True
            touched_path = True
            continue
        rep = None
        for r, new in REPLACE_LINES:
            if r.match(ln):
                rep = new
                break
        if rep is not None:
            if not inserted:
                out.append("from veraretouch_sprf import _paths as _P  # EPR-052：替代原 sys.path 注入块")
                out.append("_P.ensure_sys_path()")
                inserted = True
            touched_path = True
            out.append(rep)
            continue
        new = rewrite_imports(ln, notes)
        if new != ln:
            notes.append("导入改包内绝对导入")
        out.append(new)
    text = "\n".join(out)
    if touched_path:
        notes.append("sys.path 注入块 -> _paths")
    # provenance / K4 的同目录文件引用
    if 'HERE / "configs" / "align_frozen.toml"' in text:
        text = text.replace('HERE / "configs" / "align_frozen.toml"',
                            '_P.SPRF_LEGACY / "configs" / "align_frozen.toml"')
        notes.append("align_frozen.toml 指向 legacy configs")
    if 'HERE / "configs" / (' in text:
        text = text.replace('HERE / "configs" / (', 'cfg_path.parent / (')
        notes.append("K3 基线 config 改为与 --config 同目录（train_decoder 物化目录）")
    n0 = text
    text = re.sub(r'HERE / "([A-Za-z0-9_]+\.py)"', r'_P.src("\1")', text)
    text = re.sub(r'STAGE0 / "train_stage0\.py"', r'_P.src("train_stage0.py")', text)
    text = re.sub(r'\(HERE / ([a-z_]+)\)\.read_bytes\(\)', r'_P.src(\1).read_bytes()', text)
    text = re.sub(r'\bHERE / ([a-z_]+)\b(?! *\))', r'_P.src(\1)', text)
    text = re.sub(r'sha\(HERE / ([a-z_]+)\)', r'sha(_P.src(\1))', text)
    text = re.sub(r'sha256_file\(HERE / ([a-z_]+)\)', r'sha256_file(_P.src(\1))', text)
    if text != n0:
        notes.append('HERE/"x.py" -> _P.src("x.py")')
        if "_P." in text and "import _paths as _P" not in text:
            # 没有路径块的文件也用到了 _P
            text = text.replace("from __future__ import annotations",
                                "from __future__ import annotations\n\nfrom veraretouch_sprf import _paths as _P  # EPR-052", 1)
    if src.name == "q3vl_common.py":
        text = text.replace('MODEL_DIR = "/home/bc/data/models/Qwen3-VL-4B-Instruct"',
                            'MODEL_DIR = os.environ.get("VR_QWEN3VL_DIR", "/home/bc/data/models/Qwen3-VL-4B-Instruct")  # EPR-052：可用环境变量覆盖，默认值不变')
        text = text.replace("import sys\nfrom pathlib import Path", "import os\nimport sys\nfrom pathlib import Path", 1)
        notes.append("MODEL_DIR 加环境变量覆盖（默认值不变）")
    if src.name == "guards.py":
        text = text.replace('EVAL_ONLY_DIR = Path("/home/bc/data/builds/epr051_cot_heldout_eval")',
                            'EVAL_ONLY_DIR = Path(os.environ.get("VR_EVAL_ONLY_DIR", "/home/bc/data/builds/epr051_cot_heldout_eval"))  # EPR-052：可用环境变量覆盖，默认值不变')
        text = text.replace("import json\nfrom pathlib import Path", "import json\nimport os\nfrom pathlib import Path", 1)
        notes.append("EVAL_ONLY_DIR 加环境变量覆盖（默认值不变）")
    # 头注释
    legacy_rel = src.relative_to(REPO)
    header = (f"# 源自 {legacy_rel} @ {DATE}（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），"
              f"逐字复制/仅改导入与路径解析（EPR-052）。")
    if text.startswith("#!"):
        first, rest = text.split("\n", 1)
        text = first + "\n" + header + "\n" + rest
    else:
        text = header + "\n" + text
    return text, sorted(set(notes))


def adapter_excerpt() -> tuple[str, str]:
    src = VLM / "train_q3vl_adapt3.py"
    t = src.read_text()
    a = t.index("class Adapter(nn.Module):")
    b = t.index("def build_targets(")
    c = t.index("def readout_z_and_hidden(")
    d = t.index("def main():")
    body = t[a:b].rstrip() + "\n\n\n" + t[c:d].rstrip() + "\n"
    head = (f"# 源自 {src.relative_to(REPO)} @ {DATE}（节选：class Adapter / readout_z_and_hidden / readout_z，"
            "逐字复制，仅补导入；原 sha256 见 PROVENANCE.md）。EPR-052。\n"
            '"""S2F-B 读出 + adapter 定义（节选自 train_q3vl_adapt3.py；train/train_vlm_adapt.py 内仍保留同一份原定义）。\n\n'
            "Adapter：z_m (2560) -> 128 维 LUT latent，六槽共享权重 + 槽嵌入（零初始化 ⇒ step-0 恒等）。\n"
            "readout_z：Qwen3VLModel.last_hidden_state（已过 final norm）在第 m 段正文 token 上 span mean-pool。\n"
            "dump_readout.py 原从历史文件 train_q3vl_adapt.py 导入 Adapter；两处定义逐字相同（已 diff 核对）。\n"
            '"""\n'
            "from __future__ import annotations\n\nimport torch\nimport torch.nn as nn\n\n"
            "from veraretouch_sprf.models.vlm import q3vl_common as Q\nfrom veraretouch_sprf.data import q3vl_text as T\n\n\n")
    return head + body, sha(src)


def main() -> None:
    rows = []
    for src, rel, note in FILES:
        if not src.exists():
            sys.exit(f"缺原件 {src}")
        dst = PKG / rel
        text = src.read_text()
        new, notes = transform(src, rel, text)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(new)
        rows.append([rel, str(src.relative_to(REPO)), sha(src), None, note, "；".join(notes) or "仅加头注释"])
    ad, ad_sha = adapter_excerpt()
    (PKG / "models/vlm/adapter.py").write_text(ad)
    rows.append(["models/vlm/adapter.py", "experiments/prs/EPR-051_masked-restore-production/stage0/sprf/vlmsft/train_q3vl_adapt3.py",
                 ad_sha, None, "Adapter/readout 定义（节选）", "节选 class Adapter + readout_z_and_hidden + readout_z；补导入"])
    # ---- 重钉 train_sprf_bk5.py 的 PINS（只对主线包内存在的文件；其余名字回落 legacy，钉值不变）----
    bk5 = PKG / "train/train_sprf_bk5.py"
    t = bk5.read_text()
    m = re.search(r"PINS = \{\n(.*?)\n\}\n", t, re.S)
    assert m, "PINS 表未找到"
    pins_old = dict(re.findall(r'"([A-Za-z0-9_]+\.py)": "([0-9a-f]{64})"', m.group(1)))
    sys.path.insert(0, str(REPO))
    from veraretouch_sprf import _paths as _P  # noqa: E402
    repin = {}
    for name, old in pins_old.items():
        p = _P.src(name)
        if PKG in p.parents:
            repin[name] = (old, sha(p))
    block = m.group(1)
    for name, (old, new) in repin.items():
        block = block.replace(f'"{name}": "{old}"', f'"{name}": "{new}"')
    note = ("PINS = {  # EPR-052：主线包内文件按复制件重钉（原钉值见 PROVENANCE.md）；不在包内的历史文件回落 legacy，钉值不变\n")
    t = t.replace("PINS = {\n" + m.group(1) + "\n}\n", note + block + "\n}\n", 1)
    bk5.write_text(t)
    # ---- 新 sha 与 PROVENANCE ----
    for r in rows:
        r[3] = sha(PKG / r[0])
    lines = ["# veraretouch_sprf · PROVENANCE（EPR-052，2026-09-06）", "",
             "原件目录：`experiments/prs/EPR-051_masked-restore-production/stage0/`（**未入 git**，`.gitignore` 的 `experiments/**` 规则；",
             "故对照以原文件 sha256 为准，而非 git sha）。原件只复制不改动；在跑作业（SPRF_ADAPT_S2FB / SPRF_VLMADAPT_PASSKF_GEN）引用的原件未被触碰。", "",
             "改动类别（全部机械、不改数学与默认数值）：头注释 / sys.path 注入块 -> `_paths` / 裸导入 -> 包内绝对导入 / `HERE/\"x.py\"` -> `_P.src(\"x.py\")` / 环境变量覆盖两个默认目录常量。", "",
             "| 新文件 | 原文件 | 原 sha256 | 新 sha256 | 说明 | 改动 |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r[0]}` | `{r[1]}` | `{r[2]}` | `{r[3]}` | {r[4]} | {r[5]} |")
    lines += ["", "## train_sprf_bk5.py K4 PINS 重钉（原钉值 -> 主线包复制件 sha256）", "",
              "| 文件 | 原钉值 | 新钉值 |", "|---|---|---|"]
    for name, (old, new) in repin.items():
        lines.append(f"| `{name}` | `{old}` | `{new}` |")
    kept = [n for n in pins_old if n not in repin]
    lines += ["", f"未重钉（不在主线包内，`_P.src` 回落 legacy 原件，钉值不变）：{', '.join(f'`{k}`' for k in kept)}", ""]
    lines += ["## 未扶正（仍在 legacy 目录，按需 `_P.src` 回落）", "",
              "`train_sprf_bk_core.py`/`core2`/`core3`/`core6`/`core7`、`train_sprf_bk.py`/`bk2`/`bk3`/`bk4`/`bk6`/`bk7`、`stage_flow_bk2.py`/`bk6.py`、",
              "`train_sprf_lossabl*.py`、`train_align.py`、`stage_backend_ft.py`、`passk_*.py`、`bkfull_*ainj*.py`、`memprobe_*.py`、全部 `run_*.sh`。",
              "`eval/eval_decoder.py` 与 `models/bk_load.py` 中对这些历史 core/模型模块的懒导入行保持原样（选到这些臂时会 ImportError，主线四结果不经过它们）。", ""]
    (PKG / "PROVENANCE.md").write_text("\n".join(lines))
    print(f"copied {len(rows)} files; repinned {len(repin)} PINS; kept {len(kept)}")
    # 残留检查：包内不应再有裸的同目录导入 / sys.path 注入
    bad = []
    for f in PKG.rglob("*.py"):
        if f.name in ("sync_from_legacy.py",):
            continue
        for i, ln in enumerate(f.read_text().split("\n"), 1):
            if re.match(r"^\s*(import|from) (" + "|".join(MODMAP) + r")\b", ln):
                bad.append(f"{f.relative_to(PKG)}:{i}: {ln.strip()}")
            if "sys.path.insert" in ln and "_paths" not in str(f):
                bad.append(f"{f.relative_to(PKG)}:{i}: {ln.strip()}")
    print("residual:", len(bad))
    for b in bad:
        print("  ", b)


if __name__ == "__main__":
    main()
