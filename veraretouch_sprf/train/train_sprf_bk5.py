#!/usr/bin/env python3
# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/train_sprf_bk5.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 · stage0 · sprf · BK-FULL 入口（ADAGN+FF+AFFHEAD 按 C-LUT-FULL 全量配方；train_sprf_bk4.py 的副本，E1；转调同一 core4）。

K3：与 arm_clut_full.toml 的差集 == {arm.name, run.out_dir} ∪ {bk.arm, bk.layer_checkpoint, bk.gn_groups, bk.n_freq}（A3 逐键相同）。

K3 基线按梯子：adagn_ff vs bk_adagn.toml（只多 bk.n_freq）；adagn_ff_affhead vs bk_adagn_ff.toml（只差 bk.arm）。

差异：BK_ARMS/BK_KEYS 换成两追加臂；转调 train_sprf_bk_core3（import stage_flow_bk3）。

薄包装，不碰共享栈（E1）：`train_sprf.py` / `train_sprf_lossabl*.py` / `stage_flow.py` /
`stage_solver.py` 一字不改。沿用 `train_sprf_lossabl2.py` 的猴补丁与转调模式：

  1. `archive_assets.install(T0)`      —— v3 的 archive/ 资产布局
  2. `XL.install_subset(T0, ...)`      —— TRAIN 收窄到预注册 100k 子集；held-out 按冻结 id 表收窄
  3. 转调 `train_sprf_bk_core.main()`   —— core2 的分支副本（模型换 stage_flow_bk.BkSprfModel，
     求解器换 stage_solver_bk；base 吃整链编辑；A12a 按臂判读；日志加 ms/step 与 reserved）

本入口断言（装不上 / 对不上就停机，不静默降级）
  X1--X5  同 lossabl 入口（键表 sha / 长度 / 过滤后计数 / held-out 名单 sha / archive 补丁）
  K1  [bk] arm 存在且在 BK_ARMS；arm.name == "sprf_bk_<arm>"
  K2  损失全开：[loss] 六个 λ 键全部存在且**全部非零**（与 lossabl FULL 逐键相同）
  K3  config 自动 diff：与 configs/lossabl_full.toml 的差集 ⊆
      {arm.name, run.out_dir, data.num_workers} ∪ {bk.*}；bk.* 必须是本臂声明的键集（多一键少一键都停机）
  K4  sha 钉死：train_sprf.py（原件）/ core2（分支所基于）/ bk_core / stage_flow_bk / stage_solver_bk
      任一漂移即停机；bk_core 相对 core2 的改动只在标记段（结构核对）
  K5  stage_flow.py / stage_solver.py 原件 sha 与 lossabl provenance 记录一致（共享栈未动）

用法
  PYTHONPATH=/home/bc/VeraRetouch python train_sprf_bk.py --config configs/bk_pxfilm.toml
其余命令行参数（--smoke / --limit-samples / --stop-after ...）原样透传。
"""
from __future__ import annotations

import hashlib
import json
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
from veraretouch_sprf import _paths as _P  # EPR-052：替代原 sys.path 注入块
_P.ensure_sys_path()
STAGE0 = _P.STAGE0
REPO = _P.REPO

from veraretouch_sprf.data import train_stage0 as T0        # noqa: E402
from veraretouch_sprf.data import archive_assets as AA      # noqa: E402
from veraretouch_sprf.train import train_sprf_xl as XL       # noqa: E402

LAMBDAS = ("prev", "action", "clean", "bound", "rollout", "q50")
BK_ARMS = ("adagn_ff_affhead",)
# 每臂 [bk] 节必须恰好声明这些键（K3）
BK_KEYS = {
    "pxfilm": {"arm", "layer_checkpoint", "film_gen_hidden"},
    "loramoe": {"arm", "layer_checkpoint", "n_experts", "rank", "lora_alpha", "gate_init_std"},
    "ditblk": {"arm", "layer_checkpoint", "num_heads", "mlp_ratio", "qk_norm", "encdim_ratio",
               "t_init_std"},
    "affhead": {"arm", "layer_checkpoint"},
    "clutflow": {"arm", "layer_checkpoint", "n_luts", "lut_dim", "w_hidden", "f_width"},
    "ff": {"arm", "layer_checkpoint", "n_freq"},
    "canonfilm": {"arm", "layer_checkpoint", "ffn_ratio"},
    "adagn": {"arm", "layer_checkpoint", "gn_groups"},
    "adagn_ff": {"arm", "layer_checkpoint", "gn_groups", "n_freq"},
    "adagn_ff_affhead": {"arm", "layer_checkpoint", "gn_groups", "n_freq"},
}
# 与 FULL 允许不同的非实验语义键（K3）：身份键 + dataloader worker 数（任务卡：≤4，不饿死同卡作业）
# BK-FULL：与 arm_clut_full.toml 逐键相同，只允许 [bk] 节 + arm/run 身份键
ALLOWED_DIFF = {"arm.name", "run.out_dir"}
BASELINE_BY_ARM = {"adagn_ff_affhead": ("arm_clut_full.toml", {"bk.arm", "bk.layer_checkpoint", "bk.gn_groups", "bk.n_freq"})}

# K4 钉死值（E1 sha 冻结；提交作业后不得改动任何一项对应文件）
PINS = {  # EPR-052：主线包内文件按复制件重钉（原钉值见 PROVENANCE.md）；不在包内的历史文件回落 legacy，钉值不变
    "train_sprf.py": "5368c8fdfdbb9e7ede1131e82a7fb68969a7cfd848498ba0f9c88b933fd935cf",
    "train_sprf_lossabl_core2.py": "0127d6c90ce378e58d29a5b5beb7724d03da333b3b597dd4f5982323d992da0c",
    "train_sprf_bk_core.py": "3299e95fbac971e26aed91ed3558844bff6a8496c132ee971a77e550310e68b6",
    "train_sprf_bk_core3.py": "b3f9f1f381c24f283d6a78bef6d6bf5e489ef39fa1b596254f92bd13ca9913d5",
    "train_sprf_bk_core4.py": "eec5c142ff8478b0eae93d7e2ab355ed4e871ccb1a828687aa74b1685941899a",
    "stage_flow_bk.py": "32643506e4440e9fa8554630b4816605627c865b3a63d6b7deceba85f17269ec",
    "stage_flow_bk3.py": "59f848ce57bfc99052d19c5d657e8025ac14754f9ed6ff3d9cab626bdec77eb4",
    "stage_flow_bk4.py": "fdc26d287883e2d380ad0429253aa59e9c112493a1523e07827dd0dd7552c9a0",
    "stage_solver_bk.py": "05cac142bd59b359481749d4fa5f2e1396ed43956348ec9d54f4dbb42646a3ed",
    "stage_flow.py": "484f588d33843f9a23de42b59fe6d2027f1155c7430be252cdc65b4d1e782c70",
    "stage_solver.py": "94dfdcb5b87ee7dab6d415358d3b80d71b5845ebeef71f384fb3a91061155374",
}


def die(msg: str):
    raise SystemExit(f"train_sprf_bk: {msg}")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def check_arm_and_loss(cfg: dict) -> dict:
    bk = cfg.get("bk")
    if not bk or "arm" not in bk:
        die("K1 FAILED: 缺 [bk] arm")
    arm = bk["arm"]
    if arm not in BK_ARMS:
        die(f"K1 FAILED: bk.arm {arm!r} 不在 {BK_ARMS}")
    name = (cfg.get("arm") or {}).get("name", "")
    if name != f"sprf_bkfull_{arm}":
        die(f"K1 FAILED: arm.name {name!r} != 'sprf_bkfull_{arm}'")
    loss = cfg.get("loss") or {}
    missing = [f"w_{n}" for n in LAMBDAS if f"w_{n}" not in loss]
    if missing:
        die(f"K2 FAILED: [loss] 缺 {missing}")
    zeroed = [n for n in LAMBDAS if float(loss[f"w_{n}"]) == 0.0]
    if zeroed:
        die(f"K2 FAILED: 损失全开臂却置零了 {zeroed}")
    return dict(arm=arm, weights={f"w_{n}": float(loss[f"w_{n}"]) for n in LAMBDAS})


def check_config_diff(cfg: dict, cfg_path: Path, arm: str, baseline: Path) -> dict:
    mine = flatten(cfg)
    theirs = flatten(tomllib.loads(baseline.read_text()))
    keys = set(mine) | set(theirs)
    diff = sorted(k for k in keys if mine.get(k, KeyError) != theirs.get(k, KeyError))
    bk_keys = {k for k in mine if k.startswith("bk.")}
    want_bk = {f"bk.{k}" for k in BK_KEYS[arm]}
    if bk_keys != want_bk:
        die(f"K3 FAILED: [bk] 键集 {sorted(bk_keys)} != 本臂声明 {sorted(want_bk)}")
    extra_ok = BASELINE_BY_ARM[arm][1]
    unexpected = [k for k in diff if k not in ALLOWED_DIFF and k not in extra_ok]
    if unexpected:
        die(f"K3 FAILED: 与 {baseline.name} 除 {sorted(ALLOWED_DIFF | extra_ok)} 外还差 {unexpected}")
    if not extra_ok <= set(diff):
        die(f"K3 FAILED: 本级必须恰好多出 {sorted(extra_ok)}，实际差集 {diff}")
    if "data.num_workers" in diff and int(mine["data.num_workers"]) > 4:
        die(f"K3 FAILED: data.num_workers = {mine['data.num_workers']} > 4（任务卡上限）")
    return dict(baseline=str(baseline), baseline_sha256=sha(baseline),
                differing_keys=diff, bk_keys=sorted(bk_keys))


def check_pins(smoke: bool) -> dict:
    got = {p: sha(_P.src(p)) for p in PINS}
    bad = [p for p, want in PINS.items() if want != got[p]]
    if bad:
        if smoke:
            print(f"[bk] K4 钉死值未固化/不一致（冒烟放行）: {bad}", flush=True)
        else:
            die(f"K4 FAILED: sha 与钉死值不一致 {bad} —— E1：分支/原件已漂移，停机")
    # 结构核对：bk_core 相对 core2 的改动必须落在标记段
    import difflib
    o = (_P.src("train_sprf_bk_core.py")).read_text().split("\n")
    b = (_P.src("train_sprf_bk_core4.py")).read_text().split("\n")
    add = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, o, b, autojunk=False).get_opcodes():
        if tag != "equal":
            add += b[j1:j2]
    blob = "\n".join(add)
    for mark in ("stage_flow_bk4", "core4（叠加臂）", "skip_final_eval"):
        if mark not in blob:
            die(f"K4b FAILED: bk_core 缺少标记 {mark!r}")
    return dict(sha256=got, pinned=PINS, added_lines=len(add))


def main() -> None:
    args = sys.argv[1:]
    if "--config" not in args:
        die("缺 --config")
    cfg_path = Path(args[args.index("--config") + 1])
    cfg = tomllib.loads(cfg_path.read_text())
    sub = cfg.get("subset")
    if not sub or not sub.get("enabled", False):
        die(f"{cfg_path}: 本入口专供带 [subset] 的 v3 臂")
    smoke = "--smoke" in args or "--limit-samples" in args

    lam = check_arm_and_loss(cfg)                                     # K1/K2
    baseline = cfg_path.parent / ("smoke_clut_full.toml" if smoke else BASELINE_BY_ARM[lam["arm"]][0])
    diff = check_config_diff(cfg, cfg_path, lam["arm"], baseline)     # K3
    pins = check_pins(smoke)                                          # K4/K4b

    AA.install(T0)
    if not getattr(T0, "_SPRF_ARCHIVE_INSTALLED", False):             # X5
        die("X5 FAILED: archive_assets 补丁没装上")
    rep = XL.install_subset(T0, ("" if not sub.get("train_subset", True)
                                 else sub["keys_file"]), sub["sha256"], sub["n"],
                            sub["heldout_ids_file"], sub["heldout_ids_sha256"],
                            int(sub.get("expect_heldout_samples", 0)))
    print(f"[bk] 臂 {lam['arm']}；损失全开 λ = {lam['weights']}", flush=True)
    print(f"[bk] config diff vs {baseline.name}: {diff['differing_keys']}", flush=True)
    print(f"[bk] 子集 {sub['salt']} n={sub['n']} sha {sub['sha256'][:12]}；"
          f"bk_core sha {pins['sha256']['train_sprf_bk_core.py'][:12]}；"
          f"stage_flow_bk sha {pins['sha256']['stage_flow_bk.py'][:12]}；A12a@step2",
          flush=True)

    out = Path(cfg["run"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "bk_provenance.json").write_text(json.dumps(dict(
        epr="EPR-051/stage0/sprf/BK-FULL", recorded_by="train_sprf_bk5.py",
        arm=lam, config_diff=diff, pins=pins,
        sha256={p: sha(_P.src(p)) for p in (
            "train_sprf_bk5.py", "train_sprf_bk4.py", "train_sprf_bk_core4.py", "stage_flow_bk4.py",
            "train_sprf_bk3.py", "train_sprf_bk_core3.py", "stage_flow_bk3.py",
            "train_sprf_bk.py", "train_sprf_bk_core.py", "stage_flow_bk.py",
            "stage_solver_bk.py", "train_sprf_lossabl_core2.py", "train_sprf_xl.py",
            "archive_assets.py", "edit_cond.py", "stage_flow.py", "stage_solver.py",
            "stage_targets.py", "train_sprf.py")},
        config=str(cfg_path), config_sha256=sha(cfg_path), subset=sub),
        ensure_ascii=False, indent=1))

    from veraretouch_sprf.train import train_sprf_bk_core4 as core
    core.main()
    (out / "bk_subset_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
