#!/usr/bin/env python3
# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/train_sprf_xl.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 · stage0 · sprf · C-LUT-XL 入口（v3 数据 + 300k 固定子集）。

**薄包装,不碰被冻结的源文件。** `train_sprf.py` 此刻被在跑的 `SPRF_CLUT_S1`
sha 冻结（「进程启动后禁改源码」），所以 v3 需要的两处改动都在这里以猴补丁装上，
装完直接转调 `train_sprf.main()`：

  1. `archive_assets.install(T0)`
     补上第三种资产布局 `archive/`（v3 的 279 个分片里 275 个是它）。
     `assets/` 在的分片**逐字走原函数**，v2 口径完全不变。
  2. `install_subset(T0, ...)`
     TRAIN 过滤到**预注册的 300k 固定子集**；HELD-OUT 收窄到**冻结的 v3 id 表**
     （快照排除了新分片里落 held-out 桶的 31,011 条，而 load_shards 不排除 ——
     不管的话评测集会从 4,560 涨到 ~35,571，headline 就与 C-LUT 不可比了）。

两个补丁都在 `train_sprf.main()` 之前装好，也就在 `load_shards()` 与 DataLoader
fork 之前 —— 与 `stage_targets.install_compact_row` 同一条路子（D-compact 先例）。

运行时断言（装不上 / 数字对不上就停机，不静默降级）
  X1  keys_file 的 sha256 == `[subset] sha256`（键表没被换过）
  X2  键表长度 == `[subset] n`
  X3  过滤后 TRAIN 样本数 **恰好** == n（不是「<=n」——少了说明键表与本次
      分片集合不匹配，那是口径漂移，必须停机）
  X4a held-out id 表 sha256 == 配置（冻结名单没被换）
  X4b 保留的 held-out 全部在冻结名单内
  X4c held-out 样本数 == 预期（正式臂 4,560；对不上即停机，评测集不同则不可比）
  X5  `archive_assets` 补丁确实装上了

用法
  PYTHONPATH=/home/bc/VeraRetouch python train_sprf_xl.py --config configs/arm_clut_xl.toml
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


def die(msg: str):
    raise SystemExit(f"train_sprf_xl: {msg}")


def install_subset(t0_module, keys_file: str, want_sha: str, want_n: int,
                   ho_file: str, ho_sha: str, ho_expect: int) -> dict:
    """TRAIN 过滤到预注册 300k；HELD-OUT 过滤到**冻结的 v3 held-out id 表**。

    held-out 那一半是本次差点踩的坑：v3 快照**主动排除**了新分片里落在 held-out
    桶的行（`n_new_shard_heldout_bucket_samples_excluded = 31011`），而
    `train_stage0.load_shards` **不做**这个排除 —— 它会把那些行照常判成 held-out。
    不管的话，XL 的 held-out 会从 4,560 涨到 ~35,571：既与 CHAINEND / C-LUT 的
    headline 不可比（评测集都不是同一个），final eval 还要多跑几个小时。
    所以这里按 `snapshot_newdata_v3.heldout_ids.json` 逐 id 收窄。
    """
    if not keys_file:
        # 全量模式：不筛 TRAIN，但 held-out **仍然**按冻结名单收窄。
        # （不收窄的话 load_shards 会把新分片的 held-out 桶行也算进来，
        #  评测集从 4,560 涨到 ~35,571，与 C-LUT / XL 不可比 —— 见 D-xc14。）
        keys, kset = [], None
    else:
        keys = json.loads(Path(keys_file).read_bytes())
        kset = set(keys)
    if kset is not None:
        got_sha = hashlib.sha256("\n".join(keys).encode()).hexdigest()
        if got_sha != want_sha:                                      # X1
            die(f"X1 FAILED: {keys_file} 的 subset sha256 {got_sha[:12]} != "
                f"[subset] sha256 {want_sha[:12]} —— 键表被换过")
        if len(keys) != int(want_n):                                 # X2
            die(f"X2 FAILED: 键表长度 {len(keys)} != [subset] n {want_n}")
    ho_keys = json.loads(Path(ho_file).read_text())
    got_ho = hashlib.sha256("\n".join(ho_keys).encode()).hexdigest()
    if got_ho != ho_sha:                                             # X4a
        die(f"X4a FAILED: held-out id 表 sha256 {got_ho[:12]} != 配置 "
            f"{ho_sha[:12]} —— 冻结的 held-out 名单被换过")
    hoset = set(ho_keys)
    report: dict = {}
    orig = t0_module.load_shards

    def load_shards_subset(cfg):
        samples, blobs, meta = orig(cfg)
        n_tr = sum(1 for s in samples if not s["heldout"])
        n_ho = sum(1 for s in samples if s["heldout"])
        kept, seen, ho_drop = [], set(), 0
        for s in samples:
            if s["heldout"]:
                if s["id"] in hoset:
                    kept.append(s)
                else:
                    ho_drop += 1          # 新分片里落 held-out 桶、但快照已排除的行
                continue
            if kset is None:                     # 全量模式：TRAIN 全留
                kept.append(s)
                continue
            d = s["depth"]
            k = f'{s["id"]}|d{"full" if d is None else int(d)}'
            if k in kset:
                kept.append(s)
                seen.add(k)
        n_tr_after = sum(1 for s in kept if not s["heldout"])
        n_ho_after = sum(1 for s in kept if s["heldout"])
        if n_tr_after != int(want_n):                                # X3
            miss = (len(kset - seen) if kset is not None else 0)
            die(f"X3 FAILED: 过滤后 TRAIN {n_tr_after} != 预注册 n {want_n} "
                f"（键表里有 {miss} 个键没在本次分片集合里找到）—— "
                "键表与分片集合不匹配，口径漂移，停机")
        bad = sorted({s["id"] for s in kept if s["heldout"]} - hoset)
        if bad:                                                      # X4b
            die(f"X4b FAILED: 保留的 held-out 里有不在冻结名单的 id {bad[:3]}")
        if ho_expect and n_ho_after != int(ho_expect):               # X4c
            die(f"X4c FAILED: held-out {n_ho_after} != 预期 {ho_expect} —— "
                "评测集与 CHAINEND / C-LUT 不是同一个，headline 不可比，停机")
        report.update(mode=("full" if kset is None else "subset"),
                      train_before=n_tr, train_after=n_tr_after,
                      heldout_before=n_ho, heldout_after=n_ho_after,
                      heldout_dropped_new_shard_bucket=ho_drop,
                      keys_matched=len(seen),
                      keys_unmatched=(len(kset - seen) if kset is not None else 0))
        print(f"[subset] TRAIN {n_tr} -> {n_tr_after}（预注册 {want_n}）；"
              f"held-out {n_ho} -> {n_ho_after}（按冻结名单收窄，剔除新分片 "
              f"held-out 桶 {ho_drop} 条）", flush=True)
        return kept, blobs, meta

    t0_module.load_shards = load_shards_subset
    return report


def main() -> None:
    args = sys.argv[1:]
    if "--config" not in args:
        die("缺 --config")
    cfg_path = Path(args[args.index("--config") + 1])
    cfg = tomllib.loads(cfg_path.read_text())
    sub = cfg.get("subset")
    if not sub or not sub.get("enabled", False):
        die(f"{cfg_path}: 本入口专供带 [subset] 的 v3 臂；普通臂请直接用 train_sprf.py")

    AA.install(T0)
    if not getattr(T0, "_SPRF_ARCHIVE_INSTALLED", False):            # X5
        die("X5 FAILED: archive_assets 补丁没装上")
    rep = install_subset(T0, ("" if not sub.get("train_subset", True)
                              else sub["keys_file"]), sub["sha256"], sub["n"],
                         sub["heldout_ids_file"], sub["heldout_ids_sha256"],
                         int(sub.get("expect_heldout_samples", 0)))
    print(f"[xl] archive 布局补丁已装；子集规则 {sub['salt']} n={sub['n']} "
          f"sha {sub['sha256'][:12]}", flush=True)

    # 带外记本包装与补丁模块的 sha —— train_sprf 的 frozen_sha256 键表写死在
    # 它自己文件里，而那个文件被在跑作业冻结，不能改（同 D-xc6 的处理）。
    out = Path(cfg["run"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "xl_provenance.json").write_text(json.dumps(dict(
        epr="EPR-051/stage0/sprf/C-LUT-XL", recorded_by="train_sprf_xl.py",
        why="frozen_sha256 的键表写死在 train_sprf.py 里，该文件被在跑作业冻结，"
            "不能回头加键；这些 sha 在此带外补记。",
        sha256={p: hashlib.sha256(_P.src(p).read_bytes()).hexdigest()
                for p in ("train_sprf_xl.py", "archive_assets.py",
                          "edit_cond.py", "stage_flow.py", "stage_solver.py",
                          "stage_targets.py", "train_sprf.py")},
        config=str(cfg_path),
        config_sha256=hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        subset=sub), ensure_ascii=False, indent=1))

    from veraretouch_sprf.train import train_sprf
    train_sprf.main()
    (out / "xl_subset_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
