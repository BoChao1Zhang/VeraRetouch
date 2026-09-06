#!/usr/bin/env python3
# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/archive_assets.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 · stage0 · sprf · v3 数据前置：读 `archive/` 版 indexed_tar 资产。

snapshot_newdata_v3 的 279 个分片里,**275 个**把资产存成 `archive/` 版
indexed_tar（`archive/manifest.json` + `archive/shards/*.tar` +
`archive/indexes/*.idx.jsonl`）,只有 4 个是老的 `assets/` 目录。而
`train_stage0.asset_bytes` 只认两种布局:

    assets/<name>                      （ASSET_MODE = "dir"）
    <shard>/assets.tar + assets*.idx.jsonl（ASSET_MODE = "tar"）

全盘**没有任何** `assets.tar`,所以 `"tar"` 会立刻 die、`"auto"` 会静默退回
`"dir"` 然后在 archive-only 分片上读不到文件。这就是切 v3 的卡点。

本模块补上第三种布局,**不改任何在跑作业冻结的源码**:走
`stage_targets.install_compact_row` 同一条猴补丁路子（D-compact 的先例）,
在 `load_shards()` 之前替换 `T0.asset_bytes`。

设计要点（都是本战役付过学费的）
* **不用 sqlite**:`archive/indexes/catalog.sqlite3` 存在,但 sqlite 句柄跨线程是
  本战役明令禁止的（`READ_FANOUT` 读池是多线程）。这里与 `T0.TarAssets` 一样,
  只读 `.idx.jsonl` 进一个普通 dict + `os.pread`,天然线程安全。
* **逐 (pid, shard) 建 reader**:DataLoader fork 出的 worker 不共享句柄。
* **读前验 tar 头**:抄 `T0.TarAssets.read` 的守卫 —— 索引偏移错了会读出
  "看着像样"的字节并被静默训练,这个坑本仓已经踩过一次。
* **优先 `assets/`**:两种都在的分片（本地 27 个）行为与今天**逐位相同**,
  只有 archive-only 的分片才走新路径。所以本补丁对 v2 口径是恒等的。
* **多 tar**:`archive/manifest.json` 的 `shards` 可能不止一个,索引里每行带
  `shard` 字段,映射到对应的 tar。
"""
from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path, PurePosixPath

_READERS: dict[tuple, "ArchiveAssets"] = {}


class ArchiveAssets:
    """`archive/` 版 indexed_tar 的随机读；接口与 `T0.TarAssets` 对齐。"""

    def __init__(self, shard: Path, t0):
        self.t0 = t0
        self.root = Path(shard) / "archive"
        man_p = self.root / "manifest.json"
        if not man_p.is_file():
            t0.die(f"{shard}: 既没有 assets/ 也没有 archive/manifest.json")
        man = json.loads(man_p.read_text())
        if int(man.get("schema_version", -1)) != 2:
            t0.die(f"{man_p}: schema_version {man.get('schema_version')!r} != 2")
        if str(man.get("status")) != "complete":
            t0.die(f"{man_p}: status = {man.get('status')!r} != 'complete' —— "
                   "半成品归档不给读")
        self.manifest = man
        idx_dir, tar_dir = self.root / "indexes", self.root / "shards"
        idxs = sorted(idx_dir.glob("*.idx.jsonl"))
        if not idxs:
            t0.die(f"{idx_dir}: 没有 *.idx.jsonl")
        self.index: dict[str, tuple[str, int, int, str]] = {}
        for idx in idxs:
            with open(idx, encoding="utf-8") as fh:
                for ln, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if r.get("schema_version") != 2:
                        t0.die(f"{idx}:{ln}: schema_version "
                               f"{r.get('schema_version')!r} != 2")
                    if r["offset"] != r["offset_data"]:
                        t0.die(f"{idx}:{ln}: offset != offset_data，拒绝猜")
                    name = PurePosixPath(str(r["logical_path"])).name
                    rec = (str(r["shard"]), int(r["offset_data"]), int(r["size"]),
                           str(r["member"]))
                    for key in {name, str(r["member"])}:
                        if self.index.setdefault(key, rec) != rec:
                            t0.die(f"{idx}: 索引键 {key!r} 重复且偏移不同")
        self.tars = {p.stem.replace(".tar", ""): p for p in tar_dir.glob("*.tar")}
        for p in tar_dir.glob("*.tar"):
            self.tars[p.name[:-4]] = p
        self._fds: dict[str, int] = {}

    def _fd(self, shard_key: str) -> int:
        fd = self._fds.get(shard_key)
        if fd is None:
            p = self.tars.get(shard_key)
            if p is None:
                self.t0.die(f"{self.root}: 索引指向 tar {shard_key!r}，但 "
                            f"shards/ 里没有它（有 {sorted(self.tars)[:4]}…）")
            fd = self._fds[shard_key] = os.open(str(p), os.O_RDONLY)
        return fd

    def read(self, name: str) -> bytes:
        rec = self.index.get(name)
        if rec is None:
            raise KeyError(f"{name} 不在 {self.root}")
        shard_key, off, size, member = rec
        fd = self._fd(shard_key)

        def once() -> bytes:
            head = os.pread(fd, 512, off - 512)
            if len(head) != 512:
                raise OSError(f"short header read for {name}")
            try:
                info = tarfile.TarInfo.frombuf(head, encoding="utf-8",
                                               errors="strict")
            except Exception as exc:
                self.t0.die(f"{self.root}: {name} 在偏移 {off-512} 处没有合法 tar "
                            f"头（{exc}）—— 索引与归档不匹配")
            if info.name != member or info.size != size:
                self.t0.die(f"{self.root}: 索引说 {name} ({size}B) 在 {off}，"
                            f"归档头却说 {info.name!r} ({info.size}B) —— 拒绝读")
            buf = os.pread(fd, size, off)
            if len(buf) != size:
                raise OSError(f"short read: {len(buf)} of {size}")
            return buf

        return self.t0.with_read_retry(once, f"archive member {name} in {self.root}")


def install(t0_module) -> None:
    """把 `T0.asset_bytes` 换成「先 assets/ 后 archive/」的版本。

    `assets/<name>` 存在时**逐字走原函数**，所以 v2 的 20 个分片行为与今天完全
    一致；只有 archive-only 的分片才走新路径。逐分片判一次并缓存，避免每次读都
    多一个 stat（NFS 上很贵）。
    """
    if getattr(t0_module, "_SPRF_ARCHIVE_INSTALLED", False):
        return
    orig = t0_module.asset_bytes
    layout: dict[tuple, str] = {}

    def asset_bytes_v3(shard: str, name: str) -> bytes:
        key = (os.getpid(), str(shard))
        mode = layout.get(key)
        if mode is None:
            mode = "dir" if (Path(shard) / "assets").is_dir() else "archive"
            layout[key] = mode
        if mode == "dir":
            return orig(shard, name)
        rd = _READERS.get(key)
        if rd is None:
            rd = _READERS[key] = ArchiveAssets(Path(shard), t0_module)
        return rd.read(name)

    t0_module.asset_bytes = asset_bytes_v3
    t0_module._SPRF_ARCHIVE_INSTALLED = True


# --------------------------------------------------------------------------- #
# 逐位断言：同一分片上 assets/ 与 archive/ 必须给出**完全相同**的字节
# --------------------------------------------------------------------------- #
def verify_bitexact(shard: Path, t0, n: int = 0, seed: int = 20260830) -> dict:
    """`shard` 必须同时有 assets/ 与 archive/。返回逐位比对报告。

    这是切 v3 的准入条件：读法换了，字节不能换。任一不符即 die。
    """
    import hashlib
    import random as _r
    shard = Path(shard)
    if not (shard / "assets").is_dir():
        t0.die(f"{shard}: 没有 assets/，无法做 dir-vs-archive 对照")
    rd = ArchiveAssets(shard, t0)
    on_disk = sorted(p.name for p in (shard / "assets").iterdir() if p.is_file())
    in_index = sorted({k for k in rd.index if not k.startswith("/")}
                      & set(on_disk))
    only_disk = sorted(set(on_disk) - set(rd.index))
    only_idx = sorted({PurePosixPath(k).name for k in rd.index} - set(on_disk))
    names = list(in_index)
    if n and n < len(names):
        _r.Random(seed).shuffle(names)
        names = sorted(names[:n])
    mism = []
    for nm in names:
        a = (shard / "assets" / nm).read_bytes()
        b = rd.read(nm)
        if a != b:
            mism.append(dict(name=nm, dir_sha=hashlib.sha256(a).hexdigest()[:16],
                             arc_sha=hashlib.sha256(b).hexdigest()[:16],
                             dir_bytes=len(a), arc_bytes=len(b)))
    rep = dict(shard=str(shard), n_on_disk=len(on_disk),
               n_in_index=len(rd.index), n_compared=len(names),
               only_on_disk=len(only_disk), only_in_index=len(only_idx),
               only_on_disk_examples=only_disk[:5],
               only_in_index_examples=only_idx[:5],
               n_mismatch=len(mism), mismatches=mism[:5],
               bit_exact=(not mism and not only_disk and not only_idx))
    if mism:
        t0.die(f"逐位断言失败：{len(mism)} 个成员的 assets/ 与 archive/ 字节不同 "
               f"{mism[:2]}")
    if only_disk or only_idx:
        t0.die(f"名字集合不一致：只在盘上 {len(only_disk)}、只在索引 "
               f"{len(only_idx)} —— 归档不是该目录的忠实副本")
    return rep
