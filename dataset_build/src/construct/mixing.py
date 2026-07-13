"""样本混入策略：源图按场景分层配额 + preset 两级风格采样（大类→小类）。

场景配额：9 实类（portrait/landscape/food/still_life/architecture/night/street/
wedding/product）均匀采样；any 桶已由 scene_backfill 细分，残余 any 不参与采样。
某类供给不足时按比例把缺口重分给其余类，保证总量。

preset 采样（2026-07-12 重构，取代 grade_family 配额 + vlemb 召回）：
taxonomy.jsonl 定义 11 个互斥大类 × 75 个小类（大类=vlm_name 主色调规则映射，
小类=大类内 lab_vec KMeans + 审阅修正）。每组只采一个大类（per-source least-used，
保证同源多次采样覆盖不同大类）；组内 k 个候选从该大类的小类轮转取（每小类 1 个，
小类不足 k 则 round-robin 补齐）；小类内 preset 按 least-used 取。无 embedding、
无相关性——多样性与覆盖优先，源图↔风格的适配信号不再由采样承载。

python -m construct.mixing preview   # 对 live DB 预览分层结果
"""
from __future__ import annotations

import hashlib
import json
import sys
import threading
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

# 场景目标比例：PARA 加权、人像/风光领先（2026-07-06 定稿）。any（scene_backfill
# 细分后仍不可归类的图）不参与采样。某类供给不足时 _alloc 按余量重分保总量。
SCENE_TARGETS: Dict[str, float] = {
    "portrait": 16.0, "landscape": 14.0, "food": 12.0,
    "still_life": 10.0, "architecture": 9.0, "night": 9.0,
    "street": 8.0, "wedding": 6.0, "product": 4.0,
}


def _h(*parts: str) -> int:
    """稳定伪随机序（md5），平局打破用；保证 run 可复现且与 asset_id 字典序解耦。"""
    return int.from_bytes(hashlib.md5("|".join(parts).encode()).digest()[:8], "big")


class StyleSampler:
    """两级风格采样器（线程安全）。

    大类：per-source least-used（prior 传入跨 run 历史计数，run 内累加）。
    组内：小类按 run 级使用计数轮转，k 个槽位不足时 round-robin 补齐；
          小类内 preset 按 run 级 least-used 取，farm 路由 ≤ farm_cap_frac。
    """

    def __init__(self, taxonomy_path: str,
                 prior_major: Optional[Dict[str, Dict[str, int]]] = None):
        # tree: major -> minor -> [preset_id]（加载序即 bank 序，稳定）
        self.tree: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.major_of: Dict[str, str] = {}
        self._minor_of: Dict[str, str] = {}
        with open(taxonomy_path) as f:
            for line in f:
                r = json.loads(line)
                self.tree[r["major"]][r["minor"]].append(r["preset_id"])
                self.major_of[r["preset_id"]] = r["major"]
                self._minor_of[r["preset_id"]] = r["minor"]

        self.majors = sorted(self.tree)
        self._prior = prior_major or {}          # source_key -> {major: n}（跨 run）
        self._src_major: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._minor_cnt: Dict[str, int] = defaultdict(int)     # "major/minor" -> n
        self._preset_cnt: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def _minor_name(self, major: str, pid: str) -> str:
        return self._minor_of.get(pid, "")

    def pick_major(self, source_key: str, exclude: Optional[set] = None) -> str:
        """该源 least-used 的大类（跨 run 历史 + run 内计数）；exclude 供采空重试用。"""
        with self._lock:
            hist = self._prior.get(source_key, {})
            run = self._src_major[source_key]
            pool = [m for m in self.majors if m not in (exclude or ())] or self.majors
            return min(pool,
                       key=lambda m: (hist.get(m, 0) + run.get(m, 0), _h(source_key, m)))

    # 组内互斥子池（按小类名前缀）：IAA 系统性偏好黑白>反转（pilot600 实测黑白以 22%
    # 候选占比拿 71% 组内 top1），同组混采则反转款永远陪跑。互斥出组后各子池组内公平，
    # 大类采样份额不变。子池轮次 per-source 计数（含跨 run prior 的组数近似均摊）。
    POOL_SPLIT: Dict[str, str] = {"反转黑白": "黑白·"}

    def sample_group(self, major: str, k: int, source_key: str = "",
                     eligible: Optional[Callable[[str], bool]] = None,
                     is_farm: Optional[Callable[[str], bool]] = None,
                     farm_cap_frac: float = 0.25) -> List[str]:
        """从 major 采一组 ≤k 个 preset：小类轮转→round-robin 补齐→preset least-used。
        eligible 过滤后小类可用量不足 k 时返回实际数量（调用方自行决定弃组与否）。"""
        eligible = eligible or (lambda pid: True)
        prefix = self.POOL_SPLIT.get(major)
        if prefix is not None:
            with self._lock:
                hist = self._prior.get(source_key, {}).get(major, 0)
                nth = hist + self._src_major[source_key][major]
            # 交替 + per-source 随机初相位：多数源只落本大类一次，若固定第 0 轮=主池，
            # 全局子池组占比会趋近 0；随机初相位让两池全局各半、同源仍交替互斥。
            in_sub = ((nth + _h(source_key, major, "pool")) % 2 == 1)
            base_elig = eligible
            eligible = (lambda pid, _b=base_elig, _s=in_sub:
                        _b(pid) and (self._minor_name(major, pid).startswith(prefix) == _s))
        is_farm = is_farm or (lambda pid: False)
        farm_cap = max(1, int(k * farm_cap_frac))
        with self._lock:
            pools = {mn: [p for p in pids if eligible(p)]
                     for mn, pids in self.tree[major].items()}
            pools = {mn: v for mn, v in pools.items() if v}
            if not pools:
                return []
            minors = sorted(pools, key=lambda mn: (self._minor_cnt[f"{major}/{mn}"],
                                                   _h(source_key, major, mn)))
            picked: List[str] = []
            used = set()
            n_farm = 0
            i = 0
            # 轮转小类直到凑满 k；一整轮无新增（全部枯竭/被 farm 上限卡死）则停
            stall = 0
            while len(picked) < k and stall < len(minors):
                mn = minors[i % len(minors)]
                i += 1
                cands = sorted((p for p in pools[mn] if p not in used),
                               key=lambda p: (self._preset_cnt[p], _h(source_key, p)))
                if n_farm >= farm_cap:
                    cands = [p for p in cands if not is_farm(p)]
                if not cands:
                    stall += 1
                    continue
                stall = 0
                p = cands[0]
                picked.append(p)
                used.add(p)
                n_farm += int(bool(is_farm(p)))
                self._minor_cnt[f"{major}/{mn}"] += 1
                self._preset_cnt[p] += 1
            if picked:                    # 采空不计，调用方可换大类重试
                self._src_major[source_key][major] += 1
            return picked


def _alloc(targets: Dict[str, float], supply: Dict[str, int], total: int) -> Dict[str, int]:
    """按目标比例分配 total，供给不足的类缺口按比例重分给有余量的类。"""
    w = {k: v for k, v in targets.items() if supply.get(k, 0) > 0}
    alloc: Dict[str, int] = {}
    remaining = total
    # 迭代水填：每轮按剩余权重比例分，夹到供给上限
    for _ in range(len(w) + 1):
        if remaining <= 0 or not w:
            break
        ws = sum(w.values())
        grabbed = {}
        for k, wt in w.items():
            want = int(round(remaining * wt / ws))
            room = supply[k] - alloc.get(k, 0)
            grabbed[k] = min(want, room)
        for k, g in grabbed.items():
            alloc[k] = alloc.get(k, 0) + g
        remaining = total - sum(alloc.values())
        w = {k: wt for k, wt in w.items() if supply[k] - alloc.get(k, 0) > 0}
    # 小 total 时 round 可能全 0：按权重序补齐余量
    if remaining > 0:
        for k, _wt in sorted(targets.items(), key=lambda kv: -kv[1]):
            while remaining > 0 and supply.get(k, 0) - alloc.get(k, 0) > 0:
                alloc[k] = alloc.get(k, 0) + 1
                remaining -= 1
            if remaining <= 0:
                break
    return alloc


def stratified_sources(conn, total: int, min_iaa: float = 55.0,
                       extra_where: str = "", seed: str = "mix_v1") -> List[Any]:
    """按场景配额从 keep 池分层抽源图。返回 rows（asset_id, path, scene, iaa_mixed,
    is_portrait_pool）。类内按 md5(asset_id||seed) 伪随机排序保证可复现且去 asset_id 偏序。"""
    # 注：不再要求 saturation_mean（老 pipeline 的 stage0 标记，新 IAA 扫描行覆盖率 <1%，
    # 会把 4150 合格源饿到 30；渲染/QA 均不依赖它。2026-07-06）
    base = ("asset_type='image' AND b_quality=3 AND dup_of IS NULL "
            "AND iaa_mixed IS NOT NULL AND iaa_mixed >= %s"
            + (f" AND ({extra_where})" if extra_where else ""))
    supply_rows = conn.execute(
        f"SELECT COALESCE(scene,'any') AS scene, COUNT(*) AS n FROM assets WHERE {base} GROUP BY 1",
        (min_iaa,)).fetchall()
    supply = {r["scene"]: r["n"] for r in supply_rows}
    alloc = _alloc(SCENE_TARGETS, supply, total)
    out: List[Any] = []
    for scene, n in alloc.items():
        rows = conn.execute(
            f"SELECT asset_id, path, COALESCE(scene,'any') AS scene, iaa_mixed, is_portrait_pool "
            f"FROM assets WHERE {base} AND COALESCE(scene,'any')=%s "
            f"ORDER BY md5(asset_id || %s) LIMIT %s",
            (min_iaa, scene, seed, n)).fetchall()
        out.extend(rows)
    return out


def _preview() -> None:
    sys.path.insert(0, "/home/bc/VeraRetouch")
    from dataset_build.source_qa import db
    conn = db.connect()
    rows = stratified_sources(conn, total=2000)
    dist = defaultdict(int)
    for r in rows:
        dist[r["scene"]] += 1
    print(json.dumps({"n": len(rows), "scene_dist": dict(sorted(dist.items(), key=lambda kv: -kv[1]))},
                     ensure_ascii=False, indent=1))
    conn.close()


def _selfcheck() -> None:
    import os
    import tempfile
    # _alloc：供给不足重分 + 总量守恒
    a = _alloc({"a": 50, "b": 30, "c": 20}, {"a": 100, "b": 5, "c": 100}, 100)
    assert sum(a.values()) == 100 and a["b"] == 5, a
    # StyleSampler：3 大类（其中 mB 仅 2 小类需 round-robin 补齐；mC 单 preset 小类）
    rows = ([{"preset_id": f"a{i}", "major": "mA", "minor": f"n{i % 9}"} for i in range(45)]
            + [{"preset_id": f"b{i}", "major": "mB", "minor": f"n{i % 2}"} for i in range(20)]
            + [{"preset_id": "c0", "major": "mC", "minor": "n0"}])
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write("\n".join(json.dumps(r) for r in rows))
        tp = f.name
    s = StyleSampler(tp)
    os.unlink(tp)
    # 大类 per-source least-used：同源 3 次落 3 个不同大类
    ms = set()
    for _ in range(3):
        m = s.pick_major("src1")
        ms.add(m)
        s.sample_group(m, 8, "src1")
    assert ms == {"mA", "mB", "mC"}, ms
    # 组内小类互异（小类数≥k 时）
    g = s.sample_group("mA", 8, "src2")
    assert len(g) == 8 and len(set(g)) == 8
    # 小类不足 k：round-robin 补齐，同小类内 preset 不重复
    g = s.sample_group("mB", 8, "src3")
    assert len(g) == 8 and len(set(g)) == 8, g
    # 供给枯竭：单 preset 大类只出 1 个，不死循环
    g = s.sample_group("mC", 8, "src4")
    assert g == ["c0"], g
    # farm 上限：全 farm 时放行 max(1, k*0.25) 个
    g = s.sample_group("mA", 8, "src5", is_farm=lambda p: True)
    assert len(g) == 2, g
    # 覆盖均衡：多次采样后 mA 各 preset 使用次数差 ≤1
    for i in range(40):
        s.sample_group("mA", 8, f"s{i}")
    cnts = [s._preset_cnt[f"a{i}"] for i in range(45)]
    assert max(cnts) - min(cnts) <= 1, (min(cnts), max(cnts))
    # POOL_SPLIT 互斥子池：同源交替出 全反转组 / 全黑白组
    rows2 = ([{"preset_id": f"f{i}", "major": "反转黑白", "minor": f"冷浓·n{i % 4}"} for i in range(24)]
             + [{"preset_id": f"w{i}", "major": "反转黑白", "minor": f"黑白·n{i % 3}"} for i in range(12)])
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows2))
        tp2 = f.name
    s2 = StyleSampler(tp2)
    os.unlink(tp2)
    g1 = s2.sample_group("反转黑白", 8, "srcX")
    g2 = s2.sample_group("反转黑白", 8, "srcX")
    pool = lambda g: {p[0] for p in g}
    assert pool(g1) in ({"f"}, {"w"}) and pool(g2) in ({"f"}, {"w"}), (g1, g2)  # 组内纯池
    assert pool(g1) != pool(g2), "同源两轮必须交替换池"
    assert len(g1) == 8 and len(g2) == 8
    # 初相位打散：多源首轮的池分布应两边都有（全局各半）
    first = {("f" if s2.sample_group("反转黑白", 8, f"y{i}")[0][0] == "f" else "w") for i in range(12)}
    assert first == {"f", "w"}, first
    print("selfcheck ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "preview":
        _preview()
    else:
        _selfcheck()
