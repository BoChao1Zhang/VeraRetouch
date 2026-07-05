"""样本混入策略：源图按场景分层配额 + preset 按 grade_family 风格配额。

场景配额参照 PARA 的 semantic 类别分布（portrait 15.6% / scene 13.7% / food 11.9%
/ stilllife 10.1% / indoor 9.2% / building 9.1% / plant 8.9% / nightScene 8.7%
/ animal 12.3%），映射到本库 assets.scene 的 10 类受控词表（无 animal/plant/indoor，
有 street/wedding/product/any——any 当未细分桶）。某类供给不足时按比例把缺口
重分给其余类，保证总量。

preset 风格配额用 bank features.jsonl 的 axes.grade_family（5 类）。bank 里
stylized 占 78%，直接按召回相关性取会严重偏斜；配额化后在整个构建 run 的
粒度上把各族比例拉到目标附近（逐 source 仍按召回相关性排序，只在族超额时跳过）。

python -m construct.mixing preview   # 对 live DB 预览分层结果
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

# 场景目标比例（PARA 启发，归一前权重；any=未细分桶）
SCENE_TARGETS: Dict[str, float] = {
    "portrait": 16.0, "landscape": 14.0, "food": 12.0, "any": 12.0,
    "still_life": 10.0, "architecture": 9.0, "night": 9.0,
    "street": 8.0, "wedding": 6.0, "product": 4.0,
}

# preset 风格目标比例（grade_family 5 类；bank 原始分布 stylized 78% 需压制）
FAMILY_TARGETS: Dict[str, float] = {
    "stylized": 45.0, "teal_orange": 15.0, "vintage_film": 15.0,
    "clean_natural": 15.0, "bw": 10.0,
}


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
    return alloc


def stratified_sources(conn, total: int, min_iaa: float = 55.0,
                       extra_where: str = "", seed: str = "mix_v1") -> List[Any]:
    """按场景配额从 keep 池分层抽源图。返回 rows（asset_id, path, scene, iaa_mixed,
    is_portrait_pool）。类内按 md5(asset_id||seed) 伪随机排序保证可复现且去 asset_id 偏序。"""
    base = ("asset_type='image' AND b_quality=3 AND dup_of IS NULL "
            "AND iaa_mixed IS NOT NULL AND iaa_mixed >= %s "
            "AND saturation_mean IS NOT NULL" + (f" AND ({extra_where})" if extra_where else ""))
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


class FamilyQuota:
    """preset 风格族配额器：run 级计数，超额族的候选被跳过（除非全部超额则放行最相关的）。"""

    def __init__(self, targets: Optional[Dict[str, float]] = None):
        self.targets = dict(targets or FAMILY_TARGETS)
        tot = sum(self.targets.values())
        self.targets = {k: v / tot for k, v in self.targets.items()}
        self.counts: Dict[str, int] = defaultdict(int)
        self.total = 0

    def _over(self, fam: str) -> bool:
        if self.total < 20:          # 冷启动不限制
            return False
        share = self.counts[fam] / max(1, self.total)
        return share > self.targets.get(fam, 0.05) * 1.15   # 15% 容差

    def pick(self, candidates: Iterable[dict], k: int, fam_of=lambda c: c.get("grade_family", "stylized")) -> List[dict]:
        """从按相关性降序的候选里取 k 个，超额族跳过；不足则回填最相关的被跳者。"""
        picked, skipped = [], []
        for c in candidates:
            if len(picked) >= k:
                break
            (skipped if self._over(fam_of(c)) else picked).append(c)
        for c in skipped:
            if len(picked) >= k:
                break
            picked.append(c)
        for c in picked:
            self.counts[fam_of(c)] += 1
            self.total += 1
        return picked


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
    # _alloc：供给不足重分 + 总量守恒
    a = _alloc({"a": 50, "b": 30, "c": 20}, {"a": 100, "b": 5, "c": 100}, 100)
    assert sum(a.values()) == 100 and a["b"] == 5, a
    # FamilyQuota：超额族被压制
    q = FamilyQuota({"x": 50, "y": 50})
    cands = [{"grade_family": "x", "i": i} for i in range(40)] + [{"grade_family": "y", "i": i} for i in range(40)]
    for _ in range(10):
        q.pick(list(cands), 8)
    share_x = q.counts["x"] / q.total
    assert 0.35 <= share_x <= 0.65, q.counts
    print("selfcheck ok", dict(q.counts))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "preview":
        _preview()
    else:
        _selfcheck()
