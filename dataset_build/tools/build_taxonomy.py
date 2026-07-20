#!/usr/bin/env python3
"""preset 两级风格 taxonomy 生成（v5 结构，2026-07-17 收编进仓库）。

历史：taxonomy.jsonl 原由一次性 scratch 脚本（taxonomy_v5.py，会话 7a3478fc）产出，
未进版本控制；2026-07-17 native LUT 轴序修复后需全量重建，遂移植于此。

结构（与部署版一致）：
  L0 判据三分：翻转 = 红探针 |Δhue|>60° 且 C_ratio>0.15；黑白 = 有色5探针
     C_ratio 均值 <0.3；其余彩色。
  L1 大类 = 子集内 KMeans（彩色 k=9 色调特征 / 翻转 k=4 after-state 特征 /
     黑白 1 类），簇名 = 画像自动名（人工定名在产出后人审替换）。
  L2 小类 = 大类内 18 维感知特征 KMeans + 小簇合并。

输入（--bank 目录）：features.jsonl（lab_vec 24 维）、probe_before.json、
vlm_names.jsonl（{preset_id, vlm_name}，仅用于小类命名词频，不参与分类判定）。
输出：--out 的 jsonl（{preset_id, major, minor}）+ 同名 .summary.json。
不直接覆盖生产 taxonomy.jsonl——人审大类命名后手动替换。
"""
import argparse
import collections
import json
import math
import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

PROBES = ("red", "yellow", "green", "blue", "skin", "neutral")


def wrap(d):
    return (d + 180) % 360 - 180


class Tax:
    def __init__(self, bank: str):
        self.before = json.load(open(os.path.join(bank, "probe_before.json")))
        self.feats, self.names = {}, {}
        for l in open(os.path.join(bank, "features.jsonl")):
            r = json.loads(l)
            self.feats[r["preset_id"]] = np.array(r["lab_vec"], dtype=np.float32)
        np_path = os.path.join(bank, "vlm_names.jsonl")
        if os.path.exists(np_path):
            for l in open(np_path):
                r = json.loads(l)
                self.names[r["preset_id"]] = r.get("vlm_name") or ""

    # ---- L0 判据 ----
    def flip(self, v):
        dL, da, db, dC = v[0:4]
        b0 = self.before["red"]
        return (abs(wrap(math.degrees(math.atan2(b0["b"] + db, b0["a"] + da)) - b0["hue"])) > 60
                and (b0["C"] + dC) / b0["C"] > 0.15)

    def bw(self, v):
        crs = [max(0.0, (self.before[nm]["C"] + v[4 * i + 3]) / self.before[nm]["C"])
               for i, nm in enumerate(PROBES[:5])]
        return float(np.mean(crs)) < 0.3

    # ---- 特征行 ----
    def hue_row(self, v):
        row = []
        for i, nm in enumerate(PROBES):
            dL, da, db, dC = v[4 * i:4 * i + 4]
            b0 = self.before[nm]
            a1, b1 = b0["a"] + da, b0["b"] + db
            w = min(math.hypot(a1, b1) / max(b0["C"], 8.0), 1.5)
            h1 = math.atan2(b1, a1)
            if nm == "neutral":
                row += [a1 / 6.0, b1 / 6.0, 0.0]
            else:
                row += [math.cos(h1) * w, math.sin(h1) * w,
                        wrap(math.degrees(h1) - b0["hue"]) / 60.0]
        return row

    def state_row(self, v):
        row = []
        for i, nm in enumerate(PROBES):
            dL, da, db, dC = v[4 * i:4 * i + 4]
            b0 = self.before[nm]
            C0 = max(b0["C"], 8.0)
            row += [(b0["a"] + da) / C0, (b0["b"] + db) / C0, dL / 25.0,
                    math.log2(max(0.02, (b0["C"] + dC) / max(b0["C"], 1e-6)))]
        return row

    def percep_row(self, v):
        row = []
        for i, nm in enumerate(PROBES):
            dL, da, db, dC = v[4 * i:4 * i + 4]
            b0 = self.before[nm]
            a1, b1 = b0["a"] + da, b0["b"] + db
            cr = max(0.02, (b0["C"] + dC) / max(b0["C"], 1e-6))
            hf = (math.hypot(a1, b1) if nm == "neutral"
                  else wrap(math.degrees(math.atan2(b1, a1)) - b0["hue"]))
            row += [hf, math.log2(cr) * 30, dL]
        return row

    def auto_name(self, members):
        V = np.stack([self.feats[m] for m in members])
        dL = float(np.mean([V[:, 4 * i].mean() for i in range(6)]))
        crm = float(np.mean(
            [np.mean(np.maximum(0, (self.before[nm]["C"] + V[:, 4 * i + 3])
                                / self.before[nm]["C"]))
             for i, nm in enumerate(PROBES[:5])]))
        na = V[:, 21].mean() + self.before["neutral"]["a"]
        nb = V[:, 22].mean() + self.before["neutral"]["b"]
        cast = math.hypot(na, nb)
        hue = math.degrees(math.atan2(nb, na)) % 360
        tone = "暗调" if dL < -6 else ("高亮" if dL > 6 else "中调")
        sat = "低饱" if crm < 0.7 else ("增艳" if crm > 1.1 else "常饱")
        if cast < 4:
            c = "无偏"
        elif 15 <= hue < 100:
            c = "暖"
        elif 100 <= hue < 150:
            c = "黄绿"
        elif 150 <= hue < 235:
            c = "青绿"
        elif 235 <= hue < 300:
            c = "青蓝"
        else:
            c = "品红"
        return f"{c}{tone}{sat}"


VOCAB = ["青绿", "青橙", "青蓝", "青冷", "青品", "品红", "暖调", "冷调", "低饱和",
         "高饱和", "高亮", "压暗", "暗调", "复古", "胶片", "黑白", "去色", "去饱和",
         "赛博", "霓虹", "增艳", "撞色", "通透", "清透", "清冷", "柔雾", "高反差",
         "全色相", "消解", "自然"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="/home/bc/data/datasets/vera_directionA_1M/preset_bank_full")
    ap.add_argument("--out", required=True, help="输出 jsonl（不要直接指向生产 taxonomy.jsonl）")
    ap.add_argument("--min-de", type=float, default=6.0,
                    help="弱效果过滤：perceptual_de.jsonl 的 de_med 低于此值不入分类（部署版口径 6.0）")
    a = ap.parse_args()
    t = Tax(a.bank)
    feats, names = t.feats, t.names
    if a.min_de > 0:
        de = {}
        for l in open(os.path.join(a.bank, "perceptual_de.jsonl")):
            r = json.loads(l)
            de[r["preset_id"]] = r.get("de_med")
        n0 = len(feats)
        feats = {p: v for p, v in feats.items()
                 if de.get(p) is not None and de[p] >= a.min_de}
        t.feats = feats
        print(f"弱效果过滤 de_med>={a.min_de}: {n0} -> {len(feats)}")

    groups = {"flip": [], "bw": [], "color": []}
    for p in sorted(feats):
        v = feats[p]
        groups["bw" if t.bw(v) else ("flip" if t.flip(v) else "color")].append(p)
    # 轴序修复后（2026-07-18）反转子池只剩个位数孤点（重去饱和款的色相数值噪声，
    # 无一真反转）；孤点大类会被 per-source least-used 采样系统性偏爱，<8 时并入黑白。
    if len(groups["flip"]) < 8:
        groups["bw"] += groups["flip"]
        groups["flip"] = []
    print("L0:", {k: len(v) for k, v in groups.items()})

    major_of = {}
    for p in groups["bw"]:
        major_of[p] = "黑白去色"
    for tag, ids, k, feat_fn in (("C", groups["color"], 9, t.hue_row),
                                 ("F", groups["flip"], 4, t.state_row)):
        if not ids:
            continue
        k = min(k, len(ids))
        X = np.array([feat_fn(feats[p]) for p in ids], dtype=np.float32)
        km = KMeans(n_clusters=k, n_init=8, random_state=0).fit(X)
        for c in sorted(set(km.labels_)):
            members = [ids[i] for i in np.where(km.labels_ == c)[0]]
            label = f"{'反转·' if tag == 'F' else ''}{t.auto_name(members)}_{tag}{c}"
            for m in members:
                major_of[m] = label
    major_members = collections.defaultdict(list)
    for p, m in major_of.items():
        major_members[m].append(p)
    print("\n大类（自动名）:")
    for m, mem in sorted(major_members.items(), key=lambda kv: -len(kv[1])):
        nm = collections.Counter(names.get(x, "?") for x in mem)
        print(f"  {m} n={len(mem)}  " + " / ".join(f"{k}·{v}" for k, v in nm.most_common(3)))

    def tokens(n):
        return [w for w in VOCAB if w in n]

    out = open(a.out, "w")
    summary = {}
    for major, ids in sorted(major_members.items(), key=lambda kv: -len(kv[1])):
        ids = sorted(ids)
        n = len(ids)
        X = StandardScaler().fit_transform(np.stack([t.percep_row(feats[i]) for i in ids]))
        kmax = max(8, min(14, n // 12))
        best = None
        for k in range(min(8, max(2, n // 6)), kmax + 1):
            if k >= n:
                break
            km = KMeans(n_clusters=k, n_init=6, random_state=0).fit(X)
            sil = silhouette_score(X, km.labels_)
            if best is None or sil > best[0]:
                best = (sil, km.labels_.copy())
        labs = best[1] if best is not None else np.zeros(n, dtype=int)
        min_size = max(3, int(n * 0.02))
        while True:
            sizes = collections.Counter(labs)
            small = [c for c, s in sizes.items() if s < min_size]
            if not small or len(sizes) <= 2:
                break
            c = min(small, key=lambda c: sizes[c])
            cents = {cc: X[labs == cc].mean(0) for cc in sizes}
            tgt = min((cc for cc in sizes if cc != c),
                      key=lambda cc: np.linalg.norm(cents[cc] - cents[c]))
            labs[labs == c] = tgt
        uniq = sorted(set(labs), key=lambda c: -np.sum(labs == c))
        remap = {c: i for i, c in enumerate(uniq)}
        labs = np.array([remap[c] for c in labs])
        for c in range(len(uniq)):
            members = [ids[i] for i in np.where(labs == c)[0]]
            tok = collections.Counter(tk for m in members for tk in tokens(names.get(m, "")))
            top = [w for w, _ in tok.most_common(2)] or ["混合"]
            lab = f"{''.join(top)}_{c:02d}"
            cent = X[labs == c].mean(0)
            d2c = np.linalg.norm(X[labs == c] - cent, axis=1)
            summary.setdefault(major, {})[lab] = {
                "size": len(members),
                "reps": [members[j] for j in np.argsort(d2c)[:3]],
                "top_names": [x for x, _ in collections.Counter(
                    names.get(m, "") for m in members).most_common(3)]}
            for m in members:
                out.write(json.dumps({"preset_id": m, "major": major, "minor": lab},
                                     ensure_ascii=False) + "\n")
    out.close()
    json.dump(summary, open(a.out.replace(".jsonl", "") + ".summary.json", "w"),
              ensure_ascii=False, indent=1)
    print("\n小类结构:")
    for major, ss in sorted(summary.items(),
                            key=lambda kv: -sum(v["size"] for v in kv[1].values())):
        print(f"  {major}: {len(ss)} 小类 " + " / ".join(
            f"{k}·{v['size']}" for k, v in
            sorted(ss.items(), key=lambda kv: -kv[1]["size"])[:6]))


if __name__ == "__main__":
    main()
