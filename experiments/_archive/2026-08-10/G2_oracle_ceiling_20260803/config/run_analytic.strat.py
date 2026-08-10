"""G2 解析法 Δ_ceil 跑批：三个数据档 + 每档的 s-移植对照（Δ_shuffle 类比）。

档位
----
construct : D-CONSTRUCT(S-val) 8 级 × 24，s = GT 掩膜（判据 Δ_ceil ≥ 8 dB）
real      : D-SFT-L(S-val, normal)，s = C_GT 掩膜（判据 ≥ 1.0 dB；死刑线 < 0.4 dB）
control   : D-SFT-G(S-val, normal)，真·全局编辑 + **移植掩膜**当 s（期望 ≈ 0）

每张图都算三列（红线：每行必带 Δ_const/Δ_shuffle）：
  delta        s = 本图真值掩膜（control 档 = 移植掩膜）
  delta_donor  s = 另一张图的掩膜（Δ_shuffle）
  delta_const  s ≡ 1（Δ_const，构造上恒为 0，作实现自检）

用法:
  CUDA_VISIBLE_DEVICES=1 python3 tools/ceiling/run_analytic.py \
      --track real --index <idx.jsonl> --out <dir> [--limit 500]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/bc/VeraRetouch")

from tools.ceiling.delta_ceil import CeilResult, ceiling, confound_stats  # noqa: E402
from tools.ceiling.loader import load_mask, load_triplet  # noqa: E402

CONSTRUCT_ROOT = Path("/home/bc/VeraRetouch/experiments/tooling-wave1/T4_construct/sanity")


# --------------------------------------------------------------------------- 数据源

def iter_construct(split: str = "val", levels: list[str] | None = None):
    """D-CONSTRUCT：yield (meta, x, y, s, hw)。"""
    from PIL import Image
    root = CONSTRUCT_ROOT / split
    for line in open(root / "manifest.jsonl", encoding="utf-8"):
        m = json.loads(line)
        if levels and m["level"] not in levels:
            continue
        f = m["files"]
        x = np.asarray(Image.open(root / f["in"]).convert("RGB"), np.float32) / 255.0
        y = np.asarray(Image.open(root / f["out"]).convert("RGB"), np.float32) / 255.0
        s = np.asarray(Image.open(root / f["mask"]).convert("L"), np.float32) / 255.0
        tf = m["transform"]
        tfs = tf if isinstance(tf, list) else [tf]          # L6/L7 是多算子列表
        meta = {"id": m["uid"], "level": m["level"], "mask_kind": m["mask_kind"],
                "transform": "+".join(str(t.get("class")) for t in tfs),
                "tier": tfs[0].get("tier"), "amp": tfs[0].get("amp"),
                "source_id": m["source_id"], "alpha_achieved": m.get("alpha_achieved")}
        yield meta, x.reshape(-1, 3), y.reshape(-1, 3), s.reshape(-1), x.shape[:2]


def iter_indexed(index_path: Path, limit: int | None, donor_masks: list[np.ndarray] | None,
                 per_build: int | None = None):
    """D-SFT-L / D-SFT-G：yield (meta, x, y, s, hw)。control 档用移植掩膜当 s。

    `per_build`（D-34 补批）：**按 build 分层**均衡抽样 —— 每个 build 各取满
    `per_build` 个成功加载的样本（该 build 不足则取满该 build）。build 内仍按索引
    确定序，build 间按名字升序输出。默认 None = 原口径（全表按索引顺序截断，
    因索引本身按 build 顺序拼接 → 尾部 build 零样本，正是 D-34 要关掉的口子）。

    只改**选哪些样本**，估计器/判据/donor 池构造一律不动。
    """
    rows = [json.loads(l) for l in open(index_path, encoding="utf-8")]
    if per_build is None:
        groups: list[list[int]] = [list(range(len(rows)))]
        caps: list[int | None] = [limit]
    else:
        by: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            by.setdefault(str(r.get("build")), []).append(i)
        keys = sorted(by)
        groups = [by[k] for k in keys]
        caps = [per_build] * len(keys)
        print(f"[strat] builds={keys} per_build={per_build} "
              f"avail={[len(by[k]) for k in keys]}", flush=True)
    total = 0
    for idxs, cap in zip(groups, caps):
        n = 0
        for i in idxs:
            if cap is not None and n >= cap:
                break
            if limit is not None and total >= limit:
                return
            r = rows[i]
            donor = None
            if donor_masks:
                donor = donor_masks[i % len(donor_masks)]
            try:
                t = load_triplet(r, donor_mask=donor)
            except Exception as e:                   # 源图损坏/银行缺失
                print(f"[warn] load fail {r['candidate_id']}: {e}", flush=True)
                continue
            if t is None:
                continue
            meta = {"id": r["candidate_id"], "build": r["build"], "pool": r["pool"],
                    "source_id": r["source_id"], "group_id": r["group_id"],
                    "preset_id": r["preset_id"], "region": r.get("region"),
                    "mask_area": r.get("mask_area"), "amount": r.get("amount"),
                    "render_mode": r.get("render_mode"),
                    "s_kind": "donor_cgt" if donor is not None else "cgt"}
            n += 1
            total += 1
            yield meta, t.x, t.y, t.s, t.hw


def load_donor_pool(index_path: Path, k: int = 64) -> list[np.ndarray]:
    """从 l 系索引取 k 张 C_GT 掩膜作移植池（确定序）。"""
    rows = [json.loads(l) for l in open(index_path, encoding="utf-8")]
    out: list[np.ndarray] = []
    for r in rows:
        if len(out) >= k:
            break
        try:
            m = load_mask(r)
        except Exception:
            continue
        if m is not None and 0.02 < float(m.mean()) < 0.98:
            out.append(m)
    return out


# --------------------------------------------------------------------------- 跑批

def _donor_s(donor_masks: list[np.ndarray], idx: int, hw: tuple[int, int],
             stride: int) -> np.ndarray:
    from tools.ceiling.loader import _resize_nn
    m = _resize_nn(donor_masks[idx % len(donor_masks)], hw)
    return np.ascontiguousarray(m[::stride, ::stride].reshape(-1))


def run(track: str, index_path: Path | None, out_dir: Path, limit: int | None,
        donor_index: Path | None, levels: list[str] | None, device: str,
        with_confound: bool, alt_levels: tuple[int, ...], seed: int,
        estimator: str = "affine", also_const: bool = True,
        per_build: int | None = None, tag: str = "") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    donor_pool: list[np.ndarray] = []
    if donor_index is not None:
        donor_pool = load_donor_pool(donor_index)
        print(f"[donor] pool={len(donor_pool)}", flush=True)

    if track == "construct":
        src = iter_construct("val", levels)
    else:
        assert index_path is not None
        # control 档：s 本身就是移植掩膜
        src = iter_indexed(index_path, limit,
                           donor_pool if track == "control" else None,
                           per_build=per_build)

    per: list[dict] = []
    t0 = time.time()
    for i, (meta, x, y, s, hw) in enumerate(src):
        if track == "construct" and limit is not None and i >= limit:
            break
        r: CeilResult = ceiling(x, y, s, device=device, seed=seed, estimator=estimator)
        row = dict(meta)
        row.update(r.to_json())
        row["hw"] = list(hw)
        row["estimator"] = estimator

        # Δ_shuffle：换成另一张图的掩膜
        if donor_pool:
            sd = _donor_s(donor_pool, i + 1, hw, 1)
            if sd.shape[0] == s.shape[0]:
                rd = ceiling(x, y, sd, device=device, seed=seed, with_cv=False,
                             estimator=estimator)
                row["delta_donor"] = rd.delta
                row["delta_arm_donor"] = rd.extra["delta_arm"]
                row["psnr_4d_donor"] = rd.psnr_4d
        # Δ_const：s ≡ 1
        rc = ceiling(x, y, np.ones_like(s), device=device, seed=seed, with_cv=False,
                     estimator=estimator)
        row["delta_const"] = rc.delta
        row["delta_arm_const"] = rc.extra["delta_arm"]

        # 带宽稳定性（PLAN Step 6）：17³ 起点的替代分箱
        ra = ceiling(x, y, s, device=device, levels=alt_levels, seed=seed,
                     with_cv=False, estimator=estimator)
        row["delta_alt"] = ra.delta
        row["delta_arm_alt"] = ra.extra["delta_arm"]
        row["psnr_3d_alt"] = ra.psnr_3d

        # PLAN §4.1 字面口径（逐箱条件均值）作对照列
        if also_const and estimator != "const":
            rk = ceiling(x, y, s, device=device, seed=seed, with_cv=False,
                         estimator="const")
            row["delta_constest"] = rk.delta
            row["delta_arm_constest"] = rk.extra["delta_arm"]
            row["psnr_3d_constest"] = rk.psnr_3d
            row["psnr_4d_constest"] = rk.psnr_4d

        if with_confound:
            row.update(confound_stats(x, y, hw))
        per.append(row)
        if (i + 1) % 25 == 0:
            print(f"[{track}] {i+1} done, {time.time()-t0:.0f}s, "
                  f"median delta={np.median([p['delta'] for p in per]):.3f}", flush=True)

    with open(out_dir / f"per_image_{track}{tag}.jsonl", "w", encoding="utf-8") as f:
        for p in per:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    agg = aggregate(track, per)
    agg["sampling"] = ("stratified_by_build" if per_build is not None
                       else "index_order_truncate")
    agg["per_build_cap"] = per_build
    with open(out_dir / f"agg_{track}{tag}.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in agg.items() if k != "by_level"},
                     ensure_ascii=False, indent=2))
    return agg


def _q(a: list[float]) -> dict:
    v = np.asarray([x for x in a if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
            "min": float(v.min()), "max": float(v.max()), "std": float(v.std())}


def aggregate(track: str, per: list[dict]) -> dict:
    keys = ["delta", "delta_arm", "delta_cv", "delta_arm_cv",
            "delta_donor", "delta_arm_donor", "delta_const", "delta_arm_const",
            "delta_alt", "delta_arm_alt",
            "delta_constest", "delta_arm_constest",
            "psnr_3d", "psnr_4d", "psnr_4d_arm", "psnr_id",
            "psnr_3d_constest", "psnr_4d_constest",
            "n_s_buckets", "frac_refined", "moran_i", "blur_drop"]
    agg: dict = {"track": track, "n": len(per)}
    for k in keys:
        vals = [p[k] for p in per if k in p]
        if vals:
            agg[k] = _q(vals)
    # 带宽稳定性 Spearman（33³ 起 vs 17³ 起）
    a = np.array([p["delta"] for p in per if "delta_alt" in p])
    b = np.array([p["delta_alt"] for p in per if "delta_alt" in p])
    if a.size >= 5:
        from scipy.stats import spearmanr
        rho = spearmanr(a, b)
        agg["spearman_delta_vs_alt"] = float(rho.statistic)   # type: ignore[attr-defined]
    # 净局部信号 = delta − delta_donor（同图对照）
    d = [p["delta"] - p["delta_donor"] for p in per if "delta_donor" in p]
    if d:
        agg["delta_minus_donor"] = _q(d)
    da = [p["delta_arm"] - p["delta_arm_donor"] for p in per if "delta_arm_donor" in p]
    if da:
        agg["delta_arm_minus_donor"] = _q(da)
    aa = np.array([p["delta_arm"] for p in per if "delta_arm_alt" in p])
    ba = np.array([p["delta_arm_alt"] for p in per if "delta_arm_alt" in p])
    if aa.size >= 5:
        from scipy.stats import spearmanr
        agg["spearman_delta_arm_vs_alt"] = float(spearmanr(aa, ba).statistic)  # type: ignore[attr-defined]
    if track == "construct":
        by: dict = {}
        for lv in sorted({p["level"] for p in per}):
            sub = [p for p in per if p["level"] == lv]
            by[lv] = {"n": len(sub), "delta": _q([p["delta"] for p in sub]),
                      "delta_cv": _q([p["delta_cv"] for p in sub]),
                      "delta_arm": _q([p["delta_arm"] for p in sub]),
                      "delta_arm_cv": _q([p["delta_arm_cv"] for p in sub if "delta_arm_cv" in p]),
                      "delta_donor": _q([p["delta_donor"] for p in sub if "delta_donor" in p]),
                      "delta_arm_donor": _q([p["delta_arm_donor"] for p in sub if "delta_arm_donor" in p]),
                      "psnr_id": _q([p["psnr_id"] for p in sub]),
                      "psnr_3d": _q([p["psnr_3d"] for p in sub]),
                      "psnr_4d": _q([p["psnr_4d"] for p in sub]),
                      "psnr_4d_arm": _q([p["psnr_4d_arm"] for p in sub])}
        agg["by_level"] = by
    else:
        by = {}
        for pool in sorted({str(p.get("pool")) for p in per}):
            sub = [p for p in per if str(p.get("pool")) == pool]
            by[pool] = {"n": len(sub), "delta": _q([p["delta"] for p in sub]),
                        "delta_arm": _q([p["delta_arm"] for p in sub])}
        agg["by_pool"] = by
        # D-34 补批：分 build 稳健性（中位 + q10/q90 见 _q 的 p10/p90 字段）
        byb: dict = {}
        for b in sorted({str(p.get("build")) for p in per}):
            sub = [p for p in per if str(p.get("build")) == b]
            byb[b] = {
                "n": len(sub),
                "delta_arm": _q([p["delta_arm"] for p in sub]),
                "delta": _q([p["delta"] for p in sub]),
                "delta_arm_cv": _q([p["delta_arm_cv"] for p in sub
                                    if "delta_arm_cv" in p]),
                "delta_arm_donor": _q([p["delta_arm_donor"] for p in sub
                                       if "delta_arm_donor" in p]),
                "delta_arm_const": _q([p["delta_arm_const"] for p in sub
                                       if "delta_arm_const" in p]),
                "psnr_id": _q([p["psnr_id"] for p in sub]),
                "psnr_3d": _q([p["psnr_3d"] for p in sub]),
                "psnr_4d_arm": _q([p["psnr_4d_arm"] for p in sub]),
                "mask_area": _q([p["mask_area"] for p in sub
                                 if p.get("mask_area") is not None]),
                "frac_ge_1db": float(np.mean([p["delta_arm"] >= 1.0 for p in sub])),
                "frac_ge_04db": float(np.mean([p["delta_arm"] >= 0.4 for p in sub])),
            }
        agg["by_build"] = byb
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["construct", "real", "control"], required=True)
    ap.add_argument("--index", type=Path, default=None)
    ap.add_argument("--donor-index", type=Path, default=None,
                    help="l 系索引，用于取移植掩膜池（Δ_shuffle / control 档 s 源）")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--levels", nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alt-levels", nargs="*", type=int, default=[17, 9])
    ap.add_argument("--no-confound", action="store_true")
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--estimator", choices=["affine", "const"], default="affine")
    ap.add_argument("--no-const-column", action="store_true")
    ap.add_argument("--per-build", type=int, default=None,
                    help="按 build 分层均衡抽样，每 build 取满这么多（D-34 补批）")
    ap.add_argument("--tag", default="", help="输出文件名后缀，如 _strat")
    args = ap.parse_args()

    run(args.track, args.index, args.out, args.limit, args.donor_index,
        args.levels, args.device, not args.no_confound,
        tuple(args.alt_levels), args.seed, args.estimator,
        not args.no_const_column, args.per_build, args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
