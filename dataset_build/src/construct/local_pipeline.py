"""LOCAL Route 1 两级编排（Phase 2，用户定稿 2026-07-11）。

政策（preview_native_eval_v1 的 conservative_768，8/8 精确复原 native top-2）：
  S 选基:   全量源先选 preset（taxonomy 大类→小类采样，逻辑同 agent.process_source_local）
  P 预览:   768 长边渲 8 变体（1 径向+1 语义+2 束状+4 线性；C_GT 不存）
  Q1 初筛:  QA 8 预览 → shortlist 5 = 语义 + 径向 + 最佳束状 + 最佳线性 + 其余最高分
  N 原生:   native 渲 shortlist 5（C_GT 由后端产出）
  Q2 终选:  QA 5 native → 留 top-2 进 tier

按 preset 攒批：S 完成后 (source, base) 按 preset_id 分组处理——同组渲染背靠背走
后端（preset 解析/fits/残差缓存全命中，farm 路由也整组走），P/N 两级都按组。
QA（vLLM）与渲染流水并行：组内源的 QA 提交线程池，GPU 渲下一源不等评分。

产出 group dict 与 agent.process_source_local 完全同形，tier/下游零改动。

CLI:  python -m construct.local_pipeline run [--n 100] [--out DIR] [--dry]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from PIL import Image

from dataset_build.source_qa import config
from . import mask_synth, qa, subject_geom

PREVIEW_LONG_EDGE = 768
SHORTLIST_N = 5
FINAL_N = 2
PREVIEW_STAGE = os.path.join(config.OUT_ROOT, "renders_stage", "local_preview")


# --------------------------------------------------------------------------- #
# S：全量选基（复用 agent 的资格过滤与配额逻辑）
# --------------------------------------------------------------------------- #
def select_bases(sel, sources: list) -> list:
    """[(src_dict, base_feat)]；无合格 base 的源丢弃（Selector.select_local_base 共享逻辑）。"""
    out = []
    for src in sources:
        base = sel.select_local_base(src)
        if base:
            out.append((src, base))
    return out


def group_by_preset(pairs: list) -> list:
    groups: dict = {}
    for src, base in pairs:
        groups.setdefault(base["preset_id"], []).append((src, base))
    # 大组在前：GPU 缓存收益最大化，队尾碎组自然收敛
    return sorted(groups.values(), key=len, reverse=True)


# --------------------------------------------------------------------------- #
# P/Q1：预览级
# --------------------------------------------------------------------------- #
def _preview_source(src_path: str, out_dir: str) -> str:
    """768 长边预览源（内容寻址复用；JPEG q95 与 preview_native_eval 一致）。"""
    os.makedirs(out_dir, exist_ok=True)
    key = hashlib.sha1(f"{src_path}|{PREVIEW_LONG_EDGE}".encode()).hexdigest()[:16]
    dst = os.path.join(out_dir, f"pv_{key}.jpg")
    if not os.path.exists(dst):
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            im.thumbnail((PREVIEW_LONG_EDGE, PREVIEW_LONG_EDGE))
            tmp = f"{dst}.{os.getpid()}.tmp"   # 原子替换：崩溃不留半截 JPEG 毒化复用
            im.save(tmp, "JPEG", quality=95)
            os.replace(tmp, dst)
    return dst


def shortlist(plan: list, rendered: list, qres: dict, n: int = SHORTLIST_N) -> list:
    """conservative 政策的 shortlist：语义 + 径向 + 最佳束状 + 最佳线性 + 其余最高分。

    rendered 行带 variant_index 指回 plan；返回选中的 variant_index 列表。
    注意 bisect 补位（_mode='linear_bisect'）不冒充其槽位的模式：radial/band/linear
    槽位若当初退化成了 bisect，则该模式桶为空，名额自然让给"其余最高分"——
    无主体组（全 bisect）因此退化为纯 top-n by q，这是政策本意。"""
    def q_of(row):
        s = qres["scores"].get(row["mask_unit_id"]) or {}
        return s.get("q") or 0.0

    by_mode: dict = {}
    for row in rendered:
        mode = plan[row["variant_index"]].get("_mode")
        by_mode.setdefault(mode, []).append(row)
    picked: list = []
    for mode in ("semantic", "radial"):
        rows = by_mode.get(mode)
        if rows:
            picked.append(max(rows, key=q_of))
    for mode in ("band", "linear"):
        rows = by_mode.get(mode)
        if rows:
            picked.append(max(rows, key=q_of))
    rest = [r for r in rendered if r not in picked]
    rest.sort(key=q_of, reverse=True)
    picked += rest[: max(0, n - len(picked))]
    return [r["variant_index"] for r in picked[:n]]


# --------------------------------------------------------------------------- #
# 单源两级流水（渲染在调用方的组内串行，QA 走线程池）
# --------------------------------------------------------------------------- #
def process_one(src: dict, base: dict, cgt_dir: str) -> Optional[dict]:
    path = src["path"]
    seed = int.from_bytes(hashlib.sha1(path.encode()).digest()[:4], "big")
    plan = subject_geom.sample_plan(path, random.Random(seed))
    is_portrait = bool(src.get("is_portrait_pool"))

    # P：768 预览渲 8（C_GT 不存；预览是临时物，不进 content-addressed 永久存储）
    pv_src = _preview_source(path, PREVIEW_STAGE)
    pv_rows = mask_synth.make_local_samples(pv_src, base, plan, PREVIEW_STAGE,
                                            save_cgt=False, store=False)
    if not pv_rows:
        return None
    # Q1（阻塞点在 vLLM；调用方把整个 process_one 丢进组级线程池，互相重叠）
    pv_variants = [(r["mask_unit_id"], r["after_path"]) for r in pv_rows]
    q1 = qa.qa_rank(pv_src, pv_variants, is_portrait=is_portrait)
    keep_idx = shortlist(plan, pv_rows, q1)

    # N：native 渲 shortlist（C_GT 由后端产出）
    native_plan = [plan[i] for i in keep_idx]
    rows = mask_synth.make_local_samples(path, base, native_plan, cgt_dir)
    if not rows:
        return None
    variants = [(r["mask_unit_id"], r["after_path"]) for r in rows]
    q2 = qa.qa_rank(path, variants, is_portrait=is_portrait)

    def q_of(row):
        s = q2["scores"].get(row["mask_unit_id"]) or {}
        return s.get("q") or 0.0

    rows.sort(key=q_of, reverse=True)
    final = rows[:FINAL_N]
    return {
        "source": path, "source_asset_id": src.get("asset_id"), "is_portrait": is_portrait,
        "source_iaa": src.get("iaa_mixed"), "local": True,
        "two_level": {"preview_n": len(pv_rows), "shortlist_n": len(rows),
                      "final_n": len(final)},
        "candidates": [mask_synth.local_candidate(base, r["spec"], r,
                                                  q2["scores"].get(r["mask_unit_id"]))
                       for r in final],
    }


# --------------------------------------------------------------------------- #
# run：S → 按 preset 攒批 → 组内流水
# --------------------------------------------------------------------------- #
def run(n: int, out_dir: str, workers: int = 6) -> dict:
    from dataset_build.source_qa import db
    from . import mixing, objscore
    from .agent import Selector

    os.makedirs(out_dir, exist_ok=True)
    cgt_dir = os.path.join(out_dir, "cgt")
    os.makedirs(cgt_dir, exist_ok=True)
    sel = Selector()
    # QA-IAA scorer 单线程预热（并发 import 竞争会退化成窄带常数，见 agent.run 注释）
    _scorer = objscore._runner()
    if hasattr(_scorer, "load"):
        _scorer.load()
    min_iaa = getattr(config, "CONSTRUCT_SOURCE_IAA_MIN", config.GATE["iaa_keep_above"])
    conn = db.connect()
    rows = [dict(r) for r in mixing.stratified_sources(conn, total=2 * n, min_iaa=min_iaa)]
    conn.close()
    rows = [r for r in rows if os.path.exists(r["path"])]
    random.Random(0).shuffle(rows)
    sources = rows[:n]
    pairs = select_bases(sel, sources)
    groups = group_by_preset(pairs)
    print(f"sources={len(sources)} with_base={len(pairs)} preset_groups={len(groups)}",
          flush=True)

    n_ok = n_fail = 0
    lock = threading.Lock()
    out_path = os.path.join(out_dir, "groups.jsonl")
    gf = open(out_path, "a", encoding="utf-8")          # 流式落盘：晚期崩溃不丢已完成组
    ff = open(os.path.join(out_dir, "failures.jsonl"), "a", encoding="utf-8")

    def _one(src, base):
        nonlocal n_ok, n_fail
        try:
            g = process_one(src, base, cgt_dir)
        except subject_geom.SubjectCacheMiss as e:
            with lock:
                n_fail += 1
                ff.write(json.dumps({"source": src["path"],
                                     "error": f"cache_miss: {e}"}, ensure_ascii=False) + "\n")
                ff.flush()
            return
        except Exception as e:  # noqa: BLE001
            with lock:
                n_fail += 1
                ff.write(json.dumps({"source": src["path"],
                                     "error": repr(e)[:300]}, ensure_ascii=False) + "\n")
                ff.flush()
            return
        if g:
            with lock:
                n_ok += 1
                gf.write(json.dumps(g, ensure_ascii=False, default=str) + "\n")
                gf.flush()

    # 按源提交、按 preset 组排序：GPU 锁天然串行渲染且同 preset 背靠背（缓存命中），
    # 每源一个任务避免"单一大组串行、其余 worker 空转"的偏斜坍缩。
    ordered = [pair for grp in groups for pair in grp]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in [ex.submit(_one, src, base) for src, base in ordered]:
            f.result()
    gf.close()
    ff.close()

    summary = {"groups": n_ok, "failures": n_fail,
               "preset_groups": len(groups),
               "renders_per_source": {"preview": 8, "native": SHORTLIST_N},
               "out": out_path}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--n", type=int, default=50)
    r.add_argument("--out", default="/tmp/local_pipeline_run")
    r.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    run(a.n, a.out, a.workers)


if __name__ == "__main__":
    main()
