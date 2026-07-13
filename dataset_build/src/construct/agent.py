"""R4 end-to-end pipeline: source photo -> style-taxonomy sampling -> render N -> QA -> tier records.

Locked design (2026-07-12 重构：两级风格采样取代 vlemb 召回):
  select  = taxonomy 两级采样: 大类 per-source least-used(跨 run 历史) -> 组内小类轮转
            各取 1(不足 k 则 round-robin 补齐) -> 小类内 preset least-used。无 embedding、
            无相关性——组内同大类让 IAA 排序只比「同风格下谁执行得好」，消除跨风格 bias；
            多样性/覆盖由采样结构保证(mixing.StyleSampler)。
  render  = render.render_preset (param->GPU/LR farm, lut->trilinear)
  QA      = qa.qa_rank (IAA ArtiMuse+Charm) -> veto + merit
  tier    = tier.build (top-2 by merit -> SFT[全部 style 任务]; DPO 为副产物)

ponytail: deterministic fan-out (recall, N concurrent renders, concurrent pairwise QA) — plain
Python + ThreadPoolExecutor, NOT LangChain. No LLM routing decisions to make here; a DAG framework
would be pure ceremony. Concurrency keeps the 3 resources (SiliconFlow embed / LR farm / vLLM QA)
busy by having multiple sources in flight.

CLI:  python -m construct.agent run [--n 100] [--render-n 8] [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from dataset_build.source_qa import db, config
from . import render, qa, tier, mask_synth, mask_sam3, mixing
from .bank import PresetBank

FULL = "/home/bc/data/datasets/vera_directionA_1M/preset_bank_full"


class Selector:
    """taxonomy 两级风格采样器（global 与 local 的 preset 选择共用同一逻辑）。"""

    def __init__(self):
        self.bank = PresetBank.load(FULL)
        self.feat = {f["preset_id"]: f for f in self.bank.feats}
        self.sampler = mixing.StyleSampler(os.path.join(FULL, "taxonomy.jsonl"),
                                           prior_major=self._major_prior())
        self._farm_cache: dict = {}

    @staticmethod
    def _major_prior() -> dict:
        """跨 run 的 per-source 大类使用计数（construct_groups.style_major），
        保证同源多次采样渐进覆盖全部大类。表/列不存在或 DB 不可达时从零开始。"""
        try:
            conn = db.connect()
            rows = conn.execute(
                "SELECT source_asset_id, style_major, COUNT(*) AS n FROM construct_groups "
                "WHERE style_major IS NOT NULL GROUP BY 1, 2").fetchall()
            conn.close()
        except Exception:
            return {}
        prior: dict = {}
        for r in rows:
            prior.setdefault(r["source_asset_id"], {})[r["style_major"]] = r["n"]
        return prior

    def sample(self, src: dict, k: int, eligible=None) -> tuple:
        """(大类, [feat×≤k])。锁定 per-source least-used 大类采一组；该大类被
        eligible 滤空则换下一个大类重试（local 的资格过滤很窄，global 不会触发）。"""
        key = str(src.get("asset_id") or src.get("path") or src)
        elig_pid = (lambda pid: bool(eligible(self.feat[pid]))) if eligible else None
        tried: set = set()
        for _ in range(len(self.sampler.majors)):
            major = self.sampler.pick_major(key, exclude=tried)
            pids = self.sampler.sample_group(
                major, k, key, eligible=elig_pid,
                is_farm=lambda pid: self._is_farm(self.feat[pid]))
            if pids:
                return major, [self.feat[p] for p in pids]
            tried.add(major)
        return None, []

    def select_local_base(self, src) -> dict | None:
        """LOCAL Route 1 选基：param/xmp/无内嵌 local 资格过滤，GPU 路优先，
        大类→小类采样取 1（agent 与 local_pipeline 共用）。"""
        if not isinstance(src, dict):     # local_pipeline 传 path 的兼容
            src = {"path": src}

        def _elig(f):
            return (f.get("kind") == "param" and str(f.get("path", "")).endswith(".xmp")
                    and not f.get("has_local_mask"))
        _, feats = self.sample(src, 1, eligible=lambda f: _elig(f) and not self._is_farm(f))
        if not feats:
            _, feats = self.sample(src, 1, eligible=_elig)
        return feats[0] if feats else None

    def _is_farm(self, feat: dict) -> bool:
        """preset 是否只能农场渲（mask/未覆盖键/exotic profile）；结果按 preset_id 缓存。"""
        pid = feat.get("preset_id") or ""
        hit = self._farm_cache.get(pid)
        if hit is None:
            try:
                from gpu_render.route import route_preset
                hit = route_preset(feat["path"], feat.get("fmt") or "xmp").get("route") == "farm"
            except Exception:
                hit = True
            self._farm_cache[pid] = hit
        return hit


def process_source(sel: Selector, src: dict, render_n: int, render_workers: int = 6) -> dict:
    """One source through the full pipeline. Returns the processed group (candidates + qa)."""
    path = src["path"]
    major, cands = sel.sample(src, render_n)

    def _r(f):
        res = render.render_preset(f["path"], f["kind"], f.get("fmt"), path,
                                   preset_id=f["preset_id"])
        return (f, res["after_path"], res.get("engine")) if res.get("ok") else None
    with ThreadPoolExecutor(max_workers=render_workers) as ex:
        rendered = [x for x in ex.map(_r, cands) if x]
    variants = [(f["preset_id"], ap) for f, ap, _ in rendered]
    is_portrait = bool(src.get("is_portrait_pool"))
    qres = qa.qa_rank(path, variants, is_portrait=is_portrait)
    return {
        "source": path, "source_asset_id": src.get("asset_id"), "is_portrait": is_portrait,
        "source_iaa": src.get("iaa_mixed"), "style_major": major,
        "candidates": [{"preset_id": f["preset_id"], "kind": f["kind"], "fmt": f.get("fmt"),
                        "preset_path": f["path"], "content_hash": f.get("preset_content_hash"),
                        "after_path": ap, "engine": eng,
                        "qa": qres["scores"].get(f["preset_id"])} for f, ap, eng in rendered],
    }


def process_source_local(sel: "Selector", src: dict, n_masks: int, cgt_dir: str) -> dict | None:
    """LOCAL Route 1 (Mask v4): render one selected complete preset, then localize it
    into the per-source plan (1 radial + 1 semantic + 2 band + 4 linear; bisect fills)."""
    from . import subject_geom
    path = src["path"]
    base = sel.select_local_base(src)
    if not base:
        return None
    # Per-source RNG keeps the whole plan deterministic regardless of worker scheduling.
    seed = int.from_bytes(hashlib.sha1(path.encode()).digest()[:4], "big")
    plan = subject_geom.sample_plan(path, random.Random(seed), n=n_masks)
    rendered = mask_synth.make_local_samples(path, base, plan, cgt_dir)
    samples = [(plan[s["variant_index"]], s) for s in rendered]
    variants = [(s["mask_unit_id"], s["after_path"]) for _, s in samples]
    is_portrait = bool(src.get("is_portrait_pool"))
    qres = qa.qa_rank(path, variants, is_portrait=is_portrait) if variants else {"scores": {}}
    return {
        "source": path, "source_asset_id": src.get("asset_id"), "is_portrait": is_portrait,
        "source_iaa": src.get("iaa_mixed"), "local": True,
        "candidates": [mask_synth.local_candidate(base, g, s,
                                                  qres["scores"].get(s["mask_unit_id"]))
                       for g, s in samples],
    }


def process_source_sam3(sel: "Selector", src: dict, n_masks: int, cgt_dir: str,
                        render_workers: int = 6) -> dict | None:
    """LOCAL Route 2: a taxonomy-sampled LUT composited into a SAM3 semantic region (numpy). Candidates
    = the LUT applied to the top-K SAM3 concepts; QA picks the best region. local C_GT = SAM3 mask."""
    path = src["path"]
    if not mask_sam3.has_cache(path):
        return None
    _, luts = sel.sample(src, 1, eligible=lambda f: f.get("kind") == "lut")
    lut = luts[0] if luts else None
    cons = mask_sam3.candidate_concepts(path, n_masks)
    if not lut or not cons:
        return None

    def _one(cn):
        slug, png, _area = cn
        return (slug, mask_sam3.make_lut_local_sample(path, lut, slug, png, cgt_dir))
    with ThreadPoolExecutor(max_workers=render_workers) as ex:
        samples = [(slug, s) for slug, s in ex.map(_one, cons) if s]
    variants = [(s["mask_unit_id"], s["after_path"]) for _, s in samples]
    is_portrait = bool(src.get("is_portrait_pool"))
    qres = qa.qa_rank(path, variants, is_portrait=is_portrait) if variants else {"scores": {}}
    return {
        "source": path, "source_asset_id": src.get("asset_id"), "is_portrait": is_portrait,
        "source_iaa": src.get("iaa_mixed"), "local": True,
        "candidates": [{"preset_id": s["mask_unit_id"], "kind": "lut_in_sam3",
                        "after_path": s["after_path"], "qa": qres["scores"].get(s["mask_unit_id"]),
                        "local": {"route": "sam3", "mask_unit_id": s["mask_unit_id"],
                                  "concept": s["concept"], "concept_cn": s["concept_cn"],
                                  "cgt_path": s["cgt_path"], "area": s["area"],
                                  "base_preset_id": lut["preset_id"], "base_preset_path": lut["path"]}}
                       for _, s in samples],
    }


PERSIST_CHUNK = 25   # groups per incremental provenance write


def _load_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue   # torn tail line from a crash mid-write
    return out


def run(n: int, render_n: int, out_dir: str, src_workers: int = 12, local: bool = False,
        route: str = "geom", persist: bool = True, resume: bool = True) -> dict:
    """Lane-overlapped driver: src_workers sources in flight, each flowing
    recall -> render -> QA -> tier. The per-lane global gates (LR farm admission in
    lr_render, vLLM admission in qa) bound the real resources, so a wide in-flight
    window keeps the LR farm rendering source N+1 while vLLM judges source N —
    pipeline overlap without a queue framework. Output is STREAMED: each finished
    group appends to groups/sft/dpo.jsonl immediately and provenance is persisted
    every PERSIST_CHUNK groups, so a crash loses at most one chunk and a rerun
    with resume=True skips already-done sources."""
    os.makedirs(out_dir, exist_ok=True)
    cgt_dir = os.path.join(out_dir, "cgt")
    sel = Selector()   # both routes share taxonomy sampling (global: the group; local: the base preset)
    # QA-IAA scorer 必须在 lanes 并发前单线程预热：ArtiMuse 加载用临时 sys.modules shim
    # （iaa._temporary_artimuse_compat_modules），与其他线程的 import 竞争会把预处理链
    # 换成 stub → 打分退化成 ~46-50 窄带常数（2026-07-06 生产事故，同文件独立进程重打
    # 正常 55-69）。预热在单线程窗口完成加载即避开竞争。
    from . import objscore
    _scorer = objscore._runner()
    if hasattr(_scorer, "load"):
        _scorer.load()
    route_label = (route if local else "global")
    conn = db.connect()
    run_id = db.start_run(conn, f"construct_{route_label}",
                          {"n": n, "render_n": render_n, "out": out_dir, "local": local,
                           "route": route_label}) if persist else None
    conn.commit()
    # Route 1 (geometric mask-only) applies to ANY photo; b_subject is for Route 2 (SAM3 semantic).
    min_iaa = getattr(config, "CONSTRUCT_SOURCE_IAA_MIN", config.GATE["iaa_keep_above"])
    # 源 iaa 上界（可选）：q=0.72·abs+0.28·rel 对高分源有天花板效应——iaa>65 的源
    # preset 渲染普遍打不过它，SFT 产率 1.7→0.34/源（2026-07-06 实测）。中带源既保质量
    # 又有提升空间。CONSTRUCT_SOURCE_IAA_MAX 不设则不启用。
    max_iaa = os.environ.get("CONSTRUCT_SOURCE_IAA_MAX", "")
    extra = f"iaa_mixed < {float(max_iaa)}" if max_iaa else ""
    # 场景分层抽样（PARA 启发配额，见 mixing）；超取 2n 抗 resume/缺文件损耗
    rows = [dict(r) for r in mixing.stratified_sources(conn, total=2 * n, min_iaa=min_iaa,
                                                       extra_where=extra)]
    conn.close()
    rows = [r for r in rows if os.path.exists(r["path"])]
    random.Random(0).shuffle(rows)
    srcs = rows[:n]

    # Resume: prior streamed output in this out_dir defines what is already done.
    # ponytail: groups/sft/dpo stay in RAM for the final summary, same as before —
    # at ~100k sources stream the summary instead.
    groups = _load_jsonl(os.path.join(out_dir, "groups.jsonl")) if resume else []
    sft = _load_jsonl(os.path.join(out_dir, "sft.jsonl")) if resume else []
    dpo = _load_jsonl(os.path.join(out_dir, "dpo.jsonl")) if resume else []
    done_srcs = {g["source"] for g in groups}
    todo = [s for s in srcs if s["path"] not in done_srcs]
    if done_srcs:
        print(f"[resume] {len(done_srcs)} sources already in {out_dir}; {len(todo)} to go")

    mode = "a" if resume else "w"
    gf = open(os.path.join(out_dir, "groups.jsonl"), mode)
    sf = open(os.path.join(out_dir, "sft.jsonl"), mode)
    df = open(os.path.join(out_dir, "dpo.jsonl"), mode)
    fail_f = open(os.path.join(out_dir, "failures.jsonl"), "a")
    fail_lock = threading.Lock()

    def _proc(s):
        if not local:
            return _safe(process_source, sel, s, render_n, fail_f=fail_f, fail_lock=fail_lock)
        fn = process_source_sam3 if route == "sam3" else process_source_local
        return _safe(fn, sel, s, render_n, cgt_dir, fail_f=fail_f, fail_lock=fail_lock)

    pend_g, pend_s, pend_d = [], [], []   # provenance chunk buffers
    ptotals: dict = {}

    def _flush_provenance():
        nonlocal pend_g, pend_s, pend_d
        if not (persist and run_id is not None and pend_g):
            return
        from . import provenance
        st = provenance.persist_run(run_id, pend_g, pend_s, pend_d, route_label)
        for k, v in st.items():
            if isinstance(v, (int, float)):
                ptotals[k] = ptotals.get(k, 0) + v
        pend_g, pend_s, pend_d = [], [], []

    n_done = len(done_srcs)
    with ThreadPoolExecutor(max_workers=src_workers) as ex:
        futs = [ex.submit(_proc, s) for s in todo]
        for fut in as_completed(futs):
            g = fut.result()   # _safe never raises
            if not g:
                continue
            try:
                s, d = tier.build(g)
            except Exception as e:  # noqa: BLE001 - tier error must not lose the rendered group
                print(f"[tier-skip] {os.path.basename(g['source'])}: {type(e).__name__}: {str(e)[:100]}")
                s, d = [], []
            groups.append(g); sft.extend(s); dpo.extend(d)
            gf.write(json.dumps(g, ensure_ascii=False) + "\n"); gf.flush()
            for r in s:
                sf.write(json.dumps(r, ensure_ascii=False) + "\n")
            for r in d:
                df.write(json.dumps(r, ensure_ascii=False) + "\n")
            sf.flush(); df.flush()
            pend_g.append(g); pend_s.extend(s); pend_d.extend(d)
            if len(pend_g) >= PERSIST_CHUNK:
                _flush_provenance()
            n_done += 1
            print(f"[{n_done}/{len(srcs)}] {os.path.basename(g['source'])[:30]:30s} "
                  f"cands={len(g['candidates'])} sft={len(s)} dpo_pairs={len(d)}")
    _flush_provenance()
    for fh in (gf, sf, df, fail_f):
        fh.close()
    summary = tier.summarize(groups, sft, dpo)
    if persist and run_id is not None:
        c2 = db.connect(); db.finish_run(c2, run_id, ptotals); c2.commit(); c2.close()
        summary["postgres"] = {"run_id": run_id, **ptotals}
    print("\n=== R4 SUMMARY ===\n" + json.dumps(summary, ensure_ascii=False, indent=1))
    json.dump(summary, open(os.path.join(out_dir, "r4_summary.json"), "w"), ensure_ascii=False, indent=1)
    return summary


def _safe(fn, *args, fail_f=None, fail_lock=None):
    src = args[1]   # both process_source(sel, src, ...) and process_source_local(sel, src, ...)
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001 - one bad source must not kill the run
        print(f"[skip] {os.path.basename(src['path'])}: {type(e).__name__}: {str(e)[:120]}")
        if fail_f is not None:
            rec = json.dumps({"source": src["path"], "stage": fn.__name__,
                              "error": f"{type(e).__name__}: {str(e)[:200]}"}, ensure_ascii=False)
            with (fail_lock or threading.Lock()):
                fail_f.write(rec + "\n"); fail_f.flush()
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--n", type=int, default=100)
    r.add_argument("--render-n", type=int, default=8)
    r.add_argument("--out", default="/home/bc/data/datasets/vera_directionA_1M/r4_pilot")
    r.add_argument("--local", action="store_true", help="local mask pipeline (Mask v2)")
    r.add_argument("--route", choices=["geom", "sam3"], default="geom",
                   help="geom=Route1 preset-tone-in-mask; sam3=Route2 LUT in SAM3 region")
    r.add_argument("--no-db", action="store_true", help="skip Postgres provenance (dry test)")
    r.add_argument("--src-workers", type=int, default=12,
                   help="sources in flight (lane overlap window; real resources are "
                        "bounded by the LR farm + vLLM admission gates)")
    r.add_argument("--fresh", action="store_true",
                   help="overwrite out_dir output instead of resuming from it")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.n, a.render_n, a.out, src_workers=a.src_workers, local=a.local,
            route=a.route, persist=not a.no_db, resume=not a.fresh)


if __name__ == "__main__":
    main()
