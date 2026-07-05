"""R4 end-to-end pipeline: source photo -> vlemb recall -> render N -> 12-dim QA -> tier records.

Locked design (after R1/R2/R3):
  recall  = vlemb: source VL image-embed, centered, cosine vs centered preset-caption embeds -> top-N
            (R2: beats random/reranker across all saturation bands)
  render  = render.render_preset (param->LR farm, lut->trilinear)
  QA      = qa.qa_rank (12-dim binary F⊕R, JSON output, 2-phase, det cross-validated) -> veto + merit
  tier    = tier.build (top-2 by merit -> SFT; whole group -> DPO chosen/rejected)

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

import numpy as np

from dataset_build.source_qa import db, config
from . import sf_client, render, qa, tier, mask_synth, mask_sam3
from .bank import PresetBank

FULL = "/home/bc/data/datasets/vera_directionA_1M/preset_bank_full"
_BG_MEAN = os.path.join(FULL, "img_bg_mean.npy")


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


class Selector:
    """vlemb selector: centered cross-modal cosine (source image ↔ preset caption embeds)."""

    def __init__(self):
        self.bank = PresetBank.load(FULL)
        z = np.load(os.path.join(FULL, "text_emb.vlm_plain.npz"), allow_pickle=True)
        self.ids = list(z["ids"])
        pemb = z["emb"].astype("float32")
        self.pmu = pemb.mean(0)
        self.pemb_c = _unit(pemb - self.pmu)
        self.idx = {pid: i for i, pid in enumerate(self.ids)}
        self.feat = {f["preset_id"]: f for f in self.bank.feats}
        self.img_mu = self._bg_mean()

    def _bg_mean(self):
        if os.path.exists(_BG_MEAN):
            return np.load(_BG_MEAN)
        conn = db.connect()
        min_iaa = getattr(config, "CONSTRUCT_SOURCE_IAA_MIN", config.GATE["iaa_keep_above"])
        rows = conn.execute("SELECT path FROM assets WHERE asset_type='image' AND b_quality=3 "
                            "AND iaa_mixed IS NOT NULL AND iaa_mixed >= ? "
                            "AND dup_of IS NULL ORDER BY asset_id LIMIT 4000", (min_iaa,)).fetchall()
        conn.close()
        paths = [r["path"] for r in rows if os.path.exists(r["path"])]
        paths = paths[::max(1, len(paths) // 256)][:256]
        mu = sf_client.embed_images(paths).mean(0)
        np.save(_BG_MEAN, mu)
        return mu

    def recall(self, src_path: str, k: int) -> list:
        v = sf_client.embed_images([src_path])[0]
        vc = _unit(v - self.img_mu)
        sims = self.pemb_c @ vc
        top = np.argsort(-sims)[:k]
        return [self.feat[self.ids[i]] for i in top]


def process_source(sel: Selector, src: dict, render_n: int, render_workers: int = 6) -> dict:
    """One source through the full pipeline. Returns the processed group (candidates + qa)."""
    path = src["path"]
    cands = sel.recall(path, render_n)

    def _r(f):
        res = render.render_preset(f["path"], f["kind"], f.get("fmt"), path)
        return (f, res["after_path"]) if res.get("ok") else None
    with ThreadPoolExecutor(max_workers=render_workers) as ex:
        rendered = [x for x in ex.map(_r, cands) if x]
    variants = [(f["preset_id"], ap) for f, ap in rendered]
    is_portrait = bool(src.get("is_portrait_pool"))
    qres = qa.qa_rank(path, variants, is_portrait=is_portrait)
    after = {f["preset_id"]: ap for f, ap in rendered}
    return {
        "source": path, "source_asset_id": src.get("asset_id"), "is_portrait": is_portrait,
        "source_iaa": src.get("iaa_mixed"),
        "candidates": [{"preset_id": f["preset_id"], "kind": f["kind"], "fmt": f.get("fmt"),
                        "preset_path": f["path"], "content_hash": f.get("preset_content_hash"),
                        "after_path": after[f["preset_id"]],
                        "qa": qres["scores"].get(f["preset_id"])} for f, _ in rendered],
    }


_MASK_BANK = None


def _mask_bank():
    global _MASK_BANK
    if _MASK_BANK is None:
        import os as _os
        _MASK_BANK = mask_synth.load_bank() if _os.path.exists(mask_synth._BANK_PATH) else mask_synth.build_bank()
    return _MASK_BANK


def process_source_local(sel: "Selector", src: dict, n_masks: int, cgt_dir: str,
                         render_workers: int = 6) -> dict | None:
    """LOCAL pipeline (Mask v2 Route 1): the GLOBAL-selected preset's look applied ONLY inside a mask.
    vlemb picks an appropriate param-XMP preset (the look); we sample n mask regions and render the
    preset confined to each; QA picks the best region. local={mask_unit_id, geom, C_GT, ...}."""
    path = src["path"]
    # base preset = top vlemb param-XMP candidate (needs an .xmp to inject the mask into)
    base = next((c for c in sel.recall(path, 20)
                 if c.get("kind") == "param" and str(c.get("path", "")).endswith(".xmp")), None)
    if not base:
        return None
    bank = _mask_bank()
    # sha1, not builtin hash(): PYTHONHASHSEED randomizes hash() per process, which
    # would sample different mask geometries for the same source across restarts.
    rng = random.Random(int.from_bytes(hashlib.sha1(path.encode()).digest()[:4], "big"))

    def _one(_i):
        g = mask_synth.sample_geom(bank, rng)
        geom = mask_synth.perturb(g["geom"], rng)
        geom["__what__"] = g["what"]; geom["__type__"] = g["mask_type"]
        s = mask_synth.make_local_sample(path, base["path"], geom, cgt_dir, rng)
        return (g, s) if s else None
    with ThreadPoolExecutor(max_workers=render_workers) as ex:
        samples = [x for x in ex.map(_one, range(n_masks)) if x]
    variants = [(s["mask_unit_id"], s["after_path"]) for _, s in samples]
    is_portrait = bool(src.get("is_portrait_pool"))
    qres = qa.qa_rank(path, variants, is_portrait=is_portrait) if variants else {"scores": {}}
    return {
        "source": path, "source_asset_id": src.get("asset_id"), "is_portrait": is_portrait,
        "source_iaa": src.get("iaa_mixed"), "local": True,
        "candidates": [{"preset_id": s["mask_unit_id"], "kind": "local_from_preset",
                        "after_path": s["after_path"], "qa": qres["scores"].get(s["mask_unit_id"]),
                        "local": {"mask_unit_id": s["mask_unit_id"], "mask_type": g["mask_type"],
                                  "geom": {k: v for k, v in s["geom"].items() if not k.startswith("__")},
                                  "cgt_path": s["cgt_path"], "region": s["region"],
                                  "local_params": s["local_params"],
                                  "base_preset_id": base["preset_id"], "base_preset_path": base["path"]}}
                       for g, s in samples],
    }


def process_source_sam3(sel: "Selector", src: dict, n_masks: int, cgt_dir: str,
                        render_workers: int = 6) -> dict | None:
    """LOCAL Route 2: a vlemb-selected LUT composited into a SAM3 semantic region (numpy). Candidates
    = the LUT applied to the top-K SAM3 concepts; QA picks the best region. local C_GT = SAM3 mask."""
    path = src["path"]
    if not mask_sam3.has_cache(path):
        return None
    lut = next((c for c in sel.recall(path, 30) if c.get("kind") == "lut"), None)
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
    sel = Selector()   # both routes need vlemb (global: the look; local: the base preset for the mask)
    route_label = (route if local else "global")
    conn = db.connect()
    run_id = db.start_run(conn, f"construct_{route_label}",
                          {"n": n, "render_n": render_n, "out": out_dir, "local": local,
                           "route": route_label}) if persist else None
    conn.commit()
    # Route 1 (geometric mask-only) applies to ANY photo; b_subject is for Route 2 (SAM3 semantic).
    min_iaa = getattr(config, "CONSTRUCT_SOURCE_IAA_MIN", config.GATE["iaa_keep_above"])
    rows = [dict(r) for r in conn.execute(
        "SELECT asset_id, path, is_portrait_pool, iaa_mixed FROM assets WHERE asset_type='image' "
        "AND b_quality=3 AND iaa_mixed IS NOT NULL AND iaa_mixed >= ? "
        "AND dup_of IS NULL AND saturation_mean IS NOT NULL ORDER BY asset_id",
        (min_iaa,)).fetchall()]
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
