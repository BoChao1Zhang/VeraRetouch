"""Preset / LUT QA — two stages.

Stage 1 (all presets, cheap, no GPU):
  * parse validity (recipes.DiskRecipeParser by fmt)
  * param sanity (vlm_clean.heuristic_param_sane) + near-no-op detection
  * LUT non-identity (reject cubes that barely move colors)
  * near-duplicate dedup (param-vector / downsampled-grid signature -> dup_of head)
  * preset_content_hash (stable render cache key) on every survivor
  * has_local_mask -> status 'preset_meta_local' (routed to real LR only)

Stage 2 (survivors): REAL renders, not approximations.
  * param presets (xmp/lrtemplate, incl. the 290 local-mask ones) -> the JarvisEvo
    Lightroom task server (real LrC; masks honored).  [closes PRESET-1/PRESET-2]
  * LUT presets -> numpy trilinear (tier-3; LR has no .cube develop-preset form).
  Then paired before/after metrics (ΔE/SSIM/EMD/clip%) + NR-IQA on the real after
  + questionnaire C (region-local aware). Renders cached/keyed by
  (preset_content_hash, probe_id, engine) in render_jobs for idempotent resume.

Run: python -m dataset_build.source_qa.preset_qa stage1 [--limit N]
     python -m dataset_build.source_qa.preset_qa stage2 [--limit N] [--no-qc]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from typing import Dict, List, Optional, Tuple

from . import config, db

_PARSER = None


def _parser():
    global _PARSER
    if _PARSER is None:
        from dataset_build.recipes import DiskRecipeParser
        _PARSER = DiskRecipeParser()
    return _PARSER


# --------------------------------------------------------------------------- #
# Stage 1: metadata gate
# --------------------------------------------------------------------------- #
def _parse_preset(path: str, fmt: str):
    p = _parser()
    fmt = (fmt or "").lower()
    if fmt == "xmp":
        return "param", p.xmp_to_params(path)
    if fmt == "lrtemplate":
        return "param", p.lrtemplate_to_params(path)
    if fmt in ("cube", "3dl", "lut"):
        return "lut", p.load_cube(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xmp":
        return "param", p.xmp_to_params(path)
    if ext in (".cube", ".3dl"):
        return "lut", p.load_cube(path)
    if ext == ".lrtemplate":
        return "param", p.lrtemplate_to_params(path)
    raise ValueError(f"unknown fmt {fmt}/{ext}")


def _param_signature(params: Dict[str, Dict[str, float]]) -> str:
    items = sorted((k, round(float(v.get("value", 0.0)), 1)) for k, v in params.items()
                   if isinstance(v, dict))
    return hashlib.sha1(json.dumps(items).encode()).hexdigest()[:16]


def _content_hash(ptype: str, parsed) -> str:
    """Stable render cache key (full sha256). Canonical param dict / normalized grid."""
    import numpy as np
    if ptype == "param":
        items = sorted((k, round(float(v.get("value", 0.0)), 2)) for k, v in parsed.items()
                       if isinstance(v, dict))
        return hashlib.sha256(json.dumps(items).encode()).hexdigest()
    grid = np.asarray(parsed[0], dtype="float32")
    if float(grid.max()) > 1.5:
        grid = grid / 255.0
    return hashlib.sha256(np.round(grid, 4).tobytes()).hexdigest()


def _nonzero_params(params: Dict[str, Dict[str, float]]) -> int:
    return sum(1 for v in params.values()
              if isinstance(v, dict) and abs(float(v.get("value", 0.0) or 0.0)) > 1e-6)


_AI_MASK_PATTERNS = tuple(
    re.compile(pat, re.IGNORECASE)
    for pat in (
        r"Mask/(?:Sky|Subject|People|Person|Portrait|Background|Object|Objects)",
        r"(?:Select|Selection)(?:Sky|Subject|People|Person|Background|Object|Objects)",
        r"Semantic(?:Mask|Region)",
        r"AI(?:Mask|Selection|Denoise)",
        r"Sensei",
    )
)


def _has_ai_mask(recipe_path: str, fmt: str) -> bool:
    fmt = (fmt or "").lower()
    ext = os.path.splitext(recipe_path)[1].lower()
    if fmt not in ("xmp", "lrtemplate") and ext not in (".xmp", ".lrtemplate"):
        return False
    try:
        with open(recipe_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except OSError:
        return False
    return any(p.search(text) for p in _AI_MASK_PATTERNS)


def _lut_stats(cube) -> Tuple[float, str]:
    import numpy as np
    grid, dmin, dmax = cube
    grid = np.asarray(grid, dtype="float32")
    if grid.ndim == 2 and grid.shape[1] == 3:
        n = round(grid.shape[0] ** (1 / 3))
        grid = grid.reshape(n, n, n, 3)
    n = grid.shape[0]
    axis = np.linspace(0, 1, n, dtype="float32")
    ident = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    g = grid
    if float(g.max()) > 1.5:
        g = g / 255.0
    dev = float(np.abs(g - ident).max())
    sig = hashlib.sha1(np.round(g[:: max(1, n // 8)], 2).tobytes()).hexdigest()[:16]
    return dev, sig


def stage1(limit: Optional[int] = None) -> dict:
    conn = db.connect()
    run_id = db.start_run(conn, "preset_qa_stage1", {"limit": limit})
    where = ("asset_type='preset' AND NOT EXISTS (SELECT 1 FROM processing_events e "
             "WHERE e.asset_id=assets.asset_id AND e.stage='preset_meta')")
    sql = f"SELECT asset_id, path, fmt, kind, is_bw, is_technical, has_local_mask FROM assets WHERE {where}"
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql).fetchall()
    print(f"[preset stage1] {len(rows)} presets", file=sys.stderr)

    seen_sig: Dict[str, str] = {}
    counts = {"pass": 0, "fail": 0, "local": 0}
    reasons: Dict[str, int] = {}
    for i, r in enumerate(rows):
        aid, path, fmt = r["asset_id"], r["path"], r["fmt"]
        fail = None
        sig = content_hash = None
        detail: dict = {}
        try:
            ptype, parsed = _parse_preset(path, fmt)
            detail["ptype"] = ptype
            ai_mask = _has_ai_mask(path, fmt)
            detail["ai_mask"] = ai_mask
            if r["is_bw"]:
                fail = "is_bw"
            elif r["is_technical"]:
                fail = "is_technical"
            elif ptype == "param":
                from dataset_build.vlm_clean import heuristic_param_sane
                nz = _nonzero_params(parsed)
                detail["nonzero"] = nz
                if not heuristic_param_sane(parsed):
                    fail = "param_insane"
                elif nz == 0:
                    fail = "near_noop"
                else:
                    sig = _param_signature(parsed)
            else:  # lut
                dev, sig = _lut_stats(parsed)
                detail["lut_dev"] = round(dev, 4)
                if dev < 0.03:
                    fail = "lut_identity"
            if not fail:
                content_hash = _content_hash(ptype, parsed)
        except Exception as e:
            fail = "parse_error"
            detail["err"] = str(e)[:200]

        dup_head = None
        if not fail and sig is not None:
            if sig in seen_sig:
                dup_head = seen_sig[sig]
                detail["dup_of"] = dup_head
            else:
                seen_sig[sig] = aid

        if fail:
            counts["fail"] += 1
            reasons[fail] = reasons.get(fail, 0) + 1
            db.update_asset_fields(conn, aid, status="preset_meta_fail", auto_verdict="drop",
                                   dup_of=dup_head, has_ai_mask=1 if detail.get("ai_mask") else 0)
            db.log_event(conn, aid, "preset_meta", "ok", {"verdict": "fail", "reason": fail, **detail}, run_id)
        elif dup_head:
            counts["fail"] += 1
            reasons["dup"] = reasons.get("dup", 0) + 1
            db.update_asset_fields(conn, aid, status="preset_meta_fail", auto_verdict="drop",
                                   dup_of=dup_head, preset_content_hash=content_hash, has_ai_mask=1 if detail.get("ai_mask") else 0)
            db.log_event(conn, aid, "preset_meta", "ok", {"verdict": "dup", **detail}, run_id)
        else:
            # local-mask presets are routed to real LR only (global render is wrong)
            is_local = bool(r["has_local_mask"])
            status = "preset_meta_local" if is_local else "preset_meta_pass"
            counts["local" if is_local else "pass"] += 1
            db.update_asset_fields(conn, aid, status=status, auto_verdict="review",
                                   preset_content_hash=content_hash,
                                   has_ai_mask=1 if detail.get("ai_mask") else 0)
            db.log_event(conn, aid, "preset_meta", "ok",
                         {"verdict": ("pass_local" if is_local else "pass"),
                          "local_edit": is_local, "ai_mask": bool(detail.get("ai_mask")), **detail}, run_id)
        if (i + 1) % 2000 == 0:
            conn.commit()
            print(f"[preset stage1] {i+1}/{len(rows)} {counts}", file=sys.stderr)
    conn.commit()
    db.finish_run(conn, run_id, {"counts": counts, "fail_reasons": reasons})
    conn.close()
    print(json.dumps({"counts": counts, "fail_reasons": reasons}))
    return {"counts": counts, "fail_reasons": reasons}


# --------------------------------------------------------------------------- #
# Stage 2: real render + paired metrics + questionnaire C
# --------------------------------------------------------------------------- #
def resolve_probes(conn, k: int = None) -> List[dict]:
    """Deterministic, scene-spread probe set (stable probe_id => render cache hits).
    Honors config.PRESET_PROBE_IMAGES if pinned, else picks top-aesthetic per scene."""
    k = k or config.PRESET_PROBE_COUNT
    if config.PRESET_PROBE_IMAGES:
        rows = conn.execute(
            "SELECT asset_id, path, scene FROM assets WHERE asset_id = ANY(%s)",
            (list(config.PRESET_PROBE_IMAGES),)).fetchall()
        return [{"asset_id": r["asset_id"], "path": r["path"], "scene": r["scene"]} for r in rows]
    rows = conn.execute(
        "SELECT asset_id, path, scene FROM assets WHERE asset_type='image' AND aesthetic IS NOT NULL "
        "AND dup_of IS NULL AND corpus IN ('fivek_gold','quandian','korean') "
        "ORDER BY aesthetic DESC, asset_id LIMIT 400").fetchall()
    picked, scenes = [], set()
    for r in rows:
        if r["scene"] in scenes and len(picked) >= 1:
            continue
        picked.append({"asset_id": r["asset_id"], "path": r["path"], "scene": r["scene"]})
        scenes.add(r["scene"])
        if len(picked) >= k:
            break
    if not picked and rows:
        picked = [{"asset_id": r["asset_id"], "path": r["path"], "scene": r["scene"]} for r in rows[:k]]
    return picked


def _stage_probe(probe: dict, longedge: int = 1600) -> str:
    """Downscaled JPEG of the probe in RENDER_STAGE; submitted to LR + used as the
    'before' for paired metrics so before/after share resolution."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    out = os.path.join(config.RENDER_STAGE, f"probe_{probe['asset_id']}.jpg")
    if not os.path.exists(out):
        im = Image.open(probe["path"]); im.load(); im = im.convert("RGB")
        im.thumbnail((longedge, longedge))
        im.save(out, "JPEG", quality=95)
    return out


def _apply_cube(im, cube):
    """Trilinear-apply a cube LUT to a PIL RGB image (tier-3, LUT-only)."""
    import numpy as np
    grid, dmin, dmax = cube
    grid = np.asarray(grid, dtype="float32")
    if grid.ndim == 2 and grid.shape[1] == 3:
        n = round(grid.shape[0] ** (1 / 3))
        grid = grid.reshape(n, n, n, 3)
    if float(grid.max()) > 1.5:
        grid = grid / 255.0
    n = grid.shape[0]
    arr = np.asarray(im.convert("RGB"), dtype="float32") / 255.0
    dmin = np.asarray(dmin, dtype="float32"); dmax = np.asarray(dmax, dtype="float32")
    span = np.where((dmax - dmin) == 0, 1.0, (dmax - dmin))
    coords = np.clip((arr - dmin) / span, 0, 1) * (n - 1)
    lo = np.floor(coords).astype(int); hi = np.clip(lo + 1, 0, n - 1); fr = coords - lo
    def g(ix):
        return grid[ix[..., 0], ix[..., 1], ix[..., 2]]
    out = np.zeros_like(arr)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                idx = np.stack([np.where(np.array([dx, dy, dz])[c] == 1, hi[..., c], lo[..., c])
                                for c in range(3)], axis=-1)
                w = (np.where(dx, fr[..., 0], 1 - fr[..., 0]) *
                     np.where(dy, fr[..., 1], 1 - fr[..., 1]) *
                     np.where(dz, fr[..., 2], 1 - fr[..., 2]))
                out += g(idx) * w[..., None]
    from PIL import Image
    return Image.fromarray(np.clip(out * 255, 0, 255).astype("uint8"))


def _questionnaire_c(before_uri: str, after_uri: str, style: str, scene_aff: str,
                     region_local: bool) -> Optional[dict]:
    import requests, re, time
    C = config.QUESTIONNAIRE_C
    items = "\n".join(f"  {k} ({spec['type']}): {spec['q']}" for k, spec in C["items"].items())
    local_note = ("注意：该预设为【区域局部编辑（含 mask）】，after 已由真实 Lightroom 应用其 mask 渲染；"
                  "请就真实局部效果作答。\n" if region_local else "")
    prompt = (
        "下面两张图：第一张是 before(原图)，第二张是对其应用某『预设 look』后的真实 after。"
        f"该预设声称 style:`{style or '—'}` / scene_affinity:`{scene_aff or '—'}`。\n" + local_note +
        "请逐题回答（yn=true/false；yn3=yes/no/uncertain）并给一句理由：\n" + items + "\n"
        "只输出 JSON：{\"C1\":{\"answer\":true,\"why\":\"\"},\"C2\":{\"answer\":\"yes\",\"why\":\"\"},\"C3\":{\"answer\":true,\"why\":\"\"}}"
    )
    payload = {"model": config.VLLM_MODEL, "max_tokens": 400, "temperature": 0.1,
               "messages": [{"role": "user", "content": [
                   {"type": "image_url", "image_url": {"url": before_uri}},
                   {"type": "image_url", "image_url": {"url": after_uri}},
                   {"type": "text", "text": prompt}]}]}
    if not config.VLLM_ENABLE_THINKING:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Authorization": f"Bearer {config.VLLM_API_KEY}",
               "X-vgate-class": "qa-judge"}  # vGate priority (P1); ignored by plain vLLM
    # Bounded retry: questionnaire C had no retry, so a transient blip or a vGate
    # 429 (Retry-After) used to silently drop the result. Retry a few times with
    # backoff, honoring Retry-After when the broker sends it.
    txt = None
    for attempt in range(3):
        try:
            r = requests.post(config.VLLM_BASE_URL + "/chat/completions", json=payload,
                              headers=headers, timeout=120)
            if r.status_code == 429 and attempt < 2:
                time.sleep(float(r.headers.get("Retry-After", 2 * (attempt + 1))))
                continue
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"]
            break
        except Exception:
            if attempt == 2:
                return None
            time.sleep(2 * (attempt + 1))
    if txt is None:
        return None
    import json_repair
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
    try:
        obj = json_repair.loads(txt[txt.find("{"):] if "{" in txt else txt)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _img_uri(im) -> str:
    import base64, io
    buf = io.BytesIO(); im.convert("RGB").save(buf, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _job_id_for(conn, content_hash, probe_id, engine, recipe_id, region_local, fmt,
                preset_path, probe_path, run_id, ai_mask=False) -> str:
    jid = uuid.uuid4().hex
    db.add_render_job(conn, {"job_id": jid, "recipe_id": recipe_id,
                            "preset_content_hash": content_hash, "probe_id": probe_id,
                            "render_engine": engine, "region_local_flag": int(region_local),
                            "ai_mask_flag": int(ai_mask),
                            "fmt": fmt, "preset_path": preset_path, "probe_path": probe_path,
                            "status": "pending", "run_id": run_id})
    row = conn.execute("SELECT job_id FROM render_jobs WHERE preset_content_hash=? AND probe_id=? "
                       "AND render_engine=?", (content_hash, probe_id, engine)).fetchone()
    return row["job_id"] if row else jid


def run_stage2(limit: Optional[int] = None, run_qc: bool = True,
               workers: int = 6, probes_k: Optional[int] = None) -> dict:
    """Concurrent: presets are processed by a thread pool so the LR renders fan out
    across the connected LrC clients (serial submit_and_wait would idle 2/3 of them).
    DB access is serialized by a lock (one psycopg conn) and the shared GPU NR-IQA
    model by a gpu lock; the LR network wait + paired metrics + questionnaire C run
    in parallel. Resumable via the render_jobs cache."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    from . import paired_metrics
    from dataset_build.core import build_lr_client
    lr = build_lr_client()   # core.lr: LR pool handle (semaphore cap = config.LR_MAX_CONCURRENCY)
    _parser()  # pre-init the shared recipe parser (avoid races in worker threads)
    conn = db.connect()
    run_id = db.start_run(conn, "preset_qa_stage2", {"limit": limit, "workers": workers})
    os.makedirs(config.RENDER_STAGE, exist_ok=True)

    probes = resolve_probes(conn, probes_k)
    probe_pils = {}
    for p in probes:
        try:
            sp = _stage_probe(p)
            im = Image.open(sp); im.load(); im = im.convert("RGB")
            probe_pils[p["asset_id"]] = (p, sp, im)
        except Exception as e:
            print(f"[preset stage2] probe {p['asset_id']} failed: {e}", file=sys.stderr)
    print(f"[preset stage2] {len(probe_pils)} probes; LR health={lr.health()}", file=sys.stderr)

    iqa = None
    try:
        from .iqa import IQARunner
        iqa = IQARunner(metrics=["musiq", "clipiqa+"])
    except Exception as e:
        print(f"[preset stage2] IQA disabled: {e}", file=sys.stderr)

    # resume guard: skip presets that already produced a verdict (preset_render 'ok'
    # = rendered or near-noop); only (re)process those with no ok event yet, so a
    # retry hits exactly the previously-failed/unrendered presets, no duplicate QC.
    rows = conn.execute(
        "SELECT asset_id, path, fmt, kind, style, scene_affinity, has_local_mask, has_ai_mask, preset_content_hash "
        "FROM assets WHERE asset_type='preset' "
        "AND status IN ('preset_meta_pass','preset_meta_local','preset_render_failed','preset_needs_local_render') "
        "AND dup_of IS NULL "
        "AND NOT EXISTS (SELECT 1 FROM processing_events e WHERE e.asset_id=assets.asset_id "
        "AND e.stage='preset_render' AND e.status='ok')"
        + (f" LIMIT {int(limit)}" if limit else "")).fetchall()
    print(f"[preset stage2] {len(rows)} presets", file=sys.stderr)

    db_lock = threading.Lock()      # serialize the single psycopg connection
    gpu_lock = threading.Lock()     # serialize the shared NR-IQA GPU model
    counts = {"ok": 0, "noop": 0, "skip": 0}
    cnt_lock = threading.Lock()

    def _bump(k):
        with cnt_lock:
            counts[k] += 1

    def _process(r) -> None:
        aid = r["asset_id"]
        engine = "lrc" if r["kind"] == "param" else "lut_trilinear"
        region_local = bool(r["has_local_mask"])
        ai_mask = bool(r["has_ai_mask"])
        content_hash = r["preset_content_hash"] or aid
        cube = None
        if engine == "lut_trilinear":
            try:
                cube = _parser().load_cube(r["path"])
            except Exception as e:
                with db_lock:
                    db.log_event(conn, aid, "preset_render", "error", {"err": str(e)[:200]}, run_id)
                    conn.commit()
                _bump("skip"); return
        outdir = os.path.join(config.PREVIEW_DIR, aid)
        os.makedirs(outdir, exist_ok=True)

        paired_all, c_results = [], []
        rendered = 0
        for pid, (p, sp, before_im) in probe_pils.items():
            with db_lock:
                cached = db.cached_render(conn, content_hash, pid, engine)
            after_path = job_id = None
            if cached and cached["after_jpg_path"] and os.path.exists(cached["after_jpg_path"]):
                after_path, job_id = cached["after_jpg_path"], cached["job_id"]
            else:
                with db_lock:
                    job_id = _job_id_for(conn, content_hash, pid, engine, aid, region_local,
                                         r["fmt"], r["path"], sp, run_id, ai_mask=ai_mask)
                    conn.commit()
                if engine == "lrc":
                    res = lr.render(r["path"], r["fmt"], sp)   # core.lr: pool-capped network render
                    render_ok = bool(res and res.get("ok"))
                    after_path = res["after_path"] if render_ok else None
                    render_error = None if render_ok else json.dumps(res or {"error": "lr render failed"},
                                                                     ensure_ascii=False)[:1000]
                    with db_lock:
                        db.mark_job(conn, job_id, "done" if render_ok else "error",
                                    after_jpg_path=after_path,
                                    engine_version=(res or {}).get("engine_version"),
                                    node="lrc", error=render_error)
                        conn.commit()
                else:
                    try:
                        after_im = _apply_cube(before_im, cube)
                        after_path = os.path.join(outdir, f"{pid}_after.jpg")
                        after_im.save(after_path, "JPEG", quality=95)
                        with db_lock:
                            db.mark_job(conn, job_id, "done", after_jpg_path=after_path, node="numpy")
                            conn.commit()
                    except Exception as e:
                        with db_lock:
                            db.mark_job(conn, job_id, "error", error=str(e)[:200]); conn.commit()
            if not after_path:
                continue
            try:
                after_im = Image.open(after_path); after_im.load(); after_im = after_im.convert("RGB")
            except Exception:
                continue
            bpath = os.path.join(outdir, f"{pid}_before.jpg")
            if not os.path.exists(bpath):
                before_im.save(bpath, "JPEG", quality=92)
            pm = paired_metrics.paired(before_im, after_im)
            paired_all.append(pm)
            after_iqa = {}
            if iqa is not None:
                try:
                    with gpu_lock:
                        after_iqa = iqa.score_path(after_path)
                except Exception:
                    after_iqa = {}
            with db_lock:
                db.add_preset_preview(conn, aid, pid, bpath, after_path, after_iqa, pm,
                                      engine, int(region_local), job_id, run_id, ai_mask=int(ai_mask))
                conn.commit()
            rendered += 1
            if run_qc:
                qc = _questionnaire_c(_img_uri(before_im), _img_uri(after_im),
                                      r["style"], r["scene_affinity"], region_local)
                if qc:
                    c_results.append(qc)

        if rendered == 0:
            new_status = "preset_needs_local_render" if region_local else "preset_render_failed"
            with db_lock:
                db.update_asset_fields(conn, aid, status=new_status,
                                       auto_verdict="needs_local_render" if region_local else "review")
                db.log_event(conn, aid, "preset_render", "skip",
                             {"reason": "no render produced", "engine": engine, "local": region_local}, run_id)
                conn.commit()
            _bump("skip"); return

        mean_de = sum(pm["delta_e2000_mean"] for pm in paired_all) / len(paired_all)
        if mean_de < 1.5 or all(pm["noop_score"] == 1 for pm in paired_all):
            with db_lock:
                db.update_asset_fields(conn, aid, pass_c=0, auto_verdict="drop")
                db.log_event(conn, aid, "preset_render", "ok",
                             {"verdict": "near_noop", "mean_deltaE": round(mean_de, 3)}, run_id)
                conn.commit()
            _bump("noop"); return

        if c_results:
            items, passes = {}, []
            for k in config.QUESTIONNAIRE_C["items"]:
                yes = sum(1 for q in c_results
                          if (q.get(k) or {}).get("answer") in (True, "true", "yes", 1))
                ans = 1 if yes * 2 >= len(c_results) else 0
                why = next((str((q.get(k) or {}).get("why", "")) for q in c_results if q.get(k)), "")
                items[k] = {"answer": ans, "rationale": why[:300]}
                passes.append(ans)
            pass_c = 1 if all(pp == 1 for pp in passes) else 0
            with db_lock:
                db.add_qa(conn, aid, "C", items, model=config.VLLM_MODEL, run_id=run_id)
                db.update_asset_fields(conn, aid, pass_c=pass_c,
                                       auto_verdict=("review" if pass_c else "drop"))
                conn.commit()
        else:
            with db_lock:
                db.update_asset_fields(conn, aid, auto_verdict="review"); conn.commit()
        with db_lock:
            db.log_event(conn, aid, "preset_render", "ok",
                         {"engine": engine, "rendered": rendered,
                          "mean_deltaE": round(mean_de, 3), "local": region_local,
                          "ai_mask": ai_mask}, run_id)
            conn.commit()
        _bump("ok")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_process, r) for r in rows]
        for j, f in enumerate(as_completed(futs)):
            try:
                f.result()
            except Exception as e:
                print(f"[preset stage2] worker error: {e}", file=sys.stderr)
            if (j + 1) % 25 == 0:
                print(f"[preset stage2] {j+1}/{len(rows)} {counts}", file=sys.stderr)
    with db_lock:
        db.finish_run(conn, run_id, counts)
    conn.close()
    print(json.dumps(counts))
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stage1", "stage2"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-qc", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--probes", type=int, default=None)
    args = ap.parse_args()
    if args.stage == "stage1":
        stage1(limit=args.limit)
    else:
        run_stage2(limit=args.limit, run_qc=not args.no_qc, workers=args.workers, probes_k=args.probes)


if __name__ == "__main__":
    main()
