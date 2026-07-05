"""Track-B 退化数据构建器 v1：全图高斯算子退化 → (退化图, restore 参数, GT=源图)。

论文 §3.1 Part B 的落地（σ 已逐值对齐 Tab 8，见 recipes.SIGMA_PROFILES）：
  1. 源图从 keep 池按场景分层抽（mixing.stratified_sources，iaa_mixed 门槛）；
  2. 7 种 L/GC/SC 组合加权采样 → sample_degrade_spec（seed 确定性，可复渲）；
  3. 退化渲染走 render_backend（throughput 策略 → gpu_render 本地 batch16）；
  4. 质量门：退化幅度 ΔE00∈[gate_lo, gate_hi]（太小无学习信号、太大不可逆）+
     裁剪增量门（高光/阴影新增溢出 <25%，防不可逆信息损失）；
  5. instruction/reasoning：construct.annotate 可用则单次 VLM 调用（degrade_info
     条件化，防泄露 guard），否则安全模板回退（用户抱怨式，不含任何参数）。
  answer = spec.op_params（restore 方向原值），GT 图 = 源图本身。

Track-A（E+R 专家反演）明确不在本构建器范围（遗留，见 RESIDUAL/探索报告）。
region-local（S1 式 mask 退化）为 v2 计划，本 v1 只产 global。

用法（cwd=/home/bc/VeraRetouch，需 RENDER_BACKEND_POLICY=throughput）：
  python -m construct.degrade run --n 100 --out /home/bc/data/datasets/vera_directionA_1M/degrade_v3
  python -m construct.degrade smoke   # 3 张冒烟（打印门与产物）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import threading
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))          # src/
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))  # repo 根

import numpy as np

# 7 组合权重：单 aspect 略降权，多 aspect 组合（更接近真实退化）加权
COMBOS = ["L", "GC", "SC", "L+GC", "L+SC", "GC+SC", "L+GC+SC"]
COMBO_W = [1.0, 1.0, 1.0, 1.4, 1.4, 1.4, 1.8]
GATE_DE = (1.5, 18.0)      # 退化幅度门（下采样 ΔE00 均值）
GATE_CLIP = 0.25           # 新增溢出像素比例门
VARIANTS_PER_SOURCE = 2


def _sid(source_id: str, i: int) -> str:
    return "dg_" + hashlib.sha1(f"{source_id}|{i}|v3".encode()).hexdigest()[:16]


def _de_ds(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.color import deltaE_ciede2000, rgb2lab
    return float(deltaE_ciede2000(rgb2lab(a[::4, ::4]), rgb2lab(b[::4, ::4])).mean())


def _load01(p: str, size=None) -> np.ndarray:
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    im = Image.open(p).convert("RGB")
    if size and im.size != size:
        im = im.resize(size)
    return np.asarray(im, dtype=np.float32) / 255.0


def _clip_delta(src: np.ndarray, deg: np.ndarray) -> float:
    """退化新增的溢出像素比例（高光≥0.995 或阴影≤0.005）。"""
    hi = float((deg >= 0.995).any(-1).mean() - (src >= 0.995).any(-1).mean())
    lo = float((deg <= 0.005).all(-1).mean() - (src <= 0.005).all(-1).mean())
    return max(hi, 0.0) + max(lo, 0.0)


def _fallback_instruction(rng: random.Random) -> Dict[str, str]:
    """annotate 不可用时的安全模板：用户抱怨式，不含任何参数/方向词。"""
    opts = ["这张照片看起来不太对劲，帮我修一下。", "这张图的观感有点问题，请帮我调整到自然的状态。",
            "帮我把这张照片恢复正常。", "这张照片拍完感觉色调/光线不太舒服，请修复。"]
    return {"instruction": rng.choice(opts),
            "reasoning": ""}   # 模板回退不产 reasoning（宁缺毋泄露）


def _annotate(degraded_path: str, spec) -> Optional[Dict[str, str]]:
    try:
        from construct.annotate import annotate_winner
    except Exception:
        return None
    try:
        r = annotate_winner(degraded_path, task_type="auto", preset_meta=None,
                            degrade_info={"aspects": spec.aspects,
                                          "ops": sorted(spec.op_params)})
        if r and r.get("instruction_long"):
            return {"instruction": r["instruction_long"],
                    "instruction_short": r.get("instruction_short", ""),
                    "reasoning": r.get("reasoning", "")}
    except Exception as e:  # noqa: BLE001
        print(f"[degrade] annotate 失败回退模板: {e}", file=sys.stderr)
    return None


def build(n_sources: int, out_dir: str, min_iaa: float = 55.0,
          variants: int = VARIANTS_PER_SOURCE, use_annotate: bool = True,
          workers: int = 8) -> dict:
    os.environ.setdefault("RENDER_BACKEND_POLICY", "throughput")
    from dataset_build.source_qa import db
    from dataset_build.recipes import DiskRecipeParser
    from dataset_build.core.render_backend import get_backend, params_to_xmp
    from construct.mixing import stratified_sources

    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    shard_p = os.path.join(out_dir, "samples.jsonl")
    done = set()
    if os.path.exists(shard_p):
        with open(shard_p) as f:
            done = {json.loads(l)["sample_id"] for l in f if l.strip()}

    conn = db.connect()
    sources = stratified_sources(conn, n_sources, min_iaa=min_iaa)
    conn.close()
    parser = DiskRecipeParser(sigma_profile="aether_tab8")
    be = get_backend()
    stats = {"ok": 0, "gate_de": 0, "gate_clip": 0, "render_fail": 0, "skip_done": 0,
             "annotate_vlm": 0, "annotate_tpl": 0}
    t0 = time.time()
    slock = threading.Lock()

    def _one_source(src) -> list:
        recs = []
        src_np = None
        for i in range(variants):
            sid = _sid(src["asset_id"], i)
            if sid in done:
                with slock:
                    stats["skip_done"] += 1
                continue
            seed = int(hashlib.sha1(sid.encode()).hexdigest()[:8], 16)
            rng = random.Random(seed)
            combo = rng.choices(COMBOS, weights=COMBO_W, k=1)[0]
            spec = parser.sample_degrade_spec(combo.split("+"), seed, region_local=False)
            if not spec.op_params:
                continue
            deg_p = os.path.join(out_dir, "images", f"{sid}.jpg")
            with tempfile.NamedTemporaryFile(suffix=".xmp", delete=False) as tf:
                xmp = params_to_xmp(parser.degrade_spec_to_params(spec), tf.name)
            try:
                r = be.render_one(xmp, "xmp", src["path"], out_path=deg_p)
            finally:
                os.unlink(xmp)
            if not r.get("ok"):
                with slock:
                    stats["render_fail"] += 1
                continue
            if src_np is None:
                src_np = _load01(src["path"])
            deg_np = _load01(deg_p, size=(src_np.shape[1], src_np.shape[0]))
            de = _de_ds(src_np, deg_np)
            if not (GATE_DE[0] <= de <= GATE_DE[1]):
                with slock:
                    stats["gate_de"] += 1
                os.unlink(deg_p)
                continue
            clip = _clip_delta(src_np, deg_np)
            if clip > GATE_CLIP:
                with slock:
                    stats["gate_clip"] += 1
                os.unlink(deg_p)
                continue
            ann = _annotate(deg_p, spec) if use_annotate else None
            with slock:
                stats["annotate_vlm" if ann else "annotate_tpl"] += 1
            if not ann:
                ann = _fallback_instruction(rng)
            recs.append({
                "sample_id": sid, "task": "auto_restore",
                "source_id": src["asset_id"], "scene": src["scene"],
                "before": deg_p, "after": src["path"],
                "instruction": ann["instruction"],
                "reasoning": ann.get("reasoning", ""),
                "answer": {k: v for k, v in spec.op_params.items()},
                "recipe": {"kind": "degrade", "degrade": asdict(spec)},
                "qa": {"degrade_de": round(de, 3), "clip_delta": round(clip, 4),
                       "source_iaa": float(src["iaa_mixed"]), "engine": r.get("engine")},
            })
        return recs

    # 线程并行：GPU 渲染由 render_backend 内部互斥串行，annotate(HTTP)/解码/ΔE 门并行，
    # 流水线重叠。写盘单线程（主线程消费 futures）。
    from concurrent.futures import ThreadPoolExecutor
    with open(shard_p, "a") as out_f, ThreadPoolExecutor(max_workers=workers) as ex:
        for recs in ex.map(_one_source, sources):
            for rec in recs:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["ok"] += 1
            out_f.flush()
            if stats["ok"] and stats["ok"] % 100 < len(recs):
                el = time.time() - t0
                print(f"[degrade] ok={stats['ok']} ({stats['ok']/el*60:.0f}/min) {stats}", flush=True)
    stats["wall_min"] = round((time.time() - t0) / 60, 1)
    stats["backend"] = be.stats_snapshot()
    print(json.dumps(stats, ensure_ascii=False))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--n", type=int, default=100, help="源图数（样本≈n×2 过门率）")
    p_run.add_argument("--out", default="/home/bc/data/datasets/vera_directionA_1M/degrade_v3")
    p_run.add_argument("--min-iaa", type=float, default=55.0)
    p_run.add_argument("--variants", type=int, default=VARIANTS_PER_SOURCE)
    p_run.add_argument("--no-annotate", action="store_true")
    p_run.add_argument("--workers", type=int, default=8)
    sub.add_parser("smoke")
    a = ap.parse_args()
    if a.cmd == "smoke":
        build(3, "/tmp/claude-1001/-home-bc-VeraRetouch/degrade_smoke", use_annotate=False)
    else:
        build(a.n, a.out, min_iaa=a.min_iaa, variants=a.variants,
              use_annotate=not a.no_annotate, workers=a.workers)


if __name__ == "__main__":
    main()
