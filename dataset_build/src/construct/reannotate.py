"""存量样本离线重标注：把 v1 期产出的 instruction/reasoning 升级到 annotate v2
（img caption 条件化 + 三方面 CoT + metric 注入 + persona 去模板化）。

只改文本字段（instruction/instruction_short/reasoning/annot_src=vlm_v2），
answer/recipe/图像路径/qa 数值全部不动。输出写 <in>.v2.jsonl（原文件不动），
resume 按行号断点续跑（.v2 已有行数即偏移）。

用（iaa437 venv 非必需——不加载打分模型，base python 也可；vLLM 需在线）：
  PYTHONPATH=... python -m construct.reannotate --in .../sft.jsonl --kind global [--limit 500]
  PYTHONPATH=... python -m construct.reannotate --in .../degrade_v3/samples.jsonl --kind degrade [--limit 500]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))


def _one_global(r: dict) -> dict:
    from construct import annotate
    qa = r.get("qa") or {}
    metrics = {k: v for k, v in {
        "source_iaa": qa.get("source_iaa"), "after_iaa": qa.get("iaa_mixed"),
        "q": qa.get("q")}.items() if v is not None}
    local = r.get("local") or {}
    ann = annotate.annotate_winner(
        r["I_tar"], r.get("task_type", "auto"),
        None if r.get("task_type") in ("auto", "param") else {"vlm_name": None},
        local_region=None, local_params=None, source_path=r.get("I_in"),
        img_caption=annotate.source_caption(r.get("I_in")), metrics=metrics or None)
    out = dict(r)
    out.update({"instruction": ann["instruction_long"],
                "instruction_short": ann.get("instruction_short"),
                "reasoning": ann.get("reasoning", ""), "annot_src": "vlm_v2"})
    return out


def _one_degrade(r: dict) -> dict:
    from construct import annotate
    qa = r.get("qa") or {}
    spec = ((r.get("recipe") or {}).get("degrade") or {})
    metrics = {k: v for k, v in {
        "degrade_de": qa.get("degrade_de"), "source_iaa": qa.get("source_iaa")}.items()
        if v is not None}
    ann = annotate.annotate_winner(
        r["before"], "auto", None,
        degrade_info={"aspects": spec.get("aspects") or [], "ops": sorted(spec.get("op_params") or {})},
        source_path=r.get("after"),
        img_caption=annotate.source_caption(r.get("after")), metrics=metrics or None)
    out = dict(r)
    out.update({"instruction": ann["instruction_long"],
                "instruction_short": ann.get("instruction_short"),
                "reasoning": ann.get("reasoning", ""), "annot_src": "vlm_v2"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--kind", choices=("global", "degrade"), required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    fn = _one_global if a.kind == "global" else _one_degrade
    dst = a.inp + ".v2.jsonl" if not a.inp.endswith(".jsonl") else a.inp[:-6] + ".v2.jsonl"
    done = sum(1 for _ in open(dst)) if os.path.exists(dst) else 0
    rows = [json.loads(l) for l in open(a.inp)]
    todo = rows[done: done + a.limit if a.limit else None]
    print(f"[reann] 已有 {done}，本次 {len(todo)}", flush=True)
    ok = fb = 0
    t0 = time.time()

    def _safe(r):
        try:
            return fn(r)
        except Exception as e:  # noqa: BLE001 - 单条失败保留 v1 文本，标记回退
            out = dict(r)
            out["annot_src"] = f"v1_kept:{type(e).__name__}"
            return out

    with open(dst, "a") as f, ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, out in enumerate(ex.map(_safe, todo)):
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            ok += out["annot_src"] == "vlm_v2"
            fb += out["annot_src"] != "vlm_v2"
            if (i + 1) % 100 == 0:
                f.flush()
                el = time.time() - t0
                print(f"[reann] {i+1}/{len(todo)} v2={ok} 回退={fb} ({(i+1)/el*60:.0f}/min)", flush=True)
    print(json.dumps({"v2": ok, "kept_v1": fb, "wall_min": round((time.time()-t0)/60, 1)}))


if __name__ == "__main__":
    main()
