"""离线重打重建：对打分污染时段的 groups 重新逐张打分并重跑 tier。

背景（2026-07-06 事故）：批打分 + ArtiMuse 加载期 shim 竞争把候选 iaa 系统性压低
12-19 分，~08:47-11:40 产出的 groups 的 SFT 产率 1.7→0.11/组。候选渲染文件都在
（content-addressed store），只需重打分 + 重跑 tier 判定即可回收损失的 SFT/DPO。

输出写 sft_recovered.jsonl / dpo_recovered.jsonl（不与在跑的 build 竞争句柄），
验收时与主文件合并计数。qa 字段带 retier=true 溯源。

用（iaa437 venv，独立进程打分干净）：
  SOURCE_QA_IAA_DEVICE=cuda:1 /home/bc/.venvs/iaa437/bin/python -m construct.retier \
      --out /home/bc/data/datasets/vera_directionA_1M/construct_global_v3 \
      --from-idx 250 --to-idx 2600
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--from-idx", type=int, default=0)
    ap.add_argument("--to-idx", type=int, default=10**9)
    ap.add_argument("--q-floor", type=float, default=None, help="缺省用 tier.TAU_SFT")
    a = ap.parse_args()
    from construct import objscore, tier, qa as qmod

    lines = open(os.path.join(a.out, "groups.jsonl")).readlines()
    seg = lines[a.from_idx: a.to_idx]
    # 已在主 sft 里的 (source, preset) 不重复回收
    have = set()
    for f in ("sft.jsonl", "sft_recovered.jsonl"):
        p = os.path.join(a.out, f)
        if os.path.exists(p):
            for l in open(p):
                r = json.loads(l)
                have.add((r.get("source") or r.get("source_path"), r.get("preset_id")))
    tau = a.q_floor if a.q_floor is not None else tier.TAU_SFT

    sf = open(os.path.join(a.out, "sft_recovered.jsonl"), "a")
    n_re = n_sft = 0
    for li, l in enumerate(seg):
        g = json.loads(l)
        cands = [c for c in g.get("candidates", [])
                 if c.get("after_path") and os.path.exists(c["after_path"])]
        if not cands:
            continue
        src_iaa = objscore.mixed_value(objscore.score(g["source"]))
        rescored = []
        for c in cands:
            obj = objscore.score(c["after_path"])
            iaa = objscore.mixed_value(obj)
            q, _ = qmod._iaa_rank_q(iaa, src_iaa)
            old_veto = bool((c.get("qa") or {}).get("veto"))
            rescored.append((q, iaa, c, old_veto))
        n_re += 1
        rescored.sort(key=lambda t: -t[0])
        picked = 0
        for q, iaa, c, old_veto in rescored:
            if picked >= 2 or q < tau:
                break                      # 已按 q 降序，后面只会更低
            if old_veto:
                continue                   # det 极端 veto 与分数无关，保留原判
            key = (g["source"], c["preset_id"])
            if key in have:
                continue
            qa_new = dict(c.get("qa") or {})
            qa_new.update({"q": round(q, 4), "iaa_mixed": iaa and round(iaa, 3),
                           "source_iaa": src_iaa and round(src_iaa, 3), "retier": True})
            c2 = dict(c); c2["qa"] = qa_new
            rec = tier.make_sft_record(g, c2, picked, tier._caps())
            sf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            have.add(key)
            picked += 1
            n_sft += 1
        if n_re % 100 == 0:
            sf.flush()
            print(f"[retier] {n_re}/{len(seg)} 组，回收 sft={n_sft}", flush=True)
    sf.close()
    print(json.dumps({"groups": n_re, "sft_recovered": n_sft}))


if __name__ == "__main__":
    main()
