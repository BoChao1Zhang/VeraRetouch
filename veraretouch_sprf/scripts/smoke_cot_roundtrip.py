#!/usr/bin/env python3
"""EPR-052 自检：cot_text 一条记录的序列化/反序列化往返 + 主线包与 legacy 的 template_sha256 相等。"""
from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from veraretouch_sprf import _paths as _P
from veraretouch_sprf.data import cot_text as C

rec_path = _P.SPRF_LEGACY / "pack_teacher_20260906/samples/cot_records_2.jsonl"
rec = json.loads(rec_path.read_text().splitlines()[0])
t = C.target_text(rec)
p = C.parse_target_text(t)
a = rec["answer"]["cot"]
exact = all(p["steps"][i]["observation"] == a[i]["observation"].strip()
            and p["steps"][i]["mask"] == a[i]["mask"].strip()
            and p["steps"][i]["adjustment"] == a[i]["adjustment"].strip() for i in range(6))
segs = C.target_segments(rec)
recon = "\n".join(s + C.STAGE_TOKENS[i] for i, s in enumerate(segs))
instr, tier = C.instruction_for(rec)
spec = importlib.util.spec_from_file_location("cot_text_legacy", _P.VLMSFT_LEGACY / "cot_text.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
out = dict(key=rec["key"], chars=len(t), roundtrip_ok=p["ok"], n_segments=p["n_segments"],
           field_exact=exact, segments_reconstruct_equal=(recon == t),
           instruction_tier=tier, instruction_chars=len(instr),
           template_sha256_pkg=C.template_sha256(), template_sha256_legacy=L.template_sha256())
out["template_sha_equal"] = out["template_sha256_pkg"] == out["template_sha256_legacy"]
print(json.dumps(out, indent=1))
if not (p["ok"] and exact and recon == t and out["template_sha_equal"]):
    sys.exit("cot_text roundtrip FAILED")
