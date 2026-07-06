# Precompute probe targets from (input, gt) pairs for all manifest samples.
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from common import RESULTS, build_manifest, load_manifest, compute_targets

def one(r):
    try:
        t = compute_targets(r["input_path"], r["gt_path"])
        return r["key"], {k: v.tolist() for k, v in t.items()}
    except Exception as e:
        print("FAIL", r["key"], e)
        return r["key"], None

if __name__ == "__main__":
    man_path = os.path.join(RESULTS, "manifest.json")
    if not os.path.exists(man_path):
        os.makedirs(RESULTS, exist_ok=True)
        with open(man_path, "w", encoding="utf-8") as f:
            json.dump(build_manifest(), f, ensure_ascii=False, indent=1)
    rows = load_manifest(man_path)
    out = {}
    with ProcessPoolExecutor(max_workers=16) as ex:
        for key, t in ex.map(one, rows):
            if t is not None:
                out[key] = t
    with open(os.path.join(RESULTS, "probe_targets.json"), "w") as f:
        json.dump(out, f)
    print("targets for", len(out), "samples")
