"""Re-evaluate the frozen A0 rec-arm checkpoints on the 2^21 held-out colour
subsample, so the hypothesis-(a) table compares rec vs the E1 engine on
EXACTLY the same eval口径 (runs/rec/metrics.json is on the full 14.68M).

Output: analysis/rec_evalsub.jsonl + analysis/rec_evalsub.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from model.glut_repro import data  # noqa: E402
from model.glut_repro.model import BatchedGLUT  # noqa: E402
from model.glut_repro.train_a0 import de00_worker, psnr_8bit, psnr_float  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, ".."))


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sub = data.evalsub_indices()
    colors = data.eval_colors()[sub]
    gt_dir = os.path.join(EXP, "hypA", "gt_tmp")
    pred_dir = os.path.join(EXP, "analysis", "pred_tmp_reeval")
    os.makedirs(pred_dir, exist_ok=True)
    rows = []
    d = os.path.join(EXP, "runs", "rec")
    for fn in sorted(os.listdir(d)):
        if not fn.startswith("ckpt_chunk"):
            continue
        ck = torch.load(os.path.join(d, fn), map_location=dev,
                        weights_only=False)
        ids = ck["lut_ids"]
        m = BatchedGLUT(len(ids), ck["n_gaussians"]).to(dev)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        pred = np.empty((len(ids), colors.shape[0], 3), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, colors.shape[0], 1 << 16):
                j = min(i + (1 << 16), colors.shape[0])
                x = torch.from_numpy(colors[i:j]).to(dev)
                pred[:, i:j] = m.predict(
                    x.unsqueeze(0).expand(len(ids), -1, -1)).cpu().numpy()
        de_jobs, gts = [], []
        for k, lid in enumerate(ids):
            gp = os.path.join(gt_dir, f"{lid}.gt.npy")
            if os.path.exists(gp):
                g = np.load(gp)
            else:
                g = np.asarray(np.load(data._cache_path("a0", lid, "eval"),
                                       mmap_mode="r")[sub])
                tmp = f"{gp}.{os.getpid()}.tmp.npy"
                os.makedirs(gt_dir, exist_ok=True)
                np.save(tmp, g)
                os.replace(tmp, gp)
            gts.append(g)
            pp = os.path.join(pred_dir, f"{lid}.npy")
            np.save(pp, pred[k].astype(np.float16))
            de_jobs.append((pp, gp, slice(None)))
        from multiprocessing import Pool
        with Pool(8) as pool:
            de = pool.map(de00_worker, de_jobs)
        for k, lid in enumerate(ids):
            g = gts[k].astype(np.float32)
            rows.append({"lut_id": lid, "psnr_float": psnr_float(pred[k], g),
                         "psnr_8bit": psnr_8bit(pred[k], g),
                         "de00_mean": de[k]["mean"], "de00_p99": de[k]["p99"]})
            os.remove(os.path.join(pred_dir, f"{lid}.npy"))
        print(f"{fn}: {len(ids)} LUTs done", flush=True)
    with open(os.path.join(HERE, "rec_evalsub.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    p = np.array([r["psnr_float"] for r in rows])
    de_m = np.array([r["de00_mean"] for r in rows])
    full = json.load(open(os.path.join(EXP, "runs", "rec", "metrics.json")))
    out = {"n_luts": len(rows), "eval_part": "evalsub_2^21",
           "psnr_float_mean": float(p.mean()),
           "psnr_float_std": float(p.std()),
           "de00_mean_over_luts": float(de_m.mean()),
           "de00_p50": float(np.percentile(de_m, 50)),
           "full_eval_psnr_float_mean": full["psnr_float_mean"],
           "subsample_bias_db": float(p.mean()) - full["psnr_float_mean"]}
    with open(os.path.join(HERE, "rec_evalsub.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
