# S4 (round 3) step 1: box-prompted SAM2 masks for the 34 box test samples.
# Runs under the monetgpt_sam3 conda env (SAM2 repo + sam2_hiera_large.pt already local);
# standalone on purpose - only numpy/cv2/torch/sam2. Writes one PNG mask per sample.
# Note: brief said SAM ViT-B; we use the locally available (and stronger) SAM2 hiera-large.
import os, re, json, sys, argparse
import numpy as np
import cv2
import torch

SAM2_ROOT = "/home/bc/retouching/monetGPT/external/sam2"
CKPT = "/home/bc/retouching/monetGPT/models/sam2_hiera_large.pt"
RESULTS_R2 = os.path.expanduser("~/VeraRetouch/lens_exp_results_r2")
RESULTS_R3 = os.path.expanduser("~/VeraRetouch/lens_exp_results_r3")
BOX_RE = re.compile(r"<box>\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*</box>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS_R3, "s4_sam_masks"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = {r["key"]: r for r in json.load(open(os.path.join(RESULTS_R2, "manifest_r2.json"),
                                                encoding="utf-8"))}
    split = json.load(open(os.path.join(RESULTS_R2, "split_r2.json")))
    c0 = os.path.join(RESULTS_R2, "dumps", "c0")
    keys = sorted(k for k in split["test"] if rows[k]["has_box"]
                  and os.path.exists(os.path.join(c0, k + ".npz")))
    todo = [k for k in keys if not os.path.exists(os.path.join(args.out, k + ".png"))]
    print(f"[s4sam] keys={len(keys)} todo={len(todo)}", flush=True)
    if not todo:
        return

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2("sam2_hiera_l.yaml", CKPT, device="cuda")
    pred = SAM2ImagePredictor(model)

    for k in todo:
        r = rows[k]
        img = cv2.imread(r["input_path"], cv2.IMREAD_COLOR)[..., ::-1]
        h, w = img.shape[:2]
        pred.set_image(np.ascontiguousarray(img))
        union = np.zeros((h, w), dtype=bool)
        scores_all = []
        for m in BOX_RE.findall(r["prompt"]):
            x1, y1, x2, y2 = [float(v) for v in m]
            box = np.array([x1 * w, y1 * h, x2 * w, y2 * h])
            masks, scores, _ = pred.predict(box=box[None], multimask_output=False)
            union |= masks[0].astype(bool)
            scores_all.append(float(scores[0]))
        cv2.imwrite(os.path.join(args.out, k + ".png"), union.astype(np.uint8) * 255)
        print(f"[s4sam] {k} cov={union.mean():.3f} score={np.mean(scores_all):.3f}", flush=True)
        torch.cuda.empty_cache()
    print("[s4sam] done", flush=True)


if __name__ == "__main__":
    main()
