# E2b: retrain retouch_head+decoder (VLM frozen) on latents from a chosen layer.
# Fully offline: uses dumped all-layer latents + (input, gt) pixel supervision.
import os, sys, json, argparse, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from common import RESULTS, DUMPS, load_manifest
from e2_lib import load_head_decoder, latents_at_layer

DUMP_DIR = os.path.join(DUMPS, "baseline")
TRAIN_RES = 384


def get_split(seed=0, test_frac=1/3):
    path = os.path.join(RESULTS, "split.json")
    if os.path.exists(path):
        return json.load(open(path))
    rows = [r for r in load_manifest()
            if os.path.exists(os.path.join(DUMP_DIR, r["key"] + ".npz"))]
    rng = random.Random(seed)
    test = []
    for lang in ("CN", "EN"):
        ks = sorted(r["key"] for r in rows if r["lang"] == lang)
        rng.shuffle(ks)
        test += ks[: int(len(ks) * test_frac)]
    split = dict(test=sorted(test),
                 train=sorted(r["key"] for r in rows if r["key"] not in set(test)))
    json.dump(split, open(path, "w"), indent=1)
    return split


def load_train_arrays(keys, layer):
    rows = {r["key"]: r for r in load_manifest()}
    lats, xs, gts = [], [], []
    for k in keys:
        d = np.load(os.path.join(DUMP_DIR, k + ".npz"))
        lats.append(latents_at_layer(d, layer)[0])
        a = cv2.imread(rows[k]["input_path"], cv2.IMREAD_COLOR)
        g = cv2.imread(rows[k]["gt_path"], cv2.IMREAD_COLOR)
        if g.shape[:2] != a.shape[:2]:
            g = cv2.resize(g, (a.shape[1], a.shape[0]))
        a = cv2.resize(a, (TRAIN_RES, TRAIN_RES), interpolation=cv2.INTER_AREA)
        g = cv2.resize(g, (TRAIN_RES, TRAIN_RES), interpolation=cv2.INTER_AREA)
        xs.append(torch.from_numpy(cv2.cvtColor(a, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0)
        gts.append(torch.from_numpy(cv2.cvtColor(g, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0)
    lat = torch.stack(lats)                      # [N, 2688]
    x = (torch.stack(xs) - 0.5) * 2              # [-1, 1] input convention
    gt = torch.stack(gts)                        # [0, 1] target
    return lat, x, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=str, required=True,
                    help="tuple index, either one int or 'l1,l2,l3' per token")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=20)
    args = ap.parse_args()
    layer = tuple(int(v) for v in args.layer.split(","))
    layer = layer[0] if len(layer) == 1 else layer

    torch.manual_seed(0); np.random.seed(0)
    # fp32 cublasSgemm fails (CUBLAS_STATUS_NOT_INITIALIZED) for the decoder's
    # Linear(512->3) backward at >200k rows in this torch2.10/cu128 env; the
    # TF32/cublasLt path works, and TF32 precision is ample for this training.
    torch.backends.cuda.matmul.allow_tf32 = True
    split = get_split()
    tr_keys = split["train"]
    rng = random.Random(1)
    val_keys = sorted(rng.sample(tr_keys, max(20, len(tr_keys) // 5)))
    fit_keys = [k for k in tr_keys if k not in set(val_keys)]
    print(f"[e2b:{args.tag}] layer={layer} fit={len(fit_keys)} val={len(val_keys)}", flush=True)

    head, decoder = load_head_decoder(dtype=torch.float32)
    head.train(); decoder.train()
    params = list(head.parameters()) + list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    lat, x, gt = load_train_arrays(fit_keys, layer)
    vlat, vx, vgt = load_train_arrays(val_keys, layer)
    lat, x, gt = lat.cuda(), x.cuda(), gt.cuda()
    vlat, vx, vgt = vlat.cuda(), vx.cuda(), vgt.cuda()

    def val_loss():
        head.eval(); decoder.eval()
        tot = 0.0
        with torch.no_grad():
            for i in range(0, len(vlat), args.bs):
                z = head(vlat[i:i+args.bs])
                out = decoder(vx[i:i+args.bs], z)
                tot += F.l1_loss(out, vgt[i:i+args.bs], reduction="sum").item()
        head.train(); decoder.train()
        return tot / vgt.numel()

    best, best_state, bad = float("inf"), None, 0
    n = len(lat)
    hist = []
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=lat.device)
        tot = 0.0
        for i in range(0, n, args.bs):
            idx = perm[i:i+args.bs]
            z = head(lat[idx])
            out = decoder(x[idx], z)
            loss = F.l1_loss(out, gt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        vl = val_loss()
        hist.append(dict(epoch=ep, train_l1=tot / n, val_l1=vl))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {"head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                          "decoder": {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}}
        else:
            bad += 1
        if ep % 10 == 0 or bad == 0:
            print(f"[e2b:{args.tag}] ep{ep} train={tot/n:.4f} val={vl:.4f} best={best:.4f}", flush=True)
        if bad >= args.patience:
            print(f"[e2b:{args.tag}] early stop at ep{ep}", flush=True)
            break

    out_path = os.path.join(RESULTS, f"e2b_{args.tag}.pt")
    torch.save(dict(state=best_state, layer=layer, val_l1=best, hist=hist), out_path)
    print(f"[e2b:{args.tag}] saved {out_path} best_val_l1={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
