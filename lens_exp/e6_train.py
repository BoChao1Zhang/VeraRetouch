# E6 (round 2) training: same protocol as round-1 e2_train (L1@384, AdamW 1e-4, early stop),
# on the C0 dumps with the new 8:2 split. Variants: e6a / e6b / e6c / e6_ctrl.
import os, sys, json, argparse, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from common_r2 import RESULTS_R2, C0_DIR, load_manifest_r2, get_split_r2
from e6_lib import build_variant_model, load_latent

TRAIN_RES = 384


def load_arrays(keys, lat_kind, layers, dump_dir=C0_DIR):
    rows = {r["key"]: r for r in load_manifest_r2()}
    lats, xs, gts = [], [], []
    for k in keys:
        d = np.load(os.path.join(dump_dir, k + ".npz"))
        lats.append(load_latent(d, lat_kind, layers)[0])
        a = cv2.imread(rows[k]["input_path"], cv2.IMREAD_COLOR)
        g = cv2.imread(rows[k]["gt_path"], cv2.IMREAD_COLOR)
        if g.shape[:2] != a.shape[:2]:
            g = cv2.resize(g, (a.shape[1], a.shape[0]))
        a = cv2.resize(a, (TRAIN_RES, TRAIN_RES), interpolation=cv2.INTER_AREA)
        g = cv2.resize(g, (TRAIN_RES, TRAIN_RES), interpolation=cv2.INTER_AREA)
        xs.append(torch.from_numpy(cv2.cvtColor(a, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0)
        gts.append(torch.from_numpy(cv2.cvtColor(g, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0)
    lat = torch.stack(lats)
    x = (torch.stack(xs) - 0.5) * 2
    gt = torch.stack(gts)
    return lat, x, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["e6a", "e6b", "e6c", "e6_ctrl"])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=20)
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True  # round-1 cuBLAS fp32 bug workaround

    split = get_split_r2()
    tr_keys = split["train"]
    rng = random.Random(1)
    val_keys = sorted(rng.sample(tr_keys, max(20, len(tr_keys) // 5)))
    fit_keys = [k for k in tr_keys if k not in set(val_keys)]
    print(f"[{args.variant}] fit={len(fit_keys)} val={len(val_keys)}", flush=True)

    head, decoder, lat_kind, layers = build_variant_model(args.variant, dtype=torch.float32)
    head.train(); decoder.train()
    opt = torch.optim.AdamW(list(head.parameters()) + list(decoder.parameters()),
                            lr=args.lr, weight_decay=1e-4)

    lat, x, gt = load_arrays(fit_keys, lat_kind, layers)
    vlat, vx, vgt = load_arrays(val_keys, lat_kind, layers)
    lat, x, gt = lat.cuda(), x.cuda(), gt.cuda()
    vlat, vx, vgt = vlat.cuda(), vx.cuda(), vgt.cuda()

    def val_loss():
        head.eval(); decoder.eval()
        tot = 0.0
        with torch.no_grad():
            for i in range(0, len(vlat), args.bs):
                out = decoder(vx[i:i+args.bs], head(vlat[i:i+args.bs]))
                tot += F.l1_loss(out, vgt[i:i+args.bs], reduction="sum").item()
        head.train(); decoder.train()
        return tot / vgt.numel()

    best, best_state, best_ep, bad = float("inf"), None, -1, 0
    n = len(lat)
    hist = []
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=lat.device)
        tot = 0.0
        for i in range(0, n, args.bs):
            idx = perm[i:i+args.bs]
            out = decoder(x[idx], head(lat[idx]))
            loss = F.l1_loss(out, gt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        vl = val_loss()
        hist.append(dict(epoch=ep, train_l1=tot / n, val_l1=vl))
        if vl < best - 1e-5:
            best, bad, best_ep = vl, 0, ep
            best_state = {"head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                          "decoder": {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}}
        else:
            bad += 1
        if ep % 10 == 0 or bad == 0:
            print(f"[{args.variant}] ep{ep} train={tot/n:.4f} val={vl:.4f} best={best:.4f}@{best_ep}", flush=True)
        if bad >= args.patience:
            print(f"[{args.variant}] early stop at ep{ep}", flush=True)
            break

    out_path = os.path.join(RESULTS_R2, f"{args.variant}.pt")
    torch.save(dict(state=best_state, variant=args.variant, lat_kind=lat_kind,
                    layers=layers, val_l1=best, best_epoch=best_ep, hist=hist), out_path)
    print(f"[{args.variant}] saved {out_path} best_val_l1={best:.4f} best_ep={best_ep}", flush=True)


if __name__ == "__main__":
    main()
