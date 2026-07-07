# S5 (round 3): Logit-Lens-Loss style auxiliary supervision pilot.
# Retrain the FINAL-layer (L24) readout exactly like e6_ctrl, but add an auxiliary linear
# head that predicts the 26-dim photometric delta from the retouch head's shared penultimate
# features; joint loss = render L1 + lam * aux MSE. Backbone frozen (hard constraint), so the
# LLL regularizer can only shape the trainable readout trunk - documented deviation from the
# paper (which regularizes the backbone). Question: can supervision substitute for layer swap?
import os, sys, json, argparse, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common_r2 import RESULTS_R2, C0_DIR, get_split_r2, load_targets_r2
from common_r3 import RESULTS_R3
from e2_lib import load_head_decoder
from e6_train import load_arrays

N_TARGETS = 26


class AuxReadout(nn.Module):
    """original retouch head with an aux linear probe on the penultimate 1344-d features."""

    def __init__(self, base_head, n_targets=N_TARGETS):
        super().__init__()
        mods = list(base_head)
        assert len(mods) == 7
        self.trunk = nn.Sequential(*mods[:6])   # Linear-LN-GELU-Linear-LN-GELU
        self.final = mods[6]                    # Linear 1344 -> 2688
        self.aux = nn.Linear(1344, n_targets)

    def forward(self, x):
        f = self.trunk(x)
        return self.final(f), self.aux(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=20)
    args = ap.parse_args()
    tag = f"s5_lam{args.lam:g}"

    torch.manual_seed(0); np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True

    split = get_split_r2()
    tr_keys = split["train"]
    rng = random.Random(1)                      # same val split as e6_train
    val_keys = sorted(rng.sample(tr_keys, max(20, len(tr_keys) // 5)))
    fit_keys = [k for k in tr_keys if k not in set(val_keys)]
    print(f"[{tag}] fit={len(fit_keys)} val={len(val_keys)}", flush=True)

    base_head, decoder = load_head_decoder(dtype=torch.float32)
    head = AuxReadout(base_head).cuda().float()
    head.train(); decoder.train()
    opt = torch.optim.AdamW(list(head.parameters()) + list(decoder.parameters()),
                            lr=args.lr, weight_decay=1e-4)

    lat, x, gt = load_arrays(fit_keys, "l24", 24)
    vlat, vx, vgt = load_arrays(val_keys, "l24", 24)
    lat, x, gt = lat.cuda(), x.cuda(), gt.cuda()
    vlat, vx, vgt = vlat.cuda(), vx.cuda(), vgt.cuda()

    targets = load_targets_r2()
    T = torch.from_numpy(np.stack([targets[k] for k in fit_keys])).float()
    mu, sd = T.mean(0), T.std(0) + 1e-6
    T = ((T - mu) / sd).cuda()
    VT = ((torch.from_numpy(np.stack([targets[k] for k in val_keys])).float() - mu) / sd).cuda()

    def val_loss():
        head.eval(); decoder.eval()
        tot, aux_tot = 0.0, 0.0
        with torch.no_grad():
            for i in range(0, len(vlat), args.bs):
                z, a = head(vlat[i:i + args.bs])
                out = decoder(vx[i:i + args.bs], z)
                tot += F.l1_loss(out, vgt[i:i + args.bs], reduction="sum").item()
                aux_tot += F.mse_loss(a, VT[i:i + args.bs], reduction="sum").item()
        head.train(); decoder.train()
        return tot / vgt.numel(), aux_tot / VT.numel()

    best, best_state, best_ep, bad = float("inf"), None, -1, 0
    n = len(lat)
    hist = []
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=lat.device)
        tot = 0.0
        for i in range(0, n, args.bs):
            idx = perm[i:i + args.bs]
            z, a = head(lat[idx])
            out = decoder(x[idx], z)
            loss = F.l1_loss(out, gt[idx]) + args.lam * F.mse_loss(a, T[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        vl, va = val_loss()
        hist.append(dict(epoch=ep, train=tot / n, val_l1=vl, val_aux_mse=va))
        if vl < best - 1e-5:                    # early stop on RENDER val L1 (same as e6)
            best, bad, best_ep = vl, 0, ep
            best_state = {"head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                          "decoder": {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}}
        else:
            bad += 1
        if ep % 10 == 0 or bad == 0:
            print(f"[{tag}] ep{ep} train={tot/n:.4f} val_l1={vl:.4f} val_aux={va:.4f} "
                  f"best={best:.4f}@{best_ep}", flush=True)
        if bad >= args.patience:
            print(f"[{tag}] early stop at ep{ep}", flush=True)
            break

    out_path = os.path.join(RESULTS_R3, f"{tag}.pt")
    torch.save(dict(state=best_state, lam=args.lam, val_l1=best, best_epoch=best_ep,
                    target_mu=mu, target_sd=sd, hist=hist), out_path)
    print(f"[{tag}] saved {out_path} best_val_l1={best:.4f} best_ep={best_ep}", flush=True)


if __name__ == "__main__":
    main()
