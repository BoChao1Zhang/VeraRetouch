# E9 (round 2) SAE pilot: TopK sparse autoencoder on L11 full-sequence activations.
# ~100-line vanilla PyTorch; expansion 8x (896 -> 7168), k=32; reports val R^2 + dead rate.
import os, sys, json, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn

from common_r2 import RESULTS_R2, C0_DIR

HID = 896


class TopKSAE(nn.Module):
    def __init__(self, d_in=HID, expansion=8, k=32):
        super().__init__()
        self.k = k
        d_hid = d_in * expansion
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.enc = nn.Linear(d_in, d_hid)
        self.dec = nn.Parameter(torch.randn(d_hid, d_in) * 0.02)
        with torch.no_grad():
            self.normalize_dec()
            self.enc.weight.copy_(self.dec.data)  # tied init (encoder rows = decoder feature directions)

    @torch.no_grad()
    def normalize_dec(self):
        self.dec.data /= self.dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)

    def encode(self, x):
        pre = self.enc(x - self.b_dec)
        topv, topi = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, topi, torch.relu(topv))
        return z

    def decode(self, z):
        return z @ self.dec + self.b_dec

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def load_corpus(layer=11, max_tokens=1_500_000, seed=0):
    files = sorted(glob.glob(os.path.join(C0_DIR, "*.npz")))
    chunks = []
    for p in files:
        d = np.load(p)
        chunks.append(d[f"seq_l{layer}"])
        d.close()
    X = np.concatenate(chunks, axis=0)
    rng = np.random.default_rng(seed)
    if len(X) > max_tokens:
        X = X[rng.choice(len(X), max_tokens, replace=False)]
    else:
        rng.shuffle(X)
    return torch.from_numpy(X.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--expansion", type=int, default=8)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True

    X = load_corpus(args.layer)
    n_val = len(X) // 10
    Xv, Xt = X[:n_val].cuda(), X[n_val:].cuda()
    print(f"[e9] corpus tokens: train={len(Xt)} val={len(Xv)} dim={X.shape[1]}", flush=True)

    sae = TopKSAE(HID, args.expansion, args.k).cuda()
    with torch.no_grad():
        sae.b_dec.copy_(Xt.mean(0))
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)
    var = (Xv - Xv.mean(0)).pow(2).sum().item()

    def evaluate():
        sae.eval()
        se = 0.0
        fired = torch.zeros(sae.dec.shape[0], dtype=torch.bool, device="cuda")
        with torch.no_grad():
            for i in range(0, len(Xv), args.bs):
                xb = Xv[i:i+args.bs]
                rec, z = sae(xb)
                se += (rec - xb).pow(2).sum().item()
                fired |= (z > 0).any(dim=0)
        sae.train()
        return 1 - se / var, 1 - fired.float().mean().item()

    hist, best_r2 = [], -1
    for ep in range(args.epochs):
        perm = torch.randperm(len(Xt), device="cuda")
        tot = 0.0
        for i in range(0, len(Xt), args.bs):
            xb = Xt[perm[i:i+args.bs]]
            rec, z = sae(xb)
            loss = (rec - xb).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            sae.normalize_dec()
            tot += loss.item() * len(xb)
        r2, dead = evaluate()
        hist.append(dict(epoch=ep, train_mse=tot / len(Xt), val_r2=r2, dead_frac=dead))
        best_r2 = max(best_r2, r2)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"[e9] ep{ep} mse={tot/len(Xt):.4f} val_R2={r2:.4f} dead={dead:.3f}", flush=True)

    out = os.path.join(RESULTS_R2, f"e9_sae_l{args.layer}.pt")
    torch.save(dict(state=sae.state_dict(), layer=args.layer, expansion=args.expansion,
                    k=args.k, hist=hist), out)
    json.dump(dict(layer=args.layer, n_train=len(Xt), n_val=len(Xv),
                   val_r2=hist[-1]["val_r2"], dead_frac=hist[-1]["dead_frac"],
                   expansion=args.expansion, k=args.k),
              open(os.path.join(RESULTS_R2, "e9_sae_summary.json"), "w"), indent=1)
    print(f"[e9] saved {out} final R2={hist[-1]['val_r2']:.4f} dead={hist[-1]['dead_frac']:.3f}", flush=True)


if __name__ == "__main__":
    main()
