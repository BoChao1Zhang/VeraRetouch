# S6 (round 3) SAE scaling: (i) L11 with 32x dictionary, (ii) L11+L14+L23 concat with 8x.
# Same TopK-SAE trainer as E9 (e9_sae.TopKSAE), generalized to arbitrary d_in / corpora.
import os, sys, json, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from common_r2 import C0_DIR, RESULTS_R2
from common_r3 import RESULTS_R3
from e9_sae import TopKSAE

SEQ3_DIR = os.path.join(RESULTS_R2, "dumps", "s6_seq3")


def load_corpus(layers, max_tokens=1_500_000, seed=0):
    """single layer -> C0 dumps; multi-layer concat -> self-consistent s6_seq3 dumps."""
    src = C0_DIR if len(layers) == 1 else SEQ3_DIR
    files = sorted(glob.glob(os.path.join(src, "*.npz")))
    if not files:
        raise SystemExit(f"no dumps in {src}")
    chunks = []
    for p in files:
        d = np.load(p)
        chunks.append(np.concatenate([d[f"seq_l{l}"] for l in layers], axis=1)
                      if len(layers) > 1 else d[f"seq_l{layers[0]}"])
        d.close()
    X = np.concatenate(chunks, axis=0)
    rng = np.random.default_rng(seed)
    if len(X) > max_tokens:
        X = X[rng.choice(len(X), max_tokens, replace=False)]
    else:
        rng.shuffle(X)
    print(f"[s6sae] corpus files={len(files)} tokens={len(X)} dim={X.shape[1]}", flush=True)
    return torch.from_numpy(X.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--expansion", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tag", required=True, help="output: s6_sae_<tag>.pt")
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True

    X = load_corpus(tuple(args.layers))
    d_in = X.shape[1]
    n_val = len(X) // 10
    Xv, Xt = X[:n_val].cuda(), X[n_val:].cuda()

    sae = TopKSAE(d_in, args.expansion, args.k).cuda()
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
                xb = Xv[i:i + args.bs]
                rec, z = sae(xb)
                se += (rec - xb).pow(2).sum().item()
                fired |= (z > 0).any(dim=0)
        sae.train()
        return 1 - se / var, 1 - fired.float().mean().item()

    hist = []
    for ep in range(args.epochs):
        perm = torch.randperm(len(Xt), device="cuda")
        tot = 0.0
        for i in range(0, len(Xt), args.bs):
            xb = Xt[perm[i:i + args.bs]]
            rec, z = sae(xb)
            loss = (rec - xb).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            sae.normalize_dec()
            tot += loss.item() * len(xb)
        r2, dead = evaluate()
        hist.append(dict(epoch=ep, train_mse=tot / len(Xt), val_r2=r2, dead_frac=dead))
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"[s6sae:{args.tag}] ep{ep} mse={tot/len(Xt):.4f} val_R2={r2:.4f} dead={dead:.3f}",
                  flush=True)

    out = os.path.join(RESULTS_R3, f"s6_sae_{args.tag}.pt")
    torch.save(dict(state=sae.state_dict(), layers=list(args.layers), d_in=d_in,
                    expansion=args.expansion, k=args.k, hist=hist), out)
    json.dump(dict(tag=args.tag, layers=list(args.layers), d_in=d_in,
                   expansion=args.expansion, k=args.k, n_train=len(Xt), n_val=len(Xv),
                   val_r2=hist[-1]["val_r2"], dead_frac=hist[-1]["dead_frac"]),
              open(os.path.join(RESULTS_R3, f"s6_sae_{args.tag}_summary.json"), "w"), indent=1)
    print(f"[s6sae:{args.tag}] saved {out} R2={hist[-1]['val_r2']:.4f}", flush=True)


if __name__ == "__main__":
    main()
