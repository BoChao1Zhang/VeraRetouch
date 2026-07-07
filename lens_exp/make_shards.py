# Split the remaining C0 keys into weighted shards: local H100 + N remote 4090 workers.
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_r2 import RESULTS_R2, C0_DIR, load_manifest_r2

ap = argparse.ArgumentParser()
ap.add_argument("--n-remote", type=int, default=4)
ap.add_argument("--local-weight", type=float, default=1.8)
ap.add_argument("--out", default=os.path.join(RESULTS_R2, "shards"))
args = ap.parse_args()

rows = load_manifest_r2()
pred_dir = os.path.join(RESULTS_R2, "preds_c0")
remaining = [r["key"] for r in rows
             if not (os.path.exists(os.path.join(C0_DIR, r["key"] + ".npz"))
                     and os.path.exists(os.path.join(pred_dir, r["key"] + ".png")))]
n = len(remaining)
wsum = args.local_weight + args.n_remote
n_local = round(n * args.local_weight / wsum)
os.makedirs(args.out, exist_ok=True)
open(os.path.join(args.out, "shard_local.txt"), "w").write("\n".join(remaining[:n_local]) + "\n")
per = (n - n_local) / args.n_remote
cur = n_local
for i in range(args.n_remote):
    end = n if i == args.n_remote - 1 else n_local + round(per * (i + 1))
    open(os.path.join(args.out, f"shard_r{i+1}.txt"), "w").write("\n".join(remaining[cur:end]) + "\n")
    cur = end
print(f"remaining={n} local={n_local} remote~{per:.0f} each -> {args.out}")
