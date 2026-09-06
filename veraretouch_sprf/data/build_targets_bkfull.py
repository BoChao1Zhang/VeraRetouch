# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/vlmsft/build_targets_bkfull.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 / S2F-B targets from the BK-FULL decoder's edit_enc.

Two products:
  (1) the lut-indexed table  edit_enc(inv_table) -> (4052, 128)
  (2) per-key SLOT-ordered targets for the S2 train/val splits and heldout-d6,
      with the chain<->slot mapping asserted per key rather than assumed.

Slot/chain contract (the trap that produced linf8 16.23 once):
  readout slot m (1..6, RESTORATION order) <-> chain index k = 6 - m (0..5,
  degradation order, the order row["luts"] uses).  Materialised targets are in
  SLOT order, matching the adapter's output; the executor converts with .flip(0).
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
from veraretouch_sprf import _paths as _P  # EPR-052：替代原 sys.path 注入块
_P.ensure_sys_path()
SPRF = _P.SPRF_LEGACY
import torch
from veraretouch_sprf.models import bk_load
from veraretouch_sprf.models import align_predictor_time as AP


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bk-run", default="/home/bc/data/runs/epr051_sprf/bkfull_adagn_ff_affhead")
    ap.add_argument("--ckpt", default="ckpt_last.pt")
    ap.add_argument("--out", default="/home/bc/data/runs/epr051_vlmsft/targets_bkfull")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    run = Path(args.bk_run); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    model = bk_load.load_bk_model(run / "run_args.json", run / args.ckpt, args.device)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    print(f"[bkfull] loaded; edit_condition={model.edit_condition} "
          f"inv_table={tuple(model.inv_table.shape)} null_row={int(model.edit_null_row)}",
          flush=True)

    with torch.no_grad():
        table = AP.build_target_latents(model.cond.edit_enc, model.inv_table).detach().cpu()
    if table.shape[1] != 128:
        raise SystemExit(f"latent dim {table.shape[1]} != 128")
    if not torch.isfinite(table).all():
        raise SystemExit("non-finite target latents")
    t_sha = tensor_sha(table)
    ck_sha = sha_file(run / args.ckpt)
    torch.save(dict(target_latents=table, indexed_by="inv_table row / lut_id",
                    null_row=int(model.edit_null_row),
                    backbone_ckpt=str(run / args.ckpt), backbone_ckpt_sha256=ck_sha,
                    tensor_sha256=t_sha, backbone_arm="BK-FULL adagn_ff_affhead"),
               out / "target_latents_bkfull.pt")
    print(f"[bkfull] table {tuple(table.shape)} sha {t_sha} ckpt_sha {ck_sha[:16]}…", flush=True)

    names = list(json.loads(Path("/home/bc/data/runs/epr051_sprf/lut_inv_cache/g17/index.json")
                            .read_text())["names"])
    row_of = {n: i for i, n in enumerate(names)}
    if table.shape[0] != len(names) + 1:
        raise SystemExit(f"rows {table.shape[0]} != len(names)+1 {len(names)+1}")

    R = Path("/home/bc/data/runs/epr051_vlmsft")
    jobs = [("s2_train", R/"snap_sft2/split_train_keys.json", R/"snap_sft2/assets_index.json"),
            ("s2_val",   R/"snap_sft2/split_val_keys.json",   R/"snap_sft2/assets_index.json"),
            ("heldout_d6", R/"heldout_d6_keys.json", R/"assets_y_heldout/assets_index.json")]
    report = {}
    for tag, kf, af in jobs:
        keys = json.loads(kf.read_text())
        idx = json.loads(af.read_text())["index"]
        keys = [k for k in keys if k in idx]
        T = torch.zeros(len(keys), 6, 128)
        for i, k in enumerate(keys):
            ch = idx[k]["chain"]
            if [c["k"] for c in ch] != list(range(6)):
                raise SystemExit(f"{k}: chain k order {[c['k'] for c in ch]} != 0..5")
            for m in range(1, 7):
                kk = 6 - m                                   # slot m -> chain k
                if idx[k]["cot_step_to_chain_k"][str(m)] != kk:
                    raise SystemExit(f"{k}: recorded slot->chain {idx[k]['cot_step_to_chain_k']} "
                                     f"disagrees with 6-m at m={m}")
                r = row_of.get(ch[kk]["lut"])
                if r is None:
                    raise SystemExit(f"{k}: lut {ch[kk]['lut']} not in inverse table")
                T[i, m-1] = table[r]
        s = tensor_sha(T)
        torch.save(dict(keys=keys, targets=T, order="slot (m=1..6, chain k = 6-m)",
                        tensor_sha256=s, table_sha256=t_sha,
                        backbone_ckpt_sha256=ck_sha), out / f"targets_{tag}.pt")
        report[tag] = dict(n=len(keys), shape=list(T.shape), tensor_sha256=s)
        print(f"[bkfull] {tag}: n={len(keys)} sha={s}", flush=True)
    (out / "targets_report.json").write_text(json.dumps(
        dict(table_sha256=t_sha, backbone_ckpt=str(run/args.ckpt),
             backbone_ckpt_sha256=ck_sha, per_split=report,
             mapping="slot m (1..6, restoration) <-> chain k = 6-m, asserted per key"),
        indent=1))
    print("TARGETS_BKFULL_DONE " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
