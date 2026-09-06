# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/vlmsft/eval_vlmadapt.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 / SFT+ADAPT -- frozen SPRF executor fed with VLM-predicted latents.

This arm swaps EXACTLY ONE thing relative to T-ALIGN: the per-stage 128-d edit
latents.  The base conditioning c (SigLIP feature cache -> model.cond.base) is
untouched, the backbone is frozen, the solver/path/NFE are the backbone's own.
That is what makes the row comparable to the registered T-ALIGN / C-LUT rows.

Contracts (never merged into one row):
  oracle_text     latents read from GT CoT text (teacher forced)
  predicted_text  latents read from the model's own greedy CoT

Columns, per CLAUDE.md ("每消融行必带 Δ_const/Δ_shuffle"):
  model, identity,
  delta_const          c := 0            (cond-side, same definition as T-ALIGN)
  delta_shuffle        c := partner's c  (cond-side, same salt as T-ALIGN)
  delta_edit_null      latent := null row
  delta_edit_roll      latents rolled across active stages
  delta_latent_shuffle latents := partner's latents   (VLM-side, new to this arm)

Held-out discipline (lesson E-heldout): load_shards re-derives and inflates the
held-out set to ~35,571; it is filtered against snapshot_newdata_v3.heldout_ids
and asserted with an equality, never a subset check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from veraretouch_sprf import _paths as _P  # EPR-052：替代原 sys.path 注入块
_P.ensure_sys_path()
SPRF = _P.SPRF_LEGACY

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from veraretouch_sprf.train import train_align_time as TA  # noqa: E402  (main() is __main__-guarded)

T0, ST, SS, EC, SF = TA.T0, TA.ST, TA.SS, TA.EC, TA.SF


def die(m):
    print(f"FATAL: {m}", flush=True)
    raise SystemExit(2)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_backbone(run_args_path: Path, ckpt_path: Path, device: str):
    ra = json.loads(run_args_path.read_text())
    cd = ra["config"]
    mc = TA.MiniCfg(cd) if hasattr(TA, "MiniCfg") else None
    if mc is None:
        die("train_align_time has no MiniCfg; cannot rebuild the backbone")
    in_dim = int(ra["model"]["in_dim"])
    # n_steps lives in the backbone run_args' frozen data law; the arm config has
    # NO step_order key (verified against arm_clut_full/run_args.json).
    law_steps = int(ra["data_law"]["n_steps"])
    if law_steps != 6:
        die(f"data_law n_steps {law_steps} != 6")
    model = SF.SprfModel(in_dim, mc, law_steps, cd["flow"]["alpha_mode"],
                         cd["data"]["depth_values"]).to(device)
    inv = EC.InvLutSource(cd["edit"]["inv_cache_dir"], int(cd["edit"]["grid"]))
    model.load_inv_table(inv.load_table())
    # torch 2.6 defaults weights_only=True; these checkpoints carry numpy arrays.
    cp = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(cp["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if any(p.requires_grad for p in model.parameters()):
        die("B1 FAILED: backbone still has trainable parameters")
    return model, cd, ra, cp, law_steps, inv


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True, help="latents_*.pt from dump_readout")
    ap.add_argument("--backbone-run", required=True,
                    help="e.g. /home/bc/data/runs/epr051_sprf/arm_clut_full")
    ap.add_argument("--ckpt", default="ckpt_best.pt")
    ap.add_argument("--feats-cache", required=True)
    ap.add_argument("--heldout-ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--select-keys", default="",
                    help="evaluate exactly the keys in this json list, drawn from "
                         "ALL samples (train side included). Bypasses the held-out "
                         "4,560 equality assertion, which only applies to held-out "
                         "runs; the train/heldout composition is reported instead.")
    ap.add_argument("--depth-filter", type=int, default=0,
                    help="0 = all held-out; 6 = the d6 stratum only")
    ap.add_argument("--limit", type=int, default=0, help="quick preview")
    ap.add_argument("--backend", default="sprf", choices=["sprf", "bk"],
                    help="bk = BK-FULL decoder (bk_load); the real edit_enc then "
                         "lives at model.bk.core.edit_enc and model.cond.edit_enc "
                         "is only an ALIAS -- patching the alias alone is silent.")
    ap.add_argument("--also-oracle-lut", action="store_true",
                    help="same keys, edits := inv_lut GT rows, run BEFORE the "
                         "Identity patch; the upper-bound row of the table")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    blob = torch.load(args.latents, map_location="cpu")
    contract = blob["contract"]
    if contract not in ("oracle_text", "predicted_text", "predicted_text_s1gen"):
        die(f"unknown contract {contract!r}")
    lat_by_key = {k: blob["latents"][i] for i, k in enumerate(blob["keys"])}
    ok_by_key = {k: bool(blob["ok"][i]) for i, k in enumerate(blob["keys"])}

    run = Path(args.backbone_run)
    if args.backend == "bk":
        from veraretouch_sprf.models import bk_load
        ra = json.loads((run / "run_args.json").read_text())
        cd = ra["config"]
        model = bk_load.load_bk_model(run / "run_args.json", run / args.ckpt,
                                      args.device)
        model.eval()
        for p_ in model.parameters():
            p_.requires_grad_(False)
        cp = torch.load(run / args.ckpt, map_location="cpu", weights_only=False)
        n_steps = int(ra["data_law"]["n_steps"])
        inv = EC.InvLutSource(cd["edit"]["inv_cache_dir"], int(cd["edit"]["grid"]))
    else:
        model, cd, ra, cp, n_steps, inv = load_backbone(
            run / "run_args.json", run / args.ckpt, args.device)
    # The object the forward actually calls.  bk_load exposes model.cond.edit_enc
    # as an alias of model.bk.core.edit_enc; assigning to the alias leaves the
    # forward on the old Sequential (128 -> 19652x512 shape crash).
    real_edit_enc = (model.bk.core.edit_enc if args.backend == "bk"
                     else model.cond.edit_enc)
    ckpt_sha = sha256_file(run / args.ckpt)

    # MUST precede load_shards (train_align_time.py:1435 does the same): the
    # default compact_row drops `luts`/`grids`, and inv.for_row reads row["luts"].
    ST.install_compact_row(T0)
    bcfg = T0.Cfg(Path(ra["config_path"]))
    T0.ASSET_MODE = cd["data"]["asset_source"]
    samples, blobs, _ = T0.load_shards(bcfg)
    # MUST follow load_shards and precede any alpha rebuild: epr050_build_degradation's
    # module globals STEP_KIND / N_STEPS / PIX_CHUNK start EMPTY and are filled here
    # (train_align_time.py does the same right after load_shards).  Without it
    # T0.alpha_fields dies with "config step_order [] disagrees with the recorded chain".
    law = T0.bind_build_config(samples, cd["guard"]["build_config_sha256_allowed"])
    if int(law["n_steps"]) != n_steps:
        die(f"bind_build_config n_steps {law['n_steps']} != run_args data_law {n_steps}")
    print(f"[eval] build config bound: n_steps={law['n_steps']} "
          f"step_kind={law['step_kind']}", flush=True)
    want_ids = set(json.loads(Path(args.heldout_ids).read_text()))
    if args.select_keys:
        sel = set(json.loads(Path(args.select_keys).read_text()))
        def _k(s_):
            return f"{s_['id']}|d{s_['depth'] if s_['depth'] is not None else 'full'}"
        held = sorted([s for s in samples if _k(s) in sel],
                      key=lambda s: (s["id"], s["depth"] or 0))
        if len(held) != len(sel):
            die(f"--select-keys matched {len(held)} of {len(sel)} keys")
        n_ho = sum(1 for s in held if s["heldout"] and s["id"] in want_ids)
        print(f"[eval] select-keys: n={len(held)} heldout_side={n_ho} "
              f"train_side={len(held)-n_ho}", flush=True)
    else:
        held = [s for s in samples if s["heldout"] and s["id"] in want_ids]
        held = sorted(held, key=lambda s: (s["id"], s["depth"] or 0))
        if len(held) != 4560:
            die(f"held-out filter produced {len(held)} != 4560 (E-heldout: "
                "load_shards inflates; this must be an equality, not a subset)")
    if args.depth_filter:
        held = [s for s in held if (s["depth"] or n_steps) == args.depth_filter]
        print(f"[eval] depth filter d{args.depth_filter}: n={len(held)}", flush=True)
    if args.limit:
        held = held[: args.limit]

    # pooled c cache, assembled exactly as train_align_time.py:1532-1547 does it
    # (there is no T0.load_feature_cache helper).  Prefer the shard files frozen
    # into the backbone's run_args over a directory glob.
    want = {T0.feature_key(s) for s in held}
    want_shards = {s["shard"] for s in held}
    frozen_files = [Path(x["file"]) for x in ra.get("encoder", {}).get("files", [])
                    if x.get("shard") in want_shards]
    cache_files = (sorted(frozen_files) if frozen_files else
                   sorted(Path(args.feats_cache).glob("siglip_*.pt")))
    if not cache_files:
        die(f"no siglip_*.pt under {args.feats_cache} and none frozen in run_args")
    feats = {}
    for f in cache_files:
        for k, v in torch.load(f, map_location="cpu")["features"].items():
            if k in want:
                feats[k] = v
    miss = want - set(feats)
    if miss:
        die(f"pooled c cache missing {len(miss)} keys, e.g. {sorted(miss)[:3]}")
    print(f"[eval] pooled c cached: {len(feats)} from {len(cache_files)} files",
          flush=True)

    grid = int(cd["edit"]["grid"])
    path_mode = cd["flow"]["path_mode"]
    nfe = int(cd["solver"]["nfe_per_stage"])
    lo, hi = float(cd["solver"]["clamp_lo"]), float(cd["solver"]["clamp_hi"])
    n_pix = int(cd["eval"]["pixels_per_sample"])
    targets = TA.AP.build_target_latents(real_edit_enc, model.inv_table).detach()
    null_lat = targets[int(model.edit_null_row)].to(args.device)

    keys = [f"{s['id']}|d{s['depth'] if s['depth'] is not None else 'full'}" for s in held]
    missing = [k for k in keys if k not in lat_by_key]
    if missing:
        die(f"{len(missing)} held-out keys have no dumped latent, e.g. {missing[:3]}")
    partners = ST.shuffle_partner_index([s["id"] for s in held],
                                        cd["eval"]["shuffle_salt"], T0.source_id_of)

    def run_roll(base_c, yb, alphas, union, depth, sc, edits, d_int):
        base = model.cond.base(base_c)
        out, _ = SS.rollout_with_metrics(
            model, base, yb, ST.compose_alpha_hat(alphas), alphas, union,
            depth, sc, path_mode, SS.stages_for(path_mode, d_int), nfe,
            clamp_lo=lo, clamp_hi=hi, edits=edits)
        return out

    # ---- A-inj: oracle_lut path must equal the injected-latent path, bitwise ----
    # TRAP (4th): the backbone's edit_condition is "inv_lut", so `edits` must be
    # LONG row numbers -- it does inv_table[edits] then edit_enc.  Feeding a 128-d
    # predicted latent there dies with "inv_lut 的编辑来源必须是 long 行号".
    # T-ALIGN's own injection (train_align_time.py:1526-1530) bypasses edit_enc:
    # edit_descriptor := identity, cond.edit_enc := Identity, contract :=
    # predicted_lut.  Then a 128-d latent flows straight to the FiLM concat.
    def _batch_of(s_):
        row_ = json.loads(blobs[s_["id"]])
        x0_, y_ = T0.load_pair(s_["shard"], row_, s_["after_asset"])
        mask_ = T0.depth_mask(s_["depth"], n_steps)
        af_ = T0.alpha_fields(row_, x0_).reshape(n_steps, -1) * mask_.unsqueeze(-1)
        yf_ = y_.reshape(-1, 3)
        npx_ = int(yf_.shape[0])
        idx_ = (torch.arange(npx_) if n_pix <= 0 else
                torch.arange(0, npx_, max(1, npx_ // n_pix))[:n_pix])
        d_ = n_steps if s_["depth"] is None else int(s_["depth"])
        return dict(row=row_, x0=x0_, mask=mask_,
                    xb=x0_.reshape(-1, 3)[idx_].to(args.device).unsqueeze(0),
                    yb=yf_[idx_].to(args.device).unsqueeze(0),
                    alphas=af_[:, idx_].to(args.device).unsqueeze(0),
                    d_int=d_,
                    depth=torch.tensor([d_], device=args.device),
                    sc=torch.tensor([float(row_["calib"]["s"])], device=args.device),
                    c=T0.gather_feats(feats, [T0.feature_key(s_)], 0, args.device))

    # ---- A-lat: latent ordering must match the dump, BEFORE any rollout -------
    # TRAP (5th): the adapter is trained on SLOT-ordered targets
    # (T[m-1] = target of chain k = 6-m, restoration order), so its output is in
    # slot order.  The solver indexes edits[:, m_idx-1] in CHAIN order (stage 1 =
    # geom = k0).  Handing slot-ordered latents straight to the solver applies
    # every latent to the wrong stage AND makes latent_cos compare slot-vs-chain.
    # Conversion is a reversal: chain[k] = slot[5-k]  ->  lat.flip(0).
    dump_json = Path(args.latents).parent / (
        "dump_" + Path(args.latents).stem.replace("latents_", "") + ".json")
    dump_cos = None
    if dump_json.exists():
        dump_cos = json.loads(dump_json.read_text()).get("latent_cos_mean")
    num = 0.0
    cnt = 0
    n_skip = 0
    # A-lat must be computed over the SAME key set the dump averaged over (all
    # ok keys), not over `held`.  When --select-keys narrows the rollout to a
    # subset (e.g. only those samples whose .src.png assets are still on disk --
    # 249 of 279 shards have been archived), averaging over the subset would
    # make the 1e-6 equality fail on bookkeeping rather than on stage ordering.
    # A-lat needs only blobs[] + inv.for_row(row, None, None); it loads NO image
    # assets, so it can safely span every dumped key.
    _dump_keys = set(blob["keys"])
    alat_samples = sorted([s_ for s_ in samples
                           if f"{s_['id']}|d{s_['depth'] if s_['depth'] is not None else 'full'}"
                           in _dump_keys], key=lambda s_: (s_["id"], s_["depth"] or 0))
    if len(alat_samples) != len(_dump_keys):
        die(f"A-lat scope: matched {len(alat_samples)} of {len(_dump_keys)} dumped keys")
    print(f"[eval] A-lat scope: {len(alat_samples)} dumped keys "
          f"(rollout scope is {len(held)})", flush=True)
    for s_ in alat_samples:
        k_ = f"{s_['id']}|d{s_['depth'] if s_['depth'] is not None else 'full'}"
        # the dump averages over OK samples only; match that set exactly or the
        # 1e-6 assertion fails on bookkeeping rather than on ordering.
        if not ok_by_key.get(k_, False):
            n_skip += 1
            continue
        row_ = json.loads(blobs[s_["id"]])
        rows_ = inv.for_row(row_, None, None).to(args.device)   # chain order
        e_ = lat_by_key[k_].to(args.device).flip(0)             # slot -> chain
        cs = nn.functional.cosine_similarity(e_, targets[rows_])
        num += float(cs.sum()); cnt += int(cs.numel())
    cos_recomputed = num / max(1, cnt)
    print(f"[eval] A-lat recomputed latent_cos_mean={cos_recomputed:.10f} "
          f"dump={dump_cos} (ok-only; skipped {n_skip} not-ok)", flush=True)
    if dump_cos is not None and abs(cos_recomputed - float(dump_cos)) > 1e-6:
        die(f"A-lat FAILED: recomputed latent_cos {cos_recomputed} != dump "
            f"{dump_cos} (|d|={abs(cos_recomputed-float(dump_cos))}); the latent "
            "stage ordering does not match the dump's convention")

    guard = held[:4]
    pre = []
    for s_ in guard:
        b_ = _batch_of(s_)
        rows_ = inv.for_row(b_["row"], b_["x0"], b_["mask"]).to(args.device)
        u_ = ST.union_mask_of(b_["alphas"])
        pre.append((b_, rows_, run_roll(b_["c"], b_["yb"], b_["alphas"], u_,
                                        b_["depth"], b_["sc"], rows_.unsqueeze(0), b_["d_int"])))
    oracle_recs = []
    if args.also_oracle_lut:
        t_o = time.time()
        for i_, s_ in enumerate(held):
            b_ = _batch_of(s_)
            rows_ = inv.for_row(b_["row"], b_["x0"], b_["mask"]).to(args.device)
            u_ = ST.union_mask_of(b_["alphas"])
            o_ = run_roll(b_["c"], b_["yb"], b_["alphas"], u_, b_["depth"], b_["sc"],
                          rows_.unsqueeze(0), b_["d_int"])
            oracle_recs.append(dict(
                key=f"{s_['id']}|d{s_['depth'] if s_['depth'] is not None else 'full'}",
                depth=b_["d_int"], model=T0.err_stats(T0.linf8(o_, b_["xb"]))))
            if (i_ + 1) % 50 == 0 or i_ + 1 == len(held):
                print(f"[eval:oracle_lut] {i_+1}/{len(held)} "
                      f"{time.time()-t_o:.0f}s", flush=True)
        print("ORACLE_LUT_SUMMARY " + json.dumps(dict(n=len(oracle_recs), model={
            f: float(np.median([r["model"][f] for r in oracle_recs]))
            for f in ("p50", "p95", "p99")})), flush=True)

    frozen_edit_enc = real_edit_enc
    model.edit_descriptor = lambda src: src
    _ident = nn.Identity()
    if args.backend == "bk":
        model.bk.core.edit_enc = _ident
        object.__setattr__(model.cond, "edit_enc", _ident)
        if model.edit_encoder is not _ident:
            die(f"edit_encoder still {model.edit_encoder.__class__.__name__}")
    else:
        model.cond.edit_enc = _ident
    model.edit_condition = "predicted_lut"
    model.edit_contract = "predicted_lut"
    for j, (b_, rows_, out_a) in enumerate(pre):
        # Encode ONE STAGE AT A TIME at batch 1, exactly as the model does
        # internally (stage_flow.py:383 passes edits[:, m-1] -> edit_descriptor ->
        # inv_table[(B,)] -> edit_enc((B, descriptor_dim)) with B=1).  Encoding all
        # 6 rows in a single (6, d) call changes the bf16 reduction order and moved
        # the rollout by 7.77e-05 -- same effect as the batched-generation gate.
        lat_ = torch.stack([
            frozen_edit_enc(model.inv_table[rows_[mi]].unsqueeze(0))[0].detach()
            for mi in range(rows_.shape[0])], dim=0)
        u_ = ST.union_mask_of(b_["alphas"])
        out_b = run_roll(b_["c"], b_["yb"], b_["alphas"], u_, b_["depth"], b_["sc"],
                         lat_.unsqueeze(0), b_["d_int"])
        d_max = float((out_a - out_b).abs().max())
        if d_max != 0.0:
            die(f"A-inj FAILED on {guard[j]['id']}: injected-latent path differs "
                f"from the oracle_lut path by {d_max} (must be exactly 0)")
    print(f"[eval] A-inj PASS: injected-latent == oracle_lut path bitwise on "
          f"{len(pre)} samples", flush=True)

    recs = []
    t0 = time.time()
    for i, s in enumerate(held):
        row = json.loads(blobs[s["id"]])
        x0, y = T0.load_pair(s["shard"], row, s["after_asset"])
        mask = T0.depth_mask(s["depth"], n_steps)
        af = T0.alpha_fields(row, x0).reshape(n_steps, -1) * mask.unsqueeze(-1)
        yf = y.reshape(-1, 3)
        npx = int(yf.shape[0])
        idx = (torch.arange(npx) if n_pix <= 0 else
               torch.arange(0, npx, max(1, npx // n_pix))[:n_pix])
        xb = x0.reshape(-1, 3)[idx].to(args.device).unsqueeze(0)
        yb = yf[idx].to(args.device).unsqueeze(0)
        alphas = af[:, idx].to(args.device).unsqueeze(0)
        union = ST.union_mask_of(alphas)
        d_int = n_steps if s["depth"] is None else int(s["depth"])
        depth = torch.tensor([d_int], device=args.device)
        sc = torch.tensor([float(row["calib"]["s"])], device=args.device)
        c = T0.gather_feats(feats, [T0.feature_key(s)], 0, args.device)
        c_const = torch.zeros_like(c)
        c_shuf = T0.gather_feats(feats, [T0.feature_key(held[partners[i]])], 0,
                                 args.device)

        # .flip(0): slot order (adapter) -> chain order (solver).  See A-lat.
        ed = lat_by_key[keys[i]].to(args.device).flip(0).unsqueeze(0) \
            * mask.to(args.device).view(1, -1, 1)
        ed_part = lat_by_key[keys[partners[i]]].to(args.device).flip(0).unsqueeze(0) \
            * mask.to(args.device).view(1, -1, 1)
        ed_roll, _rep = TA.roll_active_edits(ed, depth)

        out = run_roll(c, yb, alphas, union, depth, sc, ed, d_int)
        stats = T0.err_stats(T0.linf8(out, xb))
        ident = T0.err_stats(T0.linf8(yb, xb))
        controls = {}
        for col, e_, c_ in (("delta_const", ed, c_const),
                            ("delta_shuffle", ed, c_shuf),
                            ("delta_edit_null", null_lat.view(1, 1, -1).expand(1, n_steps, -1), c),
                            ("delta_edit_roll", ed_roll, c),
                            ("delta_latent_shuffle", ed_part, c)):
            v = run_roll(c_, yb, alphas, union, depth, sc, e_, d_int)
            cs = T0.err_stats(T0.linf8(v, xb))
            controls[col] = dict(value=cs["p50"] - stats["p50"], control_p50=cs["p50"])

        rows = inv.for_row(row, x0, mask).to(args.device)
        tgt = targets[rows]
        recs.append(dict(id=s["id"], depth=d_int, geom=s["geom"],
                         rec_band=s["rec_band"], key=keys[i],
                         readout_ok=ok_by_key.get(keys[i], False),
                         n_eval_pixels=int(idx.numel()),
                         model=stats, identity=ident,
                         latent_cos=float(nn.functional.cosine_similarity(ed[0], tgt).mean()),
                         latent_l2=float((ed[0] - tgt).norm(dim=-1).mean()),
                         **controls))
        if (i + 1) % 100 == 0 or i + 1 == len(held):
            print(f"[eval:{contract}] {i+1}/{len(held)} {time.time()-t0:.0f}s", flush=True)

    def med(col, f):
        return float(np.median([r[col][f] for r in recs]))

    overall = {"model": {f: med("model", f) for f in ("p50", "p95", "p99")},
               "identity": {f: med("identity", f) for f in ("p50", "p95", "p99")}}
    for col in ("delta_const", "delta_shuffle", "delta_edit_null",
                "delta_edit_roll", "delta_latent_shuffle"):
        overall[col] = dict(value=med(col, "value"), control_p50=med(col, "control_p50"))
    overall["latent_cos"] = float(np.median([r["latent_cos"] for r in recs]))
    overall["latent_l2"] = float(np.median([r["latent_l2"] for r in recs]))

    by_depth = {}
    for d in sorted({r["depth"] for r in recs}):
        cell = [r for r in recs if r["depth"] == d]
        by_depth[str(d)] = dict(
            n=len(cell),
            model={f: float(np.median([r["model"][f] for r in cell]))
                   for f in ("p50", "p95", "p99")},
            identity_p50=float(np.median([r["identity"]["p50"] for r in cell])))

    out = dict(eval_kind="final" if not args.limit else "quick",
               contract=contract, tag=blob.get("tag", ""), n=len(recs),
               depth_filter=args.depth_filter or None,
               readout_ok_n=sum(1 for r in recs if r["readout_ok"]),
               readout_ok_rate=sum(1 for r in recs if r["readout_ok"]) / max(1, len(recs)),
               backbone=dict(run=str(run), ckpt=args.ckpt, ckpt_sha256=ckpt_sha,
                             step=int(cp.get("step", -1))),
               latents_file=args.latents,
               adapt_run=blob.get("adapt_run"), sft_run=blob.get("sft_run"),
               overall=overall, by_depth=by_depth,
               oracle_lut=(dict(n=len(oracle_recs), model={
                   f: float(np.median([r["model"][f] for r in oracle_recs]))
                   for f in ("p50", "p95", "p99")}, per_sample=oracle_recs)
                   if oracle_recs else None),
               backend=args.backend,
               model_execution_path="rollout with VLM-predicted per-stage edit latents; "
                                    "cond c and backbone unchanged",
               per_sample=recs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    head = {k: overall[k] for k in ("model", "identity", "latent_cos", "latent_l2")}
    print("EVAL_SUMMARY " + json.dumps(dict(
        contract=contract, n=len(recs), depth_filter=args.depth_filter or None,
        readout_ok_rate=out["readout_ok_rate"], **head)), flush=True)


if __name__ == "__main__":
    main()
