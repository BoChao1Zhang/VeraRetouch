#!/usr/bin/env python3
# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/batch_eval_bk.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 · BK-ABL-v2 · batch-eval（分支文件；coordinator 授权 2026-09-03 21:2x）。

与 `quick_eval` 的差别只有一处：rollout 与四个对照列按 **B 个样本一批**前向（同一模型、同一
`stage_solver_bk.rollout_with_metrics`、同一 `T0.linf8` / `T0.err_stats`），逐样本统计仍逐样本算。
数据装载 / 子集猴补丁 / 像素等距子采样 / β 场 / 逆表 / 错位伙伴表（在**全部** held-out 上算）
与 `quick_eval` 逐字同源。只出 linf8 系列注册列：
  model / identity / delta_const / delta_shuffle / delta_edit_null / delta_edit_roll / stage_mask_bitexact
（frac_le5 派生）；其余列 `columns_skipped` 显式列出（exact_solve / de00* / 条件数 / stage_* / cycle / ...）。
批内深度混合：n_stages = batch 内最大深度；深度更浅的样本在多出的阶段 β≡0，`_update` 的
where(β==0, z, ·) 逐位保持 z（stage_mask_bitexact 逐样本核验）。
校验（有 metrics.json 时）：逐样本 model/identity p50/p95/p99 差 ≤ tol（默认 1e-3）、headline 中位数
四位小数相同、stage_mask_bitexact 全 1，结果写 `metrics_batch.json.validation`。

用法
  PYTHONPATH=/home/bc/VeraRetouch python batch_eval_bk.py --config configs/bk_ff.toml --batch 8 [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
from veraretouch_sprf import _paths as _P  # EPR-052：替代原 sys.path 注入块
_P.ensure_sys_path()
STAGE0 = _P.STAGE0
REPO = _P.REPO

from veraretouch_sprf.data import train_stage0 as T0        # noqa: E402
from veraretouch_sprf.data import archive_assets as AA      # noqa: E402
from veraretouch_sprf.train import train_sprf_xl as XL       # noqa: E402

FAST_COLUMNS = ["model", "identity", "delta_const", "delta_shuffle",
                "delta_edit_null", "delta_edit_roll", "stage_mask_bitexact"]
ERR_COLS = ("model", "identity")


def die(msg):
    raise SystemExit(f"batch_eval_bk: {msg}")


def pick_core(cfg_d: dict):
    bk = cfg_d.get("bk") or {}
    arm = bk.get("arm")
    if arm in ("retinexhead", "gatefuse", "adasingle"):
        import train_sprf_bk_core6 as core
    elif arm in ("adagn_ff", "adagn_ff_affhead"):
        from veraretouch_sprf.train import train_sprf_bk_core4 as core
    elif arm == "adagn" and not (cfg_d.get("subset") or {}).get("train_subset", True):
        import train_sprf_bk_core7 as core        # BK-FULL2（全量配方；core7 = core4 + bk3）
    elif arm in ("canonfilm", "adagn"):
        import train_sprf_bk_core3 as core
    elif "hidden" in bk:
        import train_sprf_bk_core2 as core
    else:
        import train_sprf_bk_core as core
    return core


def load_item(T0, ST, s, blobs, n_steps, n_pix, edit_src):
    """与 quick_eval 逐字相同的单样本装载（像素等距子采样、β 场、s、编辑来源）。"""
    row = json.loads(blobs[s["id"]])
    x0, y = T0.load_pair(s["shard"], row, s["after_asset"])
    a = T0.alpha_fields(row, x0)
    npx = x0.shape[0] * x0.shape[1]
    idx = (torch.arange(npx) if n_pix <= 0 else
           torch.arange(0, npx, max(1, npx // n_pix))[:n_pix])
    mask = T0.depth_mask(s["depth"], n_steps)
    alphas = a.reshape(n_steps, -1)[:, idx] * mask.unsqueeze(-1)
    d_int = n_steps if s["depth"] is None else int(s["depth"])
    return dict(s=s, x=x0.reshape(-1, 3)[idx], y=y.reshape(-1, 3)[idx], alphas=alphas,
                depth=d_int, sc=float(row["calib"]["s"]), key=T0.feature_key(s),
                edits=edit_src.for_row(row, x0, mask), P=int(idx.numel()))


@torch.no_grad()
def eval_batch(items, model, feats, tokens, device, SS, ST, T0, path_mode, partner_keys):
    B = len(items)
    xb = torch.stack([it["x"] for it in items]).to(device)
    yb = torch.stack([it["y"] for it in items]).to(device)
    alphas = torch.stack([it["alphas"] for it in items]).to(device)
    depth = torch.tensor([it["depth"] for it in items], device=device)
    sc = torch.tensor([it["sc"] for it in items], device=device)
    edits = torch.stack([it["edits"] for it in items]).to(device)
    union = ST.union_mask_of(alphas)
    ahat = ST.compose_alpha_hat(alphas)
    feat = T0.gather_feats(feats, [it["key"] for it in items], tokens, device)
    n_stages = SS.stages_for(path_mode, int(depth.max()))

    def run(feat_, edits_):
        base = model.cond.base(feat_, model.edit_descriptor(edits_))
        return SS.rollout_with_metrics(model, base, yb, ahat, alphas, union, depth, sc,
                                       path_mode, n_stages, 1, return_trace=True, edits=edits_)
    pred, mtr = run(feat, edits)
    if mtr["nfe"] != n_stages:
        die(f"A5 NFE 计数失败: {mtr['nfe']} vs {n_stages}")
    e_model = T0.linf8(pred, xb)                                 # (B,P)
    e_id = T0.linf8(yb, xb)
    # 逐阶段 β_m==0 像素逐位不动（逐样本）
    tot = torch.zeros(B, dtype=torch.long); ok = torch.zeros(B, dtype=torch.long)
    by_stage = [dict() for _ in range(B)]
    prev = yb
    for j, z in enumerate(mtr["trace"], start=1):
        m = n_stages + 1 - j
        mm = torch.full((B,), m, dtype=torch.long, device=device)
        b0 = SS.stage_beta(path_mode, alphas, union, depth, mm) == 0     # (B,P)
        for i in range(B):
            n0 = int(b0[i].sum())
            same = int((z[i][b0[i]] == prev[i][b0[i]]).all(dim=-1).sum()) if n0 else 0
            tot[i] += n0; ok[i] += same
            by_stage[i][m] = dict(n_beta_zero=n0, n_bit_exact=same, all_bit_exact=(n0 == same))
        prev = z
    ctrl = {}
    ctrl["delta_const"] = T0.linf8(run(torch.zeros_like(feat), edits)[0], xb)
    ctrl["delta_shuffle"] = T0.linf8(run(T0.gather_feats(feats, partner_keys, tokens, device), edits)[0], xb)
    ctrl["delta_edit_null"] = T0.linf8(run(feat, model.null_edit(edits))[0], xb)
    ctrl["delta_edit_roll"] = T0.linf8(run(feat, model.roll_edit(edits))[0], xb)
    recs = []
    for i, it in enumerate(items):
        s = it["s"]
        cols = dict(model=dict(T0.err_stats(e_model[i:i + 1]), source="uint8_asset"),
                    identity=dict(T0.err_stats(e_id[i:i + 1]), source="uint8_asset"))
        p50 = cols["model"]["p50"]
        for c in ("delta_const", "delta_shuffle"):
            v = T0.err_stats(ctrl[c][i:i + 1])
            cols[c] = dict(value=v["p50"] - p50, source="cond_control", control_p50=v["p50"])
        for c in ("delta_edit_null", "delta_edit_roll"):
            v = T0.err_stats(ctrl[c][i:i + 1])
            cols[c] = dict(value=v["p50"] - p50, source="edit_control", control_p50=v["p50"])
        t_i = int(tot[i]); o_i = int(ok[i])
        cols["stage_mask_bitexact"] = dict(value=(o_i / t_i if t_i else None), source="per_stage_beta0",
                                           by_stage=by_stage[i],
                                           criterion="每阶段 β_m==0 的像素经该步更新后逐位不动")
        recs.append(dict(id=s["id"], depth=it["depth"], geom=s["geom"], rec_band=s["rec_band"],
                         major=s["major"], n_eval_pixels=it["P"], batch_size=B,
                         batch_n_stages=n_stages, union_frac=float(union[i].mean()),
                         alpha_hat_mean=float(ahat[i].mean()), cols=cols))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--out", default="metrics_batch.json")
    ap.add_argument("--ckpt", default="ckpt_last.pt")
    a = ap.parse_args()
    cfg_path = Path(a.config)
    cfg_d = tomllib.loads(cfg_path.read_text())
    sub = cfg_d.get("subset")
    if not sub or not sub.get("enabled", False):
        die("本入口专供带 [subset] 的 v3 臂")
    core = pick_core(cfg_d)
    SF, ST, SS, EC = core.SF, core.ST, core.SS, core.EC
    AA.install(T0)
    if not getattr(T0, "_SPRF_ARCHIVE_INSTALLED", False):
        die("X5 FAILED: archive_assets 补丁没装上")
    XL.install_subset(T0, ("" if not sub.get("train_subset", True) else sub["keys_file"]),
                      sub["sha256"], sub["n"], sub["heldout_ids_file"],
                      sub["heldout_ids_sha256"], int(sub.get("expect_heldout_samples", 0)))
    cfg = T0.Cfg(cfg_path)
    out_dir = Path(cfg.str_("run", "out_dir"))
    device = cfg.str_("run", "device")
    seed = cfg.int_("run", "seed")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    ST.install_compact_row(T0)
    T0.ASSET_MODE = cfg.str_("data", "asset_source", ("dir", "tar", "auto"))
    samples, blobs, index_meta = T0.load_shards(cfg)
    law = T0.bind_build_config(samples, cfg.list_("guard", "build_config_sha256_allowed"))
    n_steps = law["n_steps"]
    held = [s for s in samples if s["heldout"]]
    held.sort(key=lambda s: (s["id"], s["depth"] or 0))
    inputs = cfg.list_("encoder", "inputs")
    feats, cache_meta = T0.build_feature_cache(cfg, samples, device, inputs)
    tokens = int(cache_meta["tokens"])
    in_dim = cache_meta["dim"] * (1 if tokens else len(inputs))
    path_mode = cfg.str_("flow", "path_mode", ST.PATH_MODES)
    alpha_mode = cfg.str_("flow", "alpha_mode", ST.ALPHA_MODES)
    depth_values = cfg.list_("data", "depth_values", int)
    model = SF.BkSprfModel(in_dim, cfg, n_steps, alpha_mode, depth_values).to(device)
    ckpt_path = out_dir / a.ckpt
    if not ckpt_path.is_file():
        die(f"缺 {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    ckpt_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    bank_dir = cfg.str_("data", "lut_bank_dir")
    espec = SF.edit_spec(cfg)
    if espec["condition"] != "inv_lut":
        die("batch-eval 只支持 inv_lut 臂")
    inv = EC.InvLutSource(cfg.str_("edit", "inv_cache_dir"), espec["grid"])
    inv.assert_fingerprint(T0.sha256_file(Path(core.B.__file__)),
                           T0.sha256_file(Path(law["config_path"])), bank_dir)
    model.load_inv_table(inv.load_table())
    n_pix = cfg.int_("eval", "pixels_per_sample")
    final_n = cfg.int_("eval", "final_max_samples")
    final_set = T0.eval_subset(held, final_n, cfg.str_("eval", "subset_salt"))   # 全部 held-out（同 quick_eval）
    keys = [T0.feature_key(s) for s in final_set]
    partner = ST.shuffle_partner_index([s["id"] for s in final_set],
                                       cfg.str_("eval", "shuffle_salt"), T0.source_id_of)
    todo = list(range(len(final_set)))[: (a.limit or len(final_set))]
    columns_full = cfg.list_("eval", "columns")
    columns = [c for c in columns_full if c in FAST_COLUMNS]
    skipped = [c for c in columns_full if c not in columns]
    print(f"[batch] arm={model.arm} ckpt step={ck.get('step')} sha={ckpt_sha[:12]} n_eval={len(todo)}/"
          f"{len(final_set)} B={a.batch} columns={columns} skipped={skipped}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    recs: list[dict] = []
    buf: list[tuple[int, dict]] = []

    def flush():
        nonlocal buf
        if not buf:
            return
        items = [it for _, it in buf]
        pk = [keys[partner[i]] for i, _ in buf]
        recs.extend(eval_batch(items, model, feats, tokens, device, SS, ST, T0, path_mode, pk))
        buf = []
        if len(recs) % (a.batch * 25) < a.batch:
            print(f"  [batch] {len(recs)}/{len(todo)}  {(time.time() - t0) / len(recs):.2f} s/sample  "
                  f"peak alloc {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB", flush=True)
    for i in todo:
        it = load_item(T0, ST, final_set[i], blobs, n_steps, n_pix, inv)
        if buf and it["P"] != buf[0][1]["P"]:
            flush()
        buf.append((i, it))
        if len(buf) >= a.batch:
            flush()
    flush()
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    peak_res = torch.cuda.max_memory_reserved() / 2 ** 30

    def med(col, field):
        v = [r["cols"][col][field] for r in recs if r["cols"][col].get(field) is not None]
        return float(np.median(np.asarray(v, dtype=float))) if v else float("nan")
    overall = {c: ({f: med(c, f) for f in ("p50", "p95", "p99")} if c in ERR_COLS
                   else dict(value=med(c, "value"))) for c in columns}
    strata = cfg.list_("eval", "strata")
    by = {}
    for stratum in strata:
        groups = {}
        for r in recs:
            groups.setdefault(str(r[stratum]), []).append(r)
        by[stratum] = {}
        for k, v in sorted(groups.items()):
            cell = dict(n=len(v))
            for c in columns:
                fields = ("p50", "p95") if c in ERR_COLS else ("value",)
                cell[c] = {f: (float(np.median([r["cols"][c][f] for r in v if r["cols"][c].get(f) is not None]))
                               if any(r["cols"][c].get(f) is not None for r in v) else float("nan")) for f in fields}
            by[stratum][k] = cell
    frac = sum(1 for r in recs if r["cols"]["model"]["p50"] <= 5.0) / max(1, len(recs))
    smb = [r["cols"]["stage_mask_bitexact"]["value"] for r in recs]
    smb_all1 = all(v == 1.0 for v in smb if v is not None)
    print(f"\n=== batch-eval, median across {len(recs)} samples ===", flush=True)
    for c, v in overall.items():
        print(f"{c:18s} {v}", flush=True)
    print(f"frac_le5(derived)={frac:.4f}  stage_mask_bitexact all==1: {smb_all1}", flush=True)

    validation = None
    mp = out_dir / "metrics.json"
    if mp.is_file():
        ref = json.loads(mp.read_text())["heldout"]
        ref_by = {(x["id"], x["depth"]): x["cols"] for x in ref["per_sample"]}
        diffs = {f"{c}.{f}": [] for c in ERR_COLS for f in ("p50", "p95", "p99")}
        ctrl_diffs = {c: [] for c in ("delta_const", "delta_shuffle", "delta_edit_null", "delta_edit_roll")}
        missing = 0
        for r in recs:
            rc = ref_by.get((r["id"], r["depth"]))
            if rc is None:
                missing += 1; continue
            for c in ERR_COLS:
                for f in ("p50", "p95", "p99"):
                    diffs[f"{c}.{f}"].append(abs(r["cols"][c][f] - rc[c][f]))
            for c in ctrl_diffs:
                ctrl_diffs[c].append(abs(r["cols"][c]["value"] - rc[c]["value"]))
        maxd = {k: (max(v) if v else None) for k, v in diffs.items()}
        maxc = {k: (max(v) if v else None) for k, v in ctrl_diffs.items()}
        # headline 中位数（同一样本子集上）四位小数
        sub_ids = {(r["id"], r["depth"]) for r in recs}
        ref_sub = [x for x in ref["per_sample"] if (x["id"], x["depth"]) in sub_ids]
        head_ref = {f: float(np.median([x["cols"]["model"][f] for x in ref_sub])) for f in ("p50", "p95", "p99")}
        head_new = {f: overall["model"][f] for f in ("p50", "p95", "p99")}
        head_ok = all(round(head_ref[f], 4) == round(head_new[f], 4) for f in head_ref)
        per_ok = all(d is not None and d <= a.tol for d in maxd.values())
        validation = dict(reference=str(mp), n_samples=len(recs), missing_in_reference=missing,
                          tol=a.tol, max_abs_diff_per_sample=maxd, max_abs_diff_controls=maxc,
                          headline_ref_4dp={f: round(v, 4) for f, v in head_ref.items()},
                          headline_new_4dp={f: round(v, 4) for f, v in head_new.items()},
                          per_sample_within_tol=per_ok, headline_4dp_equal=head_ok,
                          stage_mask_bitexact_all_one=smb_all1,
                          passed=(per_ok and head_ok and smb_all1 and missing == 0))
        print(f"[batch] validation: per-sample max|Δ| {maxd}; controls max|Δ| {maxc}; headline ref/new "
              f"{validation['headline_ref_4dp']} / {validation['headline_new_4dp']}; PASSED={validation['passed']}",
              flush=True)

    outp = out_dir / a.out
    outp.write_text(json.dumps(dict(
        epr="EPR-051/stage0/sprf/BK-ABL-v2/batch-eval", arm=model.arm, tool=str(Path(__file__)),
        tool_sha256=T0.sha256_file(Path(__file__)), core_module=core.__name__,
        core_sha256=T0.sha256_file(Path(core.__file__)), stage_flow_module=SF.__name__,
        stage_flow_sha256=T0.sha256_file(Path(SF.__file__)), config=str(cfg_path),
        config_sha256=cfg.sha256, ckpt=str(ckpt_path), ckpt_sha256=ckpt_sha,
        ckpt_step=int(ck.get("step", -1)), batch=a.batch, n_heldout_total=len(final_set),
        n_eval=len(recs), limit=int(a.limit), columns_registered=columns, columns_skipped=skipped,
        linf8_frac_le5_derived=frac, stage_mask_bitexact_all_one=smb_all1,
        wall_seconds=round(wall, 1), seconds_per_sample=round(wall / max(1, len(recs)), 3),
        gpu_peak_gb=round(peak, 3), gpu_peak_reserved_gb=round(peak_res, 3),
        params=model.param_counts(), validation=validation,
        heldout=dict(n=len(recs), eval_kind="batch_final", step=int(ck.get("step", -1)),
                     metric="8-bit L-inf over RGB, |x_hat - before| * 255; per-sample p50/p95/p99, "
                            "then the MEDIAN across evaluated samples (batch-eval, B samples per forward)",
                     err_columns=list(ERR_COLS), scalar_columns=[c for c in columns if c not in ERR_COLS],
                     overall=overall, by=by, per_sample=recs)),
        ensure_ascii=False, indent=1))
    print(f"[batch] -> {outp}  wall {wall:.0f}s ({wall / max(1, len(recs)):.2f} s/sample) "
          f"peak alloc {peak:.2f} / reserved {peak_res:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
