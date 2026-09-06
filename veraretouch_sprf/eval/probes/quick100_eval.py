#!/usr/bin/env python3
# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/quick100_eval.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 · stage0 · sprf · X-COND **quick-100 预览评测**（非 headline）。

用某个臂的 `ckpt_best.pt` 在 **held-out 分层抽的 100 个样本**上只算三列
`model` / `identity` / `exact_solve`（linf8 p50/p95/p99）+ ΔE00 观察列。
**跳过** midpoint / 负控制 / κ / cycle / 逐阶段 / 掩膜守卫等重列 —— 那些是 final
全量表的事。

**这版数字不是 headline**：
* n = 100（final 是全量 4,560），子集由 `train_stage0.eval_subset` 按
  (geom, rec_band, depth) 分层、`sha1(salt + key)` 定序、跨层轮询取出，
  salt 落在输出 json 里；
* 子集的 identity 与全量的 identity **不是一个数**（分层构成不同），
  所以这张表只能自洽地读（model 对该子集的 identity），
  **不能**和全量表的 15.554 / 17.000 并排；
* 判读（J-C1/J-C2，判线 linf8_p50 <= 5）**一律以 final 全量表为准**，本表不参与。

    python quick100_eval.py --run /home/bc/data/runs/epr051_sprf/arm_clut \
        --device cuda:0 --n 100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
from veraretouch_sprf import _paths as _P  # EPR-052：替代原 sys.path 注入块
_P.ensure_sys_path()
STAGE0 = _P.STAGE0
REPO = _P.REPO

from veraretouch_sprf.data import train_stage0 as T0            # noqa: E402
import epr050_build_degradation as B  # noqa: E402
from veraretouch_sprf.data import stage_targets as ST           # noqa: E402
from veraretouch_sprf.models import stage_flow as SF              # noqa: E402
from veraretouch_sprf.solver import stage_solver as SS            # noqa: E402
from veraretouch_sprf.models import edit_cond as EC               # noqa: E402


def die(msg: str):
    raise SystemExit(f"quick100_eval: {msg}")


class MiniCfg:
    """只读替身：直接吃 run_args.json 里**冻结的** config 字典（与 ckpt 同源）。"""

    def __init__(self, d: dict):
        self.d = d

    def _get(self, sec, key):
        if sec not in self.d or key not in self.d[sec]:
            die(f"config: 缺 [{sec}] {key}")
        return self.d[sec][key]

    def int_(self, s, k):
        return int(self._get(s, k))

    def num(self, s, k):
        return float(self._get(s, k))

    def bool_(self, s, k):
        return bool(self._get(s, k))

    def str_(self, s, k, allowed=None):
        v = str(self._get(s, k))
        if allowed is not None and v not in allowed:
            die(f"[{s}] {k} = {v!r} 不在 {allowed}")
        return v

    def list_(self, s, k, kind=str):
        return list(self._get(s, k))


class StubLedger:
    """`exact_column` 只用到 `.note()`；这里不做 assert_wired（预览不是判读）。"""

    def __init__(self):
        self.n = {}

    def note(self, col, source=None, missing=False):
        self.n[col] = self.n.get(col, 0) + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="ckpt_best.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--salt", default="epr051-sprf-quick100:")
    ap.add_argument("--de00-max-px", type=int, default=4096)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = a.device
    run = Path(a.run)

    ras = sorted(run.glob("run_args*.json"))
    if not ras:
        die(f"{run}: 没有 run_args.json")
    ra = json.loads(ras[0].read_text())
    cd = ra["config"]
    mc = MiniCfg(cd)

    ST.install_compact_row(T0)
    T0.ASSET_MODE = cd["data"]["asset_source"]
    cfg = T0.Cfg(Path(ra["config_path"]))
    samples, blobs, _ = T0.load_shards(cfg)
    law = T0.bind_build_config(samples, cd["guard"]["build_config_sha256_allowed"])
    n_steps = law["n_steps"]
    held = sorted([s for s in samples if s["heldout"]],
                  key=lambda s: (s["id"], s["depth"] or 0))
    subset = T0.eval_subset(held, int(a.n), a.salt)
    print(f"[quick100] held-out {len(held)} -> subset {len(subset)}"
          f" (salt {a.salt!r})", flush=True)

    # ---- 模型（与 ckpt 同源的冻结 config） -------------------------------- #
    in_dim = int(ra["encoder"]["dim"]) * len(cd["encoder"]["inputs"])
    model = SF.SprfModel(in_dim, mc, n_steps, cd["flow"]["alpha_mode"],
                         cd["data"]["depth_values"]).to(dev)
    esrc = None
    econd = cd.get("edit", {}).get("condition", "off")
    if econd == "inv_lut":
        esrc = EC.InvLutSource(cd["edit"]["inv_cache_dir"], int(cd["edit"]["grid"]))
        model.load_inv_table(esrc.load_table())
    elif econd == "ref_pair":
        die("ref_pair 臂的 quick 评测需要参考池，本预览工具暂只接 inv_lut / off")
    cp = torch.load(run / a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(cp["model"])
    model.eval()
    step = cp.get("step")
    print(f"[quick100] {run.name} {a.ckpt} step={step} "
          f"contract={cd.get('edit', {}).get('contract', 'none')}", flush=True)

    pooled = {}
    want = {T0.feature_key(s) for s in subset}
    for f in sorted(Path(cd["encoder"]["cache_dir"]).glob("siglip_*.pt")):
        for k, v in torch.load(f, map_location="cpu")["features"].items():
            if k in want:
                pooled[k] = v
    missing = want - set(pooled)
    if missing:
        die(f"特征缓存缺 {len(missing)} 个 key，例如 {sorted(missing)[:3]}")

    n_pix = int(cd["eval"]["pixels_per_sample"])
    path_mode = cd["flow"]["path_mode"]
    nfe = int(cd["solver"]["nfe_per_stage"])
    lo, hi = float(cd["solver"]["clamp_lo"]), float(cd["solver"]["clamp_hi"])
    ledger = StubLedger()
    recs = []
    t0 = time.time()
    with torch.no_grad():
        for i, s in enumerate(subset):
            row = json.loads(blobs[s["id"]])
            x0, y = T0.load_pair(s["shard"], row, s["after_asset"])
            af = T0.alpha_fields(row, x0)
            npx = x0.shape[0] * x0.shape[1]
            idx = (torch.arange(npx) if n_pix <= 0 else
                   torch.arange(0, npx, max(1, npx // n_pix))[:n_pix])
            mask = T0.depth_mask(s["depth"], n_steps)
            alphas = (af.reshape(n_steps, -1)[:, idx]
                      * mask.unsqueeze(-1)).to(dev).unsqueeze(0)
            xb = x0.reshape(-1, 3)[idx].to(dev).unsqueeze(0)
            yb = y.reshape(-1, 3)[idx].to(dev).unsqueeze(0)
            d_int = n_steps if s["depth"] is None else int(s["depth"])
            depth = torch.tensor([d_int], device=dev)
            sc = torch.tensor([float(row["calib"]["s"])], device=dev)
            union = ST.union_mask_of(alphas)
            ahat = ST.compose_alpha_hat(alphas)
            base = model.cond.base(T0.gather_feats(pooled, [T0.feature_key(s)], 0, dev))
            ed = (None if esrc is None else
                  esrc.for_row(row, x0, mask).unsqueeze(0).to(dev))
            n_st = SS.stages_for(path_mode, d_int)
            pred, _ = SS.rollout_with_metrics(
                model, base, yb, ahat, alphas, union, depth, sc, path_mode,
                n_st, nfe, clamp_lo=lo, clamp_hi=hi, edits=ed)
            r = dict(id=s["id"], depth=d_int, geom=s["geom"],
                     rec_band=s.get("rec_band"))
            r["model"] = T0.err_stats(T0.linf8(pred, xb))
            r["identity"] = T0.err_stats(T0.linf8(yb, xb))
            r["exact_solve"] = T0.exact_column(s, idx, xb, ledger, "journal", dev)
            dm = EC.de00_pixels(pred[0], xb[0], int(a.de00_max_px), B.de00)
            di = EC.de00_pixels(yb[0], xb[0], int(a.de00_max_px), B.de00)
            r["de00"] = dict(p50=float(np.median(dm)),
                             p95=float(np.quantile(dm, 0.95)),
                             frac_le5=float((dm <= 5.0).mean()))
            r["de00_identity"] = dict(p50=float(np.median(di)),
                                      frac_le5=float((di <= 5.0).mean()))
            recs.append(r)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(subset)} ({time.time()-t0:.0f}s)", flush=True)

    def med(col, field):
        v = [r[col][field] for r in recs
             if col in r and r[col].get(field) is not None]
        return float(np.median(np.asarray(v, dtype=float))) if v else float("nan")

    overall = {}
    for c in ("model", "identity", "exact_solve"):
        overall[c] = {f: med(c, f) for f in ("p50", "p95", "p99")}
    overall["de00"] = dict(p50=med("de00", "p50"), p95=med("de00", "p95"),
                           frac_le5=med("de00", "frac_le5"))
    overall["de00_identity"] = dict(p50=med("de00_identity", "p50"),
                                    frac_le5=med("de00_identity", "frac_le5"))

    out = dict(
        epr="EPR-051/stage0/sprf/X-COND",
        kind="quick-100 preview",
        headline=False,
        disclaimer="quick-100,**非 headline**;n=100 的分层子集,其 identity 与全量"
                   "4,560 的 identity 不是一个数,禁与全量表(CHAINEND 15.554 / "
                   "identity 17.000)并排读。J-C1/J-C2 判读一律以 final 全量表为准。",
        run=str(run), ckpt=a.ckpt, ckpt_step=step,
        condition_contract=cd.get("edit", {}).get("contract", "none"),
        oracle=(cd.get("edit", {}).get("contract", "none") != "none"),
        n=len(recs), subset_salt=a.salt,
        subset_rule="train_stage0.eval_subset:(geom,rec_band,depth) 分层 + "
                    "sha1(salt+key) 定序 + 跨层轮询",
        eval_pixels_per_sample=n_pix, de00_max_px=int(a.de00_max_px),
        columns_computed=["model", "identity", "exact_solve", "de00",
                          "de00_identity"],
        columns_skipped=["model_midpoint", "delta_const", "delta_shuffle",
                         "delta_edit_null", "delta_edit_roll",
                         "condition_number_by_stage", "cycle", "stage_*",
                         "mask_*", "rollout_drift", "nfe_d_vs_2d",
                         "latency_by_depth"],
        generated=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        wall_seconds=round(time.time() - t0, 1),
        overall=overall, per_sample=recs)
    dst = Path(a.out) if a.out else run / "quick100.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1))

    print(f"\n=== quick-100 预览（**非 headline**，n={len(recs)}）===", flush=True)
    for c in ("model", "identity", "exact_solve"):
        v = overall[c]
        print(f"{c:14s} p50={v['p50']:9.3f} p95={v['p95']:9.3f} p99={v['p99']:9.3f}")
    print(f"{'de00':14s} p50={overall['de00']['p50']:9.3f} "
          f"p95={overall['de00']['p95']:9.3f} "
          f"frac<=5={overall['de00']['frac_le5']:.3f}  (观察列)")
    print(f"{'de00_identity':14s} p50={overall['de00_identity']['p50']:9.3f} "
          f"{'':9s}  frac<=5={overall['de00_identity']['frac_le5']:.3f}  (观察列)")
    print(f"-> {dst}", flush=True)


if __name__ == "__main__":
    main()
