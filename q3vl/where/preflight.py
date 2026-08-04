"""Protocol 14, items 4/5/6 -- the Where-A share of the mandatory preflight.

    4. real ``F_pre`` shape, aspect ratio, position encoding and one guided
       upsample check;
    5. ``BA-3-Joint`` residualisation orthogonality error, condition number and
       oracle fit success rate;
    6. ``w_dir`` sign, unit norm, ``alpha`` positivity and readout boundary tests.

Plus one repo-specific check that falls out of the freeze contract:

    4b. ``F_pre`` is invariant to Base SFT.  ``q3vl/train/freeze.py`` freezes
        ``patch_embed``, ``pos_embed`` and all 24 vision blocks, so the last
        block's output under a Base-SFT checkpoint must equal the base model's
        bit for bit.  If it does not, either the freeze leaked or the checkpoint
        is not the model we think it is.

Every check returns a row with ``id / status / detail``; ``run`` writes the
whole thing to JSON and exits non-zero on any failure, so it can gate a launch.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .basis import sign_index
from .calibrate import Calibrator
from .config import (
    CBAND_M, FPRE_DIM, MODEL_DIR, REPORT_DIR, SFT_CHECKPOINTS,
    CalibConfig, FitConfig,
)
from .fpre import (
    fpre_facts, load_vision_tower, shuffle_from_grid, unshuffle_to_grid,
)
from .readout import (
    apply_readout, band_params, bounds_report, cband_centres, cband_params, param_shapes,
)
from .upsample import ChannelOrderError, guided_upsample

__all__ = ["Check", "PreflightReport", "run_where_a_preflight"]


@dataclass
class Check:
    id: str
    status: str                      # pass | fail | skip
    detail: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "status": self.status, "message": self.message,
                "detail": self.detail}


@dataclass
class PreflightReport:
    checks: list[Check] = field(default_factory=list)
    env: dict[str, Any] = field(default_factory=dict)

    def add(self, check: Check) -> Check:
        self.checks.append(check)
        return check

    @property
    def ok(self) -> bool:
        return all(c.status != "fail" for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "n_pass": sum(c.status == "pass" for c in self.checks),
            "n_fail": sum(c.status == "fail" for c in self.checks),
            "n_skip": sum(c.status == "skip" for c in self.checks),
            "env": self.env,
            "checks": [c.to_dict() for c in self.checks],
        }


def _env() -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, cwd=Path(__file__).resolve().parents[2]).stdout.strip()
    except Exception:                                 # noqa: BLE001
        commit = ""
    return {
        "git_commit": commit,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


# --- item 6 (pure CPU, no data) ---------------------------------------------

def check_readout_boundaries() -> Check:
    """14.6 -- bounded parameterisations hold at extreme raw values."""
    detail: dict[str, Any] = {}
    bad: list[str] = []
    for raw_val in (-1e6, -50.0, 0.0, 50.0, 1e6):
        b = band_params({k: torch.tensor(raw_val, dtype=torch.float64)
                         for k in param_shapes("band")})
        c = cband_params({k: torch.full((CBAND_M,), raw_val, dtype=torch.float64)
                          for k in param_shapes("cband12")})
        if not (float(b["h"]) > 0):
            bad.append(f"h<=0 at raw={raw_val}")
        if not (1.0 <= float(b["k"]) <= 40.0):
            bad.append(f"k out of [1,40] at raw={raw_val}")
        if not (0.0 <= float(b["pi"]) <= 1.0):
            bad.append(f"pi out of [0,1] at raw={raw_val}")
        if not bool(((c["sigma"] >= 0.025) & (c["sigma"] <= 0.30)).all()):
            bad.append(f"sigma out of [0.025,0.30] at raw={raw_val}")
        detail[str(raw_val)] = {"h": float(b["h"]), "k": float(b["k"]),
                                "pi": float(b["pi"]), "sigma": float(c["sigma"][0])}
    mu = cband_centres(dtype=torch.float64)
    detail["mu_grid"] = mu.tolist()
    if not np.allclose(mu.numpy(), np.linspace(-3, 3, CBAND_M)):
        bad.append("CBand centres are not linspace(-3,3,12)")
    if not torch.allclose(mu, -torch.flip(mu, dims=(0,)), atol=1e-12):
        bad.append("CBand centre grid is not symmetric (mirror would be wrong)")
    # sanity: the readouts stay in [0,1]
    z = torch.linspace(-4, 4, 401, dtype=torch.float64)
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        rb = {k: torch.randn((), generator=g, dtype=torch.float64) for k in param_shapes("band")}
        rc = {k: torch.randn(CBAND_M, generator=g, dtype=torch.float64)
              for k in param_shapes("cband12")}
        for name, raw in (("band", rb), ("cband12", rc)):
            m = apply_readout(name, z, raw)
            if float(m.min()) < -1e-9 or float(m.max()) > 1 + 1e-9:
                bad.append(f"{name} left [0,1]: [{float(m.min())}, {float(m.max())}]")
    return Check("WA-P6a-readout-bounds", "fail" if bad else "pass", detail,
                 "; ".join(bad[:4]))


def check_upsample_order() -> Check:
    """14.4 -- exactly one guided upsample, applied to the scalar only."""
    detail: dict[str, Any] = {}
    bad: list[str] = []
    guide = torch.rand(1, 1, 128, 192, dtype=torch.float64)
    try:
        guided_upsample(torch.randn(1, 64, 8, 12, dtype=torch.float64), guide)
        bad.append("multi-channel upsample was accepted")
    except ChannelOrderError:
        detail["multichannel_refused"] = True
    out = guided_upsample(torch.randn(1, 1, 8, 12, dtype=torch.float64), guide)
    detail["scalar_shape"] = list(out.shape)
    if tuple(out.shape) != (1, 1, 128, 192):
        bad.append(f"scalar upsample shape {tuple(out.shape)}")
    const = guided_upsample(torch.full((1, 1, 8, 12), 1.25, dtype=torch.float64), guide)
    detail["constant_preserved_max_err"] = float((const - 1.25).abs().max())
    if detail["constant_preserved_max_err"] > 1e-6:
        bad.append("constant field not preserved")
    return Check("WA-P4c-upsample-order", "fail" if bad else "pass", detail, "; ".join(bad))


# --- item 4 (needs the real vision tower) -----------------------------------

def check_fpre_geometry(source, limit: int) -> tuple[Check, list]:
    """14.4 -- real shape, real aspect ratio, real spatial ordering."""
    rows: list[dict[str, Any]] = []
    prepared: list = []
    bad: list[str] = []
    for ps in source.iter_split("V_where", limit=limit):
        s = ps.sample
        geom = ps.geometry
        if s.fpre.shape != (s.grid_h * s.grid_w, FPRE_DIM):
            bad.append(f"{s.sample_id}: F_pre {tuple(s.fpre.shape)}")
        if (s.grid_h, s.grid_w) != (geom.out_h // 16, geom.out_w // 16):
            bad.append(f"{s.sample_id}: grid {(s.grid_h, s.grid_w)} != H/16 x W/16")
        a_grid = s.grid_w / s.grid_h
        a_img = geom.out_w / geom.out_h
        if abs(a_grid - a_img) / a_img > 1e-6:
            bad.append(f"{s.sample_id}: grid aspect {a_grid} != image aspect {a_img}")

        # spatial coherence: with the correct unshuffle, neighbouring tokens are
        # far more similar than random pairs.  A naive reshape scrambles 2x2
        # blocks and loses most of that gain -- this is what makes the check
        # sensitive to the merge-order bug rather than to the shapes only.
        grid = s.fpre.reshape(s.grid_h, s.grid_w, FPRE_DIM).double()
        naive = shuffle_from_grid(grid).reshape(s.grid_h, s.grid_w, FPRE_DIM)
        coh = _coherence(grid)
        coh_naive = _coherence(naive)
        rows.append({"sample_id": s.sample_id, "grid": [s.grid_h, s.grid_w],
                     "out": [geom.out_h, geom.out_w], "aspect_out": a_img,
                     "coherence": coh, "coherence_naive_reshape": coh_naive,
                     "upscaled": ps.record["image"].get("upscaled")})
        if coh <= coh_naive:
            bad.append(f"{s.sample_id}: unshuffled grid is not more coherent "
                       f"({coh:.4f} <= {coh_naive:.4f})")
        prepared.append(ps)

    detail = {"n": len(rows), "rows": rows[:20],
              "coherence_median": float(np.median([r["coherence"] for r in rows])) if rows else None,
              "coherence_naive_median": float(
                  np.median([r["coherence_naive_reshape"] for r in rows])) if rows else None}
    if not rows:
        return Check("WA-P4a-fpre-geometry", "fail", detail, "no samples"), prepared
    return Check("WA-P4a-fpre-geometry", "fail" if bad else "pass", detail,
                 "; ".join(bad[:4])), prepared


def _coherence(grid: torch.Tensor) -> float:
    """mean cosine(neighbour) - mean cosine(random pair)."""
    x = grid / (grid.norm(dim=-1, keepdim=True) + 1e-12)
    right = (x[:, :-1] * x[:, 1:]).sum(-1).mean()
    down = (x[:-1] * x[1:]).sum(-1).mean()
    flat = x.reshape(-1, x.shape[-1])
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, flat.shape[0], (2, 4096), generator=g)
    rand = (flat[idx[0]] * flat[idx[1]]).sum(-1).mean()
    return float((right + down) / 2 - rand)


def check_fpre_invariant_to_sft(model_dir: Path, ckpt: Path, device: str) -> Check:
    """4b -- the Base SFT freeze implies an identical last vision block output.

    Fail-closed: any exception (a checkpoint whose layout ``load_vision_tower``
    rejects, a truncated shard, ...) becomes a ``fail`` row rather than an
    exception that escapes the driver and takes the whole JSON report with it
    (REVIEW-impl-WhereA N-5b).
    """
    if not Path(ckpt).exists():
        return Check("WA-P4b-fpre-sft-invariance", "skip",
                     {"checkpoint": str(ckpt)}, "checkpoint does not exist yet")
    try:
        base = load_vision_tower(model_dir, dtype=torch.float32, device=device)
        tuned = load_vision_tower(ckpt, dtype=torch.float32, device=device)
        max_abs = 0.0
        n_diff = 0
        changed: list[str] = []
        sb, st = base.state_dict(), tuned.state_dict()
        missing = [k for k in sb if k not in st]
        if missing:
            return Check("WA-P4b-fpre-sft-invariance", "fail",
                         {"checkpoint": str(ckpt), "missing_in_checkpoint": missing[:8]},
                         "checkpoint does not carry the same vision tensors")
        for k in sb:
            if k.startswith("merger.") or k.startswith("deepstack_merger_list."):
                continue                               # these are trained on purpose
            d = float((sb[k] - st[k]).abs().max())
            if d > 0:
                n_diff += 1
                if len(changed) < 8:
                    changed.append(k)
            max_abs = max(max_abs, d)
    except Exception as exc:                           # noqa: BLE001 -- fail, never crash
        return Check("WA-P4b-fpre-sft-invariance", "fail",
                     {"checkpoint": str(ckpt), "error": f"{type(exc).__name__}: {exc}"},
                     "could not compare the checkpoint's vision tower")
    detail = {"checkpoint": str(ckpt), "max_abs_weight_diff": max_abs,
              "n_tensors_changed": n_diff, "changed_examples": changed}
    return Check("WA-P4b-fpre-sft-invariance", "pass" if max_abs == 0.0 else "fail",
                 detail, "" if max_abs == 0.0 else
                 "frozen vision weights changed during Base SFT")


def check_position_encoding(visual, grid_h: int = 8, grid_w: int = 12) -> Check:
    """14.4 "position encoding" -- checked directly, not via a proxy.

    ``fast_pos_embed_interpolate`` bilinearly resamples the ``num_grid_per_side``
    learned position lattice onto the image's (h, w) grid and then permutes into
    merge order.  Here that is recomputed from ``pos_embed.weight`` independently
    and compared after :func:`unshuffle_to_grid`, so both the interpolation and
    the merge-order permutation are pinned (REVIEW-impl-WhereA N-6).
    """
    try:
        with torch.no_grad():
            grid_thw = torch.tensor([[1, grid_h, grid_w]], device=next(visual.parameters()).device)
            got = visual.fast_pos_embed_interpolate(grid_thw).double()
            got_grid = unshuffle_to_grid(got, grid_h, grid_w)

            n = visual.num_grid_per_side
            w = visual.pos_embed.weight.double()
            hs = torch.linspace(0, n - 1, grid_h, dtype=torch.float64)
            ws = torch.linspace(0, n - 1, grid_w, dtype=torch.float64)
            h0 = hs.floor().long()
            w0 = ws.floor().long()
            h1 = (h0 + 1).clamp(max=n - 1)
            w1 = (w0 + 1).clamp(max=n - 1)
            dh = (hs - h0.double()).unsqueeze(1)
            dw = (ws - w0.double()).unsqueeze(0)
            want = (
                (1 - dh)[..., None] * (1 - dw)[..., None] * w[(h0[:, None] * n + w0[None, :])]
                + (1 - dh)[..., None] * dw[..., None] * w[(h0[:, None] * n + w1[None, :])]
                + dh[..., None] * (1 - dw)[..., None] * w[(h1[:, None] * n + w0[None, :])]
                + dh[..., None] * dw[..., None] * w[(h1[:, None] * n + w1[None, :])]
            )
            err = float((got_grid - want).abs().max())
            rel = err / float(want.abs().max().clamp_min(1e-30))
    except Exception as exc:                           # noqa: BLE001
        return Check("WA-P4d-position-encoding", "fail",
                     {"error": f"{type(exc).__name__}: {exc}"},
                     "could not verify the position encoding")
    detail = {"grid": [grid_h, grid_w], "num_grid_per_side": int(visual.num_grid_per_side),
              "max_abs_error": err, "max_rel_error": rel}
    ok = rel < 1e-5
    return Check("WA-P4d-position-encoding", "pass" if ok else "fail", detail,
                 "" if ok else "independent bilinear recomputation disagrees "
                               "with fast_pos_embed_interpolate after unshuffle")


# --- item 5 + rest of item 6 (needs data + a projector) ---------------------

def check_calibration_health(prepared: list, arm: str, fit_cfg: FitConfig,
                             device: str) -> list[Check]:
    """14.5 residualisation / condition number / fit success, and 14.6 on the
    latents the fit actually produced."""
    cal = Calibrator(CalibConfig(arm=arm, inner_fit=fit_cfg), device=device)
    samples = [p.sample for p in prepared]
    t0 = time.time()
    report = cal.evaluate(samples, record_fits=True)
    elapsed = time.time() - t0
    n_fits = max(1, report["n_samples"] * len(report["per_readout"]))

    diags = [r["phi_diag"] for r in report["rows"] if "phi_diag" in r]
    corr = [d["resid_corr_after"] for d in diags if d.get("resid_corr_after") is not None]
    cond_design = [d["design_gram_cond"] for d in diags]
    cond_phi = [d["phi_gram_cond"] for d in diags]
    dead = [d["n_dead_semantic"] for d in diags]
    detail5 = {
        "arm": arm,
        "n_samples": report["n_samples"],
        # N-7: this is the *starting* projector (seeded orthogonal), i.e. the
        # BA-0 configuration -- it is a gate before the run, not a calibrated
        # number.  Re-run after calibration to see how conditioning moved.
        "projector_stage": "seeded_orthogonal_start_not_calibrated",
        "projector": report["projector"],
        # N-8: these are condition numbers of the GRAM matrix, i.e. the square of
        # the matrix condition number.  The 1e10 gate is 1e5 on phi itself.
        "condition_number_basis": "gram_matrix (= matrix cond ^ 2)",
        "residual_corr_after": {"max": max(corr) if corr else None,
                                "median": float(np.median(corr)) if corr else None},
        "design_gram_cond": {"max": max(cond_design), "median": float(np.median(cond_design))},
        "phi_gram_cond": {"max": max(cond_phi), "median": float(np.median(cond_phi))},
        "n_dead_semantic": {"max": max(dead), "median": float(np.median(dead))},
        "fit": {r: {"success_rate": v["fit_success_rate"],
                    "n_ok": v["n_ok"], "n_rejected": v["n_rejected"],
                    "reject_reasons": v["reject_reasons"],
                    "headline_low_soft_iou": v["headline_low"].get("soft_iou_minmax"),
                    "headline_hi_soft_iou": v["headline_hi"].get("soft_iou_minmax")}
                for r, v in report["per_readout"].items()},
        # N-15: the inner L-BFGS is the only wall-clock bottleneck of Where-A and
        # had no measurement at all.  This is the number the S4 schedule needs.
        "throughput": {
            "device": device,
            "n_fits": n_fits,
            "elapsed_s": round(elapsed, 2),
            "s_per_fit": round(elapsed / n_fits, 3),
            "fit_cfg": dict(fit_cfg.__dict__),
            "projected_hours_per_arm_42752": round(elapsed / n_fits * 42752 * 2 / 3600, 2),
            "projected_hours_per_arm_75544": round(elapsed / n_fits * 75544 * 2 / 3600, 2),
        },
    }
    bad5 = []
    if corr and max(corr) > 1e-4:
        bad5.append(f"residualisation leaves correlation {max(corr):.2e}")
    if max(cond_phi) > 1e10:
        bad5.append(f"phi Gram condition number {max(cond_phi):.2e}")
    for r, v in report["per_readout"].items():
        if v["fit_success_rate"] < 0.90:
            bad5.append(f"{r} oracle fit success {v['fit_success_rate']:.2%} < 90%")
    c5 = Check("WA-P5-basis-conditioning", "fail" if bad5 else "pass", detail5,
               "; ".join(bad5))

    # --- B-4: the delivered-resolution path, on real data --------------------
    hi_detail: dict[str, Any] = {"n_with_hi": report["n_with_hi_res"],
                                 "upsample": report["upsample"], "per_readout": {}}
    bad_hi: list[str] = []
    if report["n_with_hi_res"] == 0:
        bad_hi.append("no sample carried mask_hi/guide_hi: the high-resolution "
                      "path was never exercised (attach_hi=False?)")
    for r, v in report["per_readout"].items():
        dom = v["s_domain"]
        hi_detail["per_readout"][r] = {
            "headline_low": v["headline_low"].get("soft_iou_minmax"),
            "headline_hi": v["headline_hi"].get("soft_iou_minmax"),
            "s_domain": dom,
        }
        if dom.get("raw_max") is None:
            continue
        # the domain assertion the CLAUDE.md s-cache contract demands: state the
        # expected domain, then show what the raw data actually did.
        if not dom.get("clamped") and (dom["raw_max"] > 3.0 or dom["raw_min"] < -3.0):
            bad_hi.append(
                f"{r}: guided upsample left the declared s domain "
                f"([{dom['raw_min']:.2f}, {dom['raw_max']:.2f}]) and clamping is off"
            )
        lo_hi = hi_detail["per_readout"][r]
        if lo_hi["headline_hi"] and lo_hi["headline_low"]:
            drop = lo_hi["headline_low"]["median"] - lo_hi["headline_hi"]["median"]
            hi_detail["per_readout"][r]["median_drop_low_to_hi"] = drop
    c_hi = Check("WA-P4e-highres-path", "fail" if bad_hi else "pass", hi_detail,
                 "; ".join(bad_hi[:3]))

    bad6, worst_norm, worst_alpha = [], 0.0, float("inf")
    n_canon = 0
    for row in report["rows"]:
        lat = row.get("latent")
        if lat is None:                     # a rejected fit has no latent (B-1)
            continue
        wd = torch.tensor(lat["w_dir"], dtype=torch.float64)
        worst_norm = max(worst_norm, abs(float(wd.norm()) - 1.0))
        worst_alpha = min(worst_alpha, lat["alpha"])
        if lat["canonical"]:
            n_canon += 1
        else:
            bad6.append(f"{row['sample_id']}/{row['readout']}: sign rule violated")
        if float(wd[sign_index(wd)]) <= 0 and lat["canonical"]:
            bad6.append(f"{row['sample_id']}: canonical flag disagrees with the sign")
        if lat["alpha"] < 0:
            bad6.append(f"{row['sample_id']}: alpha < 0")
        b = bounds_report(row["readout"],
                          {k: torch.tensor(v, dtype=torch.float64)
                           for k, v in lat["rho_raw"].items()})
        if not all(v for k, v in b.items() if k.endswith("in_bounds")):
            bad6.append(f"{row['sample_id']}/{row['readout']}: readout out of bounds")
    n_latents = sum(1 for r in report["rows"] if r.get("latent") is not None)
    detail6 = {
        "n_rows": len(report["rows"]),
        "n_latents": n_latents,
        "n_canonical": n_canon,
        "max_unit_norm_error": worst_norm,
        "min_alpha": worst_alpha if worst_alpha != float("inf") else None,
    }
    if worst_norm > 1e-9:
        bad6.append(f"||w_dir|| off by {worst_norm:.2e}")
    if n_latents == 0:
        bad6.append("no latent survived the fit")
    c6 = Check("WA-P6b-latent-invariants", "fail" if bad6 else "pass", detail6,
               "; ".join(bad6[:4]))
    return [c5, c_hi, c6]


# --- driver -----------------------------------------------------------------

def run_where_a_preflight(
    *, limit: int = 24, device: str = "cuda", arm: str = "BA-3-Joint",
    model_dir: Path = MODEL_DIR, checkpoint: Path | None = None,
    dtype: str = "bfloat16", fit_cfg: FitConfig | None = None,
    out: Path | None = None, skip_model: bool = False,
) -> PreflightReport:
    from transformers import AutoProcessor

    from .pipeline import WhereADataSource

    rep = PreflightReport(env=_env())
    rep.env.update({"limit": limit, "device": device, "arm": arm,
                    "model_dir": str(model_dir), "checkpoint": str(checkpoint or "")})
    out = Path(out) if out else REPORT_DIR / "preflight_where_a.json"

    def flush() -> None:
        """Always leave a JSON behind, even on an unexpected exception (N-5b):
        a preflight that dies without a report is indistinguishable from one that
        was never run."""
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep.to_dict(), indent=2, ensure_ascii=False))

    try:
        # data-free checks first: they gate the expensive ones
        rep.add(check_readout_boundaries())
        rep.add(check_upsample_order())

        if skip_model:
            for cid in ("WA-P4a-fpre-geometry", "WA-P4d-position-encoding",
                        "WA-P5-basis-conditioning", "WA-P4e-highres-path",
                        "WA-P6b-latent-invariants"):
                rep.add(Check(cid, "skip", {}, "--skip-model"))
        else:
            src_dir = Path(checkpoint) if checkpoint else Path(model_dir)
            visual = load_vision_tower(src_dir, dtype=getattr(torch, dtype), device=device)
            rep.env["fpre_facts"] = fpre_facts(visual)
            processor = AutoProcessor.from_pretrained(model_dir)
            # attach_hi=True: the delivered-resolution path must be exercised on
            # real data, not just on synthetic tensors (B-4).
            source = WhereADataSource(visual, processor, device=device, attach_hi=True)
            c4, prepared = check_fpre_geometry(source, limit)
            rep.add(c4)
            rep.add(check_position_encoding(visual))
            for c in check_calibration_health(
                prepared, arm, fit_cfg or FitConfig(n_random=3, max_iter=80), device
            ):
                rep.add(c)
            source.close()

        for ckpt in ([checkpoint] if checkpoint else list(SFT_CHECKPOINTS)):
            rep.add(check_fpre_invariant_to_sft(model_dir, Path(ckpt), "cpu"))
    except BaseException as exc:                       # noqa: BLE001 -- fail-closed
        rep.add(Check("WA-P0-preflight-driver", "fail",
                      {"error": f"{type(exc).__name__}: {exc}"},
                      "preflight aborted; the report below is partial"))
        flush()
        raise
    flush()
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description="Protocol 14 items 4/5/6 for Where-A")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arm", default="BA-3-Joint")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--skip-model", action="store_true",
                    help="run only the data-free checks (no GPU, no weights)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rep = run_where_a_preflight(
        limit=args.limit, device=args.device, arm=args.arm,
        model_dir=Path(args.model_dir),
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        dtype=args.dtype, out=Path(args.out) if args.out else None,
        skip_model=args.skip_model,
    )
    for c in rep.checks:
        print(f"[{c.status.upper():4}] {c.id}  {c.message}")
    print(f"\npreflight {'PASS' if rep.ok else 'FAIL'}")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
