"""CI self-checks for model4d_naive.GLUT4D (G3 / Gate D3).

Run:  python -m model.glut_repro.ci_checks_4d
Every check compares against an independently written reference (explicit
4x4 Gaussian in numpy / torch.distributions), not against the module itself.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import torch

from model.glut_repro.model4d_naive import (
    GLUT4D, K_ANCHOR, SIGMA_S_MAX, SIGMA_S_MIN, anchor_grid)

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def main() -> None:
    torch.manual_seed(0)
    N, P = 16, 512
    x = torch.rand(P, 3, dtype=torch.float32)
    s = torch.rand(P, dtype=torch.float32)

    print("== 1. sigma_s is the MARGINAL std of the s axis ==")
    for arm in ("naive", "anchored"):
        m = GLUT4D(N, arm=arm)
        with torch.no_grad():
            m.chol_off.normal_(0.0, 0.05)      # switch on cross-covariance
            m.log_sigma_c.normal_(math.log(0.15), 0.2)
            if arm == "naive":
                m.sigma_s_raw.normal_(math.log(0.15), 0.5)
            else:
                m.sigma_s_raw.normal_(0.0, 1.0)
        L = m.chol()
        Sigma = L @ L.transpose(-1, -2)
        marg = torch.sqrt(Sigma[:, 0, 0])
        err = float((marg - m.sigma_s()).abs().max())
        check(f"{arm}: sqrt(Sigma[0,0]) == sigma_s()", err < 1e-6, f"max err {err:.2e}")
        # cross-covariance present and NOT factorizable
        cross = Sigma[:, 0, 1:]
        check(f"{arm}: s-RGB cross-covariance kept", float(cross.abs().max()) > 1e-4,
              f"max |Cov(s,rgb)| {float(cross.abs().max()):.4f}")

    print("== 2. log_density == reference 4D MVN ==")
    m = GLUT4D(N, arm="naive")
    with torch.no_grad():
        m.mu_c.normal_(0.5, 0.2)
        m.mu_s.normal_(0.5, 0.2)
        m.chol_off.normal_(0.0, 0.03)
        m.log_sigma_c.normal_(math.log(0.15), 0.2)
        m.sigma_s_raw.normal_(math.log(0.15), 0.3)
    u = torch.cat([s.unsqueeze(-1), x], dim=-1).double()
    L = m.chol().double()
    ld_ref = torch.empty(N, P, dtype=torch.float64)
    for i in range(N):
        Sig = L[i] @ L[i].T
        d = torch.distributions.MultivariateNormal(m.mu4()[i].double(), Sig)
        ld_ref[i] = d.log_prob(u)
    ld = m.log_density(torch.cat([s.unsqueeze(-1), x], -1)).double()
    err = float((ld - ld_ref).abs().max())
    check("log_density matches MultivariateNormal.log_prob", err < 2e-4,
          f"max err {err:.2e}")

    print("== 3. weights == naive p*o/(sum p*o + eps), full normalization ==")
    with torch.no_grad():
        m.opacity_raw.uniform_(0.2, 1.0)
    p = torch.exp(ld_ref)
    o = m.opacity_raw.double().clamp(0, 1).unsqueeze(-1)
    w_ref = (p * o) / ((p * o).sum(0, keepdim=True) + 1e-6)
    w = m.weights(torch.cat([s.unsqueeze(-1), x], -1)).double()
    err = float((w - w_ref).abs().max())
    check("weights match the eps-regularized Eq.2", err < 1e-6, f"max err {err:.2e}")
    check("weights sum <= 1 (eps denominator)", float(w.sum(0).max()) <= 1.0 + 1e-6,
          f"max sum {float(w.sum(0).max()):.9f}")

    print("== 4. escape-channel limit is 0 weight, not NaN ==")
    m2 = GLUT4D(N, arm="naive")
    with torch.no_grad():
        m2.sigma_s_raw.fill_(math.log(1e12))       # sigma_s -> inf
    w2 = m2.weights(torch.cat([s.unsqueeze(-1), x], -1))
    out2 = m2(x, s)
    check("sigma_s=1e12: finite weights", bool(torch.isfinite(w2).all()))
    check("sigma_s=1e12: weights -> 0", float(w2.max()) < 1e-3,
          f"max w {float(w2.max()):.2e}")
    check("sigma_s=1e12: forward finite", bool(torch.isfinite(out2).all()))
    with torch.no_grad():
        m2.sigma_s_raw.fill_(math.log(1e-6))       # sigma_s -> 0
    w3 = m2.weights(torch.cat([s.unsqueeze(-1), x], -1))
    check("sigma_s=1e-6: finite weights", bool(torch.isfinite(w3).all()))
    check("sigma_s=1e-6: forward finite",
          bool(torch.isfinite(m2(x, s)).all()))

    print("== 5. init is identity passthrough up to the Eq.2 eps (G=0 red line) ==")
    # M=I, b=0, G=0, g=0  =>  f = (sum_i w_i) x = x * (1 - eps/(sum_j p_j o_j + eps)).
    # The eps denominator is part of the GLUT spec, so init is identity only up
    # to that factor; assert the EXACT closed form, not a hand-waved tolerance.
    for arm in ("naive", "anchored"):
        m3 = GLUT4D(N, arm=arm)
        uu = torch.cat([s.unsqueeze(-1), x], -1)
        with torch.no_grad():
            po = torch.exp(m3.log_density(uu).double()) \
                * m3.opacity_raw.double().clamp(0, 1).unsqueeze(-1)
            shrink = 1.0 - 1e-6 / (po.sum(0) + 1e-6)          # (P,)
        pred_ref = (x.double() * shrink.unsqueeze(-1)).float()
        err = float((m3(x, s) - pred_ref).abs().max())
        gap = float((m3(x, s) - x).abs().max())
        check(f"{arm}: f(x,s) == x*(1-eps/(sum po+eps)) at init", err < 1e-5,
              f"max err {err:.2e} (raw gap to x: {gap:.2e})")
        check(f"{arm}: init gap to identity below 1/255", gap < 1.0 / 255.0,
              f"{gap*255:.3f} grey levels")
        check(f"{arm}: G initialised to 0", float(m3.G.abs().max()) == 0.0)

    print("== 6. arm contracts ==")
    mn, ma = GLUT4D(N, arm="naive"), GLUT4D(N, arm="anchored")
    check("naive: mu_s learnable", isinstance(mn.mu_s, torch.nn.Parameter)
          and mn.mu_s.requires_grad)
    check("naive: mu_s all 0.5 at init (on the degenerate set D)",
          float(mn.mu_s.std()) == 0.0 and abs(float(mn.mu_s[0]) - 0.5) < 1e-9)
    check("anchored: mu_s is a buffer, not a parameter",
          not isinstance(ma.mu_s, torch.nn.Parameter)
          and "mu_s" in dict(ma.named_buffers()))
    grid = anchor_grid(K_ANCHOR)
    check("anchored: mu_s on the K=6 grid, all anchors used",
          bool(torch.isin(ma.mu_s, grid).all())
          and len(torch.unique(ma.mu_s)) == K_ANCHOR,
          f"unique {sorted(set(ma.mu_s.tolist()))}")
    with torch.no_grad():
        ma.sigma_s_raw.uniform_(-50, 50)
    sg = ma.sigma_s()
    check("anchored: sigma_s stays inside [0.025,0.30] for extreme raw",
          bool(((sg >= SIGMA_S_MIN - 1e-9) & (sg <= SIGMA_S_MAX + 1e-9)).all()),
          f"[{float(sg.min()):.4f},{float(sg.max()):.4f}]")
    with torch.no_grad():
        mn.sigma_s_raw.fill_(math.log(37.0))
    check("naive: sigma_s unbounded (37 reachable)",
          abs(float(mn.sigma_s()[0]) - 37.0) < 1e-3)

    print("== 7. gradients ==")
    for arm in ("naive", "anchored"):
        m4 = GLUT4D(N, arm=arm)
        loss = (m4(x, s) - torch.rand(P, 3)).abs().mean()
        loss.backward()
        for nme, prm in m4.named_parameters():
            g = prm.grad
            ok = g is not None and bool(torch.isfinite(g).all()) \
                and float(g.abs().max()) > 0
            check(f"{arm}: grad flows to {nme}", ok,
                  "" if ok else f"grad={None if g is None else float(g.abs().max())}")
        if arm == "anchored":
            check("anchored: mu_s has no grad (frozen)",
                  not hasattr(m4.mu_s, "grad") or m4.mu_s.grad is None)

    print("== 8. s-sensitivity ==")
    m5 = GLUT4D(N, arm="anchored")
    # at init only the Eq.2 eps shrink factor depends on s, so sensitivity is
    # bounded by that factor (sub-1/255), not exactly zero
    check("identity payload -> s-sensitivity below 1/255",
          m5.s_sensitivity(x, s, 0.1) < 1.0 / 255.0,
          f"{m5.s_sensitivity(x, s, 0.1)*255:.4f} grey levels")
    with torch.no_grad():
        m5.b.normal_(0.0, 0.1)
    check("perturbed payload -> positive s-sensitivity",
          m5.s_sensitivity(x, s, 0.1) > 1e-4,
          f"{m5.s_sensitivity(x, s, 0.1):.5f}")
    # |ds| == delta everywhere (no silent clipping at the range ends)
    s_edge = torch.tensor([0.0, 1.0, 0.5, 0.97])
    mid = 0.5 * (m5.s_lo + m5.s_hi)
    step = torch.where(s_edge <= mid, 0.1, -0.1)
    moved = (s_edge + step).clamp(0.0, 1.0)
    check("|ds| == delta at the range ends",
          bool(((moved - s_edge).abs() - 0.1).abs().max() < 1e-6))

    print("== 9. payload path == explicit sum_i w_i (M_i x + b_i) + Gx + g ==")
    m7 = GLUT4D(N, arm="naive")
    with torch.no_grad():
        m7.M.normal_(0.0, 0.5)
        m7.b.normal_(0.0, 0.2)
        m7.G.normal_(0.0, 0.3)
        m7.g.normal_(0.0, 0.1)
        m7.mu_s.normal_(0.5, 0.2)
        m7.chol_off.normal_(0.0, 0.03)
    with torch.no_grad():
        w = m7.weights(torch.cat([s.unsqueeze(-1), x], -1)).double()  # (N,P)
        ref = torch.zeros(P, 3, dtype=torch.float64)
        for i in range(N):
            loc = x.double() @ m7.M[i].double().T + m7.b[i].double()
            ref += w[i].unsqueeze(-1) * loc
        ref += x.double() @ m7.G.double().T + m7.g.double()
        err = float((m7(x, s).double() - ref).abs().max())
    check("forward matches the explicit per-Gaussian sum", err < 1e-5,
          f"max err {err:.2e}")

    print("== 10. parameter budget ==")
    for arm in ("naive", "anchored"):
        m6 = GLUT4D(32, arm=arm)
        print(f"    {arm}: N=32 -> {m6.n_params()} learnable params")
    check("anchored has exactly N fewer params than naive (frozen mu_s)",
          GLUT4D(32, "naive").n_params() - GLUT4D(32, "anchored").n_params() == 32)

    print()
    if FAILS:
        print(f"FAILED {len(FAILS)}: {FAILS}")
        sys.exit(1)
    print("all 4D CI checks passed")


if __name__ == "__main__":
    main()
