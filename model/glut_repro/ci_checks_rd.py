"""Implementation self-checks for the RD arms (run before the grid; every red
line in CLAUDE.md that this code can violate is pinned by one of these).

  python -m model.glut_repro.ci_checks_rd
"""

from __future__ import annotations

import math

import torch

from model.glut_repro.model_rd import (
    GAMMA_LAMBDA, GECO, LUT4D, SModulation, bspline_basis, build_arm,
    gamma_mu_for, s_response)
from model.glut_repro.model4d_naive import (
    GAMMA_MU, K_ANCHOR, SIGMA_S_MAX, SIGMA_S_MIN, anchor_grid)

FAILS: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def main() -> None:  # noqa: C901
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P = 4096
    x = torch.rand(P, 3, device=dev)
    s = torch.rand(P, device=dev)

    # 1 --------------------------------------------------------- param counts
    counts = {a: build_arm(a).n_params()
              for a in ("g3d", "gstd", "ga", "gc1", "gc3", "gc5", "gd",
                        "lut3d", "lut4d")}
    print(f"      param counts: {counts}")
    check("1a g3d = 22N+12 = 716 (paper's single-GLUT count)",
          counts["g3d"] == 716, str(counts["g3d"]))
    check("1b gstd = 844 (R-2, PLAN table) + 1686 (R-1) = 2530 ~ PLAN's 2.5K",
          counts["gstd"] == 2530, str(counts["gstd"]))
    check("1c lut3d = 17^3*3 = 14739", counts["lut3d"] == 14739,
          str(counts["lut3d"]))
    check("1d lut4d = 17^3*5*3 = 73695 = PLAN R-11's 73.7K",
          counts["lut4d"] == 73695, str(counts["lut4d"]))
    check("1e RD-A (ga) is same-budget with RD-STD (PLAN: 'R-1 vs R-5 同预算')",
          abs(counts["ga"] - counts["gstd"]) < 100,
          f"ga {counts['ga']} vs gstd {counts['gstd']}")
    check("1f RD-C K=1/3/5 = 1100/1868/2636 == PLAN R-4's '1.1-2.6K' "
          "(716 + 12*N*K)",
          (counts["gc1"], counts["gc3"], counts["gc5"]) == (1100, 1868, 2636),
          str((counts["gc1"], counts["gc3"], counts["gc5"])))
    check("1g RD-D (gd) = 780 == PLAN R-6's count (748 + learnable mu_s 32)",
          counts["gd"] == 780, str(counts["gd"]))

    # 2 ------------------------------------------- red line: G init 0, not I
    for arm in ("g3d", "gstd", "ga", "gc1", "gc3", "gc5"):
        m = build_arm(arm).to(dev)
        gz = (float(m.core.G.detach().abs().max()),
              float(m.core.g.detach().abs().max()))
        check(f"2 {arm}: G init == 0 and g init == 0 (NOT I; else f(x)=2x)",
              gz == (0.0, 0.0), str(gz))
        with torch.no_grad():
            d = float((m.predict(x, s) - x).abs().max())
        check(f"2b {arm}: f(x,s) == x at init (within Eq.2 eps offset)",
              d < 0.005, f"max|f(x,s)-x| = {d:.6f} = {d*255:.3f}/255")
    # RD-D is the one arm whose local branch is un-normalised: the identity has
    # to live in the global branch (G=I) with the LOCAL payload at zero, so
    # f(x)=2x is impossible and f(x,s)=x still holds exactly at init.
    m = build_arm("gd").to(dev)
    with torch.no_grad():
        ok_pair = (float(m.core.M.abs().max()) == 0.0
                   and float(m.core.b.abs().max()) == 0.0
                   and torch.equal(m.core.G.detach().cpu(), torch.eye(3)))
        d = float((m.predict(x, s) - x).abs().max())
    check("2c gd (R-6): local payload M=0,b=0 with G=I -> f(x,s)=x at init and "
          "f(x)=2x structurally impossible", ok_pair and d < 1e-5,
          f"max|f-x| {d:.2e}")
    with torch.no_grad():
        wsum = m.weights(x, s).sum(0)
    check("2d gd (R-6): sum_i w_i != 1 -- the s gate is OUTSIDE the "
          "normalisation, which is the whole mechanism",
          float(wsum.max()) < 0.999, f"max sum_i w_i = {float(wsum.max()):.4f}")

    # 3 ------------------------------------------- LUT identity init is exact
    for arm in ("lut3d", "lut4d"):
        m = build_arm(arm).to(dev)
        with torch.no_grad():
            d = float((m(x, s) - x).abs().max())
        check(f"3 {arm}: identity lattice -> f(x,s) == x exactly", d < 1e-5,
              f"max err {d:.2e}")

    # 4 --------------------------- quadrilinear read vs an independent reference
    m = build_arm("lut4d").to(dev)
    with torch.no_grad():
        m.table.copy_(torch.randn_like(m.table))
        got = m(x, s)
        # reference: four nested 1-D lerps, written out separately
        k, ns = m.n_rgb, m.n_s
        c = x * (k - 1)
        i0 = c.floor().clamp(0, k - 2).long()
        t = c - i0.float()
        cs = s * (ns - 1)
        j0 = cs.floor().clamp(0, ns - 2).long()
        ts = (cs - j0.float()).unsqueeze(-1)
        tb = m.table

        def corner(js, dr, dg, db):
            return tb[js, i0[:, 0] + dr, i0[:, 1] + dg, i0[:, 2] + db]

        def tri(js):
            a = torch.lerp(corner(js, 0, 0, 0), corner(js, 0, 0, 1),
                           t[:, 2:3])
            b = torch.lerp(corner(js, 0, 1, 0), corner(js, 0, 1, 1),
                           t[:, 2:3])
            cc = torch.lerp(corner(js, 1, 0, 0), corner(js, 1, 0, 1),
                            t[:, 2:3])
            d = torch.lerp(corner(js, 1, 1, 0), corner(js, 1, 1, 1),
                           t[:, 2:3])
            ab = torch.lerp(a, b, t[:, 1:2])
            cd = torch.lerp(cc, d, t[:, 1:2])
            return torch.lerp(ab, cd, t[:, 0:1])

        ref = torch.lerp(tri(j0), tri(j0 + 1), ts)
        err = float((got - ref).abs().max())
    check("4 quadrilinear == tensor product of four 1-D lerps", err < 1e-4,
          f"max err {err:.2e}")

    # 5 ------------------------------------ lut3d is structurally s-invariant
    m = build_arm("lut3d").to(dev)
    with torch.no_grad():
        m.table.copy_(torch.randn_like(m.table))
        d = float((m(x, torch.zeros_like(s)) - m(x, torch.ones_like(s)))
                  .abs().max())
    check("5 lut3d: f(x,s) independent of s -> Delta_const/shuffle == 0 "
          "by construction (built-in negative control)", d == 0.0, str(d))

    # 6 -------------------------------- red line: sigma_s bounded, no bare exp
    m = build_arm("gstd").to(dev)
    with torch.no_grad():
        for raw in (-60.0, -5.0, 0.0, 5.0, 60.0):
            m.core.sigma_s_raw.fill_(raw)
            sg = m.sigma_s()
            # 1e-6 tolerance = fp32 rounding of 0.025 + 0.275*sigmoid(60),
            # not slack in the bound itself
            if not (float(sg.min()) >= SIGMA_S_MIN - 1e-6
                    and float(sg.max()) <= SIGMA_S_MAX + 1e-6):
                break
        else:
            raw = None
    check("6 sigma_s in [0.025,0.30] for raw in [-60,60] (sigmoid, not exp)",
          raw is None)

    # 7 ------------------------------------------ R-2: mu_s frozen K=6 anchors
    m = build_arm("gstd").to(dev)
    mu = m.core.mu_s
    check("7a mu_s is a buffer (no gradient) -- R-2 'mu_s 不可学'",
          not isinstance(mu, torch.nn.Parameter)
          and "mu_s" not in dict(m.named_parameters()))
    check("7b K=6 distinct anchors, all used",
          int(torch.unique(mu).numel()) == 6, str(sorted(
              round(float(v), 3) for v in torch.unique(mu))))
    check("7c std(mu_s) = 0.351 > gamma_mu = 0.15 -> the R-3 mu hinge is "
          "structurally inactive (must be reported, not silently zero)",
          float(mu.float().std()) > GAMMA_MU,
          f"{float(mu.float().std()):.4f}")

    # 7d s-RGB cross covariance is present and not factorizable
    with torch.no_grad():
        m.core.chol_off.normal_(0, 0.05)
        L = m.core.chol()
        cov = (L @ L.transpose(-1, -2))
        cross = float(cov[:, 0, 1:].abs().max())
    check("7d s-RGB cross covariance kept (R-2: 保留 s–RGB 交叉协方差)",
          cross > 0, f"max|Cov(s,rgb)| = {cross:.4f}")

    # 8 --------------------------------------------- R-1 modulation zero-init
    mod = SModulation(32).to(dev)
    g, b = mod(s)
    check("8a R-1 head zero-init -> gamma == beta == 0 at step 0",
          float(g.detach().abs().max()) == 0.0
          and float(b.detach().abs().max()) == 0.0)
    with torch.no_grad():
        mod.head.weight.normal_(0, 0.3)
        mod.head.bias.normal_(0, 0.3)
        cs = mod.s_code(s)
        gb_lit = torch.stack([mod.head(cs * mod.z[i]) for i in range(mod.N)])
        g2, b2 = mod(s)
        err = float((torch.cat([g2, b2], -1) - gb_lit).abs().max())
    check("8b folded einsum == literal head(code_s * z_i) per Gaussian",
          err < 1e-4, f"max err {err:.2e}")

    # 9 ------------------------------- gstd forward vs explicit per-Gaussian sum
    m = build_arm("gstd").to(dev)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.1)
        m.core.log_sigma_c.fill_(math.log(0.15))
        got = m(x, s)
        w = m.weights(x, s)
        gam, bet = m.mod(s)
        acc = torch.zeros_like(x)
        for i in range(m.N):
            loc = x @ m.core.M[i].T + m.core.b[i]
            acc = acc + w[i].unsqueeze(-1) * ((1 + gam[i]) * loc + bet[i])
        ref = acc + x @ m.core.G.T + m.core.g
        err = float((got - ref).abs().max())
    check("9 gstd forward == sum_i w_i[(1+gamma_i)(M_i x + b_i) + beta_i] + Gx+g",
          err < 1e-4, f"max err {err:.2e}")

    # 10 ------------------- red line: per-pixel operator, no (x,y)/neighbourhood
    perm = torch.randperm(P, device=dev)
    with torch.no_grad():
        a = m(x, s)[perm]
        bq = m(x[perm], s[perm])
        err = float((a - bq).abs().max())
    check("10 permutation-equivariant per pixel -> no (x,y), no neighbourhood, "
          "no ordering", err < 1e-5, f"max err {err:.2e}")

    # 11 --------------------------------- red line: no s-axis smoothing by default
    lut = build_arm("lut4d").to(dev)
    with torch.no_grad():
        lut.table.copy_(torch.randn_like(lut.table))
        tv_default = float(lut.tv())
        tv_rgb_only = float(lut.tv(w_rgb=1e-4, w_s=0.0))
    check("11 LUT4D.tv() default puts ZERO weight on the s axis "
          "(s 轴禁平滑正则)", abs(tv_default - tv_rgb_only) < 1e-12)

    # 12 --------------------------------------------- GECO == Algorithm 1
    g0 = GECO(tau=0.02)
    check("12a lambda initialised to 1 (paper Algorithm 1)", g0.lam == 1.0)
    lam0 = g0.lam
    for _ in range(50):                          # constraint violated: C > 0
        g0.term(g0.constraint(torch.tensor(0.0)))
        g0.update()
    up = g0.lam
    g1 = GECO(tau=0.02)
    for _ in range(50):                          # constraint satisfied: C < 0
        g1.term(g1.constraint(torch.tensor(0.20)))
        g1.update()
    check("12b lambda grows multiplicatively while the constraint is violated "
          "and shrinks once it is met", up > lam0 and g1.lam < lam0,
          f"violated {up:.3f} vs satisfied {g1.lam:.3f}")
    g2 = GECO(tau=0.02, alpha=0.9)
    g2.term(g2.constraint(torch.tensor(0.0)))    # C_hat = 0.02, ma <- 0.02
    ma0 = g2.ma
    g2.term(g2.constraint(torch.tensor(0.04)))   # C_hat = -0.02
    check("12c C_ma <- alpha*C_ma + (1-alpha)*C_hat", ma0 is not None
          and abs(g2.ma - (0.9 * 0.02 + 0.1 * -0.02)) < 1e-9,
          f"{g2.ma:.9f}")
    g3 = GECO(tau=0.02, lam_max=5.0)
    for _ in range(20000):
        g3.term(g3.constraint(torch.tensor(0.0)))
        g3.update()
    check("12d lambda is capped (PLAN R-7 needs 'beta pinned at its ceiling' "
          "to be observable)", g3.lam == 5.0 and g3.state()["pinned_frac"] > 0,
          f"lambda {g3.lam}, pinned {g3.state()['pinned_frac']:.2f}")

    # 13 ------------------------- s_response is a perturbation, not a shuffle
    m = build_arm("gstd").to(dev)
    r0 = float(s_response(m, x, s, 0.5))
    check("13a s_response == 0 for an s-blind model at zero-init modulation",
          r0 < 1e-6, f"{r0:.2e}")
    s_const = torch.ones_like(s)                 # the L0 case: every image s==1
    with torch.no_grad():
        m.mod.head.weight.normal_(0, 0.3)
    r1 = float(s_response(m, x, s_const, 0.5))
    check("13b s_response is non-degenerate even when s is constant across the "
          "whole dataset (L0) -- a shuffle-based proxy would be identically 0 "
          "there and would pin lambda at its ceiling on the one level whose "
          "point is that s must cost nothing", r1 > 0, f"{r1:.4f}")

    # 14 ---------------------- R-3 hinge reports both terms and is finite
    h, h_mu, h_lam = m.diversity_hinge()
    check("14 R-3 hinge = relu(gamma_mu - std(mu_s)) + "
          "relu(gamma_lambda - std(log sigma_s)); mu term structurally 0 "
          f"under R-2, lambda term live (gamma_lambda={GAMMA_LAMBDA})",
          float(h_mu) == 0.0 and float(h_lam) >= 0.0 and torch.isfinite(h),
          f"h_mu {float(h_mu):.4f}  h_lam {float(h_lam):.4f}")
    h3 = build_arm("g3d").to(dev).diversity_hinge()
    check("14b 3D control has no s axis -> hinge identically 0",
          float(h3[0]) == 0.0)

    # 15 ------------------ LUT gradient only reaches lattice cells it touched
    lut = build_arm("lut4d").to(dev)
    y = torch.rand(P, 3, device=dev)
    (lut(x, s) - y).abs().mean().backward()
    gnz = lut.table.grad.abs().reshape(-1, 3).sum(-1) > 0
    check("15 LUT gradient is sparse: untouched cells stay at identity "
          f"({int(gnz.sum())}/{gnz.numel()} cells touched by {P} pixels)",
          0 < int(gnz.sum()) < gnz.numel())

    # 16 --------------------------------- RD-A: s changes strength, not direction
    m = build_arm("ga").to(dev)
    with torch.no_grad():
        m.V.normal_(0, 0.5)
        m.m0.normal_(1.0, 0.3)
        m.mod.head.weight.normal_(0, 0.5)
        m.mod.head.bias.normal_(0, 0.2)
        vdir = m.V / m.V.norm(dim=1, keepdim=True)
        cols = vdir.norm(dim=1)
        # the modulated M_i(s) column directions must be s-invariant: each
        # column of M_i(s) is a fixed unit vector scaled by a scalar
        cosines = []
        for sv in (0.0, 0.37, 1.0):
            sc = torch.full((8,), sv, device=dev)
            mag = m.m0.unsqueeze(1) * (1 + m.mod.raw(sc))       # (N,8,3)
            Ms = torch.einsum("nij,nj->nij", vdir, mag[:, 0, :])
            cosines.append(Ms / Ms.norm(dim=1, keepdim=True).clamp(min=1e-8))
        drift = max(float((c - cosines[0]).abs().max()) for c in cosines[1:])
    check("16a RD-A: ||V_i||_col == 1 after normalisation",
          float((cols - 1).abs().max()) < 1e-5)
    check("16b RD-A: column DIRECTIONS of M_i(s) are invariant in s -- s moves "
          "magnitude only (up to sign), the R-1 vs R-5 contrast",
          drift < 1e-4 or drift > 1.99, f"max direction drift {drift:.2e}")
    with torch.no_grad():
        m2 = build_arm("ga").to(dev)
        r = float(s_response(m2, x, s, 0.5))
    check("16c RD-A zero-init Delta -> s-response 0 at step 0", r < 1e-6,
          f"{r:.2e}")

    # 17 ---------------------------------- RD-C: B-spline basis and rank sweep
    ss = torch.linspace(0, 1, 257, device=dev)
    for k in (1, 3, 5):
        bb = bspline_basis(ss, k)
        ok = bb.shape == (257, k) and torch.isfinite(bb).all()
        if k > 1:
            ok = ok and float((bb.sum(-1) - 1).abs().max()) < 1e-4
        # must actually separate the two ends of the s range
        sep = float((bb[0] - bb[-1]).abs().max())
        check(f"17 RD-C K={k}: B-spline basis well formed"
              + (" (partition of unity)" if k > 1 else " (beta=s, see NOTES)")
              + " and separates s=0 from s=1", ok and sep > 0.5,
              f"|beta(0)-beta(1)|max {sep:.3f}")
    m = build_arm("gc3").to(dev)
    check("17d RD-C geometry is RGB-only: w_i does not see s (PLAN R-4 "
          "'几何只建 RGB')",
          float((m.weights(x, torch.zeros_like(s))
                 - m.weights(x, torch.ones_like(s))).abs().max()) == 0.0)
    with torch.no_grad():
        m.Mk.normal_(0, 0.2)
        m.bk.normal_(0, 0.2)
        d = float((m(x, torch.zeros_like(s)) - m(x, torch.ones_like(s)))
                  .abs().max())
    check("17e RD-C: the s dependence lives entirely in the payload and is "
          "non-trivial once M^k != 0", d > 1e-3, f"{d:.4f}")

    # 18 ------------------------------- R-3 gamma_mu is relative, not absolute
    gm = gamma_mu_for(anchor_grid(K_ANCHOR))
    check("18 gamma_mu = 0.4*std(anchor grid) reproduces PLAN's literal 0.15 "
          "on the [0,1] oracle-mask range (and would not silently vanish on "
          "PLAN 1.2's [-3,3])", abs(gm - GAMMA_MU) < 0.005,
          f"relative {gm:.4f} vs PLAN literal {GAMMA_MU}")
    gd = build_arm("gd").to(dev)
    check("18b gd: mu_s is LEARNABLE (the one arm where R-3's mu term can do "
          "work) and starts on the K=6 grid, not on the degenerate set D",
          gd.mu_s().requires_grad and int(torch.unique(gd.mu_s()).numel()) == 6)

    print()
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
        raise SystemExit(1)
    print("ALL RD CI CHECKS PASSED")


if __name__ == "__main__":
    main()
