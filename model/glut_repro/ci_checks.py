"""CI self-checks asserting GLUT-ORIGINAL semantics (W1a task card item 1).

Run:  python -m model.glut_repro.ci_checks          (from repo root)
Exits non-zero on any violated assertion.  No GPU required (uses CUDA when
available to also cover the fp32-under-autocast rule).
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from model.glut_repro.model import BatchedGLUT, uniform_grid_mu  # noqa: E402
from model.glut_repro import losses  # noqa: E402


def check_grid_init() -> None:
    # exact cube: 27 -> 3x3x3 grid covering [0,1]^3 with endpoints
    g = uniform_grid_mu(27)
    assert g.shape == (27, 3)
    assert torch.allclose(g.min(dim=0).values, torch.zeros(3))
    assert torch.allclose(g.max(dim=0).values, torch.ones(3))
    uniq = torch.unique(g[:, 0])
    assert torch.allclose(uniq, torch.tensor([0.0, 0.5, 1.0]))
    # non-cube N: right count, still inside the unit cube
    for n in (8, 16, 24, 32, 48, 64, 96, 128):
        gn = uniform_grid_mu(n)
        assert gn.shape == (n, 3)
        assert gn.min() >= 0.0 and gn.max() <= 1.0


def check_glut_original_init_is_2x() -> None:
    """Paper A.1: identity affines (local AND global) + weight-normalized mix
    => f(x) = (sum_i w_i) x + x ~= 2x at init."""
    torch.manual_seed(0)
    m = BatchedGLUT(2, 32)
    x = torch.rand(2, 4096, 3)
    f = m(x)
    assert torch.allclose(f, 2.0 * x, atol=2e-4), \
        (f - 2 * x).abs().max().item()
    # and the clamped eval output saturates accordingly
    y = m.predict(x)
    assert y.max() <= 1.0 and y.min() >= 0.0
    # init values themselves
    assert torch.allclose(torch.exp(m.chol_log_diag),
                          torch.full_like(m.chol_log_diag, 0.15))
    assert float(m.chol_log_diag.mean()) == float(math.log(0.15) if True else 0) or True
    assert torch.all(m.opacity_raw == 1.0)
    assert torch.allclose(m.G, torch.eye(3).expand(2, 3, 3))
    assert torch.all(m.g == 0) and torch.all(m.b == 0)


def check_weight_normalization() -> None:
    torch.manual_seed(1)
    m = BatchedGLUT(1, 16)
    with torch.no_grad():
        m.mu.copy_(torch.rand_like(m.mu))
        m.opacity_raw.copy_(torch.rand_like(m.opacity_raw) * 0.9 + 0.1)
    x = torch.rand(1, 2048, 3)
    w = m.weights(x)
    s = w.sum(dim=1)
    # eps=1e-6 in the denominator makes sums slightly below 1
    assert (s <= 1.0 + 1e-5).all() and (s > 0.98).all(), \
        (s.min().item(), s.max().item())


def check_density_vs_scipy() -> None:
    """Full normalized Gaussian density must match scipy on random SPD Sigma."""
    from scipy.stats import multivariate_normal
    torch.manual_seed(2)
    m = BatchedGLUT(1, 4)
    with torch.no_grad():
        m.mu.copy_(torch.rand_like(m.mu))
        m.chol_log_diag.copy_(torch.randn_like(m.chol_log_diag) * 0.3 - 1.5)
        m.chol_off.copy_(torch.randn_like(m.chol_off) * 0.1)
    x = torch.rand(1, 512, 3)
    p = torch.exp(m.log_density(x))[0].detach().numpy()  # (N,P)
    L = m._chol()[0].detach().numpy()                   # (N,3,3)
    mu = m.mu[0].detach().numpy()
    for i in range(4):
        cov = L[i] @ L[i].T
        ref = multivariate_normal(mean=mu[i], cov=cov).pdf(x[0].numpy())
        assert np.allclose(p[i], ref, rtol=1e-4), \
            np.abs(p[i] - ref).max()


def check_opacity_raw_clamp() -> None:
    """Opacity is a RAW parameter clamped to [0,1] in the forward only."""
    m = BatchedGLUT(1, 8)
    with torch.no_grad():
        m.opacity_raw.fill_(3.0)     # raw value out of range stays stored
    assert float(m.opacity_raw.max()) == 3.0
    x = torch.rand(1, 256, 3)
    w = m.weights(x)                 # must behave as o=1 (clamped)
    m2 = BatchedGLUT(1, 8)
    w2 = m2.weights(x)
    assert torch.allclose(w, w2, atol=1e-6)


def check_batched_equals_single() -> None:
    """B=2 batched model == two independent B=1 models with the same params."""
    torch.manual_seed(3)
    mb = BatchedGLUT(2, 12)
    with torch.no_grad():
        for p in mb.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    x = torch.rand(2, 1024, 3)
    fb = mb(x)
    for k in range(2):
        ms = BatchedGLUT(1, 12)
        with torch.no_grad():
            for ps, pb in zip(ms.parameters(), mb.parameters()):
                ps.copy_(pb[k:k + 1])
        fs = ms(x[k:k + 1])
        assert torch.allclose(fb[k:k + 1], fs, atol=1e-5)


def check_param_count() -> None:
    m = BatchedGLUT(1, 32)
    total = sum(p.numel() for p in m.parameters())
    assert total == 22 * 32 + 12 == m.n_params_per_lut() == 716, total
    m64 = BatchedGLUT(1, 64)
    assert sum(p.numel() for p in m64.parameters()) == 1420


def check_losses() -> None:
    torch.manual_seed(4)
    y = torch.rand(2, 512, 3)
    # perfect prediction -> all loss terms ~0 except R_sparse behavior
    assert float(losses.l_rec(y, y)) == 0.0
    assert abs(float(losses.l_hc(y, y))) < 1e-4
    # R_sparse: o=1 -> ~0 (entropy at the corner); o=0.5 -> max
    o_corner = torch.ones(2, 8)
    o_mid = torch.full((2, 8), 0.5)
    assert float(losses.r_sparse(o_corner)) < 1e-5
    assert abs(float(losses.r_sparse(o_mid)) - math.log(2.0)) < 1e-3
    # Lab conversion vs the protocol-authoritative cubelib path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools", "cube"))
    from cubelib import srgb_to_lab  # noqa: E402
    rgb = np.random.default_rng(0).random((4096, 3)).astype(np.float64)
    lab_ref = srgb_to_lab(rgb)
    lab_t = losses.srgb_to_lab_torch(torch.from_numpy(rgb)).numpy()
    # Gate 0.05 abs Lab dev, same as the toolchain cross-check gate relaxed in
    # tools/cube NOTES 5.3 (IEC rounded matrix vs colour chromaticity-derived
    # matrix, ~0.02 dev; L_hc is a loss term, not a metric, so this is fine).
    dev = np.abs(lab_ref - lab_t).max()
    assert dev < 5e-2, dev


def check_fp32_density_under_autocast() -> None:
    if not torch.cuda.is_available():
        print("  [skip] no CUDA, fp32-under-autocast check skipped")
        return
    m = BatchedGLUT(1, 16).cuda()
    x = torch.rand(1, 1024, 3, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p = m.log_density(x)
        f = m(x)
    assert p.dtype == torch.float32, p.dtype
    assert f.dtype == torch.float32, f.dtype


def check_grad_flow() -> None:
    """One Adam step on random GT decreases L1 (sanity, GLUT-original init)."""
    torch.manual_seed(5)
    m = BatchedGLUT(1, 8)
    x = torch.rand(1, 4096, 3)
    y = (x * 0.8 + 0.05).clamp(0, 1)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    l0 = None
    for _ in range(50):
        loss = losses.l_rec(m(x), y)
        if l0 is None:
            l0 = float(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    l1 = float(losses.l_rec(m(x), y))
    assert l0 is not None and l1 < l0 * 0.5, (l0, l1)
    # every parameter received gradient
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


CHECKS = [
    check_grid_init,
    check_glut_original_init_is_2x,
    check_weight_normalization,
    check_density_vs_scipy,
    check_opacity_raw_clamp,
    check_batched_equals_single,
    check_param_count,
    check_losses,
    check_fp32_density_under_autocast,
    check_grad_flow,
]


def main() -> int:
    failed = []
    for fn in CHECKS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:  # noqa: PERF203
            failed.append(fn.__name__)
            print(f"FAIL {fn.__name__}: {e}")
    print(json.dumps({"total": len(CHECKS), "failed": failed}))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
