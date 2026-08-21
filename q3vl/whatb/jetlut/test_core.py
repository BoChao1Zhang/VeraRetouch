"""EPR-032 §8.1 runtime assertions, as a runnable check.

    .venv/bin/python -m q3vl.whatb.jetlut.test_core
"""

from __future__ import annotations

import torch

from q3vl.whatb.jetlut.core import (
    Atlas, admm_lad, design_matrix, fill_distance, n_dynamic_params, pou_weights,
)


def _grid(n: int) -> torch.Tensor:
    ax = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    r, g, b = torch.meshgrid(ax, ax, ax, indexing="ij")
    return torch.stack((r.reshape(-1), g.reshape(-1), b.reshape(-1)), dim=-1)


def _fit(f: torch.Tensor, r: torch.Tensor) -> tuple[torch.Tensor, dict]:
    out = admm_lad(f, r, max_iter=4000)
    return f @ out["theta"] - r, out


def test_param_counts() -> None:
    assert n_dynamic_params(3, 2) == 822, n_dynamic_params(3, 2)
    assert n_dynamic_params(4, 2) == 1932
    assert n_dynamic_params(5, 2) == 3762
    assert n_dynamic_params(6, 2) == 6492
    assert n_dynamic_params(3, 1) == 336
    assert n_dynamic_params(4, 1) == 780
    assert n_dynamic_params(7, 1) == 4128
    assert abs(fill_distance(4) - 0.21650635) < 1e-7


def test_pou_sums_to_one() -> None:
    """§8.1 A-POU."""
    a = Atlas(m=4, c=1.0)
    x = _grid(9)
    for c in (0.4, 1.0, 2.0):
        pi = pou_weights(x, Atlas(m=4, c=c).centres(dtype=x.dtype), Atlas(m=4, c=c).sigma)
        err = (pi.sum(-1) - 1.0).abs().max().item()
        assert err < 1e-6, f"POU violated at c={c}: {err}"
    assert a.n == 64


def test_affine_and_quadratic_reproduction() -> None:
    """p=1 reproduces a global affine target; p=2 a global quadratic one.

    The global branch alone already spans affine maps, so p=1 must hit it to
    solver precision.  The quadratic target is *not* in the p=1 span, so p=2
    must be strictly better -- that separation is the whole premise of the EPR.
    """
    x = _grid(13)
    a = Atlas(m=3, c=1.0)

    mat = torch.tensor([[0.9, 0.05, 0.0], [0.0, 1.1, 0.02], [0.03, 0.0, 0.95]],
                       dtype=torch.float64)
    r_affine = (x @ mat.T + torch.tensor([0.02, -0.01, 0.03], dtype=torch.float64)) - x

    f1 = design_matrix(x, a, 1)
    res, _ = _fit(f1, r_affine)
    assert res.abs().max().item() < 1e-6, res.abs().max().item()

    r_quad = torch.stack([x[:, 0] ** 2, x[:, 1] * x[:, 2], x[:, 2] ** 2], -1) * 0.3
    e1 = _fit(f1, r_quad)[0].abs().mean().item()
    f2 = design_matrix(x, a, 2)
    e2 = _fit(f2, r_quad)[0].abs().mean().item()
    assert e2 < 1e-6, f"p=2 should reproduce a global quadratic exactly: {e2}"
    assert e1 > 100 * max(e2, 1e-12), f"p1 {e1} vs p2 {e2}"


def test_nested_monotonicity() -> None:
    """§8.1 A-nested (G-wire): p=1 is a strict subspace of p=2, so on the same
    fit set and the same atlas the L1 optimum can only go down."""
    torch.manual_seed(0)
    x = _grid(13)
    a = Atlas(m=3, c=1.0)
    # a deliberately awkward target: hue bending + highlight compression
    r = torch.stack([
        0.25 * torch.sin(4.0 * x[:, 1]) * x[:, 0],
        0.20 * (1.0 - torch.exp(-3.0 * x[:, 2])) - 0.1 * x[:, 1] ** 3,
        0.15 * x[:, 0] * x[:, 2] - 0.05 * x[:, 2],
    ], dim=-1)
    e1 = _fit(design_matrix(x, a, 1), r)[0].abs().sum().item()
    e2 = _fit(design_matrix(x, a, 2), r)[0].abs().sum().item()
    assert e2 <= e1 * (1.0 + 1e-6), f"nested violated: p1 {e1} < p2 {e2}"


def test_lad_beats_least_squares_in_l1() -> None:
    """Sanity on the solver itself: the LAD optimum must have a lower L1 than
    the ridge/L2 solution on the same design."""
    torch.manual_seed(1)
    x = _grid(11)
    a = Atlas(m=3, c=1.0)
    f = design_matrix(x, a, 1)
    r = torch.randn(x.shape[0], 3, dtype=torch.float64) * 0.05
    r[:20] += 2.0                                    # outliers: where L1 wins
    lad = _fit(f, r)[0].abs().sum().item()
    ls = torch.linalg.lstsq(f, r).solution
    l2 = (f @ ls - r).abs().sum().item()
    assert lad < l2, f"LAD {lad} !< LS {l2}"


def test_dual_certificate() -> None:
    """The reported gap is a certificate, not a residual: check it is small and
    the dual point is (numerically) in null(f^T)."""
    x = _grid(11)
    f = design_matrix(x, Atlas(m=3, c=1.0), 1)
    r = torch.stack([x[:, 0] ** 2, x[:, 1] ** 2, x[:, 2] ** 2], -1) * 0.2
    out = admm_lad(f, r, max_iter=4000)
    assert float(out["gap"].max()) < 1e-3, float(out["gap"].max())
    assert out["dual_feas"] < 1e-6, out["dual_feas"]
    assert bool((out["dual"] <= out["primal"] + 1e-9).all())


def main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all JetLUT core checks passed")


if __name__ == "__main__":
    main()
