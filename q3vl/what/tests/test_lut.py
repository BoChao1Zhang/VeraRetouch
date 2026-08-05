"""``T_gt`` parsing and the two interpolators, pinned against the repo's own.

The interesting assertions are the cross-implementation ones: the trilinear path
must agree with ``dataset_build``'s CPU oracle (the arithmetic that produced
``I_tar``) and the tetrahedral path with ``model/glut_repro``'s CI-pinned
implementation, which was itself pinned against colour-science.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from q3vl.what.lut import (
    GtLutTable,
    LutBank,
    lattice_points,
    load_gt_table,
    tetra_lookup,
    trilinear_lookup,
)


def write_cube(path: Path, size: int, fn) -> Path:
    """A real ``.cube`` file, in the format's own order (R fastest)."""
    lines = ["# test", f"LUT_3D_SIZE {size}"]
    for b in range(size):
        for g in range(size):
            for r in range(size):
                v = fn(r / (size - 1), g / (size - 1), b / (size - 1))
                lines.append(" ".join(f"{x:.6f}" for x in v))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_cube_round_trip_axis_order(tmp_path):
    """``table[r, g, b]`` really indexes R, G, B -- the one thing a transpose bug
    would silently invert, and which the 2026-07-17 ``f(B,G,R)`` bug in
    ``render_backend`` shows is not hypothetical."""
    size = 5
    p = write_cube(tmp_path / "t.cube", size, lambda r, g, b: (r, 0.5 * g, 0.25 * b))
    t = load_gt_table(p, "t")
    assert t.size == size
    for (ri, gi, bi) in [(0, 0, 0), (4, 0, 0), (0, 4, 0), (0, 0, 4), (2, 3, 1)]:
        got = t.table[ri, gi, bi]
        want = torch.tensor([ri / 4, 0.5 * gi / 4, 0.25 * bi / 4])
        assert torch.allclose(got, want, atol=1e-5), (ri, gi, bi, got, want)


def test_identity_cube_is_the_identity_function(tmp_path):
    p = write_cube(tmp_path / "id.cube", 9, lambda r, g, b: (r, g, b))
    t = load_gt_table(p, "id")
    x = torch.rand(256, 3)
    for mode in ("trilinear", "tetrahedral"):
        assert float((t.apply(x, mode) - x).abs().max()) < 1e-5


def test_trilinear_matches_dataset_build_cpu_oracle():
    # ``dataset_build.src.construct.rendering`` pulls in sqlite3, which does not
    # import under every conda env on this box.  The *production* path only needs
    # ``dataset_build.lut_io`` (numpy only), so an env where this cross-check
    # cannot run is a skip, not a failure.
    try:
        from dataset_build.src.construct.rendering import apply_lut_cpu_oracle
    except ImportError as exc:                                  # pragma: no cover
        pytest.skip(f"dataset_build renderer not importable here: {exc}")

    torch.manual_seed(0)
    size = 9
    table = torch.rand(size, size, size, 3)                    # [r, g, b]
    x = torch.rand(64, 64, 3)
    ours = trilinear_lookup(table.unsqueeze(0), x.reshape(1, -1, 3)).reshape(64, 64, 3)
    grid_bgr = np.ascontiguousarray(table.numpy().transpose(2, 1, 0, 3))
    theirs = apply_lut_cpu_oracle(x.numpy(), grid_bgr)
    assert np.abs(ours.numpy() - theirs).max() < 1e-5


def test_tetrahedral_matches_glut_repro():
    from model.glut_repro.model_rdg import tetra_lookup as ref

    torch.manual_seed(1)
    table = torch.rand(2, 11, 11, 11, 3)
    x = torch.rand(2, 512, 3)
    assert torch.allclose(tetra_lookup(table, x), ref(table, x), atol=1e-6)


def test_tetrahedral_matches_colour_science():
    colour = pytest.importorskip("colour")
    from colour.algebra import table_interpolation_tetrahedral

    torch.manual_seed(2)
    size = 9
    table = torch.rand(size, size, size, 3).double()
    x = torch.rand(256, 3).double()
    ours = tetra_lookup(table.unsqueeze(0), x.unsqueeze(0))[0].numpy()
    lut = colour.LUT3D(table.numpy())
    theirs = lut.apply(x.numpy(), interpolator=table_interpolation_tetrahedral)
    assert np.abs(ours - theirs).max() < 1e-9


def test_both_interpolators_are_exact_on_lattice_points():
    torch.manual_seed(3)
    size = 7
    table = torch.rand(1, size, size, size, 3)
    pts = lattice_points(size).unsqueeze(0)
    flat = table.reshape(1, -1, 3)
    assert float((tetra_lookup(table, pts) - flat).abs().max()) < 1e-6
    assert float((trilinear_lookup(table, pts) - flat).abs().max()) < 1e-6


def test_tetrahedral_reproduces_affine_functions_exactly():
    """The property the bake gate leans on: tetrahedral interpolation is
    piecewise linear, so an affine function survives a lattice round trip."""
    size = 9
    A = torch.tensor([[0.9, 0.1, 0.0], [0.0, 0.8, 0.1], [0.05, 0.0, 0.95]])
    c = torch.tensor([0.02, 0.01, 0.0])
    pts = lattice_points(size)
    table = (pts @ A.T + c).reshape(1, size, size, size, 3)
    x = torch.rand(1, 1024, 3)
    want = x[0] @ A.T + c
    assert float((tetra_lookup(table, x)[0] - want).abs().max()) < 1e-5


def test_domain_min_max_is_honoured(tmp_path):
    size = 5
    lines = ["LUT_3D_SIZE 5", "DOMAIN_MIN 0.2 0.2 0.2", "DOMAIN_MAX 0.8 0.8 0.8"]
    for b in range(size):
        for g in range(size):
            for r in range(size):
                lines.append(f"{r/4:.6f} {g/4:.6f} {b/4:.6f}")
    p = tmp_path / "dom.cube"
    p.write_text("\n".join(lines) + "\n")
    t = load_gt_table(p, "dom")
    # 0.2 -> the table's 0 corner, 0.8 -> its 1 corner
    lo = t.apply(torch.tensor([[0.2, 0.2, 0.2]]), "trilinear")
    hi = t.apply(torch.tensor([[0.8, 0.8, 0.8]]), "trilinear")
    assert torch.allclose(lo, torch.zeros(1, 3), atol=1e-5)
    assert torch.allclose(hi, torch.ones(1, 3), atol=1e-5)
    # and values outside the domain are clamped into it, not extrapolated
    assert torch.allclose(t.apply(torch.tensor([[0.0, 0.0, 0.0]]), "trilinear"),
                          torch.zeros(1, 3), atol=1e-5)


def test_non_finite_table_is_rejected():
    bad = torch.rand(5, 5, 5, 3)
    bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        GtLutTable("bad", bad, torch.zeros(3), torch.ones(3))


def test_lut_bank_lru_and_missing_id(tmp_path):
    paths = {}
    for i in range(4):
        p = write_cube(tmp_path / f"{i}.cube", 5,
                       lambda r, g, b, i=i: (r, g, min(1.0, b + 0.1 * i)))
        paths[f"lut{i}"] = str(p)
    bank = LutBank(path_map=paths, capacity=2)
    for k in ("lut0", "lut1", "lut2", "lut0"):
        bank.get(k)
    assert len(bank._cache) <= 2
    assert bank.n_misses >= 3
    with pytest.raises(KeyError):
        bank.get("not_a_lut")
