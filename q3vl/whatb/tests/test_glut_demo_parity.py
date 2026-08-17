"""Point-wise parity against the official GLUT demo.  This is the ground floor.

``docs/HANDOFF_whatb_2026-08-15.md`` section 6.4: the demo
(``https://color.cvc.uab.cat/assets/html/glut_editor.html``) is the only
executable official implementation, and it embeds **seven trained GLUT-32
weight sets** at line 429, so "did we read Eq.1-5 correctly" is a measurement,
not an opinion.

The fixture ``data/glut_demo_ref.json`` was produced by ``data/dump_ref.mjs``,
which splices lines 435..611 of the page -- the ``GaussianLUT`` class from its
head through the end of ``forward`` -- into node **verbatim** and evaluates the
class's own ``forward`` on 256 query colours for all seven models.  The fixture
carries the page's sha256 so a silently-changed page is detectable::

    curl --cacert <bundle-with-HARICA-GEANT-TLS-R1> \\
         https://color.cvc.uab.cat/assets/html/glut_editor.html   # HTTP 200
    sha256 863bb1cbb3a22d8a162929db52f44c9c310353a908238115fe1e4b05088147c2

(The site serves no intermediate certificate; fetch
``http://crt.harica.gr/HARICA-GEANT-TLS-R1.cer`` per AIA and append it to the
CA bundle.  ``-k`` is not used and is not needed.)

Comparison runs in float64 on CPU, clamp ``two`` (the frozen default: the demo
clamps the global branch at :574-579 and the sum at :606-610), residual on (all
seven embedded models declare ``residual: true``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from q3vl.whatb.glut import GlutParams, glut_forward

REF_PATH = Path(__file__).parent / "data" / "glut_demo_ref.json"
PAGE_SHA256 = "863bb1cbb3a22d8a162929db52f44c9c310353a908238115fe1e4b05088147c2"

#: Measured max |ours - demo| over 7 models x 256 colours x 3 channels is
#: **5.551e-16** in float64 (2026-08-15, torch 2.10.0, node v22.21.0) -- i.e.
#: 2.5 ulp of a value in [0,1].  The bar is set six orders of magnitude above
#: that so libm differences across machines cannot flake it, and still eight
#: orders below anything that could move a criterion.  float32 lands at 4.5e-7.
PARITY_ATOL = 1e-10


@pytest.fixture(scope="module")
def ref() -> dict:
    with REF_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _params_from_demo(model: dict) -> GlutParams:
    p = model["parameters"]
    t = lambda k: torch.tensor(p[k], dtype=torch.float64)
    n = int(model["num_gaussians"])
    return GlutParams(
        mu=t("positions").reshape(1, n, 3),
        chol_diag=t("cholesky_diag").reshape(1, n, 3),
        chol_off=t("cholesky_off").reshape(1, n, 3),
        opacity_logit=t("opacities_logit").reshape(1, n),
        m_local=t("color_matrices").reshape(1, n, 3, 3),
        b_local=t("color_biases").reshape(1, n, 3),
        g_matrix=t("global_matrix").reshape(1, 3, 3),
        g_bias=t("global_bias").reshape(1, 3),
    )


def test_fixture_provenance(ref: dict) -> None:
    assert ref["source_sha256"] == PAGE_SHA256
    assert ref["source_url"] == "https://color.cvc.uab.cat/assets/html/glut_editor.html"
    assert ref["source_lines"] == {"embedded_models": 429, "class_from": 435, "class_to": 611}
    assert len(ref["models"]) == 7
    assert len(ref["queries"]) == 256
    for model in ref["models"].values():
        assert model["num_gaussians"] == 32
        assert model["residual"] is True


def test_forward_matches_demo_on_all_seven_models(ref: dict) -> None:
    x = torch.tensor(ref["queries"], dtype=torch.float64)
    worst = 0.0
    per_model: dict[str, float] = {}
    for name, model in ref["models"].items():
        params = _params_from_demo(model)
        got = glut_forward(x, params, clamp="two", residual=True)
        assert got.shape == (1, 256, 3)
        want = torch.tensor(model["outputs"], dtype=torch.float64).unsqueeze(0)
        dev = (got - want).abs().max().item()
        per_model[name] = dev
        worst = max(worst, dev)
    print("\nGLUT demo parity, max |ours - demo| per model (float64):")
    for name, dev in per_model.items():
        print(f"  {dev:.3e}  {name}")
    print(f"  {worst:.3e}  WORST over 7 models x 256 colours x 3 channels")
    assert worst < PARITY_ATOL, f"worst deviation {worst:.3e} >= {PARITY_ATOL:.1e}"


def test_parity_survives_float32(ref: dict) -> None:
    """The same comparison in the dtype training will actually run in."""
    x = torch.tensor(ref["queries"], dtype=torch.float32)
    worst = 0.0
    for model in ref["models"].values():
        params = _params_from_demo(model).to(dtype=torch.float32)
        got = glut_forward(x, params, clamp="two", residual=True)
        want = torch.tensor(model["outputs"], dtype=torch.float32).unsqueeze(0)
        worst = max(worst, (got - want).abs().max().item())
    print(f"\nGLUT demo parity in float32: worst = {worst:.3e}")
    assert worst < 1e-5


def test_single_clamp_differs_only_where_the_global_branch_leaves_the_gamut(ref: dict) -> None:
    """``--clamp one`` (paper Eq.4/5) vs ``two`` (demo): the ablation is real.

    Where ``Gx + g`` already lies in ``[0,1]^3`` the two modes coincide by
    construction; the frozen default is ``two`` precisely because the paper does
    not exclude the intermediate clamp and the demo performs it.
    """
    x = torch.tensor(ref["queries"], dtype=torch.float64)
    n_diff_total = 0
    for model in ref["models"].values():
        params = _params_from_demo(model)
        two = glut_forward(x, params, clamp="two")
        one = glut_forward(x, params, clamp="one")
        glob = torch.einsum("bij,bpj->bpi", params.g_matrix, x.unsqueeze(0)) + params.g_bias.unsqueeze(1)
        inside = ((glob >= 0.0) & (glob <= 1.0)).all(dim=-1)
        assert torch.allclose(two[inside], one[inside], atol=1e-12)
        n_diff_total += int((~inside).sum())
    assert n_diff_total > 0, "no query left the global branch's gamut -- the ablation would be vacuous"
