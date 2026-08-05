"""soft-IoU must be the MIN/MAX form -- the product form is banned.

Where-A result review B-1: the product form ``sum(p*g)/sum(p + g - p*g)``
correlates **0.955 with the softness of the GT mask** and **-0.003 with fit
quality**, and its ceiling on a perfect prediction has median **0.786**.  A
§5.6 "soft-IoU >= 0.75" gate evaluated that way would rank arms by how soft
their GT masks happen to be -- structurally the same failure that got AUC
banned campaign-wide.  These tests make the form a pinned property of the
criteria path, not a default someone can flip.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from q3vl.whereb import config as C
from q3vl.whereb import metrics as M
from q3vl.whereb.losses import soft_iou


def _soft_gt(n=2000, seed=7):
    return torch.rand(n, generator=torch.Generator().manual_seed(seed))


# --- the decisive property --------------------------------------------------

def test_a_perfect_prediction_of_a_soft_gt_scores_exactly_one():
    """This is the whole issue: under the product form it does not."""
    g = _soft_gt()
    assert M.soft_iou_value(g, g) == pytest.approx(1.0, abs=1e-5)
    prod = float(soft_iou(g.double(), g.double(), "prod"))
    assert prod < 0.6, f"product form ceiling was {prod}, expected well below 1"


@pytest.mark.parametrize("v", [0.5, 0.25, 0.1, 0.9])
def test_the_ceiling_is_one_at_every_softness_level(v):
    g = torch.full((256,), v)
    assert M.soft_iou_value(g, g) == pytest.approx(1.0, abs=1e-5)


def test_the_metric_is_insensitive_to_gt_softness_when_the_prediction_is_perfect():
    """B-1's 0.955 correlation with softness must not exist for min/max."""
    scores = [M.soft_iou_value(torch.full((512,), v), torch.full((512,), v))
              for v in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert max(scores) - min(scores) < 1e-4, scores
    # the product form, by contrast, tracks softness almost perfectly
    prod = [float(soft_iou(torch.full((512,), v).double(),
                           torch.full((512,), v).double(), "prod"))
            for v in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert max(prod) - min(prod) > 0.5, prod


# --- the criteria path really uses it ---------------------------------------

def test_config_declares_minmax():
    assert C.SOFT_IOU_KIND == "minmax"


def test_soft_iou_value_follows_the_config_constant():
    src = inspect.getsource(M.soft_iou_value)
    assert "SOFT_IOU_KIND" in src, (
        "soft_iou_value must read the pinned constant, not rely on a default "
        "argument that a future edit could flip"
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_metrics_matches_minmax_and_not_product_on_random_soft_data(seed):
    g = torch.rand(512, generator=torch.Generator().manual_seed(seed))
    p = torch.rand(512, generator=torch.Generator().manual_seed(seed + 100))
    v = M.soft_iou_value(p, g)
    mm = float(soft_iou(p.double(), g.double(), "minmax"))
    pr = float(soft_iou(p.double(), g.double(), "prod"))
    assert v == pytest.approx(mm, abs=1e-9)
    assert abs(v - pr) > 1e-3, "the two forms coincided; pick a discriminating case"


def test_sample_metrics_soft_iou_is_the_minmax_form():
    g = _soft_gt(n=64)
    m = M.sample_metrics(g.reshape(8, 8), g.reshape(8, 8))
    assert m["soft_iou"] == pytest.approx(1.0, abs=1e-5)


# --- the ban ----------------------------------------------------------------

def test_no_criteria_module_selects_the_product_form():
    """Negative assertion: nothing on the criteria path may pass kind='prod'.

    AST-based, not a text scan: the docstrings on this path deliberately *name*
    the banned form to explain why it is banned, and a grep would flag the
    explanation as the offence.
    """
    import ast

    from q3vl.whereb import evaluate, losses

    def code_string_constants(mod) -> list[str]:
        tree = ast.parse(inspect.getsource(mod))
        docstrings = {
            id(node.value) for node in ast.walk(tree)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        }
        return [n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and id(n) not in docstrings]

    for mod in (M, evaluate):
        assert "prod" not in code_string_constants(mod), (
            f"{mod.__name__} selects the banned product form in code"
        )
    # losses.soft_iou keeps the branch (Where-A reports both for provenance),
    # but its default is min/max and mask_loss pins it explicitly
    assert inspect.signature(losses.soft_iou).parameters["kind"].default == "minmax"
    assert "iou_kind" in inspect.getsource(losses.mask_loss)


def test_the_oracle_ratio_denominator_is_pinned():
    """The >=85%-of-oracle gate needs its denominator stated, or a different
    tier / a different form silently changes the gate."""
    ceil = C.ORACLE_CEILING_LOW_MINMAX
    assert ceil == {"band": 0.827, "cband12": 0.835}
    # with a ceiling near 0.83, the absolute >=0.75 row implies ~90% of oracle,
    # i.e. the absolute row is the stricter of the two -- worth knowing which
    # one actually binds before reading a near-miss as an oracle problem.
    for r, c in ceil.items():
        implied = 0.75 / c
        assert implied > 0.85, (r, implied)
