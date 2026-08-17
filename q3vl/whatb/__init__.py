"""``q3vl.whatb`` -- the What-side namespace for EPR-024..029.

Clean-room by ruling (user, 2026-08-14): ``q3vl/what/``, ``model/glut_repro/``,
``gpu_render/`` and every old What experiment record are contaminated sources.
Nothing in this package reads or imports any of them --
``tests/test_no_contaminated_imports.py`` enforces that mechanically.

What lives here so far (the shared layer; per-arm modules go in
``q3vl/whatb/<epr name>/``)::

    glut.py       GLUT Eq.1-5 forward, the ONE carrier all six arms use
    generator.py  CGLUT conditional generator + the affine-only (EPR-025) mode
    gate.py       identity-anchored strength gate (EPR-027)
    guards.py     degenerate-solution guard (+ its "did it run" witness) and the
                  first-step-row contract
    zcache.py     the ONE z cache: writer, reader, start-up assertions, fp32
    evaldata.py   the ONE image / GT-alpha loader at the frozen headline size
    colorimetry.py  the ONE sRGB->CIELab / dE76 / dE00 / chroma-hue
"""

from q3vl.whatb.gate import IdentityGate, identity_gate
from q3vl.whatb.generator import (
    SEG_COLOR_HIDDEN_DIM,
    CGLUTGenerator,
    SegColorProjection,
    SharedGeometry,
    generator_param_count,
)
from q3vl.whatb.glut import (
    CLAMP_FLAG_CHOICES,
    EPS,
    LOG_2PI,
    GlutAux,
    GlutCarrier,
    GlutParams,
    glut_forward,
    glut_geometry,
    n_params_glut,
    softplus_inverse,
    uniform_grid_positions,
)
from q3vl.whatb.guards import (
    DegeneracyReport,
    DegeneracyThresholds,
    DegenerateTransform,
    LossColumnsMissing,
    StepsRowUnavailable,
    assert_first_step_columns,
    assert_transform_not_degenerate,
    record_step_witness,
    resolve_first_step_row,
)

__all__ = [
    "CGLUTGenerator",
    "CLAMP_FLAG_CHOICES",
    "DegeneracyReport",
    "DegeneracyThresholds",
    "DegenerateTransform",
    "EPS",
    "GlutAux",
    "GlutCarrier",
    "GlutParams",
    "IdentityGate",
    "LOG_2PI",
    "LossColumnsMissing",
    "SEG_COLOR_HIDDEN_DIM",
    "SegColorProjection",
    "SharedGeometry",
    "StepsRowUnavailable",
    "assert_first_step_columns",
    "assert_transform_not_degenerate",
    "generator_param_count",
    "glut_forward",
    "glut_geometry",
    "identity_gate",
    "n_params_glut",
    "record_step_witness",
    "resolve_first_step_row",
    "softplus_inverse",
    "uniform_grid_positions",
]
