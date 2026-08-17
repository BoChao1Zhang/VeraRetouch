"""The contamination ruling, enforced mechanically rather than by memory.

User ruling 2026-08-14 (``docs/HANDOFF_whatb_2026-08-15.md`` section 9):
``q3vl/what/``, ``model/glut_repro/``, ``gpu_render/``, every old What
experiment record and ``trash/`` are contaminated sources -- not one line read,
not one symbol imported.  ``q3vl/whereb/readout.py:484``
(``from q3vl.what.context import encode_color_span``) is the single existing
edge into that tree, and ``q3vl/whatb/colorspan.py`` is the sanctioned way
around it; nothing in this package may take the edge itself.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent
FORBIDDEN_ROOTS = ("q3vl.what", "model.glut_repro", "gpu_render", "trash")


def _module_names(tree: ast.AST) -> list[str]:
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.append(node.module)
    return out


def _forbidden(name: str) -> bool:
    for root in FORBIDDEN_ROOTS:
        if name == root or name.startswith(root + "."):
            # `q3vl.whatb` must not be caught by the `q3vl.what` prefix.
            if root == "q3vl.what" and name.startswith("q3vl.whatb"):
                continue
            return True
    return False


def test_no_module_in_whatb_imports_a_contaminated_tree() -> None:
    files = sorted(PACKAGE.rglob("*.py"))
    assert files, "the package should have modules"
    offences: list[str] = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        offences += [
            f"{path.relative_to(PACKAGE.parent)}: {name}"
            for name in _module_names(tree)
            if _forbidden(name)
        ]
    assert offences == [], "contaminated imports: " + "; ".join(offences)


def test_the_guard_itself_is_not_vacuous() -> None:
    assert _forbidden("q3vl.what")
    assert _forbidden("q3vl.what.context")
    assert _forbidden("gpu_render.gpu")
    assert _forbidden("model.glut_repro.forward")
    assert not _forbidden("q3vl.whatb")
    assert not _forbidden("q3vl.whatb.glut")
    assert not _forbidden("q3vl.whereb.contracts")
    assert not _forbidden("torch")


def test_whatb_imports_cleanly_without_pulling_in_the_what_tree() -> None:
    import sys

    import q3vl.whatb  # noqa: F401

    loaded = [m for m in sys.modules if m.startswith("q3vl.what.") or m == "q3vl.what"]
    assert loaded == [], f"importing q3vl.whatb loaded {loaded}"
