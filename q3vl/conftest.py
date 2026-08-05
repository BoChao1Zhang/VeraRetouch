"""Campaign-wide pytest guard: sqlite3 must be imported before torch.

Scope is deliberately ``q3vl/`` and not the repository root, so the
``dataset_build`` and ``databuild_viewer`` suites are untouched.

Campaign bug R6 (found by WHAT-IMPL, 2026-08-05).  Measured in the campaign env
``/home/bc/envs/q3vl_sft``::

    import sqlite3; import torch   -> fine
    import torch;   import sqlite3 -> ImportError, libstdc++ CXXABI_1.3.15 not found

torch loads a libstdc++ that shadows the one ``_sqlite3``'s dependency chain
(libicui18n) needs, so any process that touches torch first can never open a
published shard afterwards.  Several test modules import ``q3vl.data.shardio``
or ``q3vl.where.maskdata`` at module level, and pytest reaches torch through the
package chain before it reaches them -- so without this line the suites fail at
*collection* unless the operator remembers to export
``LD_LIBRARY_PATH=/home/bc/miniconda3/envs/llm_factory/lib``.

pytest loads this file before any test module, and ``q3vl/__init__.py`` is empty,
so this really is the first import of the session.  The entry-point scripts carry
the same guard for the same reason; see
``experiments/Q3VL_metacanvas_where_what_20260804/where_b/NOTES.md`` section 10
for the full analysis, including why the guard must NOT be added to library
modules (it would make them unimportable in any torch-first process).
"""

import sqlite3  # noqa: F401  (import order is the point)
