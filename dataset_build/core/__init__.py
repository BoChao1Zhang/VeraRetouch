"""Shared rendering backend retained for non-canonical consumers.

Canonical databuild imports :mod:`construct.rendering` directly and cannot reach
this package's farm-capable backend.
"""

from __future__ import annotations

from . import render_backend
from .render_backend import RenderBackend, get_backend

__all__ = ["RenderBackend", "get_backend", "render_backend"]
