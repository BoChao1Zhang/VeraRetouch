"""Stable cache-key helper shared by the instance-level SAM3 producer and reader."""
from __future__ import annotations

import hashlib
import os
from typing import Any


def path_key(path: Any) -> str:
    """Return the stable 16-hex directory key for one source path."""
    return hashlib.sha1(os.fspath(path).encode("utf-8")).hexdigest()[:16]


__all__ = ["path_key"]
