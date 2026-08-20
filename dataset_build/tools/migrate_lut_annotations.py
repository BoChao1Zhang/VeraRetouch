#!/usr/bin/env python3
"""Build the resumable closed-set LUT catalog used by the agent loop."""
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_build.agent_loop.lut_annotations import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
