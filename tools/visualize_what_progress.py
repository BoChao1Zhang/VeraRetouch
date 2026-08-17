#!/usr/bin/env python
"""Plot the running E031 selection-subset snapshots without publishing them."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


RUNS = {
    "MLP d=64": "/home/bc/data/logs/whatb-e031-mlp-lr1e3-20260816.log",
    "MLP d=256": "/home/bc/data/logs/whatb-e031-mlp-cd256-l3-20260816.log",
    "MLP d=512": "/home/bc/data/logs/whatb-e031-mlp-cd512-l3-20260816.log",
    "MLP no L8": "/home/bc/data/logs/whatb-e031-mlp-nol8-l3-20260816.log",
    "q-dec M=1, lr=4e-3": "/home/bc/data/logs/whatb-e031-qdec-lr4e3-20260816.log",
    "q-dec M=4, lr=3e-4": "/home/bc/data/logs/whatb-e031-qdec-mem4-lr3e4-20260816.log",
}


def _points(path: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if "step" in row and "headline" in row:
            xs.append(int(row["step"]))
            ys.append(float(row["headline"]))
    return xs, ys


def main() -> None:
    out_dir = Path("docs/assets/what_global_20260816")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(7.0, 4.4), dpi=180)
    for name in ("MLP d=64", "MLP d=256", "MLP d=512", "MLP no L8"):
        xs, ys = _points(RUNS[name])
        axis.plot(xs, ys, marker="o", markersize=3, linewidth=1.6, label=name)
    axis.set_title("MLP controls")
    axis.set_xlabel("Training step")
    axis.set_ylabel("Selection-subset DeltaE00 (lower is better)")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, fontsize=8)
    fig.suptitle("E031 running snapshots: 2,097,152 colors/step, not final boards", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "what_e031_running_mlp.png", bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.0, 4.4), dpi=180)
    for name in ("q-dec M=1, lr=4e-3", "q-dec M=4, lr=3e-4"):
        xs, ys = _points(RUNS[name])
        axis.plot(xs, ys, marker="o", markersize=3, linewidth=1.6, label=name)
    axis.set_title("Query-decoder stability scan")
    axis.set_xlabel("Training step")
    axis.set_ylabel("Selection-subset DeltaE00 (lower is better)")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, fontsize=8)
    fig.suptitle("E031 running snapshots: 2,097,152 colors/step, not final boards", y=1.03)
    fig.tight_layout()
    fig.savefig(out_dir / "what_e031_running_qdec.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
