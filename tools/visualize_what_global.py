#!/usr/bin/env python
"""Render global-boundary qualitative samples for the completed E030 MLP arm.

The selected samples are ``style`` examples, whose construction fixes alpha=1
everywhere.  The resulting panels therefore isolate the what branch: input,
LUT target, predicted global transform, and a per-pixel DeltaE00 map.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from q3vl.whatb.colorimetry import delta_e00_srgb
from q3vl.whatb.evaldata import SampleStore
from q3vl.whatb.lutdata import LutBank
from q3vl.whatb.scripts.run_epr030_arm import Epr030Config, Epr030Model
from q3vl.whatb.splits import load_index, normal_only
from q3vl.whatb.zcache import ZCache


SAMPLE_IDS = (
    "sft_002170cf2d81b95329b29ac1a3ef284f",
    "sft_0158b54ffc201a88efe73277ef399557",
    "sft_01a7fa8028998c1527f419c368c4082a",
    "sft_02ed4ee5562f824e91beac3b1ac2830f",
    "sft_0395765b96fcf1de051a6386d7aa2934",
    "sft_03cda3fdf949e7a890f7a13004dbf348",
    "sft_04260b07f1bb5473737b70843dcee2f0",
    "sft_042e9e4def70924a6f3e28c421466536",
)


def _image_array(image: torch.Tensor) -> np.ndarray:
    return image.detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)


def _model(checkpoint: Path) -> Epr030Model:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = payload["config"]
    cfg = Epr030Config(**{f.name: cfg_dict[f.name] for f in fields(Epr030Config)})
    model = Epr030Model(cfg).eval()
    model.load_state_dict(payload["model"])
    return model


def render(checkpoint: Path, out_dir: Path, metadata: Path) -> None:
    rows = {row.sample_id: row for row in normal_only(load_index("V_what"))}
    selected = [rows[sample_id] for sample_id in SAMPLE_IDS]
    if any(row.task_type != "style" for row in selected):
        raise RuntimeError("Global what panels require style samples (alpha=1).")

    model = _model(checkpoint)
    cache = ZCache("/home/bc/data/runs/whatb/zcache_v2seg/V_what.generated.none.zcache.pt")
    store = SampleStore("V_what")
    bank = LutBank()

    results = []
    for row in selected:
        image, alpha = store.load(row, device="cpu")
        if not isinstance(alpha, float) or alpha != 1.0:
            raise RuntimeError(f"{row.sample_id}: alpha is not global")
        with torch.no_grad():
            target = bank.apply_image(image, row.lut_id)
            prediction = model.transform_image(
                cache.vector(row.sample_id), image, point_chunk=16384
            )
            de_map = delta_e00_srgb(
                prediction.permute(1, 2, 0), target.permute(1, 2, 0)
            ).detach().cpu().numpy()
        results.append((row, image, target, prediction, de_map))

    column_titles = ("Input", "LUT target", "E030 MLP prediction", "Absolute DeltaE00")
    records = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for group_index, start in enumerate(range(0, len(results), 2), start=1):
        group = results[start:start + 2]
        fig, axes = plt.subplots(len(group), 4, figsize=(14.8, 7.5), dpi=180)
        axes = np.atleast_2d(axes)
        for row_idx, (row, image, target, prediction, de_map) in enumerate(group):
            panels = (_image_array(image), _image_array(target), _image_array(prediction))
            for col_idx, panel in enumerate(panels):
                axes[row_idx, col_idx].imshow(panel)
                axes[row_idx, col_idx].axis("off")
                if row_idx == 0:
                    axes[row_idx, col_idx].set_title(column_titles[col_idx], fontsize=11)

            heat = axes[row_idx, 3].imshow(de_map, cmap="magma", vmin=0.0, vmax=16.0)
            axes[row_idx, 3].axis("off")
            if row_idx == 0:
                axes[row_idx, 3].set_title(column_titles[3], fontsize=11)
            axes[row_idx, 0].text(
                0.0, -0.06, f"sample {row.sample_id[-6:]} | mean DeltaE00 {de_map.mean():.2f}",
                transform=axes[row_idx, 0].transAxes, fontsize=8, va="top"
            )
            records.append({
                "sample_id": row.sample_id,
                "lut_id": row.lut_id,
                "task_type": row.task_type,
                "alpha": "1.0 everywhere (global boundary)",
                "mean_delta_e00": float(de_map.mean()),
                "group": group_index,
            })

        fig.subplots_adjust(left=0.01, right=0.90, bottom=0.06, top=0.88,
                            wspace=0.04, hspace=0.34)
        color_axis = fig.add_axes((0.92, 0.16, 0.014, 0.64))
        fig.colorbar(heat, cax=color_axis, label="DeltaE00")
        fig.suptitle("What branch qualitative results: global boundary (alpha = 1)",
                     fontsize=14, y=0.995)
        fig.savefig(out_dir / f"what_global_e030_mlp_group_{group_index:02d}.png",
                    bbox_inches="tight")
        plt.close(fig)

    metadata.write_text(json.dumps({
        "checkpoint": str(checkpoint),
        "scope": "Global-boundary style samples only; alpha=1 everywhere.",
        "records": records,
    }, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/home/bc/data/runs/what_b/whatb_QDEC_P4_MLP/best.pt"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("docs/assets/what_global_20260816"))
    parser.add_argument("--metadata", type=Path,
                        default=Path("docs/assets/what_global_20260816/what_global_e030_mlp.json"))
    args = parser.parse_args()
    render(args.checkpoint, args.out_dir, args.metadata)


if __name__ == "__main__":
    main()
