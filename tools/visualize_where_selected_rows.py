#!/usr/bin/env python3
"""Render eight newly inferred, one-row ``where`` comparison figures.

Each PNG is intentionally one sample only.  A left-hand text panel retains the
verbatim instruction and the generated ``<where>`` span; the five visual panels
are ``Input | GT alpha | ST_LANG | SEGSAM | MATTE``.  The source image is unique
across all eight selections.

The selection is deterministic and uses *published* per-sample generated-context
scores only to select illustrative cases.  Fields and displayed IoUs are fresh
inference results produced by this script.  This distinction is recorded in the
manifest so the PNGs are never mistaken for a re-crop of the 2026-08-15 board.

The command needs the project GPU Python environment.  In this workspace,
PyTorch's bundled libstdc++ otherwise precedes conda's sqlite dependency, so use
the ``LD_PRELOAD`` invocation shown in the report's reproduction note.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


ROOT = Path("/home/bc/VeraRetouch")
RUN_ROOT = Path("/home/bc/data/runs/where_b")
V2_CHECKPOINT = "/home/bc/data/runs/q3vl_base_sft_v2seg_20260814/checkpoint-4976"
V2_GENCTX = RUN_ROOT / "genwhere_v2seg"
ARMS = ("ST_LANG", "SEGSAM", "MATTE")
RUNS = {
    "ST_LANG": RUN_ROOT / "amort_UNIQ_stlang_20260813",
    "SEGSAM": RUN_ROOT / "amort_SEGSAM_20260814",
    "MATTE": RUN_ROOT / "amort_MATTE_20260814",
}


@dataclass(frozen=True)
class SelectedSample:
    index: int
    sample_id: str
    source_image_id: str
    family: str
    role: str
    historic_scores: dict[str, float]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs/assets/where_selected_rows_20260817")
    ap.add_argument("--split", default="V_where")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--dpi", type=int, default=170)
    return ap.parse_args()


def _read_setup(run: Path) -> dict[str, Any]:
    return json.loads((run / "config/run_setup.json").read_text(encoding="utf-8"))


def _historical_scores(run: Path) -> dict[str, float]:
    """Generated-context, normal-only matched-area top-k IoUs from the board."""
    path = run / "eval_final/per_sample.jsonl"
    out: dict[str, float] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if (row.get("mode") == "generated"
                    and row.get("winner_confidence") == "normal"
                    and not row.get("uncovered", False)):
                out[str(row["sample_id"])] = float(row["hard_iou"])
    if not out:
        raise RuntimeError(f"no generated-context normal rows in {path}")
    return out


def select_samples(ds, families: dict[str, str]) -> list[SelectedSample]:
    """Two samples/family, one source image per output.

    The first band image is deliberately the lowest historical ST_LANG score to
    preserve a concrete geometry failure.  Its paired band image and all other
    images are the highest mean score over the three displayed methods.  This is
    a visual coverage rule, not a reported aggregate result.
    """
    scores = {arm: _historical_scores(RUNS[arm]) for arm in ARMS}
    candidates: dict[str, list[dict[str, Any]]] = {f: [] for f in
                                                    ("radial", "linear", "semantic", "band")}
    for i, meta in enumerate(ds.meta_rows()):
        if (meta.get("render_mode") != "local"
                or meta.get("winner_confidence") != "normal"):
            continue
        sid = str(meta["sample_id"])
        fam = families.get(sid)
        if fam not in candidates or any(sid not in scores[a] for a in ARMS):
            continue
        rec = ds.record(i)
        source = str(rec.get("source_image_id") or sid)
        candidates[fam].append({
            "index": i,
            "sample_id": sid,
            "source_image_id": source,
            "family": fam,
            "scores": {a: scores[a][sid] for a in ARMS},
        })

    used_sources: set[str] = set()
    selected: list[SelectedSample] = []

    def take(pool: list[dict[str, Any]], role: str) -> None:
        for row in pool:
            if row["source_image_id"] in used_sources:
                continue
            selected.append(SelectedSample(
                index=int(row["index"]), sample_id=str(row["sample_id"]),
                source_image_id=str(row["source_image_id"]), family=str(row["family"]),
                role=role, historic_scores={k: float(v) for k, v in row["scores"].items()},
            ))
            used_sources.add(str(row["source_image_id"]))
            return
        raise RuntimeError(f"could not choose a source-unique sample for {role}")

    mean_rank = lambda r: (-np.mean(list(r["scores"].values())), r["sample_id"])
    st_rank = lambda r: (r["scores"]["ST_LANG"], r["sample_id"])
    for family in ("radial", "linear", "semantic"):
        pool = sorted(candidates[family], key=mean_rank)
        take(pool, "positive")
        take(pool, "positive")

    take(sorted(candidates["band"], key=st_rank), "counterexample")
    take(sorted(candidates["band"], key=mean_rank), "positive")

    if len(selected) != 8 or len({s.source_image_id for s in selected}) != len(selected):
        raise AssertionError("selection must contain eight source-unique samples")
    return selected


def extract_where(generated_text: str) -> str:
    match = re.search(r"<where>.*?</where>", generated_text or "", flags=re.DOTALL)
    if not match:
        raise ValueError("generated text has no complete <where> span")
    return match.group(0)


def _make_vlm(checkpoint: str, *, device: str, attn: str, query_tokens: bool):
    from transformers import AutoProcessor

    from q3vl.train.modeling import load_model
    from q3vl.whereb.hiddens import FrozenVLM

    proc = AutoProcessor.from_pretrained(checkpoint)
    model = load_model(checkpoint, attn_implementation=attn,
                       dtype="bfloat16").to(device)
    if not query_tokens:
        return proc, model, FrozenVLM(model, proc, device=device, want_merger=True)

    from q3vl.whereb.amort.uniq4 import VARIANT4
    from q3vl.whereb.amort.uniq4b import LangQueryTokVLM

    setup = json.loads((RUNS["ST_LANG"] / "config/uniq4b_setup.json").read_text())
    VARIANT4.update(n_qtok=int(setup["n_qtok"]), lora_r=int(setup["lora_r"]),
                    head_cls=None, head_kwargs={}, vlm_ref=None)
    vlm = LangQueryTokVLM(model, proc, device=device, n_qtok=int(setup["n_qtok"]),
                          lora_r=int(setup["lora_r"]), want_merger=True)
    VARIANT4["vlm_ref"] = vlm
    return proc, model, vlm


def _new_arm_model(arm: str, *, device: str):
    """Reconstruct a new arm from its recorded run arguments and final state."""
    from q3vl.whereb.amort.model import AmortModel

    setup = _read_setup(RUNS[arm])
    cfg = dict(setup["args"])
    if arm == "SEGSAM":
        from q3vl.whereb.amort import segsam

        opts = (setup.get("model", {}).get("arm_head", {})
                .get("options", {}))
        segsam.configure(SimpleNamespace(**{
            f"segsam_{key}": value for key, value in opts.items()
        }))
    elif arm == "MATTE":
        from q3vl.whereb.amort import matte

        # The original MATTE entry uses this provider seam to retain the RGB
        # input alongside pixel GT.  It must be installed before the builder.
        matte.install_image_seam()

    model = AmortModel(
        arm,
        readout=str(cfg.get("readout", "band")),
        new_arm_defaults=not bool(cfg.get("newarm_legacy_routing", False)),
        arm_args=SimpleNamespace(**cfg),
        use_sim_field=not bool(cfg.get("no_sim_field", False)),
        use_center_prior_channel=bool(cfg.get("center_prior_channel", False)),
        with_semantic=not bool(cfg.get("no_semantic_head", False)),
        use_film=not bool(cfg.get("no_film", False)),
        pooled_w=bool(cfg.get("pooled_w", False)),
        geom_inject=bool(cfg.get("geom_inject", False)),
        geom_mode=str(cfg.get("geom_mode", "broadcast")),
        pch_size=str(cfg.get("pch_size", "full")),
        pch_impl=str(cfg.get("pch_impl", "v0")),
        seed=int(cfg.get("seed", 0)),
    ).to(device)
    state = torch.load(RUNS[arm] / "amort_final.pt", map_location=device,
                       weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model


def _st_model(*, device: str):
    from q3vl.whereb.amort.uniq4 import AmortModelV4

    setup = _read_setup(RUNS["ST_LANG"])
    cfg = dict(setup["args"])
    model = AmortModelV4(
        "UNIQ", readout=str(cfg.get("readout", "band")),
        use_sim_field=not bool(cfg.get("no_sim_field", False)),
        use_center_prior_channel=bool(cfg.get("center_prior_channel", False)),
        with_semantic=not bool(cfg.get("no_semantic_head", False)),
        use_film=not bool(cfg.get("no_film", False)),
        seed=int(cfg.get("seed", 0)),
    ).to(device)
    state = torch.load(RUNS["ST_LANG"] / "amort_final.pt", map_location=device,
                       weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model


def _new_builder(ds, selected: list[SelectedSample], families, *, device: str, attn: str):
    from q3vl.train.collator import Sft2SegCollator
    from q3vl.whereb.amort.data import AmortBatchBuilder
    from q3vl.whereb.amort import matte
    from q3vl.whereb.amort.matte import builder_kwargs
    from q3vl.whereb.amort.simfield import SimFieldNorm, WordEmbedder
    from q3vl.whereb.config import CONSTRUCT_GEOM_DB
    from q3vl.whereb.fields import load_basis
    from q3vl.whereb.readout import ReadoutBuilder, ReadoutSpec
    from q3vl.whereb.stores import GenContextStore
    from q3vl.whereb.amort.data import ConstructGeomStore

    # The MATTE checkpoint expects its PixGT object to carry the RGB image.
    # Install that arm-local provider seam before the provider is constructed.
    matte.install_image_seam()
    from q3vl.whereb.amort.pixgt import PixGTProvider

    proc, vlm_model, vlm = _make_vlm(V2_CHECKPOINT, device=device, attn=attn,
                                     query_tokens=False)
    collator = Sft2SegCollator(proc, max_length=2048, system_prompt=None)
    setup = _read_setup(RUNS["SEGSAM"])
    norm = SimFieldNorm.from_dict(setup["sim_norm"])
    selected_idx = {s.sample_id: s.index for s in selected}
    gstore = ConstructGeomStore(CONSTRUCT_GEOM_DB)
    pixgt = PixGTProvider(
        geom_store=gstore, families=families,
        mask_resolver=getattr(ds, "mask_resolver", None),
        maskviews=getattr(ds, "maskviews", None), prefer="render",
        raster_fallback="cgt1024")
    matte_args = SimpleNamespace(**_read_setup(RUNS["MATTE"])["args"])
    extra = builder_kwargs(matte_args)
    builder = AmortBatchBuilder(
        collator, vlm, load_basis("BA-3-Joint").to(device),
        embedder=WordEmbedder.from_checkpoint(V2_CHECKPOINT, proc.tokenizer),
        norm=norm, genctx=GenContextStore(V2_GENCTX / ds.split),
        families=families, id_to_index=selected_idx, dataset=ds, device=device,
        attn_implementation=attn, checkpoint=V2_CHECKPOINT,
        readout=ReadoutBuilder(collator.tokenizer, ReadoutSpec(kind="seg_where"),
                               on_missing_tag="last"),
        pixgt=pixgt, **extra)
    return builder, vlm_model


def _st_builder(ds, selected: list[SelectedSample], families, *, device: str, attn: str):
    from q3vl.train.collator import Sft2SegCollator
    from q3vl.whereb.amort.data import AmortBatchBuilder
    from q3vl.whereb.amort.simfield import SimFieldNorm, WordEmbedder
    from q3vl.whereb.fields import load_basis
    from q3vl.whereb.stores import GenContextStore

    checkpoint = _read_setup(RUNS["ST_LANG"])["args"]["checkpoint"]
    proc, vlm_model, vlm = _make_vlm(checkpoint, device=device, attn=attn,
                                     query_tokens=True)
    collator = Sft2SegCollator(proc, max_length=2048, system_prompt=None)
    setup = _read_setup(RUNS["ST_LANG"])
    selected_idx = {s.sample_id: s.index for s in selected}
    builder = AmortBatchBuilder(
        collator, vlm, load_basis("BA-3-Joint").to(device),
        embedder=WordEmbedder.from_checkpoint(checkpoint, proc.tokenizer),
        norm=SimFieldNorm.from_dict(setup["sim_norm"]),
        genctx=GenContextStore(V2_GENCTX / ds.split), families=families,
        id_to_index=selected_idx, dataset=ds, device=device,
        attn_implementation=attn, checkpoint=checkpoint)
    return builder, vlm_model


def infer(builder, model, sample) -> tuple[torch.Tensor, float]:
    from q3vl.whereb.metrics import gt_area_k, hard_iou, topk_mask

    x = builder.build([sample], ["generated"])[0]
    with torch.no_grad():
        cond = model.cond_of(x.cond_h, x.cond_mask, x.word_ids, x.word_offsets)
        out = model.forward_geo(
            x.feat, cond, x.phi_dir, sim=x.sim, center=x.center, geom=x.geom,
            guide_hi=x.guide_hi, grid_h=x.grid_h, grid_w=x.grid_w,
            h_where=x.cond_h, h_mask=x.cond_mask, h_cond=x.h_cond, sample=x)
    field = out["m_low"].detach().float().cpu()
    gt = x.gt_low.detach().float().cpu()
    k = gt_area_k(gt)
    return field, hard_iou(topk_mask(field, k), topk_mask(gt, k))


def render_one(*, out: Path, sample, info: SelectedSample, reasoning: str,
               fields: dict[str, torch.Tensor], ious: dict[str, float], dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    from q3vl.whereb.viz import overlay_grid_on_image

    image = sample.image_tensor()
    gt = sample.mask_target_hi().float()
    gh, gw = fields["ST_LANG"].shape
    gt_low = torch.nn.functional.interpolate(gt[None, None], size=(gh, gw),
                                              mode="area")[0, 0]
    overlays = {"GT alpha": gt_low, **fields}

    fig = plt.figure(figsize=(25.5, 4.8), constrained_layout=True)
    grid = fig.add_gridspec(1, 6, width_ratios=(1.55, 1, 1, 1, 1, 1), wspace=0.025)
    text_ax = fig.add_subplot(grid[0, 0])
    text_ax.set_axis_off()
    text_ax.add_patch(FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96, transform=text_ax.transAxes,
        boxstyle="round,pad=0.025,rounding_size=0.012",
        facecolor="#f4f7f9", edgecolor="#334e5c", linewidth=1.4))
    instruction = textwrap.fill(sample.instruction.strip(), width=42)
    where = textwrap.fill(reasoning.strip(), width=42)
    text_ax.text(0.07, 0.93,
                 f"{info.family.upper()}  |  {info.role}\n{info.sample_id[-12:]}",
                 transform=text_ax.transAxes, va="top", fontsize=10.5,
                 fontweight="bold", color="#18323d")
    text_ax.text(0.07, 0.78, f"Instruction\n{instruction}",
                 transform=text_ax.transAxes, va="top", fontsize=8.5,
                 linespacing=1.32, color="#16242b")
    text_ax.text(0.07, 0.34, f"Generated reasoning\n{where}",
                 transform=text_ax.transAxes, va="top", fontsize=8.5,
                 linespacing=1.32, color="#16242b")

    labels = ("Input", "GT alpha", "ST_LANG", "SEGSAM", "MATTE")
    for col, label in enumerate(labels, start=1):
        ax = fig.add_subplot(grid[0, col])
        ax.set_axis_off()
        if label == "Input":
            ax.imshow(image.permute(1, 2, 0).cpu().numpy())
            subtitle = f"source: {info.source_image_id}"
        else:
            overlay, stats = overlay_grid_on_image(
                overlays[label], image, alpha=0.55, fixed=(0.0, 1.0),
                allow_all_valid=True)
            ax.imshow(np.clip(overlay, 0.0, 1.0))
            subtitle = (f"IoU={ious[label]:.3f}" if label in ious
                        else f"GT area={float(gt_low.mean()):.3f}")
            ax.text(0.02, 0.02, subtitle, transform=ax.transAxes, fontsize=9,
                    color="white", va="bottom", ha="left",
                    bbox={"facecolor": "#111827", "alpha": 0.78, "pad": 2.5,
                          "edgecolor": "none"})
        ax.set_title(label, fontsize=13, pad=8)
    fig.savefig(out, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from q3vl.whereb.amort.data import family_labels
    from q3vl.whereb.data import open_dataset
    from q3vl.whereb.stores import GenContextStore

    ds, _ = open_dataset(args.split, need_mask=True)
    local_idx = [i for i, row in enumerate(ds.meta_rows())
                 if row.get("render_mode") == "local"
                 and row.get("winner_confidence") == "normal"]
    families = family_labels(ds, local_idx)
    selected = select_samples(ds, families)
    genctx = GenContextStore(V2_GENCTX / args.split)

    # New-arm VLM and heads share one fresh forward per sample.
    new_builder, new_vlm_model = _new_builder(ds, selected, families,
                                              device=args.device, attn=args.attn)
    new_models = {arm: _new_arm_model(arm, device=args.device)
                  for arm in ("SEGSAM", "MATTE")}
    new_out: dict[str, tuple[Any, dict[str, torch.Tensor], dict[str, float]]] = {}
    for pick in selected:
        sample = ds[pick.index]
        fields, ious = {}, {}
        for arm, model in new_models.items():
            field, iou = infer(new_builder, model, sample)
            fields[arm], ious[arm] = field, iou
        new_out[pick.sample_id] = (sample, fields, ious)

    del new_models, new_builder, new_vlm_model
    gc.collect()
    torch.cuda.empty_cache()

    # ST_LANG requires its original in-context query-token and language-LoRA VLM.
    st_builder, st_vlm_model = _st_builder(ds, selected, families,
                                           device=args.device, attn=args.attn)
    st_model = _st_model(device=args.device)
    manifest_samples = []
    for pick in selected:
        sample, fields, ious = new_out[pick.sample_id]
        st_field, st_iou = infer(st_builder, st_model, sample)
        fields = {"ST_LANG": st_field, **fields}
        ious = {"ST_LANG": st_iou, **ious}
        text = genctx.record(pick.sample_id)["generated_text"]
        where = extract_where(text)
        png = args.out / f"where_{pick.family}_{pick.role}_{pick.sample_id[-12:]}.png"
        render_one(out=png, sample=sample, info=pick, reasoning=where,
                   fields=fields, ious=ious, dpi=args.dpi)
        manifest_samples.append({
            "sample_id": pick.sample_id,
            "source_image_id": pick.source_image_id,
            "family": pick.family,
            "role": pick.role,
            "instruction": sample.instruction,
            "generated_where": where,
            "historical_generated_topk_iou_for_selection": pick.historic_scores,
            "fresh_v2seg_context_topk_iou": ious,
            "figure": png.name,
        })
        print(f"wrote {png.name}", flush=True)

    manifest = {
        "generated": "2026-08-17",
        "split": args.split,
        "selection": {
            "population": "local, winner_confidence=normal",
            "rule": (
                "source-unique across all eight images; radial/linear/semantic use "
                "descending historical mean IoU over ST_LANG, SEGSAM and MATTE; "
                "band includes the historical ST_LANG minimum counterexample plus "
                "the best remaining mean-IoU positive example"),
            "historical_score_context": "generated",
        },
        "inference": {
            "context": "fresh inference using v2seg generated context cache",
            "genctx": str(V2_GENCTX / args.split),
            "displayed_arms": {
                arm: {"run": str(RUNS[arm]), "weights": "amort_final.pt"}
                for arm in ARMS
            },
            "metric": "matched-area top-k IoU; k is GT alpha area at the low grid",
            "note": (
                "Fresh displayed IoUs need not equal the historic board because "
                "ST_LANG is replayed with the newer v2seg generated context."),
        },
        "layout": {
            "one_sample_per_png": True,
            "left_text_box": "verbatim instruction plus generated <where> reasoning",
            "visual_columns": ["Input", "GT alpha", "ST_LANG", "SEGSAM", "MATTE"],
            "field_scale": [0.0, 1.0],
        },
        "samples": manifest_samples,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
