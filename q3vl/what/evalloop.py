"""In-loop evaluation: the ``eval_fn`` protocol 10.4's ``eval_steps: 500`` needs.

Review NF-2 found that nothing was calling the evaluation library: ``run_what.py``
constructed the trainer without an ``eval_fn``, so ``eval_steps`` was unimplemented,
amendment A-4's two boards had no producer, and -- worst -- blocker B-2's checkpoint
protection was **inert in production**, because with no eval report there is no
``best()``, so nothing was ever marked ``best_protected`` and the rolling deletion
kept eating the first six checkpoints exactly as before.

The main agent's ruling is route (a) plus an offline complement:

* **here**: every ``eval_steps`` steps, on a *fixed deterministic subset* of
  ``V_what``, both contexts, **LUT-function metrics only** -- the 2048 query
  points of protocol 9.1, no full-image render.  That is what keeps the wall
  clock bounded; a full render of 897 images twice per eval would cost more than
  the training step it interrupts;
* **offline** (:mod:`q3vl.what.scripts.evaluate_what`): the complete ``V_what``,
  both contexts, image metrics, protocol 12.2 partitions and 12.3 controls.
  **That** is the input to ``main_board`` and the basis of the final selection.

The two are deliberately not the same measurement, and the online one is named
accordingly (``ONLINE_SELECTION_KEY`` is a *proxy*).  What matters is that the
proxy is **monotone in the same direction** as the offline primary key: both are
CIEDE2000 on local samples, smaller is better, generated context.  The proxy
decides which checkpoint files survive the rolling deletion; the offline board
decides which checkpoint wins.  A proxy that ranked differently would let the
rolling deletion throw away the file the offline board later wants -- which is
B-2 again, one level up.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from .config import (
    CONTEXT_GENERATED,
    CONTEXT_GT,
    CONTEXT_MODES,
    EVAL_SUBSET_SEED,
    EVAL_SUBSET_SIZE,
    MASK_AREA_BINS,
    ONLINE_SELECTION_KEY,
    SELECTION_CONTEXT,
    ArmConfig,
)
from .evaluate import arm_metrics, context_report, summarise
from .gaussians import bake, render
from .lut import tetra_lookup
from .metrics import bake_metrics, lut_metrics

__all__ = ["stable_order_key", "mask_area_bin", "stratum_of", "build_eval_subset",
           "subset_digest", "make_eval_fn", "evaluate_subset_once",
           "eval_subset_rows"]


# --- the fixed deterministic subset -----------------------------------------

def stable_order_key(sample_id: str, seed: int) -> str:
    """A per-(seed, sample) ordering key that is identical in every process.

    ``random.shuffle`` and ``torch.randperm`` would both do, but only inside one
    process: the subset has to be reproducible from a manifest months later, on a
    different machine, possibly under a different Python.  A sha256 is.
    """
    return hashlib.sha256(f"{seed}|{sample_id}".encode()).hexdigest()


def mask_area_bin(area: float | None) -> str:
    """Protocol 12.2's mask-area stratum.  ``None`` when no mask source exists."""
    if area is None:
        return "unknown"
    for i, edge in enumerate(MASK_AREA_BINS):
        if area < edge:
            return f"a{i}"
    return f"a{len(MASK_AREA_BINS)}"


def stratum_of(row: Mapping[str, Any], use_mask_area: bool) -> tuple[str, ...]:
    """``(build, render_mode[, mask_area_bin])`` -- the strata the ruling names."""
    key = (str(row.get("build")), str(row.get("render_mode")))
    if use_mask_area:
        key = key + (mask_area_bin(row.get("mask_area")),)
    return key


def build_eval_subset(
    rows: Sequence[Mapping[str, Any]],
    n: int = EVAL_SUBSET_SIZE,
    seed: int = EVAL_SUBSET_SEED,
) -> tuple[list[int], dict[str, Any]]:
    """A fixed, stratified, deterministic subset of ``V_what``.

    ``rows`` carry ``sample_id`` / ``build`` / ``render_mode`` and, when a mask
    source is available, ``mask_area``.  Whether mask area was actually used is
    **recorded in the manifest** rather than assumed: the maskviews are published
    by Where-A's own job and may not exist when an arm starts, and a subset that
    silently stopped stratifying by mask area would quietly change what the
    online proxy measures.

    Allocation is proportional with largest remainders, so every stratum that
    exists in ``V_what`` is represented and the subset's build/render_mode mix
    matches the split's.  Within a stratum, members are taken in
    :func:`stable_order_key` order.
    """
    if n <= 0:
        raise ValueError(f"subset size must be positive, got {n}")
    use_mask_area = any(r.get("mask_area") is not None for r in rows)
    groups: dict[tuple[str, ...], list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(stratum_of(r, use_mask_area), []).append(i)
    for k in groups:
        groups[k].sort(key=lambda i: stable_order_key(str(rows[i]["sample_id"]), seed))

    total = len(rows)
    n = min(n, total)
    # largest-remainder proportional allocation, deterministic tie-break by key
    exact = {k: len(v) * n / total for k, v in groups.items()}
    take = {k: min(len(groups[k]), int(v)) for k, v in exact.items()}
    remaining = n - sum(take.values())
    order = sorted(groups, key=lambda k: (-(exact[k] - int(exact[k])), k))
    for k in order:
        if remaining <= 0:
            break
        if take[k] < len(groups[k]):
            take[k] += 1
            remaining -= 1
    while remaining > 0:                       # strata exhausted; top up the big ones
        grew = False
        for k in sorted(groups, key=lambda k: (-len(groups[k]), k)):
            if remaining > 0 and take[k] < len(groups[k]):
                take[k] += 1
                remaining -= 1
                grew = True
        if not grew:
            break

    picked: list[int] = []
    for k in sorted(groups):
        picked.extend(groups[k][: take[k]])
    picked.sort(key=lambda i: stable_order_key(str(rows[i]["sample_id"]), seed))

    manifest = {
        "schema": "q3vl.what.evalsubset/1",
        "seed": seed,
        "requested": n,
        "n": len(picked),
        "n_population": total,
        "strata_keys": (["build", "render_mode"] +
                        (["mask_area_bin"] if use_mask_area else [])),
        "mask_area_used": use_mask_area,
        "mask_area_bins": list(MASK_AREA_BINS) if use_mask_area else None,
        "strata": {"|".join(k): {"population": len(groups[k]), "taken": take[k]}
                   for k in sorted(groups)},
        "sample_ids": [str(rows[i]["sample_id"]) for i in picked],
    }
    manifest["digest"] = subset_digest(manifest["sample_ids"])
    return picked, manifest


def subset_digest(sample_ids: Sequence[str]) -> str:
    """Digest of the *content* of the subset, order-independent.

    Goes into the run's config digest so that "which 256 samples the online proxy
    was computed on" is part of the run's identity rather than a re-derivable
    convention.
    """
    h = hashlib.sha256()
    for s in sorted(sample_ids):
        h.update(s.encode())
        h.update(b"\n")
    return h.hexdigest()


def eval_subset_rows(dataset, maskviews=None) -> list[dict[str, Any]]:
    """Stratification rows for a split, without loading a single image.

    ``mask_area`` comes from Where-A's published low-resolution mask view -- a
    ~40x32 float array per sample, so 897 of them is a trivial read and gives a
    real area rather than a proxy.  When the maskviews have not been published
    yet the field is ``None`` and :func:`build_eval_subset` records that it
    stratified on ``(build, render_mode)`` only, instead of silently dropping a
    stratum the ruling asked for.
    """
    rows: list[dict[str, Any]] = []
    for i, ref in enumerate(dataset.refs):
        meta = dict(ref.meta)
        render_mode = meta.get("render_mode")
        if render_mode is None:
            from .config import LOCAL_BUILDS

            render_mode = "local" if meta.get("build") in LOCAL_BUILDS else "global"
        area: float | None = None
        if maskviews is not None and render_mode == "local":
            try:
                if maskviews.has(ref.sample_id, maskviews.LOW):
                    area = float(maskviews.mask_low(ref.sample_id).mean())
            except (KeyError, OSError, ValueError):
                area = None
        elif render_mode == "global":
            area = 1.0
        rows.append({"index": i, "sample_id": ref.sample_id,
                     "build": meta.get("build"), "render_mode": render_mode,
                     "mask_area": area})
    return rows


# --- one evaluation pass -----------------------------------------------------

@torch.no_grad()
def evaluate_subset_once(
    model,
    builder,
    dataset,
    indices: Sequence[int],
    arm_cfg: ArmConfig,
    *,
    context: str,
    micro_batch: int = 4,
    bake_size: int | None = None,
) -> list[dict[str, Any]]:
    """LUT-function metrics for one context over the subset -- no image render.

    Protocol 9.1's 2048 query points are already built by the batch builder, and
    ``T_gt`` at those points is already a target, so the per-sample row costs one
    forward plus one 33^3 bake.  The bake is kept because the protocol 12.1 gate
    lives on it and a checkpoint that drifts out of the gate should be visible
    at 500 steps, not at the end of the epoch.
    """
    if context not in CONTEXT_MODES:
        raise ValueError(f"unknown context {context!r}")
    was_training = model.training
    model.eval()
    prev_stats = getattr(model, "collect_pool_stats", True)
    model.collect_pool_stats = False
    rows: list[dict[str, Any]] = []
    try:
        for i in range(0, len(indices), micro_batch):
            chunk = list(indices[i:i + micro_batch])
            samples = [dataset[j] for j in chunk]
            batch = builder.build(samples, [context] * len(samples))
            out = model(**batch.inputs)
            params = {k: (v.float() if torch.is_tensor(v) else v)
                      for k, v in out.params.items()}
            x = torch.stack([t["x"] for t in batch.targets]).float()
            t_gt = torch.stack([t["t_gt"] for t in batch.targets]).float()
            t_pred = render(params, x, arm_cfg.lut)
            cube = bake(params, arm_cfg.lut, bake_size) if bake_size else \
                bake(params, arm_cfg.lut)
            t_read = tetra_lookup(cube, x.clamp(0, 1))
            kind = batch.targets[0]["query_kind"]
            for b, tgt in enumerate(batch.targets):
                row: dict[str, Any] = {
                    "sample_id": tgt["sample_id"], "lut_id": tgt.get("lut_id"),
                    "context": context,
                    "natural_weighting": tgt.get("natural_weighting"),
                    **{k: v for k, v in (tgt.get("meta") or {}).items()
                       if k in ("build", "render_mode", "winner_confidence",
                                "upscaled")},
                }
                for ci, name in enumerate(("uniform", "natural")):
                    sel = kind == ci
                    if bool(sel.any()):
                        row.update(lut_metrics(t_pred[b][sel], tgt["t_gt"][sel],
                                               f"{name}_"))
                row.update(lut_metrics(t_pred[b], tgt["t_gt"], "lut_"))
                row.update(bake_metrics(t_pred[b:b + 1], t_read[b:b + 1]))
                rows.append(row)
    finally:
        model.collect_pool_stats = prev_stats
        if was_training:
            model.train()
    return rows


def make_eval_fn(
    model,
    builder,
    dataset,
    indices: Sequence[int],
    arm_cfg: ArmConfig,
    *,
    n_trainable: int,
    micro_batch: int = 4,
    subset_manifest: Mapping[str, Any] | None = None,
    out_dir: Path | None = None,
) -> Callable[[int], dict[str, Any]]:
    """The ``eval_fn`` :class:`~q3vl.what.trainer.WhatTrainer` calls every 500 steps.

    Returns one report per call containing:

    * ``contexts.gt`` / ``contexts.generated`` -- two full ``arm_metrics`` rows
      (amendment A-4 item 2: never averaged together);
    * ``gap`` -- the paired generated-minus-teacher difference (A-4 item 3's
      "how much does this arm depend on GT colour reasoning");
    * the generated row's selection keys promoted to the top level, so
      ``WhatTrainer.best`` -- and therefore B-2's checkpoint protection -- reads
      the **generated** board, the same one the offline selection will use.

    ``local_image_de00_median`` is deliberately **absent**: this pass does not
    render images, and a key that does not mean what its name says elsewhere is
    worse than a missing one.  The online proxy is ``ONLINE_SELECTION_KEY``.
    """
    ids = list(indices)

    def eval_fn(step: int) -> dict[str, Any]:
        t0 = time.time()
        per_context: dict[str, Any] = {}
        rows_by_context: dict[str, list[dict[str, Any]]] = {}
        for ctx in (CONTEXT_GT, CONTEXT_GENERATED):
            rows = evaluate_subset_once(model, builder, dataset, ids, arm_cfg,
                                        context=ctx, micro_batch=micro_batch)
            rows_by_context[ctx] = rows
            m = arm_metrics(rows, arm=arm_cfg.arm, step=step,
                            n_trainable=n_trainable, context=ctx)
            local = [r for r in rows if r.get("render_mode") == "local"]
            m[ONLINE_SELECTION_KEY] = summarise(local or rows).get(
                "lut_de00_median_median")
            per_context[ctx] = m

        sel = per_context[SELECTION_CONTEXT]
        report: dict[str, Any] = {
            "step": step,
            "kind": "online_subset_lut_function",
            "selection_context": SELECTION_CONTEXT,
            "online_selection_key": ONLINE_SELECTION_KEY,
            "n_subset": len(ids),
            "subset_digest": (subset_manifest or {}).get("digest"),
            "contexts": per_context,
            "gap": context_report(
                [per_context[CONTEXT_GT], per_context[CONTEXT_GENERATED]],
                metric=ONLINE_SELECTION_KEY),
            "eval_seconds": round(time.time() - t0, 2),
        }
        # promote the generated board's gate + selection keys
        report[ONLINE_SELECTION_KEY] = sel.get(ONLINE_SELECTION_KEY)
        for k in ("gate_pass", "bake_mae_mean", "bake_err_p99", "bake_non_finite",
                  "lut_non_finite", "lut_de00_p90"):
            report[k] = sel.get(k)
        if out_dir is not None:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            with (Path(out_dir) / "eval_per_sample.jsonl").open("a",
                                                               encoding="utf-8") as fh:
                for ctx, rows in rows_by_context.items():
                    for r in rows:
                        fh.write(json.dumps({"step": step, **r},
                                            ensure_ascii=False, sort_keys=True) + "\n")
        return report

    return eval_fn
