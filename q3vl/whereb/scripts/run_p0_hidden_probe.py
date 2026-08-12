"""P0 (P0a) -- layer-wise linear probe of the ``<where>`` hidden states for geometry.

The question, in one line: the generated ``<where>`` *text* describes the mask
geometry at ~82% word-level accuracy; does the ``<where>`` **hidden state**
carry that geometry *more losslessly* than the sampled text does?  If a linear
probe on the hidden beats the text, a hidden-side readout (attention readout /
query bridge, RESEARCH_geometry-extraction-arch §4 arms B1/C) can beat the
text-sampling loss and the three-arm comparison is justified.

Discipline notes that are structural rather than stylistic:

* **Labels never come from text.**  They come from the construction-side
  ``.vrmeta.json`` (``slot_id`` -> shape family, ``region`` -> direction), which
  is the only ground truth for the geometry and is completely independent of
  what either the GT or the generated ``<where>`` span happens to say.  Extent
  slots are absent from vrmeta, so they stay 0 **and are excluded from every
  reported number**.
* **One MaskResolver per thread.**  It holds sqlite handles; sharing one fails a
  load-dependent fraction of lookups (72% of 75,544 inside a real run) while
  leaving the surviving histogram looking perfectly normal.
* **Two context arms, not one.**  The task card specifies the GT (teacher-forced)
  span, which is the information-theoretic upper bound: it answers "what *can*
  the hidden carry".  But that arm is fed the answer -- the GT text names the
  geometry -- so a high number there is partly "read back the tokens we just
  supplied", which the layer-0 (token-embedding) row makes visible.  The
  **generated** arm conditions on the model's own 82%-accurate span and is the
  arm that is directly comparable to the text baselines.  Both are run; both are
  reported; neither is allowed to stand in for the other.
* **No AUC anywhere** (project red line).  This is a discrete-slot probe, so the
  criteria are accuracy / F1 / exact-set match, each against an explicit
  constant-predictor baseline and a label-permutation control.  The direction
  group in particular is 76% ``center``, so a bare "dir_any = 0.9" would mean
  nothing without the constant column.

Run directly (no queue): one frozen forward over 400 samples, micro-batch 4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

# --- the label space ---------------------------------------------------------
# vrmeta populates 9 of the 20 GEOM_SLOTS.  The other 11 (extent, plus the
# text-only shape_oval / dir_edge / dir_horizontal / dir_vertical / dir_diagonal)
# have no ground truth on the construction side and are excluded everywhere.
SHAPE_SLOTS = ("shape_radial", "shape_linear", "shape_band", "shape_semantic")
DIR_SLOTS = ("dir_center", "dir_bottom", "dir_left", "dir_right", "dir_top")
POPULATED = SHAPE_SLOTS + DIR_SLOTS
EXCLUDED_NOTE = (
    "vrmeta records slot_id (shape family) and region (direction) only. The 6 "
    "extent slots (ext_large/ext_small/ext_whole/ext_partial/ext_soft/ext_hard) "
    "and the 5 text-only slots (shape_oval, dir_edge, dir_horizontal, "
    "dir_vertical, dir_diagonal) have NO construction-side ground truth: they "
    "are held at 0 in the label vector and are EXCLUDED from every accuracy, "
    "F1 and group metric reported here. All numbers below concern the 9 "
    "populated slots only."
)


# --- provenance --------------------------------------------------------------
def _provenance(repo: str = "/home/bc/VeraRetouch") -> dict[str, Any]:
    out: dict[str, Any] = {"git_commit": "unknown", "working_tree_dirty": None,
                           "changed_files": []}
    try:
        out["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo).decode().strip()
        st = subprocess.check_output(["git", "status", "--porcelain"],
                                     cwd=repo).decode().strip()
        out["working_tree_dirty"] = bool(st)
        out["changed_files"] = [ln[3:] for ln in st.splitlines()][:80]
    except Exception:  # noqa: BLE001
        pass
    try:
        import hashlib as _h

        h = _h.sha256()
        for extra in ("q3vl/whereb/scripts/run_p0_hidden_probe.py",
                      "q3vl/whereb/amort/geomparse.py",
                      "q3vl/whereb/hiddens.py", "q3vl/whereb/context.py"):
            h.update(Path(repo, extra).read_bytes())
        out["source_sha256"] = h.hexdigest()
    except Exception:  # noqa: BLE001
        pass
    return out


def split_half(sample_id: str) -> int:
    """sha1(sample_id) % 2 -- 0 = probe-train, 1 = probe-test.

    The campaign's split rule family (content hash, not an index prefix), so the
    halves stay stable under any reordering of the split index and carry no
    correlation with build order.
    """
    return int(hashlib.sha1(sample_id.encode("utf-8")).hexdigest(), 16) % 2


# --- per-layer <where> pooling ----------------------------------------------
class WherePooler:
    """Mask-mean-pool over the ``<where>`` positions of every decoder layer.

    ``output_hidden_states=True`` would allocate ``(L+1, B, T, 2560)`` and throw
    all but a pooled row of it away.  Hooking each layer and pooling *inside the
    hook* keeps the peak at one layer's activations, which is what lets this
    coexist with two training jobs on the same cards.
    """

    def __init__(self, lm: torch.nn.Module):
        self.lm = lm
        self.n_layers = len(lm.layers)
        self.embed = getattr(lm, "embed_tokens", None)
        self.mask: torch.Tensor | None = None
        self.pooled: dict[str, torch.Tensor] = {}
        self._handles: list[Any] = []

    # layer keys: "0" = token embeddings (before layer 1), "1".."L" = the output
    # of decoder layer i, "L_norm" = the last layer after the final RMSNorm --
    # the state `q3vl.whereb.hiddens` calls H_where (config.WHERE_HIDDEN_LAYER
    # = -1, WHERE_HIDDEN_FINAL_NORM = True).
    def layer_keys(self) -> list[str]:
        keys = ([] if self.embed is None else ["0"])
        keys += [str(i) for i in range(1, self.n_layers + 1)]
        keys += [f"{self.n_layers}_norm"]
        return keys

    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        assert self.mask is not None
        m = self.mask.to(h.device, dtype=torch.float32)
        hf = h.float()
        s = (hf * m.unsqueeze(-1)).sum(dim=1)
        return (s / m.sum(dim=1, keepdim=True).clamp(min=1.0)).to("cpu")

    def _layer_fn(self, i: int):
        def fn(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output
            self.pooled[str(i + 1)] = self._pool(h)
            if i == self.n_layers - 1:
                self.pooled[f"{self.n_layers}_norm"] = self._pool(self.lm.norm(h))
        return fn

    def _embed_fn(self, _m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        self.pooled["0"] = self._pool(h)

    def attach(self) -> None:
        if self.embed is not None:
            self._handles.append(self.embed.register_forward_hook(self._embed_fn))
        for i, layer in enumerate(self.lm.layers):
            self._handles.append(layer.register_forward_hook(self._layer_fn(i)))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


# --- the probe ---------------------------------------------------------------
def _fit(X: torch.Tensor, Y: torch.Tensor, lam: float, max_iter: int) -> tuple:
    """Multi-label one-vs-rest logistic regression, L2, LBFGS.

    Objective: ``mean_{n,k} BCE(logits, y) + lam * ||W||_F^2 / (n * k)``.
    sklearn is not installed in this environment, so this is the whole probe.
    """
    n, d = X.shape
    k = Y.shape[1]
    W = torch.zeros(d, k, device=X.device, dtype=X.dtype, requires_grad=True)
    b = torch.zeros(k, device=X.device, dtype=X.dtype, requires_grad=True)
    lossf = torch.nn.BCEWithLogitsLoss(reduction="mean")
    opt = torch.optim.LBFGS([W, b], max_iter=max_iter, history_size=20,
                            line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7, tolerance_change=1e-9)

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = lossf(X @ W + b, Y) + lam * (W * W).sum() / (n * k)
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def _bce(X, Y, W, b) -> float:
    with torch.no_grad():
        return float(torch.nn.functional.binary_cross_entropy_with_logits(
            X @ W + b, Y).item())


def select_lambda(X: torch.Tensor, Y: torch.Tensor, grid: Sequence[float],
                  folds: int, max_iter: int, seed: int) -> tuple[float, dict]:
    """K-fold CV **inside the train half only** -- the test half is never touched.

    Fixing one lambda a priori would be a coin flip at n~200 / d=2560 (the
    difference between under- and over-regularised is the whole result), and
    tuning on test would be the other kind of mistake.  CV on train is the
    leak-free option, and the chosen value is reported per layer.
    """
    n = X.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    cuts = [perm[i::folds] for i in range(folds)]
    scores: dict[str, float] = {}
    for lam in grid:
        tot = 0.0
        for f in range(folds):
            te = cuts[f]
            tr = torch.cat([cuts[j] for j in range(folds) if j != f])
            W, b = _fit(X[tr], Y[tr], lam, max_iter)
            tot += _bce(X[te], Y[te], W, b)
        scores[f"{lam:g}"] = tot / folds
    best = min(scores, key=lambda k: scores[k])
    return float(best), scores


# --- group metrics -----------------------------------------------------------
def _f1(pred: np.ndarray, true: np.ndarray) -> float:
    tp = float((pred & true).sum())
    fp = float((pred & ~true).sum())
    fn = float((~pred & true).sum())
    return (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else float("nan")


def group_metrics(prob: np.ndarray, Y: np.ndarray) -> dict[str, Any]:
    """Everything reported per layer.  ``prob`` and ``Y`` are (n, 9)."""
    pred = prob >= 0.5
    true = Y >= 0.5
    ns = len(SHAPE_SLOTS)
    out: dict[str, Any] = {"n": int(Y.shape[0]), "per_slot": {}}
    for j, name in enumerate(POPULATED):
        out["per_slot"][name] = {
            "acc": float((pred[:, j] == true[:, j]).mean()),
            "f1": _f1(pred[:, j], true[:, j]),
            "n_pos_true": int(true[:, j].sum()),
            "n_pos_pred": int(pred[:, j].sum()),
        }
    out["macro_acc"] = float(np.mean([v["acc"] for v in out["per_slot"].values()]))
    out["macro_f1"] = float(np.nanmean([v["f1"] for v in out["per_slot"].values()]))

    # --- shape group: argmax over the 4 shape slots vs the true family. This is
    # the number directly comparable to the text's 0.820.
    has_shape = true[:, :ns].any(axis=1)
    if has_shape.any():
        gold = true[:, :ns][has_shape].argmax(axis=1)
        got = prob[:, :ns][has_shape].argmax(axis=1)
        out["shape_argmax_acc"] = float((gold == got).mean())
        out["shape_n"] = int(has_shape.sum())
        cm = np.zeros((ns, ns), dtype=int)
        for a, p in zip(gold, got):
            cm[a, p] += 1
        out["shape_confusion"] = {SHAPE_SLOTS[a]: {SHAPE_SLOTS[p]: int(cm[a, p])
                                                   for p in range(ns)}
                                  for a in range(ns)}
    else:
        out["shape_argmax_acc"] = float("nan")
        out["shape_n"] = 0
    # like-for-like with the text baselines, which are multi-hot and may fire
    # zero or several shape words
    sp, st = pred[:, :ns], true[:, :ns]
    out["shape_any_overlap"] = float(((sp & st).any(axis=1)).mean())
    out["shape_exact_set"] = float((sp == st).all(axis=1).mean())

    dp, dt = pred[:, ns:], true[:, ns:]
    out["dir_exact"] = float((dp == dt).all(axis=1).mean())
    out["dir_any"] = float(((dp & dt).any(axis=1)).mean())
    nonempty = dt.any(axis=1)
    out["dir_n_true_nonempty"] = int(nonempty.sum())
    out["dir_any_nonempty"] = (float(((dp & dt).any(axis=1))[nonempty].mean())
                               if nonempty.any() else float("nan"))
    return out


def constant_baseline(Ytr: np.ndarray, Yte: np.ndarray) -> dict[str, Any]:
    """The zero-information predictor: per slot, the train-majority label.

    Mandatory column.  The direction group is ~76% ``dir_center`` on this split,
    so any direction number without it is unreadable -- the same failure mode the
    campaign's centre-prior red line exists to prevent.
    """
    maj = (Ytr.mean(axis=0) >= 0.5).astype(np.float32)
    prob = np.tile(maj, (Yte.shape[0], 1))
    # argmax over the shape block for a constant predictor is degenerate, so use
    # the train-majority FAMILY (the most frequent single shape slot) instead
    ns = len(SHAPE_SLOTS)
    prob = prob.copy()
    prob[:, :ns] = 0.0
    prob[:, int(Ytr[:, :ns].sum(axis=0).argmax())] = 1.0
    out = group_metrics(prob, Yte)
    out["majority_vector"] = {n: float(v) for n, v in zip(POPULATED, maj)}
    return out


# --- main --------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="V_where")
    ap.add_argument("--checkpoint",
                    default="/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--contexts", default="gt,generated",
                    help="comma list: gt (teacher-forced, upper bound) and/or "
                         "generated (the model's own 82%%-accurate span)")
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260811)
    # the smoke run selected the top of a 1e-3..1e2 grid at every layer, i.e. the
    # grid was truncating; it now spans two decades past that so the CV choice is
    # interior rather than clipped
    ap.add_argument("--lam-grid", default="1e-2,1e-1,1,10,100,1000,10000")
    ap.add_argument("--cv-folds", type=int, default=3)
    ap.add_argument("--lbfgs-iter", type=int, default=120)
    # Measured, not assumed: one LBFGS fit of this shape (202x2560 -> 9) costs
    # 0.07 s single-threaded on CPU and ~4.8 s on cuda:0 while two trainings are
    # resident -- 70x, because the problem is far too small to amortise kernel
    # launches and loses every scheduling race on a contended card. 8 CPU threads
    # is also worse than 1 (1.26 s) for the same reason, so the probe stage
    # pins itself to a single thread.
    ap.add_argument("--probe-device", default="cpu")
    ap.add_argument("--probe-threads", type=int, default=1)
    args = ap.parse_args(argv)

    t_start = time.time()

    # RLIMIT_NOFILE soft -> hard, before anything opens a descriptor: the vrmeta
    # lookup opens one sqlite catalogue per build batch from `workers` threads,
    # and at the inherited soft limit of 1024 that surfaces as sqlite "unable to
    # open database file" rather than as an error anyone would recognise.
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f"RLIMIT_NOFILE {soft} -> {resource.getrlimit(resource.RLIMIT_NOFILE)[0]}",
              flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"WARN could not raise RLIMIT_NOFILE: {exc}", flush=True)

    import threading
    from concurrent.futures import ThreadPoolExecutor

    from transformers import AutoProcessor

    from q3vl.train.collator import Sft2SegCollator
    from q3vl.train.modeling import load_model
    from q3vl.where.maskdata import MaskResolver
    from q3vl.whereb.amort.geomparse import (GEOM_SLOTS, geom_features,
                                             geom_features_from_vrmeta)
    from q3vl.whereb.config import GENCTX_DIR
    from q3vl.whereb.context import generated_context, gt_context
    from q3vl.whereb.data import _PromptShim, open_dataset
    from q3vl.whereb.hiddens import resolve_language_model
    from q3vl.whereb.stores import GenContextStore

    slot_index = {name: i for i, (name, _) in enumerate(GEOM_SLOTS)}
    keep_cols = [slot_index[n] for n in POPULATED]

    out_dir = Path(args.out)
    (out_dir / "config").mkdir(parents=True, exist_ok=True)
    print(f"out {out_dir}", flush=True)

    contexts = [c.strip() for c in args.contexts.split(",") if c.strip()]
    for c in contexts:
        if c not in ("gt", "generated"):
            raise SystemExit(f"unknown context {c!r}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -- data -------------------------------------------------------------
    # need_mask=False: this probe never looks at a mask, and resolving .cgt.png
    # for 400 samples is pure latency.
    ds, ds_info = open_dataset(args.split, need_mask=False)
    rows = ds.meta_rows()
    idx = [i for i, r in enumerate(rows) if r.get("render_mode") == "local"]
    if args.limit:
        idx = idx[: args.limit]
    print(f"{args.split}: {len(ds)} samples, {len(idx)} local", flush=True)

    # -- labels from the construction side (never from text) ---------------
    print("resolving .vrmeta.json labels (one MaskResolver per thread)...", flush=True)
    t0 = time.time()
    local = threading.local()

    def _res() -> MaskResolver:
        r = getattr(local, "res", None)
        if r is None:
            r = local.res = MaskResolver(verify="none", suffix=".vrmeta.json")
        return r

    label_errors: dict[str, int] = {}

    def _label(i: int):
        rec = ds.record(i)
        sid = rec["sample_id"]
        try:
            r = _res()
            vm = json.loads(r.read_bytes(r.resolve(rec)).decode("utf-8"))
            slot_id, region = vm.get("slot_id"), vm.get("region")
            v = geom_features_from_vrmeta(slot_id, region)
            return i, sid, v, str(slot_id or ""), str(region or ""), None
        except Exception as exc:  # noqa: BLE001
            key = f"{type(exc).__name__}: {str(exc)[:100]}"
            label_errors[key] = label_errors.get(key, 0) + 1
            return i, sid, None, "", "", key

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        label_rows = list(ex.map(_label, idx))
    print(f"  {time.time() - t0:.0f}s", flush=True)

    skipped: dict[str, int] = {}
    skipped_ids: dict[str, list[str]] = {}

    def _skip(reason: str, sid: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1
        skipped_ids.setdefault(reason, []).append(sid)

    kept: list[dict[str, Any]] = []
    for i, sid, v, slot_id, region, err in label_rows:
        if v is None:
            _skip(f"vrmeta_lookup_failed::{err}", sid)
            continue
        vv = v[keep_cols]
        if vv[: len(SHAPE_SLOTS)].sum() != 1:
            _skip(f"no_shape_family(slot_id={slot_id!r})", sid)
            continue
        kept.append({"index": i, "sample_id": sid, "label": vv,
                     "slot_id": slot_id, "region": region,
                     "half": split_half(sid)})
    print(f"labels: {len(kept)} usable / {len(idx)} local; skipped {dict(skipped)}",
          flush=True)
    if not kept:
        raise SystemExit("no labelled samples")

    fam_hist: dict[str, int] = {}
    for k in kept:
        fam = str(k["slot_id"]).rsplit("-", 1)[0]
        fam_hist[fam] = fam_hist.get(fam, 0) + 1
    reg_hist: dict[str, int] = {}
    for k in kept:
        reg_hist[k["region"]] = reg_hist.get(k["region"], 0) + 1
    print(f"  families {fam_hist}", flush=True)
    print(f"  regions  {dict(sorted(reg_hist.items(), key=lambda kv: -kv[1]))}",
          flush=True)

    # -- frozen VLM ---------------------------------------------------------
    print(f"loading {args.checkpoint} ({args.dtype}, attn={args.attn})...", flush=True)
    proc = AutoProcessor.from_pretrained(args.checkpoint)
    tok = proc.tokenizer
    model = load_model(args.checkpoint, attn_implementation=args.attn,
                       dtype=args.dtype).to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    collator = Sft2SegCollator(proc, max_length=2048, system_prompt=None)
    lm = resolve_language_model(model)
    pooler = WherePooler(lm)
    layer_keys = pooler.layer_keys()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    close_id = int(tok("</where>", add_special_tokens=False)["input_ids"][0])
    print(f"  n_text_layers={pooler.n_layers}  layer_keys={len(layer_keys)}  "
          f"embed_hook={'yes' if pooler.embed is not None else 'no'}", flush=True)

    genctx = None
    if "generated" in contexts:
        genctx = GenContextStore(Path(GENCTX_DIR) / args.split)
        print(f"genctx {args.split}: {len(genctx.rows)} rows", flush=True)

    # -- read the split ONCE ------------------------------------------------
    # The prompt is identical across context arms (only the <where> token ids
    # differ), so decoding the images and tokenising the prompts a second time
    # for the second arm would be pure NFS + CPU waste on a box already running
    # two trainings.
    print("\nloading samples + prompts (once, shared by all context arms)...",
          flush=True)
    t0 = time.time()
    base: list[dict[str, Any]] = []
    for k in kept:
        s = ds[k["index"]]
        try:
            enc = collator.encode_one(_PromptShim(s))
        except Exception as exc:  # noqa: BLE001
            _skip(f"collator::{type(exc).__name__}", s.sample_id)
            continue
        base.append({"sample_id": s.sample_id, "image": s.image,
                     "prompt_ids": enc["input_ids"][:enc["n_prompt_tokens"]],
                     "where_text": s.where_text, "label": k["label"],
                     "half": k["half"], "slot_id": k["slot_id"],
                     "region": k["region"]})
    print(f"  {len(base)} samples in {time.time() - t0:.0f}s", flush=True)

    # -- feature extraction --------------------------------------------------
    def extract(mode: str) -> dict[str, Any]:
        print(f"\n=== extracting {mode} context features ===", flush=True)
        items: list[dict[str, Any]] = []
        local_skip: dict[str, int] = {}
        fmt_failures = 0
        for b in base:
            sid = b["sample_id"]
            if mode == "gt":
                try:
                    ctx = gt_context(tok, sid, b["where_text"])
                except Exception as exc:  # noqa: BLE001
                    local_skip[f"gt_context::{type(exc).__name__}"] = \
                        local_skip.get(f"gt_context::{type(exc).__name__}", 0) + 1
                    continue
            else:
                try:
                    rec = genctx.record(sid)
                except Exception:  # noqa: BLE001
                    local_skip["genctx_missing"] = local_skip.get("genctx_missing", 0) + 1
                    continue
                ctx = generated_context(sid, rec["generated_ids"], close_id,
                                        text=rec.get("generated_text", ""),
                                        eos_id=tok.eos_token_id)
                fmt_failures += int(ctx.format_failure)
            if not ctx.token_ids:
                local_skip["empty_where_span"] = local_skip.get("empty_where_span", 0) + 1
                continue
            items.append({**b, "where_ids": list(ctx.token_ids),
                          "span_text": tok.decode(ctx.token_ids),
                          "format_failure": bool(ctx.format_failure)})
        for r, c in local_skip.items():
            skipped[f"{mode}::{r}"] = skipped.get(f"{mode}::{r}", 0) + c
        print(f"  {len(items)} samples, skipped {dict(local_skip)}, "
              f"format_failures {fmt_failures}", flush=True)

        feats: dict[str, list[torch.Tensor]] = {k: [] for k in layer_keys}
        pooler.attach()
        t_ex = time.time()
        try:
            for start in range(0, len(items), args.micro_batch):
                chunk = items[start:start + args.micro_batch]
                toks = [it["prompt_ids"] + it["where_ids"] for it in chunk]
                max_len = max(len(t) for t in toks)
                input_ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long)
                attn = torch.zeros((len(chunk), max_len), dtype=torch.long)
                wmask = torch.zeros((len(chunk), max_len), dtype=torch.float32)
                for j, (t, it) in enumerate(zip(toks, chunk)):
                    input_ids[j, : len(t)] = torch.tensor(t, dtype=torch.long)
                    attn[j, : len(t)] = 1
                    n_p, n_w = len(it["prompt_ids"]), len(it["where_ids"])
                    wmask[j, n_p:n_p + n_w] = 1.0
                assert float(wmask.sum(1).min()) > 0
                img = proc.image_processor(images=[it["image"] for it in chunk],
                                          do_resize=False, return_tensors="pt")
                dtype = next(model.parameters()).dtype
                pooler.mask = wmask
                pooler.pooled = {}
                with torch.no_grad():
                    kw = dict(
                        input_ids=input_ids.to(args.device),
                        attention_mask=attn.to(args.device),
                        pixel_values=img["pixel_values"].to(args.device, dtype),
                        image_grid_thw=img["image_grid_thw"].to(args.device),
                    )
                    # model.model skips the lm_head: a (B, T, 151k) logits tensor
                    # is ~1.2 GiB at B=4/T=2048 and nothing here reads it.
                    try:
                        model.model(use_cache=False, **kw)
                    except TypeError:
                        model(**kw)
                missing = [k for k in layer_keys if k not in pooler.pooled]
                if missing:
                    raise RuntimeError(f"no hidden captured for layers {missing[:5]}")
                for kk in layer_keys:
                    feats[kk].append(pooler.pooled[kk])
                del kw
                if (start // args.micro_batch) % 20 == 0:
                    done = min(start + args.micro_batch, len(items))
                    print(f"  {done}/{len(items)}  "
                          f"{time.time() - t_ex:.0f}s  "
                          f"peak {torch.cuda.max_memory_allocated(args.device) / 2**30:.1f}GiB",
                          flush=True)
        finally:
            pooler.detach()
            pooler.pooled = {}
            pooler.mask = None
        X = {k: torch.cat(v, dim=0).numpy().astype(np.float32) for k, v in feats.items()}
        Y = np.stack([it["label"] for it in items]).astype(np.float32)
        half = np.array([it["half"] for it in items], dtype=np.int64)
        print(f"  features {X[layer_keys[0]].shape} x {len(layer_keys)} layers "
              f"({time.time() - t_ex:.0f}s)", flush=True)
        return {"X": X, "Y": Y, "half": half, "items": items,
                "format_failures": fmt_failures}

    # -- text baselines, recomputed on this exact sample set ----------------
    def text_baseline(items, Y, mask) -> dict[str, Any]:
        """Frozen-vocabulary parse of the span text vs the vrmeta code.

        Recomputed here rather than quoted so that the probe and the text are
        scored on the *same* samples, the *same* test half and the *same* 9
        slots -- otherwise "hidden beats text" could be a split artefact.
        """
        P = np.stack([geom_features(it["span_text"]) for it in items])[:, keep_cols]
        return group_metrics(P[mask].astype(np.float64), Y[mask])

    probe_dev = torch.device(args.probe_device or args.device)
    lam_grid = [float(x) for x in args.lam_grid.split(",")]

    results: dict[str, Any] = {}
    per_layer_rows: list[dict[str, Any]] = []

    def _dump_per_layer() -> None:
        with (out_dir / "per_layer.jsonl").open("w", encoding="utf-8") as fh:
            for r in per_layer_rows:
                fh.write(json.dumps(r) + "\n")

    default_threads = torch.get_num_threads()
    for mode in contexts:
        torch.set_num_threads(default_threads)
        data = extract(mode)
        torch.set_num_threads(args.probe_threads)
        X, Y, half = data["X"], data["Y"], data["half"]
        tr, te = half == 0, half == 1
        n_tr, n_te = int(tr.sum()), int(te.sum())
        print(f"\n--- probing {mode}: train {n_tr} / test {n_te} ---", flush=True)
        if n_tr < 20 or n_te < 20:
            raise SystemExit(f"split too small: {n_tr}/{n_te}")

        Ytr_t = torch.from_numpy(Y[tr]).to(probe_dev)
        Yte_np = Y[te]
        rng = np.random.default_rng(args.seed)
        perm_idx = rng.permutation(n_tr)          # label-permutation control

        mode_layers: dict[str, Any] = {}
        for kk in layer_keys:
            Xa = X[kk]
            mu = Xa[tr].mean(axis=0)
            sd = Xa[tr].std(axis=0)
            sd = np.maximum(sd, 1e-6)
            Xs = (Xa - mu) / sd
            Xtr = torch.from_numpy(Xs[tr]).to(probe_dev)
            Xte = torch.from_numpy(Xs[te]).to(probe_dev)

            lam, cv = select_lambda(Xtr, Ytr_t, lam_grid, args.cv_folds,
                                    args.lbfgs_iter, args.seed)
            W, b = _fit(Xtr, Ytr_t, lam, args.lbfgs_iter)
            with torch.no_grad():
                prob = torch.sigmoid(Xte @ W + b).cpu().numpy().astype(np.float64)
                prob_tr = torch.sigmoid(Xtr @ W + b).cpu().numpy().astype(np.float64)
            m = group_metrics(prob, Yte_np)
            m_tr = group_metrics(prob_tr, Y[tr])

            Wp, bp = _fit(Xtr, Ytr_t[torch.from_numpy(perm_idx).to(probe_dev)],
                          lam, args.lbfgs_iter)
            with torch.no_grad():
                prob_p = torch.sigmoid(Xte @ Wp + bp).cpu().numpy().astype(np.float64)
            mp = group_metrics(prob_p, Yte_np)

            row = {
                "context": mode, "layer_key": kk,
                "layer": (pooler.n_layers if kk.endswith("_norm") else int(kk)),
                "post_final_norm": kk.endswith("_norm"),
                "lambda": lam, "cv_bce": cv,
                "test": m, "train": m_tr,
                "perm_control": {k2: mp[k2] for k2 in
                                 ("shape_argmax_acc", "shape_any_overlap",
                                  "shape_exact_set", "dir_exact", "dir_any",
                                  "macro_acc", "macro_f1")},
            }
            mode_layers[kk] = row
            per_layer_rows.append(row)
            _dump_per_layer()          # a late stage must never discard this
            print(f"  L{kk:>8}  lam={lam:<6g} shape_acc={m['shape_argmax_acc']:.4f}  "
                  f"shape_any={m['shape_any_overlap']:.4f}  "
                  f"dir_exact={m['dir_exact']:.4f}  dir_any={m['dir_any']:.4f}  "
                  f"macroF1={m['macro_f1']:.4f}  (perm {mp['shape_argmax_acc']:.4f})",
                  flush=True)

        best_key = max(mode_layers, key=lambda k: mode_layers[k]["test"]["shape_argmax_acc"])
        results[mode] = {
            "n_total": int(Y.shape[0]),
            "n_train": n_tr, "n_test": n_te,
            "format_failures": data["format_failures"],
            "layers": mode_layers,
            "best_layer_key": best_key,
            "best_layer": mode_layers[best_key]["layer"],
            "best_post_final_norm": mode_layers[best_key]["post_final_norm"],
            "best": mode_layers[best_key]["test"],
            "constant_baseline": constant_baseline(Y[tr], Yte_np),
            "text_parse_baseline_test": text_baseline(data["items"], Y, te),
            "text_parse_baseline_all": text_baseline(data["items"], Y,
                                                     np.ones(len(Y), dtype=bool)),
        }
        # per-sample audit at the best layer -- optional, so a failure here must
        # not cost the layer curve that is already in hand
        try:
          with (out_dir / f"per_sample_{mode}.jsonl").open("w", encoding="utf-8") as fh:
            Xa = X[best_key]
            mu, sd = Xa[tr].mean(axis=0), np.maximum(Xa[tr].std(axis=0), 1e-6)
            Xs = torch.from_numpy(((Xa - mu) / sd)[te]).to(probe_dev)
            lam = mode_layers[best_key]["lambda"]
            Xtr = torch.from_numpy(((Xa - mu) / sd)[tr]).to(probe_dev)
            W, b = _fit(Xtr, Ytr_t, lam, args.lbfgs_iter)
            with torch.no_grad():
                pr = torch.sigmoid(Xs @ W + b).cpu().numpy()
            te_items = [it for it, f in zip(data["items"], te) if f]
            for it, p in zip(te_items, pr):
                fh.write(json.dumps({
                    "sample_id": it["sample_id"], "slot_id": it["slot_id"],
                    "region": it["region"], "half": "test",
                    "label": {n: float(v) for n, v in zip(POPULATED, it["label"])},
                    "prob": {n: round(float(v), 4) for n, v in zip(POPULATED, p)},
                    "pred_shape": SHAPE_SLOTS[int(p[:len(SHAPE_SLOTS)].argmax())],
                    "true_shape": SHAPE_SLOTS[int(np.argmax(it["label"][:len(SHAPE_SLOTS)]))],
                    "span_text": it["span_text"][:400],
                    "format_failure": it["format_failure"],
                }) + "\n")
        except Exception as exc:  # noqa: BLE001
            print(f"WARN per-sample audit for {mode} failed: {exc}", flush=True)
        torch.set_num_threads(default_threads)
        del data, X

    # -- deliverables --------------------------------------------------------
    _dump_per_layer()

    TEXT_BASE = {"shape_any_overlap": 0.820, "shape_exact_set": 0.386,
                 "direction_any_overlap": 0.582,
                 "source": "pre-existing measurement of the GENERATED <where> text "
                           "(word-level ~82%), supplied in the task card; "
                           "recomputed on this exact sample/test split as "
                           "text_parse_baseline_* below"}

    verdicts = {}
    for mode in contexts:
        acc = results[mode]["best"]["shape_argmax_acc"]
        if acc > 0.82 + 1e-9:
            v = ("hidden carries the geometry MORE losslessly than the sampled "
                 "text; a hidden-side readout is justified")
        elif abs(acc - 0.82) <= 0.02:
            v = ("hidden ~= text: text sampling is not the loss; the bottleneck "
                 "is downstream")
        else:
            v = "hidden is WORSE than the text; the text route is preferable"
        verdicts[mode] = {"best_layer_shape_argmax_acc": acc,
                          "vs_text_0.820": round(acc - 0.820, 4), "verdict": v}

    metrics = {
        "experiment": "P0 / P0a -- layer-wise linear probe of <where> hidden states",
        "question": ("does the <where> hidden state carry the mask geometry more "
                     "losslessly than the sampled <where> text (82% word-level)?"),
        "split": args.split, "render_mode": "local",
        "n_local": len(idx), "n_labelled": len(kept),
        "split_rule": "sha1(sample_id) % 2 -> 0 train / 1 test",
        "label_source": ".vrmeta.json (construction side): slot_id -> shape family, "
                        "region -> direction slots. No text parsing.",
        "excluded_slots_note": EXCLUDED_NOTE,
        "populated_slots": list(POPULATED),
        "family_histogram": fam_hist, "region_histogram": reg_hist,
        "text_baselines_given": TEXT_BASE,
        "verdict": verdicts,
        "probe": {
            "implementation": "torch nn.Linear-equivalent (W,b) + BCEWithLogitsLoss, "
                              "LBFGS strong-wolfe; sklearn is not installed",
            "objective": "mean_{n,k} BCE + lambda * ||W||_F^2 / (n*k)",
            "lambda_grid": lam_grid, "cv_folds": args.cv_folds,
            "lambda_selected_by": "K-fold CV on the TRAIN half only (test never seen)",
            "standardisation": "per-feature mean/std from the TRAIN half only",
            "lbfgs_max_iter": args.lbfgs_iter,
        },
        "contexts": results,
        "skipped": skipped,
        "skipped_ids": {k: v[:20] for k, v in skipped_ids.items()},
        "label_errors": label_errors,
        "runtime_sec": round(time.time() - t_start, 1),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")

    setup = {
        "argv": sys.argv, "args": vars(args), "seed": args.seed,
        "provenance": _provenance(), "dataset_info": ds_info,
        "python": sys.version, "torch": torch.__version__,
        "n_text_layers": pooler.n_layers, "layer_keys": layer_keys,
        "device": args.device, "attn_implementation": args.attn, "dtype": args.dtype,
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated(args.device) / 2**30, 2)
        if torch.cuda.is_available() else None,
    }
    (out_dir / "config" / "run_setup.json").write_text(json.dumps(setup, indent=1),
                                                       encoding="utf-8")
    print(f"wrote {out_dir / 'metrics.json'}", flush=True)

    # plotting is the last, optional stage: metrics.json and per_layer.jsonl are
    # already on disk, and a matplotlib failure must not discard them
    try:
        _plot(out_dir / "probe_curve.png", results, contexts, pooler.n_layers)
    except Exception as exc:  # noqa: BLE001
        import traceback

        print(f"WARN probe_curve.png failed: {exc}", flush=True)
        traceback.print_exc()

    # -- compact table -------------------------------------------------------
    for mode in contexts:
        print(f"\n================ {mode} context ================", flush=True)
        print(f"{'layer':>8} {'shape_acc':>10} {'shape_any':>10} "
              f"{'dir_exact':>10} {'dir_any':>9} {'macroF1':>8} {'lam':>7}")
        for kk in layer_keys:
            r = results[mode]["layers"][kk]
            m = r["test"]
            print(f"{kk:>8} {m['shape_argmax_acc']:>10.4f} {m['shape_any_overlap']:>10.4f} "
                  f"{m['dir_exact']:>10.4f} {m['dir_any']:>9.4f} "
                  f"{m['macro_f1']:>8.4f} {r['lambda']:>7g}")
        cb = results[mode]["constant_baseline"]
        tb = results[mode]["text_parse_baseline_test"]
        print(f"{'CONST':>8} {cb['shape_argmax_acc']:>10.4f} {cb['shape_any_overlap']:>10.4f} "
              f"{cb['dir_exact']:>10.4f} {cb['dir_any']:>9.4f} {cb['macro_f1']:>8.4f}")
        print(f"{'TEXT':>8} {tb['shape_argmax_acc']:>10.4f} {tb['shape_any_overlap']:>10.4f} "
              f"{tb['dir_exact']:>10.4f} {tb['dir_any']:>9.4f} {tb['macro_f1']:>8.4f}"
              "   <- frozen-vocab parse of the same span, same test half")
        b = results[mode]
        print(f"BEST layer {b['best_layer_key']}: shape_acc={b['best']['shape_argmax_acc']:.4f} "
              f"vs text 0.820 -> {verdicts[mode]['verdict']}")
    print(f"\ndone in {time.time() - t_start:.0f}s -> {out_dir}", flush=True)
    return 0


def _plot(path: Path, results: dict[str, Any], contexts: list[str],
          n_layers: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(contexts), figsize=(7 * len(contexts), 5),
                             squeeze=False)
    for ax, mode in zip(axes[0], contexts):
        layers = results[mode]["layers"]
        keys = [k for k in layers if not layers[k]["post_final_norm"]]
        keys.sort(key=lambda k: layers[k]["layer"])
        xs = [layers[k]["layer"] for k in keys]
        for field, style, label in (
            ("shape_argmax_acc", "-o", "shape argmax acc (4-way)"),
            ("shape_any_overlap", "-s", "shape any-overlap"),
            ("dir_exact", "-^", "direction exact-set"),
            ("dir_any", "-v", "direction any-overlap"),
        ):
            ax.plot(xs, [layers[k]["test"][field] for k in keys], style,
                    ms=3, lw=1.3, label=label)
        ax.plot(xs, [layers[k]["perm_control"]["shape_argmax_acc"] for k in keys],
                ":", color="0.5", lw=1.0, label="shape acc, permuted labels")
        cb = results[mode]["constant_baseline"]
        ax.axhline(0.820, color="crimson", lw=1.6, ls="--",
                   label="text shape any-overlap 0.820")
        ax.axhline(cb["shape_argmax_acc"], color="darkorange", lw=1.0, ls="-.",
                   label=f"const shape acc {cb['shape_argmax_acc']:.3f}")
        ax.axhline(cb["dir_any"], color="olive", lw=1.0, ls="-.",
                   label=f"const dir any {cb['dir_any']:.3f}")
        nk = [k for k in layers if layers[k]["post_final_norm"]]
        if nk:
            ax.plot([n_layers], [layers[nk[0]]["test"]["shape_argmax_acc"]], "*",
                    ms=13, color="black", label="H_where (final RMSNorm)")
        ax.set_xlim(-0.5, n_layers + 0.5)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(np.arange(0.0, 1.01, 0.1))
        ax.set_xlabel("decoder layer index (0 = token embeddings)")
        ax.set_ylabel("test-half accuracy")
        ax.set_title(f"{mode} <where> context  "
                     f"(n_train={results[mode]['n_train']}, "
                     f"n_test={results[mode]['n_test']})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower left", ncol=2)
    fig.suptitle("P0a: layer-wise linear probe of <where> hidden states -> "
                 "vrmeta geometry (9 populated slots; extent excluded)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
