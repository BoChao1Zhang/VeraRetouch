"""PR-AMORT P1 / P3' -- train one arm and publish its board.

    P1       tower -> grid codes -> w(71) -> frozen Phi-71 + guided upsample
    P3prime  tower -> field           -> guided upsample          (no Phi)

Same data, same conditioning, same loss, same criteria; the only difference is
whether the Phi-71 intermediate layer sits in the feed-forward path.  That is
the question the pair exists to answer, and it is open precisely because E5
proved Phi-71 is *not* a bottleneck (it reproduces any field it is handed) --
which says nothing about whether it helps a network that has to *produce* one.

The VLM is frozen throughout and runs once per micro-batch, yielding F_pre, the
merger output and H_where together (protocol 2.3).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _provenance() -> dict:
    """Commit **and** working-tree state.

    Recording a bare HEAD would be misleading here: the arm's own source is not
    committed, so HEAD does not describe the code that ran.  The dirty flag, the
    changed-file list and a content hash of the amort package make the run
    reproducible from the artefact rather than from an assumption.
    """
    repo = "/home/bc/VeraRetouch"
    out = {"git_commit": "unknown", "working_tree_dirty": None,
           "changed_files": [], "amort_source_sha256": None}
    try:
        out["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo).decode().strip()
        st = subprocess.check_output(["git", "status", "--porcelain"],
                                     cwd=repo).decode().strip()
        out["working_tree_dirty"] = bool(st)
        out["changed_files"] = [ln[3:] for ln in st.splitlines()][:80]
    except Exception:
        pass
    try:
        import hashlib

        h = hashlib.sha256()
        for f in sorted(Path(repo, "q3vl/whereb/amort").rglob("*.py")):
            h.update(f.read_bytes())
        for extra in ("q3vl/where/fpre.py", "q3vl/whereb/hiddens.py",
                      "q3vl/whereb/scripts/run_amort_arm.py"):
            h.update(Path(repo, extra).read_bytes())
        out["amort_source_sha256"] = h.hexdigest()
    except Exception:
        pass
    return out


def _git_commit() -> str:
    return _provenance()["git_commit"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=["P1", "P3prime", "SHAPE3"])
    ap.add_argument("--checkpoint",
                    default="/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976")
    ap.add_argument("--basis", default="BA-3-Joint")
    ap.add_argument("--readout", default="band")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--eval-split", default="V_where")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--attn", default="eager",
                    help="the similarity-field norm is only valid under the "
                         "kernel it was fitted with; see simfield.SimFieldNorm")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out-root", default="/home/bc/data/runs/where_b")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--effective-batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-hours", type=float, default=3.6)
    ap.add_argument("--max-steps", type=int, default=1200,
                    help="review U4: the A/B is matched on STEPS, not wall clock; "
                         "both arms must use the same value")
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--eval-limit", type=int, default=None)
    ap.add_argument("--norm-samples", type=int, default=256)
    ap.add_argument("--teacher-fraction", type=float, default=0.5)
    # ablation switches (pre-registered rows; the main arm uses the defaults)
    ap.add_argument("--no-sim-field", action="store_true")
    ap.add_argument("--center-prior-channel", action="store_true")
    ap.add_argument("--no-film", action="store_true")
    ap.add_argument("--geom-inject", action="store_true",
                    help="B2: concat the parsed <where> geometry code as "
                         "broadcast spatial channels (bypasses the pooled path)")
    ap.add_argument("--geom-shuffle", action="store_true",
                    help="B2 negative control: permute the geometry slots, "
                         "preserving the active-bit count")
    ap.add_argument("--geom-source", default="parsed",
                    choices=["parsed", "vrmeta"],
                    help="parsed = generated <where> text (~82%%); vrmeta = "
                         "construction-side GT code (go/no-go upper bound)")
    ap.add_argument("--geom-mode", default="broadcast", choices=["broadcast", "pch"],
                    help="broadcast = 21 constant channels on the stem (lower "
                         "bound form, NOT resumable); pch = zero-initialised "
                         "prototype/cross-attention residual on the tower's "
                         "penultimate features (proposal D-1, resumable)")
    ap.add_argument("--pch-size", default="full", choices=["full", "lite"],
                    help="PCH capacity档: full ~3.9M, lite ~0.5M (G6 ablation)")
    ap.add_argument("--freeze-base", action="store_true",
                    help="train ONLY the injector (proposal §2.4).  This is what "
                         "makes M0 exactly the resumed checkpoint's own board: "
                         "with the backbone frozen no part of Delta can be "
                         "continued training rather than injection")
    ap.add_argument("--pooled-w", action="store_true",
                    help="registered control arm: pool to one vector before w(71)")
    ap.add_argument("--sdf-weight", type=float, default=None,
                    help="P2 shaped-weight recalibration (pre-registered default 0.10)")
    ap.add_argument("--area-weight", type=float, default=None,
                    help="P2 shaped-weight recalibration (pre-registered default 0.05)")
    ap.add_argument("--eik-weight", type=float, default=0.0,
                    help="SHAPE3: soft eikonal |grad s|=1 on the s field")
    ap.add_argument("--curv-weight", type=float, default=0.0,
                    help="P2 structural pack: excess-curvature penalty")
    ap.add_argument("--mono-weight", type=float, default=0.0,
                    help="P2 structural pack: monotonicity hinge")
    ap.add_argument("--sep-weight", type=float, default=None,
                    help="override the pre-registered separation weight (0.30)")
    ap.add_argument("--resume", default=None,
                    help="checkpoint to continue from (weights only; a fresh "
                         "schedule is started and recorded as such)")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training and re-score --resume's checkpoint "
                         "(rebuilds a board that was missing a criterion)")
    ap.add_argument("--no-semantic-head", action="store_true")
    ap.add_argument("--train-context", default="mixed",
                    choices=["mixed", "gt", "shuffled", "fixed_phrase"],
                    help="'shuffled'/'fixed_phrase' are the training-ceiling rows")
    ap.add_argument("--quick-eval-limit", type=int, default=200)
    args = ap.parse_args(argv)

    t_start = time.time()

    # ---------------------------------------------------------------------
    # Raise RLIMIT_NOFILE before anything opens a file.
    # ---------------------------------------------------------------------
    # Queue-run jobs inherit pueued's soft limit of **1024**, while an
    # interactive shell here gets 1048576.  That difference is invisible until
    # it bites: this arm opens sqlite catalogues for 35 dataset batches from
    # several worker threads on top of a loaded VLM and three published stores,
    # and at 1024 it exhausts descriptors -- surfacing as sqlite
    # "unable to open database file" (55,512 lookups) plus Errno 24, i.e. an
    # 83.7% mask-family failure that reproduces *not at all* when the same code
    # is run by hand.  The process genuinely needs the descriptors, so it asks
    # for them rather than tiptoeing around the limit.
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f"RLIMIT_NOFILE {soft} -> {resource.getrlimit(resource.RLIMIT_NOFILE)[0]}",
              flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"WARN could not raise RLIMIT_NOFILE: {exc}", flush=True)

    from transformers import AutoProcessor

    from q3vl.train.collator import Sft2SegCollator
    from q3vl.train.modeling import load_model
    from q3vl.whereb.amort.data import (AmortBatchBuilder, ForeignIndex,
                                        family_labels)
    from q3vl.whereb.amort.evaluate import evaluate_arm
    from q3vl.whereb.amort.losses import LossWeights
    from q3vl.whereb.amort.model import AmortModel
    from q3vl.whereb.amort.resume import load_resumable
    from q3vl.whereb.amort.simfield import (SimFieldNorm, WordEmbedder,
                                            similarity_field, subject_nouns)
    from q3vl.whereb.amort.trainer import AmortTrainConfig, AmortTrainer
    from q3vl.whereb.context import ShuffleIndex
    from q3vl.whereb.data import open_dataset
    from q3vl.whereb.fields import load_basis
    from q3vl.whereb.hiddens import EncodeItem, FrozenVLM
    from q3vl.whereb.stores import GenContextStore
    from q3vl.whereb.config import GENCTX_DIR

    run_name = args.run_name or f"amort_{args.arm}"
    run_dir = Path(args.out_root) / run_name
    (run_dir / "config").mkdir(parents=True, exist_ok=True)
    print(f"run_dir {run_dir}", flush=True)

    # -- frozen model ------------------------------------------------------
    proc = AutoProcessor.from_pretrained(args.checkpoint)
    model_vlm = load_model(args.checkpoint, attn_implementation=args.attn,
                           dtype=args.dtype).to(args.device).eval()
    vlm = FrozenVLM(model_vlm, proc, device=args.device, want_merger=True)
    collator = Sft2SegCollator(proc, max_length=2048, system_prompt=None)
    basis = load_basis(args.basis).to(args.device)
    embedder = WordEmbedder.from_checkpoint(args.checkpoint, proc.tokenizer)

    # -- data --------------------------------------------------------------
    # Review U7 / data discipline: `winner_confidence == "low"` must not enter
    # the SFT main training set.  It stays in the EVAL split so the stratum can
    # be reported, but the headline there is normal-only (see evaluate.py).
    train_ds, train_info = open_dataset(args.train_split, need_mask=True,
                                        exclude_low=True)
    eval_ds, eval_info = open_dataset(args.eval_split, need_mask=True)
    tr_rows = train_ds.meta_rows()
    ev_rows = eval_ds.meta_rows()
    # local only; global has zero part in training (registered ruling)
    train_idx = [i for i, r in enumerate(tr_rows) if r.get("render_mode") == "local"]
    eval_idx = [i for i, r in enumerate(ev_rows) if r.get("render_mode") == "local"]
    if args.train_limit:
        train_idx = train_idx[: args.train_limit]
    if args.eval_limit:
        eval_idx = eval_idx[: args.eval_limit]
    print(f"train local {len(train_idx)}  eval local {len(eval_idx)}", flush=True)

    tr_shuffle_rows = train_ds.shuffle_records()
    ev_shuffle_rows = eval_ds.shuffle_records()
    train_shuffle = ShuffleIndex(tr_shuffle_rows, seed=0)
    eval_shuffle = ShuffleIndex(ev_shuffle_rows, seed=0)
    foreign = ForeignIndex(tr_shuffle_rows, seed=args.seed)
    def _genctx(split: str):
        try:
            st = GenContextStore(Path(GENCTX_DIR) / split)
            print(f"genctx {split}: {len(st.rows)} rows", flush=True)
            return st
        except Exception as exc:  # noqa: BLE001
            print(f"WARN no genctx for {split}: {exc}", flush=True)
            return None

    genctx = _genctx(args.eval_split)
    genctx_train = _genctx(args.train_split)

    print("fetching mask families (construction-side .vrmeta.json)...", flush=True)
    t0 = time.time()
    fam_train = family_labels(train_ds, train_idx)
    fam_eval = family_labels(eval_ds, eval_idx)
    fam_hist: dict[str, int] = {}
    for v in fam_train.values():
        fam_hist[v] = fam_hist.get(v, 0) + 1
    print(f"  {time.time()-t0:.0f}s  train families {fam_hist}", flush=True)

    id_to_index_tr = {train_ds.record(i)["sample_id"]: i for i in train_idx}
    id_to_index_ev = {eval_ds.record(i)["sample_id"]: i for i in eval_idx}

    # -- fit the arm-wide similarity normalisation -------------------------
    # Arm constants, over every valid cell of a sample of the training split.
    # Never per image (s-cache contract red line).
    print(f"fitting similarity-field norm on {args.norm_samples} samples...", flush=True)
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(train_idx), size=min(args.norm_samples, len(train_idx)),
                      replace=False)
    pool: list[np.ndarray] = []
    for start in range(0, len(pick), 8):
        chunk = [train_idx[int(j)] for j in pick[start:start + 8]]
        samples = [train_ds[i] for i in chunk]
        items = []
        for s in samples:
            from q3vl.whereb.data import _PromptShim

            enc = collator.encode_one(_PromptShim(s, instruction=s.instruction))
            n_p = enc["n_prompt_tokens"]
            items.append(EncodeItem(sample_id=s.sample_id, image=s.image,
                                    prompt_ids=enc["input_ids"][:n_p], where_ids=()))
        for s, e in zip(samples, vlm.encode(items)):
            nouns = subject_nouns(s.where_text or "")
            if not nouns or e.f_merger is None:
                continue
            emb = embedder(nouns[0]).to(e.f_merger.device)
            pool.append(similarity_field(e.f_merger, emb, "dot").cpu().numpy())
    from q3vl.whereb.amort.simfield import fit_norm

    norm = fit_norm(pool, kind="dot", gain=1.0, sigma=1.0,
                    attn_implementation=args.attn, checkpoint=args.checkpoint,
                    n_samples=len(pool))
    print(f"  norm center={norm.center:.4f} scale={norm.scale:.4f} "
          f"domain=[{norm.raw_min:.3f}, {norm.raw_max:.3f}] "
          f"cells={norm.n_cells} ({time.time()-t0:.0f}s)", flush=True)

    # -- model -------------------------------------------------------------
    model = AmortModel(
        args.arm, readout=args.readout,
        use_sim_field=not args.no_sim_field,
        use_center_prior_channel=args.center_prior_channel,
        with_semantic=not args.no_semantic_head,
        use_film=not args.no_film,
        pooled_w=args.pooled_w,
        geom_inject=args.geom_inject,
        geom_mode=args.geom_mode,
        pch_size=args.pch_size,
        seed=args.seed,
    ).to(args.device)
    if args.resume:
        sd = torch.load(args.resume, map_location=args.device)
        report = load_resumable(model, sd["model"])
        print(f"resumed weights from {args.resume} (step {sd.get('step')}); "
              "a NEW schedule starts here and run_setup records it", flush=True)
        print(f"  resume: {json.dumps(report)}", flush=True)
    if args.freeze_base:
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("pch.")
        n_pch = sum(p.numel() for n, p in model.named_parameters()
                    if n.startswith("pch."))
        if not n_pch:
            raise SystemExit("--freeze-base with no injector: nothing would train")
        print(f"FROZEN BASE: only the injector trains ({n_pch:,} params); "
              "M0 is therefore the resumed checkpoint itself", flush=True)
    print(f"trainable params {model.n_trainable():,}", flush=True)

    _kw = {}
    if args.sep_weight is not None:
        _kw["sep"] = float(args.sep_weight)
    if args.curv_weight:
        _kw["curv"] = float(args.curv_weight)
    if args.mono_weight:
        _kw["mono"] = float(args.mono_weight)
    if args.eik_weight:
        _kw["eik"] = float(args.eik_weight)
    if args.sdf_weight is not None:
        _kw["sdf"] = float(args.sdf_weight)
    if args.area_weight is not None:
        _kw["area"] = float(args.area_weight)
    weights_pre = LossWeights(**_kw)
    common = dict(embedder=embedder, norm=norm, device=args.device,
                  attn_implementation=args.attn, checkpoint=args.checkpoint,
                  use_center_prior_channel=args.center_prior_channel,
                  # review U1: the pair guard must use the same margin the hinge
                  # does, or it would reject the wrong set of pairs
                  sep_margin=weights_pre.sep_margin,
                  geom_inject=args.geom_inject,
                  geom_shuffle=args.geom_shuffle,
                  geom_source=args.geom_source)
    train_builder = AmortBatchBuilder(
        collator, vlm, basis, shuffle_index=train_shuffle, foreign_index=foreign,
        genctx=genctx_train, families=fam_train, id_to_index=id_to_index_tr,
        dataset=train_ds, **common)
    eval_builder = AmortBatchBuilder(
        collator, vlm, basis, shuffle_index=eval_shuffle, foreign_index=None,
        genctx=genctx, families=fam_eval, id_to_index=id_to_index_ev,
        dataset=eval_ds, **common)

    weights = weights_pre
    cfg = AmortTrainConfig(
        arm=args.arm, learning_rate=args.lr, effective_batch=args.effective_batch,
        micro_batch=args.micro_batch, epochs=args.epochs, eval_steps=args.eval_steps,
        save_steps=args.save_steps, seed=args.seed, max_hours=args.max_hours,
        teacher_fraction=args.teacher_fraction,
        max_steps=args.max_steps,
        train_context=args.train_context,
    )

    # Review N4: quick eval drives `best()` once U3 is wired, so it must not be
    # the set the arm is reported on.  Split V_where local deterministically by
    # sha1(sample_id) -- a rule of the same family as the campaign's splits, not
    # an index prefix -- and select on one half, report the other separately.
    import hashlib

    def _half(i: int) -> int:
        sid = eval_ds.record(i)["sample_id"]
        return int(hashlib.sha1(sid.encode()).hexdigest(), 16) % 2

    sel_idx = [i for i in eval_idx if _half(i) == 0]
    hold_idx = [i for i in eval_idx if _half(i) == 1]
    quick_idx = sel_idx[: args.quick_eval_limit]
    print(f"selection half {len(sel_idx)}  holdout half {len(hold_idx)}", flush=True)

    def quick_eval(step: int) -> dict[str, Any]:
        board = evaluate_arm(model, eval_builder, eval_ds, quick_idx,
                             quick=True, batch_size=8)
        model.train()
        return {"local_soft_iou_median": board.get("local_soft_iou_median"),
                "gate_pass": board.get("gate_pass"),
                "gate": board.get("gate"),
                "swap_subject_delta": board.get("swap_subject_delta", {}).get("delta"),
                "antonym_median_abs_delta":
                    board.get("antonym_invariance", {}).get("median_abs_delta"),
                "corr_center_minus_gt":
                    board["contexts"]["gt"]["corr_center_minus_corr_gt"]["delta"]
                    if "gt" in board["contexts"] else None}

    # only the TRAINING builder may fall back; the eval builder must keep
    # reporting `uncovered` (review B1)
    train_builder.allow_context_fallback = True
    trainer = AmortTrainer(model, train_builder, train_ds, train_idx, cfg, weights,
                           run_dir=run_dir, device=args.device, eval_fn=quick_eval)

    setup = {**trainer.setup(), "provenance": _provenance(),
             "args": vars(args), "train_info": train_info, "eval_info": eval_info,
             "train_family_hist": fam_hist,
             "shuffle_coverage_train": train_shuffle.coverage(),
             "shuffle_coverage_eval": eval_shuffle.coverage(),
             "sim_norm": norm.to_dict(),
             "resumed_from": args.resume,
             "vlm": vlm.facts()}
    (run_dir / "config" / "run_setup.json").write_text(json.dumps(setup, indent=2, default=str),
                                                       encoding="utf-8")
    (run_dir / "config" / "loss_preregistration.json").write_text(
        json.dumps({"weights": weights.to_dict(),
                    "form": "L = bce*BCE_soft + sdf*SDF_boundary + area*area_band"
                            " + fake*empty_mask + sep*paired_separation",
                    "separation": "relu(margin - (d(m,gt_partner) - d(m,gt_own)))",
                    "antonym": "reported only; never a loss term (E3 erratum)",
                    "iou_as_target": False, "dice_as_target": False,
                    "seed": args.seed, "provenance": _provenance()}, indent=2),
        encoding="utf-8")
    print(json.dumps({k: setup[k] for k in ("arm", "total_optimizer_steps",
                                            "n_train_samples")}), flush=True)

    if args.eval_only:
        # Re-score an existing checkpoint without touching its weights.  Used
        # when a board has to be rebuilt because a pre-registered criterion was
        # missing from it -- the arm is NOT retrained, so the numbers stay
        # comparable to the ones already published.
        print("EVAL ONLY: no training; re-scoring the resumed checkpoint",
              flush=True)
        final_dir = run_dir / "eval_final"
        board = evaluate_arm(model, eval_builder, eval_ds, eval_idx,
                             batch_size=8, want_hi=False, out_dir=final_dir,
                             progress=True)
        board["arm"] = args.arm
        board["eval_only"] = True
        board["resumed_from"] = args.resume
        board["builder_facts"] = train_builder.facts()
        board["elapsed_hours"] = round((time.time() - t_start) / 3600.0, 3)
        (final_dir / "metrics.json").write_text(
            json.dumps(board, indent=2, default=str), encoding="utf-8")
        print(json.dumps({
            "eval_only": True,
            "topk_iou_median_normal_only": board.get("topk_iou_median_normal_only"),
            "shape_residual": board.get("criteria_columns", {})
                                   .get("shape_residual", {}).get("median"),
            "shape_residual_gt": board.get("criteria_columns", {})
                                      .get("shape_residual", {})
                                      .get("gt_control", {}).get("median"),
        }), flush=True)
        return 0

    state = trainer.train()

    # -- checkpoint selection (review U3) ----------------------------------
    # Pre-registration: all three hard gates must pass to be selectable, then
    # median top-k IoU decides.  Never a val loss.  If nothing passes, that is
    # the finding -- the arm is reported as having no selectable checkpoint
    # rather than silently delivering whatever the wall clock happened to stop on.
    best = trainer.best()
    selection = {"selected": None, "reason": "", "n_checkpoints": len(state.checkpoints),
                 "n_gate_pass": sum(1 for c in state.checkpoints if c.get("gate_pass"))}
    if best is not None:
        ckpt = run_dir / f"amort_step{best['step']}.pt"
        if not ckpt.exists() and best["step"] == state.step:
            # The last eval record is written at the final step, but the last
            # *save* is named `amort_final.pt` -- `amort_step{final}.pt` only
            # exists when the final step happens to be a multiple of save_steps.
            # Without this the run reports "no selectable checkpoint" even
            # though every checkpoint passed its gates (observed: 3/3 gate-pass,
            # selected None).  Harmless to the numbers here, because the final
            # state and the selected checkpoint are the same weights -- but the
            # record would have understated a passing arm.
            ckpt = run_dir / "amort_final.pt"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=args.device)["model"])
            selection.update(selected=best["step"], reason="gate_pass + best median top-k IoU",
                             selected_metric=best.get("local_soft_iou_median"))
            print(f"selected checkpoint step{best['step']} "
                  f"(top-k IoU {best.get('local_soft_iou_median')})", flush=True)
        else:
            selection.update(reason=f"gate-passing step{best['step']} but {ckpt} missing")
    else:
        selection.update(reason="NO SELECTABLE CHECKPOINT: no checkpoint passed all "
                                "three pre-registered hard gates")
        print("WARNING no checkpoint passed the hard gates; reporting the final "
              "state and marking the arm as not having cleared its gates", flush=True)

    # -- final board -------------------------------------------------------
    print("final evaluation...", flush=True)
    final_dir = run_dir / "eval_final"
    board = evaluate_arm(model, eval_builder, eval_ds, eval_idx,
                         batch_size=8, want_hi=False, out_dir=final_dir,
                         progress=True)
    board["arm"] = args.arm
    board["checkpoint_selection"] = selection
    board["steps_trained"] = state.step
    board["max_steps_configured"] = args.max_steps
    board["stopped_reason"] = state.stopped_reason
    board["warnings"] = state.warnings
    board["builder_facts"] = train_builder.facts()
    board["elapsed_hours"] = round((time.time() - t_start) / 3600.0, 3)
    try:
        hold = evaluate_arm(model, eval_builder, eval_ds, hold_idx,
                            contexts=("gt", "shuffled"), batch_size=8,
                            want_hi=False)
        hold.pop("_rows", None)
        board["holdout_half"] = {
            "n": len(hold_idx),
            "topk_iou_median": hold.get("topk_iou_median"),
            "gate": hold.get("gate"),
            "note": "sha1-disjoint from the half used for checkpoint selection",
        }
    except Exception as exc:  # noqa: BLE001
        board["holdout_half"] = {"error": f"{type(exc).__name__}: {exc}"}
    rows = board.pop("_rows", {})
    (final_dir / "metrics.json").write_text(json.dumps(board, indent=2, default=str),
                                            encoding="utf-8")
    print(json.dumps({"arm": args.arm,
                      "local_soft_iou_median": board.get("local_soft_iou_median"),
                      "gate": board.get("gate"),
                      "steps": state.step,
                      "hours": board["elapsed_hours"]}, indent=2), flush=True)

    # visualisation (fixed 0..1 colour scale, valid cells only, exact inverse map)
    try:
        from q3vl.whereb.amort.viz import write_panels

        write_panels(model, eval_builder, eval_ds, eval_idx, rows,
                     run_dir / "viz", n_each=6)
    except Exception as exc:  # noqa: BLE001
        print(f"WARN viz failed: {type(exc).__name__}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
