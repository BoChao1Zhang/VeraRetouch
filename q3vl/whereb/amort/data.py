"""Batch construction for PR-AMORT: one frozen forward -> everything both arms need.

What this adds on top of :class:`q3vl.whereb.data.BatchBuilder`

* the **merger output** comes back from the same forward (``want_merger=True``),
  which is the only new thing the VLM has to do for this campaign;
* the **similarity field** is built, softened and squashed with the frozen
  arm-wide constants, and the consumer-side domain assertion is run on the raw
  field before the squash (s-cache contract: producer declares, consumer checks);
* the **orientation/shape words** are parsed from the instruction text;
* the **partner GT mask** (same source image, different edit) is loaded for the
  separation term;
* the **mask family** is fetched from the construction-side ``.vrmeta.json``,
  which is the only surviving copy of that label.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from q3vl.where.upsample import area_resize, luma_guide
from q3vl.whereb.context import ShuffleIndex, gt_context, shuffled_context
from q3vl.whereb.fields import phi_dir_fast
from q3vl.whereb.data import _PromptShim
from q3vl.whereb.hiddens import EncodeItem

from .model import upsample_sim_to_grid
from .simfield import SimFieldNorm, WordEmbedder, gaussian_soften, similarity_field, \
    subject_nouns

__all__ = ["ORIENTATION_WORDS", "word_ids_of", "family_labels", "is_semantic_text",
           "ForeignIndex", "AmortSampleInputs", "AmortBatchBuilder"]


#: Frozen 32-slot vocabulary, parsed from the **instruction text only**.
#: Never from the generated ``<where>`` span: DELTA §7 downgraded its geometry
#: to untrustworthy (local exact-match 0/30, errors biased toward "large centred
#: ellipse"), and NOTES §4 makes "orientation words must be parsed from the
#: instruction" an explicit clause of the conditioning contract.
ORIENTATION_WORDS: tuple[str, ...] = (
    "left", "right", "top", "bottom", "upper", "lower", "middle", "center",
    "centre", "corner", "edge", "side", "background", "foreground", "sky",
    "ground", "horizon", "front", "back", "near", "far", "whole", "entire",
    "overall", "band", "strip", "gradient", "oval", "round", "circular",
    "diagonal", "subject",
)
_WORD_INDEX = {w: i for i, w in enumerate(ORIENTATION_WORDS)}


def word_ids_of(instruction: str) -> list[int]:
    """Indices of the orientation/shape words present in the instruction."""
    words = set(re.findall(r"[a-z]+", (instruction or "").lower()))
    return sorted({_WORD_INDEX[w] for w in words if w in _WORD_INDEX})


def is_semantic_text(where_text: str) -> bool:
    """The binary half of the zero-training type-word router (100.0% on generated).

    Only the semantic/geometric split is consumed.  The four-way split is 74.2%
    and every one of its errors is internal to the three geometric families,
    which share a single path -- so those errors have no consequence, and reading
    the finer label would import 26% noise for nothing.
    """
    from q3vl.whereb.scripts.where_typeword_router import route

    return route(where_text or "")[0] == "semantic"


def family_labels(ds, indices: Sequence[int], workers: int = 32) -> dict[str, str]:
    """``sample_id -> {radial, semantic, band, linear, unknown}``.

    The label survives **only** on the construction side: the split index, the
    ``.rec.json`` and the maskview ``meta`` all drop it, and ``build`` (l1-l6) is
    a production batch code with no relation to geometry (every build contains
    all four families).  Fetched live rather than published, so it cannot drift
    from the construction side (NOTES §5-A conservative default).
    """
    import json
    import threading

    from q3vl.where.maskdata import MaskResolver
    from q3vl.whereb.scripts.mask_type_stats import family_of

    # Defensive: the entrypoint already raises this, but family_labels is the
    # one consumer that actually needs the descriptors, so it must not depend on
    # a caller having done it.  Queue jobs inherit a soft limit of 1024.
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except Exception:  # noqa: BLE001
        pass

    # ONE RESOLVER PER THREAD.  A shared MaskResolver is not thread-safe: it
    # holds sqlite connections to the batch catalogues, and sqlite objects may
    # not cross threads.  Sharing one raises `InterfaceError: bad parameter or
    # other API misuse` on a load-dependent fraction of lookups -- 3/2000 in
    # isolation, but **72% of 75,544** inside the real run, where far more
    # catalogues are open and a VLM is competing for the process.  The failures
    # are unbiased across families, so the surviving histogram looks perfectly
    # normal (30.9/25.9/25.9/17.3 vs the published 30.8/25.5/26.2/17.5) and the
    # only visible symptom is a large `unknown` bucket.
    local = threading.local()

    def _res() -> MaskResolver:
        r = getattr(local, "res", None)
        if r is None:
            r = local.res = MaskResolver(verify="none", suffix=".vrmeta.json")
        return r

    errors: dict[str, int] = {}

    def one(i: int) -> tuple[str, str]:
        rec = ds.record(i)
        try:
            r = _res()
            vm = json.loads(r.read_bytes(r.resolve(rec)).decode("utf-8"))
            return rec["sample_id"], family_of(vm.get("slot_id"))
        except Exception as exc:  # noqa: BLE001
            key = f"{type(exc).__name__}: {str(exc)[:80]}"
            errors[key] = errors.get(key, 0) + 1
            return rec["sample_id"], "unknown"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = dict(ex.map(one, indices))

    # Loud, not silent.  The published census (NOTES §1) resolved 75,544/75,544
    # with zero failures, so any material `unknown` fraction means the lookup is
    # broken -- and because it degrades only the *reporting* path (strata, m_sem,
    # routing confusion) rather than training, nothing else would ever complain.
    n_unknown = sum(1 for v in out.values() if v == "unknown")
    frac = n_unknown / max(1, len(out))
    if frac > 0.05:
        raise RuntimeError(
            f"mask family lookup failed for {n_unknown}/{len(out)} ({frac:.1%}) "
            f"samples; the published census resolved 100%. Top errors: "
            f"{sorted(errors.items(), key=lambda kv: -kv[1])[:3]}"
        )
    if n_unknown:
        print(f"  family lookup: {n_unknown}/{len(out)} unknown "
              f"({frac:.2%}), errors={sorted(errors.items(), key=lambda kv: -kv[1])[:3]}",
              flush=True)
    return out


class ForeignIndex:
    """A cross-image instruction for the empty-mask term.

    Distinct from :class:`ShuffleIndex`, and the difference is the whole point:
    a *shuffled* partner is another edit of the **same** picture (so its subject
    is present, and the right answer is a different mask), whereas a *foreign*
    instruction comes from a **different** picture (so its subject is typically
    absent, and the right answer is no mask at all).  NOTES §7.2 warns explicitly
    that these two must not be merged.

    The subject-noun guard is what keeps the label honest: if the foreign
    instruction happens to name something this image also contains ("the sky"),
    the empty-mask target would be wrong, so such a pairing is rejected and the
    sample simply does not serve as a fake that epoch.
    """

    def __init__(self, rows: Sequence[dict], seed: int = 0):
        self.rows = list(rows)
        self.by_id = {r["sample_id"]: r for r in self.rows}
        self.nouns = {r["sample_id"]: set(subject_nouns(r.get("where", "") or ""))
                      for r in self.rows}
        self.images = {r["sample_id"]: r.get("source_image_id") for r in self.rows}
        self._g = torch.Generator().manual_seed(seed)
        self.n_rejected = 0

    def foreign_for(self, sample_id: str, tries: int = 8) -> dict | None:
        own_img = self.images.get(sample_id)
        own_nouns = self.nouns.get(sample_id, set())
        n = len(self.rows)
        for _ in range(tries):
            j = int(torch.randint(0, n, (1,), generator=self._g))
            cand = self.rows[j]
            cid = cand["sample_id"]
            if self.images.get(cid) == own_img:
                continue
            if self.nouns.get(cid, set()) & own_nouns:
                self.n_rejected += 1
                continue
            return cand
        return None


@dataclass
class AmortSampleInputs:
    sample_id: str
    feat: torch.Tensor                  # (1, 1024, gh, gw)
    sim: torch.Tensor | None            # (1, 1, gh, gw), squashed to [0,1]
    center: torch.Tensor | None         # (1, 1, gh, gw)
    cond_h: torch.Tensor                # (1, T, 2560)
    cond_mask: torch.Tensor             # (1, T)
    word_ids: torch.Tensor
    word_offsets: torch.Tensor
    phi_dir: torch.Tensor               # (P, 71)
    guide_hi: torch.Tensor | None
    gt_low: torch.Tensor                # (gh, gw)
    gt_hi: torch.Tensor | None
    gt_partner_low: torch.Tensor | None
    grid_h: int
    grid_w: int
    is_fake: bool
    family: str
    route_semantic: bool
    geom: Any = None
    meta: dict[str, Any] = field(default_factory=dict)


class AmortBatchBuilder:
    """Samples -> per-sample inputs.  The VLM runs once per micro-batch."""

    def __init__(
        self,
        collator,
        vlm,
        basis,
        *,
        embedder: WordEmbedder,
        norm: SimFieldNorm,
        phi_cfg=None,
        shuffle_index: ShuffleIndex | None = None,
        foreign_index: ForeignIndex | None = None,
        genctx=None,
        families: dict[str, str] | None = None,
        id_to_index: dict[str, int] | None = None,
        dataset=None,
        device: str | torch.device = "cuda",
        want_hi: bool = False,
        use_center_prior_channel: bool = False,
        sep_margin: float = 0.05,
        geom_inject: bool = False,
        geom_shuffle: bool = False,
        geom_source: str = "parsed",
        geom_seed: int = 20260811,
        attn_implementation: str = "",
        checkpoint: str = "",
    ):
        self.collator = collator
        self.tokenizer = collator.tokenizer
        self.vlm = vlm
        self.basis = basis
        self.embedder = embedder
        self.norm = norm
        self.phi_cfg = phi_cfg
        self.shuffle_index = shuffle_index
        self.foreign_index = foreign_index
        self.genctx = genctx
        self.close_id = int(self.tokenizer("</where>", add_special_tokens=False)["input_ids"][0])
        self.eos_id = self.tokenizer.eos_token_id
        self.families = families or {}
        self.id_to_index = id_to_index or {}
        self.dataset = dataset
        self.device = torch.device(device)
        self.want_hi = want_hi
        self.use_center_prior_channel = use_center_prior_channel
        self.domain_reports: list[dict[str, float]] = []
        self._partner_cache: dict[str, torch.Tensor] = {}
        self._nouns_cache: dict[str, set] = {}
        #: must match LossWeights.sep_margin -- the guard below rejects pairs
        #: whose best attainable advantage is under it (review U1)
        self.sep_margin = float(sep_margin)
        self.sep_reject: dict[str, int] = {}
        #: training may fall back when a fixed context is impossible for a
        #: sample; evaluation never may (it must report `uncovered` instead)
        self.allow_context_fallback = False
        self.context_fallbacks: dict[str, int] = {}
        #: consumer-side domain enforcement (review B2)
        self.domain_violations = 0
        self.geom_inject = geom_inject
        #: negative control: permute the slots, preserve the active-bit count
        self.geom_shuffle = geom_shuffle
        #: "parsed" = from the generated <where> text (deployable, ~82% quality)
        #: "vrmeta" = the construction-side ground truth (go/no-go upper bound)
        self.geom_source = geom_source
        self._vrmeta_cache: dict[str, Any] = {}
        self._geom_rng = np.random.default_rng(geom_seed)
        if not getattr(vlm, "want_merger", False):
            raise RuntimeError(
                "AmortBatchBuilder needs the merger output; build the FrozenVLM "
                "with want_merger=True"
            )
        # s-cache contract, producer/consumer halves.  Fails loudly at
        # construction rather than shifting every field in the run by a silent
        # kernel mismatch (measured: eager vs sdpa move the merger output by
        # rel-max 0.12 in bf16).
        self.norm.assert_compatible(attn_implementation, checkpoint)

    # -- helpers ------------------------------------------------------------
    def _partner_gt_low(self, sample_id: str, grid_h: int, grid_w: int,
                        gt_own: torch.Tensor | None = None):
        """The partner GT for the separation term -- **only when the pair is legal**.

        Review U1.  ``ShuffleIndex`` groups by ``(source_image_id, render_mode)``
        and deranges within the group, so a "partner" is another edit of the same
        picture -- which on this data is **usually the same subject with a
        different colour instruction**.  Measured on V_where local (n=382 pairs):

          * **65.2%** of pairs share a subject noun;
          * **15.2%** have ``d(gt_own, gt_partner) < margin`` (58/382), and 9 pairs
            are bit-identical.

        Both are fatal to the term as written:

        1. Applying separation pressure to a same-subject pair **is** building an
           antonym separation loss -- the one thing NOTES §7.2 forbids outright,
           arriving by way of ShuffleIndex rather than by the antonym context.
           It can train the arm to fail its own registered ``antonym_ok`` gate.
        2. The hinge's best achievable advantage is exactly
           ``d(gt_own, gt_partner)``, so when that is below ``margin`` the term
           stays active **at the correct answer** and keeps a constant-magnitude
           gradient pushing away from it.

        So the pair must be legal on both counts, mirroring the guard
        :meth:`ForeignIndex.foreign_for` already applies. Rejections are counted
        and land in ``facts()`` -- otherwise "the separation term did nothing"
        and "the pairs were degenerate" are indistinguishable, which the
        pre-registered failure picture §3-4 explicitly warns about.
        """
        if self.shuffle_index is None or self.dataset is None:
            return None
        pid = self.shuffle_index.partner_of(sample_id)
        if pid is None or pid not in self.id_to_index:
            self.sep_reject["no_partner"] = self.sep_reject.get("no_partner", 0) + 1
            return None
        # (a) subject nouns must be disjoint -- otherwise it is an antonym pair
        own_n = self._nouns_of(sample_id)
        par_n = self._nouns_of(pid)
        if own_n and par_n and (own_n & par_n):
            self.sep_reject["same_subject"] = self.sep_reject.get("same_subject", 0) + 1
            return None
        key = f"{pid}@{grid_h}x{grid_w}"
        if key not in self._partner_cache:
            try:
                ps = self.dataset[self.id_to_index[pid]]
                m = ps.mask_target_hi()
            except Exception:
                self.sep_reject["load_failed"] = self.sep_reject.get("load_failed", 0) + 1
                return None
            self._partner_cache[key] = area_resize(
                m[None, None].float(), (grid_h, grid_w))[0, 0]
        gt_par = self._partner_cache[key].to(self.device)
        # (b) the margin must be attainable at the correct answer
        if gt_own is not None:
            d = float((gt_own.float() - gt_par.float()).abs().mean())
            if d < self.sep_margin:
                self.sep_reject["gt_too_close"] = self.sep_reject.get("gt_too_close", 0) + 1
                return None
        self.sep_reject["accepted"] = self.sep_reject.get("accepted", 0) + 1
        return gt_par

    def _vrmeta_code(self, sample_id: str):
        """GT code from the construction side; cached, thread-local resolver."""
        import json as _json

        from .geomparse import geom_features_from_vrmeta

        if sample_id not in self._vrmeta_cache:
            v = None
            try:
                from q3vl.where.maskdata import MaskResolver

                r = getattr(self, "_vres", None)
                if r is None:
                    r = self._vres = MaskResolver(verify="none",
                                                  suffix=".vrmeta.json")
                idx = self.id_to_index.get(sample_id)
                if idx is not None and self.dataset is not None:
                    vm = _json.loads(
                        r.read_bytes(r.resolve(self.dataset.record(idx))).decode())
                    v = geom_features_from_vrmeta(vm.get("slot_id"), vm.get("region"))
            except Exception:
                v = None
            self._vrmeta_cache[sample_id] = v
        return self._vrmeta_cache[sample_id]

    def _geom_of(self, text: str, sample_id: str = ""):
        """Geometry code for the conditioning; source per `geom_source`."""
        if not self.geom_inject:
            return None
        import numpy as _np

        from .geomparse import GEOM_DIM, geom_features, shuffle_features

        if self.geom_source == "vrmeta":
            v = self._vrmeta_code(sample_id)
            if v is None:
                v = _np.zeros(GEOM_DIM, dtype=_np.float32)
        else:
            v = geom_features(text or "")
        if self.geom_shuffle:
            v = shuffle_features(v, self._geom_rng)
        return torch.from_numpy(v).to(self.device)

    def _nouns_of(self, sample_id: str) -> set[str]:
        if sample_id not in self._nouns_cache:
            row = None
            if self.shuffle_index is not None:
                row = self.shuffle_index.by_id.get(sample_id)
            where = (row or {}).get("where", "") if row else ""
            if not where and sample_id in self.id_to_index and self.dataset is not None:
                try:
                    where = self.dataset.record(self.id_to_index[sample_id]).get("where", "")
                except Exception:
                    where = ""
            self._nouns_cache[sample_id] = set(subject_nouns(where or ""))
        return self._nouns_cache[sample_id]

    def _sim_field(self, res, where_text: str, grid_h: int, grid_w: int):
        nouns = subject_nouns(where_text or "")
        if not nouns or res.f_merger is None:
            return None
        emb = self.embedder(nouns[0]).to(res.f_merger.device)
        raw = similarity_field(res.f_merger, emb, self.norm.kind)
        # consumer-side domain assertion on the RAW field, before any squash
        self.domain_reports.append(self.norm.assert_in_domain(raw))
        soft = gaussian_soften(raw, self.norm.sigma)
        return upsample_sim_to_grid(self.norm.squash(soft), grid_h, grid_w)

    # -- context selection --------------------------------------------------
    def context_for(self, s, mode: str):
        """``(context, instruction, is_fake)`` for one sample.

        ``foreign`` is the empty-mask control and is the only mode that sets
        ``is_fake``; ``shuffled`` swaps to another edit of the **same** image and
        keeps a real (different) target.  Keeping them distinct is a NOTES §7.2
        requirement, not a stylistic choice.
        """
        from q3vl.whereb.context import (antonym_context, fixed_phrase_context,
                                         generated_context, irrelevant_words_context,
                                         null_context)

        if mode == "foreign":
            cand = (self.foreign_index.foreign_for(s.sample_id)
                    if self.foreign_index is not None else None)
            if cand is None:
                return gt_context(self.tokenizer, s.sample_id, s.where_text), \
                    s.instruction, False
            return (shuffled_context(self.tokenizer, cand["sample_id"],
                                     cand["where"], cand["instruction"]),
                    cand["instruction"], True)
        if mode == "gt":
            return gt_context(self.tokenizer, s.sample_id, s.where_text), \
                s.instruction, False
        if mode == "generated":
            if self.genctx is None:
                raise RuntimeError(
                    "generated context requested but no GenContextStore is "
                    "attached; never fall back to GT")
            rec = self.genctx.record(s.sample_id)
            ctx = generated_context(s.sample_id, rec["generated_ids"], self.close_id,
                                    text=rec.get("generated_text", ""),
                                    eos_id=self.eos_id)
            return ctx, s.instruction, False
        if mode == "shuffled":
            if self.shuffle_index is None:
                raise RuntimeError("shuffled context requested but no ShuffleIndex")
            pid = self.shuffle_index.partner_of(s.sample_id)
            if pid is None:
                # EVALUATION must not silently pair across images -- an
                # uncovered sample is reported as uncovered (review B1).  But in
                # TRAINING (`--train-context shuffled`, the ceiling row) the same
                # raise kills the run on the first partnerless sample: ~4.5% of
                # the split has none, so the arm died within a few hundred steps.
                # Fall back to the sample's own context and COUNT it, so the
                # ceiling row can report exactly how much of it was not actually
                # shuffled instead of pretending the condition was uniform.
                if self.allow_context_fallback:
                    self.context_fallbacks["shuffled_no_partner"] = (
                        self.context_fallbacks.get("shuffled_no_partner", 0) + 1)
                    return (gt_context(self.tokenizer, s.sample_id, s.where_text),
                            s.instruction, False)
                raise KeyError(f"{s.sample_id} has no shuffle partner")
            row = self.shuffle_index.by_id[pid]
            return (shuffled_context(self.tokenizer, pid, row["where"],
                                     row["instruction"]),
                    row["instruction"], False)
        if mode == "antonym":
            return antonym_context(self.tokenizer, s.sample_id, s.instruction,
                                   s.where_text), s.instruction, False
        if mode == "fixed_phrase":
            return fixed_phrase_context(self.tokenizer, s.sample_id), \
                s.instruction, False
        if mode == "irrelevant_words":
            return irrelevant_words_context(self.tokenizer, s.sample_id), \
                s.instruction, False
        if mode == "null":
            return null_context(), s.instruction, False
        raise ValueError(f"unknown context mode {mode!r}")

    # -- one micro-batch ----------------------------------------------------
    def build(self, samples: Sequence[Any],
              modes: Sequence[str] | Sequence[bool]) -> list[AmortSampleInputs]:
        # Back-compat: the training loop passes booleans (fake or not).  numpy
        # bools are not `True`, so compare by value rather than identity.
        norm_modes = [m if isinstance(m, str) else ("foreign" if bool(m) else "gt")
                      for m in modes]
        items, ctxs, instructions = [], [], []
        for s, mode in zip(samples, norm_modes):
            ctx, instr, fake = self.context_for(s, mode)
            ctxs.append((ctx, bool(fake)))
            instructions.append(instr)
            enc = self.collator.encode_one(_PromptShim(s, instruction=instr))
            n_p = enc["n_prompt_tokens"]
            items.append(EncodeItem(sample_id=s.sample_id, image=s.image,
                                    prompt_ids=enc["input_ids"][:n_p],
                                    where_ids=ctx.token_ids))

        encoded = self.vlm.encode(items)
        out: list[AmortSampleInputs] = []
        for s, e, (ctx, fake), instr in zip(samples, encoded, ctxs, instructions):
            gh, gw = e.grid_h, e.grid_w
            flat = e.f_pre.reshape(gh * gw, -1)
            with torch.no_grad():
                sem64 = self.basis(flat.to(self.basis.weight.device))
                img = s.image_tensor()
                img_low = area_resize(img.unsqueeze(0), (gh, gw))[0]
                phi = phi_dir_fast(sem64, img_low.to(sem64.device), gh, gw, self.phi_cfg)

            feat = flat.transpose(0, 1).reshape(1, -1, gh, gw).to(self.device)
            # the sim field must follow the CONDITIONING text, not the sample's
            # own: under `shuffled`/`foreign` the subject noun is the partner's,
            # which is exactly what makes those controls informative.
            sim = self._sim_field(e, ctx.text or s.where_text, gh, gw)
            center = None
            if self.use_center_prior_channel:
                from q3vl.whereb.metrics import center_prior_field

                cp = center_prior_field(gh, gw, device=self.device)
                center = ((cp - cp.min()) / (cp.max() - cp.min() + 1e-9)).reshape(1, 1, gh, gw)

            gt_hi = s.mask_target_hi().float()
            gt_low = area_resize(gt_hi[None, None], (gh, gw))[0, 0].to(self.device)
            wid = word_ids_of(instr)
            fam = self.families.get(s.sample_id, "unknown")
            out.append(AmortSampleInputs(
                sample_id=s.sample_id,
                feat=feat,
                sim=None if sim is None else sim.to(self.device),
                center=center,
                cond_h=e.h_where.unsqueeze(0).to(self.device),
                cond_mask=torch.ones(1, max(1, e.h_where.shape[0]),
                                     dtype=torch.bool, device=self.device),
                word_ids=torch.tensor(wid or [0], dtype=torch.long, device=self.device),
                word_offsets=torch.tensor([0], dtype=torch.long, device=self.device),
                phi_dir=phi.to(self.device),
                guide_hi=luma_guide(img.unsqueeze(0)).to(self.device) if self.want_hi else None,
                gt_low=gt_low,
                gt_hi=gt_hi.to(self.device) if self.want_hi else None,
                gt_partner_low=(None if fake else
                                self._partner_gt_low(s.sample_id, gh, gw, gt_low)),
                grid_h=gh, grid_w=gw,
                is_fake=fake,
                family=fam,
                # the router reads the CONDITIONING text, which at deployment is
                # the generated span -- that is the 396/396 binary decision NOTES
                # §3 measured, and evaluating it on GT text would flatter it.
                route_semantic=is_semantic_text(ctx.text or s.where_text),
                geom=self._geom_of(ctx.text or s.where_text, s.sample_id),
                meta={"instruction": instr, "n_words": len(wid),
                      "render_mode": s.meta.get("render_mode"),
                      "build": s.meta.get("build"),
                      "winner_confidence": s.meta.get("winner_confidence")},
            ))
        return out

    def facts(self) -> dict[str, Any]:
        rep = self.domain_reports
        return {
            "norm": self.norm.to_dict(),
            "n_domain_checks": len(rep),
            "max_frac_outside_tol": max((r["frac_outside_tol"] for r in rep), default=0.0),
            "observed_raw_min": min((r["raw_min"] for r in rep), default=None),
            "observed_raw_max": max((r["raw_max"] for r in rep), default=None),
            "foreign_rejected": getattr(self.foreign_index, "n_rejected", 0),
            # review U1: without these, "the separation term did nothing" and
            # "the pairs were degenerate" cannot be told apart
            "sep_pair_disposition": dict(self.sep_reject),
            "sep_usable_fraction": (
                self.sep_reject.get("accepted", 0)
                / max(1, sum(self.sep_reject.values()))),
            "sep_margin": self.sep_margin,
            "context_fallbacks": dict(self.context_fallbacks),
            "domain_violations": self.domain_violations,
        }
