"""Protocol 14, items 7/8/9 -- the Where-B share of the mandatory preflight.

    7. the GT / generated / null / shuffled context data flows do not cross;
    8. prove that ``Q_where`` does not read ``H_color`` (and ``Q_color`` does not
       read ``H_where`` -- that half belongs to Stage-What);
    9. prove that no main arm's input contains ``I_tar``, a GT mask, a GT LUT or
       an oracle latent.

Plus one repo-specific check that the two-context design lives or dies on:

    7b. the teacher and the generated context go through the *same* hidden-state
        contract.  Feeding the GT ``<where>`` ids through the generated path must
        reproduce the teacher hidden states bit for bit; if it does not, every
        "generated context is worse" number is uninterpretable.

Item 8's proof has two independent halves and both are checked:

* **structural** -- no function in the model path has a parameter, and no module
  in the model path has an identifier, containing ``color``;
* **numerical** -- attention is causal, so ``H_where`` is bit-identical whether
  or not a ``<color>`` segment follows.  That is checked on a real forward when
  a model is supplied.

Every check returns ``id / status / detail``; :func:`run_where_b_preflight`
writes the whole thing to JSON and exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import subprocess
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch

from q3vl.train.constants import COLOR_CLOSE, COLOR_OPEN, WHERE_CLOSE, WHERE_OPEN

from . import connector as _connector_mod
from . import heads as _heads_mod
from . import model as _model_mod
from . import qwhere as _qwhere_mod
from .config import ARM_IDS, MODEL_DIR, REPORT_DIR, WHERE_CONTEXT_MAX_TOKENS, arm_config
from .context import (
    BalancedContextSampler,
    ShuffleIndex,
    generated_context,
    gt_context,
    null_context,
    shuffled_context,
)
from .data import META_KEYS
from .model import MODEL_INPUT_KEYS, WhereBModel

__all__ = ["Check", "PreflightReport", "run_where_b_preflight"]

MODEL_PATH_MODULES = (_qwhere_mod, _connector_mod, _heads_mod, _model_mod)


@dataclass
class Check:
    id: str
    status: str                       # pass | fail | skip
    detail: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "status": self.status, "message": self.message,
                "detail": self.detail}


#: every check id this preflight is *required* to produce a row for.  A missing
#: id is a failure, not a silent pass (review blocker B1): the previous version
#: reported ``ok=True, n_skip=0`` while the two model checks had never been
#: constructed, so an operator reading "preflight PASS" would conclude that
#: protocol 14 items 7b/8b had run when nothing had.
REQUIRED_CHECKS: tuple[str, ...] = (
    "WB-P7-context-flows",
    "WB-P8-no-h-color",
    "WB-P9-no-target-leak",
    "WB-P-zero-init-gates",
    "WB-P-param-table",
    "WB-P7b-hidden-contract",
    "WB-P8b-h-where-causal-independence",
)
#: the subset that a real VLM forward is the only way to satisfy
MODEL_CHECKS: tuple[str, ...] = (
    "WB-P7b-hidden-contract",
    "WB-P8b-h-where-causal-independence",
)


@dataclass
class PreflightReport:
    checks: list[Check] = field(default_factory=list)
    env: dict[str, Any] = field(default_factory=dict)
    required: tuple[str, ...] = REQUIRED_CHECKS

    def add(self, c: Check) -> Check:
        self.checks.append(c)
        return c

    @property
    def ids(self) -> set[str]:
        return {c.id for c in self.checks}

    @property
    def missing(self) -> list[str]:
        return [c for c in self.required if c not in self.ids]

    @property
    def ok(self) -> bool:
        """No failure **and** no missing required row.

        ``skip`` rows still count as present: a skip is an explicit, visible
        statement that a check did not run, which is exactly what was absent
        before.  ``complete`` below is the stronger flag that gates S5.
        """
        return not self.missing and all(c.status != "fail" for c in self.checks)

    @property
    def complete(self) -> bool:
        """Every required check present and actually executed (no skips)."""
        return not self.missing and all(c.status == "pass" for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        skipped = [c.id for c in self.checks if c.status == "skip"]
        return {
            "ok": self.ok,
            "complete": self.complete,
            "n_pass": sum(c.status == "pass" for c in self.checks),
            "n_fail": sum(c.status == "fail" for c in self.checks),
            "n_skip": len(skipped),
            "skipped": skipped,
            "missing_required": self.missing,
            "required": list(self.required),
            "env": self.env,
            "checks": [c.to_dict() for c in self.checks],
        }


def _env() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[2],
        ).stdout.strip()
    except Exception:                                   # noqa: BLE001
        commit = ""
    return {
        "git_commit": commit, "python": platform.python_version(),
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
    }


# --- item 7: the four context flows ----------------------------------------

def check_context_flows(tokenizer) -> Check:
    bad: list[str] = []
    detail: dict[str, Any] = {}

    where_a = "subject: the jellyfish; edit scope: stays within the jellyfish"
    where_b = "subject: the church and hillside; edit scope: a vertical band"
    instr_a = "Please make the jellyfish darker and cooler."
    instr_b = "Please warm the church and the hillside behind it."
    close_id = int(tokenizer(WHERE_CLOSE, add_special_tokens=False)["input_ids"][0])

    gt = gt_context(tokenizer, "sA", where_a)
    if gt.provenance != "sA" or gt.format_failure:
        bad.append("gt context is not self-provenanced / flagged a failure")

    nul = null_context()
    if nul.n_tokens != 0:
        bad.append("null context carries tokens")

    partner_records = [
        {"sample_id": "sA", "source_image_id": "img1", "render_mode": "local",
         "where": where_a, "instruction": instr_a},
        {"sample_id": "sB", "source_image_id": "img1", "render_mode": "local",
         "where": where_b, "instruction": instr_b},
        {"sample_id": "sC", "source_image_id": "img2", "render_mode": "local",
         "where": where_a, "instruction": instr_a},
    ]
    shuf = ShuffleIndex(partner_records, seed=0)
    if shuf.partner_of("sA") != "sB" or shuf.partner_of("sB") != "sA":
        bad.append("shuffle partners are not a derangement inside the image group")
    if shuf.partner_of("sC") is not None:
        bad.append("a singleton group was given a partner (cross-image leak)")
    sh = shuffled_context(tokenizer, "sB", where_b, instr_b)
    if sh.provenance == "sA" or sh.token_ids == gt.token_ids:
        bad.append("shuffled context reused the sample's own where text")
    # protocol 5.4 swaps instruction AND where context (review blocker B3)
    if sh.instruction != instr_b:
        bad.append("shuffled context did not swap the instruction")
    if gt.instruction is not None or nul.instruction is not None:
        bad.append("a non-shuffled context carries an instruction override")
    try:
        shuffled_context(tokenizer, "sB", where_b, "")
        bad.append("shuffled context accepted an empty partner instruction")
    except ValueError:
        detail["empty_partner_instruction_refused"] = True
    try:
        ShuffleIndex([{"sample_id": "x", "source_image_id": "i",
                       "render_mode": "local", "where": where_a}], seed=0)
        bad.append("ShuffleIndex accepted a record with no instruction")
    except ValueError:
        detail["shuffle_record_requires_instruction"] = True

    # generated: closed
    closed_ids = gt.token_ids + [999, 1000]
    g_ok = generated_context("sA", closed_ids, close_id)
    if g_ok.token_ids != gt.token_ids or g_ok.format_failure:
        bad.append("a properly closed generation was not cut at </where>")

    # generated: never closed -> fixed boundary, flagged, and NOT the GT
    runaway = [7] * (WHERE_CONTEXT_MAX_TOKENS + 40)
    g_bad = generated_context("sA", runaway, close_id)
    if len(g_bad.token_ids) != WHERE_CONTEXT_MAX_TOKENS:
        bad.append("an unclosed generation was not cut at the fixed boundary")
    if not g_bad.format_failure or g_bad.stop_reason != "no_close_tag":
        bad.append("an unclosed generation was not recorded as a format failure")
    if g_bad.token_ids == gt.token_ids:
        bad.append("FALLBACK TO GT: an unclosed generation returned the GT span")
    sig = inspect.signature(generated_context)
    if any("gt" in p or "teacher" in p or "where_text" == p for p in sig.parameters):
        bad.append(f"generated_context can reach GT text: {list(sig.parameters)}")

    # the 50/50 mix
    sampler = BalancedContextSampler(64, micro_batch=4, seed=0)
    seen: set[int] = set()
    for batch in sampler:
        modes = [m for _, m in batch]
        if modes.count("gt") != 2 or modes.count("generated") != 2:
            bad.append(f"a micro-batch is not 50/50: {modes}")
            break
        for i, _ in batch:
            if i in seen:
                bad.append(f"sample {i} appears in two micro-batches")
            seen.add(i)
    if set(sampler.teacher_pool) & set(sampler.generated_pool):
        bad.append("teacher and generated pools overlap")

    detail.update({
        "gt_tokens": gt.n_tokens, "null_tokens": nul.n_tokens,
        "shuffled_provenance": sh.provenance,
        "shuffled_swaps_instruction": sh.instruction == instr_b,
        "generated_closed": g_ok.to_dict(), "generated_unclosed": g_bad.to_dict(),
        "boundary": WHERE_CONTEXT_MAX_TOKENS,
        "sampler": sampler.facts(), "shuffle": shuf.coverage(),
        "generated_context_signature": list(sig.parameters),
    })
    return Check("WB-P7-context-flows", "fail" if bad else "pass", detail, "; ".join(bad[:4]))


# --- item 8: Q_where must not read H_color ---------------------------------

def _code_identifiers(path: Path) -> set[str]:
    """Identifiers in a module's *code*, with comments and strings removed."""
    out: set[str] = set()
    with path.open("rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type == tokenize.NAME:
                out.add(tok.string)
    return out


def check_no_h_color(model: WhereBModel | None = None) -> Check:
    bad: list[str] = []
    detail: dict[str, Any] = {}

    sigs = {
        "WhereBModel.forward": list(inspect.signature(WhereBModel.forward).parameters),
        "ConnectorStream.forward": list(
            inspect.signature(_connector_mod.ConnectorStream.forward).parameters),
        "ConnectorBlock.forward": list(
            inspect.signature(_connector_mod.ConnectorBlock.forward).parameters),
        "LatentHeads.forward": list(
            inspect.signature(_heads_mod.LatentHeads.forward).parameters),
    }
    for name, params in sigs.items():
        hits = [p for p in params if "color" in p.lower()]
        if hits:
            bad.append(f"{name} accepts {hits}")
    detail["signatures"] = sigs

    leaks = {}
    for mod in MODEL_PATH_MODULES:
        idents = _code_identifiers(Path(mod.__file__))
        hits = sorted(i for i in idents if "color" in i.lower())
        leaks[mod.__name__] = hits
        if hits:
            bad.append(f"{mod.__name__} references {hits} in code")
    detail["identifier_scan"] = leaks

    detail["model_input_keys"] = list(MODEL_INPUT_KEYS)
    if any("color" in k for k in MODEL_INPUT_KEYS):
        bad.append("MODEL_INPUT_KEYS mentions colour")

    if model is not None:
        # a colour-shaped tensor cannot even be passed in
        try:
            model(**{k: None for k in MODEL_INPUT_KEYS}, h_color=torch.zeros(1))  # type: ignore[arg-type]
            bad.append("forward accepted an h_color keyword")
        except TypeError:
            detail["h_color_keyword_rejected"] = True
        except Exception:                                # noqa: BLE001
            detail["h_color_keyword_rejected"] = True
    return Check("WB-P8-no-h-color", "fail" if bad else "pass", detail, "; ".join(bad[:4]))


def check_h_where_causal_independence(vlm, collator, sample) -> Check:
    """The numerical half of item 8, on a real forward.

    Same image, same instruction, same ``<where>`` body, two different
    ``<color>`` bodies -> identical ``H_where``.  Causal attention guarantees it;
    this asserts the implementation actually slices what it claims to.
    """
    from .data import _PromptShim
    from .hiddens import EncodeItem

    tok = collator.tokenizer
    enc = collator.encode_one(_PromptShim(sample))
    prompt = enc["input_ids"][: enc["n_prompt_tokens"]]
    where_ids = tok(f"{WHERE_OPEN}{sample.where_text}{WHERE_CLOSE}",
                    add_special_tokens=False)["input_ids"]

    def run(color_text: str) -> torch.Tensor:
        color_ids = tok(f"{COLOR_OPEN}{color_text}{COLOR_CLOSE}",
                        add_special_tokens=False)["input_ids"]
        item = EncodeItem(sample.sample_id, sample.image, prompt,
                          list(where_ids) + list(color_ids))
        h = vlm.encode([item])[0].h_where
        return h[: len(where_ids)]

    a = run("warm highlights, lifted shadows")
    b = run("cool shadows, deep contrast, an entirely different plan")
    same = bool(torch.equal(a, b))
    delta = float((a - b).abs().max())
    return Check(
        "WB-P8b-h-where-causal-independence", "pass" if same else "fail",
        {"max_abs_diff": delta, "n_where_tokens": int(a.shape[0]),
         "hidden_dim": int(a.shape[-1])},
        "" if same else "H_where changed when only the <color> body changed",
    )


def check_hidden_contract(vlm, collator, sample) -> Check:
    """7b -- the teacher path and the generated path are the same function.

    The GT ids are pushed through the generated-context builder and then through
    :meth:`FrozenVLM.encode`; the result must equal the teacher path bit for bit.
    """
    from .data import _PromptShim
    from .hiddens import EncodeItem

    tok = collator.tokenizer
    close_id = int(tok(WHERE_CLOSE, add_special_tokens=False)["input_ids"][0])
    enc = collator.encode_one(_PromptShim(sample))
    prompt = enc["input_ids"][: enc["n_prompt_tokens"]]

    teacher = gt_context(tok, sample.sample_id, sample.where_text)
    as_generated = generated_context(sample.sample_id, teacher.token_ids + [1, 2, 3],
                                     close_id)
    if as_generated.token_ids != teacher.token_ids:
        return Check("WB-P7b-hidden-contract", "fail",
                     {"teacher_tokens": teacher.n_tokens,
                      "generated_tokens": as_generated.n_tokens},
                     "the generated span extractor does not reproduce the GT span")
    h1 = vlm.encode([EncodeItem(sample.sample_id, sample.image, prompt,
                                teacher.token_ids)])[0].h_where
    h2 = vlm.encode([EncodeItem(sample.sample_id, sample.image, prompt,
                                as_generated.token_ids)])[0].h_where
    same = bool(torch.equal(h1, h2))
    return Check(
        "WB-P7b-hidden-contract", "pass" if same else "fail",
        {"max_abs_diff": float((h1 - h2).abs().max()),
         "shape": list(h1.shape), "vlm": vlm.facts()},
        "" if same else "teacher and generated hidden extraction differ",
    )


# --- item 9: no target may reach a model input ------------------------------

def check_no_target_leak(model: WhereBModel | None = None, batch=None) -> Check:
    bad: list[str] = []
    detail: dict[str, Any] = {"model_input_keys": list(MODEL_INPUT_KEYS),
                              "sample_meta_keys": list(META_KEYS)}
    banned = ("mask", "oracle", "lut", "target", "tar", "latent", "baked", "preset")
    for k in MODEL_INPUT_KEYS:
        low = k.lower()
        for b in banned:
            if b in low and not low.startswith(("f_pre_mask", "h_where_mask")):
                bad.append(f"model input {k!r} looks like a target")
    if "image" in META_KEYS:
        bad.append("META_KEYS keeps the raw `image` block, which carries image.baked (I_tar)")

    params = list(inspect.signature(WhereBModel.forward).parameters)
    detail["forward_params"] = params
    for p in params:
        low = p.lower()
        if any(b in low for b in ("oracle", "lut", "gt_", "i_tar")):
            bad.append(f"forward parameter {p!r} is a target")

    if model is not None and batch is not None:
        batch.check_inputs()
        with torch.no_grad():
            base = model(**batch.inputs)
        for t in batch.targets:                     # poison every target
            for k, v in list(t.items()):
                if isinstance(v, torch.Tensor):
                    t[k] = torch.full_like(v, float("nan"))
        with torch.no_grad():
            after = model(**batch.inputs)
        same = bool(torch.equal(base.w_raw, after.w_raw) and torch.equal(base.w0, after.w0))
        detail["taint_test_output_unchanged"] = same
        if not same:
            bad.append("model output changed when the targets were poisoned with NaN")
    return Check("WB-P9-no-target-leak", "fail" if bad else "pass", detail,
                 "; ".join(bad[:4]))


# --- parameter inventory ----------------------------------------------------

def check_parameter_table(arms: Sequence[str] = ARM_IDS) -> Check:
    from .model import parameter_table

    rows = parameter_table(tuple(arms))
    by_struct: dict[str, set[int]] = {}
    for r in rows:
        by_struct.setdefault(r["structure"], set()).add(r["n_trainable_params"])
    bad = []
    joint8 = next((r for r in rows if r["structure"] == "MC8-Joint"), None)
    joint16 = next((r for r in rows if r["structure"] == "MC16-Joint"), None)
    split16 = next((r for r in rows if r["structure"] == "MC16-SplitHead"), None)
    dual16 = next((r for r in rows if r["structure"] == "MC16-DualCanvas"), None)
    if joint8 and joint16 and joint16["n_trainable_params"] <= joint8["n_trainable_params"]:
        bad.append("MC16-Joint is not larger than MC8-Joint")
    if joint16 and split16 and split16["n_trainable_params"] <= joint16["n_trainable_params"]:
        bad.append("MC16-SplitHead is not larger than MC16-Joint")
    if split16 and dual16 and dual16["n_trainable_params"] <= split16["n_trainable_params"]:
        bad.append("MC16-DualCanvas is not larger than MC16-SplitHead")
    return Check("WB-P-param-table", "fail" if bad else "pass",
                 {"rows": rows}, "; ".join(bad))


def check_zero_init_gates() -> Check:
    """Protocol 5.1 -- every cross-attention gate starts at exactly zero, so the
    connector output at step 0 does not depend on ``H_where`` or ``F_pre``."""
    bad = []
    detail = {}
    for arm in ("W01", "W08"):
        m = WhereBModel(arm_config(arm)).eval()
        gates = m.gate_values()
        detail[arm] = gates
        for stream, g in gates.items():
            if any(v != 0.0 for v in g["gate_text"] + g["gate_vis"]):
                bad.append(f"{arm}/{stream}: a cross-attn gate is not zero at init")
        b, p, t = 2, 40, 7
        f = torch.randn(b, p, 1024)
        pos = torch.randn(b, p, 2)
        h1 = torch.randn(b, t, 2560)
        h2 = torch.randn(b, t, 2560)
        with torch.no_grad():
            o1 = m(f, pos, None, h1, None)
            o2 = m(f * 3 + 1, pos, None, h2, None)
        d = float((o1.w_raw - o2.w_raw).abs().max())
        detail[f"{arm}_input_independence_max_diff"] = d
        if d > 1e-6:
            bad.append(f"{arm}: output depends on the input at init (diff {d:.2e})")
    return Check("WB-P-zero-init-gates", "fail" if bad else "pass", detail, "; ".join(bad))


# --- driver -----------------------------------------------------------------

def run_model_checks(
    rep: PreflightReport,
    *,
    model_dir: Path,
    checkpoint: Path | None,
    device: str,
    dtype: str,
    attn: str,
    split: str,
    sample_index: int,
) -> None:
    """Actually construct the VLM and run protocol 14 items 7b/8b.

    This is the body review blocker B1 said was missing: before this, passing
    ``--with-model`` merely removed two ``skip`` rows and nothing was executed.
    Any failure here is recorded as a ``fail`` row (never swallowed), so the
    report cannot come back green because the loader crashed.
    """
    from transformers import AutoProcessor

    from q3vl.train.collator import Sft2SegCollator
    from q3vl.train.modeling import load_model
    from q3vl.train.tokens import register_special_tokens

    from .data import open_dataset
    from .hiddens import FrozenVLM

    src = Path(checkpoint) if checkpoint else Path(model_dir)
    try:
        processor = AutoProcessor.from_pretrained(model_dir)
        register_special_tokens(processor.tokenizer)
        collator = Sft2SegCollator(processor, max_length=2048, system_prompt=None)
        model = load_model(str(src), attn_implementation=attn, dtype=dtype)
        model = model.to(device).eval()
        vlm = FrozenVLM(model, processor, device=device)
        ds, ds_info = open_dataset(split, need_mask=False,
                                   limit=sample_index + 1)
        if len(ds) <= sample_index:
            raise RuntimeError(f"{split} has {len(ds)} samples, need index {sample_index}")
        sample = ds[sample_index]
        rep.env["model_checks"] = {
            "weights": str(src), "device": device, "dtype": dtype, "attn": attn,
            "split": split, "sample_id": sample.sample_id, "vlm": vlm.facts(),
            "dataset": ds_info,
        }
    except Exception as exc:                       # noqa: BLE001
        for cid in MODEL_CHECKS:
            rep.add(Check(cid, "fail", {"error": f"{type(exc).__name__}: {exc}"},
                          "could not build the VLM / sample for the model checks"))
        return

    for cid, fn in (("WB-P7b-hidden-contract", check_hidden_contract),
                    ("WB-P8b-h-where-causal-independence",
                     check_h_where_causal_independence)):
        try:
            got = fn(vlm, collator, sample)
            if got.id != cid:
                raise AssertionError(f"{fn.__name__} returned id {got.id!r}, expected {cid!r}")
            rep.add(got)
        except Exception as exc:                   # noqa: BLE001
            rep.add(Check(cid, "fail", {"error": f"{type(exc).__name__}: {exc}"},
                          "the check itself raised"))


def run_where_b_preflight(
    *,
    model_dir: Path = MODEL_DIR,
    out: Path | None = None,
    skip_model: bool = True,
    arms: Sequence[str] = ARM_IDS,
    checkpoint: Path | None = None,
    device: str = "cuda",
    dtype: str = "bfloat16",
    attn: str = "flash_attention_2",
    split: str = "V_where",
    sample_index: int = 0,
) -> PreflightReport:
    rep = PreflightReport(env=_env())
    rep.env.update({"model_dir": str(model_dir), "skip_model": skip_model,
                    "checkpoint": str(checkpoint or "")})

    from transformers import AutoProcessor

    from q3vl.train.tokens import register_special_tokens

    processor = AutoProcessor.from_pretrained(model_dir)
    register_special_tokens(processor.tokenizer)

    rep.add(check_context_flows(processor.tokenizer))
    rep.add(check_no_h_color(WhereBModel(arm_config("W01")).eval()))
    rep.add(check_no_target_leak())
    rep.add(check_zero_init_gates())
    rep.add(check_parameter_table(arms))
    if skip_model:
        for cid in MODEL_CHECKS:
            rep.add(Check(cid, "skip", {}, "--skip-model (needs a real VLM forward)"))
    else:
        run_model_checks(rep, model_dir=model_dir, checkpoint=checkpoint,
                         device=device, dtype=dtype, attn=attn, split=split,
                         sample_index=sample_index)

    if rep.missing:                                 # belt and braces for B1
        for cid in rep.missing:
            rep.add(Check(cid, "fail", {}, "required check never ran"))

    out = Path(out) if out else REPORT_DIR / (
        "preflight_where_b_cpu.json" if skip_model else "preflight_where_b.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep.to_dict(), indent=2, ensure_ascii=False))
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description="Protocol 14 items 7/8/9 for Where-B")
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--out", default=None)
    ap.add_argument("--with-model", action="store_true",
                    help="also run protocol 14 items 7b/8b on a real VLM forward")
    ap.add_argument("--checkpoint", default=None,
                    help="Base SFT checkpoint; defaults to the base model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--split", default="V_where")
    ap.add_argument("--sample-index", type=int, default=0)
    args = ap.parse_args()
    rep = run_where_b_preflight(
        model_dir=Path(args.model_dir),
        out=Path(args.out) if args.out else None,
        skip_model=not args.with_model,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        device=args.device, dtype=args.dtype, attn=args.attn,
        split=args.split, sample_index=args.sample_index,
    )
    for c in rep.checks:
        print(f"[{c.status.upper():4}] {c.id}  {c.message}")
    if rep.missing:
        print(f"\nMISSING REQUIRED CHECKS: {rep.missing}")
    print(f"\npreflight {'PASS' if rep.ok else 'FAIL'}"
          f"  (complete={rep.complete}; skipped={[c.id for c in rep.checks if c.status == 'skip']})")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
