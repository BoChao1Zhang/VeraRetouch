# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/vlmsft/q3vl_common.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 / SFT+ADAPT -- Qwen3-VL-4B base: loading, LoRA surface, span readout.

用户裁决(2026-09-02):基座 = Qwen3-VL-4B-Instruct(不再用原论文 VeraRetouch VLM)。
读出主线 = 逐段 span mean-pool(EPR-033 color_span_pool 形态),special token 位
单点读出降为消融行。

Checkpoint (local, no download needed):
  dir      /home/bc/data/models/Qwen3-VL-4B-Instruct
  repo     Qwen/Qwen3-VL-4B-Instruct
  revision ebb281ec70b05090aa6165b016eac8ec08e71b17   (.hfd/repo_metadata.json `sha`,
           REVISION=main at fetch time, lastModified 2025-10-15T16:15:55Z)
  params   4,437,815,808 BF16;hidden 2560;36 text layers;vocab 151,936;
           tie_word_embeddings=true(权重索引里没有 lm_head)。
  class    Qwen3VLForConditionalGeneration(config.architectures);model_type qwen3_vl。
  载入版本 config.json 写 transformers_version 4.57.0.dev0;本机
           /home/bc/envs/q3vl_sft(transformers 4.57.1 / torch 2.10.0)实测可 import
           Qwen3VLForConditionalGeneration。

Readout contract — imported, not re-declared (q3vl/whereb/contracts.py:30,32):
  SEGMENT_HIDDEN_LAYER = -1, SEGMENT_HIDDEN_FINAL_NORM = True.
  i.e. the last decoder block's output, passed through the tower's final norm,
  cast to fp32, then mean-pooled over the segment's token positions.

Image contract — spec 5, via the repo's own q3vl.train.imageproc.prepare_image
  (short side 512, aspect preserved, both sides aligned to 32, bicubic, EXIF
  applied), handed to the HF image processor with do_resize=False so the planned
  geometry is the geometry used; assert_grid_matches cross-checks the grid.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

from veraretouch_sprf import _paths as _P  # EPR-052：替代原 sys.path 注入块
_P.ensure_sys_path()
REPO_ROOT = str(_P.REPO)

import torch  # noqa: E402
from torch import nn  # noqa: E402

from veraretouch_sprf.data.cot_text import STAGE_TOKENS  # noqa: E402

MODEL_DIR = os.environ.get("VR_QWEN3VL_DIR", "/home/bc/data/models/Qwen3-VL-4B-Instruct")  # EPR-052：可用环境变量覆盖，默认值不变
MODEL_REPO = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"

Z_DIM = 2560
N_TEXT_LAYERS = 36

# EPR-033 主行(q3vl/whatb/loraspan.py:131-140),逐项照抄
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LORA_LAST_N_BLOCKS = 8
LORA_TARGET_SUFFIXES: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

IGNORE_INDEX = -100


def die(m: str):
    print(f"FATAL: {m}", flush=True)
    raise SystemExit(2)


# --------------------------------------------------------------------------- #
# module resolution (same semantics as q3vl/whereb/hiddens.py:47-75)
# --------------------------------------------------------------------------- #
def resolve_language_model(model: nn.Module) -> nn.Module:
    for path in (("model", "language_model"), ("language_model",), ("model",)):
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "layers") and hasattr(obj, "norm"):
            return obj
    die("cannot find the text decoder (model.model.language_model with .layers/.norm)")


def resolve_visual(model: nn.Module) -> nn.Module:
    for path in (("model", "visual"), ("visual",)):
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "blocks"):
            return obj
    die("cannot find the vision tower (model.model.visual with .blocks)")


def resolve_mm_model(model: nn.Module) -> nn.Module:
    """The Qwen3VLModel (vision + text), i.e. the thing whose forward returns
    `last_hidden_state` ALREADY passed through the text tower's final norm
    (modeling_qwen3_vl.py: `hidden_states = self.norm(hidden_states)` then
    `Qwen3VLModelOutputWithPast(last_hidden_state=outputs.last_hidden_state)`).

    Reading it directly instead of hooking the last block has three effects:
      * the tensor is a normal graph node, so gradient checkpointing is safe;
      * `lm_head` is never called, so the 151,936-wide vocab logits for the whole
        sequence are never materialised;
      * NO extra norm may be applied on top -- doing so would double-norm.
    """
    for path in (("base_model", "model", "model"), ("model", "model"), ("model",)):
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "language_model") and hasattr(obj, "visual"):
            return obj
    die("cannot find Qwen3VLModel (expected .model with .language_model and .visual)")


class LastLayerHook:
    """Capture ONE decoder layer's output (q3vl/whereb/hiddens.py:78).

    output_hidden_states=True would allocate (L+1, B, T, 2560) and throw all but
    one slice away.
    """

    def __init__(self, lm: nn.Module, layer: int = -1):
        self.layer = lm.layers[layer]
        self.captured: torch.Tensor | None = None
        self._h = None

    def _fn(self, _m, _i, output):
        self.captured = output[0] if isinstance(output, tuple) else output

    def __enter__(self):
        self._h = self.layer.register_forward_hook(self._fn)
        return self

    def __exit__(self, *exc):
        if self._h is not None:
            self._h.remove()
        self._h = None
        return False


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_processor(model_max_length: int = 8192):
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL_DIR)
    tk = proc.tokenizer
    n_before = len(tk)
    n_new = tk.add_tokens(STAGE_TOKENS, special_tokens=True)
    tk.model_max_length = model_max_length
    return proc, n_before, n_new


def stage_token_ids(tokenizer) -> list[int]:
    ids = []
    for t in STAGE_TOKENS:
        v = tokenizer(t, add_special_tokens=False).input_ids
        if len(v) != 1:
            die(f"stage token {t} is not atomic: {v}")
        ids.append(v[0])
    return ids


def load_model(processor, dtype=torch.bfloat16, device="cuda:0",
               attn_implementation: str = "eager", grad_checkpointing: bool = False,
               weights_dir: str | None = None):
    """attn_implementation="eager" is the campaign default: EPR-033 launches pass
    --vlm-attn eager explicitly ("FA2/SDPA must not be fallen back to silently",
    q3vl/whatb/scripts/run_lora_span_arm.py:130-131).

    grad_checkpointing: the EPR-033 A-9 prohibition is HOOK-SPECIFIC -- HF's
    checkpointing hands a forward hook a tensor outside the autograd graph.  A
    checkpointed block's RETURN VALUE is in the graph, so reading
    Qwen3VLModel(...).last_hidden_state instead of hooking the last block makes
    checkpointing safe.  This arm therefore takes the return-value route and
    enables checkpointing with use_reentrant=False.  The span-pool gradient is
    still verified at runtime (A12-style) rather than assumed.
    """
    from transformers import Qwen3VLForConditionalGeneration
    src = weights_dir or MODEL_DIR
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        src, dtype=dtype, attn_implementation=attn_implementation)

    # TRAP (measured 2026-09-02): the tokenizer has 151,669 real entries but the
    # embedding matrix has config.vocab_size = 151,936 rows -- Qwen pads it.  The
    # 6 stage tokens therefore take ids 151,669..151,674, which ALREADY have rows.
    # Calling resize_token_embeddings(len(tokenizer)) here would SHRINK the matrix
    # from 151,936 to 151,675 and silently destroy 261 rows, so the resize happens
    # only in the genuine grow case.
    n_emb = model.get_input_embeddings().weight.shape[0]
    tok_len = len(processor.tokenizer)
    if tok_len > n_emb:
        model.resize_token_embeddings(tok_len)
        n_emb = model.get_input_embeddings().weight.shape[0]
    new_ids = stage_token_ids(processor.tokenizer)
    if len(new_ids) != len(STAGE_TOKENS):
        die(f"expected {len(STAGE_TOKENS)} stage tokens, got {len(new_ids)}")
    if max(new_ids) >= n_emb:
        die(f"stage token id {max(new_ids)} >= embedding rows {n_emb}")
    first_new = min(new_ids)

    emb = model.get_input_embeddings().weight.data
    if weights_dir is None:
        # fresh start: init the new rows to the mean of the pre-existing REAL rows.
        # When RESUMING from a checkpoint these rows are already trained -- re-running
        # this would silently wipe them, so it is skipped.
        emb[new_ids] = emb[:first_new].mean(dim=0, keepdim=True)
        out = model.get_output_embeddings()
        if out is not None and out.weight.data_ptr() != emb.data_ptr():
            out.weight.data[new_ids] = out.weight.data[:first_new].mean(dim=0, keepdim=True)
    else:
        print(f"[load] resume: stage-token rows {new_ids} kept from checkpoint "
              "(mean-init skipped)", flush=True)

    for p_ in model.parameters():
        p_.requires_grad_(False)
    model.config.use_cache = False
    if grad_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    facts = dict(n_emb_rows=n_emb, tokenizer_len=tok_len, new_ids=new_ids,
                 first_new_id=first_new, resized=bool(tok_len > n_emb))
    return model.to(device), facts


def enable_new_token_rows(model, facts):
    """Only the stage-token rows train.  Weights are tied to lm_head, so without
    this the model could never emit <vr_stage_m>; with a naive full-matrix grad it
    would instead fine-tune the entire output head."""
    emb = model.get_input_embeddings()
    emb.weight.requires_grad_(True)
    mask = torch.zeros(emb.weight.shape[0], 1, device=emb.weight.device,
                       dtype=emb.weight.dtype)
    mask[facts["new_ids"]] = 1.0
    emb.weight.register_hook(lambda g: g * mask)
    return emb.weight, mask


# --------------------------------------------------------------------------- #
# LoRA surface (q3vl/whatb/loraspan.py:520-556 -- fully-qualified names)
# --------------------------------------------------------------------------- #
def lora_target_names(model: nn.Module, last_n: int = LORA_LAST_N_BLOCKS,
                      suffixes: Sequence[str] = LORA_TARGET_SUFFIXES):
    """Fully-qualified names, never bare suffixes.

    peft's `check_target_module_exists` accepts a bare "q_proj" and would then
    adapt the vision tower too.  A full name takes peft's exact-match branch.
    """
    lm = resolve_language_model(model)
    n_layers = len(lm.layers)
    if not 1 <= int(last_n) <= n_layers:
        die(f"last_n must be 1..{n_layers}, got {last_n}")
    by_id = {id(mod): name for name, mod in model.named_modules()}
    lm_name = by_id.get(id(lm))
    if lm_name is None:
        die("the language tower is not a submodule of the model")
    idxs = list(range(n_layers - int(last_n), n_layers))
    names = []
    for i in idxs:
        for suf in suffixes:
            full = f"{lm_name}.layers.{i}.self_attn.{suf}"
            if not any(n == full for n, _ in model.named_modules()):
                die(f"declared LoRA target {full} does not exist")
            names.append(full)
    return names, idxs


def attach_lora(model: nn.Module, r=LORA_R, alpha=LORA_ALPHA,
                dropout=LORA_DROPOUT, last_n=LORA_LAST_N_BLOCKS):
    from peft import LoraConfig, get_peft_model
    targets, idxs = lora_target_names(model, last_n=last_n)
    cfg = LoraConfig(r=int(r), lora_alpha=int(alpha), lora_dropout=float(dropout),
                     bias="none", target_modules=list(targets),
                     init_lora_weights=True, task_type=None)
    peft_model = get_peft_model(model, cfg)
    # A_lora: the adapted surface must equal the declared surface, as a set.
    got = {n.rsplit(".lora_A", 1)[0].replace("base_model.model.", "")
           for n, _ in peft_model.named_parameters() if ".lora_A." in n}
    want = set(targets)
    if got != want:
        die(f"A_lora FAILED: adapted {len(got)} modules, declared {len(want)}; "
            f"extra={sorted(got - want)[:4]} missing={sorted(want - got)[:4]}")
    facts = dict(r=int(r), alpha=int(alpha), dropout=float(dropout),
                 last_n_blocks=int(last_n), block_indices=idxs,
                 n_targets=len(targets), targets=targets,
                 n_lora_params=sum(p.numel() for n, p in peft_model.named_parameters()
                                   if "lora_" in n and p.requires_grad))
    return peft_model, facts


def assert_frozen_area(model: nn.Module, extra_trainable: Sequence[str] = ()):
    """A_freeze: only LoRA params (and explicitly named extras) may train."""
    bad = [n for n, p in model.named_parameters()
           if p.requires_grad and "lora_" not in n
           and not any(e in n for e in extra_trainable)]
    if bad:
        die(f"A_freeze FAILED: {len(bad)} non-LoRA trainable params, e.g. {bad[:4]}")


# --------------------------------------------------------------------------- #
# image (spec 5)
# --------------------------------------------------------------------------- #
def prepare_image_spec5(pil):
    from q3vl.train.imageproc import prepare_image
    return prepare_image(pil)


def assert_grid(geom, grid_thw):
    from q3vl.train.imageproc import assert_grid_matches
    assert_grid_matches(geom, grid_thw)

def unfreeze_full(model: nn.Module):
    """Full fine-tune surface: EVERYTHING trainable except the vision tower.

    `model.visual` (patch_embed, blocks, merger, deepstack mergers, pos_embed) stays
    frozen; the language tower, the multimodal projector-side params and the tied
    embedding / lm_head (hence the 6 stage-token rows) all train.
    """
    vis = resolve_visual(model)
    vis_ids = {id(p_) for p_ in vis.parameters()}
    n_tr = n_fr = 0
    for p_ in model.parameters():
        if id(p_) in vis_ids:
            p_.requires_grad_(False); n_fr += p_.numel()
        else:
            p_.requires_grad_(True); n_tr += p_.numel()
    return dict(n_trainable=n_tr, n_frozen_visual=n_fr)


def assert_frozen_area_full(model: nn.Module):
    """A_freeze (full-FT口径): every vision-tower param frozen, and the language
    side actually trainable.  Mirror of assert_frozen_area for the LoRA arms."""
    vis = resolve_visual(model)
    vis_ids = {id(p_) for p_ in vis.parameters()}
    bad = [n for n, p_ in model.named_parameters()
           if id(p_) in vis_ids and p_.requires_grad]
    if bad:
        die(f"A_freeze(full) FAILED: {len(bad)} vision-tower params trainable, "
            f"e.g. {bad[:4]}")
    n_tr = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    if n_tr == 0:
        die("A_freeze(full) FAILED: nothing is trainable")
    emb = model.get_input_embeddings()
    if not emb.weight.requires_grad:
        die("A_freeze(full) FAILED: embedding (tied to lm_head) is not trainable")
    return n_tr
