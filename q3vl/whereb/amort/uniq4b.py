"""ST_LANG: in-context query tokens + language-only LoRA -- NEW FILE.

Why this arm exists (2026-08-13 午): the visual-LoRA drift verdict killed
both the bridge+LoRA cell (step 400) and the ST continuation (step ~520,
accumulated drift), while ST itself survived exactly 1200 steps.  The clean
2x2 factorial and any continued-training future for the ST line therefore
both require the language-only form.  Implementation: construct the parent
QueryTokVLM with ``lora=False`` (token machinery + checkpointing + grad
gate all inherited), then inject the language-only adapters exactly as
uniq3c does -- zero duplicated plumbing.
"""

from __future__ import annotations

from .uniq4 import QueryTokVLM

__all__ = ["LangQueryTokVLM"]


class LangQueryTokVLM(QueryTokVLM):
    def __init__(self, model, processor, device="cuda", *, n_qtok: int = 8,
                 lora_r: int = 16, **kw):
        super().__init__(model, processor, device, n_qtok=n_qtok,
                         lora=False, **kw)
        from peft import LoraConfig, inject_adapter_in_model

        cfg = LoraConfig(
            r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0, bias="none",
            target_modules=r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj)$")
        inject_adapter_in_model(cfg, model)
        n = 0
        for name, p in model.named_parameters():
            if "lora_" in name:
                if "visual" in name:
                    raise RuntimeError(
                        f"language-only LoRA leaked into the visual tower: {name}")
                p.requires_grad_(True)     # bf16 storage (user ruling)
                n += p.numel()
        if not n:
            raise RuntimeError("LoRA injection matched no language modules")
        self.lora = True                   # facts() reports it
        self.n_lora_params = n

    def facts(self):
        return {**super().facts(), "lora": "language-only"}
