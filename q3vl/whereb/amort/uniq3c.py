"""Language-side-only LoRA (EPR-011 LORA_P3 retry) -- NEW FILE.

LORA_P3 (visual+language LoRA) was executed at step ~130 by the sim-field
domain assertion: the P3' head leans on the similarity channel, so its
gradients push the visual tower / merger -- the sim PRODUCER -- out of the
domain the arm-wide norm was fitted in.  This variant removes the mechanism
at the root: LoRA touches ONLY the language stack (negative-lookahead
excludes anything under `visual`), so `f_merger` and `F_pre` are
bit-identical to the base model and the sim domain cannot drift by
construction.  The frozen-premise diagnostic for the P3' head therefore
becomes "does LANGUAGE-side finetuning beat the anchor", which is the only
form the P3 head admits.

Grad gating, checkpointing and the trainer-optimiser exposure follow
uniq3.TrainableVLM (the canary-tested pattern).
"""

from __future__ import annotations

import torch

from q3vl.whereb.hiddens import FrozenVLM

from .uniq3 import _ENCODE_NOGRADLESS

__all__ = ["LangOnlyLoRAVLM"]


class LangOnlyLoRAVLM(FrozenVLM):
    _grad_on: bool = False

    def __init__(self, model, processor, device="cuda", *, lora_r: int = 16,
                 lora_alpha: int = 32, **kw):
        super().__init__(model, processor, device, **kw)
        from peft import LoraConfig, inject_adapter_in_model

        cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none",
            target_modules=r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj)$")
        inject_adapter_in_model(cfg, model)
        n = 0
        for name, p in model.named_parameters():
            if "lora_" in name:
                if "visual" in name:
                    raise RuntimeError(
                        f"language-only LoRA leaked into the visual tower: "
                        f"{name} -- the whole point of this variant is that "
                        "the sim producer stays bit-identical")
                p.requires_grad_(True)     # bf16 storage (user ruling)
                n += p.numel()
        if not n:
            raise RuntimeError("LoRA injection matched no language modules")
        self.n_lora_params = n
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        model.train()

    def encode(self, items):
        if self._grad_on and torch.is_grad_enabled():
            return _ENCODE_NOGRADLESS(self, items)
        with torch.no_grad():
            return _ENCODE_NOGRADLESS(self, items)

    def facts(self):
        return {**super().facts(), "lora": "language-only",
                "n_lora_params": self.n_lora_params,
                "gradient_checkpointing": True}
