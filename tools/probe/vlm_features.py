"""PR-1/PR-3 · 冻结 VLM 逐层特征抽取（INF-7 探针工具链）。

对一张图 + 一条指令做**一次 teacher-forced 前向**（无生成），导出 PLAN §3
「取点 layer×module 双扫」要求的全部探针位点：

  vit.stem / vit.s{i} / vit.out   视觉塔（FastViTHD）逐 stage（空间均值池化）
  conn.out                        connector 输出（= 喂进 LLM 的 image token 均值）
  llm.resid.L{l}                  LLM 第 l 层残差流（image token 均值），l=0..23
  llm.attn.L{l} / llm.mlp.L{l}    LLM 每层三位点的另外两个（子层输出，image token 均值）
  llm.last.L{l}                   末位置（文本侧 last token）
  llm.sink.L{l}                   **sink/register 单独一路**：该层范数最大的 image token
  llm.lat3.L{l}                   **control latent 单独一路**：三个 retouch special token
                                  拼接（3×896=2688）——l=23 即线上真实读出口
                                  （VeraRetouch.py:405-425 的 retouch_head 输入）

红线遵守：
- 特征**禁逐图归一化**：本模块只做「跨 token 平均池化」，不做任何逐图仿射；
  标准化留给探针拟合端，且只用**训练折统计量**。
- C3 token 置换干预对象 = **整段 image token**（非 last token）。
- 需要 attention 时必须 eager（本模块默认不取 attention；取则断言非 None）。

用法见 experiments/PR13_probe_whatwhere_20260803/run_extract.py。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MODEL_PATH_DEFAULT = "/home/bc/data/models/VeraRetouch"
CONFIG_ADD_DEFAULT = str(REPO / "configs" / "infer_config.yaml")
GRID = 16
N_IMG_TOKENS = GRID * GRID
N_LAYERS = 24
HID = 896
GL_TOKENS = ("<retouch_light>", "<retouch_color&temp>", "<retouch_colormixer>")

# 固定中性 prompt（颜色探针与空间探针共用，保证 H3 两条曲线同口径）
NEUTRAL_INSTRUCTION = "Retouch this photo."


def _style_prompt(instruction: str) -> str:
    from llava.constants import DEFAULT_IMAGE_TOKEN, TASK_STYLE_RETOUCH_TOKEN
    from llava.conversation import conv_templates

    qs = (f"{TASK_STYLE_RETOUCH_TOKEN}\nNow, you are acting as a Retouch Agent. "
          "I will provide an image and an instruction, please give me a retouch plan "
          f"and retouch tokens.\n Instruction: {instruction}")
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


class VLMFeatureExtractor:
    """冻结 VLM 逐层探针位点抽取器。

    randomize: None（真实权重）/ "llm"（C2a：LLM 随机重初始化，视觉塔+connector 保留）
               / "all"（C2b：视觉塔+connector+LLM 全部随机重初始化）
    """

    def __init__(self, model_path: str = MODEL_PATH_DEFAULT,
                 config_add: str = CONFIG_ADD_DEFAULT,
                 device: str = "cuda:0", randomize: str | None = None,
                 seed: int = 20260803):
        import torch
        import yaml
        from box import Box
        from transformers import AutoTokenizer

        from llava.model.VeraRetouch import VeraRetouchForCausalLLM_Unified
        from llava.utils import disable_torch_init

        with open(config_add, "r", encoding="utf-8") as f:
            cfg_add = Box(yaml.safe_load(f))
        cfg_add.project_name = "pr13_probe"
        cfg_add.freeze_retouch_decoder = True

        disable_torch_init()
        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, model_max_length=8192, padding_side="right", use_fast=False)
        self.model = VeraRetouchForCausalLLM_Unified.from_pretrained(
            model_path, config_add=cfg_add, torch_dtype=torch.bfloat16,
            attn_implementation="eager").to(device)
        self.model.eval()
        if self.model.config._attn_implementation != "eager":
            raise RuntimeError("attn_implementation != eager（DOSSIER §5.4 红线）")

        self.gl_ids = [self.tokenizer(t, add_special_tokens=False).input_ids[0]
                       for t in GL_TOKENS]
        assert self.gl_ids[0] == 151646, self.gl_ids
        self.model.register_special_token_idx(*self.gl_ids)
        self.model.lens_track_spans = True
        self.image_processor = self.model.get_vision_tower().image_processor

        self.randomize = randomize
        if randomize is not None:
            self._randomize(randomize, seed)

        self._install_hooks()
        self._token_perm: np.ndarray | None = None   # C3

    # -- C2 随机骨干 --------------------------------------------------------
    def _randomize(self, which: str, seed: int) -> None:
        torch = self.torch
        g = torch.Generator(device="cpu").manual_seed(seed)
        lm = self.model.get_model()
        targets = []
        for name, mod in lm.named_modules():
            is_vis = ("vision_tower" in name) or ("mm_projector" in name)
            if which == "llm" and is_vis:
                continue
            targets.append((name, mod))
        std = float(getattr(self.model.config, "initializer_range", 0.02))
        n = 0
        for name, mod in targets:
            for pname, p in mod.named_parameters(recurse=False):
                w = torch.empty(p.shape, dtype=torch.float32).normal_(0.0, std, generator=g)
                if "bias" in pname or p.ndim == 1:
                    # LayerNorm/RMSNorm 权重置 1、bias 置 0（否则前向数值爆炸，
                    # 得到的不是「随机表征」而是 NaN）
                    w = torch.ones_like(w) if "bias" not in pname else torch.zeros_like(w)
                p.data.copy_(w.to(p.dtype).to(p.device))
                n += 1
        print(f"[C2 randomize={which}] reinit {n} param tensors (std={std})", flush=True)

    # -- hooks --------------------------------------------------------------
    def _install_hooks(self) -> None:
        self._cap: dict[str, "object"] = {}
        vt = self.model.get_vision_tower().vision_tower.model   # FastViTHD

        def mk(key):
            def hook(_m, _i, out):
                self._cap[key] = out
            return hook

        self._handles = []
        self._handles.append(vt.patch_embed.register_forward_hook(mk("vit.stem")))
        for i, blk in enumerate(vt.network):
            self._handles.append(blk.register_forward_hook(mk(f"vit.s{i}")))
        self._n_vit_stages = len(vt.network)
        self._handles.append(vt.conv_exp.register_forward_hook(mk("vit.out")))

        layers = self.model.get_model().layers
        for li, layer in enumerate(layers):
            self._handles.append(
                layer.self_attn.register_forward_hook(mk(f"attn.{li}")))
            self._handles.append(
                layer.mlp.register_forward_hook(mk(f"mlp.{li}")))

    # -- C3 token 置换 ------------------------------------------------------
    def set_token_permutation(self, perm: np.ndarray | None) -> None:
        """C3 对照：置换喂进 LLM 的**整段 image token**（红线：非 last token）。"""
        self._token_perm = perm

    # -- 输入构造 -----------------------------------------------------------
    def _prepare(self, img_u8: np.ndarray, instruction: str):
        import torch
        from PIL import Image

        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import process_images_, tokenizer_image_token

        prompt = _style_prompt(instruction) + "".join(GL_TOKENS)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors="pt").unsqueeze(0).to(self.device)
        # 守卫：末尾三个 token 必须正好是三个 GL special token（否则 lat3 取错位置）
        tail = input_ids[0, -3:].tolist()
        if tail != self.gl_ids:
            raise RuntimeError(f"prompt 末尾 token {tail} != GL ids {self.gl_ids}")
        pil = Image.fromarray(img_u8)
        image_tensor = process_images_([pil], self.image_processor)[0]
        images = [image_tensor.to(torch.bfloat16).to(self.device)]
        return input_ids, images, pil.size

    # -- 主入口 -------------------------------------------------------------
    def encode(self, img_u8: np.ndarray, instruction: str = NEUTRAL_INSTRUCTION,
               want_tokens: bool = False) -> dict:
        """返回 {position_name: np.float32 vector}；want_tokens 时附 tokens (24,256,896) fp16。"""
        import torch

        input_ids, images, image_size = self._prepare(img_u8, instruction)
        self._cap.clear()
        self.model.lens_image_spans = []

        ctx = []
        if self._token_perm is not None:
            ctx.append(self._perm_patch())
        with torch.inference_mode():
            if ctx:
                with ctx[0]:
                    out = self.model(input_ids=input_ids, images=images,
                                     image_sizes=[image_size], use_cache=False,
                                     output_hidden_states=True, return_dict=True)
            else:
                out = self.model(input_ids=input_ids, images=images,
                                 image_sizes=[image_size], use_cache=False,
                                 output_hidden_states=True, return_dict=True)

        span = self.model.lens_image_spans[0]
        i0, i1, total = span
        if i1 - i0 != N_IMG_TOKENS:
            raise RuntimeError(f"image span {span} != {N_IMG_TOKENS}")
        hs = out.hidden_states                       # len 25，[0]=embedding 输出
        seq_len = hs[0].shape[1]
        if seq_len != total:
            raise RuntimeError(f"seq_len {seq_len} != span total {total}")

        # 三个 GL token 的位置：序列末尾三个（prompt 尾部固定追加）
        lat_pos = [seq_len - 3, seq_len - 2, seq_len - 1]

        # 全部位点先在 GPU 上算好，**一次性**搬回 CPU（159 次 .cpu() 会拖慢 ~1 s/前向）
        names: list[str] = []
        vecs: list["object"] = []

        def add(name, t):
            names.append(name)
            vecs.append(t.reshape(-1).float())

        for k in sorted(self._cap):
            if not k.startswith("vit."):
                continue
            v = self._cap[k]
            v = v[0] if isinstance(v, (tuple, list)) else v
            t = v.float()
            add(k, t.mean(dim=(-2, -1))[0] if t.ndim == 4 else t.mean(dim=1)[0])

        # connector 输出 = embedding 层上的 image token（hidden_states[0]）
        add("conn.out", hs[0][0, i0:i1].float().mean(0))

        tok_stack = [] if want_tokens else None
        for li in range(N_LAYERS):
            h = hs[li + 1][0].float()                 # (T, 896) 第 li 层输出
            him = h[i0:i1]                            # (256, 896)
            add(f"llm.resid.L{li}", him.mean(0))
            add(f"llm.last.L{li}", h[-4])             # 末位置（三个 GL token 之前）
            add(f"llm.sink.L{li}", him[him.norm(dim=-1).argmax()])
            add(f"llm.lat3.L{li}", h[lat_pos])
            a = self._cap.get(f"attn.{li}")
            a = a[0] if isinstance(a, (tuple, list)) else a
            add(f"llm.attn.L{li}", a[0, i0:i1].float().mean(0))
            m = self._cap.get(f"mlp.{li}")
            m = m[0] if isinstance(m, (tuple, list)) else m
            add(f"llm.mlp.L{li}", m[0, i0:i1].float().mean(0))
            if want_tokens:
                tok_stack.append(him.to(torch.float16))

        dims = [int(v.shape[0]) for v in vecs]
        flat = torch.cat(vecs).cpu().numpy().astype(np.float32)
        feats: dict[str, np.ndarray] = {}
        s = 0
        for nm, d in zip(names, dims):
            feats[nm] = flat[s:s + d]
            s += d
        feats["readout.actual"] = feats["llm.lat3.L23"]   # 线上 retouch_head 输入
        if want_tokens:
            tok_stack = torch.stack(tok_stack).cpu().numpy()
        # D-0 记账：末层 image token 范数 outlier 占比（median + 3*MAD）
        nrm = hs[-1][0, i0:i1].float().norm(dim=-1).cpu().numpy()
        med = float(np.median(nrm))
        mad = float(np.median(np.abs(nrm - med)))
        out_d = {"feats": feats,
                 "outlier_frac": float(np.mean(nrm > med + 3.0 * mad)) if mad > 0 else 0.0}
        if want_tokens:
            out_d["tokens"] = tok_stack                   # (24, 256, 896) fp16
        return out_d

    def _perm_patch(self):
        """上下文管理器：在 encode_images 之后置换 image token 顺序（C3）。"""
        import contextlib

        import torch
        model = self.model
        perm = torch.as_tensor(self._token_perm, dtype=torch.long, device=self.device)
        orig = model.encode_images

        def patched(imgs):
            f = orig(imgs)
            return f[:, perm, :] if f.ndim == 3 else f

        @contextlib.contextmanager
        def cm():
            model.encode_images = patched
            try:
                yield
            finally:
                model.encode_images = orig
        return cm()


def position_names(n_vit_stages: int = 8) -> list[str]:
    names = ["vit.stem"] + [f"vit.s{i}" for i in range(n_vit_stages)] + ["vit.out", "conn.out"]
    for fam in ("resid", "attn", "mlp", "last", "sink", "lat3"):
        names += [f"llm.{fam}.L{l}" for l in range(N_LAYERS)]
    names.append("readout.actual")
    return names
