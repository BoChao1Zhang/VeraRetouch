# `tools/readout/clip_naclip` — vendored OpenAI-CLIP fork (RO-1 骨干)

零训练 self-self 读出（EXPERIMENTS_v3 §2.1 RO-1）的**单一骨干**。三个官方算子
（SCLIP / NACLIP / ClearCLIP）都是"最后一个 resblock 的 attention 手术"，把它们放进同一份
权重、同一份预处理里，才能回答任务卡问的"三家谁强"——分别 clone 三个仓库会同时换掉
open_clip 版本、fp16/fp32、输入尺度，三者不可比。

## 出处（全部一手核实，2026-08-03）

| 文件 | 来源 | commit |
|---|---|---|
| `clip.py` `model.py` `simple_tokenizer.py` `bpe_simple_vocab_16e6.txt.gz` `__init__.py` | https://github.com/sinahmr/NACLIP `clip/` | `0cac3a651f753b11315ea799cfbfabf79da1bd76` |
| `imagenet_template.py` | 同上 `prompts/imagenet_template.py`（80 条 openai 模板） | 同上 |
| `LICENSE.NACLIP` | NACLIP MIT License (c) 2024 Sina Hajimiri | 同上 |

NACLIP 的 `clip/` 本身是 OpenAI CLIP (`github.com/openai/CLIP`) 的 fork（文件头有声明），
其 `VisionTransformer.custom_attn` 已内建 `naclip / nonly / kk / csa / vanilla` 五种策略，
其中 **`csa` 与 SCLIP 官方实现逐字相同**（已对 SCLIP@`3608360267b6130c1ef18090d7289f17c771cb90`
`clip/model.py:283-313` 核对：`softmax(qqᵀ·scale)+softmax(kkᵀ·scale)`，且 SCLIP 的末层
走 `x = x + custom_attn(...)`、`x = x + mlp(...)`，即本文件的 `arch='vanilla'`）。

## VeraRetouch 的三处改动（全部标了 `VeraRetouch add-on`）

1. **`attn_strategy='clearclip'`**（`set_params` + `custom_attn`）：
   `softmax(qqᵀ·scale)`，逐字取自 ClearCLIP@`ad68a404d55d48d27330b93554eb64a234ff717f`
   `open_clip/transformer.py:614-616`。ClearCLIP 官方的 `ignore_residual=True` +
   `last_n_layers=1`（同文件 `:520-528`）= 末层丢残差丢 FFN，与本文件 `arch='reduced'`
   的 `x = custom_attn(...)` 语义等价，故断言 `clearclip ⇒ arch='reduced'`。
2. **`extra_tokens` 测试时寄存器**：位置编码之后、`ln_pre` 之前追加 N 个零嵌入 token，
   逐字对齐 test-time-registers@`860df43515c8d8e9e90952af25a46c26e4469570`
   `clip/clip/transformer.py:805-822`（arXiv 2506.08010, NeurIPS'25）。
   NACLIP 的高斯邻域偏置矩阵同步补零行/列（寄存器无空间位置）。
3. **`self._last_hidden`**：`ln_post` 之前的 hidden（NLD），供 D-0 的高范数 outlier 诊断使用。

除此之外**未改动任何一行**（可 `diff` 回原仓库核对）。

## 权重

`/home/bc/data/models/openai_clip/ViT-B-16.pt`，直接取自 OpenAI 官方 URL
`openaipublic.azureedge.net/clip/models/5806e77c…/ViT-B-16.pt`，
sha256 = `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`
（与 URL 内嵌的期望哈希一致，`clip.py::_download` 的校验口径）。
