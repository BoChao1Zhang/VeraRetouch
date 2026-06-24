# VeraRetouch 架构报告:ImgEncoder / Render Encoder / LLM Special Token

> 整理日期:2026-06-21
> 范围:推理(inference)主链路。重点回答三个问题 —— 图像如何进 LLM、修图参数如何编码、特殊 token 如何把 LLM 和渲染器粘起来。

---

## 0. 一句话总览

VeraRetouch 是一个 **LLaVA(Qwen2)多模态大模型 + 像素级修图渲染器** 的拼装体。三者关系:

```
输入图像 ──ImgEncoder(vision tower)──▶ 视觉 token ──┐
                                                    ├─▶ Qwen2 LLM ──生成──▶ 文本(problem/plan) + 3 个修图 special token
任务/指令文本 ──tokenizer──▶ 文本 token ──────────┘                                    │
                                                                                       │ 取这 3 个 token 的最后层 hidden state
                                                                                       ▼
                                                                  retouch_head (MLP, 2688→2688)
                                                                                       │ = control latent
                                                                                       ▼
输入图像(原始像素) ──────────────────────────────────────▶ retouch_decoder ──▶ 修图后图像
```

**关键洞察:** 代码库里存在两套"把修图意图变成 control latent"的路径,它们共用同一个 `decoder`:

1. **独立渲染器路径(`RetouchRenderer`)** —— 用一对 before/after 参考图,经 **Render Encoder** 产出 2688 维 control latent。这是离线/参考式修图。
2. **统一模型路径(`VeraRetouchForCausalLLM_Unified`)** —— 用 **LLM 的 3 个 special token 的 hidden state**(经 `retouch_head`)产出同样 2688 维 control latent。这里 Render Encoder 被 LLM + special token "替换"掉了,只复用 `decoder`。

理解了这一点,三个模块就串成一条线了。

---

## 1. ImgEncoder(图像编码器 / Vision Tower)

作用:把输入图像编码成一串视觉 token,投影到 LLM 的隐空间,作为 `<image>` 占位符的填充内容。这是标准 LLaVA 范式。

### 1.1 编码主链路

`llava/model/llava_arch.py:141-143`

```python
def encode_images(self, images):
    image_features = self.get_model().get_vision_tower()(images)  # (B, num_patches, vision_hidden)
    image_features = self.get_model().mm_projector(image_features)  # (B, num_patches, llm_hidden)
    return image_features
```

两步:**vision tower 提特征 → mm_projector 投影到 LLM 维度**。

### 1.2 可选的三种 backbone(工厂选择)

`llava/model/multimodal_encoder/builder.py:6-19` 根据配置名选择:

| Backbone | 文件 | 关键类 | 典型输入→输出 |
|---|---|---|---|
| CLIP ViT | `multimodal_encoder/clip_encoder.py:7` | `CLIPVisionTower` / `CLIPVisionTowerS2` | (B,3,336,336) → (B,256,768) |
| MobileCLIP(FastViT) | `multimodal_encoder/mobileclip_encoder.py:13` | `MobileCLIPVisionTower` | (B,3,1024,1024) → (B,256,3072) |
| SigLIP | (HF `SiglipVisionModel`) | — | (B,3,384,384) → (B,729,1152) |

- CLIP:取倒数某层 hidden state,`feature_select` 去掉 CLS token 只留 patch(`clip_encoder.py:48-56`)。
- MobileCLIP:FastViT 输出 `[B,C,H,W]` reshape 成 `[B,H*W,C]`(`mobileclip_encoder.py:60-68`)。
- S2 版本:多尺度(如 336/672/1008)拼接,`hidden_size` 翻倍。

### 1.3 投影器(Vision → LLM)

`llava/model/multimodal_projector/builder.py`:`linear` / `mlp{N}x_gelu` / `identity` 三选一。把 vision hidden(768/1152/3072)线性/MLP 投影到 LLM hidden(Qwen2 = 896)。

### 1.4 视觉 token 如何嵌入序列

见第 3 节 —— `<image>` 在 tokenize 时被替换成哨兵 `IMAGE_TOKEN_INDEX = -200`,再在 `prepare_inputs_labels_for_multimodal` 里把该位置的 embedding 换成上面 `encode_images` 的输出。

> 注:另有一个**完全独立**的 `model/colormlp_v2.py:SiglipEncoder` 等,它属于下面的 Render Encoder 体系,**不是**给 LLM 喂图用的 vision tower,不要混淆。

---

## 2. Render Encoder(渲染器:修图参数编码器 + 解码器)

代码集中在 `model/retouch_render.py`、`model/colormlp_v2.py`、`model/resnet.py`、`model/vgg.py`,配置在 `configs/renderer_config.py`。

### 2.1 顶层容器 `RetouchRenderer`

`model/retouch_render.py:257-318`。结构 = **encoder(Render Encoder)+ decoder**:

```python
def forward(self, x, ref_o, ref_t, mask, chunk=-1):
    control_feat = self.encoder(ref_o, ref_t)        # Render Encoder:一对参考图 → 2688 维 latent
    control_feat = control_feat * mask_expend         # 三段 mask(光/色温/混色)可分别开关
    output = self.decoder(x, control_feat)            # 把 latent 作用到输入图 x 上
    return output
```

- `x`:待修图像;`ref_o / ref_t`:同一参考图的"修前/修后"对;`mask`:3 维开关。
- 这条路径用于"参考式修图"——给定一对示例,把同样的调整迁移到新图。

### 2.2 Render Encoder 的实现(可插拔,~20 种)

`RetouchRenderer.__init__`(`retouch_render.py:263-299`)按名字选 encoder。输入恒为**两张图**(修前 ref_o、修后 ref_t),输出恒为 `(B, latent_dim*3)`(`latent_dim=896` → **2688**):

| 家族 | 文件 | backbone | 输入分辨率 |
|---|---|---|---|
| `SiglipEncoder` / `SiglipEncoderDual` | `colormlp_v2.py:13-160` | 冻结 SigLIP(+可选 LoRA / Perceiver 采样) | 384×384 |
| `Resnet18Encoder` + ~15 变体(CBAM/FPN/Interact/VQ…) | `resnet.py:6-3409` | ResNet | 512×512 |
| `VGG16/19Encoder` | `vgg.py:6-154` | VGG | 512×512 |
| `ColorIlluminationEncoder` | `retouch_render.py:184-255` | 自定义 CNN + 差分特征 | 任意 |

共同模式:对两张图分别提特征 → 拼接/做差(`feat2-feat1`、`feat2/feat1`)→ 投影到 `latent_dim*3`。差分编码捕捉"做了什么调整"而非"图像内容"。

### 2.3 Decoder(把 latent 渲染成像素)

`model/colormlp_v2.py`:

- `ConditionalMLPDecoder`(`162-236`):逐像素 MLP。输入 `(B,3,H,W)` + control latent,条件注入方式 `cond_method ∈ {add, cat, adain, cross_attn}`,末端激活 `sigmoid/hard_sigmoid/...`。`forward_chunk` 分块处理大图。
- `ConditionalMLPDecoderDualAdaLN`(`376-440`):接受 `(B,2,latent)` 双条件。
- 本质是一个**全局色彩/影调映射网络**(类似可学习的 3D-LUT/曲线),不改内容只改颜色影调。

### 2.4 配置

`configs/renderer_config.py` 列了 17+ 个组合(`RetouchRenderer_Resnet18Encoder_InputCatMixedCBAM` 等),统一 `latent_dim=896`。

---

## 3. LLM Special Token(把 LLM 和渲染器粘起来的机制)

这是 VeraRetouch 区别于普通 LLaVA 的核心设计。

### 3.1 Token 定义

`llava/constants.py`。分四类:

- **图像**:`<image>`(L9),哨兵 `IMAGE_TOKEN_INDEX = -200`(L8);另有 `<im_patch>/<im_start>/<im_end>`。
- **修图控制(核心 3 个)**:`<retouch_light>`(L17)、`<retouch_color&temp>`(L18)、`<retouch_colormixer>`(L19)—— 分别对应 **光线 / 色温色调 / 颜色混合** 三组调整。
- **问题分析 / 修图方案**:`<problem_*_start/end>`、`<plan_*_start/end>`(L20-31),把 LLM 的 CoT 结构化成"光/全局色/局部色"三段。
- **任务类型**:`<Auto_Retouch_Task>` / `<Style_Retouch_Task>` / `<Professional_Retouch_Task>`(L32-34)。

### 3.2 注册到 tokenizer + embedding 初始化

- 加载时 `tokenizer.add_tokens([...], special_tokens=True)` 后 `resize_token_embeddings`(`builder.py:160-167`、`llava_arch.py:334-377`)。
- **新 token 的 embedding 用现有所有 token embedding 的均值初始化**(不是随机),输入/输出 embedding 都做(`llava_arch.py:344-353`、`train_qwen.py:241-254`)。

### 3.3 推理时拿到 token id 并注册进模型

`inference.py:60-63`:

```python
light_idx      = tokenizer(DEFAULT_RETOUCH_LIGHT_TOKEN,      add_special_tokens=False).input_ids[0]
colortemp_idx  = tokenizer(DEFAULT_RETOUCH_COLORTEMP_TOKEN,  add_special_tokens=False).input_ids[0]
colormixer_idx = tokenizer(DEFAULT_RETOUCH_COLORMIXER_TOKEN, add_special_tokens=False).input_ids[0]
model.register_special_token_idx(light_idx, colortemp_idx, colormixer_idx)
```

`register_special_token_idx`(`VeraRetouch.py:39-78`)把这些 id 存成模型属性。

### 3.4 输入侧:`<image>` token 的 embedding 替换

`llava/mm_utils.py:214-233` `tokenizer_image_token`:按 `"<image>"` 字符串切分 prompt,在切点插入哨兵 `-200`。

`VeraRetouch.py:200-314`(`prepare_inputs_labels_for_multimodal`):

1. `torch.where(input_ids == IMAGE_TOKEN_INDEX)` 定位图像位置;
2. 按哨兵把序列切成纯文本段,各段过 `embed_tokens` 得文本 embedding;
3. 在哨兵处插入 `encode_images` 产出的视觉 token;
4. 交错拼成 `[文本emb, 图像emb, 文本emb, ...]`,再 pad。

> 注:`VeraRetouch.py:229-231` 有一段**已注释**的 `<retouch_occ>` 输入侧替换逻辑(用参考 retouch token 填充),当前推理链路未启用 —— 现版本走的是下面 3.5 的"输出侧 hidden state 提取"。

### 3.5 输出侧:3 个修图 token → control latent(关键)

`VeraRetouch.py:358-410`,`generate` 用 `output_hidden_states=True`:

```python
output_ids   = outputs.sequences          # [bs, seq_len]
hidden_states = outputs.hidden_states      # 每步每层 hidden

# 对每个修图 token 求 mask,找到它在生成序列中第一次出现的位置
light_latent      = hidden_states[light_indices[0]][-1][i].squeeze()      # 该步最后一层 hidden, [hidden=896]
colortemp_latent  = hidden_states[colortemp_indices[0]][-1][i].squeeze()
colormixer_latent = hidden_states[colormixer_indices[0]][-1][i].squeeze()

retouch_latent = torch.stack([light, colortemp, colormixer]).view(-1).unsqueeze(0)  # [1, 3*896=2688]
retouch_latent = self.retouch_head(retouch_latent)                                  # MLP → 2688
if retouch_masks is not None:
    retouch_latent = retouch_latent * mask_expend     # 三段分别可关
retouched_img = self.retouch_decoder(input_img, retouch_latent)                     # 渲染成像素
```

- **`retouch_head`**(`VeraRetouch.py:29-37`):`Linear(2688→1344)→LN→GELU→Linear→LN→GELU→Linear(→2688)`。把 3 个 token 的语言 hidden 翻译成渲染器认识的 control latent。
- **`retouch_decoder`**(`VeraRetouch.py:23`):`get_retouch_decoder_model(config_add.retouch_decoder_name).decoder` —— **正是第 2 节那个 `RetouchRenderer` 的 `.decoder` 部分**。也就是说统一模型只借用渲染器的 decoder,encoder 角色由"LLM + 3 个 special token + retouch_head"承担。

### 3.6 维度对齐

`configs/infer_config.yaml`:

```yaml
retouch_decoder_name: RetouchRenderer_Resnet18Encoder_InputCatMixedCBAM
retouch_head_in_dim:  2688   # = 3 × Qwen2_hidden(896)
retouch_head_hidden_dim: 1344
retouch_head_out_dim: 2688   # = decoder 期望的 control latent(latent_dim*3 = 896*3)
```

3 个 special token × 每个 896 维 hidden = 2688,正好对上独立渲染器里 `latent_dim*3` 的 control latent 维度 —— 两条路径在 decoder 接口上完全等价。

### 3.7 Prompt 排布示例

`data/infer_dataset.py:60-66`(Auto 模式),Qwen2 chat 模板(`conversation.py:407-415`):

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<image>
<Auto_Retouch_Task>
Now, you are acting as a Retouch Agent ... state the problems (lighting/global_color/specific color), give the solution and retouch tokens.<|im_end|>
<|im_start|>assistant
```

模型在 assistant 段先生成结构化的 problem/plan 文本(用 `<problem_*>`/`<plan_*>` 包裹),最后吐出 `<retouch_light>` / `<retouch_color&temp>` / `<retouch_colormixer>` 三个 token,它们的 hidden state 就是修图"指令向量"。

---

## 4. 端到端推理时序

```
1. 构造 prompt:<image> + <Task> + 指令
2. tokenizer_image_token:<image> → 哨兵 -200
3. ImgEncoder:vision tower + mm_projector → 视觉 token
4. prepare_inputs_labels_for_multimodal:哨兵处填入视觉 token → inputs_embeds
5. Qwen2.generate(output_hidden_states=True):
     生成 problem/plan 文本 + <retouch_light>/<retouch_color&temp>/<retouch_colormixer>
6. 取 3 个 token 的最后层 hidden → stack(2688) → retouch_head → control latent
7. (可选)retouch_masks 按光/色温/混色三段开关
8. retouch_decoder(原图, control latent) → 修图后图像
```

---

## 5. 关键文件 / 行号速查

| 模块 | 文件:行 | 说明 |
|---|---|---|
| Token 常量 | `llava/constants.py:8-34` | 全部 special token + 哨兵 |
| ImgEncoder 主链 | `llava/model/llava_arch.py:141-143` | vision tower + projector |
| Vision tower 工厂 | `llava/model/multimodal_encoder/builder.py:6-19` | CLIP/MobileCLIP/SigLIP |
| CLIP tower | `llava/model/multimodal_encoder/clip_encoder.py:7,48-76` | |
| MobileCLIP tower | `llava/model/multimodal_encoder/mobileclip_encoder.py:13,60-68` | FastViT |
| 投影器 | `llava/model/multimodal_projector/builder.py` | linear/mlp/identity |
| `<image>` tokenize | `llava/mm_utils.py:214-233` | 插哨兵 -200 |
| 视觉 token 注入 | `llava/model/VeraRetouch.py:200-314` | embedding 替换 |
| 统一模型 / retouch_head | `llava/model/VeraRetouch.py:17-37` | 容器 + head MLP |
| token idx 注册 | `llava/model/VeraRetouch.py:39-78`;`inference.py:60-63` | |
| **special token → latent → 渲染** | `llava/model/VeraRetouch.py:358-410` | 核心机制 |
| Render Encoder 容器 | `model/retouch_render.py:257-318` | encoder+decoder |
| Render Encoder 选择 | `model/retouch_render.py:263-299` | ~20 种 |
| SigLIP/Decoder | `model/colormlp_v2.py:13-440` | encoder + 条件 MLP decoder |
| ResNet 家族 | `model/resnet.py:6-3409` | 15+ 变体 |
| 维度配置 | `configs/infer_config.yaml`;`configs/renderer_config.py` | 2688 对齐 |
| Prompt 排布 | `data/infer_dataset.py:60-66`;`llava/conversation.py:407-415` | |

---

## 6. 设计要点提炼

1. **三段式可控**:光线 / 色温色调 / 颜色混合 拆成 3 个 special token,既让 LLM 显式"分通道"决策,又能在渲染端通过 mask 独立开关。
2. **special token 当"软参数寄存器"**:不解码成离散数值,而是直接取其 hidden state 当连续控制向量 —— 避免了"LLM 输出数字字符串再 parse"的精度/格式损失。
3. **encoder 可替换、decoder 复用**:参考式修图(image-pair → Render Encoder)与指令式修图(LLM special token → retouch_head)产出同维度 latent,共享同一 decoder,接口统一。
4. **embedding 均值初始化**:新增 special token 用现有词表均值初始化,冷启动更稳。

> 待确认:`<problem_*>`/`<plan_*>` 这批 token 目前只见于定义与注册,推理侧主要消费 3 个 `<retouch_*>` token;前者更多服务于结构化 CoT 文本与训练监督,实际是否进 `retouch_head` 需结合训练代码(`llava/train/`)进一步核对。
