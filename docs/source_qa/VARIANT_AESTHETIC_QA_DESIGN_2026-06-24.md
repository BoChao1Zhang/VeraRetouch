# Variant aesthetic QA — binary F⊕R questionnaire (≥10-dim) — design for review

Final aesthetic gate for the 50k construct set. Per item: SOURCE photo (第一张) + variant = source with
ONE color-grade preset really applied (第二张). Judge decides, source-conditioned: veto broken renders +
rank the good ones (keep top-2 SFT, full set DPO). Method = binary forward/reverse, position-coded,
rule-cleaned (reuse source_qa qa_clean machinery). Judge = local 35B (qwen3_5-35b-a3b), 2 images/prompt.

Designed by a 5-lens workflow (colorist / portrait / art-director / contest-judge / technical-QC) +
adversarial critique. Critique P0/P1 fixes are APPLIED below.

## 12 dimensions (6 veto + 6 merit). Each = independent F⊕R pair. F=1 good/present; R=1 the DEFECT.
Reverse claims name a DISTINCT concrete symptom (not ¬F), so both-1 is a real "didn't look" contradiction.

### VETO dims (failure ⇒ unusable). Skip chroma dims when variant is B&W (measured cf<8).
| key | name | F (1=good) | R (1=defect, independent symptom) | portrait |
|-----|------|-----------|-----------------------------------|----------|
| WBCAST | 中性色罩 | 本应中性的区域(白墙/灰物/雪/纸)保持干净中性,无非本意偏色 | 中性物被染上明显黄绿/品红/青蓝脏色罩(非原图本有氛围) | |
| MEMORY_COLOR | 记忆色合理 | 天空/草木/肤色等记忆色仍是可信自然的色相 | 记忆色被扭到不可信色相(天发青绿、草发黄、肤发橙/品红) | |
| SKINTONE | 肤色还原 | 人物肤色落在健康肤色带、像真人 | 肤色发橙红塑料/黄疸泛绿/品红猪肝/被环境染青 | ✓ |
| HILIGHT_CLIP | 高光死白 | 高光保留质感与过渡(受控的亮) | 高光成片纯白死区/硬切边,相比原图烧掉亮部细节 | |
| SHADOW_CRUSH | 暗部死黑 | 暗部保留层次细节 | 暗部出现断层式色块/色阶台阶(posterize),非自然低调 | |
| SAT_CLIP | 过饱和溢出 | 鲜艳处仍有层次,无溢色 | 色相溢出糊成无层次实色块/边缘出血/霓虹断层 | |

### MERIT dims (quality points, rank survivors). R tightened so "loud edit" doesn't fire several at once.
| key | name | F (1=good) | R (1=defect) | portrait |
|-----|------|-----------|--------------|----------|
| INTENT_HARMONY | 调色意图/和谐 | 全图色调成体系、有明确专业意图、配色和谐 | 颜色改动随机失控/区域色调互相矛盾/像乱加一层不搭的色 | |
| MOOD_FIT | 氛围契合 | 营造连贯情绪且契合本图题材场景 | 情绪与画面内容错位矛盾(暖场压成阴冷等) | |
| SUBJECT_POP | 主体凸显 | 相比原图主体更分离立体讨喜 | 相比原图主体被压平/与背景糊成一团 | |
| PREMIUM | 高级感 | 高端专业可上刊的精致质感(干净通透不发灰发雾) | 廉价业余一键滤镜感/发灰发脏发雾浑浊 | |
| RESTRAINT | 克制 | 强度恰到好处、服务画面 | 用力过猛:对比拉爆/HDR脏感/描边/浓到失真、喧宾夺主 | |
| BETTER_THAN_SRC | 优于原图 | 整体明确比第一张更好更高级 | 其实不如原图(更难看/更脏/更廉价) | |

## Controls (anti-cheating; reuse POS_MAP scramble + qa_clean.parse_codes)
| key | kind | question | expected |
|-----|------|----------|----------|
| ANCHOR | anchor | 你是否看到了两张图像 | 1 |
| HON_POS | honesty_pos | 第二张是第一张套用调色后的版本吗 | 1 |
| HON_NEG | honesty_neg | 这两张是完全无关的图像吗 | 0 |
| GRAY_TRAP | trap | 第二张整体几乎无彩色、接近黑白吗 | =1 if measured cf<8 else 0 |

## Scoring rule (critique-corrected)
1. **Reliability**: parse ok (parse_codes) ∧ ANCHOR=1 ∧ HON_POS=1 ∧ HON_NEG=0 ∧ GRAY_TRAP matches measured.
   F⊕R contradiction = **both-1 ONLY** (claims good AND broken). **both-0 PASSES** (honest "clean but
   plain" middle — the twice-burned lesson; counting veto both-0 as contradiction systematically discards
   restrained non-garish renders → biases toward loud edits). contradiction tolerance ≤1 → else reask@temp0.
2. **Applicability skips**: drop portrait dims when no face; drop chroma dims (WBCAST/MEMORY_COLOR/SKINTONE/
   SAT_CLIP/INTENT_HARMONY) when variant is B&W; recompute veto_n / merit_n.
3. **Veto** = any veto-dim with **F=0 ∧ R=1** (defect asserted AND quality denied — not R=1 alone).
   Start CONSERVATIVE: only veto-dims **corroborated by a deterministic metric** hard-drop; subtle ones
   demote merit until per-dim FPR is measured on a gold set (κ-gated, like PRO_DROP_ENABLED).
4. **merit_score** = #(merit F=1 ∧ R=0) / merit_n.
5. **Keep**: reliable ∧ not vetoed ∧ merit_score ≥ τ (τ tuned on pilot). top-2 per source → SFT; whole
   group → DPO (chosen=kept-top, rejected=vetoed/low-merit). signed q for DPO = merit_score − (veto?·penalty).

## Deterministic CV complements (NOT VLM dims — the 35B can't see these at 2 small images, P0-3)
Run alongside; hard-drop on these is trustworthy and corroborates the veto dims:
- highlight-clip frac, crushed frac, overall over/under-exposure (luma).
- extreme oversaturation backstop (colorfulness ≫ source, kept loose to avoid killing vivid presets).
- banding/posterize detector (gradient-step histogram), halo/edge-fringe — replaces VLM ARTIFACT dim.
- ΔE direction cross-check: did saturation/exposure go UP — cross-checks the model's saturation bias.

## Cost (P2): 24 F/R + 4 control ≈ 28 bits/call. With both-0 fixed, reask rate drops (the real cost win).
Optional 2-phase: 6 veto questions first; run 6 merit questions only on veto=0 survivors.

## Dropped vs the raw synthesis (per critique)
- SKINTEX (waxy/over-smooth): color-grade presets don't retouch skin → N/A, dropped.
- ARTIFACT/banding as a VLM veto: not perceptible at 2 small images → moved to deterministic CV.
- MUDDY as separate veto: folded into PREMIUM (cheap/foggy = premium failure, not unusable) → merit.
- OVERSAT "internal micro-layering": not low-res visible → kept only the clipping/bleeding part (SAT_CLIP).
- DESTRUCT_TRAP no-op: front gate already drops no-ops → repurposed as the ΔE-direction CV cross-check.
