# IAA Final Evaluation Report

Date: 2026-07-08
Direction: ArtiMuse + Charm mixed IAA (QA-AES questionnaire is historical baseline only).

## 1. Summary Tables

Metrics: PLCC (Pearson), SRCC (Spearman), KRCC (Kendall tau-b). Coverage = matched
rows / benchmark rows. Higher is better for the three correlations.

### ArtiMuse-10K (1002 public test images, GT = public `gt_score` mapped 1–10 → 0–100)

| model | PLCC | SRCC | KRCC | coverage | provenance |
| --- | --- | --- | --- | --- | --- |
| ArtiMuse | 0.6237 | 0.6150 | 0.4411 | 1002/1002 | reused |
| **ArtiMuse+Charm mixed (0.75/0.25)** | **0.6205** | **0.6146** | **0.4381** | 1001/1002 | **new (this run)** |
| Charm-AVA-frequency | 0.2896 | 0.2951 | 0.1986 | 1001/1002 | reused |
| QA-AES questionnaire baseline | 0.3250 | 0.3066 | 0.2191 | 997/1002 | **new (this run)** |
| OneAlign (context) | 0.3038 | 0.3172 | 0.2159 | 1002/1002 | reused |

### Photographer-IAA (1800 rows, GT = `gt_aesthetic_mean_0_100`)

| model | PLCC | SRCC | KRCC | coverage | provenance |
| --- | --- | --- | --- | --- | --- |
| ArtiMuse | 0.8555 | 0.8343 | 0.6494 | 1800/1800 | reused |
| Charm-PARA-random (NOT final setting) | 0.9782 | 0.9425 | 0.8048 | 1800/1800 | reused |
| Charm-AVA-frequency (final setting) | — | — | — | 622/1800 partial | **incomplete — see §4** |
| ArtiMuse+Charm mixed (0.75/0.25) | — | — | — | — | **blocked on Charm-AVA-freq** |
| QA-AES questionnaire baseline | 0.8876 | 0.6522 | 0.5311 | 1792/1800 | **new (this run)** |
| OneAlign (context) | 0.8491 | 0.4199 | 0.2886 | 1800/1800 | reused |
| AesExpert-LLaVA-7B (context) | 0.8004 | 0.6038 | 0.4714 | 1800/1800 | reused |

## 2. Mixed vs QA-AES verdict

Rule: mixed score = `0.75 * ArtiMuse + 0.25 * Charm` on shared `benchmark_id`, evaluated
only where both models produced a valid score.

**ArtiMuse-10K — mixed clearly beats QA-AES on both metrics.**
- PLCC: mixed 0.6205 vs QA-AES 0.3250 → mixed wins by +0.2955.
- SRCC: mixed 0.6146 vs QA-AES 0.3066 → mixed wins by +0.3080.
- Coverage is near-full and comparable on both sides (mixed 1001/1002, QA-AES 997/1002),
  so the verdict is not a coverage artifact.

**Photographer-IAA — verdict pending.** The final Charm-AVA-frequency run did not finish
(§4), so the Photographer mixed score was not computed. What can be stated with the
current data, coverage-qualified:
- QA-AES on Photographer: PLCC 0.8876 / SRCC 0.6522 (coverage 1792/1800).
- The QA-AES PLCC is inflated by a heavily zero-skewed prediction distribution:
  1282 of 1792 scored images have `merit_frac = 0` (pred = 0). PLCC rewards the strong
  0-vs-nonzero separation; SRCC 0.6522 is the more honest ranking signal.
- ArtiMuse alone already reaches PLCC 0.8555 / SRCC 0.8343 on Photographer, higher than
  QA-AES on SRCC, so an ArtiMuse-weighted mix is expected to match or beat QA-AES on
  SRCC — but this is an expectation, not a measured result. Do not report a Photographer
  mixed-vs-QA-AES conclusion until Charm-AVA-frequency is completed.

## 3. Provenance: reused / new / not done

**Reused existing prediction + metric files (not re-run):**
- ArtiMuse-10K: ArtiMuse, Charm-AVA-frequency, OneAlign, OneAlign-paper-iaa-prompt,
  AesExpert-LLaVA, HumanAesExpert.
- Photographer-IAA: ArtiMuse, Charm-PARA-random, OneAlign, AesExpert-LLaVA, HumanAesExpert.

**New this run:**
- ArtiMuse-10K `ArtiMuse+Charm 0.75/0.25` mixed predictions + metrics
  (`predictions/artimuse_charm075_025.jsonl`, `metrics/artimuse_charm075_025.metrics.json`).
- QA-AES questionnaire baseline on **both** benchmarks
  (`predictions/qa_aes_merit_frac.jsonl`, `metrics/qa_aes_merit_frac.metrics.json` on each).

**Not completed:**
- Photographer-IAA Charm-AVA-frequency: stopped at 622/1800 and killed on request because
  the 40-worker CPU preprocessing saturated the shared workstation (load ~73, ~1.8 GB/s
  read). The 622 saved rows are valid (0 errors) but insufficient for a final metric.
- Photographer-IAA ArtiMuse+Charm mixed: blocked by the above (needs the full
  Charm-AVA-frequency prediction file).

## 4. Why Photographer Charm-AVA-frequency is unfinished

The Charm tokenizer's importance computation is a CPU bottleneck (~40 s/image regardless
of machine), so 1800 images is ~20 CPU-hours single-threaded. To fit a time budget it was
parallelized with 40 CPU preprocessing workers feeding one GPU scorer
(`run_charm_predictions_mp.py`). Two obstacles:
1. On the remote 8×4090 box (`exp-remote`) the original multiprocessing pool deadlocked
   after a per-image CUDA OOM corrupted the pool (large tensors through the pool pipe +
   `received 0 items of ancdata`). Fixed by having workers save tensors to disk and return
   a path, plus per-image GPU OOM try/except.
2. Moved to the local H100 (GPU 0). The 40-worker preprocessing made the shared
   workstation unusably laggy (CPU load + heavy read I/O), and the run was killed by user
   request at 622/1800. An intermediate reboot had already reset it once (200 → resume).

To finish it later (cheapest, least local impact): run on `exp-remote` GPU 0 with
`--resume`; the remote already has the `lens` conda env with `Charm-tokenizer`,
`ml_collections`, the `Ava_large_charm.pth` checkpoint, and the `para_v1` images synced.
Command shape:

```
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 ~/miniconda3/envs/lens/bin/python \
  dataset_build/iaa_benchmark/run_charm_predictions_mp.py \
  --input  ~/data/datasets/photographer_iaa_benchmark/model_inputs/generic_image_score.jsonl \
  --output ~/data/datasets/photographer_iaa_benchmark/predictions/charm_ava_frequency.jsonl \
  --checkpoint ~/data/models/Charm/Ava_large_charm.pth \
  --training-dataset ava --patch-selection frequency --backbone facebook/dinov2-large \
  --device cuda:0 --workers 12 --resume
```

Then compute the mix and metrics locally:

```
python dataset_build/iaa_benchmark/mix_predictions.py \
  --pred-a .../predictions/artimuse.jsonl --pred-b .../predictions/charm_ava_frequency.jsonl \
  --model-name "ArtiMuse+Charm-0.75/0.25" --output .../predictions/artimuse_charm075_025.jsonl
python dataset_build/iaa_benchmark/evaluate_predictions.py \
  --metadata .../para_v1/metadata.csv --predictions .../predictions/artimuse_charm075_025.jsonl \
  --output .../metrics/artimuse_charm075_025.metrics.json --model-name "ArtiMuse+Charm-0.75/0.25"
```

## 5. QA-AES questionnaire baseline — method and mapping

- Runner: `dataset_build/iaa_benchmark/run_qa_aes_predictions.py`, calling
  `source_qa.qa_runner.run_aes` against the card-1 vLLM server
  (`qwen3_5-35b-a3b`, OpenAI-compatible, `http://localhost:8003/v1`).
- **benchmark_id mapping rule:** the questionnaire is run directly on the image paths in
  each benchmark's `metadata.csv` (ArtiMuse-10K `image_path`; Photographer-IAA
  `absolute_image_path`), keyed by `benchmark_id`. **No DB join** — the `vera_source_qa`
  `assets` table has zero AES runs on the `artimuse10k` and `para` corpora, so a
  join-based mapping was not possible and would have produced no rows. Running the
  questionnaire fresh on the benchmark images is the only reliable mapping and is what was
  done.
- `max_face_frac` is unavailable for benchmark images, so every image used the no-face AES
  questionnaire variant (`aes_pos_map(has_face=False)`).
- Score: only records with `reliable == true` and non-null `merit_frac` are scored;
  `pred_score_0_100 = merit_frac * 100`.
- Coverage / drops:
  - ArtiMuse-10K: 997/1002 scored; 5 unreliable (4 honesty-trap, 1 parse-dup).
  - Photographer-IAA: 1792/1800 scored; 8 unreliable (7 parse-missing, 1 parse-dup).
- Distribution caveat (Photographer): `merit_frac` is a fraction of merits over 9 probes,
  so it is discretized to multiples of 1/9 and strongly zero-inflated on Photographer
  (1282/1792 = 0). This is why its PLCC (0.8876) far exceeds its SRCC (0.6522); prefer
  SRCC/KRCC for ranking comparisons.

## 6. Background notes (carried from prior handoff, unchanged)

- ArtiMuse-10K ArtiMuse local SRCC 0.6150 / PLCC 0.6237 matches the paper Table 3
  (SRCC 0.614 / PLCC 0.627) — no reproduction gap. Uses native
  `InternVLChatModel.score()` (448 resize, ImageNet norm, BF16); vLLM path not validated.
- OneAlign's low ArtiMuse-10K numbers reflect the off-the-shelf / AVA-generalization
  context, not target-dataset fine-tuning; paper-style prompt did not move results.
- Charm's existing Photographer result is Charm-PARA-random, NOT the final
  Charm-AVA-frequency, and PARA-random is essentially in-distribution for PARA (hence the
  0.97 PLCC) — it is not a fair cross-dataset generalization number.

## 7. Artifact locations

- ArtiMuse-10K root: `/home/bc/data/datasets/artimuse10k_eval/`
  (`predictions/`, `metrics/`, `metrics/summary.md`).
- Photographer-IAA root: `/home/bc/data/datasets/photographer_iaa_benchmark/`
  (`predictions/`, `metrics/`, `metrics/summary.md`; partial
  `predictions/charm_ava_frequency.jsonl` = 622 rows).
- Scripts (new/modified this run): `dataset_build/iaa_benchmark/mix_predictions.py`,
  `run_qa_aes_predictions.py`, `run_charm_predictions_mp.py`.
