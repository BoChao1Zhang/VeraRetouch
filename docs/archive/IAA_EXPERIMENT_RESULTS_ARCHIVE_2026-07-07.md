# IAA Experiment Results Archive

Date: 2026-07-07

This archives the completed IAA benchmark results available before the handoff. No new experiments were run for this archive.

## Active Direction

- Active IAA model direction: ArtiMuse + Charm mixed scoring.
- QA analysis branch is historical only: `archive/qa-analysis-datagen-v2`.
- Existing QA archive: `docs/archive/QA_ANALYSIS_ARCHIVE_2026-07-04.md`.

## Artifact Roots

- ArtiMuse-10K eval root: `/home/bc/data/datasets/artimuse10k_eval`
- ArtiMuse-10K report: `/home/bc/data/datasets/artimuse10k_eval/REPORT.md`
- ArtiMuse-10K metrics summary: `/home/bc/data/datasets/artimuse10k_eval/metrics/summary.md`
- Photographer-IAA eval root: `/home/bc/data/datasets/photographer_iaa_benchmark`
- Photographer-IAA benchmark root: `/home/bc/data/datasets/photographer_iaa_benchmark/para_v1`
- Photographer-IAA report: `/home/bc/data/datasets/photographer_iaa_benchmark/REPORT.md`
- Photographer-IAA metrics summary: `/home/bc/data/datasets/photographer_iaa_benchmark/metrics/summary.md`

## ArtiMuse-10K Completed Results

Dataset: 1002 public test images. Ground truth is the public JSON `gt_score`, linearly mapped from 1-10 to 0-100.

| model | n | coverage | plcc | srcc | krcc | rmse | mae | bias | pred_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ArtiMuse | 1002 | 1002 | 0.6237 | 0.6150 | 0.4411 | 15.7188 | 12.3333 | 4.2065 | 51.6947 |
| OneAlign | 1002 | 1002 | 0.3038 | 0.3172 | 0.2159 | 23.4116 | 18.6635 | 6.3736 | 53.8617 |
| OneAlign-paper-iaa-prompt | 1002 | 1002 | 0.3051 | 0.3184 | 0.2165 | 23.6838 | 18.8793 | 6.4653 | 53.9534 |
| Charm-AVA-frequency | 1001 | 1001 | 0.2896 | 0.2951 | 0.1986 | 18.0579 | 14.1979 | 1.8980 | 49.3560 |
| AesExpert-LLaVA-7B | 1002 | 1002 | 0.1352 | 0.1602 | 0.1247 | 29.3028 | 23.5878 | 16.4091 | 63.8972 |
| HumanAesExpert-8B-metavoter-max1 | 1002 | 1002 | 0.0230 | 0.0200 | 0.0140 | 31.8935 | 27.2860 | 23.2172 | 70.7053 |

Completed prediction files:

- `/home/bc/data/datasets/artimuse10k_eval/predictions/artimuse.jsonl`
- `/home/bc/data/datasets/artimuse10k_eval/predictions/charm_ava_frequency.jsonl`
- `/home/bc/data/datasets/artimuse10k_eval/predictions/onealign.jsonl`
- `/home/bc/data/datasets/artimuse10k_eval/predictions/onealign_paper_iaa.jsonl`
- `/home/bc/data/datasets/artimuse10k_eval/predictions/aesexpert_llava.jsonl`
- `/home/bc/data/datasets/artimuse10k_eval/predictions/humanaesexpert_metavoter_max1.jsonl`

Notes:

- Charm-AVA-frequency missed one huge 6000x4008 image, `artimuse10k:5_174`, due to OOM even when retried alone.
- ArtiMuse used the official native `InternVLChatModel.score()` path with 448 resize, ImageNet normalization, BF16.
- ArtiMuse result matches the paper closely: local SRCC 0.6150 / PLCC 0.6237 versus paper SRCC 0.614 / PLCC 0.627.
- OneAlign local results around PLCC 0.304-0.305 are close to the AVA-only/generalization setting, not the target-dataset fine-tuned paper setting.

## Photographer-IAA Completed Results

Benchmark: 1800 rows, 9 categories, 8 modes, 25 images per category/mode.

| model | n | coverage | plcc | srcc | krcc | rmse | mae | bias | pred_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Charm-PARA-random | 1800 | 1800 | 0.9782 | 0.9425 | 0.8048 | 4.8228 | 3.6835 | 1.6099 | 29.9412 |
| OneAlign | 1800 | 1800 | 0.8491 | 0.4199 | 0.2886 | 14.9007 | 12.2296 | -9.2810 | 19.0504 |
| AesExpert-LLaVA-7B | 1800 | 1800 | 0.8004 | 0.6038 | 0.4714 | 16.0653 | 12.4553 | 0.2464 | 28.5778 |
| ArtiMuse | 1800 | 1800 | 0.8555 | 0.8343 | 0.6494 | 10.9709 | 7.2202 | 1.6081 | 29.9395 |
| HumanAesExpert-8B-metavoter-max1 | 1800 | 1800 | 0.4152 | 0.5107 | 0.3588 | 39.9023 | 34.0539 | 33.4044 | 61.7358 |

Completed prediction files:

- `/home/bc/data/datasets/photographer_iaa_benchmark/predictions/charm_para_random.jsonl`
- `/home/bc/data/datasets/photographer_iaa_benchmark/predictions/artimuse.jsonl`
- `/home/bc/data/datasets/photographer_iaa_benchmark/predictions/onealign.jsonl`
- `/home/bc/data/datasets/photographer_iaa_benchmark/predictions/aesexpert_llava.jsonl`
- `/home/bc/data/datasets/photographer_iaa_benchmark/predictions/humanaesexpert_metavoter_max1.jsonl`

Notes:

- Photographer-IAA currently has Charm-PARA-random, not final Charm-AVA-frequency.
- AesExpert direct vLLM serving failed with local vLLM/Transformers because config model type `llava_llama` was not recognized; native LLaVA fallback was used.
- UNIAA-LLaVA was not evaluated because no official/public checkpoint was found from the tried names/mirrors.

## Speed And VRAM Notes

- ArtiMuse vLLM throughput has not been validated. Existing ArtiMuse speed smoke log is native, not vLLM: 5 Photographer-IAA images finished with final tqdm rate about 3.27 img/s after loading.
- VRAM budget reference: `docs/DATABUILD_VRAM_BUDGET_2026-07-05.md`.
- Single-model benchmark runner observed/estimated peaks: OneAlign 19GB, HumanAesExpert 16GB, ArtiMuse 15GB, AesExpert 14GB, Charm <2GB.
- source_qa MixedIAA budget: ArtiMuse + Charm + pyiqa small models about 20GB when loaded together.
- A full build vLLM replica reserves about 83.2GB on an H100 97.9GB card, so MixedIAA should be run off-peak or with vLLM memory utilization reduced.

## Missing Or Not Final

- ArtiMuse+Charm mixed metrics were not found as completed benchmark artifacts.
- Photographer-IAA Charm-AVA-frequency was not found as a completed final result; only Charm-PARA-random is archived above.
- QA-AES questionnaire baseline metrics were not found as completed benchmark artifacts for ArtiMuse-10K or Photographer-IAA. The reliable baseline definition for future comparison is `pred_score_0_100 = merit_frac * 100` on reliable records with a documented benchmark_id mapping.
