# IAA Benchmark Direction

Date: 2026-07-04

The active IAA scoring direction is:

- ArtiMuse for expert-aligned aesthetic scoring and ArtiMuse-10K compatibility.
- Charm as a complementary ViT-based IAA scorer.

Recent benchmark artifacts:

- Photographer-IAA PARA benchmark: `/home/bc/data/datasets/photographer_iaa_benchmark/para_v1`
- ArtiMuse-10K public test eval: `/home/bc/data/datasets/artimuse10k_eval`

Current ArtiMuse-10K headline:

- ArtiMuse: PLCC 0.6237, SRCC 0.6150 on 1002 public test images.
- Charm-AVA-frequency: PLCC 0.2896, SRCC 0.2951 on 1001/1002 images.

Implementation notes:

- ArtiMuse currently runs through the official native `InternVLChatModel.score()`
  path in `dataset_build/iaa_benchmark/run_artimuse_predictions.py`.
- Charm currently runs through `dataset_build/iaa_benchmark/run_charm_predictions.py`
  with AVA large checkpoint and deterministic frequency patch selection.
- Treat vLLM serving for ArtiMuse as unvalidated until the custom score head is
  proven compatible with the OpenAI-compatible vLLM path.
