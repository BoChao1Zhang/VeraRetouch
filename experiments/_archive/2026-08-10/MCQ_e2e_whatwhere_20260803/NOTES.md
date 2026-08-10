# MCQ-E2E: MetaCanvas-style queries for what / where

Date: 2026-08-03

## Questions

This experiment has two independent, end-to-end supervised tracks.  They use
the same VLM/query implementation but do **not** pretend that one dataset has
both labels.

1. **WHAT / LUT (GPU 0).** Can `I_in + instruction` be mapped through a
   spatial canvas into a compact N=48 GLUT whose baked 33^3 cube represents
   the requested preset, including held-out preset IDs?
2. **WHERE / w14 (GPU 1).** Can the same spatial canvas, followed by eight
   small task queries, predict the fixed 14-dimensional basis coefficients and
   recover the requested region under a fixed global readout?

G1--G4 are global-only and contain no masks.  They supervise the first
question only.  D-CONSTRUCT supplies masks and fitted w14 targets for the
second question.  A positive result on both tracks is evidence that the
interface can carry both kinds of information; it is not evidence that the
G1--G4 rows themselves contain spatial supervision.

## Paper-grounded architecture

MetaCanvas (arXiv:2512.11464v1, Sections 3.2--3.4) appends a 2D canvas after
the multimodal context, uses MRoPE, aligns it with a vanilla Transformer, and
fuses it patch-by-patch into the downstream latent through a zero-initialised
residual.  Its small T2I experiment uses 16x16=256 canvas tokens; Table 3 shows
that removing the spatial connector or fusing before patchification hurts.

The local VLM is Llava-Qwen2 rather than Qwen2.5-VL and has no multimodal
RoPE.  The faithful, explicit approximation used here is:

```
I_in + instruction
  -> frozen VeraRetouch backbone + rank-32 LoRA
  -> append 16x16 learned query grid after the instruction
  -> add learned Fourier(x,y) coordinates
  -> route query/image-token pairs from L11/L17/L23
  -> patch-wise query + same-cell image-token fusion
  -> one bidirectional Transformer block + zero-init residual
  -> task decoder
       LUT: 48 primitive queries + 1 global query -> 1,116 GLUT numbers
       w14: 8 task queries -> 14 coefficients
```

The 1D `metaquery` control uses eight learned queries, no 2D coordinates and
no patch-wise image fusion.  Everything downstream, including LoRA rank,
optimiser, renderer, data and step budget, is identical.

The VLM, visual tower and projector base weights are frozen.  LoRA, canvas
queries, layer router, connector and task decoder are trained from the final
renderer/mask loss.  This is end-to-end task supervision in the same sense as
MetaCanvas's optional MLLM-LoRA arm; it is not a frozen-feature probe.

## Data and leakage discipline

### LUT track

- Inputs: normal-confidence winners from prod-g1/g2/g3/g4.
- True `I_in` is ranged-read from the indexed image-bank tar.  `.in.jpg` is
  forbidden because it is only a preview.
- Target: `recipe.preset_id` joined to the canonical D-CUBE 33^3 table.
- Train: S-train x P-train.
- Selection: S-val x P-train plus S-val x P-val.
- Final untouched test: S-test x P-test.
- The six g4 source IDs missing from the split authority are dropped, never
  assigned ad hoc.

### w14 track

- Inputs: D-CONSTRUCT train/val source-disjoint manifests.
- Standard rows use the real templated instruction and fitted `mask` w14.
- L6 additionally contributes two instruction-specific examples (`rega` and
  `regb`) with separately fitted `maska`/`maskb` w14 targets.
- `alpha` is capped at 100 for regression conditioning; mask reconstruction
  still uses the fixed global `sigmoid(6s)` readout and no per-image
  normalisation.

No new sample assets are materialised as small-file datasets.  The LUT track
uses the existing durable indexed-tar assets by random access; experiment
cache contains only bounded, rebuildable manifests and checkpoints.

## Losses and gates

### LUT

- main: L1 over half uniform-cube and half natural-image colours;
- prior: squared raw renderer outputs;
- metrics: DeltaE00 p50/p90, PSNR, variance ratio, instruction shuffle,
  cross-image shuffle, held-out P-test, 33^3 tetrahedral bake readback;
- positive structural evidence: canvas beats metaquery at the same step budget
  on P-test and neither collapses (`variance_ratio > 0.6`);
- absolute convergence gate remains DeltaE00 p50 < 1.5 and p90 < 3.0.

### w14

- warm-up: robust-standardised Huber(w14), with direction cosine enabled after
  50 regression-only startup steps so the zero-initialised readout does not
  create a singular first-step cosine gradient;
- main: fixed-readout BCE mask loss + a residual coefficient loss;
- metrics: soft-IoU, ordinary mask AUC, L6 AUC_target, paired instruction
  shuffle delta, coefficient RMSE, alpha by level;
- positive gate: AUC_target >= 0.65 and paired shuffle degradation above
  machine zero; offline-fit soft-IoU is an upper bound, not a model score.
- checkpoint selection evaluates the full 410-item source-disjoint inner pool;
  prefix truncation is forbidden because it leaves only one L6 A/B pair.

Both tracks report canvas-query mean cosine and effective rank.  Mean cosine
near 1 or effective rank near 1 is query replication and invalidates any
"spatial canvas" interpretation even if task loss falls.

The training DataLoader uses a dedicated fixed generator shared by both
architecture arms.  This is essential for the LUT track, where 3,000 steps
cover only 39.37% of the 45,724-row train pool; relying on the global RNG after
architecture-specific parameter initialisation would expose the two arms to
different subsets and confound the structural comparison.
