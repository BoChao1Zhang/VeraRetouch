# LUT Renderer Online Calibration Pilot

This directory compares two style renderers on the same 4,000 native 3D LUTs:

- Large Shared-Geometry CGLUT-32 from the GLUT paper and the repository protocol.
- VeraRetouch's native pretrained `ConditionalMLPDecoder`, calibrated with one
  trainable 2,688D latent per LUT.

For every training occurrence, the data loader deterministically samples a held-in
source image for a LUT. The target is rendered online from the canonical
`grid[b,g,r]` LUT, never read from a pre-rendered target. Both methods also receive
the complete native LUT lattice as an auxiliary objective, so a style is not judged
only on colors occurring in one photograph.

## Scope

This is a renderer-calibration pilot informed by
`docs/LUT_RENDERER_EXPERIMENT_PROTOCOL.md`, not a protocol-complete P0-P6 run. Its
pre-registered deviations are:

- 4,000 native `.cube` LUTs only; the 581 baked LUTs are excluded.
- All 4,000 LUTs are training/seen styles; there is no unseen-style claim.
- Exactly 500 held-out source images replace the 10k seen-style test panel.
- It trains renderers only: no mask gate, VLM cache, adaptor SFT, or three-seed
  significance claim.
- Natural-image pairs are synthesized online in addition to full-grid supervision.
- Vera preset latents start at zero because this pilot does not build the protocol's
  five-reference teacher prototypes. The official decoder weights are loaded and
  jointly calibrated.

The CGLUT local affine is represented as an identity-centered residual,
`(M_i - I)x + b_i`, added to the global affine. This is algebraically equivalent in
capacity to the paper's local affine, makes the protocol's identity initialization
well-defined, and avoids the paper text's otherwise contradictory double-identity
initial output.

## Reproducible Run

The launcher restricts CUDA visibility to physical GPU 1:

```bash
bash experiments/lut_renderer_pilot/run.sh
```

Individual stages:

```bash
/home/bc/miniconda3/bin/python -m experiments.lut_renderer_pilot.prepare \
  --config experiments/lut_renderer_pilot/config.yaml
CUDA_VISIBLE_DEVICES=1 /home/bc/miniconda3/bin/python \
  -m experiments.lut_renderer_pilot.train --config experiments/lut_renderer_pilot/config.yaml \
  --model cglut
CUDA_VISIBLE_DEVICES=1 /home/bc/miniconda3/bin/python \
  -m experiments.lut_renderer_pilot.train --config experiments/lut_renderer_pilot/config.yaml \
  --model vera
CUDA_VISIBLE_DEVICES=1 /home/bc/miniconda3/bin/python \
  -m experiments.lut_renderer_pilot.evaluate --config experiments/lut_renderer_pilot/config.yaml
```

Use `--smoke` on either training command for one tiny update. Outputs include frozen
manifests, configs, checkpoints, JSONL training logs, per-sample metrics, aggregate
metrics, and qualitative comparison sheets under the configured `output_root`.

