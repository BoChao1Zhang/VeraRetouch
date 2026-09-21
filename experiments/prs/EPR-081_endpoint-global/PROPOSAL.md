# EPR-081: instruction-conditioned endpoint-only global continuation

Status: author-confirmed specification; training NOT launched.

## Confirmed information boundary (2026-09-21)

The author confirmed: retain the final user instruction, remove process information.
The model receives only the before image and final instruction, followed by one
shared eight-token readout group. Predict one continuous color code and apply it
globally (support identically one). The only supervised image is after.
No stage descriptions, stage markers, intermediate states, masks, local code
targets, prototype labels, code MSE, or InfoNCE may enter this continuation loss.
Global/style replay must also omit cached CoT, not just the former local stream.

## Matched budget

- Same pretrained global readout initialization as PIX/ENDPT2: EPR-071 R best@800,
  SHA256 4309e517e524fdf74a08881ad017b009036a4d217aad6adc5e94f825edfb9fbf.
- Same image identities and batch key order, style subset and all MMArt pairs;
  collapse each local chain to its before/after pair offline.
- 3,200 updates, effective batch 16, seed 20260918, cosine horizon 6,250,
  same optimizer and 1:1 replay sampling. Full-image pixel MAE only.
- The initialization is shared and may contain prior process learning. This
  experiment removes process information during continuation, not from the
  entire model's training history.

## Required isolation and tests before launch

1. Export a pair-only manifest with key, before, after, final instruction and
   source hashes. Keep trajectory construction outside the training loader.
2. Check endpoint equality against the existing float-state construction;
   do not silently substitute quantized observed PNGs for float loss inputs.
3. Test that adding/removing process fields cannot change model inputs or loss;
   reject manifests carrying such fields at the training boundary.
4. Assert exactly one readout group and no stage markers/assistant CoT.
5. Verify optimizer, schedule, batch keys and counts against matched controls.
6. Smoke-test finite forward loss, backward gradients, save/reload, and
   target-free global evaluation before the full run.

## Interpretation

Existing ENDPT2 is rollout-objective training WITH process inputs and is not
Endpoint-only. EPR-080 validly compares its objective to step-objective training.
EPR-081 is a global endpoint-only baseline; relative to six-stage systems it
changes both process supervision and execution architecture, so it measures
their combined contribution, not an isolated stage-loss effect.

## Scheduling

Do not interrupt existing jobs or overlap the queued exclusive efficiency test.
Do not claim a running experiment before smoke checks and an actual launch.
