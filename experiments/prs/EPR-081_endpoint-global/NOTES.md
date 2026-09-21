# EPR-081 execution record

## 2026-09-21: implementation and CPU preflight

Author confirmed final user instruction remains; all process information is removed
from continuation inputs. This is **global endpoint-only continuation**, not the
existing six-code rollout-objective control (ENDPT2).

### Information boundary

- `tools/epr081_export_pairs.py` is the only component allowed to read trajectory
  construction annotations. It reconstructs the original float32 before/after
  endpoints and exports packed NPZ shards. It does not use quantized observed PNGs
  as the pixel-loss input. The VLM image remains the same observed before view
  used by the matched controls.
- `epr081_pairs.py` accepts exactly key, final instruction, array shard/prefix and
  array digests. Unexpected fields (including CoT, supports, stage positions,
  intermediate states, LUT labels and code targets) are rejected.
- Both former local examples and global replay receive one eight-token readout
  group, without stage markers or generated assistant text. Style replay uses the
  rewritten request alone, without the old request-to-CoT scaffold. MMArt uses the
  original final request without the scaffold. Local uses the same tier-selected
  final user instruction as the control, without its six cached stage segments.
- The shared initialization is the R best@800 checkpoint (SHA in PROPOSAL.md).
  It can contain earlier process learning. This experiment isolates continuation
  information/architecture, not the full pretraining history.

### Matched controls

Offline sampler reconstruction matched **all 3,200 update key digests** against
`ENDPT2/full/run/steps.jsonl`, not merely the seed or source proportions.
The source pool is identical: style 15,000, MMArt 16,163, local chains 75,311.
Only pairs actually visited in this fixed 3,200-update budget need materialization;
the entire eligible pool and exact selection/order are preserved in `pool_audit.json`
and `batch_plan.json`.

Optimizer: shared initialization; effective batch 16; 3,200 updates; 6,250-step
cosine horizon and 100-step warmup; seed 20260918; AdamW betas (0.9,0.999), epsilon
1e-8, weight decay .01, LoRA LR 1e-4, head/readout LR 1e-3, gradient norm cap 1.
Loss: single global transformation, support one, un-clipped endpoint MAE at the
same flattened pixel stride 2. No code MSE or InfoNCE. Four-example microbatches
preserve effective batch size; they do not reintroduce a stage dimension.

### Tests completed

- Five CPU tests pass: strict schema, scaffold rejection, exactly one readout
  group/no stage tokens, all 3,200 schedule multipliers, and float32 shard
  round-trip/digest corruption detection.
- CPU preflight exported 32 examples (first global and local batches), then read
  every array back and checked its digest/dtype/shape. No float endpoint was
  silently replaced with an 8-bit rendering.
- Artifacts:
  `/home/bc/nfsvfs/bc/data/runs/epr081_endpoint_global_20260921/preflight_pairs/`.

### Resource-safe queue; not a completed training run

`tools/epr081_queue.py` waits for the existing efficiency queue to report
`finished`, then requires card 0 to be empty for four successive checks. It never
stops another process. The stages are:

1. Reconstruct the same 32 endpoints on the accelerator; compare to CPU float
   endpoints with max absolute tolerance 2e-6 and exact VLM-view equality.
2. Two-update model smoke: finite losses/gradients, nonzero head/readout/LoRA
   gradients, checkpoint round-trip, and target-free global rendering.
3. Full pair-only offline export on the same device.
4. Full 3,200-update training, saving every 400 updates and final checkpoint.

At this entry, GPU smoke and full training have **not** run. Queue state is the
source of truth; a waiting queue is not a started training job. A failure stops
the queue and preserves logs instead of launching the next stage.

The queue was started as user service `epr081-endpoint-queue.service` (initial PID
1259772). Verified state: `waiting_after_efficiency`, efficiency unfinished.
Status file: `/home/bc/nfsvfs/bc/data/runs/epr081_endpoint_global_20260921/queue.json`.

### Interpretation

Compared with the six-stage systems, this baseline changes both input process
information and execution architecture. Report it as the combined global
endpoint-only baseline, not as an isolated intermediate-loss ablation. ENDPT2
remains the controlled step-objective versus rollout-objective comparison with
shared process inputs.
