# Dataset-Build Runbook — fire-ready GPU schedule (2×H100)

Goal: when the GPU heartbeat (`/home/bc/data/datasets/_scratch/gpu_heartbeat.sh`) signals idle,
launch a schedule that keeps **both** H100s saturated until the 1M Direction-A set is built.

## GPU layout decision — "mirror" (both cards fully used)
Each card runs a self-contained vertical pipeline:
- **GPU0**: vLLM `qwen3-vl-8b` @ `:8001`  +  VeraRetouch renderer (orchestrator `--shard 0/2`)
- **GPU1**: vLLM `qwen3-vl-8b` @ `:8002`  +  VeraRetouch renderer (orchestrator `--shard 1/2`)

vLLM `gpu-memory-utilization=0.55` (~53 GB) leaves ~40 GB/card for the renderer (chunked 512² tiles)
+ raw-decode buffers. The orchestrator is CPU/IO heavy too (decode, recipe parse, C_GT PNG IO),
so cleaning (vLLM) and rendering overlap → both cards stay hot. SAM3 cannot co-exist with the
renderer in one process (transformers 5.2 vs llava), so SAM3 is a **separate bounded pass** (below).

## Bottleneck
VeraRetouch teacher render ≈ 1–3 samples/s/GPU (autoregressive VLM decode) is the long pole;
vLLM cleaning overlaps with it. ⇒ ~740K (Wave 1) ≈ 740k / (≈2.5/s × 2) ≈ **~40 h wall** with the
mirror layout (resumable, shardable to more workers if a 3rd/4th process fits). Recipe-based ⇒
**no before/after pixels are stored** (only recipe + C_GT PNG + JSONL ≈ 26 GB total at 1M).

## Stream → SAM3 dependency
| needs SAM3? | streams | budget |
|---|---|---|
| **NO** (launch immediately) | S1 (degrade, exact mask), S5/S6/S7 (global) | **740,000** |
| YES (Wave 2, after precompute) | S2, S3 (mask=sam3) | 220,000 |
| conditional | S4 (ppr10k masks; corrupt → folds into S1) | 40,000 |

---

## ORDERED TASK QUEUE (run top-to-bottom when GPUs free)

### Phase 0 — pre-flight (CPU, already done / no GPU)
- [x] TAD66K extracted → `_scratch/TAD66K` ; [x] registry indexes in `out_root` ; [x] rawpy decode verified.
- [ ] (lazy) extract award `.rar`/`.zip` packs on demand; PPR10K masks stay `available:false`.

### Phase 1 — SMOKE (≈3 min, validate end-to-end on tiny N; single vLLM)
```bash
cd /home/bc/VeraRetouch
bash dataset_build/launch_dual.sh smoke      # 1 vLLM @8001(GPU0) + renderer on GPU1, --pilot 60 --stream S1,S5,S6,S7 --out-suffix _smoke
```
Then inspect: shard JSONL records, `rejects.jsonl` reasons, a few C_GT PNGs, contact sheet. Fix any real-run bug. (This is the first time real weights touch real data — expect to iterate here.)

Contract gate:
```bash
python -m dataset_build.audit --config dataset_build/config.yaml --out-root /home/bc/data/datasets/vera_directionA_1M_smoke
python -m dataset_build.stage0_probe /home/bc/data/datasets/vera_directionA_1M_smoke/shards/S1/*.jsonl \
  /home/bc/data/datasets/vera_directionA_1M_smoke/shards/S7/*.jsonl \
  --config dataset_build/config.yaml --limit 64 --out /home/bc/data/datasets/vera_directionA_1M_smoke/stage0_probe.jsonl
```

### Phase 2 — WAVE-1 pilot (≈2.7K, dual-GPU mirror, full no-SAM3 mix)
```bash
bash dataset_build/launch_dual.sh pilot      # 2 vLLM(@8001/@8002) + 2 sharded workers, --pilot 2700 --stream S1,S5,S6,S7
```
Validate quality at scale → contact sheets per stream (`out_root/contact_sheets/`). Sign-off gate.
Also run `dataset_build.audit` and a larger S1/S7 `stage0_probe`; use the reported `er_recon_psnr`
drop-rate before locking final S1/S7 budgets.

### Phase 3 — WAVE-1 full (740K, dual-GPU mirror, resumable)  ← the long run
```bash
bash dataset_build/launch_dual.sh full S1,S5,S6,S7
# resumable: re-run the same line to continue; add more workers with --shard i/N if GPUs allow
```

### Phase 4 — SAM3 C_GT precompute for S2/S3 (dual-GPU, bounded, monetgpt_sam3 env)
*(prereq: registry indexes exist; run in the `monetgpt_sam3` env. `sam3_precompute.py`
defaults to the S2/S3 source pools and writes `sam3_cache/<path_key>/<concept>.png`.)*
```bash
# data-parallel over the unique S2/S3 source list, both cards:
CUDA_VISIBLE_DEVICES=0 $SAM3_PY -m dataset_build.sam3_precompute --shard 0/2 &
CUDA_VISIBLE_DEVICES=1 $SAM3_PY -m dataset_build.sam3_precompute --shard 1/2 &
wait   # writes concept PNGs into out_root/sam3_cache/<path_key>/<concept>.png
```

### Phase 5 — WAVE-2 full (S2,S3 = 220K, dual-GPU mirror, cached masks)
```bash
# set models.sam3.use_cache: true in the per-worker config first
bash dataset_build/launch_dual.sh full S2,S3
```

### Phase 6 — finalize
- Merge shard manifests → `out_root/manifest_index.jsonl`; dedup; final QA stats; train/val split.

---

## OPEN CODE TASKS
Phases 1–3 = 740K need neither SAM3 nor cache changes and can run first. Phase 4/5
code is present: `dataset_build/sam3_precompute.py` writes the cache in the SAM3 env,
and `run.py` builds `mask_cache.CachedMasker` when `models.sam3.use_cache:true`.

## Monitoring during the runs
- `tail -f out_root/logs/*.log` ; per-shard manifest counts ; `nvidia-smi dmon` to confirm both cards hot.
- Disk: C_GT PNGs dominate; ~26 GB @ 1M. Source extractions (TAD66K/awards) live in `_scratch`.
