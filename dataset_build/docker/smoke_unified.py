"""In-image self-test for the unified core image (Dockerfile.unified).

Validates that vLLM's base image + the added deps can host SAM3 + the LLaVA/Qwen2
teacher renderer + pyiqa on ONE env (transformers 5.7 / torch 2.11), without
disturbing torch/transformers/vllm. Run INSIDE the container with the repo +
/home/bc/data bind-mounted and cwd=/home/bc/VeraRetouch:

    docker exec <ctr> bash -lc 'cd /home/bc/VeraRetouch && /usr/bin/python3 dataset_build/docker/smoke_unified.py'

Validated 2026-06-16: SAM3 model load OK; renderer load+forward OK; pyiqa OK;
renderer parity vs base(tf4.57) = max|Δ|=2/255, PSNR 56.85 dB. See
UNIFIED_CONCURRENCY_DESIGN_v2 §10.
"""
import os, sys, time, traceback
# make the repo root importable regardless of how this nested script is invoked
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
IMG = "/home/bc/data/datasets/_scratch/TAD66K/cathassos33455152516.jpg"


def main() -> int:
    import torch, transformers
    print(f"env: transformers {transformers.__version__}  torch {torch.__version__}")
    rc = 0

    # 1) SAM3 model (built-in transformers 5.7 classes). Processor/tokenizer is a
    #    separate, currently-broken data artifact -> model-only load here.
    try:
        from transformers import Sam3VideoModel
        t0 = time.time()
        vm = Sam3VideoModel.from_pretrained("/home/bc/data/models", local_files_only=True,
                                            dtype=torch.bfloat16).eval().to("cuda")
        n = sum(p.numel() for p in vm.detector_model.parameters()) / 1e6
        print(f"[sam3]     model load OK in {time.time()-t0:.1f}s  detector≈{n:.0f}M")
    except Exception:
        rc = 1; print("[sam3] FAIL"); traceback.print_exc()

    # 2) VeraRetouch / LLaVA teacher renderer (load + one forward).
    try:
        t0 = time.time()
        from dataset_build.render import VeraRetouchRenderer
        from dataset_build.contracts import PARAM_KEYS
        r = VeraRetouchRenderer(model_path="/home/bc/data/models/VeraRetouch", device="cuda")
        r._ensure_loaded()
        params = {k: {"value": 0.0} for k in PARAM_KEYS}
        out = r.render([IMG], [params], batch_size=1)[0]
        print(f"[renderer] load+forward OK in {time.time()-t0:.1f}s  out={None if out is None else (out.shape, str(out.dtype))}")
    except Exception:
        rc = 1; print("[renderer] FAIL"); traceback.print_exc()

    # 3) pyiqa NR-IQA.
    try:
        import pyiqa
        for name in ("musiq", "clipiqa+"):
            m = pyiqa.create_metric(name, device="cuda")
            print(f"[pyiqa]    {name}: {float(m(IMG)):.3f}")
    except Exception:
        rc = 1; print("[pyiqa] FAIL"); traceback.print_exc()

    print("SMOKE OK" if rc == 0 else "SMOKE FAILED")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
