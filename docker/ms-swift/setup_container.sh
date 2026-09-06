#!/bin/bash
# EPR-052 ENG-2：容器内环境安装（只改容器写层，不改宿主）。在容器内执行：bash /workspace/VeraRetouch/docker/ms-swift/setup_container.sh
set -euo pipefail
MS=/data/runs/epr052_rl/ms-swift          # 宿主 /home/bc/data/runs/epr052_rl/ms-swift，git clone 自 236a1f19
OUT=/workspace/VeraRetouch/docker/ms-swift
echo "[setup] image swift/vllm/torch before:"; python -c "import swift,torch;print('swift',swift.__version__,'torch',torch.__version__)"; python -c "import vllm;print('vllm',vllm.__version__)" || true
git config --global --add safe.directory $MS; cd $MS && git rev-parse HEAD | tee $OUT/MS_SWIFT_COMMIT.txt
pip uninstall -y ms-swift >/dev/null 2>&1 || true
pip install -e . 2>&1 | tail -3
pip install "transformers==4.57.1" "qwen_vl_utils>=0.0.14" "trl>=0.26,<1.0" 2>&1 | tail -5
echo "[setup] versions after:"
python -c "import swift;print('swift', swift.__version__)"   # main@236a1f19 的 CLI 无 --version（KeyError）
python - <<'PY'
import importlib.metadata as m, torch
for p in ["ms-swift","transformers","trl","peft","qwen_vl_utils","vllm","accelerate","datasets","deepspeed","flash_attn","torch","torchvision"]:
    try: print(f"{p}=={m.version(p)}")
    except Exception as e: print(f"{p}: MISSING")
print("cuda", torch.cuda.is_available(), torch.cuda.device_count(), torch.version.cuda)
PY
pip check 2>&1 | head -20 || true
pip freeze > $OUT/FREEZE.txt && echo "[setup] wrote $OUT/FREEZE.txt ($(wc -l < $OUT/FREEZE.txt) lines)"
