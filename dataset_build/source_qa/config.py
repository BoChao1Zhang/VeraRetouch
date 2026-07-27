"""Configuration used only by retained source ingest, SAM3, OneAlign, and farm support."""
from __future__ import annotations

import os


OUT_ROOT = os.environ.get("VERA_OUT_ROOT", "/mnt/nfs/bc/data/builds")
SOURCE_INDEX = os.path.join(OUT_ROOT, "source_index.jsonl")
TAG_CACHE_DIR = os.path.join(OUT_ROOT, "tag_cache")

PG_DSN = os.environ.get("SOURCE_QA_PG_DSN", "postgresql:///vera_source_qa")

VLLM_BASE_URL = os.environ.get("SOURCE_QA_VLLM", "http://127.0.0.1:8003/v1")
VLLM_MODEL = "qwen3_5-35b-a3b"
VLLM_API_KEY = "EMPTY"
VLLM_IMAGE_LONGEDGE = 768
VLLM_CONCURRENCY = 16
VLLM_MAX_TOKENS = 1100

IAA_DEVICE = os.environ.get("SOURCE_QA_IAA_DEVICE", "cuda:0")

QA_ROOT = os.path.join(OUT_ROOT, "source_qa")
RENDER_STAGE = os.path.join(QA_ROOT, "render_stage")
LR_SERVER_URL = os.environ.get("SOURCE_QA_LR_URL", "http://127.0.0.1:8081")
LR_POLL_WAIT = 25
LR_JOB_TIMEOUT = 1200
LR_HTTP_TIMEOUT = 60
LR_MAX_CONCURRENCY = int(os.environ.get("SOURCE_QA_LR_CONCURRENCY", "6"))
RENDER_MAX_ATTEMPTS = 3
