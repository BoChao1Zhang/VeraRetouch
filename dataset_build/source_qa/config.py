"""Central paths / model endpoints / thresholds / questionnaires for source-QA.

Source QA cleans the BUILD INPUTS (before any S1-S8 stream runs):
  * input images   (source_index.jsonl, 124k)  -> dedup + NR-IQA + questionnaire A/B
  * preset/LUT looks (recipe_index.jsonl, 12k)  -> metadata gate + REAL render + questionnaire C

Storage is PostgreSQL (psycopg3) — see db.py for why we left SQLite. Nothing here
loads weights at import; runners construct models lazily.

This module is the single source of truth for the questionnaires, gate thresholds
and metric directions; both the runners and the web UI read them from here.
"""
from __future__ import annotations

import os

# --- on-disk dataset artifacts (read-only inputs to QA) ----------------------
OUT_ROOT = "/home/bc/data/datasets/vera_directionA_1M"
SOURCE_INDEX = os.path.join(OUT_ROOT, "source_index.jsonl")
RECIPE_INDEX = os.path.join(OUT_ROOT, "recipe_index.jsonl")
TAG_CACHE_DIR = os.path.join(OUT_ROOT, "tag_cache")

# Cleaned indexes the BUILD consumes after QA (written by apply.py; the build's
# config.yaml storage.{source,recipe}_index points here, falling back to raw).
SOURCE_INDEX_CLEANED = os.path.join(OUT_ROOT, "source_index.cleaned.jsonl")
RECIPE_INDEX_CLEANED = os.path.join(OUT_ROOT, "recipe_index.qa.jsonl")

# --- QA-owned artifacts (writable) -------------------------------------------
QA_ROOT = os.path.join(OUT_ROOT, "source_qa")
THUMB_DIR = os.path.join(QA_ROOT, "thumbs")
PREVIEW_DIR = os.path.join(QA_ROOT, "preset_previews")

# --- PostgreSQL --------------------------------------------------------------
# Dedicated DB on the shared local PG16; library-level isolation from other apps.
PG_DSN = os.environ.get(
    "SOURCE_QA_PG_DSN",
    "postgresql://vera:vera@127.0.0.1:5432/vera_source_qa",
)

# --- vLLM (reuse the running multimodal Qwen3.5-35B reasoning server) ---------
# Default points at the vGate broker (Phase 0): least-outstanding routing over
# the live reason_g0:8001/reason_g1:8002 replicas (see dataset_build/core/broker).
# Rollback: set SOURCE_QA_VLLM=http://localhost:8002/v1 (or edit this default).
VLLM_BASE_URL = os.environ.get("SOURCE_QA_VLLM", "http://localhost:8003/v1")
VLLM_MODEL = "qwen3_5-35b-a3b"
VLLM_API_KEY = "EMPTY"
VLLM_IMAGE_LONGEDGE = 768          # downscale images sent to the judge
VLLM_CONCURRENCY = 16              # HTTP fan-out for LLM QA (cap to avoid engine overload)
# The 35B is a REASONING model: thinking ON emits a long CoT and the structured
# JSON gets truncated. QA wants the answer, not the CoT -> disable thinking.
VLLM_ENABLE_THINKING = False
# Headroom for the graded questionnaire (8 items + rationales-on-no/uncertain +
# caption). 700 was the truncation risk; 1100 leaves margin.
VLLM_MAX_TOKENS = 1100

# --- NR-IQA (pyiqa, GPU0) -----------------------------------------------------
IQA_DEVICE = os.environ.get("SOURCE_QA_IQA_DEVICE", "cuda:0")
IQA_METRICS = ["musiq", "clipiqa+", "niqe", "brisque"]   # + laplacian (cv2) in iqa.py
# Cheap deterministic detectors computed alongside IQA (closes QA-1/QA-6), all
# from established libraries (no hand-rolled algorithms): megapixels/longedge from
# the native decode (width/height are NULL in the index), noise_sigma via
# skimage.restoration.estimate_sigma, max_face_frac via cv2 Haar (portrait pool).
# Compression artifacts are covered by BRISQUE (already computed) + questionnaire
# B_comp, so no bespoke blockiness metric is hand-written.
IQA_EXTRA = ["laplacian", "noise_sigma", "megapixels", "longedge"]
# higher_is_better per metric (for UI + gate direction)
IQA_HIGHER_BETTER = {
    "musiq": True, "clipiqa+": True, "clipiqa": True, "maniqa": True,
    "niqe": False, "brisque": False, "laplacian": True,
    "noise_sigma": False,
    "megapixels": True, "longedge": True, "max_face_frac": True,
    "aesthetic_model": True, "aesthetic_vlm": True,
}
IQA_BATCH = 16
IQA_LONGEDGE = 1024                # downscale before IQA (keeps texture, bounds VRAM)
FACE_DETECT = True                 # run a cheap face detector on the portrait pool

# --- aesthetic scorer (reuse dataset_build/aesthetic.py weights) --------------
AESTHETIC_CLIP_DIR = "/home/bc/data/models/clip-vit-large-patch14"
AESTHETIC_MLP_PATH = "/home/bc/data/models/laion_aesthetic_sac_logos_ava1_l14_linearMSE.pth"

# --- source-image dedup (dedup.py) -------------------------------------------
DEDUP_PHASH_HAMMING = 6            # near-dup merge cutoff on the 64-bit DCT pHash
DEDUP_LSH_BANDS = 8               # 8 bands x 8 bits for O(n) candidate generation

# --- preset rendering (real-Lightroom render-QA) -----------------------------
# Real renders go through the already-running JarvisEvo Lightroom task server
# (FastAPI, reverse-connection: Win/Mac LrC clients poll it). We submit
# {probe photo, xmp preset} via /api/submit_task_with_files, long-poll
# /api/task_status, then GET /api/download_task_result. A real LR develop preset
# (XMP) includes local-mask corrections, so the mask presets render faithfully.
# Engine tiers, best-fidelity first: lrc = real Lightroom (source-of-record);
# darktable = deterministic Linux fallback (LUTs / param when no LR client up);
# lut_trilinear = numpy LUT pre-filter only, never QA-of-record.
LR_SERVER_URL = os.environ.get("SOURCE_QA_LR_URL", "http://127.0.0.1:8081")
LR_POLL_WAIT = 25                  # long-poll seconds per task_status (server caps ~25)
LR_JOB_TIMEOUT = 1200              # total seconds to wait for one render before giving up
LR_HTTP_TIMEOUT = 60               # per-HTTP read timeout (> any single short poll)
RENDER_ENGINE_PRIORITY = ["lrc", "darktable", "lut_trilinear"]
RENDER_STAGE = os.path.join(QA_ROOT, "render_stage")   # generated xmp / staged probes
RENDER_OUT_FMT = "jpg"             # LR client exports jpg; darktable tier can emit tif
RENDER_MAX_ATTEMPTS = 3
DARKTABLE_CLI = os.environ.get("SOURCE_QA_DARKTABLE_CLI", "darktable-cli")
# A FIXED probe set (asset_ids) so probe_id is stable and the render cache hits.
# Empty -> resolved deterministically on first stage2 and persisted to the
# qa_probe_set table (preset_qa.resolve_probe_set).
PRESET_PROBE_IMAGES: list[str] = []
PRESET_PROBE_COUNT = 6

# --- auto-gate thresholds -----------------------------------------------------
# ABSOLUTE hard floors (gate preamble; never relativized per corpus) + cheap
# deterministic tech floors (auto-drop before the 35B judge) + soft IQA bands
# (relative-mode tail). Human review is still the final word for keep/drop.
GATE = {
    # absolute hard floors (preamble). CONSERVATIVE starting values meant to catch
    # genuine thumbnails, NOT to nuke a structurally-low-res corpus. NOTE: tad66k
    # (the largest corpus) is web-scraped at ~800px / ~0.4MP, so a 1.5MP / 1024px
    # floor would drop ~100% of it — re-tune per corpus against a gold set, and
    # decide explicitly whether 800px is an acceptable restoration target.
    "min_megapixels": 0.30,          # ~670x450; below = thumbnail
    "min_megapixels_portrait": 0.50,
    "min_longedge": 640,
    "min_longedge_portrait": 800,
    "min_face_frac": 0.04,            # portrait pool: largest face must cover >=4% of frame
    # cheap deterministic tech floors -> auto 'drop' BEFORE the LLM (triage)
    "tech_musiq_drop_below": 25.0,
    "tech_sharp_drop_below": 30.0,
    # soft IQA floors (absolute mode + relative-mode bad tail)
    "musiq_drop_below": 30.0,
    "niqe_drop_above": 9.0,
    "brisque_drop_above": 65.0,
    "laplacian_drop_below": 40.0,
    "noise_sigma_drop_above": 18.0,       # starting guess
    # comfortable 'keep' band
    "musiq_keep_above": 55.0,
    "clipiqa_keep_above": 0.50,
    "aesthetic_vlm_keep_above": 5.5,      # starting guess; aesthetic now a soft keep vote
}

# Pass-rule thresholds for the graded questionnaire B (suitability).
QA_PASS = {
    "b_quality_min": 2,    # 0..3
    "b_comp_min": 2,
    "b_subject_min": 1,
    "b_face_min": 1,       # portrait pool only
}

# --- canonical questionnaires (single source of truth; llm_qa + UI share) -----
# Item types: 'yn' (binary hard-fail), 'yn3' (yes/no/uncertain; uncertain->review),
# 'grade' (integer 0-3). 'portrait_only' items are emitted only for the portrait pool.
QUESTIONNAIRE_A = {
    "title": "真实性 / 完整性门 (validity)",
    "pass_rule": "all_yes",
    "items": {
        "A1": {"q": "这是一张真实拍摄/扫描的自然照片吗？（非屏幕截图/海报/纯图形/拼图/AI 生成）", "type": "yn"},
        "A2": {"q": "画面无叠加污染吗？（无水印/平台 logo/文字条/边框/拼贴分割/二维码）", "type": "yn"},
        "A3": {"q": "主体完整、非极端裁切或纯色测试图/碎片吗？", "type": "yn"},
    },
}
QUESTIONNAIRE_B = {
    "title": "高画质 + 内容/场景适用性门 (suitability)",
    "items": {
        "B_safe": {"q": "内容安全合规吗？（无 NSFW/暴力/敏感）", "type": "yn"},
        "B_scene": {"q": "画面内容与场景标签 `{scene}` 一致吗？（场景为 any/空 时答『不确定』，不扣分）", "type": "yn3"},
        "B_quality": {"q": "作为退化-还原目标的画质等级：0=原生严重缺陷(明显压缩块/噪点/失焦/上采样痕迹) 1=勉强 2=良好 3=优秀。注意：已精修成品按其画质评分，不因看起来已调色而扣分。", "type": "grade"},
        "B_comp": {"q": "压缩/伪影等级：0=严重 JPEG 块/带状/色噪 1=可见 2=轻微 3=干净无伪影。", "type": "grade"},
        "B_subject": {"q": "主体/构图清晰度：0=无聊空镜/纯文字图 1=弱 2=较好 3=明确强主体。", "type": "grade"},
        "B_face": {"q": "人脸清晰、无明显塑料感/严重模糊吗？0=不可用 1=勉强 2=良好 3=优秀。", "type": "grade", "portrait_only": True},
    },
}
QUESTIONNAIRE_C = {
    "title": "预设 look 质量 (preset, on real renders)",
    "pass_rule": "all_yes",
    "items": {
        "C1": {"q": "after 是专业、克制、非破坏性的修饰吗？（非过曝/死黑/塑料感/严重偏色/halo）", "type": "yn"},
        "C2": {"q": "after 的风格与其声称的 style/scene_affinity 一致吗？（声称为空时答『不确定』）", "type": "yn3"},
        "C3": {"q": "该 look 有训练价值吗？（有明确可学的编辑意图，非近似 no-op 也非极端怪异）", "type": "yn"},
    },
}


def ensure_dirs() -> None:
    for d in (QA_ROOT, THUMB_DIR, PREVIEW_DIR, RENDER_STAGE):
        os.makedirs(d, exist_ok=True)
