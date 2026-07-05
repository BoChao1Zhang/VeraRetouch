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
    "artimuse": True, "charm": True, "iaa_mixed": True,
}
IQA_BATCH = 16
IQA_LONGEDGE = 1024                # downscale before IQA (keeps texture, bounds VRAM)
FACE_DETECT = True                 # run a cheap face detector on the portrait pool

# --- aesthetic scorer (reuse dataset_build/aesthetic.py weights) --------------
AESTHETIC_CLIP_DIR = "/home/bc/data/models/clip-vit-large-patch14"
AESTHETIC_MLP_PATH = "/home/bc/data/models/laion_aesthetic_sac_logos_ava1_l14_linearMSE.pth"

# --- IAA scorer (Artimuse + Charm mixed aesthetic score, GPU0 by default) -----
IAA_DEVICE = os.environ.get("SOURCE_QA_IAA_DEVICE", IQA_DEVICE)
IAA_MODEL_VERSION = "artimuse+charm-ava-frequency"
IAA_ARTIMUSE_MODEL_PATH = os.environ.get("SOURCE_QA_ARTIMUSE_MODEL", "/home/bc/data/models/ArtiMuse")
IAA_ARTIMUSE_REPO_DIR = os.environ.get("SOURCE_QA_ARTIMUSE_REPO", "/home/bc/code/iaa_models/ArtiMuse")
IAA_ARTIMUSE_USE_FLASH_ATTN = bool(int(os.environ.get("SOURCE_QA_ARTIMUSE_FLASH_ATTN", "0")))
IAA_CHARM_MODEL_DIR = os.environ.get("SOURCE_QA_CHARM_MODEL_DIR", "/home/bc/data/models/Charm")
IAA_CHARM_CHECKPOINT = os.environ.get(
    "SOURCE_QA_CHARM_CHECKPOINT",
    os.path.join(IAA_CHARM_MODEL_DIR, "Ava_large_charm.pth"),
)
IAA_CHARM_PATCH_SELECTION = os.environ.get("SOURCE_QA_CHARM_PATCH_SELECTION", "frequency")
IAA_CHARM_TRAINING_DATASET = os.environ.get("SOURCE_QA_CHARM_TRAINING_DATASET", "ava")
IAA_CHARM_BACKBONE = os.environ.get("SOURCE_QA_CHARM_BACKBONE", "facebook/dinov2-large")
IAA_ARTIMUSE_WEIGHT = float(os.environ.get("SOURCE_QA_ARTIMUSE_WEIGHT", "0.75"))
IAA_CHARM_WEIGHT = float(os.environ.get("SOURCE_QA_CHARM_WEIGHT", "0.25"))
IAA_IN_IQA = bool(int(os.environ.get("SOURCE_QA_IAA_IN_IQA", "1")))
IAA_IN_CLEAN = bool(int(os.environ.get("SOURCE_QA_IAA_IN_CLEAN", "1")))
IAA_REQUIRE_FOR_KEEP = bool(int(os.environ.get("SOURCE_QA_IAA_REQUIRE_FOR_KEEP", "1")))

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
# LR pool admission cap (core.LrClient semaphore): concurrent submit+poll cycles
# against the farm, independent of preset_qa's thread-pool size. The server
# enforces 1 in-flight render per client, so this does NOT create write-lock
# collisions; 2x the online clients keeps a small pending backlog on the server
# so each machine picks up its next task the moment it reports the previous one
# (1:1 left every client idle for a full submit round-trip between renders —
# measured 2026-07-05: ~71/min at 3, saturated at 6). Env-overridable.
LR_MAX_CONCURRENCY = int(os.environ.get("SOURCE_QA_LR_CONCURRENCY", "6"))
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
    # mixed IAA floors (Artimuse + Charm, normalized 0..100)
    "iaa_drop_below": 35.0,
    "iaa_keep_above": 55.0,
    # comfortable 'keep' band
    "musiq_keep_above": 55.0,
    "clipiqa_keep_above": 0.50,
    "aesthetic_vlm_keep_above": 5.5,      # starting guess; aesthetic now a soft keep vote
}

CONSTRUCT_SOURCE_IAA_MIN = float(os.environ.get("CONSTRUCT_SOURCE_IAA_MIN", str(GATE["iaa_keep_above"])))

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


# =============================================================================
# 三套二元问卷 + 反作弊位置码设计 (设计文档 LLM_QA_QUESTIONNAIRE_DESIGN_2026-06-17)
# -----------------------------------------------------------------------------
# 取代图像侧 QUESTIONNAIRE_A/B 与 QUESTIONNAIRE_C 的判级用途(旧列保留仅兼容)。
# 机制(§1.1-1.3): 只问二元真/假题; 每关键判断配 F + 逻辑反向 R(F⊕R 互斥);
# 确定性陷阱/锚点; 题号去语义化(两位"位置码")+固定乱序(每对 F/R 间隔≥⌈N/3⌉不相邻)。
#
# POS_MAP[pos] = (base_claim, polarity, audit_id)
#   polarity ∈ {F, R, trap, anchor, anchor_neg, honesty_pos, honesty_neg}
#   base_claim: F/R 对的基名(REAL/K_compose/PRO...); 陷阱/锚点的 claim 名
#   audit_id  : 落库 item 用的可读审计 ID(REAL_F / ANCHOR / T1b ...); 模型永不可见
# QTEXT[audit_id] = 中文题面(prompt + Web UI 用)
# 模型只看到 pos→题面(无维度/极性/配对); 清洗器据 POS_MAP 还原 claim+极性判 F⊕R。
# 三套 SCRAMBLE_SEED 固定(记 pilot/manifest)，改题/加题须同步 POS_MAP 并过间隔单测。
# =============================================================================

# ---- 流程 1: 摄影图像技术质量 IMQ (§1) --------------------------------------
IMQ_QUESTIONNAIRE_TAG = "IMQ"
IMQ_SCRAMBLE_SEED = 73101            # 固定; 决定 non-portrait/portrait 两态乱序(已物化于下表)
IMQ_FACE_MIN = 0.012                 # FACE_GT / no_face 的 max_face_frac 阈(与 AES_FACE_MIN 分开)
DEF_DROP = 3                         # defect_count ≥ 此值 → drop(仅 DEFECT_HARD_DROP=True)
DEF_REVIEW = 2                       # defect_count ≥ 此值(且<DEF_DROP)或 weak_subject → review
IMQ_CONTRA_TOL = 1                   # IMQ 软对允许的最大矛盾数(INTACT 现计入软对)
IMQ_REASK_MAX = 1                    # 不可信重问上限(重洗题序); 仍不可信 → review
DEFECT_HARD_DROP = False             # 缺陷计数是否准硬 drop(金标 κ≥0.60 才 True)
ENABLE_T_COLOR = False               # 启用 COLOR 陷阱(需先补 is_bw_img 双条件且假阳<2%)
ISBW_SAT_MEAN_MAX = 10.0             # is_bw_img 条件一: HSV S(0..255)均值 < 此阈
ISBW_AB_VAR_MAX = 35.0               # is_bw_img 条件二: lab a*/b* 色度方差 < 此阈(双条件均满足)
# IQA 是否参与 clean 判定。2026-06-22 用户拍板关闭: clean 纯靠两阶段 LLM QA(IMQ+审美),
# IQA(musiq/clipiqa/niqe/noise)不再做软尾降级/keep-band/hard-drop。默认关; 置 1 恢复。
IQA_IN_CLEAN = bool(int(os.environ.get("SOURCE_QA_IQA_IN_CLEAN", "0")))
MUSIQ_DROP_BELOW = 30.0              # reconcile 软尾: keep 但 musiq< → review
NIQE_CONFLICT_ABOVE = 9.0            # reconcile 软尾: niqe> 视为冲突
# R4 回调(G4, cleaner_threshold): 实测 pilot noise_sigma 中位数 232、p90 839(非 0..25 sigma 量纲),
# 旧值 18 触发率 99.5% → 把全部 keep 误降 review。回调到 ≈p90=800(只降最噪 ~10%), 据真实分布。
NOISE_SIGMA_DROP_ABOVE = 800.0       # reconcile 软尾: noise_sigma> 视为冲突
DROP_KAPPA_MIN = 0.60                # §6: 逐题 Cohen's κ ≥ 此值才允许进硬 drop
KAPPA_PASS = {"REAL": False, "CLEAN": False, "FACE": False}   # 标定前全 False → 降 review

# IMQ 题面(audit_id → 中文题面), §1.3 最终版
IMQ_QTEXT = {
    "REAL_F": "这是相机拍摄或扫描得到的真实自然照片。",
    "REAL_R": "画面带有非自然来源的破绽：界面/截图元素（状态栏、光标、菜单栏、播放控件）、印刷海报排版、3D/CG 渲染感，或 AI 生成的异常细节（畸形手指/文字/拼接）。",
    # R2 修订(G2): 去掉"画面干净"前缀(被误读成"画面不杂乱"致内容繁杂的干净图误判 CLEAN_F=0),
    # 只问是否存在叠加物, 与 CLEAN_R 严格互补。
    "CLEAN_F": "照片上没有叠加任何水印、台标/平台 logo、文字条/字幕、加框、二维码或多图拼贴的分割线。",
    "CLEAN_R": "画面上能找到具体的叠加物：水印/台标、文字条/字幕、明显边框、二维码，或多图拼贴的分割线。",
    "INTACT_F": "这是一张内容完整的正常照片，不是被极端裁切的图像碎片、纯色块或测试图/色卡。",
    "INTACT_R": "这张图不是内容完整的正常照片（它是极端裁切碎片、纯色块或测试图/色卡）。",
    "SHARP_F": "画面主体清晰、对焦准确。",
    "SHARP_R": "画面主体失焦、脱焦或整体发虚，看不清细节。",
    "NOISE_F": "画面纯净，没有明显的噪点或颗粒感。",
    "NOISE_R": "画面有明显的噪点、彩色噪声或颗粒感（尤其暗部/天空）。",
    "COMP_F": "画面没有可见的 JPEG 压缩方块、阶梯状色带或块状色噪。",
    "COMP_R": "画面有可见的 JPEG 压缩方块、平滑区域的阶梯色带或块状色噪。",
    "UPSC_F": "画面看起来是原生清晰度，没有被放大插值后的糊边、锯齿或过度平滑塑料感。（原生分辨率低/小图本身不算放大痕迹）",
    "UPSC_R": "画面有被低分辨率放大插值的痕迹：边缘糊化、锯齿或过度平滑的塑料感。",
    "EXPO_F": "曝光基本正常，没有大面积纯白死白或纯黑死黑的丢失细节区域。",
    "EXPO_R": "画面有大面积过曝纯白或欠曝纯黑的死区，完全丢失细节。",
    "OVERCOOK_F": "后期克制自然：没有过度处理的破坏痕迹（无 halo 描边/亮边、无高光涂抹丢层次、无塑料感过度磨皮、无 HDR 脏渲染、无色阶断裂）。",
    "OVERCOOK_R": "画面有过度后期的破坏痕迹：高反差边缘出现 halo/亮边、皮肤或天空被过度平滑成塑料感、高光被涂抹丢层次，或出现 HDR 脏渲染/色调分离(posterize)。",
    "SUBJ_F": "画面有明确的主体或清晰的构图意图。",
    "SUBJ_R": "画面空洞、没有明确主体，或基本只是纯文字/截图式内容。",
    "FACE_F": "画面中有清晰、可用的人脸（五官清楚、不模糊、不塑料）。",
    "FACE_R": "画面中没有清晰可用的人脸（无人脸，或人脸严重模糊/塑料感/被遮挡）。",
    # R2 修订: 去方位词"上方"(歧义致 yes 率坍到 0.11, 模型看得到图却答 0)
    "ANCHOR": "你能看到这张图片的画面内容（不是空白或纯黑）。",
    "LANDSCAPE": "这张图是横向构图（宽度大于高度）。",
    "FACE_GT": "画面中能看到人的脸。",
    "COLOR": "这是一张彩色照片（不是黑白/单色照片）。",
}
# non-portrait 展示乱序 (§1.3, N=22); pos → (claim, polarity, audit_id)
IMQ_POS_MAP = {
    "01": ("REAL", "F", "REAL_F"),     "02": ("SHARP", "F", "SHARP_F"),
    "03": ("NOISE", "F", "NOISE_F"),   "04": ("ANCHOR", "anchor", "ANCHOR"),
    "05": ("CLEAN", "F", "CLEAN_F"),   "06": ("COMP", "F", "COMP_F"),
    "07": ("INTACT", "F", "INTACT_F"), "08": ("UPSC", "F", "UPSC_F"),
    "09": ("EXPO", "F", "EXPO_F"),     "10": ("SUBJ", "F", "SUBJ_F"),
    "11": ("OVERCOOK", "F", "OVERCOOK_F"), "12": ("REAL", "R", "REAL_R"),
    "13": ("NOISE", "R", "NOISE_R"),   "14": ("LANDSCAPE", "trap", "LANDSCAPE"),
    "15": ("SHARP", "R", "SHARP_R"),   "16": ("COMP", "R", "COMP_R"),
    "17": ("CLEAN", "R", "CLEAN_R"),   "18": ("UPSC", "R", "UPSC_R"),
    "19": ("INTACT", "R", "INTACT_R"), "20": ("OVERCOOK", "R", "OVERCOOK_R"),
    "21": ("EXPO", "R", "EXPO_R"),     "22": ("SUBJ", "R", "SUBJ_R"),
}
# portrait 展示乱序 (§1.3, N=25; 加 FACE_F/FACE_R/FACE_GT)
IMQ_POS_MAP_PORTRAIT = {
    "01": ("REAL", "F", "REAL_F"),     "02": ("SHARP", "F", "SHARP_F"),
    "03": ("NOISE", "F", "NOISE_F"),   "04": ("ANCHOR", "anchor", "ANCHOR"),
    "05": ("CLEAN", "F", "CLEAN_F"),   "06": ("COMP", "F", "COMP_F"),
    "07": ("INTACT", "F", "INTACT_F"), "08": ("UPSC", "F", "UPSC_F"),
    "09": ("FACE_GT", "trap", "FACE_GT"), "10": ("EXPO", "F", "EXPO_F"),
    "11": ("SUBJ", "F", "SUBJ_F"),     "12": ("OVERCOOK", "F", "OVERCOOK_F"),
    "13": ("FACE", "F", "FACE_F"),     "14": ("LANDSCAPE", "trap", "LANDSCAPE"),
    "15": ("REAL", "R", "REAL_R"),     "16": ("NOISE", "R", "NOISE_R"),
    "17": ("SHARP", "R", "SHARP_R"),   "18": ("CLEAN", "R", "CLEAN_R"),
    "19": ("COMP", "R", "COMP_R"),     "20": ("INTACT", "R", "INTACT_R"),
    "21": ("UPSC", "R", "UPSC_R"),     "22": ("EXPO", "R", "EXPO_R"),
    "23": ("SUBJ", "R", "SUBJ_R"),     "24": ("OVERCOOK", "R", "OVERCOOK_R"),
    "25": ("FACE", "R", "FACE_R"),
}
IMQ_SYSTEM_PROMPT = (
    "你是严格的照片技术质检员。只依据所给这一张图片作答；不解释、不推理、不输出多余文字。\n"
    "你只判断\"技术与有效性质量\"，不评价美感、风格或调色好坏。注意：已精修/已调色的成片是正常照片，"
    "不要因为\"看起来已调色/已修图\"就判为非自然或扣画质——只看真实技术缺陷与破坏性痕迹。"
    "原生分辨率低的小图本身不是\"放大插值痕迹\"。\n"
    "下面是若干二元判断题，每题有一个两位题号（如 03）。逐题判断该陈述对这张图是否成立：成立=1，不成立=0。\n"
    "【重要】题目之间彼此独立、没有配对或正反关系；每一题都要重新看这张图独立判断，不要根据别的题的答案来推断本题。"
    "每题必须作答，拿不准时给最可能的判断，不得留空、不得答\"不确定\"。\n"
    "特别注意：仔细检查四角与边缘是否有半透明水印、版权/署名文字、字幕条，以及画面是否为多图拼贴"
    "（有分割线/多格）——这些都算\"叠加物\"，不要漏看。\n"
    "题目（按题号）：\n{items}\n"
    "作答格式：每题输出\"题号紧跟答案\"（03 成立写 031，不成立写 030），各题用一个空格分隔，"
    "按题号从小到大，每题号只出现一次。只输出这一串，不要其它任何内容。"
)

# ---- 流程 2: 摄影图像审美 AES (§2; 软信号 merit_frac, 永不 drop) -----------
AES_QUESTIONNAIRE_TAG = "aes"
AES_SCRAMBLE_SEED = 24207
AES_FACE_MIN = 0.04                  # max_face_frac> 此值才判"确有人脸" → 发 M3/M4/T4
AES_CONTRA_TOL = 1                   # 门5 软对违反容差; viol> 即 reliable=False
MERIT_KEEP_FRAC = 0.65              # merit_frac>= 记 aes_keep_vote=1(仅独立观测软栏)
AES_RETRY = 1                        # reliable=False 重问次数; 仍不可信 → merit_frac=NULL→review
AES_TEMPERATURE = 0.1
HAS_EXIF_FIXED = False               # iqa decode 是否已加 exif_transpose; False→T2 软监控不入硬门
HAS_ISBW_IMG = False                 # iqa 是否已算 is_bw_img; False→T3 不发题不入硬门

AES_QTEXT = {
    "K1": "构图明显经过经营、优于随手拍：主体落点 / 留白 / 线条引导有自觉安排，而非仅居中端正。",
    "K2": "构图随意失衡：主体贴边或被边缘切割、画面明显歪斜或重心失衡。",
    "K3": "取景边缘干净利落：四边与角落没有多余杂物或半截割裂的元素。",
    "K4": "取景潦草局促：边缘塞进多余杂物或被切一半的元素。",
    "L1": "光线明显讲究：有清晰的方向感，光影塑造出立体与氛围（非平光直照）。",
    "L2": "照明平板无方向感：光线平铺呆滞，画面发闷。",
    "L3": "影调层次丰富：高光到阴影的明暗过渡顺滑且有细腻层次。",
    "L4": "影调灰平一团：明暗缺乏层次、整体发灰扁平。",
    "C1": "色彩有自觉的整体调性：配色和谐统一、有呼应或克制（黑白 / 单色图：影调统一克制、调性鲜明）。",
    "C2": "色彩严重偏色发脏：整体明显发浊 / 脏污或偏色失调（黑白 / 单色图：影调脏浊不统一）。",
    "D1": "有明显的空间纵深：前中后景或虚实关系清楚，画面立体不平板。",
    "D2": "画面平板无纵深：前后景糊成一片、缺乏立体感。",
    "D3": "主体与背景明显分离：主体清楚地从背景中凸显出来。",
    "D4": "主体淹没于背景：主体与背景粘连难分、无法凸显。",
    "M1": "有强而明确的视觉中心：一眼锁定主体，注意力被有效引导。",
    "M2": "无视觉中心：画面空洞或元素平均散乱，看不出重点。",
    "M3": "人物神态 / 瞬间到位：表情或动作自然有感染力（决定性瞬间）。",
    "M4": "人物神态僵硬尴尬：表情 / 动作呆板别扭、瞬间没抓住。",
    "N1": "画面整洁：背景干净不杂乱，没有抢眼的干扰物。",
    "N2": "画面杂乱：背景堆满杂物或有抢眼干扰物，注意力被分散。",
    # R2 修订: 去方位词"上方"(同 IMQ ANCHOR 根因), 保持 H1=1/H2=0 诚实对偶
    "H1": "这是一张照片（不是纯文字截图或纯色块）。",
    "H2": "这张画面完全空白、没有任何可见内容。",
    "T1": "你能看到这张图片的画面内容（不是空白或纯黑）。",
    "T2": "这张图是横向的（宽大于高）。",
    "T3": "这是一张彩色照片（不是黑白 / 单色）。",
    "T4": "画面中能看到人脸。",
}
# non-portrait 基线展示乱序 (§2.3, N=22)
AES_POS_MAP = {
    "01": ("K_compose", "F", "K1"), "02": ("L_light", "F", "L1"),
    "03": ("C_harmony", "F", "C1"), "04": ("ANCHOR", "anchor", "T1"),
    "05": ("K_frame", "F", "K3"),   "06": ("L_tone", "F", "L3"),
    "07": ("D_depth", "F", "D1"),   "08": ("M_focus", "F", "M1"),
    "09": ("N_clean", "F", "N1"),   "10": ("D_sep", "F", "D3"),
    "11": ("HON", "honesty_pos", "H1"), "12": ("K_compose", "R", "K2"),
    "13": ("L_light", "R", "L2"),   "14": ("LANDSCAPE", "trap", "T2"),
    "15": ("C_harmony", "R", "C2"), "16": ("K_frame", "R", "K4"),
    "17": ("L_tone", "R", "L4"),    "18": ("D_depth", "R", "D2"),
    "19": ("M_focus", "R", "M2"),   "20": ("N_clean", "R", "N2"),
    "21": ("D_sep", "R", "D4"),     "22": ("HON", "honesty_neg", "H2"),
}
# portrait 展示乱序 (§2.3, N=25; 加 M_moment(M3/M4)/FACE(T4))
AES_POS_MAP_PORTRAIT = {
    "01": ("K_compose", "F", "K1"), "02": ("L_light", "F", "L1"),
    "03": ("C_harmony", "F", "C1"), "04": ("ANCHOR", "anchor", "T1"),
    "05": ("K_frame", "F", "K3"),   "06": ("L_tone", "F", "L3"),
    "07": ("D_depth", "F", "D1"),   "08": ("M_focus", "F", "M1"),
    "09": ("FACE", "trap", "T4"),   "10": ("N_clean", "F", "N1"),
    "11": ("D_sep", "F", "D3"),     "12": ("M_moment", "F", "M3"),
    "13": ("HON", "honesty_pos", "H1"), "14": ("K_compose", "R", "K2"),
    "15": ("L_light", "R", "L2"),   "16": ("LANDSCAPE", "trap", "T2"),
    "17": ("C_harmony", "R", "C2"), "18": ("K_frame", "R", "K4"),
    "19": ("L_tone", "R", "L4"),    "20": ("D_depth", "R", "D2"),
    "21": ("M_focus", "R", "M2"),   "22": ("N_clean", "R", "N2"),
    "23": ("D_sep", "R", "D4"),     "24": ("M_moment", "R", "M4"),
    "25": ("HON", "honesty_neg", "H2"),
}
AES_SYSTEM_PROMPT = (
    "你是严谨的摄影审美评审。只依据所给这一张图片作答；不解释、不推理、不输出任何多余文字。\n"
    "下面是固定的若干道判断题，每题有一个两位题号（如 03）。逐题判真假：陈述为真写 1，为假写 0。\n"
    "评的是【画面结构性审美】（构图取景、光线影调、色彩协调、空间层次、主体显著、画面整洁），"
    "与清晰度/噪点/压缩等技术质量无关；不要因为图片分辨率低或看起来已调色而扣分。\n"
    "注意：很多题问的不是\"有没有\"，而是\"是否达到明显高于随手拍的较高水准\"——只有该维度确实出类拔萃、"
    "一眼可见的优点才写 1；大多数普通成片在多数维度上只是及格/平庸，应写 0。一张图很少在所有维度都出众，"
    "不要一律给好评，也不要因为图片看起来精致/已调色就默认每项都打 1。\n"
    "【重要】题目之间彼此独立、没有配对或正反关系；每一题都要重新看这张图独立判断，不要根据别的题的答案来推断本题。"
    "每题都必须答；看不准时给出你最可能的判断，不得留空、不得答\"不确定\"。\n"
    "题目（按题号）：\n{items}\n"
    "作答格式：把【题号和答案紧贴】写出（03 为真写 031，为假写 030），题与题之间空一格，"
    "按题号从小到大，每个题号恰好出现一次。只输出这一串题号答案，不要任何其它文字。"
)

# ---- 流程 3: Preset QA (§3; 6 固定探针 before/after, coherence_score) -------
PRESET_QA_TAG = "preset"
PRESET_SCRAMBLE_SEED = 51811

# preset 功能打 tag — vLLM 命名 (grounding=确定性 LAB 指标 + QA, 证据=6 探针 before/after 拼图)。
# 命名策略: 受控 axes(来自确定性测量, 可查询一致) + vLLM 自由 look 名(可读) + 一句功能描述。
# 测量为准: vLLM 的 axes 必须与测量一致, 它负责"命名/综合"而非"测量"。
PRESET_TAG_SYSTEM_PROMPT = (
    "你是资深调色师。任务: 根据一个 Lightroom 预设在 6 个『标准探针』上的真实 before→after "
    "客观 LAB 指标, 命名该预设的功能(把画面变成什么风格的 look), 与具体图像内容无关。\n"
    "6 探针覆盖主要色彩与影调: 红/黄/绿/蓝 四个高饱和主色相 + 肤色人像 + 中性低饱和, "
    "故预设对各色彩、肤色、明暗的处理都会暴露在指标里。\n"
    "你只会收到【逐探针 LAB 客观指标表】+【聚合判读】(程序在真实渲染上算出, 不可推翻), 不给图。\n"
    "规则: (1) 只能基于给出的客观指标命名, 不臆造未测到的效果; "
    "(2) 测量为准——axes 必须与判读一致(判读说中性灰偏暖就不能命名冷调); "
    "(3) 描述这一 look 的功能, 不描述具体图像内容; "
    "(4) **色温/色罩只看『中性』『肤色』行的 Δa*/Δb***——红/黄/绿/蓝行本身就是高饱和彩色, "
    "其 chroma 被压缩是正常的, 不要据此把预设叫成『冷调/赛博』; 只有中性/肤色确实被推冷才算冷调; "
    "(5) 不要滥用『赛博/霓虹』, 仅当中性/肤色无明显冷移、却有强烈增艳(色探针相对 chroma 大幅为正)时才用。\n"
    "(6) **区分度要求(重点)**: name 可粗, 但你必须逐一刻画 6 个探针各自的走向(red/yellow/green/blue/"
    "skin/neutral), 用表里的 ΔL/Δa*/Δb*/相对chroma/色相旋转说话——例如『蓝: 压暗-12 并转青(Δhue+18°)、"
    "去饱-40%』。同一大类(如多个『复古胶片暗调』)的不同预设, 其 per_probe 细节必须不同; 抓住该预设"
    "区别于同类的最显著 1-2 个处理写进 caption。不得把不同预设糊成同一句。\n"
    "只输出 JSON。"
)
# 自由 look 名示例词库(提示风格粒度, 非枚举): 暖调褪色胶片/青橙电影感/冷调通透/日系小清新/
# 高饱港风/复古胶片/黑白胶片/油画暗调/HDR 风光/奶油肤色...
TAU_NOOP_MEAN = 1.5                  # param(LR) order=0 前置 no-op 门: mean ΔE< → near_noop drop
TAU_NOOP_MEAN_LUT = 2.5             # LUT 前置 no-op 门阈
TAU_NOOP_LOW = 2.5                  # param T2 真值: ΔE< → t2_truth=1(近似 no-op)
TAU_NOOP_LOW_LUT = 3.5             # LUT T2 真值阈
TAU_CHANGE = 3.0                    # 文案/监控参考"明显变化"线(param)
TAU_CHANGE_LUT = 4.0               # 同上(LUT)
TAU_SSIM = 0.90                     # T3 真值: ssim< → t3_truth=1(结构破坏); 须 200-preset 回调
PRESET_CONTRA_TOL = 0               # COH 软对违反容差: H1==H2 即 1 次违反, >0 才 contradiction_soft
MIN_RELIABLE_PROBES = 3             # 投票所需最少 reliable 探针; 不足→review
VOTE_THRESH = 0.5                   # PRO/INTENT 跨探针多数投票阈
COH_VOTE_THRESH = 0.66             # COH 投票阈(2/3, 更严)
W_PRO = 0.5                         # coherence_score 权重
W_COH = 0.5
PRESET_REASK_MAX = 1                # 单探针重问次数(parse/anchor temp=0; trap/contra 打乱+temp0.3)
PRO_DROP_ENABLED = False            # κ-gate: PRO κ≥0.6 才 True, 方允许 not_professional→drop
INTENT_DROP_ENABLED = False         # 同上, INTENT
# T2/T3 硬剔除前提: 200-preset 试跑验证 t2_truth(1占比∈[10%,90%]) / t3_truth∈[3%,97%];
# 未验证前降级软标记(记 trap_fail 但不剔除探针)。pilot 起始一律 False (§3.6/§3.10)。
PRESET_T2_HARD = False
PRESET_T3_HARD = False

PRESET_QTEXT = {
    # R2 修订: 旧 P1 含"专业、克制、自然"风格措辞, 强风格化 LUT look 被判 P1=0∧P2=0(both-0
    # 占 LUT 59%) → 误触 contradiction_hard:PRO, reliable 仅 53.8%。改为 P2 的严格否定(只问
    # 破坏性缺陷是否缺失, 风格强弱不算问题), 让 both-0 几乎消失、强风格干净 look 得 P1=1。
    "P1": "after 是一次可用的专业成品：即便是强烈、大胆的风格化调色，画面本身也没有明显的破坏性问题（无大面积过曝死白、无大面积死黑、无明显塑料感、无严重整体偏色、无怪异/脏渲染）。",
    "P2": "after 存在明显破坏性问题：大面积过曝死白或死黑、明显塑料感、严重整体偏色、或像被极端/怪异地处理过（之一或多项）。",
    "I1": "after 相对 before 有明确、可学习的编辑方向（可清楚看出在做调色/影调/对比的处理）。",
    "I2": "after 几乎等于 before、看不出实质编辑（近似 no-op）。",
    "H1": "after 相对 before 的调整是一个连贯一致的整体 look：色调与影调走向统一，像同一套专业预设作用于全图。",
    "H2": "after 的调整随机或局部失控：不同区域色调走向互相矛盾、有突兀的局部色块/断裂，不像统一的 look。",
    # R2 修订: 去方位词"上面"(同 IMQ ANCHOR 根因, 预修; preset 尚未真渲染)
    "T1": "我看到了两张图（before 和 after）。",
    "T1b": "我只看到一张图。",
    "T2": "after 与 before 几乎一模一样、看不出肉眼可辨的差异。",
    "T3": "after 出现明显的结构破坏 / 色阶断裂 / 细节涂抹涂糊（不是自然的调色）。",
}
# 展示乱序 (§3.3, N=10; 全 preset 共用一份, system prompt 可 prefix-cache)
PRESET_POS_MAP = {
    "01": ("PRO", "F", "P1"),       "02": ("INTENT", "F", "I1"),
    "03": ("CHANGED", "trap", "T2"), "04": ("COH", "F", "H1"),
    "05": ("ANCHOR", "anchor", "T1"), "06": ("PRO", "R", "P2"),
    "07": ("DESTRUCT", "trap", "T3"), "08": ("INTENT", "R", "I2"),
    "09": ("COH", "R", "H2"),       "10": ("ANCHOR", "anchor_neg", "T1b"),
}
PRESET_SYSTEM_PROMPT = (
    "你是严格的修图预设质检员。下面给你两张图：第 1 张是原图(before)，第 2 张是对其应用某预设后的真实成品(after)。\n"
    "仅依据这两张图作答，不解释、不推理、不输出多余文字。\n"
    "共有 10 道判断题，每题有一个两位题号(如 03)。逐题判断该陈述对这两张图是否为真：为真=1，为假=0。\n"
    "【重要】题目之间彼此独立、没有配对或正反关系；每一题都要重新看这两张图独立判断，"
    "不要根据别的题的答案来推断本题。必须每题都答；看不准也要给出最可能的判断，不得留空。\n"
    "题目（按题号）：\n{items}\n"
    "作答格式：每题写出\"题号+答案\"并紧贴(题号 03 答真写 031，答假写 030)，题间用一个空格，"
    "按题号从小到大，每个题号恰好出现一次。只输出这一串，不要任何其它文字。"
)

# 6 固定探针确定性选取 (§3.2); resolve_probes 在 images 上按这些列取极值/匹配
PRESET_PROBE_SLOTS = [
    {"slot": 1, "desc": "肤色/人脸区域", "where": "is_portrait_pool=1 AND max_face_frac BETWEEN 0.08 AND 0.45", "order": "aesthetic DESC"},
    {"slot": 2, "desc": "大面积高光/吹白", "where": "highlight_frac IS NOT NULL", "order": "highlight_frac DESC, mean_luma DESC"},
    {"slot": 3, "desc": "高饱和/白平衡", "where": "saturation_mean IS NOT NULL", "order": "saturation_mean DESC"},
    {"slot": 4, "desc": "中性基线/偏色放大镜", "where": "saturation_mean IS NOT NULL AND is_bw_img IS DISTINCT FROM 1", "order": "saturation_mean ASC"},
    {"slot": 5, "desc": "中等暗部/暗部提亮", "where": "shadow_frac IS NOT NULL AND max_face_frac IS NOT NULL", "order": "abs(shadow_frac-0.15) ASC, abs(max_face_frac-0.15) ASC"},
    {"slot": 6, "desc": "大面积暗部/死黑", "where": "shadow_frac IS NOT NULL", "order": "shadow_frac DESC, mean_luma ASC"},
]


# LAB 主色探针 (2026-06-22 用户指令): luma 探针不覆盖色彩类别, preset 区分度不足。
# 改为 4 主色相(chroma 加权 hue 直方图选取)+ 肤色 + 中性。hue=atan2(b*,a*)∈[0,360)。
# 4 主色相中心(°)各 ±HUE_HALF; 选 chroma 加权 mass 最高且 chroma 足够者。
PRESET_HUE_HALF = 35
PRESET_PROBE_SLOTS_LAB = [
    {"slot": 1, "name": "red",     "desc": "红/暖色主导",  "kind": "hue", "hue_center": 25},
    {"slot": 2, "name": "yellow",  "desc": "黄色主导",     "kind": "hue", "hue_center": 90},
    {"slot": 3, "name": "green",   "desc": "绿色主导",     "kind": "hue", "hue_center": 150},
    {"slot": 4, "name": "blue",    "desc": "蓝色主导",     "kind": "hue", "hue_center": 290},
    {"slot": 5, "name": "skin",    "desc": "肤色/人脸",    "kind": "skin"},
    {"slot": 6, "name": "neutral", "desc": "中性/低饱和",  "kind": "neutral"},
]
# preset 功能打 tag 阈值。关键: 色温/色罩在【中性探针】上读(极饱和色探针的全图 Δa/Δb
# 被 chroma 压缩主导, 失真); 饱和用色探针的【相对】chroma 变化。
PRESET_TAG_THRESH = {
    "warm_db": 3.0,        # 中性探针 Δb* > → 暖; < -→ 冷
    "tint_da": 3.0,        # 中性探针 Δa* > → 品红色罩; < -→ 绿色罩
    "sat_rel": 0.10,       # 色探针相对 chroma 变化 (after-before)/before > → 增艳; < -→ 去饱和
    "bw_after_C": 6.0,     # 色探针 after 平均 chroma < → 黑白
    "contrast_ratio_hi": 1.12,   # L 标准差比 after/before > → 强对比
    "contrast_ratio_lo": 0.90,   # < → 平/褪
    "lift_dLmin_blacks": 6.0,    # 暗部(before L<25) ΔL > → 提亮暗部(褪色胶片)
    "crush_dLmin_blacks": -6.0,  # < → 压黑
    "teal_hue_rot": 12.0,        # teal-orange: 蓝探针色相向青/teal 旋转(°)
    "warm_keep_C": 18.0,         # 且暖探针(red/skin) after chroma 保留(不被压成灰)
}


def imq_pos_map(is_portrait: bool) -> dict:
    return IMQ_POS_MAP_PORTRAIT if is_portrait else IMQ_POS_MAP


def aes_pos_map(has_face: bool) -> dict:
    return AES_POS_MAP_PORTRAIT if has_face else AES_POS_MAP


def render_items_block(pos_map: dict, qtext: dict) -> str:
    """按位置码升序列出 '位置码：题面'(无维度/极性), 供 system prompt {items} 填充。"""
    return "\n".join(f"{p}：{qtext[pos_map[p][2]]}" for p in sorted(pos_map))


def ensure_dirs() -> None:
    for d in (QA_ROOT, THUMB_DIR, PREVIEW_DIR, RENDER_STAGE):
        os.makedirs(d, exist_ok=True)
