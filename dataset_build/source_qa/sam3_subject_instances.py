"""实例级主体 mask precompute（Mask v4：VLM 选主体 + 实例/群体 union）。

把 2026-07-11 在 50 张审核样本上定稿的 v6 流程（eval_subject_instance_selector）
生产化：VLM 源图标注（single/group、花草纳入、画中画排除）→ SAM3 实例 proposals
（single 走中心点过滤，group 全量，密集群体 union 回退）→ VLM 选实例（群体多选
union）→ 退化守卫（mask 面积 >0.85 / 多成员包络 >0.67 舍弃）→ 清理（≥最大组件
0.5% 连通域保留 + 填孔）。

产物（source path 使用稳定 ``path_key``）：
    <cache>/<path_key>/subject.png    # 清理后的主体 mask（L 8bit；实例或群体 union）
    <cache>/<path_key>/subject.json   # status/scope/prompt/守卫指标/版本，见 _write_meta

status: ready | no_subject | sam_miss | no_center_candidate |
        degenerate_full_frame | group_envelope_too_large |
        subject_label_error | selection_failed

两个子命令解耦运行环境（DB 依赖只在 dump-pool）：
    # base env（有 DB 依赖）：导出源图池
    python -m dataset_build.source_qa.sam3_subject_instances dump-pool --out /home/bc/data/datasets/vera_directionA_1M/subject_pool.jsonl
    # monetgpt_sam3 env（SAM3 + VLM）：消费池子跑重建，可分片可续跑
    conda run -n monetgpt_sam3 python -m dataset_build.source_qa.sam3_subject_instances run \
        --pool /home/bc/data/datasets/vera_directionA_1M/subject_pool.jsonl \
        --device cuda:0 [--shard 0/1] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from dataset_build.mask_cache import path_key  # noqa: E402

# Instance-level subject cache root.
CACHE = os.environ.get(
    "CONSTRUCT_SUBJECT_CACHE", "/home/bc/data/datasets/vera_directionA_1M/subject_cache")
PIPELINE_VERSION = "subject_v1 (eval v6 2026-07-11)"


# --------------------------------------------------------------------------- #
# dump-pool（base env：唯一碰 DB 的路径）
# --------------------------------------------------------------------------- #
def dump_pool(out_path: str) -> None:
    from . import db
    conn = db.connect()
    rows = conn.execute(
        "SELECT c.asset_id, s.path FROM source_captions c "
        "JOIN assets s USING(asset_id) ORDER BY c.asset_id").fetchall()
    conn.close()
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"asset_id": r["asset_id"], "source_path": r["path"]},
                               ensure_ascii=False) + "\n")
            n += 1
    print(f"pool: {n} sources -> {out_path}")


# --------------------------------------------------------------------------- #
# mask 清理（与 50 张验收版一致：0.5% 连通域保留 + 边框种子填孔）
# --------------------------------------------------------------------------- #
def clean_mask(hard: np.ndarray) -> np.ndarray:
    import cv2
    u8 = hard.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
        thr = max(64, int(0.005 * areas.max()))  # 0.5%：群体成员/截断肢体保留，碎屑清除
        keep = np.zeros(n, bool)
        keep[1:] = areas >= thr
        u8 = keep[lab].astype(np.uint8)
    pad = np.pad(u8, 1)  # 1px 零边框：mask 贴边时 floodFill 种子仍在背景上
    h, w = pad.shape
    cv2.floodFill(pad, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    return (u8 | (1 - pad[1:-1, 1:-1])).astype(bool)


def _save_subject_png(mask: np.ndarray, out_path: str) -> None:
    from PIL import Image
    tmp = out_path + ".tmp"
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(
        tmp, format="PNG", compress_level=1)
    os.replace(tmp, out_path)


def _write_meta(cache_dir: str, payload: dict) -> None:
    payload = dict(payload)
    payload.setdefault("version", PIPELINE_VERSION)
    payload["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = os.path.join(cache_dir, "subject.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, os.path.join(cache_dir, "subject.json"))


# --------------------------------------------------------------------------- #
# 批量 SAM3 前向（GPU0 大 batch；后处理复用 eval 工具的 _proposals_from_result）
# --------------------------------------------------------------------------- #
MASK_LONG_EDGE = 1536   # proposal/subject mask 分辨率封顶：几何归一化，α 消费端自会 resize；
                        # native 后处理在 10MP 图上是 20-45× 的 CPU/显存放大器（0.15 img/s 事故）

_POST_POOL = None


def _post_pool() -> ThreadPoolExecutor:
    """SAM 后处理共享线程池（每批新建/销毁一个 pool 是纯开销）。"""
    global _POST_POOL
    if _POST_POOL is None:
        _POST_POOL = ThreadPoolExecutor(max_workers=8)
    return _POST_POOL


def _shutdown_post_pool() -> None:
    global _POST_POOL
    if _POST_POOL is not None:
        _POST_POOL.shutdown(wait=True, cancel_futures=True)
        _POST_POOL = None


def _capped(native: tuple) -> tuple:
    h, w = native
    s = MASK_LONG_EDGE / max(h, w)
    if s >= 1.0:
        return (h, w)
    return (max(1, int(round(h * s))), max(1, int(round(w * s))))


def _sam_forward_batch(masker, sam_rows: list, img_works: list, min_score: float,
                       dedupe_iou: float) -> list:
    from dataset_build.tools.eval_subject_instance_selector import (
        _proposals_from_result, _sam_proposals)
    torch = masker._torch
    try:
        images, sizes = [], []
        for r in sam_rows:
            image, native = masker._as_pil(r["source_path"])   # native = (H, W)
            images.append(image)
            sizes.append(_capped(native))
        inputs = masker._proc(
            images=images, text=[r["main_subject"] for r in sam_rows],
            return_tensors="pt").to(masker._det.device)
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = masker._det(**inputs)
        results = masker._proc.post_process_instance_segmentation(
            outputs, threshold=float(min_score),
            mask_threshold=masker.mask_threshold,
            target_sizes=[(h, w) for (h, w) in sizes])
        per_img = (time.perf_counter() - started) / max(len(sam_rows), 1)
        # 每图后处理（mask 下载/IoU 去重/PNG 存盘）并行化，别让 GPU 等 CPU
        out = list(_post_pool().map(
            lambda t: _proposals_from_result(torch, t[1], t[0], (t[2][1], t[2][0]),
                                             per_img, t[3], min_score, dedupe_iou),
            zip(sam_rows, results, sizes, img_works)))
        del outputs, results, inputs
        torch.cuda.empty_cache()   # 大块瞬时分配不留在缓存池（防碎片化膨胀）
        return out
    except Exception as e:  # noqa: BLE001 - 批前向失败退回逐张（含 OOM/尺寸兼容问题）
        print(f"[sam-batch] fallback to per-image: {type(e).__name__}: {str(e)[:150]}",
              file=sys.stderr, flush=True)
        out = []
        for r, work in zip(sam_rows, img_works):
            try:
                out.append(_sam_proposals(masker, r, work, min_score, dedupe_iou))
            except Exception as e2:  # noqa: BLE001
                out.append({"asset_id": r["asset_id"], "source_path": r["source_path"],
                            "main_subject": r["main_subject"], "proposals": [],
                            "status": "sam_miss", "error": str(e2)[:200]})
        return out


# --------------------------------------------------------------------------- #
# run（monetgpt_sam3 env）
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace, rows_override: list[dict] | None = None) -> dict[str, str]:
    # v6 已验证逻辑直接复用 eval 工具（prompts/解析/proposals/过滤/dense 回退/overlay）
    from dataset_build.masking import Sam3Masker
    from dataset_build.tools.eval_subject_instance_selector import (
        _call_selector, _call_subject_label, _dense_group_union,
        _focus_proposals, _prepare_overlays)

    # 续跑集合用一次 listdir 构建：对 56.8k 源逐个 os.path.exists 是冷 inode 随机读，
    # 在共享数据盘上曾把扫描拖到 ~40min；源图可读性由标注阶段兜底（source_unreadable）。
    cache_root = os.fspath(getattr(args, "cache_root", CACHE))
    os.makedirs(cache_root, exist_ok=True)
    done_keys = set()
    if rows_override is None:
        for d in os.listdir(cache_root):
            if os.path.exists(os.path.join(cache_root, d, "subject.json")):
                done_keys.add(d)
    todo = []
    if rows_override is None:
        with open(args.pool, encoding="utf-8") as f:
            input_rows = [json.loads(line) for line in f if line.strip()]
    else:
        input_rows = [dict(row) for row in rows_override]
    for row in input_rows:
        key = path_key(row["source_path"])
        if args.shard_n > 1 and int(key, 16) % args.shard_n != args.shard_i:
            continue
        if key in done_keys:
            continue  # 续跑：已处理
        row["cache_dir"] = os.path.join(cache_root, key)
        todo.append(row)
        if args.limit and len(todo) >= args.limit:
            break
    print(f"pending: {len(todo)} (shard {args.shard_i}/{args.shard_n})", flush=True)
    if not todo:
        return {}

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    masker = None
    stats: dict[str, int] = {}
    statuses: dict[str, str] = {}
    pool = ThreadPoolExecutor(max_workers=args.vlm_workers)

    def bump(status: str, asset_id: str | None = None) -> None:
        stats[status] = stats.get(status, 0) + 1
        if asset_id is not None:
            statuses[asset_id] = status

    def finalize(record: dict, row: dict) -> str:
        """选择结果 → 守卫 → 清理 → 落盘。返回最终 status。"""
        from PIL import Image
        cache_dir = row["cache_dir"]
        os.makedirs(cache_dir, exist_ok=True)
        ann = record.get("subject_annotation", {}).get("a", {})
        base_meta = {
            "asset_id": row["asset_id"],
            "source_path": row["source_path"],
            "scope": ann.get("subject_scope"),
            "sam_prompt": ann.get("sam_prompt"),
            "description": ann.get("description"),
            "expected_count": ann.get("expected_count"),
            "dense_union": bool(record.get("dense_union")),
            "raw_proposal_count": record.get("raw_proposal_count"),
        }
        sel = (record.get("selection") or {}).get("a") or {}
        if sel.get("status") == "transport_error":
            # 瞬时故障（vLLM 重启/过载）不落盘：留给下次 resume 重跑，绝不永久降级
            return "transport_skipped"
        if sel.get("decision") != "select":
            status = ("no_subject" if sel.get("reason_code") == "pure_landscape"
                      else "selection_failed")
            _write_meta(cache_dir, dict(base_meta, status=status,
                                        decision=sel.get("decision"),
                                        reason=sel.get("reason_code") or sel.get("status")))
            return status
        ids = sel.get("stable_instance_ids") or []
        props = [p for p in record["proposals"] if p["stable_id"] in ids]
        if not props:
            _write_meta(cache_dir, dict(base_meta, status="selection_failed",
                                        reason="selected_id_missing"))
            return "selection_failed"
        area = sum(float(p["area"]) for p in props)
        env = ((max(p["bbox"][2] for p in props) - min(p["bbox"][0] for p in props))
               * (max(p["bbox"][3] for p in props) - min(p["bbox"][1] for p in props)))
        guard_meta = dict(base_meta, n_members=len(props),
                          selected_area=round(area, 4), selected_envelope=round(env, 4),
                          confidence=sel.get("confidence"))
        multi = len(props) > 1 or bool(record.get("dense_union"))
        if area > args.max_mask_area:
            _write_meta(cache_dir, dict(guard_meta, status="degenerate_full_frame"))
            return "degenerate_full_frame"
        if ann.get("subject_scope") == "group" and multi and env > args.max_group_envelope:
            _write_meta(cache_dir, dict(guard_meta, status="group_envelope_too_large"))
            return "group_envelope_too_large"
        union = None
        for p in props:
            m = np.asarray(Image.open(p["mask_path"]).convert("L"), np.uint8) >= 128
            union = m if union is None else (union | m)
        cleaned = clean_mask(union)
        if not cleaned.any():
            _write_meta(cache_dir, dict(guard_meta, status="selection_failed",
                                        reason="empty_after_clean"))
            return "selection_failed"
        _save_subject_png(cleaned, os.path.join(cache_dir, "subject.png"))
        _write_meta(cache_dir, dict(guard_meta, status="ready",
                                    mask_area_cleaned=round(float(cleaned.mean()), 4)))
        return "ready"

    def _post_and_select(record: dict, row: dict, label: dict, img_work: Path) -> str:
        """SAM 之后的全部 CPU/VLM 工作（focus/dense/overlay/选择/守卫/落盘），跑线程池。
        任何单图异常就地记账（post_select_error），绝不让毒丸图打穿整个 run。"""
        try:
            return _post_and_select_inner(record, row, label, img_work)
        except Exception as e:  # noqa: BLE001 - 与 _label_one 同款单图遏制
            os.makedirs(row["cache_dir"], exist_ok=True)
            _write_meta(row["cache_dir"], {"asset_id": row["asset_id"],
                                           "source_path": row["source_path"],
                                           "status": "post_select_error",
                                           "reason": f"{type(e).__name__}: {e}"[:200]})
            return "post_select_error"

    def _post_and_select_inner(record: dict, row: dict, label: dict, img_work: Path) -> str:
        cache_dir = row["cache_dir"]
        try:
            record["subject_annotation"] = {"a": label}
            if record.get("status") == "ready":
                _focus_proposals(record, args.focus_radius)
            if len(record.get("proposals", ())) > args.max_proposals:
                if label.get("subject_scope") == "group" and record.get("status") == "ready":
                    _dense_group_union(record, img_work)
                else:
                    record["status"] = "too_many_instances"
            if record.get("status") != "ready":
                os.makedirs(cache_dir, exist_ok=True)
                _write_meta(cache_dir, {"asset_id": row["asset_id"],
                                        "source_path": row["source_path"],
                                        "scope": label.get("subject_scope"),
                                        "sam_prompt": label.get("sam_prompt"),
                                        "status": record.get("status"),
                                        "reason": record.get("error")})
                return record.get("status", "unknown")
            _prepare_overlays(record, img_work, args.seed)
            _, _, sel = _call_selector(
                record, "a", args.vlm_base_url, args.vlm_model, args.vlm_timeout,
                api_key=getattr(args, "vlm_api_key", "EMPTY"),
            )
            record["selection"] = {"a": sel}
            return finalize(record, row)
        finally:
            shutil.rmtree(img_work, ignore_errors=True)

    def _label_one(r: dict) -> dict:
        try:
            return _call_subject_label(
                r, "a", args.vlm_base_url, args.vlm_model, args.vlm_timeout,
                api_key=getattr(args, "vlm_api_key", "EMPTY"),
            )[2]
        except Exception as e:  # noqa: BLE001 - 源图缺失/损坏等，单图记账不拖垮批
            return {"status": "source_unreadable", "error": type(e).__name__}

    def label_chunk(chunk: list) -> list:
        return list(pool.map(_label_one, chunk))

    chunks = [todo[i:i + args.chunk] for i in range(0, len(todo), args.chunk)]
    prefetch = ThreadPoolExecutor(max_workers=1)
    try:
        next_labels = prefetch.submit(label_chunk, chunks[0])
        t0 = time.perf_counter()
        n_done = 0
        for ci, chunk in enumerate(chunks):
            labels = next_labels.result()
            if ci + 1 < len(chunks):   # SAM 忙时预取下一块标注，vLLM(卡1)保持高占用
                next_labels = prefetch.submit(label_chunk, chunks[ci + 1])
            sam_rows, img_works, sam_meta = [], [], []
            sel_futures = []
            for row, label in zip(chunk, labels):
                cache_dir = row["cache_dir"]
                if label.get("status") == "transport_error":
                    bump("transport_skipped", row["asset_id"])
                    continue
                if label.get("status") != "ok":
                    os.makedirs(cache_dir, exist_ok=True)
                    _write_meta(cache_dir, {"asset_id": row["asset_id"],
                                            "source_path": row["source_path"],
                                            "status": "subject_label_error",
                                            "reason": label.get("status")})
                    bump("subject_label_error", row["asset_id"])
                    continue
                if not label.get("has_localizable_subject"):
                    os.makedirs(cache_dir, exist_ok=True)
                    _write_meta(cache_dir, {"asset_id": row["asset_id"],
                                            "source_path": row["source_path"],
                                            "status": "no_subject",
                                            "reason": label.get("reason_code")})
                    bump("no_subject", row["asset_id"])
                    continue
                sam_rows.append(dict(row, main_subject=str(label.get("sam_prompt") or "")))
                img_works.append(work / row["asset_id"])
                sam_meta.append((row, label))
            if sam_rows and masker is None:
                masker = Sam3Masker(device=args.device, score_threshold=args.min_score)
                masker._ensure_loaded()
            for b0 in range(0, len(sam_rows), args.sam_batch):
                b1 = b0 + args.sam_batch
                records = _sam_forward_batch(masker, sam_rows[b0:b1], img_works[b0:b1],
                                             args.min_score, args.dedupe_iou)
                for record, (row, label), img_work in zip(records, sam_meta[b0:b1],
                                                          img_works[b0:b1]):
                    sel_futures.append((row["asset_id"], pool.submit(
                        _post_and_select, record, row, label, img_work)))
            for asset_id, fut in sel_futures:
                bump(fut.result(), asset_id)
            n_done += len(chunk)
            rate = n_done / max(time.perf_counter() - t0, 1e-6)
            eta_h = (len(todo) - n_done) / max(rate, 1e-6) / 3600
            print(f"[{n_done}/{len(todo)}] {rate:.2f} img/s eta={eta_h:.1f}h "
                  f"{json.dumps(stats, ensure_ascii=False)}", flush=True)
        print("done:", json.dumps(stats, ensure_ascii=False))
        return statuses
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        prefetch.shutdown(wait=True, cancel_futures=True)
        _shutdown_post_pool()
        if masker is not None:
            del masker
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def relabel_sources(
    rows: list[dict],
    *,
    cache_root: str,
    device: str,
    seed: int,
    vlm_base_url: str,
    vlm_api_key: str,
    vlm_model: str,
) -> dict[str, str]:
    """Force a bounded batch of source rows through the retained instance protocol."""
    args = argparse.Namespace(
        pool=None,
        cache_root=cache_root,
        device=device,
        shard_i=0,
        shard_n=1,
        limit=0,
        chunk=max(1, min(256, len(rows))),
        sam_batch=8,
        work_dir=os.path.join(cache_root, "_instance_work"),
        min_score=0.30,
        dedupe_iou=0.92,
        max_proposals=16,
        focus_radius=0.025,
        max_mask_area=0.85,
        max_group_envelope=0.67,
        seed=seed,
        vlm_base_url=vlm_base_url,
        vlm_api_key=vlm_api_key,
        vlm_model=vlm_model,
        vlm_workers=32,
        vlm_timeout=120.0,
    )
    return run(args, rows_override=rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump-pool")
    d.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--pool", required=True)
    r.add_argument("--cache-root", default=CACHE)
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--shard", default="0/1")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--chunk", type=int, default=256)
    r.add_argument("--sam-batch", type=int, default=8)
    r.add_argument("--work-dir", default=os.path.join(CACHE, "_instance_work"))
    r.add_argument("--min-score", type=float, default=0.30)
    r.add_argument("--dedupe-iou", type=float, default=0.92)
    r.add_argument("--max-proposals", type=int, default=16)
    r.add_argument("--focus-radius", type=float, default=0.025)
    r.add_argument("--max-mask-area", type=float, default=0.85)
    r.add_argument("--max-group-envelope", type=float, default=0.67)
    r.add_argument("--seed", type=int, default=20260711)
    r.add_argument("--vlm-base-url",
                   default=os.environ.get("SOURCE_QA_VLLM", "http://localhost:8003/v1"))
    r.add_argument("--vlm-model", default="qwen3_5-35b-a3b")
    r.add_argument("--vlm-workers", type=int, default=32)
    r.add_argument("--vlm-timeout", type=float, default=120.0)
    a = ap.parse_args()
    if a.cmd == "dump-pool":
        dump_pool(a.out)
    else:
        a.shard_i, a.shard_n = (int(x) for x in a.shard.split("/"))
        run(a)


if __name__ == "__main__":
    main()
