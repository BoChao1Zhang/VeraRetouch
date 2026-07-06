"""scene='any' 桶细分类回填：用已有 source_captions 文本走 vLLM 分到 9 类受控词表。

采样池 79% 压在 any 桶（2026-07-06 实测 25,503/32,295），9 类均匀采样前必须细分。
caption 覆盖率 100%，纯文本分类即可（不重发图像，单条 ~200 tok，比视觉路快 20x）。
失败/不确定的保持 any，重跑自然续传（已更新的行不再命中 WHERE scene='any'）。

用法:
    PYTHONPATH=/home/bc/VeraRetouch:/home/bc/VeraRetouch/dataset_build/src \
    /home/bc/.venvs/iaa437/bin/python -m dataset_build.source_qa.scene_backfill [--limit N]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import requests

from . import config, db

SCENES = ("portrait", "landscape", "food", "still_life", "architecture",
          "night", "street", "wedding", "product")
BATCH = 10
WORKERS = 24
_POOL_WHERE = ("asset_type='image' AND b_quality=3 AND dup_of IS NULL "
               "AND iaa_mixed IS NOT NULL AND iaa_mixed>=55 "
               "AND COALESCE(scene,'any')='any'")

_PROMPT = """你是照片场景分类器。对下面每条图片描述，从这 9 个类别中选一个：
portrait(人像特写/单人主体) landscape(自然风光) food(食物) still_life(静物摆拍)
architecture(建筑) night(夜景) street(街拍/城市生活) wedding(婚礼婚纱) product(商品图)
无法确定或都不符合时用 any。只输出 JSON 数组，长度与输入条数一致，如 ["portrait","any",...]。

{items}"""


def _classify_batch(rows: list) -> dict:
    items = "\n".join(
        f"{i+1}. {r['caption']}（主体: {r['main_subject'] or '未知'}）"
        for i, r in enumerate(rows))
    resp = requests.post(
        f"{config.VLLM_BASE_URL}/chat/completions",
        json={"model": config.VLLM_MODEL, "temperature": 0, "max_tokens": 200,
              "chat_template_kwargs": {"enable_thinking": False},
              "messages": [{"role": "user", "content": _PROMPT.format(items=items)}]},
        timeout=120)
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    m = re.search(r"\[.*\]", text, re.S)
    labels = json.loads(m.group(0)) if m else []
    if len(labels) != len(rows):
        raise ValueError(f"标签数 {len(labels)} != 输入数 {len(rows)}")
    return {r["asset_id"]: lab for r, lab in zip(rows, labels) if lab in SCENES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（冒烟用）")
    args = ap.parse_args()

    conn = db.connect()
    rows = conn.execute(
        f"SELECT a.asset_id, c.caption, c.main_subject FROM assets a "
        f"JOIN source_captions c USING(asset_id) WHERE {_POOL_WHERE} "
        f"ORDER BY a.asset_id" + (f" LIMIT {args.limit}" if args.limit else "")
    ).fetchall()
    conn.close()
    batches = [rows[i:i + BATCH] for i in range(0, len(rows), BATCH)]
    print(f"[scene_backfill] {len(rows)} 张 any 待分类，{len(batches)} 批", flush=True)

    lock = threading.Lock()
    stats = {"done": 0, "updated": 0, "failed": 0}

    def work(batch: list) -> None:
        wconn = db.connect()
        try:
            labels = _classify_batch(batch)
            for aid, scene in labels.items():
                wconn.execute("UPDATE assets SET scene=%s WHERE asset_id=%s", (scene, aid))
            wconn.commit()
            with lock:
                stats["updated"] += len(labels)
        except Exception as e:  # noqa: BLE001 - 单批失败保持 any，重跑续传
            with lock:
                stats["failed"] += len(batch)
            print(f"[scene_backfill] 批失败({len(batch)}条): {str(e)[:100]}",
                  file=sys.stderr, flush=True)
        finally:
            wconn.close()
            with lock:
                stats["done"] += len(batch)
                if stats["done"] % 500 < BATCH:
                    print(f"[scene_backfill] {stats['done']}/{len(rows)} "
                          f"updated={stats['updated']} failed={stats['failed']}", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(work, batches))
    print(f"[scene_backfill] 完成: {json.dumps(stats)}", flush=True)

    conn = db.connect()
    print("[scene_backfill] 分类后池内分布:")
    for r in conn.execute(
            "SELECT COALESCE(scene,'any') s, count(*) FROM assets WHERE "
            "asset_type='image' AND b_quality=3 AND dup_of IS NULL "
            "AND iaa_mixed IS NOT NULL AND iaa_mixed>=55 GROUP BY 1 ORDER BY 2 DESC"):
        print(f"  {r['s']:<14} {r['count']}")
    conn.close()


if __name__ == "__main__":
    main()
