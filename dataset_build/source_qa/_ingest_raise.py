"""RAISE-6k: extract embedded full-res JPEG preview from each NEF (CPU-cheap,
~0.8s vs ~17s for a full demosaic) into jpg_preview/, then upsert as image assets
(corpus='raise'). The two-stage LLM QA (run-img --all) then picks them up.
Run: python -m dataset_build.source_qa._ingest_raise [--limit N]
"""
from __future__ import annotations
import io, os, sys, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from . import db

RAW_DIR = "/home/bc/data/datasets/RAISE-6k/raw"
JPG_DIR = "/home/bc/data/datasets/RAISE-6k/jpg_preview"


def _extract(nef: str) -> str | None:
    """NEF embedded JPEG → jpg_preview/<stem>.jpg; return jpg path or None."""
    import rawpy
    from PIL import Image
    stem = os.path.splitext(os.path.basename(nef))[0]
    out = os.path.join(JPG_DIR, stem + ".jpg")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out
    try:
        with rawpy.imread(nef) as r:
            th = r.extract_thumb()
        if th.format == rawpy.ThumbFormat.JPEG:
            with open(out, "wb") as f:
                f.write(th.data)
        else:  # bitmap thumb → encode JPEG
            Image.fromarray(th.data).save(out, "JPEG", quality=95)
        return out if os.path.getsize(out) > 0 else None
    except Exception as e:  # noqa: BLE001
        print(f"[raise] {stem} FAIL: {str(e)[:80]}", file=sys.stderr)
        return None


def main(limit=None, workers=12):
    os.makedirs(JPG_DIR, exist_ok=True)
    nefs = sorted(os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR) if f.lower().endswith(".nef"))
    if limit:
        nefs = nefs[:limit]
    print(f"[raise] {len(nefs)} NEF → extract embedded JPEG", file=sys.stderr)
    conn = db.connect()
    run_id = db.start_run(conn, "ingest_raise", {"n": len(nefs)})
    lock = threading.Lock()
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_extract, n): n for n in nefs}
        for j, fu in enumerate(as_completed(futs)):
            nef = futs[fu]
            jpg = fu.result()
            stem = os.path.splitext(os.path.basename(nef))[0]
            if jpg:
                with lock:
                    db.upsert_asset(conn, {"asset_id": "raise_" + stem, "asset_type": "image",
                                           "corpus": "raise", "path": jpg})
                    ok += 1
                    if ok % 500 == 0:
                        conn.commit(); print(f"[raise] {ok} ingested", file=sys.stderr)
            else:
                fail += 1
    conn.commit()
    db.finish_run(conn, run_id, {"ok": ok, "fail": fail})
    conn.close()
    print(f"[raise] DONE: ingested={ok} failed={fail}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    main(limit=a.limit, workers=a.workers)
