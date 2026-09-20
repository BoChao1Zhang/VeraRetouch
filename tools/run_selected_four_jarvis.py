#!/usr/bin/env python3
"""Render selected JarvisEvo round-one parameters with the official LrC farm."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("/home/bc/data/runs/paper_selected_four_pix3200_20260920"))
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--wait-client-seconds", type=float, default=0,
                    help="Wait for an eligible client before submitting any task")
    args = ap.parse_args()
    farm = args.root / "jarvisevo_lr_farm"
    os.environ["VERA_OUT_ROOT"] = str(farm / "qa_root")
    os.environ["SOURCE_QA_LR_URL"] = args.url
    os.environ["SOURCE_QA_LR_CONCURRENCY"] = "1"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import requests
    from PIL import Image
    from q3vl.whatb.pubbench.epr037_lr_render import params_to_lua
    from dataset_build.source_qa import config, lr_render

    jobs = [json.loads(line) for line in (args.root / "jarvisevo_jobs.jsonl").read_text().splitlines() if line.strip()]
    Path(config.RENDER_STAGE).mkdir(parents=True, exist_ok=True)
    audit_path = farm / "render_progress.jsonl"
    completed = {}
    if audit_path.exists():
        for line in audit_path.read_text().splitlines():
            row = json.loads(line)
            if row.get("ok"):
                completed[row["sample_id"]] = row
    prepared = []
    for job in jobs:
        source = Path(job["input_path"])
        params = Path(job["out_json"])
        rec = json.loads(params.read_text())
        if not rec.get("ok") or rec.get("n_rounds") != 1:
            raise RuntimeError(f"No valid round-one inference: {params}")
        if rec["input_path"] != str(source) or rec["instruction"] != job["instruction"]:
            raise RuntimeError(f"Inference job mismatch: {params}")
        with Image.open(source) as im:
            if [im.height, im.width] != job["expect_hw"]:
                raise RuntimeError(f"Input dimensions changed: {source}")
        lua = farm / "lua" / f"{job['sample_id']}.lua"
        ok, why, has_mask = params_to_lua(params, lua)
        if not ok:
            raise RuntimeError(f"Official Lua conversion failed: {params}: {why}")
        prepared.append((job, source, params, lua, has_mask))
    if args.prepare_only:
        print(f"Prepared {len(prepared)} official Lua presets; no render submitted.", flush=True)
        return 0

    deadline = time.monotonic() + args.wait_client_seconds
    announced = False
    while True:
        health = requests.get(args.url + "/api/health", timeout=10)
        health.raise_for_status()
        clients = requests.get(args.url + "/api/clients", timeout=10).json()
        eligible = [c for c in clients.get("clients", []) if c.get("status") == "online" and c.get("eligible")]
        if eligible or time.monotonic() >= deadline:
            break
        if not announced:
            print("Waiting for an eligible Lightroom client; no task submitted.", flush=True)
            announced = True
        time.sleep(min(10, max(0, deadline - time.monotonic())))
    if not eligible:
        print("BLOCKED: no eligible online Lightroom Classic client; no task submitted.", flush=True)
        return 2
    for job, source, params, lua, has_mask in prepared:
        out = source.parent / "jarvisevo.png"
        if out.exists():
            previous = completed.get(job["sample_id"], {})
            if (previous.get("output_sha256") == sha256(out)
                    and previous.get("input_sha256") == sha256(source)
                    and previous.get("params_sha256") == sha256(params)):
                print(f"{job['sample_id']}: existing audited official render verified", flush=True)
                continue
            raise RuntimeError(f"Refusing to overwrite an unverified existing result: {out}")
        source_hash = sha256(source)
        row = {"sample_id": job["sample_id"], "input_path": str(source),
               "input_sha256": source_hash, "params_path": str(params),
               "params_sha256": sha256(params), "lua_sha256": sha256(lua),
               "has_ai_mask": has_mask, "engine": "lrc", "n_rounds": 1,
               "source_format": "jpeg_q0.8", "url": args.url,
               "clients_at_start": clients, "expect_hw": job["expect_hw"], "attempts": []}
        for attempt in range(3):
            result = lr_render.submit_and_wait(str(source), str(lua))
            row["attempts"].append(result)
            if result.get("ok") or not lr_render._is_retryable(result):
                break
            if attempt < 2:
                time.sleep(min(3 * 2 ** attempt, 20))
        row["ok"] = bool(result.get("ok"))
        if row["ok"]:
            rendered = Path(result["after_path"])
            row["jpeg_sha256"] = sha256(rendered)
            status = requests.get(args.url + "/api/task_status/" + result["task_id"], timeout=10)
            status.raise_for_status()
            row["official_task_status"] = status.json()
            with Image.open(rendered) as im:
                im.load()
                if [im.height, im.width] != job["expect_hw"]:
                    row.update(ok=False, error="size_mismatch", got_hw=[im.height, im.width])
                elif sha256(source) != source_hash:
                    row.update(ok=False, error="input_bytes_changed")
                else:
                    tmp = farm / f"{job['sample_id']}.png.part"
                    im.convert("RGB").save(tmp, format="PNG", compress_level=6)
                    tmp.replace(out)
                    row.update(output_path=str(out), output_sha256=sha256(out))
        row["timestamp"] = time.time()
        with audit_path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        print(f"{job['sample_id']}: ok={row['ok']} task={result.get('task_id')}", flush=True)
        if not row["ok"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
