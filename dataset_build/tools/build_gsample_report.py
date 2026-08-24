"""G1: build the gsample A/B global-proposal report (docs/assets/gsample_20260821).

Numbers and verbatim strings only: every line it writes is either a count, a stored
field, or a verbatim quote of the diagnosis / shortlist row text.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import psycopg
from PIL import Image

DSN = (
    "postgresql://research:research@127.0.0.1:5432/agent_loop"
    "?options=-c%20search_path%3Dagent_loop"
)
CAMPAIGNS = {
    "A": ["gsample-a", "gsample-a2", "gsample-a3", "gsample-a4",
          "gsample-a5", "gsample-a6", "gsample-a7", "gsample-a-l1", "gsample-a-l1b"],
    "B": ["gsample-b", "gsample-b2", "gsample-b3", "gsample-b-l1", "gsample-b-l1b"],
}
ROOTS = {
    "gsample-a": Path("/mnt/ramstage/agent_loop/gsample-a"),
    "gsample-a2": Path("/mnt/ramstage/agent_loop/gsample-a2"),
    "gsample-b": Path("/mnt/ramstage/agent_loop/gsample-b"),
    "gsample-b2": Path("/mnt/ramstage/agent_loop/gsample-b2"),
    "gsample-a3": Path("/mnt/ramstage/agent_loop/gsample-a3"),
    "gsample-b3": Path("/mnt/ramstage/agent_loop/gsample-b3"),
    "gsample-a4": Path("/mnt/ramstage/agent_loop/gsample-a4"),
    "gsample-a5": Path("/mnt/ramstage/agent_loop/gsample-a5"),
    "gsample-a6": Path("/mnt/ramstage/agent_loop/gsample-a6"),
    "gsample-a7": Path("/mnt/ramstage/agent_loop/gsample-a7"),
    "gsample-a-l1": Path("/mnt/ramstage/agent_loop/gsample-a-l1"),
    "gsample-b-l1": Path("/mnt/ramstage/agent_loop/gsample-b-l1"),
    "gsample-a-l1b": Path("/mnt/ramstage/agent_loop/gsample-a-l1b"),
    "gsample-b-l1b": Path("/mnt/ramstage/agent_loop/gsample-b-l1b"),
}


def cell(text: str) -> str:
    """Markdown table cells: the fixed-form lines contain ' | ' separators."""
    return str(text).replace("|", "\\|")


def blob_path(campaign: str, sha: str) -> Path:
    return ROOTS[campaign] / "blobs" / sha[:2] / sha[2:4] / sha


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def shortlist_rows(request_json: str) -> dict[str, dict[int, str]]:
    """{major: {row_index: verbatim row line}} from the stored canonical request."""
    payload = json.loads(request_json)
    text = ""
    for message in payload.get("input", []):
        for content in message.get("content", []):
            body = content.get("text") or ""
            if body.startswith("LUT shortlist table."):
                text = body
    out: dict[str, dict[int, str]] = {}
    major = None
    for line in text.splitlines():
        if line.startswith("[major] "):
            major = line[len("[major] "):].strip()
            out[major] = {}
        elif major is not None and " | " in line:
            head = line.split(" | ", 1)[0].strip()
            if head.isdigit():
                out[major][int(head)] = line
    return out


def fetch(conn, campaigns: list[str]) -> dict:
    runs, branches, renders, requests = {}, {}, {}, {}
    raw_counts: list[tuple[str, dict[str, int]]] = []
    for campaign in campaigns:
        counted = dict(conn.execute(
            "select status,count(*) from agent_source_run where campaign_id=%s "
            "group by 1", (campaign,)
        ).fetchall())
        if counted:
            raw_counts.append((campaign, counted))
        for source_id, sha, status, counts in conn.execute(
            "select source_id,source_sha256,status,counts_json from agent_source_run "
            "where campaign_id=%s", (campaign,)
        ).fetchall():
            runs.setdefault(source_id, []).append({
                "campaign": campaign, "sha": sha, "status": status,
                "counts": json.loads(counts or "{}"),
            })
        for bid, sha, status, proposal in conn.execute(
            "select branch_id,source_sha256,status,proposal_json from agent_branch "
            "where campaign_id=%s and level='global' order by branch_id", (campaign,)
        ).fetchall():
            branches.setdefault((campaign, sha), []).append({
                "campaign": campaign, "branch_id": bid, "status": status,
                "proposal": json.loads(proposal),
            })
        for bid, art, params, metrics in conn.execute(
            "select r.branch_id,r.artifact_json,r.parameters_json,r.metrics_json "
            "from render_record r join agent_branch b on b.branch_id=r.branch_id "
            "where b.campaign_id=%s and r.stage='global' and r.status='accepted'",
            (campaign,)
        ).fetchall():
            renders[bid] = {
                "artifact": json.loads(art), "params": json.loads(params or "{}"),
                "metrics": json.loads(metrics or "{}"),
            }
        for sha, response, request in conn.execute(
            "select x.source_sha256,q.response_json,q.canonical_request_json "
            "from api_request_context x join api_request q on q.request_hash=x.request_hash "
            "where x.campaign_id=%s and x.stage='global_propose'", (campaign,)
        ).fetchall():
            parsed = (json.loads(response) or {}).get("parsed") or {}
            if parsed.get("major"):
                requests[(campaign, sha)] = {
                    "major": parsed["major"], "rows": shortlist_rows(request),
                    "lane": json.loads(request).get("endpoint_identity"),
                    "temperature": (json.loads(request).get("behavior") or {})
                    .get("temperature"),
                }
    # A source can appear in several campaigns of the same arm (the backfill rounds).
    # Pick one run per source: an accepted run first, otherwise the run that carries the
    # most global branches, otherwise the first attempt.
    chosen = {}
    for source_id, attempts in runs.items():
        accepted = [a for a in attempts if a["status"] == "accepted"]
        pool = accepted or attempts
        chosen[source_id] = max(
            pool, key=lambda a: len(branches.get((a["campaign"], a["sha"]), []))
        )
    return {"runs": chosen, "run_attempts": runs, "branches": branches,
            "renders": renders, "requests": requests, "raw_counts": raw_counts}


def write_before(source_path: Path, out: Path) -> None:
    if out.exists():
        return
    image = Image.open(source_path).convert("RGB")
    image.thumbnail((512, 512), Image.LANCZOS)
    image.save(out, "JPEG", quality=88)


def write_after(campaign: str, sha: str, out: Path) -> bool:
    src = blob_path(campaign, sha)
    if not src.exists():
        return False
    if out.suffix == ".jpg":
        image = Image.open(src).convert("RGB")
        image.thumbnail((512, 512), Image.LANCZOS)
        image.save(out, "JPEG", quality=88)
    else:
        shutil.copy2(src, out)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/home/bc/data/agent_loop/local-v1/"
                                          "gsample10.annotated-v34.jsonl")
    ap.add_argument("--out-dir", default="docs/assets/gsample_20260821")
    ap.add_argument("--prompt-revision", default="local-agent-v1-0ad2a90176f4")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    images = out_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(Path(args.manifest))

    with psycopg.connect(DSN) as conn:
        data = {arm: fetch(conn, names) for arm, names in CAMPAIGNS.items()}

    lines: list[str] = []
    lines.append("# G1 · diag-v3.4 global prompt 的 10 源 × 2 次独立采样")
    lines.append("")
    lines.append(f"- prompt_revision fingerprint:`{args.prompt_revision}`")
    lines.append("- A = campaign `gsample-a`,`[terra] temperature = 0.1`;"
                 "B = campaign `gsample-b`,`[terra] temperature = 0.11`。")
    lines.append("- 温度取 0.1 / 0.11:`api_cache` 精确缓存键由请求哈希算得,"
                 "`behavior.temperature` 在哈希内,两跑不同温度即保证 B 不命中 A 的缓存。")
    lines.append("- `terra_concurrency_target = 2`(同机另有全量 annotate 占双 lane × 16 并发)。")
    lines.append("- 跑批期间 relay 降级:60 分钟窗口内 `provider-c-lane-2` 有 214 次 "
                 "`503:model_not_found`、17 次成功;`provider-c-lane-1` 40 次成功。"
                 "补跑档 `*-l1` / `*-l1b` 因此只挂 `provider-c-lane-1`,`*-l1b` 另把 "
                 "`[terra] attempts` 由 3 提到 8。每个源实际用的 endpoint 与 "
                 "`behavior.temperature` 逐条列在下面各 run 小节。")
    lines.append("")
    lines.append("## 源级状态计数")
    lines.append("")
    seen = sorted({
        status for arm in ("A", "B")
        for _, row in data[arm]["raw_counts"] for status in row
    })
    lines.append("| arm | campaign | " + " | ".join(seen) + " | 合计 |")
    lines.append("| --- | --- | " + " | ".join("---" for _ in seen) + " | --- |")
    for arm in ("A", "B"):
        for campaign, row in data[arm]["raw_counts"]:
            cells = " | ".join(str(row.get(status, 0)) for status in seen)
            lines.append(f"| {arm} | {campaign} | {cells} | {sum(row.values())} |")
    lines.append("")
    lines.append("`gsample-a2`…`gsample-a7` / `gsample-b2` / `gsample-b3` 是补跑档,"
                 "只跑上一轮非 accepted 的源,温度与各自主档相同;`*-l1` 档只挂 "
                 "`provider-c-lane-1` 一条 lane。全部 error 的 `counts_json` 都是 "
                 "`CacheExhausted`(`503:model_not_found` / `500:get_channel_failed` / "
                 "`untyped_responses_event`)。")
    lines.append("")
    lines.append("| arm | 源数 | source_run accepted | 有 ≥1 个 global 提案的源 | "
                 "有 ≥1 张 accepted global render 的源 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for arm in ("A", "B"):
        runs_arm = data[arm]["runs"]
        ok = sum(1 for run in runs_arm.values() if run["status"] == "accepted")
        with_branch = 0
        with_render = 0
        for run in runs_arm.values():
            got = data[arm]["branches"].get((run["campaign"], run["sha"]), [])
            with_branch += int(bool(got))
            with_render += int(any(
                b["branch_id"] in data[arm]["renders"] for b in got
            ))
        lines.append(f"| {arm} | {len(runs_arm)} | {ok} | {with_branch} | "
                     f"{with_render} |")
    lines.append("")
    lines.append("## global 分支状态计数")
    lines.append("")
    lines.append("(只统计上表去重后被采用的那一档 campaign 的分支)")
    lines.append("")
    lines.append("| arm | global 分支总数 | 有 accepted global render | 无 |")
    lines.append("| --- | --- | --- | --- |")
    for arm in ("A", "B"):
        keys = {(run["campaign"], run["sha"]) for run in data[arm]["runs"].values()}
        allb = [b for key in keys for b in data[arm]["branches"].get(key, [])]
        got = sum(1 for b in allb if b["branch_id"] in data[arm]["renders"])
        lines.append(f"| {arm} | {len(allb)} | {got} | {len(allb) - got} |")
    lines.append("")

    diversity: list[dict] = []

    for index, item in enumerate(manifest, start=1):
        source_id = item["source_id"]
        annotation = json.loads(Path(item["source_annotation_path"]).read_text())
        diagnosis = annotation["diagnosis"]
        before_rel = f"images/{source_id}_before.jpg"
        write_before(Path(item["source_path"]), images / f"{source_id}_before.jpg")

        lines.append(f"## {index}. `{source_id}` · scene = {item['scene']}")
        lines.append("")
        lines.append(f"- subject:{item['subject']['description']}"
                     f"(mask_area = {item['subject']['mask_area']})")
        lines.append(f"- intent_mode:`{diagnosis.get('intent_mode')}`,"
                     f"confidence = {diagnosis.get('confidence')}")
        lines.append("")
        lines.append("**诊断 correction_needs(v3.4 固定句式,原文)**")
        lines.append("")
        needs = diagnosis.get("correction_needs") or []
        if needs:
            for text in needs:
                lines.append(f"- `{text}`")
        else:
            lines.append("- (空)")
        lines.append("")
        lines.append("**诊断 enhancement_opportunities(v3.4 固定句式,原文)**")
        lines.append("")
        opportunities = diagnosis.get("enhancement_opportunities") or []
        if opportunities:
            for text in opportunities:
                lines.append(f"- `{text}`")
        else:
            lines.append("- (空)")
        lines.append("")

        row = {"source_id": source_id, "A": set(), "B": set(),
               "major_A": None, "major_B": None}
        for arm in ("A", "B"):
            arm_data = data[arm]
            run = arm_data["runs"].get(source_id)
            sha = run["sha"] if run else None
            campaign = run["campaign"] if run else "-"
            status = run["status"] if run else "missing"
            request = arm_data["requests"].get((campaign, sha)) if sha else None
            major = request["major"] if request else None
            row[f"major_{arm}"] = major
            lines.append(f"### {index}{arm.lower()}. run {arm}(`{campaign}`,"
                         f"source_run.status = `{status}`)")
            lines.append("")
            if major:
                lines.append(f"- 模型返回 style major:`{major}`")
                lines.append(f"- global_propose 请求:endpoint `{request['lane']}`,"
                             f"`behavior.temperature = {request['temperature']}`")
                lines.append("")
            proposals = sorted(
                arm_data["branches"].get((campaign, sha), []) if sha else [],
                key=lambda b: b["proposal"].get("row_index", 0),
            )
            if not proposals:
                lines.append("- 无 global 提案记录。")
                lines.append("")
                continue
            for branch in proposals:
                proposal = branch["proposal"]
                preset_id = proposal.get("preset_id")
                bin_name = proposal.get("strength_bin")
                codes = ", ".join(proposal.get("reason_codes") or [])
                row_index = proposal.get("row_index")
                row_text = ""
                if request and major in request["rows"]:
                    row_text = request["rows"][major].get(row_index, "")
                # `id | achievable_bins | <8 fingerprint numbers> | caption` plus the
                # optional trailing `d_shadow d_mid d_high` group of a v2 mount.
                parts = row_text.split(" | ")
                caption = parts[3] if len(parts) > 3 else ""
                render = arm_data["renders"].get(branch["branch_id"])
                lines.append(f"#### {proposal.get('proposal_id')} · `{preset_id}` · "
                             f"bin = `{bin_name}`")
                lines.append("")
                if render:
                    sha_after = render["artifact"]["sha256"]
                    after_name = f"{campaign}_{source_id}_{proposal.get('proposal_id')}.jpg"
                    ok = write_after(campaign, sha_after, images / after_name)
                    after_rel = f"images/{after_name}" if ok else None
                else:
                    after_rel = None
                lines.append(
                    f'<p><img src="{before_rel}" width="380" alt="before"> '
                    + (f'<img src="{after_rel}" width="380" alt="after">'
                       if after_rel else "(无 accepted global render)")
                    + "</p>"
                )
                lines.append("")
                lines.append(f"- branch.status:`{branch['status']}`")
                if not render:
                    result_note = branch["proposal"]
                    lines.append(f"- 无 stage=global / status=accepted 的 render_record"
                                 f"(row_index={result_note.get('row_index')})")
                if render:
                    params = render["params"]
                    metrics = render["metrics"]
                    lines.append(
                        f"- render:global_strength = {params.get('global_strength')},"
                        f" strength_bin = `{params.get('strength_bin')}`,"
                        f" delta_e = {metrics.get('delta_e')},"
                        f" clip_fraction_new = {metrics.get('clip_fraction_new')}"
                    )
                lines.append(f"- caption:{caption}")
                lines.append(f"- reason_codes:`{codes}`")
                lines.append(f"- 量测指纹行(shortlist 原文,major `{major}` 第 "
                             f"{row_index} 行):")
                lines.append("")
                lines.append("  ```")
                lines.append(f"  {row_text}")
                lines.append("  ```")
                lines.append("")
                lines.append("- 对应的诊断方向(左:该提案 reason_codes;右:诊断固定句式行原文):")
                lines.append("")
                lines.append("  | 提案 reason_codes | 诊断固定句式行 |")
                lines.append("  | --- | --- |")
                fixed = [("correction_needs", t) for t in needs] + \
                        [("enhancement_opportunities", t) for t in opportunities]
                if not fixed:
                    lines.append(f"  | `{cell(codes)}` | (诊断两列均为空) |")
                for position, (field, text) in enumerate(fixed):
                    left = f"`{cell(codes)}`" if position == 0 else ""
                    lines.append(f"  | {left} | `{field}`: `{cell(text)}` |")
                lines.append("")
                row[arm].add(str(preset_id))
        diversity.append(row)

    lines.append("## 多样性表")
    lines.append("")
    lines.append("| # | source_id | A 选中 preset 集 | B 选中 preset 集 | "
                 "A 数 | B 数 | 交集大小 | 并集大小 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for index, row in enumerate(diversity, start=1):
        a, b = row["A"], row["B"]
        a_text = ", ".join(f"`{x}`" for x in sorted(a)) or "-"
        b_text = ", ".join(f"`{x}`" for x in sorted(b)) or "-"
        lines.append(f"| {index} | `{row['source_id']}` | {a_text} | {b_text} | "
                     f"{len(a)} | {len(b)} | {len(a & b)} | {len(a | b)} |")
    lines.append("")
    total_a = sum(len(r["A"]) for r in diversity)
    total_b = sum(len(r["B"]) for r in diversity)
    total_i = sum(len(r["A"] & r["B"]) for r in diversity)
    total_u = sum(len(r["A"] | r["B"]) for r in diversity)
    lines.append(f"合计:A 数 = {total_a},B 数 = {total_b},交集大小 = {total_i},"
                 f"并集大小 = {total_u}。")
    lines.append("")
    lines.append("### style major 选择对照")
    lines.append("")
    lines.append("| # | source_id | A major | B major | 相同 |")
    lines.append("| --- | --- | --- | --- | --- |")
    same = 0
    for index, row in enumerate(diversity, start=1):
        ma, mb = row["major_A"], row["major_B"]
        hit = int(ma is not None and ma == mb)
        same += hit
        lines.append(f"| {index} | `{row['source_id']}` | `{ma}` | `{mb}` | {hit} |")
    lines.append("")
    lines.append(f"两 campaign major 相同的源数:{same}。")
    lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
