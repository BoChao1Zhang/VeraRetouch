#!/usr/bin/env python3
"""Build a human-review report for a random sample of offline v3.4 diagnoses.

For each sampled source the report shows, side by side, the 512px source
preview and the histogram board that the diagnose call actually saw (located
in the artifact store by ``provenance.board_sha256``), followed by the five
diagnosis fields verbatim.

Usage::

    python dataset_build/tools/build_diagnose_sample_report.py \
        --manifest /home/bc/data/agent_loop/local-v1/sources5k.annotated-v34.jsonl \
        --seed 20260824 --n 30 \
        --out docs/assets/diagnose_v34_sample30_20260824
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

# fixed-sentence vocabulary (task card S30)
AXES = {
    "exposure",
    "band_lightness",
    "contrast",
    "hue",
    "cast",
    "saturation",
}
SCOPES = {
    "global",
    "shadows",
    "midtones",
    "highlights",
    "subject",
    "skin",
    "sky",
    "foliage",
    "background",
}

FIVE_FIELDS = [
    "correction_needs",
    "enhancement_opportunities",
    "evidence",
    "forbidden_directions",
    "preserve_intent",
]

PANEL_RE = re.compile(r"panel\s*\d", re.IGNORECASE)
EFFORT_RE = re.compile(rb'"reasoning_effort"\s*:\s*"([a-z]+)"')


# --------------------------------------------------------------------------- #
# parsing helpers
# --------------------------------------------------------------------------- #
def split_components(line: str) -> list[str]:
    """Split an enhancement_opportunities line into its component clauses.

    Format: ``<style phrase> => <component 1> ; <component 2> ...``
    A correction_needs line is a bare component and returns ``[line]``.
    """
    body = line.split("=>", 1)[1] if "=>" in line else line
    return [c.strip() for c in body.split(";") if c.strip()]


def check_component(comp: str) -> tuple[bool, str]:
    """Return (ok, reason) for a ``<axis>|<scope>|<state>|<move>`` clause."""
    parts = [p.strip() for p in comp.split("|")]
    if len(parts) != 4:
        return False, f"segments={len(parts)}"
    axis, scope, state, move = parts
    if axis not in AXES:
        return False, f"axis={axis!r}"
    if scope not in SCOPES:
        return False, f"scope={scope!r}"
    if not state:
        return False, "empty state"
    if not move:
        return False, "empty move"
    return True, ""


def style_phrase(line: str) -> str:
    return line.split("=>", 1)[0].strip() if "=>" in line else ""


# --------------------------------------------------------------------------- #
# artifact helpers
# --------------------------------------------------------------------------- #
def blob_path(store: Path, sha: str) -> Path:
    return store / sha[0:2] / sha[2:4] / sha


def make_preview(src: Path, dst: Path, max_side: int) -> bool:
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side), Image.LANCZOS)
            im.save(dst, "JPEG", quality=88)
        return True
    except Exception:
        return False


def full_effort_tally(ann_dir: Path) -> Counter:
    tally: Counter = Counter()
    for p in sorted(ann_dir.glob("*.json")):
        m = EFFORT_RE.search(p.read_bytes())
        tally[m.group(1).decode() if m else "missing"] += 1
    return tally


# --------------------------------------------------------------------------- #
# markdown helpers
# --------------------------------------------------------------------------- #
def md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def bullet_block(items: list[str]) -> list[str]:
    if not items:
        return ["_(empty)_", ""]
    return [f"{i + 1}. {it}" for i, it in enumerate(items)] + [""]


# --------------------------------------------------------------------------- #
def build(args: argparse.Namespace) -> None:
    manifest = Path(args.manifest)
    out_dir = Path(args.out)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    store = Path(args.blob_store)

    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    sample = random.Random(args.seed).sample(rows, args.n)

    recs = []
    for row in sample:
        ann = json.loads(Path(row["source_annotation_path"]).read_text())
        diag = ann["diagnosis"]
        prov = ann.get("provenance", {})
        usage = prov.get("usage", {})
        sid = row["source_id"]

        # images
        src_rel = board_rel = None
        src_png = img_dir / f"{sid}.source.jpg"
        if make_preview(Path(row["source_path"]), src_png, args.preview_px):
            src_rel = f"images/{src_png.name}"
        sha = prov.get("board_sha256")
        if sha:
            bp = blob_path(store, sha)
            if bp.is_file():
                dst = img_dir / f"{sid}.board.png"
                shutil.copyfile(bp, dst)
                board_rel = f"images/{dst.name}"

        # fixed-sentence compliance
        comps = []  # (field, line_idx, comp, ok, reason)
        for field in ("correction_needs", "enhancement_opportunities"):
            for li, line in enumerate(diag.get(field) or []):
                for comp in split_components(line):
                    ok, reason = check_component(comp)
                    comps.append((field, li, comp, ok, reason))

        # panel citations, counted per item across the five fields
        panel_items = {
            f: sum(1 for it in (diag.get(f) or []) if PANEL_RE.search(it))
            for f in FIVE_FIELDS
        }

        recs.append(
            dict(
                sid=sid,
                scene=row.get("scene") or ann.get("scene") or "unknown",
                subject=(row.get("subject") or {}).get("description", ""),
                mask_area=(row.get("subject") or {}).get("mask_area"),
                diag=diag,
                effort=prov.get("reasoning_effort", "?"),
                model=prov.get("model", "?"),
                prompt_rev=prov.get("diagnose_prompt_revision", "?"),
                board_rev=prov.get("board_revision", "?"),
                in_tok=usage.get("input_tokens"),
                out_tok=usage.get("output_tokens"),
                src_rel=src_rel,
                board_rel=board_rel,
                comps=comps,
                panel_items=panel_items,
                n_corr=len(diag.get("correction_needs") or []),
                n_opp=len(diag.get("enhancement_opportunities") or []),
            )
        )

    tally = full_effort_tally(Path(args.annotations_dir)) if args.full_tally else None
    L = render(args, recs, tally)
    (out_dir / "report.md").write_text("\n".join(L))
    print(f"wrote {out_dir / 'report.md'} ({len(recs)} sources)")


def render(args, recs, tally) -> list[str]:
    n = len(recs)
    L: list[str] = []
    L += [
        f"# S30 - offline diagnose v3.4 随机 {n} 源人工审阅板",
        "",
        f"- 标注清单: `{args.manifest}` (5000 行)",
        f"- 抽样方法: `random.Random({args.seed}).sample(rows, {args.n})`,rows 为清单文件的原始行序",
        f"- 随机种子: **{args.seed}**",
        f"- 源图预览: 由 `source_path` 现做 {args.preview_px}px 长边缩略 (LANCZOS)",
        f"- 直方图板: 按 `provenance.board_sha256` 从 `{args.blob_store}` 取出并复制进 `images/`",
        f"- 生成脚本: `dataset_build/tools/build_diagnose_sample_report.py`",
        "",
    ]

    revs = Counter(r["prompt_rev"] for r in recs)
    boards = Counter(r["board_rev"] for r in recs)
    models = Counter(r["model"] for r in recs)
    L += [
        "运行标识(抽样 30 源):",
        "",
        f"- model: {', '.join(f'`{k}` x{v}' for k, v in models.items())}",
        f"- diagnose_prompt_revision: {', '.join(f'`{k}` x{v}' for k, v in revs.items())}",
        f"- board_revision: {', '.join(f'`{k}` x{v}' for k, v in boards.items())}",
        "",
        "## effort 口径",
        "",
        "口径直接读 `provenance.reasoning_effort`(标注 json 内的显式字段),无需从 token 数反推。",
        "",
    ]
    if tally:
        L += ["全量 5000 源:", ""]
        L += ["| reasoning_effort | 源数 |", "| --- | --- |"]
        for k, v in sorted(tally.items()):
            L.append(f"| {k} | {v} |")
        L.append("")

    eff_cnt = Counter(r["effort"] for r in recs)
    L += [f"抽样 {n} 源: " + ", ".join(f"`{k}` {v}" for k, v in sorted(eff_cnt.items())), ""]

    # ---------------- summary tables ---------------- #
    L += ["## 汇总表", "", "### 1. intent_mode 分布", ""]
    im_all = Counter(r["diag"].get("intent_mode") for r in recs)
    modes = ["correction_led", "enhancement_led", "mixed"]
    modes += [m for m in im_all if m not in modes]
    L += ["| intent_mode | 全部 | high | low |", "| --- | --- | --- | --- |"]
    for m in modes:
        hi = sum(1 for r in recs if r["diag"].get("intent_mode") == m and r["effort"] == "high")
        lo = sum(1 for r in recs if r["diag"].get("intent_mode") == m and r["effort"] == "low")
        L.append(f"| {m} | {im_all.get(m, 0)} | {hi} | {lo} |")
    L += [f"| 合计 | {n} | {eff_cnt.get('high', 0)} | {eff_cnt.get('low', 0)} |", ""]

    for title, key in (
        ("### 2. correction_needs 条数分布", "n_corr"),
        ("### 3. enhancement_opportunities(机会)条数分布", "n_opp"),
    ):
        L += [title, ""]
        c = Counter(r[key] for r in recs)
        L += ["| 条数 | 全部 | high | low |", "| --- | --- | --- | --- |"]
        for k in sorted(c):
            hi = sum(1 for r in recs if r[key] == k and r["effort"] == "high")
            lo = sum(1 for r in recs if r[key] == k and r["effort"] == "low")
            L.append(f"| {k} | {c[k]} | {hi} | {lo} |")
        tot = sum(r[key] for r in recs)
        th = sum(r[key] for r in recs if r["effort"] == "high")
        tl = sum(r[key] for r in recs if r["effort"] == "low")
        L += [f"| 总条数 | {tot} | {th} | {tl} |"]
        mh = th / max(1, eff_cnt.get("high", 0))
        ml = tl / max(1, eff_cnt.get("low", 0))
        L += [f"| 每源均值 | {tot / n:.2f} | {mh:.2f} | {ml:.2f} |", ""]

    # panel citations
    L += ["### 4. 引用 panel 号的条目数(按五字段计条)", ""]
    L += ["| 字段 | 全部条目 | 其中引用 panel | high 引用 | low 引用 |", "| --- | --- | --- | --- | --- |"]
    for f in FIVE_FIELDS:
        tot_items = sum(len(r["diag"].get(f) or []) for r in recs)
        cit = sum(r["panel_items"][f] for r in recs)
        ch = sum(r["panel_items"][f] for r in recs if r["effort"] == "high")
        cl = sum(r["panel_items"][f] for r in recs if r["effort"] == "low")
        L.append(f"| {f} | {tot_items} | {cit} | {ch} | {cl} |")
    tot_items = sum(len(r["diag"].get(f) or []) for r in recs for f in FIVE_FIELDS)
    cit = sum(sum(r["panel_items"].values()) for r in recs)
    ch = sum(sum(r["panel_items"].values()) for r in recs if r["effort"] == "high")
    cl = sum(sum(r["panel_items"].values()) for r in recs if r["effort"] == "low")
    L += [f"| 合计 | {tot_items} | {cit} | {ch} | {cl} |", ""]
    ph = ch / max(1, eff_cnt.get("high", 0))
    pl = cl / max(1, eff_cnt.get("low", 0))
    L += [f"每源引用 panel 的条目数均值 - 全部 {cit / n:.2f} | high {ph:.2f} | low {pl:.2f}", ""]

    # per-source table
    L += ["### 5. 每源明细", ""]
    L += [
        "| # | source_id | scene | effort | in/out tokens | intent_mode | conf | corr | opp | 机会组件行 | 引用 panel 条目 | 不合规行 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, r in enumerate(recs, 1):
        n_bad = sum(1 for c in r["comps"] if not c[3])
        n_opp_comp = sum(1 for c in r["comps"] if c[0] == "enhancement_opportunities")
        L.append(
            f"| {i} | `{r['sid']}` | {r['scene']} | {r['effort']} | {r['in_tok']}/{r['out_tok']} | "
            f"{r['diag'].get('intent_mode')} | {r['diag'].get('confidence')} | {r['n_corr']} | "
            f"{r['n_opp']} | {n_opp_comp} | {sum(r['panel_items'].values())} | {n_bad} |"
        )
    L.append("")

    # token usage by effort
    L += ["### 6. token 用量(按口径)", "", "| 口径 | 源数 | in 均值 | out 均值 | out 最小 | out 最大 |", "| --- | --- | --- | --- | --- | --- |"]
    for k in ("high", "low"):
        sel = [r for r in recs if r["effort"] == k]
        if not sel:
            continue
        ins = [r["in_tok"] or 0 for r in sel]
        outs = [r["out_tok"] or 0 for r in sel]
        L.append(
            f"| {k} | {len(sel)} | {sum(ins) / len(sel):.0f} | {sum(outs) / len(sel):.0f} | "
            f"{min(outs)} | {max(outs)} |"
        )
    L.append("")

    # compliance
    all_comps = [c for r in recs for c in r["comps"]]
    bad = [(r, c) for r in recs for c in r["comps"] if not c[3]]
    L += [
        "### 7. 固定句式合规",
        "",
        "解析口径: `correction_needs` 每行本身即一个组件行;`enhancement_opportunities` 每行按 "
        "`风格简述 => 组件行1 ; 组件行2 …` 先切 `=>` 再按 `;` 切成组件行。每个组件行按 `|` 切四段,"
        "要求 axis ∈ {exposure, band_lightness, contrast, hue, cast, saturation}、"
        "scope ∈ {global, shadows, midtones, highlights, subject, skin, sky, foliage, background}、"
        "state 与 move 非空。",
        "",
        "| 来源字段 | 组件行数 | 合规 | 不合规 |",
        "| --- | --- | --- | --- |",
    ]
    for f in ("correction_needs", "enhancement_opportunities"):
        sel = [c for c in all_comps if c[0] == f]
        nb = sum(1 for c in sel if not c[3])
        L.append(f"| {f} | {len(sel)} | {len(sel) - nb} | {nb} |")
    nb_all = sum(1 for c in all_comps if not c[3])
    L += [f"| 合计 | {len(all_comps)} | {len(all_comps) - nb_all} | {nb_all} |", ""]

    if bad:
        L += ["不合规样例:", "", "| source_id | 字段 | 组件行 | 判据 |", "| --- | --- | --- | --- |"]
        for r, c in bad[:40]:
            L.append(f"| `{r['sid']}` | {c[0]} | `{md_escape(c[2])}` | {md_escape(c[4])} |")
        L.append("")
    else:
        L += ["不合规行数: 0(无样例)。", ""]

    # axis / scope tallies
    axis_c = Counter(c[2].split("|")[0].strip() for c in all_comps if c[3])
    scope_c = Counter(c[2].split("|")[1].strip() for c in all_comps if c[3])
    L += ["### 8. 合规组件行的 axis / scope 计数", "", "| axis | 计数 | scope | 计数 |", "| --- | --- | --- | --- |"]
    ax = sorted(axis_c.items(), key=lambda kv: -kv[1])
    sc = sorted(scope_c.items(), key=lambda kv: -kv[1])
    for i in range(max(len(ax), len(sc))):
        a = f"{ax[i][0]} | {ax[i][1]}" if i < len(ax) else " | "
        s = f"{sc[i][0]} | {sc[i][1]}" if i < len(sc) else " | "
        L.append(f"| {a} | {s} |")
    L.append("")

    # missing images
    miss = [r["sid"] for r in recs if not r["src_rel"] or not r["board_rel"]]
    L += [f"图片缺失源: {miss if miss else '无'}", ""]

    # ---------------- per-source sections ---------------- #
    L += ["---", "", "## 逐源", ""]
    for i, r in enumerate(recs, 1):
        d = r["diag"]
        L += [f"### {i}. `{r['sid']}` ({r['scene']}, effort={r['effort']})", ""]
        imgs = []
        if r["src_rel"]:
            imgs.append(f'<img src="{r["src_rel"]}" width="340">')
        if r["board_rel"]:
            imgs.append(f'<img src="{r["board_rel"]}" width="420">')
        L += [" ".join(imgs), ""]
        L += [
            f"- subject: {r['subject']} (mask_area={r['mask_area']})",
            f"- intent_mode: `{d.get('intent_mode')}` | confidence: `{d.get('confidence')}` | "
            f"reasoning_effort: `{r['effort']}`",
            f"- tokens in/out: {r['in_tok']} / {r['out_tok']} | "
            f"引用 panel 的条目数: {sum(r['panel_items'].values())}",
            "",
        ]
        L += ["**correction_needs**", ""]
        L += bullet_block(list(d.get("correction_needs") or []))
        L += ["**enhancement_opportunities**", ""]
        opps = list(d.get("enhancement_opportunities") or [])
        if not opps:
            L += ["_(empty)_", ""]
        else:
            for j, line in enumerate(opps, 1):
                sp = style_phrase(line)
                L.append(f"{j}. `{md_escape(sp)}` =>")
                for comp in split_components(line):
                    ok, reason = check_component(comp)
                    flag = "" if ok else f"  **[不合规: {reason}]**"
                    L.append(f"   - `{md_escape(comp)}`{flag}")
            L.append("")
        for f in ("evidence", "forbidden_directions", "preserve_intent"):
            L += [f"**{f}**", ""]
            L += bullet_block(list(d.get(f) or []))
        L += ["", "---", ""]

    L += ["## 抽样的 30 个 source_id", "", "```"]
    L += [r["sid"] for r in recs]
    L += ["```", ""]
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/home/bc/data/agent_loop/local-v1/sources5k.annotated-v34.jsonl")
    ap.add_argument("--annotations-dir", default="/home/bc/data/agent_loop/local-v1/annotations_v34")
    ap.add_argument("--blob-store", default="/mnt/ramstage/agent_loop/local-v2-pilot/blobs")
    ap.add_argument("--out", default="docs/assets/diagnose_v34_sample30_20260824")
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--preview-px", type=int, default=512)
    ap.add_argument("--no-full-tally", dest="full_tally", action="store_false")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
