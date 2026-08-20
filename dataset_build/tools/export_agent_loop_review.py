"""Static HTML review of one agent-loop campaign, laid out as sidebar + detail pane.

Reads the PostgreSQL audit tables (``agent_source_run`` / ``agent_branch`` /
``proposal_audit`` / ``render_record`` / ``api_request*``) plus the content-addressed
artifacts written by the run, and emits a self-contained directory with one page and
one thumbnail per referenced image.  No legacy review batch (eval100 / scene samples)
is read.

The page has a fixed left source list (thumbnail, id, terminal status, leaf counts,
all/accepted filter) and a right detail pane per source.  Each source -> global -> local
chain is shown as one wide ``source | global_after | final_after`` triple captioned with
strength and ΔE only; every other parameter lives in a collapsed ``details`` block.

Usage:
    python -m dataset_build.tools.export_agent_loop_review \
        [--campaign local-v1-iter1] [--out-dir docs/assets/agent_loop_review_<campaign>] \
        [--limit N]
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image

from dataset_build.agent_loop.artifacts import ArtifactStore


DEFAULT_ROOT_PARENTS = (
    Path("/mnt/ramstage/agent_loop"), Path("/home/bc/data/agent_loop"),
)
DEFAULT_CATALOG_DB = Path("/var/cache/veradata/global.sqlite3")
FINGERPRINT_KEYS = (
    "dL", "dSat", "contrast", "cast_hue", "cast_mag", "hue_rot",
    "highlight_dL", "shadow_dL",
)
GLOBAL_BIN_ORDER = ("natural", "medium", "bold")
LOCAL_BIN_ORDER = ("subtle", "moderate", "strong")


# --------------------------------------------------------------------------- helpers


# C1b items 3-5. Shared by every questionnaire builder that samples committed leaves.
# `winner_confidence` is written on the committed leaf by the agent loop: "normal" when
# the chain validator passed, "low" otherwise. Campaign discipline keeps `low` out of
# main training and out of evaluation GT, so a questionnaire sample that does not carry
# the field cannot be audited against that rule after the fact.
WINNER_CONFIDENCE_UNKNOWN = "unknown"


def winner_confidence_counts(values: Iterable[Any]) -> dict[str, int]:
    """Per-level counts of a sampled population's `winner_confidence`."""
    counted: Counter[str] = Counter()
    for value in values:
        text = str(value).strip() if value not in (None, "") else ""
        counted[text or WINNER_CONFIDENCE_UNKNOWN] += 1
    return dict(sorted(counted.items()))


def winner_confidence_warning(counts: Mapping[str, int], *, filtered: bool) -> str:
    """One warning line for the top of a builder's stats block, or "" when clean.

    The builders deliberately do NOT drop `low` rows (the current campaign is entirely
    `low`, so a filter returns the empty set), which makes the printed warning the only
    thing standing between the operator and a silently `low`-only questionnaire.
    """
    low = int(counts.get("low", 0)) + int(counts.get(WINNER_CONFIDENCE_UNKNOWN, 0))
    total = sum(int(value) for value in counts.values())
    if low == 0:
        return ""
    return (
        f"WARNING winner_confidence: {low}/{total} sampled rows are low or unknown "
        f"(counts={json.dumps(dict(counts), sort_keys=True)}); "
        f"winner_confidence filtering is {'ON' if filtered else 'OFF'} in this builder, "
        "and campaign discipline keeps winner_confidence=low out of main training and "
        "out of evaluation GT."
    )


def print_winner_confidence_warning(counts: Mapping[str, int], *, filtered: bool) -> str:
    line = winner_confidence_warning(counts, filtered=filtered)
    if line:
        print(line, file=sys.stderr)
    return line


def _loads(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    return json.loads(text) if text else None


def _number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int,)):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _join(values: Iterable[Any]) -> str:
    items = [str(item) for item in values if item not in (None, "")]
    return ", ".join(items) if items else "-"


# ------------------------------------------------------------------- artifact access


class ArtifactResolver:
    """Try several artifact roots, then the landed-tar global catalog."""

    def __init__(self, roots: Sequence[Path], catalog_db: Path | None) -> None:
        self.stores: list[ArtifactStore] = []
        for root in roots:
            self.stores.append(ArtifactStore(root, catalog_db=catalog_db))
        if not self.stores:
            raise SystemExit("no artifact root found; pass --artifact-root")
        self.missing: Counter[str] = Counter()

    def path_for(self, ref: Mapping[str, Any] | str) -> Path:
        last: Exception | None = None
        for store in self.stores:
            try:
                return store.path_for(ref)
            except (FileNotFoundError, ValueError) as exc:  # noqa: PERF203
                last = exc
        raise FileNotFoundError(str(last))

    def read_json(self, ref: Mapping[str, Any] | str) -> Any:
        return json.loads(self.path_for(ref).read_bytes())


def _default_roots(explicit: Sequence[Path]) -> list[Path]:
    if explicit:
        return [Path(item).expanduser() for item in explicit]
    roots: list[Path] = []
    for parent in DEFAULT_ROOT_PARENTS:
        if not parent.is_dir():
            continue
        roots.extend(sorted(
            child for child in parent.iterdir() if (child / "blobs").is_dir()
        ))
    return roots


class Thumbnails:
    """Deterministic short-edge thumbnails, one file per referenced blob."""

    def __init__(self, resolver: ArtifactResolver, out_dir: Path, short_edge: int) -> None:
        self.resolver = resolver
        self.dir = out_dir / "img"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.short_edge = int(short_edge)
        self.written: dict[str, str] = {}
        self.missing: set[str] = set()
        self.failed: dict[str, str] = {}

    def rel(self, ref: Mapping[str, Any] | None) -> str | None:
        """Relative path of the thumbnail, or None when the blob is unavailable."""
        if not isinstance(ref, Mapping) or not ref.get("sha256"):
            return None
        digest = str(ref["sha256"])
        if digest in self.written:
            return self.written[digest]
        if digest in self.missing or digest in self.failed:
            return None
        try:
            source = self.resolver.path_for(ref)
        except FileNotFoundError:
            self.missing.add(digest)
            return None
        media = str(ref.get("media_type") or "")
        suffix = ".png" if media == "image/png" else ".jpg"
        name = f"{digest[:16]}-{self.short_edge}{suffix}"
        target = self.dir / name
        if not target.is_file():
            try:
                self._render(source, target, suffix)
            except Exception as exc:  # pragma: no cover - malformed blob
                self.failed[digest] = f"{type(exc).__name__}: {exc}"
                return None
        self.written[digest] = f"img/{name}"
        return self.written[digest]

    def _render(self, source: Path, target: Path, suffix: str) -> None:
        with Image.open(source) as handle:
            image = handle.copy()
        if min(image.size) > self.short_edge:
            scale = self.short_edge / min(image.size)
            size = tuple(max(1, round(value * scale)) for value in image.size)
            image = image.resize(size, Image.Resampling.LANCZOS)
        temporary = target.with_name(target.name + ".tmp")
        if suffix == ".png":
            image.convert("L").save(temporary, "PNG", compress_level=6)
        else:
            image.convert("RGB").save(
                temporary, "JPEG", quality=88, optimize=False, progressive=False,
                subsampling=0,
            )
        temporary.replace(target)

    def prune(self) -> int:
        """Drop files nothing on the page references (stale sizes, retired blobs)."""
        keep = {Path(value).name for value in self.written.values()}
        removed = 0
        for path in sorted(self.dir.iterdir()):
            if path.is_file() and path.name not in keep:
                path.unlink()
                removed += 1
        return removed


# ---------------------------------------------------------------------- audit access


def _connect(dsn: str):
    import psycopg

    return psycopg.connect(dsn)


def _rows(conn, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        names = [column.name for column in cur.description]
        return [dict(zip(names, row)) for row in cur.fetchall()]


def _latest_campaign(conn) -> str:
    rows = _rows(
        conn,
        "SELECT campaign_id, max(started_at) AS latest FROM agent_source_run "
        "GROUP BY campaign_id ORDER BY latest DESC, campaign_id LIMIT 1",
    )
    if not rows:
        raise SystemExit("no campaign found in agent_source_run")
    return str(rows[0]["campaign_id"])


def _collect(conn, campaign: str, limit: int) -> dict[str, Any]:
    sources = _rows(
        conn,
        "SELECT source_id, source_sha256, status, started_at, updated_at, counts_json, "
        "manifest_json FROM agent_source_run WHERE campaign_id=%s "
        "ORDER BY started_at, source_id",
        (campaign,),
    )
    status_all = Counter(str(row["status"]) for row in sources)
    if limit > 0:
        sources = sources[:limit]
    keys = [row["source_sha256"] for row in sources]
    branches = _rows(
        conn,
        "SELECT branch_id, source_sha256, parent_id, level, status, proposal_json, "
        "result_json, created_at FROM agent_branch WHERE campaign_id=%s "
        "AND source_sha256 = ANY(%s) ORDER BY created_at, branch_id",
        (campaign, keys),
    ) if keys else []
    audits = _rows(
        conn,
        "SELECT branch_id, level, preset_id, scorer_top1, scorer_top3, scorer_top1_raw, "
        "direction_cosine FROM proposal_audit WHERE campaign_id=%s "
        "AND source_sha256 = ANY(%s)",
        (campaign, keys),
    ) if keys else []
    renders = _rows(
        conn,
        "SELECT render_hash, branch_id, stage, status, parameters_json, metrics_json, "
        "created_at FROM render_record WHERE source_sha256 = ANY(%s) "
        "ORDER BY created_at, render_hash",
        (keys,),
    ) if keys else []
    usage = _rows(
        conn,
        "SELECT c.stage AS stage, count(*) AS requests, sum(c.cache_hit) AS cache_hits, "
        "sum(COALESCE((r.usage_json::jsonb->>'input_tokens')::bigint, 0)) AS input_tokens, "
        "sum(COALESCE((r.usage_json::jsonb->>'cached_tokens')::bigint, 0)) AS cached_tokens, "
        "sum(COALESCE((r.usage_json::jsonb->>'output_tokens')::bigint, 0)) AS output_tokens "
        "FROM api_request_context c JOIN api_request r ON r.request_hash=c.request_hash "
        "WHERE c.campaign_id=%s GROUP BY c.stage ORDER BY c.stage",
        (campaign,),
    )
    return {
        "sources": sources, "status_all": status_all, "branches": branches,
        "audits": audits, "renders": renders, "usage": usage,
    }


# ------------------------------------------------------------------------- rendering


def _kv(pairs: Sequence[tuple[str, Any]]) -> str:
    cells = "".join(
        f"<div class='k'>{_esc(key)}</div><div class='v'>{value}</div>"
        for key, value in pairs
    )
    return f"<div class='kv'>{cells}</div>"


def _figure(thumbs: Thumbnails, ref: Mapping[str, Any] | None, caption: str) -> str:
    if not isinstance(ref, Mapping) or not ref.get("sha256"):
        return (
            f"<figure><div class='absent'>not rendered</div>"
            f"<figcaption>{_esc(caption)}</figcaption></figure>"
        )
    rel = thumbs.rel(ref)
    digest = str(ref.get("sha256"))[:12]
    if rel is None:
        body = f"<div class='missing'>blob missing<br><code>{_esc(digest)}</code></div>"
    else:
        body = f"<img src='{_esc(rel)}' alt='{_esc(caption)}' loading='lazy'>"
    return f"<figure>{body}<figcaption>{_esc(caption)}</figcaption></figure>"


def _gate(render_row: Mapping[str, Any] | None) -> str:
    if render_row is None:
        return "-"
    metrics = _loads(render_row.get("metrics_json")) or {}
    target = ""
    if "target_low" in metrics:
        target = (f" · target [{_number(metrics.get('target_low'))},"
                  f"{_number(metrics.get('target_high'))}]")
    return f"{render_row.get('status')}{target}"


# --------------------------------------------------------------- chain (three-up view)


def _chain_block(
    source_ref: Mapping[str, Any] | None, branch: Mapping[str, Any],
    leaf: Mapping[str, Any] | None, index: int, thumbs: Thumbnails,
) -> str:
    """One source -> global_after -> final_after row, captioned with strength/ΔE only."""
    result = branch["result"]
    g_proposal = result.get("proposal") or branch.get("proposal") or {}
    g_render = result.get("global_render") or {}
    g_parameters = g_render.get("parameters") or {}
    g_metrics = g_render.get("metrics") or {}
    has_leaf = leaf is not None
    leaf = leaf or {}
    l_proposal = leaf.get("proposal") or {}
    l_render = leaf.get("render") or {}
    l_parameters = l_render.get("parameters") or {}
    l_metrics = l_render.get("metrics") or {}
    commit = str(leaf.get("commit_status") or "")

    g_bin = g_parameters.get("strength_bin") or g_proposal.get("strength_bin")
    l_bin = l_parameters.get("strength_bin") or l_proposal.get("strength_bin")
    global_line = (
        f"<b>global</b> <span class='bin'>{_esc(g_bin)}</span>"
        f" · strength <b>{_number(g_parameters.get('global_strength'), 3)}</b>"
        f" · ΔE <b>{_number(g_metrics.get('delta_e'), 2)}</b>"
    )
    if has_leaf:
        local_line = (
            f"<b>local</b> <span class='bin'>{_esc(l_bin)}</span>"
            f" · strength <b>{_number(l_parameters.get('local_strength'), 3)}</b>"
            f" · mask-weighted ΔE <b>{_number(l_metrics.get('delta_e'), 2)}</b>"
        )
    else:
        local_line = "<b>local</b> <span class='dimtext'>no leaf</span>"

    badge_class = "ok" if commit == "committed" else "no"
    label = commit or ("not committed" if has_leaf else "no leaf")
    badges = [f"<span class='badge {badge_class}'>{_esc(label)}</span>"]
    if has_leaf and str(leaf.get("status") or "") not in ("", "validator_skipped"):
        badges.append(f"<span class='badge'>{_esc(leaf.get('status'))}</span>")
    if int(leaf.get("repair_count") or 0):
        badges.append(f"<span class='badge'>repair={_esc(leaf.get('repair_count'))}</span>")
    reject = leaf.get("reject_reason") or result.get("reject_reason")
    if reject:
        badges.append(f"<span class='badge warn'>{_esc(reject)}</span>")

    figures = (
        _figure(thumbs, source_ref, "source")
        + _figure(thumbs, g_render.get("artifact"), "global_after")
        + _figure(thumbs, l_render.get("artifact"), "final_after")
    )
    return (
        f"<div class='chain'><div class='chdr'><span class='cidx'>#{index}</span>"
        + "".join(badges)
        + f"<code class='dimtext'>{_esc(str(branch['branch_id'])[-8:])}"
        + (f" / {_esc(str(leaf.get('branch_id'))[-8:])}" if has_leaf else "")
        + "</code></div>"
        f"<div class='trip'>{figures}</div>"
        f"<div class='lines'><div>{global_line}</div><div>{local_line}</div></div>"
        "</div>"
    )


# ----------------------------------------------------------- collapsed parameter dump


def _param_row(
    index: int, branch: Mapping[str, Any], leaf: Mapping[str, Any] | None,
    masks: Mapping[str, Mapping[str, Any]], audits: Mapping[str, Mapping[str, Any]],
    renders: Mapping[str, Mapping[str, Any]],
) -> str:
    result = branch["result"]
    g_proposal = result.get("proposal") or branch.get("proposal") or {}
    g_render = result.get("global_render") or {}
    g_parameters = g_render.get("parameters") or {}
    g_metrics = g_render.get("metrics") or {}
    g_audit = audits.get(str(branch["branch_id"])) or {}
    leaf = leaf or {}
    l_proposal = leaf.get("proposal") or {}
    l_render = leaf.get("render") or {}
    l_parameters = l_render.get("parameters") or {}
    l_metrics = l_render.get("metrics") or {}
    l_audit = audits.get(str(leaf.get("branch_id"))) or {}
    validation = leaf.get("validation") or {}
    mask = masks.get(str(l_proposal.get("mask_id"))) or {}
    cells = [
        f"#{index}",
        f"<code>{_esc(str(branch['branch_id'])[-8:])}</code>",
        _esc(branch.get("status")),
        f"<code>{_esc(g_proposal.get('preset_id'))}</code>",
        _esc(g_proposal.get("row_index")),
        _esc(g_parameters.get("strength_bin") or g_proposal.get("strength_bin")),
        _number(g_parameters.get("global_strength"), 5),
        _number(g_metrics.get("delta_e"), 4),
        _number(g_metrics.get("clip_fraction_new"), 5),
        _esc(_gate(renders.get(str(branch["branch_id"])))),
        _esc(_join(g_proposal.get("reason_codes") or [])),
        _esc(f"{g_audit.get('scorer_top1')} / {g_audit.get('scorer_top3')} / "
             f"{g_audit.get('scorer_top1_raw')}") if g_audit else "-",
        _esc(json.dumps(result.get("local_shortlist_deficits") or [], ensure_ascii=False)),
        f"<code>{_esc(str(leaf.get('branch_id') or '-')[-8:])}</code>",
        _esc(leaf.get("status") or "-"),
        _esc(leaf.get("commit_status") or "-"),
        f"<code>{_esc(l_proposal.get('preset_id'))}</code>",
        _esc(l_proposal.get("row_index")),
        _esc(l_parameters.get("strength_bin") or l_proposal.get("strength_bin")),
        _number(l_parameters.get("local_strength"), 5),
        _number(l_metrics.get("delta_e"), 4),
        _number(l_metrics.get("clip_fraction_new"), 5),
        _esc(_gate(renders.get(str(leaf.get("branch_id"))))),
        _esc(_join(l_proposal.get("reason_codes") or [])),
        _esc(f"{l_proposal.get('scorer_rank_offered')} / {l_proposal.get('scorer_rank_raw')}"),
        _number(l_audit.get("direction_cosine"), 4) if l_audit else "-",
        f"<code>{_esc(l_proposal.get('mask_id'))}</code>",
        _esc(mask.get("family") or "-"),
        _esc(mask.get("direction") or "-"),
        _esc(mask.get("center_hint") or "-"),
        _number(mask.get("half_area"), 4),
        _number(mask.get("support_area"), 4),
        _number(mask.get("subject_area"), 4),
        _number(l_metrics.get("applied_alpha_effective_alpha_mean"), 4),
        _number(l_metrics.get("applied_alpha_subject_high_coverage"), 4),
        _number(l_metrics.get("applied_alpha_subject_support_coverage"), 4),
        _esc(leaf.get("visible_region") or l_proposal.get("visible_region") or "-"),
        _esc(leaf.get("repair_count")),
        _esc(leaf.get("reject_reason") or result.get("reject_reason") or "-"),
        _esc(f"mode={validation.get('mode', '-')} passed={validation.get('passed')} "
             f"skipped={validation.get('skipped', False)} "
             f"defects={len(validation.get('defects') or [])}") if validation else "-",
        _esc(leaf.get("winner_confidence") or "-"),
    ]
    return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"


PARAM_COLUMNS = (
    "chain", "global_branch", "global_status", "g preset_id", "g row_index", "g bin",
    "g strength", "g ΔE", "g clip_new", "g render gate", "g reason_codes",
    "g scorer top1/top3/raw", "local_shortlist_deficits",
    "leaf_branch", "leaf_status", "commit_status", "l preset_id", "l row_index", "l bin",
    "l local_strength", "l mask-weighted ΔE", "l clip_new", "l render gate",
    "l reason_codes", "l scorer offered/raw", "direction_cosine", "mask_id",
    "mask family", "mask direction", "center_hint", "half_area", "support_area",
    "subject_area", "applied_alpha mean", "subject_high_cov", "subject_support_cov",
    "visible_region", "repair", "reject_reason", "validator", "winner_confidence",
)


def _shortlist_block(shortlist: Mapping[str, Any] | None) -> str:
    if not shortlist:
        return "<p class='warn'>global shortlist artifact unavailable</p>"
    by_major = shortlist.get("by_major") or {}
    deficits = list(shortlist.get("quota_deficits") or [])
    corrections = [row for row in deficits
                   if str(row.get("reason", "")).startswith("correction_")]
    quotas = [row for row in deficits if row not in corrections]
    parts = [
        _kv([
            ("offered_majors", _esc(_join(shortlist.get("offered_majors") or []))),
            ("shortlist_rows", sum(len(rows) for rows in by_major.values())),
            ("quota_deficits", len(quotas)),
            ("correction_filter", _esc(_join(
                f"{row.get('reason')}(before={row.get('majors_before')},"
                f"after={row.get('majors_after')},"
                f"dropped={_join(row.get('dropped_majors') or [])})"
                for row in corrections
            )) if corrections else "none"),
        ])
    ]
    head = "".join(f"<th>{_esc(key)}</th>" for key in FINGERPRINT_KEYS)
    for major in sorted(by_major):
        rows = by_major[major]
        body = []
        for row in rows:
            fingerprint = row.get("fingerprint") or {}
            cells = "".join(
                f"<td class='num'>{_number(fingerprint.get(key))}</td>"
                for key in FINGERPRINT_KEYS
            )
            body.append(
                f"<tr><td class='num'>"
                f"{_esc(row.get('row_index', row.get('scorer_rank_offered')))}</td>"
                f"<td><code>{_esc(row.get('preset_id'))}</code></td>"
                f"<td>{_esc(_join(row.get('achievable_bins') or []))}</td>"
                f"{cells}"
                f"<td class='cap'>{_esc(row.get('caption'))}</td></tr>"
            )
        parts.append(
            f"<h5>major <code>{_esc(major)}</code> &middot; {len(rows)} rows</h5>"
            f"<div class='scroll'><table class='grid'><thead><tr><th>row</th>"
            f"<th>preset_id</th><th>achievable_bins</th>{head}<th>caption</th></tr>"
            f"</thead><tbody>{''.join(body)}</tbody></table></div>"
        )
    if quotas:
        parts.append(
            "<h5>quota deficits</h5><pre>"
            + _esc(json.dumps(quotas, ensure_ascii=False, sort_keys=True, indent=1))
            + "</pre>"
        )
    return "".join(parts)


# ------------------------------------------------------------------------ source pane


def _sort_leaves(leaves: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(leaves, key=lambda leaf: (
        int(leaf.get("repair_count") or 0),
        int((leaf.get("proposal") or {}).get("row_index") or 0),
        str(leaf.get("branch_id")),
    ))


def _source_pane(
    source: Mapping[str, Any], tree: Mapping[str, Any] | None,
    annotation: Mapping[str, Any] | None, shortlist: Mapping[str, Any] | None,
    branches: Sequence[Mapping[str, Any]], audits: Mapping[str, Mapping[str, Any]],
    renders: Mapping[str, Mapping[str, Any]], thumbs: Thumbnails,
) -> tuple[str, str]:
    """Return (pane html, sidebar item html) for one source."""
    tree = tree or {}
    diagnosis = tree.get("diagnosis") or {}
    counts = _loads(source.get("counts_json")) or {}
    masks = {str(row["mask_id"]): row for row in (tree.get("mask_bank") or [])}
    source_ref = tree.get("source_artifact")
    source_id = str(source["source_id"])
    pane_id = "pane-" + source_id
    status = str(source["status"])

    globals_ = [row for row in branches if row["level"] == "global"]
    locals_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in branches:
        if row["level"] == "local":
            locals_by_parent[str(row.get("parent_id"))].append(row["result"])

    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any] | None]] = []
    for branch in globals_:
        leaves = _sort_leaves(locals_by_parent.get(str(branch["branch_id"]), []))
        if leaves:
            pairs.extend((branch, leaf) for leaf in leaves)
        else:
            pairs.append((branch, None))
    committed = [pair for pair in pairs
                 if str((pair[1] or {}).get("commit_status")) == "committed"]
    others = [pair for pair in pairs
              if str((pair[1] or {}).get("commit_status")) != "committed"]
    ordered = [(index, branch, leaf) for index, (branch, leaf)
               in enumerate(committed + others, start=1)]

    committed_html = "".join(
        _chain_block(source_ref, branch, leaf, index, thumbs)
        for index, branch, leaf in ordered[:len(committed)]
    ) or "<p class='dimtext'>no committed chain</p>"
    others_html = "".join(
        _chain_block(source_ref, branch, leaf, index, thumbs)
        for index, branch, leaf in ordered[len(committed):]
    )

    needs = diagnosis.get("correction_needs") or []
    headline = " · ".join(str(item) for item in (
        diagnosis.get("intent_mode"), tree.get("style_major"),
        (annotation or {}).get("scene"),
    ) if item)
    summary_line = str(needs[0]) if needs else ""

    def _bullets(key: str) -> str:
        items = diagnosis.get(key) or []
        if not items:
            return "<em>none</em>"
        return "<ul>" + "".join(f"<li>{_esc(item)}</li>" for item in items) + "</ul>"

    param_rows = "".join(
        _param_row(index, branch, leaf, masks, audits, renders)
        for index, branch, leaf in ordered
    ) or f"<tr><td colspan='{len(PARAM_COLUMNS)}'><em>none</em></td></tr>"
    head = "".join(f"<th>{_esc(name)}</th>" for name in PARAM_COLUMNS)

    details = (
        "<details class='dump'><summary>all parameters (shortlist / reason codes / "
        "fingerprints / direction cosine / masks / gates)</summary>"
        "<h4>source</h4>"
        + _kv([
            ("source_sha256", f"<code>{_esc(source['source_sha256'])[:24]}</code>"),
            ("scene", _esc((annotation or {}).get("scene"))),
            ("subject",
             _esc(((annotation or {}).get("subject") or {}).get("description"))),
            ("intent_mode", _esc(diagnosis.get("intent_mode"))),
            ("diagnosis confidence", _number(diagnosis.get("confidence"), 3)),
            ("style_major", _esc(tree.get("style_major"))),
            ("terminal status", _esc(status)),
            ("reject_reasons", _esc(_join(
                counts.get("reject_reasons") or tree.get("reject_reasons") or []))),
            ("counts", _esc(json.dumps(counts, ensure_ascii=False, sort_keys=True))),
            ("committed_leaf_ids", _esc(_join(tree.get("committed_leaf_ids") or []))),
        ])
        + "<h4>diagnosis</h4><div class='row'>"
        + f"<div class='half'><b>correction_needs</b>{_bullets('correction_needs')}</div>"
        + "<div class='half'><b>enhancement_opportunities</b>"
        + f"{_bullets('enhancement_opportunities')}</div></div>"
        + "<h4>per-chain parameters</h4>"
        + f"<div class='scroll'><table class='grid'><thead><tr>{head}</tr></thead>"
        + f"<tbody>{param_rows}</tbody></table></div>"
        + "<h4>shortlist actually sent to Terra</h4>"
        + _shortlist_block(shortlist)
        + "</details>"
    )

    pane = (
        f"<section class='pane' id='{_esc(pane_id)}' hidden>"
        f"<div class='panehdr'><h2>{_esc(source_id)}</h2>"
        f"<span class='badge big {_esc(status)}'>{_esc(status)}</span>"
        f"<span class='hl'>{_esc(headline)}</span>"
        f"<span class='badge'>chains={len(pairs)}</span>"
        f"<span class='badge'>committed={len(committed)}</span></div>"
        + (f"<p class='diag'>{_esc(summary_line)}</p>" if summary_line else "")
        + f"<div class='chains'>{committed_html}</div>"
        + (f"<details class='rest'><summary>other chains "
           f"({len(others)}: cap_excluded / rejected)</summary>"
           f"<div class='chains'>{others_html}</div></details>" if others else "")
        + details
        + "</section>"
    )

    thumb = thumbs.rel(source_ref) if isinstance(source_ref, Mapping) else None
    picture = (f"<img src='{_esc(thumb)}' alt='' loading='lazy'>" if thumb
               else "<span class='nothumb'></span>")
    item = (
        f"<li data-target='{_esc(pane_id)}' data-status='{_esc(status)}'>"
        f"{picture}<span class='si'><span class='sid'>{_esc(source_id)}</span>"
        f"<span class='meta'><span class='badge {_esc(status)}'>{_esc(status)}</span>"
        f"<span class='n'>{len(committed)}/{len(pairs)} leaves</span></span></span></li>"
    )
    return pane, item


def _summary_pane(
    campaign: str, data: Mapping[str, Any], exported: Sequence[Mapping[str, Any]],
    branch_index: Mapping[str, list[dict[str, Any]]], thumbs: Thumbnails,
) -> str:
    status_all = data["status_all"]
    leaf_status: Counter[str] = Counter()
    leaf_commit: Counter[str] = Counter()
    global_status: Counter[str] = Counter()
    global_bins: Counter[str] = Counter()
    local_bins: Counter[str] = Counter()
    committed_global_bins: Counter[str] = Counter()
    reasons_global: Counter[str] = Counter()
    reasons_local: Counter[str] = Counter()
    for source in exported:
        for row in branch_index.get(str(source["source_sha256"]), []):
            result = row["result"]
            proposal = result.get("proposal") or row.get("proposal") or {}
            if row["level"] == "global":
                global_status[str(row["status"])] += 1
                global_bins[str(proposal.get("strength_bin"))] += 1
                reasons_global.update(str(code) for code in proposal.get("reason_codes") or [])
            else:
                leaf_status[str(row["status"])] += 1
                leaf_commit[str(result.get("commit_status") or "none")] += 1
                local_bins[str(proposal.get("strength_bin"))] += 1
                reasons_local.update(str(code) for code in proposal.get("reason_codes") or [])
                if str(result.get("commit_status")) == "committed":
                    committed_global_bins[str(result.get("global_strength_bin"))] += 1

    def _counter_table(title: str, counter: Counter[str], order: Sequence[str] = ()) -> str:
        keys = [key for key in order if key in counter] + sorted(
            key for key in counter if key not in set(order)
        )
        rows = "".join(
            f"<tr><td>{_esc(key)}</td><td class='num'>{counter[key]}</td></tr>"
            for key in keys
        ) or "<tr><td colspan='2'><em>none</em></td></tr>"
        return (
            f"<div class='card'><h4>{_esc(title)}</h4><table class='grid'>"
            f"<tbody>{rows}</tbody></table></div>"
        )

    usage_rows = "".join(
        f"<tr><td>{_esc(row['stage'])}</td><td class='num'>{row['requests']}</td>"
        f"<td class='num'>{row['cache_hits']}</td><td class='num'>{row['input_tokens']}</td>"
        f"<td class='num'>{row['cached_tokens']}</td>"
        f"<td class='num'>{row['output_tokens']}</td></tr>"
        for row in data["usage"]
    ) or "<tr><td colspan='6'><em>none</em></td></tr>"
    return (
        "<section class='pane summary' id='pane-summary' hidden>"
        f"<h1>agent-loop review · <code>{_esc(campaign)}</code></h1>"
        + _kv([
            ("sources in campaign", sum(status_all.values())),
            ("sources exported", len(exported)),
            ("global branches", sum(global_status.values())),
            ("local branches", sum(leaf_status.values())),
            ("thumbnails written", len(thumbs.written)),
            ("blobs missing", len(thumbs.missing)),
            ("thumbnails failed", len(thumbs.failed)),
        ])
        + "<div class='cards'>"
        + _counter_table("source terminal status (campaign)", status_all)
        + _counter_table("global branch status", global_status)
        + _counter_table("leaf status", leaf_status)
        + _counter_table("leaf commit_status", leaf_commit)
        + _counter_table("global bin coverage", global_bins, GLOBAL_BIN_ORDER)
        + _counter_table("local bin coverage", local_bins, LOCAL_BIN_ORDER)
        + _counter_table(
            "committed leaves by global bin", committed_global_bins, GLOBAL_BIN_ORDER)
        + _counter_table("reason_codes (global)", reasons_global)
        + _counter_table("reason_codes (local)", reasons_local)
        + "</div>"
        + "<h4>token usage by stage</h4><table class='grid'><thead><tr><th>stage</th>"
        "<th>requests</th><th>cache_hits</th><th>input_tokens</th><th>cached_tokens</th>"
        f"<th>output_tokens</th></tr></thead><tbody>{usage_rows}</tbody></table>"
        "</section>"
    )


STYLE = """
:root { color-scheme: light; --side: 250px; }
* { box-sizing: border-box; }
[hidden] { display: none !important; }
body { font: 13px/1.5 -apple-system,Segoe UI,Roboto,"Noto Sans CJK SC",sans-serif;
       margin: 0; color: #16191c; background: #fff; }
h1 { font-size: 20px; margin: 0 0 8px; } h2 { font-size: 16px; margin: 0; }
h4 { font-size: 13px; margin: 14px 0 4px; } h5 { font-size: 12px; margin: 10px 0 3px; }
code { font-family: ui-monospace,Menlo,Consolas,monospace; font-size: 11px; }
aside { position: fixed; top: 0; left: 0; bottom: 0; width: var(--side);
        border-right: 1px solid #d7dde3; display: flex; flex-direction: column;
        background: #f7f9fb; }
aside .head { padding: 8px 10px; border-bottom: 1px solid #e2e7ec; }
aside .head .ttl { font-weight: 600; font-size: 12px; word-break: break-all; }
#tabs { display: flex; gap: 6px; margin-top: 6px; }
#tabs button { font: inherit; font-size: 11px; padding: 2px 9px; cursor: pointer;
               border: 1px solid #c6ced6; background: #fff; border-radius: 10px; }
#tabs button.on { background: #2f6fb0; border-color: #2f6fb0; color: #fff; }
#list { list-style: none; margin: 0; padding: 0; overflow-y: auto; flex: 1 1 auto; }
#list li { display: flex; gap: 7px; align-items: center; padding: 4px 8px; cursor: pointer;
           border-bottom: 1px solid #edf1f4; }
#list li:hover { background: #eef3f8; }
#list li.on { background: #dbe8f5; }
#list img, #list .nothumb { width: 46px; height: 34px; object-fit: cover; flex: 0 0 46px;
                            border: 1px solid #ccd2d8; border-radius: 2px;
                            background: #e9edf1; }
#list .si { min-width: 0; display: flex; flex-direction: column; }
#list .sid { font-family: ui-monospace,Menlo,Consolas,monospace; font-size: 11px;
             overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
#list .meta { display: flex; gap: 5px; align-items: center; font-size: 10px; }
#list .n { color: #5a6672; }
main { margin-left: var(--side); padding: 12px 16px 60px; }
.panehdr { display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
           padding-bottom: 6px; border-bottom: 1px solid #e2e7ec; }
.panehdr .hl { color: #5a6672; font-size: 12px; }
p.diag { color: #3d474f; margin: 6px 0 2px; max-width: 1100px; }
.badge { background: #e6ebf0; border-radius: 10px; padding: 0 7px; font-size: 10px;
         white-space: nowrap; }
.badge.big { font-size: 12px; }
.badge.accepted, .badge.ok { background: #d7efdb; }
.badge.error, .badge.source_rejected, .badge.warn { background: #f7dcd8; }
.badge.no { background: #e6ebf0; color: #5a6672; }
.chains { display: flex; flex-direction: column; gap: 14px; margin-top: 10px; }
.chain { border: 1px solid #e2e7ec; border-radius: 5px; padding: 6px 8px 8px;
         background: #fcfdfe; }
.chdr { display: flex; gap: 7px; align-items: center; flex-wrap: wrap; margin-bottom: 4px; }
.cidx { font-size: 11px; color: #5a6672; font-weight: 600; }
.trip { display: flex; gap: 6px; align-items: flex-start; }
.trip figure { margin: 0; flex: 1 1 0; min-width: 0; text-align: center; }
.trip img { width: 100%; height: auto; display: block; border: 1px solid #ccd2d8;
            border-radius: 2px; }
figcaption { font-size: 10px; color: #5a6672; }
.lines { margin-top: 3px; font-size: 11.5px; color: #2c343b; }
.lines .bin { background: #eef2f6; border-radius: 3px; padding: 0 5px; }
.dimtext { color: #8a939c; }
.missing, .absent { width: 100%; aspect-ratio: 3 / 2; display: flex; align-items: center;
           justify-content: center; border: 1px dashed #c0504d; color: #c0504d;
           font-size: 11px; text-align: center; border-radius: 2px; }
.absent { border-color: #c6ced6; color: #8a939c; }
.kv { display: grid; grid-template-columns: max-content 1fr; gap: 1px 10px;
      align-items: baseline; }
.kv .k { color: #5a6672; font-size: 11px; }
.kv .v { font-size: 12px; word-break: break-word; }
.row { display: flex; gap: 14px; align-items: flex-start; flex-wrap: wrap; }
.row > .half { flex: 1 1 420px; min-width: 300px; }
.row > .half b { display: block; }
.scroll { overflow-x: auto; max-width: 100%; }
table.grid { border-collapse: collapse; font-size: 11px; }
table.grid th, table.grid td { border: 1px solid #dde2e7; padding: 2px 5px;
                               text-align: left; vertical-align: top;
                               white-space: nowrap; }
table.grid th { background: #eef2f6; font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
td.cap { max-width: 380px; white-space: normal; }
ul { margin: 2px 0 2px 16px; padding: 0; }
pre { background: #f2f4f7; padding: 6px; overflow-x: auto; font-size: 11px; }
.cards { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 8px; }
.card { flex: 0 1 260px; }
.card table.grid { width: 100%; }
.warn { color: #c0504d; }
details.dump, details.rest { margin-top: 14px; border: 1px solid #e2e7ec;
                             border-radius: 4px; padding: 6px 8px; background: #fbfcfd; }
summary { cursor: pointer; font-weight: 600; font-size: 12px; }
"""

SCRIPT = """
(function () {
  var list = document.getElementById('list');
  var tabs = document.getElementById('tabs');
  var main = document.querySelector('main');
  var panes = main.querySelectorAll('section.pane');
  function show(id) {
    for (var i = 0; i < panes.length; i++) { panes[i].hidden = panes[i].id !== id; }
    var items = list.querySelectorAll('li');
    for (var j = 0; j < items.length; j++) {
      var on = items[j].getAttribute('data-target') === id;
      items[j].className = on ? 'on' : '';
    }
    window.scrollTo(0, 0);
  }
  function route() {
    var id = (location.hash || '').slice(1) || 'pane-summary';
    if (!document.getElementById(id)) { id = 'pane-summary'; }
    show(id);
  }
  list.addEventListener('click', function (event) {
    var node = event.target;
    while (node && node !== list && !node.getAttribute('data-target')) {
      node = node.parentNode;
    }
    if (node && node !== list) { location.hash = node.getAttribute('data-target'); }
  });
  tabs.addEventListener('click', function (event) {
    var button = event.target;
    if (!button || button.tagName !== 'BUTTON') { return; }
    var filter = button.getAttribute('data-filter');
    var all = tabs.querySelectorAll('button');
    for (var i = 0; i < all.length; i++) {
      all[i].className = all[i] === button ? 'on' : '';
    }
    var items = list.querySelectorAll('li[data-status]');
    for (var j = 0; j < items.length; j++) {
      items[j].hidden = !(filter === 'all'
                          || items[j].getAttribute('data-status') === filter);
    }
  });
  window.addEventListener('hashchange', route);
  route();
})();
"""


# ------------------------------------------------------------------------------ main


def export(args: argparse.Namespace) -> dict[str, Any]:
    dsn = args.dsn or os.environ.get("VERARETOUCH_AGENT_POSTGRES_DSN")
    if not dsn:
        raise SystemExit(
            "no PostgreSQL DSN: pass --dsn or export VERARETOUCH_AGENT_POSTGRES_DSN"
        )
    with _connect(dsn) as conn:
        campaign = args.campaign or _latest_campaign(conn)
        data = _collect(conn, campaign, int(args.limit))
    out_dir = Path(args.out_dir) if args.out_dir else \
        Path("docs/assets") / f"agent_loop_review_{campaign}"
    out_dir.mkdir(parents=True, exist_ok=True)
    resolver = ArtifactResolver(_default_roots(args.artifact_root), args.catalog_db)
    thumbs = Thumbnails(resolver, out_dir, args.thumb_short_edge)

    branch_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in data["branches"]:
        row["proposal"] = _loads(row.pop("proposal_json")) or {}
        row["result"] = _loads(row.pop("result_json")) or {}
        branch_index[str(row["source_sha256"])].append(row)
    audits = {str(row["branch_id"]): row for row in data["audits"]}
    renders: dict[str, dict[str, Any]] = {}
    for row in data["renders"]:
        renders[str(row["branch_id"])] = row  # ordered by created_at: last wins

    unreadable: list[str] = []
    pending: list[str] = []
    sections: list[str] = []
    items: list[str] = []
    for source in data["sources"]:
        tree = annotation = shortlist = None
        manifest_ref = _loads(source["manifest_json"])
        if not isinstance(manifest_ref, Mapping):
            # record_source_start leaves manifest_json NULL until the source finishes.
            pending.append(str(source["source_id"]))
        else:
            try:
                tree = resolver.read_json(manifest_ref)
            except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                unreadable.append(f"{source['source_id']}:tree:{type(exc).__name__}")
        if tree:
            for key, target in (("source_annotation_artifact", "annotation"),
                                ("global_shortlist_artifact", "shortlist")):
                ref = tree.get(key)
                if not ref:
                    continue
                try:
                    value = resolver.read_json(ref)
                except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                    unreadable.append(f"{source['source_id']}:{target}:{type(exc).__name__}")
                    continue
                if target == "annotation":
                    annotation = value
                else:
                    shortlist = value
        pane, item = _source_pane(
            source, tree, annotation, shortlist,
            branch_index.get(str(source["source_sha256"]), []),
            audits, renders, thumbs,
        )
        sections.append(pane)
        items.append(item)

    summary = _summary_pane(campaign, data, data["sources"], branch_index, thumbs)
    accepted = sum(1 for source in data["sources"] if str(source["status"]) == "accepted")
    sidebar = (
        "<aside><div class='head'>"
        f"<div class='ttl'><a href='#pane-summary'>{_esc(campaign)}</a></div>"
        f"<div id='tabs'><button data-filter='all' class='on'>all "
        f"{len(data['sources'])}</button>"
        f"<button data-filter='accepted'>accepted {accepted}</button></div></div>"
        f"<ul id='list'>{''.join(items)}</ul></aside>"
    )
    page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>agent-loop review {_esc(campaign)}</title>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<style>{STYLE}</style></head><body>"
        + sidebar + "<main>" + summary + "".join(sections) + "</main>"
        + f"<script>{SCRIPT}</script></body></html>\n"
    )
    index = out_dir / "index.html"
    temporary = index.with_name("index.html.tmp")
    temporary.write_text(page, encoding="utf-8")
    temporary.replace(index)
    pruned = thumbs.prune()

    stats = {
        "campaign": campaign,
        "page": str(index),
        "sources": len(data["sources"]),
        "global_branches": sum(
            1 for rows in branch_index.values() for row in rows if row["level"] == "global"
        ),
        "local_branches": sum(
            1 for rows in branch_index.values() for row in rows if row["level"] == "local"
        ),
        "thumbnails": len(thumbs.written),
        "thumbnail_short_edge": thumbs.short_edge,
        "thumbnails_pruned": pruned,
        "missing_blobs": len(thumbs.missing),
        "thumbnail_failures": len(thumbs.failed),
        "unreadable_json": unreadable,
        "manifest_pending": pending,
        "artifact_roots": [str(store.root) for store in resolver.stores],
    }
    (out_dir / "export_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return stats


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--campaign", default=None,
                        help="campaign_id; default = newest campaign in agent_source_run")
    result.add_argument("--out-dir", type=Path, default=None,
                        help="default docs/assets/agent_loop_review_<campaign>")
    result.add_argument("--limit", type=int, default=0, help="0 = every source")
    result.add_argument("--dsn", default=None,
                        help="default $VERARETOUCH_AGENT_POSTGRES_DSN")
    result.add_argument("--artifact-root", type=Path, action="append", default=[],
                        help="repeatable; default = every */blobs under "
                             + " and ".join(str(item) for item in DEFAULT_ROOT_PARENTS))
    result.add_argument("--catalog-db", type=Path,
                        default=DEFAULT_CATALOG_DB if DEFAULT_CATALOG_DB.is_file() else None,
                        help="landed-tar global catalog used when a blob left staging")
    result.add_argument("--thumb-short-edge", type=int, default=512)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    stats = export(args)
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
