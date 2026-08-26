"""Blind-review material pack for one agent-loop smoke campaign (EPR-048).

Reads an exported audit JSON (``audit.smoke5.json``: ``tables.agent_source_run`` rows,
each carrying a ``manifest_json`` artifact reference to the source's edit-tree manifest)
plus the content-addressed artifact store the run wrote, and emits, per source:

  * ``contact_sheet.<source_id>.png`` - the section 10.1 sheet: one ``source`` panel, one
    ``global-<i>`` panel per formal global branch, and per committed leaf a
    ``edit-<k>`` final render panel plus an ``edit-<k> mask`` overlay panel.  Panel
    captions are drawn from a closed neutral vocabulary only (see ``PANEL_CAPTION_RE``);
    no preset id, no LUT/style name, no metric, no revision string ever reaches a pixel.
  * ``contact_sheet.<source_id>.sidecar.json`` - the internal key (panel code -> leaf id,
    preset id, mask id, render hash).  NOT for the blind reviewer.

and, once per pack, ``audit_summary.md`` (numbers only) and ``review_manifest.json``
(the machine-readable Phase 4 input: sheet path + panel code list, nothing else).

Usage:
    PYTHONPATH=dataset_build/src:. .venv/bin/python -m dataset_build.tools.build_smoke_review_pack \
        --audit experiments/prs/EPR-047_wiring/audit.smoke5.json \
        --out-dir docs/assets/r4_smoke5_review_20260825
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset_build.agent_loop.artifacts import ArtifactStore


BUILD_ID = "smoke-review-pack-epr048"

# ---------------------------------------------------------------- blind-review redline

# Every string that reaches a sheet pixel must match this closed vocabulary.  A caption
# is generated, never copied from run data, so the whitelist is the real guard; the
# blocklist below is the belt-and-braces pass that also runs over the review manifest.
PANEL_CAPTION_RE = re.compile(r"^(source|global-\d+|edit-\d+|edit-\d+ mask)$")
SHEET_CAPTION_RE = re.compile(r"^sheet [A-Z]$")

FORBIDDEN_PATTERNS: tuple[tuple[str, str], ...] = (
    ("preset_id", r"rcp_[0-9a-f]+"),
    ("preset_word", r"preset"),
    ("style_family", r"fam_\d+"),
    ("score_word", r"scor"),
    ("lut_word", r"\blut\b"),
    ("metric_delta_e", r"delta_e|Δe|\bde\d"),
    ("chain_name", r"r4g4l4|g4_|l4_|\bg4\b|\bl4\b|\br4\b"),
    ("campaign_word", r"campaign"),
    ("model_name", r"gpt-|qwen|terra|opus|claude"),
    ("revision_word", r"revision|thread_id|prompt_rev"),
    ("sha256", r"[0-9a-f]{64}"),
    ("float_number", r"\d+\.\d+"),
    ("branch_id", r"\b(?:global|local)_[0-9a-f]{20,}"),
    ("mask_id", r"mask_[0-9a-f]{12,}"),
)


# The subset that identifies the producing system or an individual candidate.  Applied to
# every reviewer-visible file, including the ones that legitimately carry paths/constants.
IDENTITY_PATTERN_NAMES = (
    "preset_id", "preset_word", "style_family", "score_word", "lut_word",
    "metric_delta_e", "campaign_word", "model_name", "revision_word", "sha256",
    "branch_id", "mask_id",
)


class BlindLeak(AssertionError):
    """A string bound for reviewer-visible material carries an identity or a metric."""


def assert_no_leak(text: str, *, where: str) -> None:
    lowered = str(text).lower()
    for name, pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, lowered):
            raise BlindLeak(f"{where}: {name!r} pattern matched in {text!r}")


def assert_no_identity(text: str, *, where: str) -> None:
    lowered = str(text).lower()
    lookup = dict(FORBIDDEN_PATTERNS)
    for name in IDENTITY_PATTERN_NAMES:
        if re.search(lookup[name], lowered):
            raise BlindLeak(f"{where}: {name!r} pattern matched")


def assert_panel_caption(text: str) -> str:
    if not PANEL_CAPTION_RE.match(text):
        raise BlindLeak(f"panel caption outside the closed vocabulary: {text!r}")
    assert_no_leak(text, where="panel caption")
    return text


def assert_sheet_caption(text: str) -> str:
    if not SHEET_CAPTION_RE.match(text):
        raise BlindLeak(f"sheet caption outside the closed vocabulary: {text!r}")
    assert_no_leak(text, where="sheet caption")
    return text


# --------------------------------------------------------------------------- rendering

PANEL_W = 512
PANEL_H = 384
GAP = 10
CAPTION_H = 26
HEADER_H = 34
BG = (12, 14, 17)
PANEL_BG = (22, 25, 29)
CAPTION_FG = (238, 241, 244)
# Fixed overlay colour, identical on every sheet and every mask.  No per-image scaling
# of the alpha is applied: weight = OVERLAY_WEIGHT * alpha, alpha as stored.
OVERLAY_RGB = (255, 96, 0)
OVERLAY_WEIGHT = 0.55
OVERLAY_EDGE_LEVEL = 0.5


def _font(size: int, *, bold: bool = False) -> Any:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.is_file() \
        else ImageFont.load_default()


def _outline(binary: np.ndarray) -> np.ndarray:
    """Inner four-neighbour boundary of a boolean field (no scipy/cv2 dependency)."""
    inner = np.asarray(binary, dtype=bool).copy()
    inner[:-1, :] &= binary[1:, :]
    inner[1:, :] &= binary[:-1, :]
    inner[:, :-1] &= binary[:, 1:]
    inner[:, 1:] &= binary[:, :-1]
    return np.asarray(binary, dtype=bool) & ~inner


def overlay_mask(image: Image.Image, alpha: np.ndarray) -> Image.Image:
    """Fixed-colour mask overlay.  `alpha` is used as stored, never re-normalised."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    weight = (OVERLAY_WEIGHT * np.clip(alpha, 0.0, 1.0))[..., None]
    rgb = rgb * (1.0 - weight) + np.asarray(OVERLAY_RGB, dtype=np.float32) * weight
    rgb[_outline(alpha >= OVERLAY_EDGE_LEVEL)] = np.asarray(OVERLAY_RGB, dtype=np.float32)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")


def _panel(image: Image.Image, caption: str) -> Image.Image:
    assert_panel_caption(caption)
    body = image.convert("RGB").copy()
    body.thumbnail((PANEL_W, PANEL_H), getattr(Image, "Resampling", Image).LANCZOS)
    panel = Image.new("RGB", (PANEL_W, PANEL_H + CAPTION_H), PANEL_BG)
    panel.paste(body, ((PANEL_W - body.width) // 2,
                       CAPTION_H + (PANEL_H - body.height) // 2))
    ImageDraw.Draw(panel).text((8, 4), caption, fill=CAPTION_FG, font=_font(15, bold=True))
    return panel


def compose_sheet(panels: Sequence[tuple[str, Image.Image]], sheet_code: str,
                  *, n_globals: int) -> Image.Image:
    """Row 0 = source + every global; then two (edit, edit mask) pairs per row."""
    assert_sheet_caption(sheet_code)
    rows: list[list[Image.Image]] = []
    drawn = [_panel(image, caption) for caption, image in panels]
    head = 1 + n_globals
    rows.append(drawn[:head])
    rest = drawn[head:]
    for start in range(0, len(rest), 4):
        rows.append(rest[start:start + 4])
    ncols = max(len(row) for row in rows)
    cell_w, cell_h = PANEL_W + GAP, PANEL_H + CAPTION_H + GAP
    sheet = Image.new(
        "RGB", (GAP + ncols * cell_w, HEADER_H + GAP + len(rows) * cell_h), BG
    )
    ImageDraw.Draw(sheet).text(
        (GAP, 8), sheet_code, fill=CAPTION_FG, font=_font(19, bold=True)
    )
    for r, row in enumerate(rows):
        for c, panel in enumerate(row):
            sheet.paste(panel, (GAP + c * cell_w, HEADER_H + GAP + r * cell_h))
    return sheet


# ------------------------------------------------------------------------ audit reading


def load_audit(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# Tables whose rows are content-addressed and therefore shared by every run of the same
# campaign (a re-run that produces a byte-identical render/branch re-uses the row the
# first run inserted, keeping the first run's `created_at`).  They must NOT be filtered:
# the manifests of the selected run reference them by hash.
def filter_audit(audit: Mapping[str, Any], *, prompt_revision: str | None = None,
                 since_epoch: float | None = None) -> dict[str, Any]:
    """Restrict a multi-run audit export to the rows of one run.

    Selection only - nothing about how a record is built or rendered changes.  Two
    independent cuts, both optional:

      * ``prompt_revision``: keep the ``agent_source_run`` rows whose
        ``prompt_revision`` equals (or, for ``rev:pass-N`` values, starts with) it.
      * ``since_epoch``: keep the ``agent_source_run`` rows started at/after it and the
        ``api_attempt`` rows started at/after it (the per-attempt clock is
        ``started_at``; a cached/pending request row from an earlier run can collect
        new attempts in a later run, so the request tables stay whole and only the
        attempts are cut).
    """
    tables = {name: list(rows) for name, rows in audit["tables"].items()}
    runs = tables.get("agent_source_run", [])
    if prompt_revision is not None:
        runs = [row for row in runs
                if str(row.get("prompt_revision", "")) == prompt_revision
                or str(row.get("prompt_revision", "")).startswith(prompt_revision + ":")]
        if not runs:
            raise ValueError(f"no agent_source_run row with prompt_revision "
                             f"{prompt_revision!r}")
    if since_epoch is not None:
        runs = [row for row in runs if float(row["started_at"]) >= since_epoch]
        tables["api_attempt"] = [row for row in tables.get("api_attempt", [])
                                 if float(row["started_at"]) >= since_epoch]
    tables["agent_source_run"] = runs
    return {**audit, "tables": tables}


def source_input_images(audit: Mapping[str, Any]) -> dict[str, str]:
    """`source_sha256` -> sha256 of the render-resolution source the renderer consumed."""
    found: dict[str, set[str]] = collections.defaultdict(set)
    for row in audit["tables"]["render_record"]:
        if row.get("stage") == "global":
            found[str(row["source_sha256"])].add(str(row["input_json"]["input_image_sha256"]))
    resolved: dict[str, str] = {}
    for key, values in found.items():
        if len(values) != 1:
            raise ValueError(f"source {key} has {len(values)} global render inputs")
        resolved[key] = values.pop()
    return resolved


def attempts_by_source(audit: Mapping[str, Any],
                       manifests: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    """Per-source `api_attempt` count.

    A resolved request carries an `api_request_context` row with the source hash.  A
    request whose every attempt failed stays `pending` and never gets one, so it is
    attributed through the artifact sha256 references its prompt carries: every
    artifact in a manifest belongs to exactly one source.
    """
    tables = audit["tables"]
    context = {str(row["request_hash"]): str(row["source_sha256"])
               for row in tables["api_request_context"]}
    owner: dict[str, set[str]] = collections.defaultdict(set)
    sha_re = re.compile(r"\b[0-9a-f]{64}\b")
    for source_sha, manifest in manifests.items():
        for digest in sha_re.findall(json.dumps(manifest, sort_keys=True)):
            owner[digest].add(source_sha)
    per_request = collections.Counter(
        str(row["request_hash"]) for row in tables["api_attempt"]
    )
    counts: collections.Counter[str] = collections.Counter()
    unattributed = 0
    for row in tables["api_request"]:
        digest = str(row["request_hash"])
        source_sha = context.get(digest)
        if source_sha is None:
            hits: set[str] = set()
            for candidate in sha_re.findall(json.dumps(row["canonical_request_json"],
                                                       sort_keys=True)):
                hits |= owner.get(candidate, set())
            if len(hits) != 1:
                unattributed += per_request[digest]
                continue
            source_sha = hits.pop()
        counts[source_sha] += per_request[digest]
    if unattributed:
        raise ValueError(f"{unattributed} api_attempt rows could not be attributed")
    return dict(counts)


def load_manifests(audit: Mapping[str, Any],
                   store: ArtifactStore) -> dict[str, dict[str, Any]]:
    return {
        str(row["source_sha256"]): json.loads(store.read_bytes(row["manifest_json"]))
        for row in audit["tables"]["agent_source_run"]
    }


def collect_sources(audit: Mapping[str, Any],
                    manifests: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One record per source: panel plan, sidecar key, audit numbers."""
    runs = sorted(audit["tables"]["agent_source_run"], key=lambda row: row["started_at"])
    inputs = source_input_images(audit)
    attempts = attempts_by_source(audit, manifests)
    records: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        source_sha = str(run["source_sha256"])
        manifest = manifests[source_sha]
        branches = [b for b in manifest["branches"] if b["status"] == "formal_global"]
        globals_: list[dict[str, Any]] = []
        edits: list[dict[str, Any]] = []
        deficits: collections.Counter[str] = collections.Counter()
        for scope_key in ("g4_deficits", "l4_source_deficits"):
            for item in manifest.get(scope_key, []):
                deficits[f"{item.get('scope', scope_key)}/{item['reason']}"] += 1
        for b_index, branch in enumerate(branches, start=1):
            code = f"global-{b_index}"
            globals_.append({
                "panel": code,
                "branch_id": branch["branch_id"],
                "artifact": branch["global_render"]["artifact"],
                "preset_id": branch["global_render"]["parameters"]["preset_id"],
                "strength_bin": branch["global_render"]["parameters"]["strength_bin"],
                "global_strength": branch["global_render"]["parameters"]["global_strength"],
                "delta_e": branch["global_render"]["metrics"]["delta_e"],
                "render_hash": branch["global_render"]["render_hash"],
                "l4_verdict": branch["l4_verdict"]["achieved"],
                "l4_pick_source": branch["l4_pick_source"],
                "l4_refine_source": branch["l4_refine_source"],
            })
            for item in branch.get("l4_deficits", []):
                deficits[f"{item['scope']}/{item['reason']}"] += 1
            for leaf in branch["leaves"]:
                if leaf["commit_status"] != "committed":
                    continue
                k = len(edits) + 1
                edits.append({
                    "panel": f"edit-{k}",
                    "mask_panel": f"edit-{k} mask",
                    "parent_panel": code,
                    "leaf_id": leaf["branch_id"],
                    "artifact": leaf["render"]["artifact"],
                    "applied_alpha_artifact": leaf["render"]["applied_alpha_artifact"],
                    "mask_id": leaf["proposal"]["mask_id"],
                    "mask_family": leaf["mask_family"],
                    "mask_role": leaf["mask_role"],
                    "action": leaf["proposal"]["action"],
                    "sign": int(leaf["proposal"]["sign"]),
                    "intent": leaf["intent"],
                    "preset_id": leaf["proposal"]["preset_id"],
                    "strength_bin": leaf["render"]["effective_strength_bin"],
                    "local_strength": leaf["render"]["parameters"]["local_strength"],
                    "delta_e": leaf["render"]["metrics"]["delta_e"],
                    "render_hash": leaf["render"]["render_hash"],
                    "status": leaf["status"],
                    "validator_status": leaf["validator_status"],
                    "audit_notes": list(leaf["audit"]),
                })
        records.append({
            "index": index,
            "sheet_code": f"sheet {chr(ord('A') + index)}",
            "source_id": str(run["source_id"]),
            "source_sha256": source_sha,
            "source_image_sha256": inputs[source_sha],
            "status": str(run["status"]),
            "counts": dict(run["counts_json"]),
            "manifest_sha256": str(run["manifest_json"]["sha256"]),
            "branch_count": len(branches),
            "leaf_count": len(edits),
            "globals": globals_,
            "edits": edits,
            "deficits": dict(sorted(deficits.items())),
            "api_attempts": int(attempts.get(source_sha, 0)),
        })
    return records


# ------------------------------------------------------------------------- deliverables


def build_sheet(record: Mapping[str, Any], store: ArtifactStore) -> Image.Image:
    with Image.open(store.local_path(str(record["source_image_sha256"]))) as handle:
        source = handle.convert("RGB")
    panels: list[tuple[str, Image.Image]] = [("source", source)]
    for entry in record["globals"]:
        with Image.open(store.path_for(entry["artifact"])) as handle:
            panels.append((entry["panel"], handle.convert("RGB")))
    for entry in record["edits"]:
        with Image.open(store.path_for(entry["artifact"])) as handle:
            rendered = handle.convert("RGB")
        panels.append((entry["panel"], rendered))
        with Image.open(store.path_for(entry["applied_alpha_artifact"])) as handle:
            alpha = np.asarray(handle.convert("L").resize(
                rendered.size, getattr(Image, "Resampling", Image).BILINEAR
            ), dtype=np.float32) / 255.0
        panels.append((entry["mask_panel"], overlay_mask(rendered, alpha)))
    return compose_sheet(panels, str(record["sheet_code"]),
                         n_globals=len(record["globals"]))


def sidecar_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "build_id": BUILD_ID,
        "note": "internal key - NOT to be shown to a blind reviewer",
        "sheet_code": record["sheet_code"],
        "source_id": record["source_id"],
        "source_sha256": record["source_sha256"],
        "source_image_sha256": record["source_image_sha256"],
        "manifest_sha256": record["manifest_sha256"],
        "panels": (
            [{"panel": "source", "artifact_sha256": record["source_image_sha256"]}]
            + [{"panel": g["panel"], "branch_id": g["branch_id"],
                "preset_id": g["preset_id"], "strength_bin": g["strength_bin"],
                "artifact_sha256": g["artifact"]["sha256"],
                "render_hash": g["render_hash"]} for g in record["globals"]]
            + [item for e in record["edits"] for item in (
                {"panel": e["panel"], "parent_panel": e["parent_panel"],
                 "leaf_id": e["leaf_id"], "preset_id": e["preset_id"],
                 "mask_id": e["mask_id"], "action": e["action"], "sign": e["sign"],
                 "artifact_sha256": e["artifact"]["sha256"],
                 "render_hash": e["render_hash"]},
                {"panel": e["mask_panel"], "leaf_id": e["leaf_id"],
                 "mask_id": e["mask_id"],
                 "alpha_sha256": e["applied_alpha_artifact"]["sha256"],
                 "overlay_rgb": list(OVERLAY_RGB), "overlay_weight": OVERLAY_WEIGHT},
            )]
        ),
    }


def review_manifest(records: Sequence[Mapping[str, Any]], out_dir: Path) -> dict[str, Any]:
    sheets = []
    for record in records:
        codes = ["source"] + [g["panel"] for g in record["globals"]]
        for entry in record["edits"]:
            codes += [entry["panel"], entry["mask_panel"]]
        for code in codes:
            assert_panel_caption(code)
        sheets.append({
            "sheet_code": record["sheet_code"],
            "sheet_path": str(out_dir / f"contact_sheet.{record['source_id']}.png"),
            "panel_codes": codes,
            "panel_count": len(codes),
            # Which whole-image panel each edit sits under.  Structure only: no candidate
            # table, no preset, no metric.
            "panel_parents": {e["panel"]: e["parent_panel"] for e in record["edits"]},
        })
    payload = {
        "build_id": BUILD_ID,
        "overlay": {"rgb": list(OVERLAY_RGB), "weight": OVERLAY_WEIGHT,
                    "edge_level": OVERLAY_EDGE_LEVEL},
        "checklist_dimensions": ["usability", "diversity"],
        "sheets": sheets,
    }
    for sheet in sheets:
        assert_sheet_caption(str(sheet["sheet_code"]))
    # The manifest also carries file paths and the fixed overlay constants, so the strict
    # caption vocabulary does not apply to it; what must never appear is an identity.
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    assert_no_identity(body, where="review_manifest")
    return payload


def audit_summary(records: Sequence[Mapping[str, Any]], audit: Mapping[str, Any]) -> str:
    lines = [
        "# r4g4l4-smoke5 review pack - audit summary",
        "",
        f"build_id: {BUILD_ID}",
        f"campaign_id: {audit['config']['campaign_id']}",
        "internal document: it carries preset ids and metrics and is NOT part of the "
        "blind-review material (`review_manifest.json` is).",
        "",
        "## per source",
        "",
        "| sheet | source_id | status | branches | leaves | validator_pass | "
        "validator_skipped | api_attempts | deficit rows |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| {record['sheet_code']} | {record['source_id']} | {record['status']} | "
            f"{record['branch_count']} | {record['leaf_count']} | "
            f"{record['counts']['validator_pass_leaves']} | "
            f"{record['counts']['validator_skipped_leaves']} | "
            f"{record['api_attempts']} | {sum(record['deficits'].values())} |"
        )
    lines += ["", "## per leaf", "",
              "| sheet | panel | parent | mask family | mask role | action | sign | "
              "strength bin | delta_e reached | validator_status | commit audit |",
              "|---|---|---|---|---|---|---:|---|---:|---|---|"]
    for record in records:
        for entry in record["edits"]:
            lines.append(
                f"| {record['sheet_code']} | {entry['panel']} | {entry['parent_panel']} | "
                f"{entry['mask_family']} | {entry['mask_role']} | {entry['action']} | "
                f"{entry['sign']:+d} | {entry['strength_bin']} | "
                f"{entry['delta_e']:.3f} | {entry['validator_status']} | "
                f"{','.join(entry['audit_notes']) or '-'} |"
            )
    lines += ["", "## per global", "",
              "| sheet | panel | strength bin | global_strength | delta_e reached | "
              "verdict | pick source | refine source |",
              "|---|---|---|---:|---:|---|---|---|"]
    for record in records:
        for entry in record["globals"]:
            lines.append(
                f"| {record['sheet_code']} | {entry['panel']} | {entry['strength_bin']} | "
                f"{entry['global_strength']:.4f} | {entry['delta_e']:.3f} | "
                f"{entry['l4_verdict']} | {entry['l4_pick_source']} | "
                f"{entry['l4_refine_source']} |"
            )
    classes = sorted({key for record in records for key in record["deficits"]})
    lines += ["", "## deficit counts by class", "",
              "| scope/reason | " + " | ".join(r["sheet_code"] for r in records) + " | total |",
              "|---|" + "---:|" * (len(records) + 1)]
    for name in classes:
        row = [record["deficits"].get(name, 0) for record in records]
        lines.append(f"| {name} | " + " | ".join(str(v) for v in row)
                     + f" | {sum(row)} |")
    total_row = [sum(record["deficits"].values()) for record in records]
    lines.append("| TOTAL | " + " | ".join(str(v) for v in total_row)
                 + f" | {sum(total_row)} |")
    attempts = collections.Counter(
        str(row["error_type"]) for row in audit["tables"]["api_attempt"]
    )
    lines += ["", "## api_attempt error_type (campaign total)", "",
              "| error_type | count |", "|---|---:|"]
    for name, count in sorted(attempts.items()):
        lines.append(f"| {'ok' if name == 'None' else name} | {count} |")
    lines.append(f"| TOTAL | {sum(attempts.values())} |")
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------------ reconciliation


def reconcile(records: Sequence[Mapping[str, Any]], audit: Mapping[str, Any],
              manifests: Mapping[str, Mapping[str, Any]],
              review: Mapping[str, Any]) -> tuple[int, list[str]]:
    """Re-derive every summary number from the audit tables.  Returns (checks, failures)."""
    tables = audit["tables"]
    checks = 0
    failures: list[str] = []

    def check(name: str, left: Any, right: Any) -> None:
        nonlocal checks
        checks += 1
        if left != right:
            failures.append(f"{name}: pack={left!r} audit={right!r}")

    runs = {str(row["source_sha256"]): row for row in tables["agent_source_run"]}
    branch_rows: collections.Counter[str] = collections.Counter()
    for row in tables["agent_branch"]:
        if row["level"] == "global" and row["status"] == "formal_global":
            branch_rows[str(row["source_sha256"])] += 1
    renders = {str(row["render_hash"]): row for row in tables["render_record"]}
    attempts = attempts_by_source(audit, manifests)
    sheets = {str(entry["sheet_code"]): entry for entry in review["sheets"]}

    check("source_count", len(records), len(tables["agent_source_run"]))
    for record in records:
        key = record["source_sha256"]
        run = runs[key]
        prefix = record["sheet_code"]
        check(f"{prefix}/source_id", record["source_id"], str(run["source_id"]))
        check(f"{prefix}/status", record["status"], str(run["status"]))
        check(f"{prefix}/committed_leaves", record["leaf_count"],
              int(run["counts_json"]["committed_leaves"]))
        check(f"{prefix}/formal_globals", record["branch_count"],
              int(run["counts_json"]["formal_globals"]))
        check(f"{prefix}/branch_rows", record["branch_count"], branch_rows[key])
        check(f"{prefix}/validator_skipped",
              sum(1 for e in record["edits"] if e["status"] == "validator_skipped"),
              int(run["counts_json"]["validator_skipped_leaves"]))
        check(f"{prefix}/validator_pass",
              sum(1 for e in record["edits"] if e["status"] == "validator_pass"),
              int(run["counts_json"]["validator_pass_leaves"]))
        check(f"{prefix}/api_attempts", record["api_attempts"], attempts.get(key, 0))
        check(f"{prefix}/panel_count", int(sheets[prefix]["panel_count"]),
              1 + record["branch_count"] + 2 * record["leaf_count"])
        check(f"{prefix}/leaf_ids",
              sorted(e["leaf_id"] for e in record["edits"]),
              sorted(str(x) for x in manifests[key]["committed_leaf_ids"]))
        for entry in list(record["globals"]) + list(record["edits"]):
            row = renders.get(str(entry["render_hash"]))
            check(f"{prefix}/{entry['panel']}/render_row", row is not None, True)
            if row is None:
                continue
            check(f"{prefix}/{entry['panel']}/delta_e",
                  round(float(entry["delta_e"]), 6),
                  round(float(row["metrics_json"]["delta_e"]), 6))
            check(f"{prefix}/{entry['panel']}/artifact",
                  entry["artifact"]["sha256"], str(row["artifact_json"]["sha256"]))
            check(f"{prefix}/{entry['panel']}/preset_id",
                  entry["preset_id"], str(row["parameters_json"]["preset_id"]))
    check("api_attempt_total", sum(r["api_attempts"] for r in records),
          len(tables["api_attempt"]))
    return checks, failures


# ------------------------------------------------------------------------------- driver


def build_pack(audit_path: Path, out_dir: Path, artifact_root: Path | None = None,
               *, prompt_revision: str | None = None,
               since_epoch: float | None = None) -> dict[str, Any]:
    audit = load_audit(audit_path)
    if prompt_revision is not None or since_epoch is not None:
        audit = filter_audit(audit, prompt_revision=prompt_revision,
                             since_epoch=since_epoch)
    root = Path(artifact_root) if artifact_root else Path(audit["config"]["artifact_root"])
    store = ArtifactStore(root)
    manifests = load_manifests(audit, store)
    records = collect_sources(audit, manifests)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Results land before the optional stages.
    summary_path = out_dir / "audit_summary.md"
    summary_path.write_text(audit_summary(records, audit), encoding="utf-8")
    manifest = review_manifest(records, out_dir)
    (out_dir / "review_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for record in records:
        (out_dir / f"contact_sheet.{record['source_id']}.sidecar.json").write_text(
            json.dumps(sidecar_payload(record), ensure_ascii=False, indent=2,
                       sort_keys=True) + "\n",
            encoding="utf-8",
        )
    checks, failures = reconcile(records, audit, manifests, manifest)

    sheets = []
    for record in records:
        path = out_dir / f"contact_sheet.{record['source_id']}.png"
        build_sheet(record, store).save(path)
        sheets.append({"path": str(path), "size": Image.open(path).size,
                       "panels": 1 + record["branch_count"] + 2 * record["leaf_count"]})
    return {
        "build_id": BUILD_ID,
        "out_dir": str(out_dir),
        "run_filter": {"prompt_revision": prompt_revision, "since_epoch": since_epoch},
        "sources": len(records),
        "sheets": sheets,
        "panel_counts": {r["sheet_code"]: 1 + r["branch_count"] + 2 * r["leaf_count"]
                         for r in records},
        "reconciled_checks": checks,
        "reconcile_failures": failures,
        "leaves_total": sum(r["leaf_count"] for r in records),
        "branches_total": sum(r["branch_count"] for r in records),
        "api_attempts_total": sum(r["api_attempts"] for r in records),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path,
                        default=Path("experiments/prs/EPR-047_wiring/audit.smoke5.json"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("docs/assets/r4_smoke5_review_20260825"))
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--prompt-revision", default=None,
                        help="keep only the agent_source_run rows of this run")
    parser.add_argument("--since-epoch", type=float, default=None,
                        help="keep only runs started, and api_attempt rows started, "
                             "at/after this unix timestamp")
    args = parser.parse_args(argv)
    report = build_pack(args.audit, args.out_dir, args.artifact_root,
                        prompt_revision=args.prompt_revision,
                        since_epoch=args.since_epoch)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["reconcile_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
