"""EPR-046b: Style Card annotation for the 4,051-LUT library (candidate-set flow).

Pipeline (spec: ``docs/确定性召回与 Global - Local 精排优化方案（R4 - G4 - L4）.md`` §3.1B,
revised by EPR-046b)::

    candidates  per LUT, every frozen vocabulary word whose numeric assertion is
                already true on that LUT, grouped by tag field (no thresholds and
                no measured values ever leave this step)
    ask         VLM sees the contact sheet + the four candidate lists and returns
                the visually obvious subset of each list + one English summary
    check       dataset_build/tools/style_vocab_check.check_tags -> accepted /
                rejected; annotation_quality = clean | conflicted (programmatic).
                By construction every selected word already passes, so a
                ``conflicted`` card is a bug signal, not a normal outcome.

There is no second annotation round: whatever comes back (after at most one
out-of-candidate re-ask) is final.

Hard invariants asserted at start-up (``preflight``):

* the frozen vocabulary payload sha256 equals :data:`EXPECTED_VOCAB_SHA`;
* the contact-sheet ``params_sha256`` equals :data:`EXPECTED_SHEET_PARAMS_SHA`;
* the checker answers a four-case negative self-test (pass / assertion-false /
  unknown-word / missing-feature) exactly as documented.

Every card goes through ``check_tags`` and through the subset guard
(``selected words`` subseteq ``candidate words``) before it is written; the call
counters are carried in ``ASSERTIONS`` and land in the progress and report files,
so "defined but never wired" is visible in the output numbers.

The VLM never sees a number: its whole context is the contact sheet PNG, the
``preset_id``, the sidecar ``strength_capacity`` and the candidate word lists.
It cannot emit a numeric field either -- the strict ``json_schema`` has no
numeric slot and the four tag arrays are ``enum``-bound to this LUT's candidates.

Usage::

    .venv/bin/python -m dataset_build.tools.build_style_cards --limit 10 \\
        --out experiments/prs/EPR-046_style-card-annotation/pilot_v2/style_cards.pilot.jsonl
    .venv/bin/python -m dataset_build.tools.build_style_cards --workers 12
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import threading
import time
import tomllib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from dataset_build.tools import style_vocab_check as svc

# --- fixed inputs -------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]
SHEETS_DIR = Path("/home/bc/data/scratch/lut_contact_sheets/sheets")
SIDECAR_DIR = Path("/home/bc/data/scratch/lut_contact_sheets/sidecar")
SHEET_PARAMS = Path("/home/bc/data/scratch/lut_contact_sheets/params.json")
ANNOTATIONS = Path("/home/bc/data/scratch/lut_reannotate/out/annotations.closed-v1.jsonl")
FINGERPRINTS = Path("/home/bc/data/scratch/lut_reannotate/out/segment_fingerprints.v2.jsonl")
VOCAB = REPO / "experiments/prs/EPR-036_style-vocab/vocab.v1.json"
OUT_JSONL = Path("/home/bc/data/scratch/lut_reannotate/out/style_cards.v1.jsonl")
CACHE_DIR = Path("/home/bc/data/scratch/lut_style_cards/cache")
PROGRESS = Path("/home/bc/data/scratch/lut_style_cards/progress.json")
CREDENTIALS = REPO / "databuild.prod-l8-local400k-20260812.toml"
LANE = "provider-c-lane-1"

EXPECTED_VOCAB_SHA = "354f09cb667dcb4f4c05eaaa9c522831f41d1d5936a6a3314e9cc5e3cdb29390"
EXPECTED_SHEET_PARAMS_SHA = (
    "6fb27f4789f31b8f9807181407a1373fd03497777e433a17e13aa7d10ed5c212"
)

# --- pre-registered constants -------------------------------------------------------
CARD_SCHEMA = "lut-style-card-v1"
SCHEMA_NAME = "lut_style_card_v1"
PROMPT_REVISION = "style-card-v2-candidates"
#: prompt-level tag-count guidance (not schema-enforced: OpenAI strict structured
#: outputs ignores minItems/maxItems).  Violations are counted, never rejected.
#: A field whose candidate set is smaller than the lower bound is asked for the
#: whole candidate set instead (see :func:`count_guidance_for`).
COUNT_GUIDANCE: dict[str, tuple[int, int]] = {
    "tone_tags": (2, 4),
    "palette_tags": (2, 4),
    "semantic_affinity": (1, 3),
    "semantic_risks": (0, 3),
}
#: placeholder enum value for a field with an empty candidate set: a strict
#: ``json_schema`` enum may not be empty.  It is never a vocabulary word, so the
#: post-check strips it and the field lands empty (counted as a deficit).
NO_CANDIDATE_TOKEN = "none"
#: pilot stop-gate (task card EPR-046b b).
PILOT_MAX_CONFLICTED = 2
PILOT_MAX_TOKENS_PER_CARD = 20_000
#: one re-ask when a word escapes this LUT's candidate set despite the enum.
MAX_VOCAB_RETRIES = 1
PROGRESS_EVERY = 500

# --- transport ----------------------------------------------------------------------
REASONING_EFFORT = "low"
TEMPERATURE = 0.3
MAX_OUTPUT_TOKENS = 2048
TIMEOUT_SECONDS = 420.0
ATTEMPTS = 3
IMAGE_DETAIL = "high"

# --- runtime assertion counters -----------------------------------------------------
ASSERTIONS: Counter = Counter()
_ASSERT_LOCK = threading.Lock()


def note_assertion(name: str, count: int = 1) -> None:
    with _ASSERT_LOCK:
        ASSERTIONS[name] += count


def require_assertions(*names: str) -> None:
    missing = [name for name in names if ASSERTIONS.get(name, 0) <= 0]
    if missing:
        raise SystemExit(f"[assert] pre-registered check never ran: {missing}")


# --- prompt -------------------------------------------------------------------------
SHEET_GUIDE = """\
The image is one contact sheet for a single colour LUT. Its panels, top to bottom:

1. `neutral_ramp.before` / `neutral_ramp.after_full` -- a neutral grey ramp before and
   after the LUT at full strength. Read black point, white point and any colour cast
   that appears in what used to be neutral grey.
2. `hue_chart.before` / `hue_chart.after_full` -- an 8-hue x 3-lightness colour chart
   before and after. Read per-hue shifts in hue angle, saturation and lightness.
3. Four fixed scenes, each as three panels `before | full | normalized`:
   portrait/skin, foliage/sky, architecture/daylight, night/mixed. `normalized` is the
   LUT re-scaled to a common perceptual strength, `full` is the LUT at strength 1.

Judge the LUT from the whole sheet. The four scenes are the same four photographs for
every LUT in the library, so any difference you see between `before` and `full` is the
LUT and nothing else.
"""

CANDIDATE_RULE = """\
Below are four candidate lists built for this one LUT. They are already filtered: every
word in them is a valid description of this LUT, and no word outside them is. Choosing
is your only job -- keep the words whose effect is visually obvious on the sheet and
drop the rest.

* choose a word when you can point at where you see it on the sheet;
* drop a word when the effect is there but too faint to notice, or when another word in
  the same list says the same thing more precisely;
* a shorter honest answer beats a longer one; an empty RISKS list is common;
* `semantic_affinity` words are scene categories the look suits -- keep the ones the
  look actually flatters, not every scene it would not ruin.

You may not use any word that is not printed in the list for that field.
"""

OUTPUT_RULE = """\
Answer with a single JSON object with exactly these five keys:

  "tone_tags"         : {tone_lo}-{tone_hi} words from TONE
  "palette_tags"      : {palette_lo}-{palette_hi} words from PALETTE
  "semantic_affinity" : {aff_lo}-{aff_hi} words from AFFINITY
  "semantic_risks"    : {risk_lo}-{risk_hi} words from RISKS
  "summary"           : one English sentence describing the look

Rules for `summary`: English, one sentence, no digits, no percentages, no measurement
of any kind, no LUT name, no brand or film-stock name. Describe the look, not numbers.

Never output a number anywhere in the answer. Never write a word that is not printed in
that field's candidate list. Never repeat a word inside one list. When a candidate list
is empty, answer with an empty list for that field.
"""

FIELD_LABELS: tuple[tuple[str, str], ...] = (
    ("tone_tags", "TONE"),
    ("palette_tags", "PALETTE"),
    ("semantic_affinity", "AFFINITY"),
    ("semantic_risks", "RISKS"),
)


def candidate_sets(
    features: Mapping[str, Any], vocab: Mapping[str, Any]
) -> dict[str, list[str]]:
    """Per-field words whose frozen numeric assertion is already true on this LUT.

    A word whose feature is absent from the preset is *not* a candidate (the same
    verdict :func:`style_vocab_check.check_tags` would give it).  Vocabulary order
    is preserved, so the candidate list is deterministic.
    """
    out: dict[str, list[str]] = {}
    for field in TAG_FIELDS_ORDER:
        words: list[str] = []
        for word, term in (vocab.get(field) or {}).items():
            try:
                if svc.evaluate_term(features, term):
                    words.append(word)
            except svc.StyleVocabError:
                continue
        out[field] = words
    note_assertion("candidate_sets")
    return out


def count_guidance_for(candidates: Mapping[str, Sequence[str]]) -> dict[str, tuple[int, int]]:
    """Clamp the pre-registered per-field counts to what the candidate set can supply."""
    bounds: dict[str, tuple[int, int]] = {}
    for field, (low, high) in COUNT_GUIDANCE.items():
        size = len(candidates.get(field) or [])
        bounds[field] = (min(low, size), min(high, size))
    return bounds


def candidate_deficits(candidates: Mapping[str, Sequence[str]]) -> list[str]:
    """Fields whose candidate set is empty; recorded as-is, never back-filled."""
    return [field for field in TAG_FIELDS_ORDER if not (candidates.get(field) or [])]


def _candidate_block(
    candidates: Mapping[str, Sequence[str]], bounds: Mapping[str, tuple[int, int]]
) -> str:
    lines = []
    for field, label in FIELD_LABELS:
        words = list(candidates.get(field) or [])
        low, high = bounds[field]
        if not words:
            lines.append(f"{label} (no candidate for this LUT -- answer with []):")
            lines.append("  (empty)")
            continue
        span = f"choose {low}" if low == high else f"choose {low}-{high}"
        lines.append(f"{label} ({len(words)} candidates, {span}):")
        lines.append("  " + ", ".join(words))
    return "\n".join(lines)


CAPACITY_NOTE = {
    "normal": (
        "strength_capacity = normal: the `normalized` column is this LUT scaled down to "
        "the common perceptual strength, so the `full` column is stronger than typical."
    ),
    "weak": (
        "strength_capacity = weak: this LUT cannot reach the common perceptual strength "
        "even at strength 1, so the `normalized` column is identical to `full`. Its "
        "effect is intrinsically subtle; do not compensate by choosing stronger words."
    ),
}


def build_prompt(
    preset_id: str, capacity: str, candidates: Mapping[str, Sequence[str]]
) -> str:
    bounds = count_guidance_for(candidates)
    counts = {
        "tone_lo": bounds["tone_tags"][0],
        "tone_hi": bounds["tone_tags"][1],
        "palette_lo": bounds["palette_tags"][0],
        "palette_hi": bounds["palette_tags"][1],
        "aff_lo": bounds["semantic_affinity"][0],
        "aff_hi": bounds["semantic_affinity"][1],
        "risk_lo": bounds["semantic_risks"][0],
        "risk_hi": bounds["semantic_risks"][1],
    }
    return "\n".join([
        "You are annotating one colour LUT for a retrieval index. Answer in English.",
        "",
        SHEET_GUIDE,
        CANDIDATE_RULE,
        "=== candidate words for this LUT ===",
        _candidate_block(candidates, bounds),
        "",
        "=== output ===",
        OUTPUT_RULE.format(**counts),
        "=== context (this is the whole context; there is nothing else) ===",
        f"preset_id: {preset_id}",
        CAPACITY_NOTE.get(capacity, f"strength_capacity = {capacity}"),
    ])


TAG_FIELDS_ORDER: tuple[str, ...] = (
    "tone_tags", "palette_tags", "semantic_affinity", "semantic_risks",
)


def build_schema(candidates: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    """Strict schema whose four tag enums are *this LUT's* candidate sets."""
    properties: dict[str, Any] = {}
    for field in TAG_FIELDS_ORDER:
        words = list(candidates.get(field) or [])
        properties[field] = {
            "type": "array",
            "items": {
                "type": "string",
                "enum": words or [NO_CANDIDATE_TOKEN],
            },
        }
    properties["summary"] = {"type": "string"}
    return {
        "type": "object",
        "properties": properties,
        "required": list(TAG_FIELDS_ORDER) + ["summary"],
        "additionalProperties": False,
    }


# --- provider -----------------------------------------------------------------------
def lane_credentials(path: Path, identity: str) -> tuple[str, str, str]:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    annotation = data["annotation"]
    for row in annotation["external_endpoints"]:
        if str(row.get("id")) == identity:
            return (
                str(row["base_url"]).rstrip("/"), str(row["api_key"]),
                str(annotation["external_model"]),
            )
    raise KeyError(f"endpoint {identity!r} is absent from {path}")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def request_hash(
    *, model: str, prompt: str, image_sha: str, schema: Mapping[str, Any], stage: str
) -> str:
    identity = {
        "prompt_revision": PROMPT_REVISION,
        "stage": stage,
        "model": model,
        "prompt_sha256": sha256_text(prompt),
        "image_sha256": image_sha,
        "image_detail": IMAGE_DETAIL,
        "schema_sha256": sha256_text(
            json.dumps(schema, ensure_ascii=False, sort_keys=True)
        ),
        "behavior": {
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "reasoning_effort": REASONING_EFFORT,
        },
    }
    return sha256_text(json.dumps(identity, ensure_ascii=False, sort_keys=True))


class ResponseCache:
    """Durable per-request cache: one JSON file per canonical request hash."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def path(self, digest: str) -> Path:
        return self.root / digest[:2] / f"{digest}.json"

    def get(self, digest: str) -> dict[str, Any] | None:
        path = self.path(digest)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        with self._lock:
            self.hits += 1
        return payload

    def put(self, digest: str, payload: Mapping[str, Any]) -> None:
        path = self.path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)
        with self._lock:
            self.misses += 1


class Provider:
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self._local = threading.local()

    def client(self) -> Any:
        client = getattr(self._local, "client", None)
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                base_url=self.base_url, api_key=self.api_key, max_retries=0,
                timeout=TIMEOUT_SECONDS,
            )
            self._local.client = client
        return client

    def ask(self, prompt: str, image_b64: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(1, ATTEMPTS + 1):
            try:
                response = self.client().responses.create(
                    model=self.model,
                    input=[{"role": "user", "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image",
                         "image_url": f"data:image/png;base64,{image_b64}",
                         "detail": IMAGE_DETAIL},
                    ]}],
                    stream=False,
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    store=False,
                    reasoning={"effort": REASONING_EFFORT},
                    text={"format": {
                        "type": "json_schema", "name": SCHEMA_NAME,
                        "strict": True, "schema": dict(schema),
                    }},
                    timeout=TIMEOUT_SECONDS,
                )
                text = str(getattr(response, "output_text", "") or "")
                if not text:
                    raise ValueError("responses_output_empty")
                usage = getattr(response, "usage", None)
                return {
                    "parsed": json.loads(text),
                    "usage": {
                        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
                    },
                    "transport_attempts": attempt,
                }
            except Exception as exc:  # retried; the last one is re-raised
                last = exc
                if attempt < ATTEMPTS:
                    time.sleep(min(2 ** (attempt - 1), 20))
        raise RuntimeError(f"style card request failed after {ATTEMPTS} attempts: {last}")


# --- checking -----------------------------------------------------------------------
def normalise_answer(parsed: Mapping[str, Any]) -> dict[str, Any]:
    """Coerce the model answer into the card shape, de-duplicating each list."""
    answer: dict[str, Any] = {}
    for field in TAG_FIELDS_ORDER:
        seen: list[str] = []
        for value in parsed.get(field) or []:
            word = str(value).strip()
            if word and word not in seen:
                seen.append(word)
        answer[field] = seen
    answer["summary"] = str(parsed.get("summary", "") or "").strip()
    return answer


def all_tags(answer: Mapping[str, Any]) -> list[str]:
    tags: list[str] = []
    for field in TAG_FIELDS_ORDER:
        tags.extend(answer.get(field) or [])
    return tags


def out_of_candidates(
    answer: Mapping[str, Any], candidates: Mapping[str, Sequence[str]]
) -> list[str]:
    """Words absent from this LUT's candidate list for their field (post-check gate)."""
    bad: list[str] = []
    for field in TAG_FIELDS_ORDER:
        legal = set(candidates.get(field) or [])
        bad.extend(word for word in (answer.get(field) or []) if word not in legal)
    return bad


def assert_subset(
    answer: Mapping[str, Any], candidates: Mapping[str, Sequence[str]], preset_id: str
) -> None:
    """Pre-write guard: no word may reach disk that was not offered as a candidate."""
    bad = out_of_candidates(answer, candidates)
    if bad:
        raise SystemExit(
            f"[assert] {preset_id}: selected words outside the candidate set: {bad}"
        )
    note_assertion("subset_guard")


def count_violations(
    answer: Mapping[str, Any], candidates: Mapping[str, Sequence[str]] | None = None
) -> list[str]:
    bounds = (
        count_guidance_for(candidates) if candidates is not None
        else dict(COUNT_GUIDANCE)
    )
    problems = []
    for field in TAG_FIELDS_ORDER:
        low, high = bounds[field]
        size = len(answer.get(field) or [])
        if not low <= size <= high:
            problems.append(f"{field}={size} outside [{low},{high}]")
    return problems


def check_card(
    answer: Mapping[str, Any], features: Mapping[str, Any], vocab: Mapping[str, Any]
) -> dict[str, Any]:
    """The only path onto disk: every card is judged by ``style_vocab_check``."""
    rows = svc.check_tags(features, all_tags(answer), vocab)
    note_assertion("check_tags")
    accepted = svc.accepted_tags(rows)
    quality = svc.annotation_quality(rows)
    note_assertion("annotation_quality")
    return {
        "rows": rows,
        "accepted_tags": accepted,
        "rejected_tags": [
            {k: row[k] for k in ("tag", "tag_field", "assertion", "value", "reason")}
            for row in rows if not row["pass"]
        ],
        "annotation_quality": quality,
    }


# --- preflight ----------------------------------------------------------------------
def preflight(vocab_path: Path, sheet_params: Path) -> dict[str, Any]:
    vocab = svc.load_vocab(vocab_path)
    actual = svc.vocab_sha256(vocab)
    registered = (vocab.get("meta") or {}).get("sha256")
    if actual != EXPECTED_VOCAB_SHA or registered != EXPECTED_VOCAB_SHA:
        raise SystemExit(
            f"[assert] vocab sha mismatch: file={registered} recomputed={actual} "
            f"expected={EXPECTED_VOCAB_SHA}"
        )
    note_assertion("vocab_sha256")

    params = json.loads(sheet_params.read_text(encoding="utf-8"))
    if str(params.get("params_sha256")) != EXPECTED_SHEET_PARAMS_SHA:
        raise SystemExit(
            f"[assert] contact-sheet params_sha256 {params.get('params_sha256')} != "
            f"{EXPECTED_SHEET_PARAMS_SHA}"
        )
    note_assertion("sheet_params_sha256")

    self_test(vocab)
    return vocab


def self_test(vocab: Mapping[str, Any]) -> None:
    """Negative self-test: the checker must fail the three failure modes it claims."""
    features = {"segments.shadows.dL": -20.0, "scene_affinity": []}
    cases = [
        ("deep-shadows", True, "ok"),
        ("lifted-shadows", False, "assertion-false"),
        ("not-a-real-word", False, "unknown-word"),
        ("portrait", False, "assertion-false"),
        ("high-contrast", False, None),  # feature absent -> StyleVocabError text
    ]
    rows = svc.check_tags(features, [tag for tag, _, _ in cases], vocab)
    for row, (tag, expect_pass, expect_reason) in zip(rows, cases):
        if row["tag"] != tag or bool(row["pass"]) is not expect_pass:
            raise SystemExit(f"[assert] checker self-test failed on {tag}: {row}")
        if expect_reason is not None and row["reason"] != expect_reason:
            raise SystemExit(
                f"[assert] checker self-test reason for {tag}: {row['reason']!r} != "
                f"{expect_reason!r}"
            )
    if svc.accepted_tags(rows) != ["deep-shadows"]:
        raise SystemExit(f"[assert] accepted_tags self-test: {svc.accepted_tags(rows)}")
    if svc.annotation_quality(rows) != "conflicted":
        raise SystemExit("[assert] annotation_quality self-test: expected conflicted")
    if svc.annotation_quality(rows[:1]) != "clean":
        raise SystemExit("[assert] annotation_quality self-test: expected clean")

    # candidate-set self-test: a true word is offered, a false word is not, a word
    # whose feature is missing is not, and the subset guard fires on an intruder.
    cands = candidate_sets(features, vocab)
    if "deep-shadows" not in cands["tone_tags"]:
        raise SystemExit("[assert] candidate self-test: deep-shadows must be offered")
    if "lifted-shadows" in cands["tone_tags"]:
        raise SystemExit("[assert] candidate self-test: lifted-shadows must not be")
    if "high-contrast" in cands["tone_tags"]:
        raise SystemExit(
            "[assert] candidate self-test: a word with a missing feature must not be"
        )
    if cands["semantic_affinity"]:
        raise SystemExit(
            "[assert] candidate self-test: empty scene_affinity offers no affinity word"
        )
    assert_subset({"tone_tags": ["deep-shadows"]}, cands, "<self-test>")
    try:
        assert_subset({"tone_tags": ["lifted-shadows"]}, cands, "<self-test>")
    except SystemExit:
        pass
    else:
        raise SystemExit("[assert] subset guard self-test: intruder was not caught")
    note_assertion("checker_self_test")


# --- catalog ------------------------------------------------------------------------
def load_catalog(sheets_dir: Path, sidecar_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sheet in sorted(sheets_dir.glob("*.png")):
        preset_id = sheet.stem
        sidecar = sidecar_dir / f"{preset_id}.json"
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if str(payload.get("params_sha256")) != EXPECTED_SHEET_PARAMS_SHA:
            raise SystemExit(
                f"[assert] {preset_id}: sidecar params_sha256 != frozen sheet params"
            )
        rows.append({
            "preset_id": preset_id,
            "sheet_path": sheet,
            "strength_capacity": str(payload.get("strength_capacity", "")),
            "sheet_png_sha256": str((payload.get("sheet") or {}).get("png_sha256", "")),
        })
    return rows


def load_done(path: Path) -> dict[str, dict[str, Any]]:
    """Last record wins, so the append-only ledger is resumable and pass-2 safe."""
    done: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[str(row["preset_id"])] = row
    return done


# --- annotation ---------------------------------------------------------------------
def annotate_one(
    entry: Mapping[str, Any],
    *,
    vocab: Mapping[str, Any],
    features: Mapping[str, Any],
    provider: Provider,
    cache: ResponseCache,
    stage: str = "pass1",
) -> dict[str, Any]:
    preset_id = str(entry["preset_id"])
    image_bytes = Path(entry["sheet_path"]).read_bytes()
    image_sha = sha256_bytes(image_bytes)
    if entry.get("sheet_png_sha256") and image_sha != entry["sheet_png_sha256"]:
        raise SystemExit(f"[assert] {preset_id}: sheet PNG sha != sidecar png_sha256")
    image_b64 = base64.b64encode(image_bytes).decode("ascii")

    candidates = candidate_sets(features, vocab)
    schema = build_schema(candidates)
    prompt = build_prompt(preset_id, str(entry["strength_capacity"]), candidates)

    usage_total = {"input_tokens": 0, "output_tokens": 0}
    calls = 0
    cached_calls = 0
    vocab_retries = 0
    answer: dict[str, Any] = {}
    for attempt in range(MAX_VOCAB_RETRIES + 1):
        message = prompt
        if attempt:
            message = (
                prompt
                + "\n\n=== your previous answer used words outside the candidate lists ===\n"
                + json.dumps(answer, ensure_ascii=False)
                + "\n\nEvery word must appear verbatim in that field's candidate list "
                  "above. Answer again."
            )
        digest = request_hash(
            model=provider.model, prompt=message, image_sha=image_sha,
            schema=schema, stage=f"{stage}#{attempt}",
        )
        hit = cache.get(digest)
        if hit is None:
            reply = provider.ask(message, image_b64, schema)
            cache.put(digest, {
                "request_hash": digest, "preset_id": preset_id, "stage": stage,
                "attempt": attempt, "model": provider.model,
                "prompt_sha256": sha256_text(message), "image_sha256": image_sha,
                "parsed": reply["parsed"], "usage": reply["usage"],
            })
            calls += 1
            usage_total["input_tokens"] += reply["usage"]["input_tokens"]
            usage_total["output_tokens"] += reply["usage"]["output_tokens"]
            parsed = reply["parsed"]
        else:
            cached_calls += 1
            parsed = hit["parsed"]
        answer = normalise_answer(parsed)
        bad = out_of_candidates(answer, candidates)
        if not bad:
            break
        vocab_retries += 1
        if attempt == MAX_VOCAB_RETRIES:
            for field in TAG_FIELDS_ORDER:
                legal = set(candidates[field])
                answer[field] = [w for w in answer[field] if w in legal]

    assert_subset(answer, candidates, preset_id)
    verdict = check_card(answer, features, vocab)
    record = {
        "schema": CARD_SCHEMA,
        "preset_id": preset_id,
        **{field: answer[field] for field in TAG_FIELDS_ORDER},
        "summary": answer["summary"],
        "annotation_quality": verdict["annotation_quality"],
        "accepted_tags": verdict["accepted_tags"],
        "rejected_tags": verdict["rejected_tags"],
        "candidates": {field: list(candidates[field]) for field in TAG_FIELDS_ORDER},
        "candidate_counts": {
            field: len(candidates[field]) for field in TAG_FIELDS_ORDER
        },
        "candidate_deficit": candidate_deficits(candidates),
        "selected_counts": {
            field: len(answer[field]) for field in TAG_FIELDS_ORDER
        },
        "count_violations": count_violations(answer, candidates),
        "vocab_sha": EXPECTED_VOCAB_SHA,
        "sheet_params_sha": EXPECTED_SHEET_PARAMS_SHA,
        "sheet_png_sha256": image_sha,
        "strength_capacity": str(entry["strength_capacity"]),
        "prompt_revision": PROMPT_REVISION,
        "model": provider.model,
        "endpoint": provider.base_url,
        "lane": LANE,
        "pass": 1,
        "vocab_retries": vocab_retries,
        "api_calls": calls,
        "cache_hits": cached_calls,
        "usage": usage_total,
    }
    return record


# --- driver -----------------------------------------------------------------------
class Ledger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._handle = self.path.open("a", encoding="utf-8")

    def append(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock:
            self._handle.write(line)
            self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def run_stage(
    entries: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    vocab: Mapping[str, Any],
    library: Mapping[str, Any],
    provider: Provider,
    cache: ResponseCache,
    ledger: Ledger,
    workers: int,
    progress_path: Path,
    on_record: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    lock = threading.Lock()
    started = time.time()

    def work(entry: Mapping[str, Any]) -> None:
        preset_id = str(entry["preset_id"])
        try:
            record = annotate_one(
                entry, vocab=vocab, features=library[preset_id],
                provider=provider, cache=cache, stage=stage,
            )
        except SystemExit:
            raise
        except Exception as exc:
            with lock:
                failures.append({"preset_id": preset_id, "error": repr(exc)})
            print(f"[fail] {stage} {preset_id}: {exc!r}", flush=True)
            return
        ledger.append(record)
        with lock:
            records.append(record)
            done = len(records)
        if on_record is not None:
            on_record(record)
        if done % PROGRESS_EVERY == 0 or done == len(entries):
            snapshot = summarise(records)
            snapshot.update({
                "stage": stage, "done": done, "planned": len(entries),
                "failed": len(failures), "elapsed_s": round(time.time() - started, 1),
                "assertions": dict(ASSERTIONS),
                "cache": {"hits": cache.hits, "misses": cache.misses},
            })
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=1) + "\n",
                encoding="utf-8",
            )
            print(f"[progress] {json.dumps(snapshot, ensure_ascii=False)}", flush=True)

    if workers <= 1:
        for entry in entries:
            work(entry)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, entries))
    return records, failures


def summarise(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    records = list(records)
    quality = Counter(str(r["annotation_quality"]) for r in records)
    word_hits: Counter = Counter()
    rejected_word: Counter = Counter()
    rejected_by_field: Counter = Counter()
    proposed_by_field: Counter = Counter()
    reject_reason: Counter = Counter()
    empty_field: Counter = Counter()
    candidate_by_field: Counter = Counter()
    deficit_by_field: Counter = Counter()
    selected_hist: dict[str, Counter] = {field: Counter() for field in TAG_FIELDS_ORDER}
    input_tokens = sum(int((r.get("usage") or {}).get("input_tokens", 0)) for r in records)
    output_tokens = sum(
        int((r.get("usage") or {}).get("output_tokens", 0)) for r in records
    )
    per_card_tokens = [
        int((r.get("usage") or {}).get("input_tokens", 0))
        + int((r.get("usage") or {}).get("output_tokens", 0))
        for r in records
    ]
    billed = [value for value in per_card_tokens if value > 0]
    n_rejected = 0
    empty_cards = 0
    for record in records:
        counts = record.get("candidate_counts") or {}
        for field in TAG_FIELDS_ORDER:
            candidate_by_field[field] += int(counts.get(field, 0))
        for field in record.get("candidate_deficit") or []:
            deficit_by_field[str(field)] += 1
        for field in TAG_FIELDS_ORDER:
            selected_hist[field][len(record.get(field) or [])] += 1
        for field in TAG_FIELDS_ORDER:
            words = record.get(field) or []
            proposed_by_field[field] += len(words)
            if not words:
                empty_field[field] += 1
        if not all_tags(record):
            empty_cards += 1
        for word in record.get("accepted_tags") or []:
            word_hits[word] += 1
        for row in record.get("rejected_tags") or []:
            n_rejected += 1
            rejected_by_field[str(row.get("tag_field"))] += 1
            rejected_word[str(row.get("tag"))] += 1
            reject_reason[str(row.get("reason"))] += 1
    return {
        "n": len(records),
        "clean": quality.get("clean", 0),
        "conflicted": quality.get("conflicted", 0),
        "n_tags_proposed": sum(proposed_by_field.values()),
        "n_tags_rejected": n_rejected,
        "mean_rejected_per_card": round(n_rejected / len(records), 4) if records else 0.0,
        "proposed_by_field": dict(proposed_by_field),
        "candidates_by_field": dict(candidate_by_field),
        "mean_candidates_by_field": {
            field: round(candidate_by_field[field] / len(records), 4)
            for field in TAG_FIELDS_ORDER
        } if records else {},
        "candidate_deficit_by_field": dict(deficit_by_field),
        "selected_count_hist": {
            field: dict(sorted(selected_hist[field].items()))
            for field in TAG_FIELDS_ORDER
        },
        "rejected_by_field": dict(rejected_by_field),
        "reject_reason": dict(reject_reason),
        "empty_field_cards": dict(empty_field),
        "empty_cards": empty_cards,
        "accepted_word_hits": dict(word_hits.most_common()),
        "rejected_word_hits": dict(rejected_word.most_common()),
        "vocab_retries": sum(int(r.get("vocab_retries", 0)) for r in records),
        "count_violations": sum(
            1 for r in records if r.get("count_violations")
        ),
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens,
            "per_lut": round((input_tokens + output_tokens) / len(records), 1)
            if records else 0.0,
            "per_billed_call": round(sum(billed) / len(billed), 1) if billed else 0.0,
            "max_per_card": max(per_card_tokens) if per_card_tokens else 0,
        },
        "api_calls": sum(int(r.get("api_calls", 0)) for r in records),
        "cache_hits": sum(int(r.get("cache_hits", 0)) for r in records),
    }


def verify_ledger(
    records: Iterable[Mapping[str, Any]],
    library: Mapping[str, Any],
    vocab: Mapping[str, Any],
) -> dict[str, int]:
    """Re-run the checker over every finished card and assert the stored labels.

    This is the run-time wiring of the pre-registered criterion: it fires on every
    run, including a pure resume where no new card was annotated, so a ledger whose
    ``annotation_quality`` was written by anything other than ``check_tags`` cannot
    survive a second run.
    """
    checked = 0
    for record in records:
        preset_id = str(record["preset_id"])
        recomputed = candidate_sets(library[preset_id], vocab)
        stored = record.get("candidates")
        if stored is not None and {
            field: list(stored.get(field) or []) for field in TAG_FIELDS_ORDER
        } != recomputed:
            raise SystemExit(f"[assert] {preset_id}: stored candidate set drifted")
        assert_subset(record, recomputed, preset_id)
        verdict = check_card(record, library[preset_id], vocab)
        checked += 1
        if verdict["annotation_quality"] != record.get("annotation_quality"):
            raise SystemExit(
                f"[assert] {preset_id}: stored annotation_quality "
                f"{record.get('annotation_quality')!r} != recomputed "
                f"{verdict['annotation_quality']!r}"
            )
        if verdict["accepted_tags"] != list(record.get("accepted_tags") or []):
            raise SystemExit(f"[assert] {preset_id}: stored accepted_tags drifted")
    note_assertion("verify_ledger")
    return {"verified": checked}


def rewrite_sorted(path: Path) -> tuple[int, str]:
    """Collapse the append-only ledger to one final row per preset_id, sorted."""
    done = load_done(path)
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_bytes(path.read_bytes())
    lines = [
        json.dumps(done[key], ensure_ascii=False, sort_keys=True) + "\n"
        for key in sorted(done)
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(lines), encoding="utf-8")
    tmp.replace(path)
    return len(done), sha256_bytes(path.read_bytes())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_JSONL)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--progress", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--sheets", type=Path, default=SHEETS_DIR)
    parser.add_argument("--sidecar", type=Path, default=SIDECAR_DIR)
    parser.add_argument("--sheet-params", type=Path, default=SHEET_PARAMS)
    parser.add_argument("--vocab", type=Path, default=VOCAB)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--fingerprints", type=Path, default=FINGERPRINTS)
    parser.add_argument("--lane", default=LANE)
    parser.add_argument("--limit", type=int, default=0,
                        help="annotate only the first N preset_ids (lexicographic)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pilot-gate", action="store_true",
                        help="apply the EPR-046b task-card b stop-gate after the run")
    parser.add_argument("--dump-prompt", type=Path, default=None,
                        help="write the prompt of the first preset and exit")
    args = parser.parse_args(argv)

    vocab = preflight(args.vocab, args.sheet_params)
    catalog = load_catalog(args.sheets, args.sidecar)
    if args.limit:
        catalog = catalog[: args.limit]

    library = svc.load_library(args.annotations, args.fingerprints)
    missing = [row["preset_id"] for row in catalog if row["preset_id"] not in library]
    if missing:
        raise SystemExit(f"[assert] {len(missing)} sheets have no numeric features")

    if args.dump_prompt is not None:
        entry = catalog[0]
        preset_id = str(entry["preset_id"])
        candidates = candidate_sets(library[preset_id], vocab)
        args.dump_prompt.parent.mkdir(parents=True, exist_ok=True)
        args.dump_prompt.write_text(
            build_prompt(preset_id, str(entry["strength_capacity"]), candidates),
            encoding="utf-8",
        )
        print(f"[prompt] written to {args.dump_prompt}")
        return 0

    base_url, api_key, model = lane_credentials(CREDENTIALS, args.lane)
    provider = Provider(base_url, api_key, model)
    cache = ResponseCache(args.cache_dir)
    progress_path = args.progress or (args.out.parent / f"{args.out.stem}.progress.json")
    ledger = Ledger(args.out)
    report: dict[str, Any] = {
        "schema": CARD_SCHEMA,
        "prompt_revision": PROMPT_REVISION,
        "vocab_sha": EXPECTED_VOCAB_SHA,
        "sheet_params_sha": EXPECTED_SHEET_PARAMS_SHA,
        "model": model, "endpoint": base_url, "lane": args.lane,
        "behavior": {
            "temperature": TEMPERATURE, "reasoning_effort": REASONING_EFFORT,
            "max_output_tokens": MAX_OUTPUT_TOKENS, "image_detail": IMAGE_DETAIL,
            "transport": "nonstream", "attempts": ATTEMPTS,
        },
        "count_guidance": {k: list(v) for k, v in COUNT_GUIDANCE.items()},
        "planned": len(catalog),
    }

    try:
        done = load_done(args.out)
        pending = [row for row in catalog if row["preset_id"] not in done]
        print(f"[plan] catalog={len(catalog)} pending={len(pending)} "
              f"resumed={len(catalog) - len(pending)} workers={args.workers}",
              flush=True)
        fresh, failures = run_stage(
            pending, stage="pass1", vocab=vocab, library=library,
            provider=provider, cache=cache, ledger=ledger,
            workers=args.workers, progress_path=progress_path,
        )
        done = load_done(args.out)
        rows = [done[row["preset_id"]] for row in catalog if row["preset_id"] in done]
        report["pass1"] = summarise(rows)
        report["pass1"]["fresh_this_run"] = len(fresh)
        report["pass1"]["failed"] = len(failures)
        report["pass1"]["failures"] = failures
        _flush_report(args, report)
        if args.pilot_gate:
            gate = report["pass1"]
            max_tokens = gate["tokens"]["max_per_card"]
            stop = (
                gate["conflicted"] > PILOT_MAX_CONFLICTED
                or max_tokens > PILOT_MAX_TOKENS_PER_CARD
            )
            report["pilot_gate"] = {
                "max_conflicted": PILOT_MAX_CONFLICTED,
                "max_tokens_per_card": PILOT_MAX_TOKENS_PER_CARD,
                "conflicted": gate["conflicted"],
                "observed_max_tokens_per_card": max_tokens,
                "stop": stop,
            }
            _flush_report(args, report)
            print(f"[gate] {json.dumps(report['pilot_gate'], ensure_ascii=False)}",
                  flush=True)
    finally:
        ledger.close()

    n_rows, digest = rewrite_sorted(args.out)
    final = load_done(args.out)
    report["verify"] = verify_ledger(final.values(), library, vocab)
    report["final"] = summarise(final.values())
    report["final"]["rows"] = n_rows
    report["final"]["sha256"] = digest
    report["final"]["by_pass"] = dict(
        Counter(int(r.get("pass", 1)) for r in final.values())
    )
    report["assertions"] = dict(ASSERTIONS)
    report["cache"] = {"hits": cache.hits, "misses": cache.misses}
    _flush_report(args, report)

    require_assertions(
        "vocab_sha256", "sheet_params_sha256", "checker_self_test",
        "check_tags", "annotation_quality", "verify_ledger",
        "candidate_sets", "subset_guard",
    )
    print(json.dumps({
        "out": str(args.out), "rows": n_rows, "sha256": digest,
        "final": {k: report["final"][k] for k in
                  ("n", "clean", "conflicted", "n_tags_proposed", "n_tags_rejected",
                   "tokens")},
        "assertions": dict(ASSERTIONS),
    }, ensure_ascii=False, indent=1))
    return 0


def _flush_report(args: argparse.Namespace, report: Mapping[str, Any]) -> None:
    path = args.report or (args.out.parent / f"{args.out.stem}.report.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
