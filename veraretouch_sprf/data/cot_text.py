"""EPR-051 / SFT+ADAPT -- frozen instruction template + CoT target serialisation.

Everything in this file is a FROZEN口径 object.  `template_sha256()` hashes the
instruction text, the per-step scaffold, the special-token names and the
serialiser source together, so a silent wording drift changes the sha.

口径 note (recorded, not silently decided):
  The annotation prompt showed the labeller THREE images (CURRENT=x0, TARGET=y,
  and a per-step breakdown sheet) and asked for a grade written FROM the current
  photo TO the target look.  The student sees ONLY y (the graded / degraded
  image) plus this fixed instruction -- that is the distillation setting the
  task card prescribes.  The instruction below is worded so the task is
  well-posed from y alone: "reconstruct the six-move grade that produced this
  photo", with each move's observation describing the state BEFORE that move.

Special tokens: one per stage, appended at the END of stage m's text.  Causal
attention means an end-of-segment position has attended over the whole segment,
whereas a segment-initial token has not -- the readout wants the former.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

STAGE_TOKENS = [f"<vr_stage_{m}>" for m in range(1, 7)]

INSTRUCTION = (
    "This photograph carries a finished colour grade that was built in six "
    "moves, applied one after another. Reconstruct that grade from this image "
    "alone. The six moves run in a fixed arc: first the global grade over the "
    "whole frame, then the masked tonal and hue moves, and finally the subject "
    "move. For every move, state the problem the move addresses at that point "
    "of the grade, the mask it works in, and the correction it applies with the "
    "hue bands and the direction and size of each shift."
)

STEP_SCAFFOLD = "Move {m}.\nObservation: {observation}\nMask: {mask}\nAdjustment: {adjustment}\n"


def target_text(record: dict) -> str:
    """6 segments, each closed by its own stage token."""
    cot = record["answer"]["cot"]
    if len(cot) != 6:
        raise ValueError(f"expected 6 cot steps, got {len(cot)}")
    out = []
    for m, c in enumerate(cot, 1):
        if c.get("step") != m:
            raise ValueError(f"step index {c.get('step')} != {m}")
        seg = STEP_SCAFFOLD.format(m=m,
                                   observation=c["observation"].strip(),
                                   mask=c["mask"].strip(),
                                   adjustment=c["adjustment"].strip())
        out.append(seg + STAGE_TOKENS[m - 1])
    return "\n".join(out)


def target_segments(record: dict) -> list[str]:
    """The 6 segment bodies, WITHOUT their closing stage token.

    target_text(record) == "\n".join(seg + STAGE_TOKENS[m-1] for m, seg in ...).
    Kept separate so a span-pool readout can build token ranges by construction
    instead of searching for boundaries in a flat id list.
    """
    cot = record["answer"]["cot"]
    if len(cot) != 6:
        raise ValueError(f"expected 6 cot steps, got {len(cot)}")
    out = []
    for m, c in enumerate(cot, 1):
        if c.get("step") != m:
            raise ValueError(f"step index {c.get('step')} != {m}")
        out.append(STEP_SCAFFOLD.format(m=m,
                                        observation=c["observation"].strip(),
                                        mask=c["mask"].strip(),
                                        adjustment=c["adjustment"].strip()))
    return out


# --------------------------------------------------------------------------- #
# per-sample instruction (2026-09-04 ruling): the prompt carries the sample's own
# editing instruction instead of one fixed sentence.  Tier is chosen by
# sha1(salt + ':' + key), so it is per-sample deterministic, replayable, and
# identical across epochs and across train/eval.
# --------------------------------------------------------------------------- #
INSTRUCTION_SALT = "epr051-inst-mix-v1"
INSTRUCTION_MIX = (("instruction_long", 70), ("instruction_medium", 20),
                   ("instruction_short", 10))


def instruction_tier(key: str, salt: str = INSTRUCTION_SALT) -> str:
    b = int(hashlib.sha1(f"{salt}:{key}".encode()).hexdigest()[:8], 16) % 100
    acc = 0
    for name, pct in INSTRUCTION_MIX:
        acc += pct
        if b < acc:
            return name
    return INSTRUCTION_MIX[-1][0]


def instruction_for(record: dict, key: str | None = None,
                    salt: str = INSTRUCTION_SALT) -> tuple[str, str]:
    """-> (instruction text, tier name).  Raises if the record lacks the field."""
    k = key or record["key"]
    tier = instruction_tier(k, salt)
    txt = (record.get("answer") or {}).get(tier)
    if not txt or not str(txt).strip():
        raise ValueError(f"{k}: record has no {tier!r}")
    return str(txt).strip(), tier


def instruction_mix_counts(records, salt: str = INSTRUCTION_SALT) -> dict:
    c = {n: 0 for n, _ in INSTRUCTION_MIX}
    for r in records:
        c[instruction_tier(r["key"], salt)] += 1
    tot = max(1, sum(c.values()))
    return dict(counts=c, pct={k: round(100.0 * v / tot, 4) for k, v in c.items()},
                n=tot, salt=salt, mix=list(INSTRUCTION_MIX))


def parse_target_text(text: str) -> dict:
    """Inverse of target_text, tolerant enough to score a generated sample.

    Returns {'ok': bool, 'n_segments': int, 'missing_tokens': [...],
             'steps': [{'observation','mask','adjustment'} or None] * 6}
    """
    steps: list = [None] * 6
    missing = []
    for m in range(1, 7):
        tok = STAGE_TOKENS[m - 1]
        if tok not in text:
            missing.append(tok)
    # split on the stage tokens in order
    cur = text
    for m in range(1, 7):
        tok = STAGE_TOKENS[m - 1]
        if tok not in cur:
            continue
        seg, cur = cur.split(tok, 1)
        d = {}
        for field, label in (("observation", "Observation:"), ("mask", "Mask:"),
                             ("adjustment", "Adjustment:")):
            if label in seg:
                tail = seg.split(label, 1)[1]
                for nxt in ("Observation:", "Mask:", "Adjustment:", "Move "):
                    if nxt in tail:
                        tail = tail.split(nxt, 1)[0]
                d[field] = tail.strip()
        if len(d) == 3:
            steps[m - 1] = d
    n_ok = sum(1 for s in steps if s is not None)
    return dict(ok=(n_ok == 6 and not missing), n_segments=n_ok,
                missing_tokens=missing, steps=steps)


def template_sha256() -> str:
    payload = json.dumps(dict(instruction=INSTRUCTION, scaffold=STEP_SCAFFOLD,
                              tokens=STAGE_TOKENS,
                              source=Path(__file__).read_text()),
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


if __name__ == "__main__":
    import sys
    rec = json.loads(open("/home/bc/data/runs/epr051_vlmsft/snap_sft1/records.jsonl").readline())
    t = target_text(rec)
    print(t[:1200])
    print("...")
    print("template_sha256 =", template_sha256())
    print("chars =", len(t))
    p = parse_target_text(t)
    print("roundtrip ok =", p["ok"], "n_segments =", p["n_segments"])
    a = rec["answer"]["cot"]
    same = all(p["steps"][i]["observation"] == a[i]["observation"].strip()
               and p["steps"][i]["mask"] == a[i]["mask"].strip()
               and p["steps"][i]["adjustment"] == a[i]["adjustment"].strip()
               for i in range(6))
    print("roundtrip field-exact =", same)
