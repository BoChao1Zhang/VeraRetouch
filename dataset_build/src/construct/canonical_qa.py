"""OneAlign ranking, deterministic veto, and top-2 SFT winner selection."""
from __future__ import annotations

import math
import queue
import threading
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from dataset_build.tools.archive_reader import open_image


SFT_THRESHOLD = 0.50


class QaError(RuntimeError):
    """OneAlign preflight or ranking failed."""


class Scorer(Protocol):
    def score(self, path: str) -> float | None: ...


class OneAlignScorer:
    """Explicit OneAlign-only scorer; environment variables cannot select another model."""

    def __init__(self, device: str = "cuda:0"):
        from dataset_build.source_qa.iaa import OneAlignRunner

        self.device = device
        self.runner = OneAlignRunner(device=device)

    @classmethod
    def create(cls, device: str = "cuda:0") -> "OneAlignScorer":
        scorer = cls(device)
        try:
            scorer.runner.load()
        except Exception as exc:  # noqa: BLE001
            raise QaError(f"OneAlign preflight failed: {type(exc).__name__}: {exc}") from exc
        return scorer

    def score(self, path: str) -> float | None:
        try:
            result = self.runner.score_path(path)
        except Exception as exc:  # noqa: BLE001 - convert model/runtime errors at the QA boundary
            raise QaError(
                f"OneAlign inference failed: {type(exc).__name__}: {exc}"
            ) from exc
        value = result.get("onealign", result.get("iaa_mixed"))
        return _validated(value)

    def score_batch(self, paths: list[str]) -> list[float | None]:
        """Score a chunk of images in one forward pass with score()'s error boundary.

        The returned length is checked by ``_score_paths``, which is the boundary
        every ``Scorer`` crosses.
        """
        if not paths:
            return []
        try:
            values = self.runner.score_paths(list(paths))
        except Exception as exc:  # noqa: BLE001 - convert model/runtime errors at the QA boundary
            raise QaError(
                f"OneAlign inference failed: {type(exc).__name__}: {exc}"
            ) from exc
        return [_validated(value) for value in values]


class OneAlignScorerPool:
    """Bounded pool of independent OneAlign instances on one QA device."""

    def __init__(self, scorers: tuple[OneAlignScorer, ...]):
        if not scorers:
            raise ValueError("OneAlign scorer pool cannot be empty")
        self.device = scorers[0].device
        self._scorers = scorers
        self._available: queue.Queue[OneAlignScorer] = queue.Queue(len(scorers))
        self._preflight_lock = threading.Lock()
        self._preflight_warmed = False
        for scorer in scorers:
            if scorer.device != self.device:
                raise ValueError("OneAlign scorer pool devices must match")
            self._available.put_nowait(scorer)

    @classmethod
    def create(
        cls, device: str = "cuda:0", *, instances: int = 2
    ) -> "OneAlignScorerPool":
        if instances < 1:
            raise ValueError("OneAlign scorer instances must be positive")
        return cls(tuple(OneAlignScorer.create(device) for _ in range(instances)))

    def score(self, path: str) -> float | None:
        # Startup preflight warms every copy so the first real concurrent QA pair
        # does not inherit model-specific kernel initialization.
        if not self._preflight_warmed:
            with self._preflight_lock:
                if not self._preflight_warmed:
                    values = [scorer.score(path) for scorer in self._scorers]
                    self._preflight_warmed = True
                    return values[0]
        scorer = self._available.get()
        try:
            return scorer.score(path)
        finally:
            self._available.put_nowait(scorer)

    def score_batch(self, paths: list[str]) -> list[float | None]:
        scorer = self._available.get()
        try:
            return scorer.score_batch(paths)
        finally:
            self._available.put_nowait(scorer)


def _validated(value: Any) -> float:
    if value is None:
        raise QaError("OneAlign inference returned no score")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 100.0:
        raise QaError(f"OneAlign inference returned invalid score: {score!r}")
    return score


@dataclass(frozen=True, slots=True)
class RankedCandidates:
    candidates: tuple[dict[str, Any], ...]
    winner_ids: tuple[str, ...]
    source_score: float | None
    # OneAlign points between the best and second-best *scored* candidate, and
    # what the margin policy made of it.  ``None`` margin means the group had
    # only one scored candidate; ``None`` confidence means no candidate cleared
    # ``SFT_THRESHOLD``, so there was no selection for the policy to judge.
    winner_margin: float | None = None
    winner_confidence: str | None = None


def _stats(path: str) -> dict[str, float]:
    image = open_image(path)
    image.thumbnail((256, 256))
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    rg = red - green
    yb = 0.5 * (red + green) - blue
    colorfulness = float(
        np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())
    )
    luma = 0.299 * red + 0.587 * green + 0.114 * blue
    return {
        "colorfulness": colorfulness,
        "highlight": float((pixels.max(-1) > 250).mean()),
        "shadow": float((luma < 8).mean()),
        "luma": float(luma.mean()),
    }


def deterministic_veto(after: dict[str, float], source: dict[str, float]) -> tuple[bool, list[str]]:
    flags: list[str] = []
    if after["highlight"] > 0.70 or after["luma"] > 218.0:
        flags.append("extreme_highlight")
    if after["shadow"] > 0.45 or after["luma"] < 25.0:
        flags.append("extreme_shadow")
    if after["colorfulness"] > max(200.0, 4.0 * source["colorfulness"]):
        flags.append("extreme_color")
    return bool(flags), flags


def _quality(onealign: float | None, source_score: float | None) -> tuple[float, float]:
    if onealign is None:
        return 0.0, 0.5
    absolute = max(0.0, min(1.0, onealign / 100.0))
    improvement = (
        0.5 if source_score is None
        else 0.5 + 0.5 * math.tanh(2.0 * ((onealign - source_score) / 100.0))
    )
    return max(0.0, min(1.0, 0.72 * absolute + 0.28 * improvement)), improvement


def _score_paths(scorer: Scorer, paths: list[str], batch_size: int) -> list[float | None]:
    """Score paths in order, batching the forward passes when the scorer supports it."""
    if batch_size <= 1 or not hasattr(scorer, "score_batch"):
        return [scorer.score(path) for path in paths]
    scores: list[float | None] = []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        values = list(scorer.score_batch(chunk))
        if len(values) != len(chunk):
            raise QaError("OneAlign batch scoring returned a mismatched number of scores")
        scores.extend(values)
    return scores


def rank_candidates(source_path: str, candidates: list[dict[str, Any]],
                    scorer: Scorer, *, batch_size: int = 8,
                    abstain_margin: float = 0.0,
                    low_margin: float = 0.0) -> RankedCandidates:
    """Score, rank and pick at most two SFT winners.

    ``abstain_margin`` / ``low_margin`` are the rank1-rank2 OneAlign gaps below
    which the selection is respectively refused and flagged.  They default to
    the historical policy-free behaviour so the ranking primitive carries no
    opinion of its own; ``[render] qa_winner_margin_*`` is the single source of
    truth and ``agent.py`` always passes it.
    """
    if len(candidates) != 8:
        raise QaError("OneAlign ranking requires exactly eight accepted candidates")
    source_stats = _stats(source_path)
    prepared: list[tuple[dict[str, Any], str, bool, list[str]]] = []
    for candidate in candidates:
        candidate_id = candidate.get("candidate_id")
        after_path = candidate.get("after_path")
        if not candidate_id or not after_path:
            raise QaError("candidate_id and after_path are required for ranking")
        cached_stats = candidate.get("_qa_stats")
        after_stats = (
            {key: float(value) for key, value in cached_stats.items()}
            if isinstance(cached_stats, dict)
            else _stats(str(after_path))
        )
        veto, flags = deterministic_veto(after_stats, source_stats)
        # The postprocess cache is process-local and must never enter groups.jsonl.
        durable_candidate = {
            key: value for key, value in candidate.items() if key != "_qa_stats"
        }
        prepared.append((durable_candidate, str(after_path), veto, flags))

    # The source plus every non-vetoed candidate share the forward passes; vetoed
    # candidates keep their unscored (onealign=None) semantics.
    scored_paths = [source_path] + [row[1] for row in prepared if not row[2]]
    scores = iter(_score_paths(scorer, scored_paths, batch_size))
    source_score = next(scores)
    ranked: list[dict[str, Any]] = []
    for candidate, _after_path, veto, flags in prepared:
        onealign = None if veto else next(scores)
        q, improvement = _quality(onealign, source_score)
        qa = {
            "onealign": None if onealign is None else round(onealign, 4),
            "source_onealign": None if source_score is None else round(source_score, 4),
            "q": 0.0 if veto else round(q, 6),
            "improvement": round(improvement, 6),
            "reliable": onealign is not None and not veto,
            "veto": veto,
            "veto_flags": flags,
            "qa_mode": "onealign",
        }
        ranked.append({**candidate, "qa": qa})
    ranking = sorted(
        range(len(ranked)),
        key=lambda index: (
            bool(ranked[index]["qa"]["veto"]),
            not bool(ranked[index]["qa"]["reliable"]),
            -float(ranked[index]["qa"]["q"]),
            str(ranked[index]["candidate_id"]),
        ),
    )
    for rank, index in enumerate(ranking, start=1):
        ranked[index] = {**ranked[index], "rank": rank}
    winners = [
        ranked[index]["candidate_id"]
        for index in ranking
        if ranked[index]["qa"]["reliable"]
        and not ranked[index]["qa"]["veto"]
        and float(ranked[index]["qa"]["q"]) >= SFT_THRESHOLD
    ][:2]
    # WP5 stage C: the OneAlign order only separates rank1 from rank2 once their
    # scores are a couple of points apart (56% agreement below that, 92% above
    # ten), so the gap between the two best scored candidates is the resolution
    # this selection is entitled to.  ``q`` is monotone in ``onealign`` for a
    # fixed source score, which makes these two exactly ranks 1 and 2, and the
    # gap is read off the rounded journal values so it is reproducible from
    # ``groups.jsonl`` alone.
    scored = [
        ranked[index] for index in ranking
        if ranked[index]["qa"]["reliable"] and not ranked[index]["qa"]["veto"]
    ]
    margin = (
        float(scored[0]["qa"]["onealign"]) - float(scored[1]["qa"]["onealign"])
        if len(scored) >= 2 else None
    )
    confidence: str | None = None
    if winners:
        if margin is None:
            # One scored candidate and seven vetoed or unscored ones: there is no
            # near-tie to lose, so the winner stands, but nothing corroborates it
            # either.  Structural, not a threshold outcome — it stays ``low`` even
            # when the thresholds are zeroed out.
            confidence = "low"
        elif margin < abstain_margin:
            confidence, winners = "abstain", []
        elif margin < low_margin:
            confidence = "low"
        else:
            confidence = "normal"
    return RankedCandidates(
        tuple(ranked), tuple(winners), source_score,
        None if margin is None else round(margin, 4), confidence,
    )
