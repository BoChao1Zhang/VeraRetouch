"""OneAlign ranking, deterministic veto, and top-2 SFT winner selection."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from PIL import Image


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
        result = self.runner.score_path(path)
        value = result.get("onealign", result.get("iaa_mixed"))
        return None if value is None else float(value)


@dataclass(frozen=True, slots=True)
class RankedCandidates:
    candidates: tuple[dict[str, Any], ...]
    winner_ids: tuple[str, ...]
    source_score: float | None


def _stats(path: str) -> dict[str, float]:
    with Image.open(path) as image:
        image.load()
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


def rank_candidates(source_path: str, candidates: list[dict[str, Any]],
                    scorer: Scorer) -> RankedCandidates:
    if len(candidates) != 8:
        raise QaError("OneAlign ranking requires exactly eight accepted candidates")
    source_stats = _stats(source_path)
    source_score = scorer.score(source_path)
    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = candidate.get("candidate_id")
        after_path = candidate.get("after_path")
        if not candidate_id or not after_path:
            raise QaError("candidate_id and after_path are required for ranking")
        after_stats = _stats(str(after_path))
        veto, flags = deterministic_veto(after_stats, source_stats)
        onealign = None if veto else scorer.score(str(after_path))
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
    return RankedCandidates(tuple(ranked), tuple(winners), source_score)
