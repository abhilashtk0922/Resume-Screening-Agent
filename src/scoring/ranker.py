"""Deterministic ranking and explanation helpers for scored candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict


class RankedCandidate(TypedDict, total=False):
    """Scored candidate data enriched with rank and explanation."""

    rank: int
    ranking_explanation: str
    candidate_name: str
    final_score: float
    similarity_score: float
    skill_match_score: float
    experience_score: float
    education_score: float
    matched_skills: list[str]
    missing_skills: list[str]
    effective_weights: dict[str, float]
    score_breakdown: dict[str, dict[str, Any]]


def rank_candidates(scored_candidates: Sequence[Mapping[str, Any]]) -> list[RankedCandidate]:
    """Sort scored candidates and attach deterministic ranks.

    Candidates are ordered by final score, NLP similarity, skill match score,
    and candidate name. Score fields sort from highest to lowest; names sort
    alphabetically so ties are stable and repeatable.
    """
    ranked_inputs = list(scored_candidates)
    sorted_candidates = sorted(ranked_inputs, key=_ranking_key)

    ranked: list[RankedCandidate] = []
    for index, candidate in enumerate(sorted_candidates, start=1):
        candidate_data: RankedCandidate = dict(candidate)  # type: ignore[assignment]
        candidate_data["rank"] = index
        candidate_data["ranking_explanation"] = generate_explanation(candidate_data)
        ranked.append(candidate_data)
    return ranked


def generate_explanation(candidate: Mapping[str, Any]) -> str:
    """Create a concise explanation from calculated scoring data only."""
    name = _display_name(candidate.get("candidate_name"))
    final_score = _number(candidate.get("final_score"))
    similarity_score = _number(candidate.get("similarity_score"))
    skill_match_score = _number(candidate.get("skill_match_score"))
    matched_skills = _string_list(candidate.get("matched_skills"))
    missing_skills = _string_list(candidate.get("missing_skills"))

    parts = [
        f"{name} has an overall {_match_strength(final_score)} match "
        f"with a final score of {final_score:.1f}/100."
    ]

    if similarity_score is not None:
        parts.append(f"NLP similarity is {similarity_score:.1f}/100.")
    if skill_match_score is not None:
        parts.append(f"Required skill match is {skill_match_score:.1f}/100.")
    if matched_skills:
        parts.append(f"Matched skills: {', '.join(matched_skills)}.")
    if missing_skills:
        parts.append(f"Missing skills: {', '.join(missing_skills)}.")

    return " ".join(parts)


def _ranking_key(candidate: Mapping[str, Any]) -> tuple[float, float, float, str]:
    """Return a stable key for descending score order and ascending names."""
    return (
        -_score(candidate.get("final_score")),
        -_score(candidate.get("similarity_score")),
        -_score(candidate.get("skill_match_score")),
        _normalized_name(candidate.get("candidate_name")),
    )


def _score(value: Any) -> float:
    """Convert possibly missing score values into safe numeric values."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _number(value: Any) -> float:
    """Return a score clamped to the public 0-to-100 range."""
    return min(100.0, max(0.0, _score(value)))


def _normalized_name(value: Any) -> str:
    """Normalize candidate names while keeping unnamed candidates last in ties."""
    if isinstance(value, str) and value.strip():
        return value.strip().casefold()
    return "zzzz unnamed candidate"


def _display_name(value: Any) -> str:
    """Return a readable candidate name for explanations."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "Unnamed candidate"


def _string_list(value: Any) -> list[str]:
    """Return a clean list of strings from optional list-like data."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _match_strength(score: float) -> str:
    """Translate a numeric score into a conservative match label."""
    if score >= 80:
        return "excellent"
    if score >= 65:
        return "strong"
    if score >= 50:
        return "moderate"
    return "limited"
