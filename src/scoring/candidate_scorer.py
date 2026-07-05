"""Transparent weighted scoring for candidates against a job description."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypedDict

from src.extractors.candidate_extractor import (
    CandidateInfo,
    estimate_experience_years,
    extract_candidate_info,
    extract_education,
)
from src.extractors.skills import extract_skills


BASE_WEIGHTS = {
    "similarity": 50.0,
    "skills": 30.0,
    "experience": 10.0,
    "education": 10.0,
}


class CandidateScore(TypedDict):
    """Explainable candidate score and its supporting details."""

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


def score_candidates(
    job_description: str,
    resume_texts: Sequence[str],
    candidate_infos: Sequence[CandidateInfo] | None = None,
) -> list[CandidateScore]:
    """Score multiple resumes using one shared TF-IDF similarity calculation.

    Candidate details are extracted from each resume unless pre-extracted
    ``candidate_infos`` are supplied. Output order matches ``resume_texts``.
    """
    # Import here so the transparent scoring helpers remain importable even
    # before the optional ML stack is loaded. Batch scoring still uses the
    # shared similarity engine exactly once for all resumes.
    from src.scoring.similarity import calculate_similarities

    similarities = calculate_similarities(job_description, resume_texts)

    if candidate_infos is None:
        infos = [extract_candidate_info(text) for text in resume_texts]
    else:
        if len(candidate_infos) != len(resume_texts):
            raise ValueError("candidate_infos must match the number of resumes")
        infos = list(candidate_infos)

    return [
        calculate_candidate_score(job_description, info, similarity["score"])
        for info, similarity in zip(infos, similarities)
    ]


def score_candidate(
    job_description: str,
    resume_text: str,
    candidate_info: CandidateInfo | None = None,
) -> CandidateScore:
    """Convenience wrapper for scoring one candidate."""
    infos = [candidate_info] if candidate_info is not None else None
    return score_candidates(job_description, [resume_text], infos)[0]


def calculate_candidate_score(
    job_description: str,
    candidate_info: CandidateInfo,
    similarity_score: float,
) -> CandidateScore:
    """Combine similarity, skills, experience, and education into one score.

    Base weights are 50%, 30%, 10%, and 10%. If the job description does not
    contain a detectable requirement for a component, that component is
    inactive and its weight is redistributed proportionally among active
    components. Similarity is always active.
    """
    if not isinstance(job_description, str):
        raise TypeError("job_description must be a string")

    similarity = _clamp_score(similarity_score)
    required_skills = extract_skills(job_description)
    candidate_skills = set(candidate_info.get("skills", []))
    matched_skills = [skill for skill in required_skills if skill in candidate_skills]
    missing_skills = [skill for skill in required_skills if skill not in candidate_skills]
    skill_score = (
        len(matched_skills) / len(required_skills) * 100.0
        if required_skills
        else 0.0
    )

    required_experience = estimate_experience_years(job_description)
    candidate_experience = candidate_info.get("estimated_experience_years")
    experience_score = _experience_score(candidate_experience, required_experience)

    required_education = extract_education(job_description)
    candidate_education = set(candidate_info.get("education", []))
    education_score = (
        len(candidate_education.intersection(required_education))
        / len(required_education)
        * 100.0
        if required_education
        else 0.0
    )

    active_signals = {
        "similarity": True,
        "skills": bool(required_skills),
        "experience": required_experience is not None,
        "education": bool(required_education),
    }
    effective_weights = _redistribute_weights(active_signals)
    component_scores = {
        "similarity": similarity,
        "skills": skill_score,
        "experience": experience_score,
        "education": education_score,
    }

    breakdown: dict[str, dict[str, Any]] = {}
    for component, score in component_scores.items():
        effective_weight = effective_weights[component]
        contribution = score * effective_weight / 100.0
        breakdown[component] = {
            "score": score,
            "base_weight": BASE_WEIGHTS[component],
            "effective_weight": effective_weight,
            "contribution": contribution,
            "requirement_detected": active_signals[component],
        }

    final_score = _clamp_score(sum(item["contribution"] for item in breakdown.values()))
    return {
        "candidate_name": candidate_info.get("name", ""),
        "final_score": final_score,
        "similarity_score": similarity,
        "skill_match_score": _clamp_score(skill_score),
        "experience_score": _clamp_score(experience_score),
        "education_score": _clamp_score(education_score),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "effective_weights": effective_weights,
        "score_breakdown": breakdown,
    }


def _experience_score(
    candidate_years: int | float | None,
    required_years: int | float | None,
) -> float:
    """Return proportional experience credit when a requirement exists."""
    if required_years is None or candidate_years is None:
        return 0.0
    if required_years <= 0:
        return 100.0
    return _clamp_score(float(candidate_years) / float(required_years) * 100.0)


def _redistribute_weights(active_signals: dict[str, bool]) -> dict[str, float]:
    """Redistribute inactive weights proportionally across active components."""
    active_total = sum(
        BASE_WEIGHTS[name] for name, is_active in active_signals.items() if is_active
    )
    return {
        name: (BASE_WEIGHTS[name] / active_total * 100.0 if is_active else 0.0)
        for name, is_active in active_signals.items()
    }


def _clamp_score(score: float) -> float:
    """Keep a numeric score inside the public 0-to-100 range."""
    return min(100.0, max(0.0, float(score)))
