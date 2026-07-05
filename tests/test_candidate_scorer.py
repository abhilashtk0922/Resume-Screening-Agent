"""Tests for transparent weighted candidate scoring."""

import pytest

from src.extractors.candidate_extractor import CandidateInfo
from src.scoring.candidate_scorer import (
    calculate_candidate_score,
    score_candidates,
)


def candidate(
    skills: list[str] | None = None,
    experience: int | float | None = None,
    education: list[str] | None = None,
) -> CandidateInfo:
    """Create minimal structured candidate data for scoring tests."""
    return {
        "name": "Aarav Sharma",
        "email": "",
        "phone": "",
        "skills": skills or [],
        "education": education or [],
        "estimated_experience_years": experience,
    }


def test_full_skill_match():
    result = calculate_candidate_score(
        "Requires Python, Docker, and AWS.",
        candidate(["Python", "Docker", "AWS"]),
        similarity_score=70,
    )

    assert result["skill_match_score"] == 100.0
    assert result["matched_skills"] == ["Python", "AWS", "Docker"]
    assert result["missing_skills"] == []


def test_partial_skill_match():
    result = calculate_candidate_score(
        "Requires Python, Docker, and AWS.",
        candidate(["Python", "Docker"]),
        similarity_score=50,
    )

    assert result["skill_match_score"] == pytest.approx(200 / 3)
    assert result["missing_skills"] == ["AWS"]


def test_no_skill_match():
    result = calculate_candidate_score(
        "Requires Python and Docker.",
        candidate(["Java"]),
        similarity_score=30,
    )

    assert result["skill_match_score"] == 0.0
    assert result["matched_skills"] == []
    assert result["missing_skills"] == ["Python", "Docker"]


def test_exact_weighted_calculation_when_all_signals_exist():
    job = "Python Docker. 4 years of experience. B.Tech in Computer Science."
    info = candidate(
        ["Python", "Docker"],
        experience=4,
        education=["Bachelor of Technology", "Computer Science"],
    )

    result = calculate_candidate_score(job, info, similarity_score=80)

    assert result["effective_weights"] == {
        "similarity": 50.0,
        "skills": 30.0,
        "experience": 10.0,
        "education": 10.0,
    }
    assert result["final_score"] == pytest.approx(90.0)


def test_missing_experience_requirement_gets_zero_weight():
    result = calculate_candidate_score(
        "Python developer with B.Tech.",
        candidate(["Python"], education=["Bachelor of Technology"]),
        similarity_score=60,
    )

    assert result["experience_score"] == 0.0
    assert result["effective_weights"]["experience"] == 0.0
    assert result["score_breakdown"]["experience"]["requirement_detected"] is False


def test_missing_education_requirement_gets_zero_weight():
    result = calculate_candidate_score(
        "Python developer with 3 years of experience.",
        candidate(["Python"], experience=3),
        similarity_score=60,
    )

    assert result["education_score"] == 0.0
    assert result["effective_weights"]["education"] == 0.0


def test_dynamic_weights_are_redistributed_proportionally():
    result = calculate_candidate_score(
        "Python developer",
        candidate(["Python"]),
        similarity_score=60,
    )

    assert result["effective_weights"]["similarity"] == pytest.approx(62.5)
    assert result["effective_weights"]["skills"] == pytest.approx(37.5)
    assert sum(result["effective_weights"].values()) == pytest.approx(100.0)
    assert result["final_score"] == pytest.approx(75.0)


def test_scores_are_clamped_to_boundaries():
    high = calculate_candidate_score("Python", candidate(["Python"]), 150)
    low = calculate_candidate_score("Python", candidate([]), -20)

    for result in (high, low):
        score_fields = (
            "final_score",
            "similarity_score",
            "skill_match_score",
            "experience_score",
            "education_score",
        )
        assert all(0.0 <= result[field] <= 100.0 for field in score_fields)


def test_batch_scoring_preserves_candidates_and_uses_existing_modules():
    resumes = [
        "Aarav Sharma\nPython Docker developer",
        "Meera Patel\nJava developer",
    ]

    results = score_candidates("Python Docker developer", resumes)

    assert [result["candidate_name"] for result in results] == [
        "Aarav Sharma",
        "Meera Patel",
    ]
    assert results[0]["final_score"] > results[1]["final_score"]
