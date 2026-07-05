"""Tests for deterministic candidate ranking."""

from src.scoring.ranker import generate_explanation, rank_candidates


def scored_candidate(
    name: str,
    final: float,
    similarity: float = 50,
    skills: float = 50,
) -> dict:
    """Build minimal scored candidate data."""
    return {
        "candidate_name": name,
        "final_score": final,
        "similarity_score": similarity,
        "skill_match_score": skills,
        "experience_score": 0,
        "education_score": 0,
        "matched_skills": ["Python"],
        "missing_skills": ["Docker"],
        "effective_weights": {"similarity": 50.0, "skills": 50.0},
        "score_breakdown": {"similarity": {"score": similarity}},
    }


def test_ranking_order():
    ranked = rank_candidates(
        [
            scored_candidate("Candidate B", 60),
            scored_candidate("Candidate A", 90),
            scored_candidate("Candidate C", 75),
        ]
    )

    assert [candidate["candidate_name"] for candidate in ranked] == [
        "Candidate A",
        "Candidate C",
        "Candidate B",
    ]


def test_rank_assignment():
    ranked = rank_candidates([scored_candidate("A", 80), scored_candidate("B", 70)])

    assert [candidate["rank"] for candidate in ranked] == [1, 2]


def test_deterministic_ties():
    ranked = rank_candidates(
        [
            scored_candidate("Zara", 80, similarity=70, skills=80),
            scored_candidate("Aman", 80, similarity=70, skills=80),
            scored_candidate("Nina", 80, similarity=75, skills=60),
        ]
    )

    assert [candidate["candidate_name"] for candidate in ranked] == [
        "Nina",
        "Aman",
        "Zara",
    ]


def test_explanation_generation():
    explanation = generate_explanation(scored_candidate("Aarav", 82, 76, 90))

    assert "Aarav" in explanation
    assert "82.0/100" in explanation
    assert "NLP similarity is 76.0/100" in explanation
    assert "Matched skills: Python" in explanation
    assert "Missing skills: Docker" in explanation


def test_missing_names_are_handled_gracefully():
    ranked = rank_candidates([scored_candidate("", 70)])

    assert ranked[0]["candidate_name"] == ""
    assert "Unnamed candidate" in ranked[0]["ranking_explanation"]


def test_empty_candidate_list():
    assert rank_candidates([]) == []


def test_scoring_data_preservation():
    original = scored_candidate("Aarav", 88)
    ranked = rank_candidates([original])

    assert ranked[0]["final_score"] == original["final_score"]
    assert ranked[0]["effective_weights"] == original["effective_weights"]
    assert ranked[0]["score_breakdown"] == original["score_breakdown"]
