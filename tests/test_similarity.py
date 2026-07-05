"""Tests for the deterministic TF-IDF similarity engine."""

import pytest

from src.scoring.similarity import calculate_similarities


def test_identical_documents_receive_full_similarity():
    text = "Python machine learning engineer builds production NLP systems"

    result = calculate_similarities(text, [text])[0]

    assert result["raw_similarity"] == pytest.approx(1.0)
    assert result["score"] == pytest.approx(100.0)


def test_related_document_scores_above_unrelated_document():
    job = "Python machine learning NLP model development with scikit-learn"
    related = "Python developer building machine learning NLP models"
    unrelated = "Chef experienced in pastry baking and kitchen operations"

    results = calculate_similarities(job, [related, unrelated])

    assert results[0]["score"] > results[1]["score"]
    assert results[0]["score"] > 0


def test_unrelated_documents_have_zero_similarity():
    results = calculate_similarities(
        "Python cloud engineering",
        ["culinary arts pastry baking"],
    )

    assert results[0] == {"raw_similarity": 0.0, "score": 0.0}


def test_one_resume_returns_one_result():
    results = calculate_similarities("Python API developer", ["Python Flask developer"])

    assert len(results) == 1


def test_multiple_resumes_preserve_input_order():
    results = calculate_similarities(
        "Python data science",
        ["Python data science", "Python", "graphic design"],
    )

    assert len(results) == 3
    assert results[0]["score"] > results[1]["score"] > results[2]["score"]


def test_empty_inputs_are_handled():
    assert calculate_similarities("Python developer", []) == []
    assert calculate_similarities("Python developer", [""])[0]["score"] == 0.0

    with pytest.raises(ValueError, match="must not be empty"):
        calculate_similarities("   ", ["Python developer"])


def test_stop_word_only_corpus_returns_zero_scores():
    results = calculate_similarities("the and or", ["", "the and"])

    assert results == [
        {"raw_similarity": 0.0, "score": 0.0},
        {"raw_similarity": 0.0, "score": 0.0},
    ]


def test_all_scores_stay_within_boundaries():
    results = calculate_similarities(
        "Python machine learning engineer",
        ["Python machine learning engineer", "Python", "", "accounting finance"],
    )

    assert all(0.0 <= result["raw_similarity"] <= 1.0 for result in results)
    assert all(0.0 <= result["score"] <= 100.0 for result in results)


def test_batch_of_more_than_ten_resumes():
    resumes = [f"Python data engineer project {index}" for index in range(12)]

    results = calculate_similarities("Python data engineer", resumes)

    assert len(results) == 12
    assert all(result["score"] > 0 for result in results)


@pytest.mark.parametrize("invalid_resumes", ["one resume", ["valid", None]])
def test_invalid_resume_inputs_raise_clear_errors(invalid_resumes):
    with pytest.raises(TypeError):
        calculate_similarities("Python developer", invalid_resumes)
