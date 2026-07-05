"""TF-IDF and cosine-similarity comparison for job and resume text."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityResult(TypedDict):
    """Similarity values for one resume."""

    raw_similarity: float
    score: float


def calculate_similarities(
    job_description: str,
    resumes: Sequence[str],
) -> list[SimilarityResult]:
    """Compare one job description with multiple resumes.

    A single TF-IDF vectorizer is fit on the job description and every resume
    together. Each result contains raw cosine similarity in the range 0 to 1
    and the same value normalized to a score from 0 to 100. Results retain the
    order of the supplied resumes.

    Args:
        job_description: Non-empty job description text.
        resumes: Resume texts to compare, including empty texts if needed.

    Raises:
        TypeError: If the job description or a resume is not a string.
        ValueError: If the job description is empty or whitespace-only.
    """
    if not isinstance(job_description, str):
        raise TypeError("job_description must be a string")
    if not job_description.strip():
        raise ValueError("job_description must not be empty")
    if isinstance(resumes, (str, bytes)) or not isinstance(resumes, Sequence):
        raise TypeError("resumes must be a sequence of strings")
    if any(not isinstance(resume, str) for resume in resumes):
        raise TypeError("every resume must be a string")
    if not resumes:
        return []

    # Fit once so every document uses the same vocabulary and IDF weights.
    corpus = [job_description, *resumes]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )

    try:
        document_vectors = vectorizer.fit_transform(corpus)
    except ValueError as exc:
        # A corpus made entirely of stop words has no comparable vocabulary.
        if "empty vocabulary" in str(exc).lower():
            return [_build_result(0.0) for _ in resumes]
        raise

    job_vector = document_vectors[0:1]
    resume_vectors = document_vectors[1:]
    raw_similarities = cosine_similarity(job_vector, resume_vectors).ravel()

    return [_build_result(float(value)) for value in raw_similarities]


def _build_result(raw_similarity: float) -> SimilarityResult:
    """Clamp floating-point output and create one public result mapping."""
    bounded_similarity = min(1.0, max(0.0, raw_similarity))
    return {
        "raw_similarity": bounded_similarity,
        "score": bounded_similarity * 100.0,
    }
