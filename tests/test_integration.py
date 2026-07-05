"""Integration tests for the complete deterministic screening workflow."""

from pathlib import Path

from src.extractors.candidate_extractor import extract_candidate_info
from src.parsers.document_parser import extract_text
from src.scoring.candidate_scorer import score_candidates
from src.scoring.ranker import rank_candidates


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_JD = ROOT / "data" / "sample_job_description.txt"
SAMPLE_RESUMES = ROOT / "data" / "sample_resumes"


def test_complete_workflow_one_resume():
    job_description = extract_text(SAMPLE_JD)
    resume_path = SAMPLE_RESUMES / "aarav_sharma.txt"
    resume_text = extract_text(resume_path)
    candidate_info = extract_candidate_info(resume_text)

    scores = score_candidates(job_description, [resume_text], [candidate_info])
    ranked = rank_candidates(scores)

    assert len(ranked) == 1
    assert ranked[0]["rank"] == 1
    assert ranked[0]["candidate_name"] == "Aarav Sharma"
    assert ranked[0]["ranking_explanation"]
    assert 0 <= ranked[0]["final_score"] <= 100


def test_batch_workflow_with_at_least_10_resumes():
    job_description = extract_text(SAMPLE_JD)
    resume_paths = sorted(SAMPLE_RESUMES.glob("*.txt"))
    resume_texts = [extract_text(path) for path in resume_paths]
    candidate_infos = [extract_candidate_info(text) for text in resume_texts]

    scores = score_candidates(job_description, resume_texts, candidate_infos)
    enriched_scores = [
        {**score, "filename": path.name}
        for score, path in zip(scores, resume_paths)
    ]
    ranked = rank_candidates(enriched_scores)

    assert len(ranked) >= 10
    assert [candidate["rank"] for candidate in ranked] == list(range(1, len(ranked) + 1))
    assert ranked == sorted(
        ranked,
        key=lambda candidate: (
            -candidate["final_score"],
            -candidate["similarity_score"],
            -candidate["skill_match_score"],
            (candidate.get("candidate_name") or "zzzz unnamed").casefold(),
        ),
    )
    assert all(candidate["ranking_explanation"] for candidate in ranked)
    assert all(0 <= candidate["final_score"] <= 100 for candidate in ranked)
    assert all(0 <= candidate["similarity_score"] <= 100 for candidate in ranked)
