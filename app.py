"""Streamlit app for the AI Resume Screening Agent."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from src.extractors.candidate_extractor import extract_candidate_info
from src.parsers.document_parser import DocumentParserError, extract_text
from src.scoring.candidate_scorer import score_candidates
from src.scoring.ranker import rank_candidates


TABLE_COLUMNS = [
    "Rank",
    "Candidate",
    "Final Score",
    "NLP Similarity",
    "Skill Match",
    "Experience Score",
    "Education Score",
]


def main() -> None:
    """Render and run the Streamlit application."""
    st.set_page_config(page_title="AI Resume Screening Agent", page_icon="📄", layout="wide")

    st.title("AI Resume Screening Agent")
    st.write(
        "Rank candidates against a Job Description using NLP-based similarity "
        "and transparent scoring."
    )

    st.header("1. Job Description")
    job_text = st.text_area(
        "Paste Job Description text",
        height=220,
        placeholder="Paste the role description, requirements, skills, education, and experience here...",
    )
    job_file = st.file_uploader(
        "Or upload a Job Description file",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=False,
    )

    st.header("2. Resume Upload")
    resume_files = st.file_uploader(
        "Upload one or more resumes",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="PDF, DOCX, and TXT files are supported. The app can process 10+ resumes.",
    )

    if st.button("Analyze Candidates", type="primary"):
        run_analysis(job_text, job_file, resume_files)


def run_analysis(job_text: str, job_file: Any, resume_files: list[Any] | None) -> None:
    """Execute the full deterministic screening workflow."""
    job_description = _load_job_description(job_text, job_file)
    if not job_description:
        st.error("Please paste or upload a readable Job Description before analyzing.")
        return
    if not resume_files:
        st.error("Please upload at least one resume.")
        return

    parsed_resumes: list[dict[str, Any]] = []
    file_errors: list[dict[str, str]] = []

    progress = st.progress(0, text="Parsing resumes...")
    with st.spinner("Analyzing candidates..."):
        for index, uploaded_file in enumerate(resume_files, start=1):
            try:
                resume_text = extract_text(uploaded_file, filename=uploaded_file.name)
                candidate_info = extract_candidate_info(resume_text)
                parsed_resumes.append(
                    {
                        "filename": uploaded_file.name,
                        "text": resume_text,
                        "candidate_info": candidate_info,
                    }
                )
            except DocumentParserError as exc:
                file_errors.append({"filename": uploaded_file.name, "error": str(exc)})
            except Exception as exc:  # Defensive guard so one bad file cannot stop the batch.
                file_errors.append(
                    {
                        "filename": uploaded_file.name,
                        "error": f"Unexpected processing error: {exc}",
                    }
                )
            progress.progress(index / len(resume_files), text=f"Processed {index}/{len(resume_files)} files")

        if not parsed_resumes:
            st.error("No resumes could be analyzed. Please check the uploaded files.")
            _display_file_errors(file_errors)
            return

        scores = score_candidates(
            job_description,
            [resume["text"] for resume in parsed_resumes],
            [resume["candidate_info"] for resume in parsed_resumes],
        )

    enriched_scores = []
    for resume, score in zip(parsed_resumes, scores):
        enriched = {
            **score,
            "filename": resume["filename"],
            "email": resume["candidate_info"]["email"],
            "phone": resume["candidate_info"]["phone"],
            "education": resume["candidate_info"]["education"],
            "estimated_experience_years": resume["candidate_info"]["estimated_experience_years"],
            "extracted_skills": resume["candidate_info"]["skills"],
        }
        enriched_scores.append(enriched)

    ranked_results = rank_candidates(enriched_scores)
    _display_summary_metrics(len(resume_files), ranked_results)
    _display_file_errors(file_errors)
    _display_results_table(ranked_results)
    _display_candidate_details(ranked_results)
    _display_downloads(ranked_results)


def _load_job_description(job_text: str, job_file: Any) -> str:
    """Load a Job Description from pasted text or an uploaded file."""
    if job_file is not None:
        try:
            return extract_text(job_file, filename=job_file.name)
        except DocumentParserError as exc:
            st.error(f"Could not read Job Description file: {exc}")
            return ""
    return job_text.strip()


def _display_summary_metrics(uploaded_count: int, ranked_results: list[dict[str, Any]]) -> None:
    """Show high-level screening metrics."""
    scores = [float(result["final_score"]) for result in ranked_results]
    top_candidate = ranked_results[0].get("candidate_name") or "Unnamed candidate"

    st.header("3. Summary")
    columns = st.columns(5)
    columns[0].metric("Total Resumes Uploaded", uploaded_count)
    columns[1].metric("Successfully Analyzed Resumes", len(ranked_results))
    columns[2].metric("Top Candidate", top_candidate)
    columns[3].metric("Highest Score", f"{max(scores):.1f}")
    columns[4].metric("Average Score", f"{sum(scores) / len(scores):.1f}")


def _display_file_errors(file_errors: list[dict[str, str]]) -> None:
    """Show per-file parsing failures without interrupting valid results."""
    if not file_errors:
        return
    with st.expander("Files skipped due to parsing errors", expanded=False):
        for error in file_errors:
            st.warning(f"{error['filename']}: {error['error']}")


def _display_results_table(ranked_results: list[dict[str, Any]]) -> None:
    """Display ranked candidate scores in a compact table."""
    st.header("4. Ranked Results")
    table = pd.DataFrame(
        [
            {
                "Rank": result["rank"],
                "Candidate": result.get("candidate_name") or "Unnamed candidate",
                "Final Score": round(float(result["final_score"]), 2),
                "NLP Similarity": round(float(result["similarity_score"]), 2),
                "Skill Match": round(float(result["skill_match_score"]), 2),
                "Experience Score": round(float(result["experience_score"]), 2),
                "Education Score": round(float(result["education_score"]), 2),
            }
            for result in ranked_results
        ],
        columns=TABLE_COLUMNS,
    )
    st.dataframe(table, use_container_width=True, hide_index=True)


def _display_candidate_details(ranked_results: list[dict[str, Any]]) -> None:
    """Show transparent details for every ranked candidate."""
    st.header("5. Candidate Details")
    for result in ranked_results:
        name = result.get("candidate_name") or "Unnamed candidate"
        label = f"#{result['rank']} — {name} — {float(result['final_score']):.1f}/100"
        with st.expander(label):
            left, right = st.columns(2)
            left.write(f"**Filename:** {result.get('filename', '')}")
            left.write(f"**Email:** {result.get('email') or 'Not found'}")
            left.write(f"**Phone:** {result.get('phone') or 'Not found'}")
            left.write(
                "**Estimated Experience:** "
                f"{result.get('estimated_experience_years') if result.get('estimated_experience_years') is not None else 'Not found'}"
            )
            left.write(f"**Education:** {_join_or_empty(result.get('education'))}")
            left.write(f"**Extracted Skills:** {_join_or_empty(result.get('extracted_skills'))}")

            right.write(f"**Matched Skills:** {_join_or_empty(result.get('matched_skills'))}")
            right.write(f"**Missing Skills:** {_join_or_empty(result.get('missing_skills'))}")
            right.write(f"**NLP Similarity:** {float(result['similarity_score']):.2f}")
            right.write(f"**Skill Match Score:** {float(result['skill_match_score']):.2f}")
            right.write(f"**Experience Score:** {float(result['experience_score']):.2f}")
            right.write(f"**Education Score:** {float(result['education_score']):.2f}")
            right.write(f"**Final Score:** {float(result['final_score']):.2f}")

            st.write("**Effective Weights**")
            st.json(_json_safe(result.get("effective_weights", {})))
            st.write("**Score Breakdown**")
            st.json(_json_safe(result.get("score_breakdown", {})))
            st.write("**Ranking Explanation**")
            st.info(result.get("ranking_explanation", ""))


def _display_downloads(ranked_results: list[dict[str, Any]]) -> None:
    """Provide CSV and JSON exports."""
    st.header("6. Export Results")
    export_rows = [_json_safe(result) for result in ranked_results]
    csv_data = pd.DataFrame(export_rows).to_csv(index=False)
    json_data = json.dumps(export_rows, indent=2, ensure_ascii=False)

    col1, col2 = st.columns(2)
    col1.download_button(
        "Download CSV",
        csv_data,
        file_name="resume_screening_results.csv",
        mime="text/csv",
    )
    col2.download_button(
        "Download JSON",
        json_data,
        file_name="resume_screening_results.json",
        mime="application/json",
    )


def _join_or_empty(value: Any) -> str:
    """Format list-like values for display."""
    if isinstance(value, list) and value:
        return ", ".join(str(item) for item in value)
    return "Not found"


def _json_safe(value: Any) -> Any:
    """Recursively convert data into JSON-serializable Python objects."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if pd.isna(value) if not isinstance(value, (dict, list, str)) else False:
        return None
    return value


if __name__ == "__main__":
    main()
