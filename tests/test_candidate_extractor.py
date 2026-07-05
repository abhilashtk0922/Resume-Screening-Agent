"""Tests for deterministic candidate information extraction."""

from src.extractors.candidate_extractor import (
    estimate_experience_years,
    extract_candidate_from_document,
    extract_candidate_info,
    extract_education,
    extract_email,
    extract_name,
    extract_phone,
)


SAMPLE_RESUME = """Aarav Sharma
aarav.sharma+jobs@example.co.in | +91 98765 43210

Professional Summary
Python developer with 5+ years of professional experience.

Skills
Python, PYTHON, FastAPI, Docker, PostgreSQL

Education
B.Tech in Computer Science
"""


def test_extracts_conservative_name_from_opening_line():
    assert extract_name(SAMPLE_RESUME) == "Aarav Sharma"
    assert extract_name("RESUME\nProfessional Summary\nPython developer") == ""


def test_extracts_common_email_format():
    assert extract_email(SAMPLE_RESUME) == "aarav.sharma+jobs@example.co.in"


def test_extracts_international_and_indian_phone_formats():
    assert extract_phone(SAMPLE_RESUME) == "+91 98765 43210"
    assert extract_phone("Phone: +1 (415) 555-2671") == "+1 (415) 555-2671"


def test_reuses_skills_taxonomy():
    result = extract_candidate_info(SAMPLE_RESUME)

    assert result["skills"] == ["Python", "PostgreSQL", "Docker", "FastAPI"]


def test_duplicate_skills_and_education_are_removed():
    result = extract_candidate_info("Python python PYTHON\nB.Tech and Bachelor of Technology")

    assert result["skills"] == ["Python"]
    assert result["education"] == ["Bachelor of Technology"]


def test_extracts_education_degrees_and_fields():
    text = "B.E. Information Technology; MCA; M.Sc Computer Science; PhD"

    assert extract_education(text) == [
        "Bachelor of Engineering",
        "MCA",
        "Master of Science",
        "PhD",
        "Computer Science",
        "Information Technology",
    ]


def test_lowercase_be_is_not_mistaken_for_engineering_degree():
    assert extract_education("Candidate will be available immediately.") == []


def test_estimates_only_explicit_experience():
    assert estimate_experience_years(SAMPLE_RESUME) == 5
    assert estimate_experience_years("Experience: 3.5 years") == 3.5
    assert estimate_experience_years("Worked from 2018 to 2024") is None


def test_missing_information_uses_sensible_empty_values():
    assert extract_candidate_info("") == {
        "name": "",
        "email": "",
        "phone": "",
        "skills": [],
        "education": [],
        "estimated_experience_years": None,
    }


def test_document_interface_reuses_existing_txt_parser():
    result = extract_candidate_from_document(
        SAMPLE_RESUME.encode("utf-8"),
        filename="candidate.txt",
    )

    assert result["name"] == "Aarav Sharma"
    assert result["skills"] == ["Python", "PostgreSQL", "Docker", "FastAPI"]
