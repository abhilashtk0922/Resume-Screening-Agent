"""Tests for deterministic technical skill extraction."""

from src.extractors.skills import SKILLS, extract_skills


def test_matches_normal_skills():
    text = "Built Python services with FastAPI, PostgreSQL, Docker, and AWS."

    assert extract_skills(text) == ["Python", "PostgreSQL", "AWS", "Docker", "FastAPI"]


def test_matching_is_case_insensitive():
    text = "PYTHON, pytorch, DOCKER, and streamLIT"

    assert extract_skills(text) == ["Python", "Docker", "PyTorch", "Streamlit"]


def test_aliases_return_one_canonical_skill_each():
    text = (
        "NLP and Natural Language Processing; LLM and Large Language Models; "
        "RAG and Retrieval Augmented Generation; REST API and REST APIs."
    )

    assert extract_skills(text) == [
        "Natural Language Processing",
        "Large Language Models",
        "RAG",
        "REST API",
    ]


def test_repeated_mentions_do_not_create_duplicates():
    assert extract_skills("Python python PYTHON and pandas pandas") == ["Python", "pandas"]
    assert len(SKILLS) == len(set(skill.casefold() for skill in SKILLS))


def test_short_and_symbol_skills_match_as_complete_tokens():
    text = "Experience: C, C++, C#, R, and SQL."

    assert extract_skills(text) == ["C", "C++", "C#", "R", "SQL"]


def test_cplusplus_and_csharp_do_not_also_match_plain_c():
    assert extract_skills("C++ and C# developer") == ["C++", "C#"]


def test_unrelated_words_do_not_produce_false_matches():
    text = (
        "We respond reactively to angularity while tracking express.jsmith, "
        "learning from narrative reports and discussing mysqlite systems."
    )

    assert extract_skills(text) == []


def test_java_does_not_match_inside_javascript_and_sql_not_inside_mysql():
    assert extract_skills("JavaScript with MySQL") == ["JavaScript", "MySQL"]


def test_custom_taxonomy_is_supported():
    taxonomy = {"Data Visualization": ("data viz", "visualisation")}

    assert extract_skills("Strong DATA VIZ experience", taxonomy) == ["Data Visualization"]
