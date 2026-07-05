"""Conservative, deterministic extraction of candidate information."""

from __future__ import annotations

import re
from typing import TypedDict

from src.extractors.skills import extract_skills
from src.parsers.document_parser import DocumentSource, extract_text, normalize_text


class CandidateInfo(TypedDict):
    """Structured candidate details extracted from resume text."""

    name: str
    email: str
    phone: str
    skills: list[str]
    education: list[str]
    estimated_experience_years: int | float | None


EMAIL_PATTERN = re.compile(
    r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])",
    re.IGNORECASE,
)
PHONE_PATTERN = re.compile(r"(?<!\w)\+?\d[\d \t().-]{7,}\d(?!\w)")

EDUCATION_ALIASES: dict[str, tuple[str, ...]] = {
    "Bachelor of Engineering": ("Bachelor of Engineering", "B.E.", "BE"),
    "Bachelor of Technology": ("Bachelor of Technology", "B.Tech", "BTech"),
    "Bachelor of Science": ("Bachelor of Science", "B.Sc", "BSc"),
    "BCA": ("BCA",),
    "MCA": ("MCA",),
    "Master of Technology": ("Master of Technology", "M.Tech", "MTech"),
    "Master of Science": ("Master of Science", "M.Sc", "MSc"),
    "MBA": ("MBA",),
    "PhD": ("PhD", "Ph.D."),
    "Computer Science": ("Computer Science",),
    "Information Science": ("Information Science",),
    "Information Technology": ("Information Technology",),
}

_NAME_LABEL_PATTERN = re.compile(r"^\s*(?:candidate\s+)?name\s*[:\-]\s*(.+)$", re.IGNORECASE)
_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z.'-]*(?:\s+[A-Za-z][A-Za-z.'-]*){1,3}$")
_NAME_BLOCKLIST = {
    "curriculum vitae",
    "resume",
    "professional summary",
    "profile summary",
    "contact information",
    "work experience",
    "technical skills",
}
_JOB_TITLE_WORDS = {
    "analyst",
    "architect",
    "consultant",
    "developer",
    "engineer",
    "intern",
    "manager",
    "scientist",
    "specialist",
}


def extract_candidate_info(text: str) -> CandidateInfo:
    """Extract structured candidate details from already-parsed resume text.

    Missing fields use empty strings, empty lists, or ``None``. The function
    only reports information present in the supplied text.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    cleaned_text = normalize_text(text)
    return {
        "name": extract_name(cleaned_text),
        "email": extract_email(cleaned_text),
        "phone": extract_phone(cleaned_text),
        "skills": extract_skills(cleaned_text),
        "education": extract_education(cleaned_text),
        "estimated_experience_years": estimate_experience_years(cleaned_text),
    }


def extract_candidate_from_document(
    source: DocumentSource,
    filename: str | None = None,
) -> CandidateInfo:
    """Parse a supported document and extract its candidate information."""
    return extract_candidate_info(extract_text(source, filename=filename))


def extract_name(text: str) -> str:
    """Return a conservative name candidate from the resume's opening lines."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines[:8]:
        labelled_name = _NAME_LABEL_PATTERN.match(line)
        if labelled_name and _is_plausible_name(labelled_name.group(1)):
            return labelled_name.group(1).strip()

    for line in lines[:5]:
        if _is_plausible_name(line):
            return line
    return ""


def extract_email(text: str) -> str:
    """Return the first common email address found, or an empty string."""
    match = EMAIL_PATTERN.search(text)
    return match.group(0) if match else ""


def extract_phone(text: str) -> str:
    """Return the first plausible international phone number found."""
    for match in PHONE_PATTERN.finditer(text):
        candidate = match.group(0).strip()
        digit_count = sum(character.isdigit() for character in candidate)
        if 10 <= digit_count <= 15:
            return candidate
    return ""


def extract_education(text: str) -> list[str]:
    """Return canonical degree and field names found in resume text."""
    found: list[str] = []
    for canonical_name, aliases in EDUCATION_ALIASES.items():
        if any(_contains_phrase(text, alias) for alias in aliases):
            found.append(canonical_name)
    return found


def estimate_experience_years(text: str) -> int | float | None:
    """Estimate experience from explicit statements such as '5 years experience'."""
    patterns = (
        r"\b(\d{1,2}(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?"
        r"(?:(?:professional|relevant|work|industry)\s+)?experience\b",
        r"\bexperience\s*(?:of|:)?\s*(\d{1,2}(?:\.\d+)?)\+?\s*(?:years?|yrs?)\b",
    )
    estimates: list[float] = []
    for pattern in patterns:
        estimates.extend(float(value) for value in re.findall(pattern, text, re.IGNORECASE))

    sensible_estimates = [value for value in estimates if 0 <= value <= 50]
    if not sensible_estimates:
        return None

    estimate = max(sensible_estimates)
    return int(estimate) if estimate.is_integer() else estimate


def _is_plausible_name(value: str) -> bool:
    """Check strict formatting rules used by conservative name extraction."""
    candidate = re.sub(r"\s+", " ", value).strip()
    if candidate.casefold() in _NAME_BLOCKLIST:
        return False
    if "@" in candidate or "http" in candidate.casefold():
        return False
    words = candidate.split()
    if any(word.casefold().strip(".'-") in _JOB_TITLE_WORDS for word in words):
        return False
    if not all(
        next((character for character in word if character.isalpha()), "").isupper()
        for word in words
    ):
        return False
    return _NAME_PATTERN.fullmatch(candidate) is not None


def _contains_phrase(text: str, phrase: str) -> bool:
    """Match an education phrase without finding it inside another word."""
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(phrase)}(?![A-Za-z0-9_])"
    flags = 0 if phrase == "BE" else re.IGNORECASE
    return re.search(pattern, text, flags) is not None
