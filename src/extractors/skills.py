"""Deterministic taxonomy and matching helpers for technical skills."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence


# Each key is the name returned by ``extract_skills``. Values include accepted
# spellings and abbreviations. Add another entry here to extend the taxonomy.
SKILL_ALIASES: dict[str, tuple[str, ...]] = {
    "Python": ("Python",),
    "Java": ("Java",),
    "C": ("C",),
    "C++": ("C++",),
    "C#": ("C#",),
    "R": ("R",),
    "JavaScript": ("JavaScript",),
    "TypeScript": ("TypeScript",),
    "React": ("React",),
    "Angular": ("Angular",),
    "Vue": ("Vue",),
    "Node.js": ("Node.js", "NodeJS"),
    "Express.js": ("Express.js", "ExpressJS"),
    "MongoDB": ("MongoDB",),
    "MySQL": ("MySQL",),
    "PostgreSQL": ("PostgreSQL", "Postgres"),
    "SQL": ("SQL",),
    "Redis": ("Redis",),
    "AWS": ("AWS", "Amazon Web Services"),
    "Azure": ("Azure",),
    "Google Cloud": ("Google Cloud", "GCP", "Google Cloud Platform"),
    "Docker": ("Docker",),
    "Kubernetes": ("Kubernetes", "K8s"),
    "Git": ("Git",),
    "GitHub": ("GitHub",),
    "Linux": ("Linux",),
    "Machine Learning": ("Machine Learning",),
    "Deep Learning": ("Deep Learning",),
    "Natural Language Processing": ("Natural Language Processing", "NLP"),
    "Computer Vision": ("Computer Vision",),
    "TensorFlow": ("TensorFlow",),
    "PyTorch": ("PyTorch",),
    "scikit-learn": ("scikit-learn", "scikit learn", "sklearn"),
    "pandas": ("pandas",),
    "NumPy": ("NumPy",),
    "Generative AI": ("Generative AI", "GenAI", "Gen AI"),
    "Large Language Models": ("Large Language Models", "Large Language Model", "LLM", "LLMs"),
    "RAG": ("RAG", "Retrieval Augmented Generation"),
    "LangChain": ("LangChain",),
    "REST API": ("REST API", "REST APIs"),
    "FastAPI": ("FastAPI",),
    "Flask": ("Flask",),
    "Django": ("Django",),
    "HTML": ("HTML",),
    "CSS": ("CSS",),
    "Streamlit": ("Streamlit",),
}

# Flat, duplicate-free view useful for forms, reports, and inspection.
SKILLS: tuple[str, ...] = tuple(SKILL_ALIASES)


def extract_skills(
    text: str,
    taxonomy: Mapping[str, Sequence[str]] | None = None,
) -> list[str]:
    """Return canonical skills found in ``text`` without duplicates.

    Matching is case-insensitive and uses token-aware regular expressions.
    Callers may provide a custom ``taxonomy`` mapping canonical skill names to
    accepted aliases. Results follow taxonomy order, making output stable.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    selected_taxonomy = SKILL_ALIASES if taxonomy is None else taxonomy
    found: list[str] = []
    seen: set[str] = set()

    for canonical_name, aliases in selected_taxonomy.items():
        candidates = _unique_aliases((canonical_name, *aliases))
        if any(_contains_alias(text, alias) for alias in candidates):
            identity = canonical_name.casefold()
            if identity not in seen:
                found.append(canonical_name)
                seen.add(identity)

    return found


def _unique_aliases(aliases: Sequence[str]) -> tuple[str, ...]:
    """Remove repeated aliases case-insensitively, longest aliases first."""
    unique = {alias.strip().casefold(): alias.strip() for alias in aliases if alias.strip()}
    return tuple(sorted(unique.values(), key=lambda alias: (-len(alias), alias.casefold())))


def _contains_alias(text: str, alias: str) -> bool:
    """Match one alias as a complete technical token or phrase."""
    escaped_alias = re.escape(alias)

    # Plain C needs an extra guard so C++ and C# do not also imply C.
    if alias.casefold() == "c":
        pattern = rf"(?<![A-Za-z0-9_]){escaped_alias}(?![A-Za-z0-9_+#])"
    else:
        pattern = rf"(?<![A-Za-z0-9_]){escaped_alias}(?![A-Za-z0-9_])"

    return re.search(pattern, text, flags=re.IGNORECASE) is not None
