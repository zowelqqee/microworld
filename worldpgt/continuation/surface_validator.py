"""Deterministic surface validation for realized continuations."""

from __future__ import annotations

from dataclasses import dataclass, field
import re


@dataclass
class SurfaceValidationResult:
    ok: bool
    risk_score: float
    reasons: list[str] = field(default_factory=list)
    matched_patterns: list[str] = field(default_factory=list)


_BAD_PATTERNS = [
    ("to_closed", re.compile(r"\bto closed\b", re.IGNORECASE)),
    ("to_swam", re.compile(r"\bto swam\b", re.IGNORECASE)),
    ("to_lifted", re.compile(r"\bto lifted\b", re.IGNORECASE)),
    ("to_brought", re.compile(r"\bto brought\b", re.IGNORECASE)),
    ("current_cast_his_line", re.compile(r"\bcurrent cast his line\b", re.IGNORECASE)),
    ("repeated_subject", re.compile(r"\bthe\s+([a-z0-9']+)\s+the\s+\1\b", re.IGNORECASE)),
    (
        "malformed_until_subject",
        re.compile(r"\buntil\s+the\s+[a-z0-9']+\s+(the|it|its)\b", re.IGNORECASE),
    ),
    (
        "malformed_before_it_operator",
        re.compile(r"\bbefore\s+it\s+lifted\s+steel\s+the\s+operator\b", re.IGNORECASE),
    ),
    (
        "malformed_before_wings",
        re.compile(r"\bbefore\s+its\s+wings\s+spread\s+its\s+wings\b", re.IGNORECASE),
    ),
    (
        "connector_subject_and",
        re.compile(r"\b(until|before|when|after|while)\s+the\s+[a-z0-9']+\s+and\b", re.IGNORECASE),
    ),
    (
        "repeated_spread_wings",
        re.compile(r"\bspread its wings\b.*\bspread its wings\b", re.IGNORECASE),
    ),
    ("repeated_hit", re.compile(r"\bhit\b.*\bhit the ball\b", re.IGNORECASE)),
    (
        "repeated_open_account",
        re.compile(r"\bopened an account\b.*\bopen an account\b", re.IGNORECASE),
    ),
    ("repeated_lifted", re.compile(r"\blifted\b.*\blifted\b", re.IGNORECASE)),
]

_BOUNDARY_RE = re.compile(
    r"\b(until|before|when)\s+the\s+([a-z0-9']+)\s*$",
    re.IGNORECASE,
)


def validate_surface_text(prompt: str, continuation: str) -> SurfaceValidationResult:
    matched = [
        name
        for name, pattern in _BAD_PATTERNS
        if pattern.search(continuation)
    ]
    if _has_prompt_boundary_repeated_subject(prompt, continuation):
        matched.append("prompt_boundary_repeated_subject")

    reasons = [f"surface_pattern={name}" for name in matched]
    return SurfaceValidationResult(
        ok=not matched,
        risk_score=float(len(matched)),
        reasons=reasons,
        matched_patterns=matched,
    )


def _has_prompt_boundary_repeated_subject(prompt: str, continuation: str) -> bool:
    match = _BOUNDARY_RE.search(prompt.strip())
    if match is None:
        return False
    expected_start = f"the {match.group(2).lower()}"
    if continuation.startswith(prompt):
        suffix = continuation[len(prompt) :].strip().lower()
    else:
        suffix = continuation.strip().lower()
    return suffix.startswith(expected_start)
