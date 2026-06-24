"""Support checks for explicit reasoning conclusions."""

from __future__ import annotations

from worldpgt.cognition.types import (
    AllowedConclusion,
    ConclusionSupportCheck,
    EvidenceWorkspace,
)


def validate_conclusion_support(
    conclusion: AllowedConclusion,
    workspace: EvidenceWorkspace,
    *,
    forbidden_claims: tuple[str, ...] = (),
) -> ConclusionSupportCheck:
    issues: list[str] = []
    text_norm = conclusion.text.lower()

    for forbidden in forbidden_claims:
        forbidden_norm = forbidden.strip().lower()
        if forbidden_norm and forbidden_norm in text_norm:
            issues.append(f"forbidden claim leaked: {forbidden}")

    if conclusion.kind == "profile_summary":
        allowed_terms = _workspace_terms(workspace, conclusion.evidence_roles)
        for term in _profile_summary_terms(conclusion.text):
            if term.lower() not in allowed_terms:
                issues.append(f"unsupported profile term: {term}")

    if conclusion.kind == "mechanism_gap":
        if "works by" in text_norm:
            issues.append("mechanism gap conclusion claims a mechanism")

    return ConclusionSupportCheck(
        conclusion_kind=conclusion.kind,
        ok=not issues,
        issues=tuple(issues),
    )


def validate_trace_conclusions(
    conclusions: tuple[AllowedConclusion, ...],
    workspace: EvidenceWorkspace,
    *,
    forbidden_claims: tuple[str, ...] = (),
) -> tuple[ConclusionSupportCheck, ...]:
    return tuple(
        validate_conclusion_support(
            conclusion,
            workspace,
            forbidden_claims=forbidden_claims,
        )
        for conclusion in conclusions
    )


def _workspace_terms(
    workspace: EvidenceWorkspace,
    roles,
) -> set[str]:
    terms: set[str] = set()
    for role in roles:
        for text in workspace.texts_for(role):
            for term in _extract_terms(text):
                terms.add(term.lower())
    return terms


def _profile_summary_terms(text: str) -> list[str]:
    marker = "through "
    if marker not in text:
        return []
    tail = text.split(marker, 1)[1].strip().rstrip(".")
    return _split_terms(tail)


def _extract_terms(text: str) -> list[str]:
    cleaned = str(text)
    for prefix in (
        "develops ",
        "produces ",
        "operates ",
        "publishes ",
        "provides ",
        "enables ",
        "uses ",
        "works by ",
        "is used for ",
        "is owned by ",
        "owns ",
        "is classified as ",
        "is known for ",
    ):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    cleaned = cleaned.replace(", including ", ", ")
    return _split_terms(cleaned)


def _split_terms(text: str) -> list[str]:
    return [
        part
        for part in (p.strip() for p in text.replace(" and ", ", ").split(","))
        if part and not part.endswith("other named items")
    ]
