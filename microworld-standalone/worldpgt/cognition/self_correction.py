"""Self-correction for generated answer text.

This module is a surface guard. It reads a generated draft plus the speech plan
that produced it, critiques risky wording, and applies bounded repairs without
adding new facts.
"""

from __future__ import annotations

import re

from worldpgt.cognition.types import (
    AnswerDraft,
    CritiqueFinding,
    RepairAction,
    SelfCorrectionTrace,
)
from worldpgt.entity_qa.semantic_speech_planner import SpeechPlan


def self_correct_answer(draft_text: str, plan: SpeechPlan) -> SelfCorrectionTrace:
    """Critique and repair generated text against the current speech plan."""

    draft = AnswerDraft(text=draft_text)
    findings: list[CritiqueFinding] = []
    repairs: list[RepairAction] = []
    text = draft_text

    mechanism_repair = _repair_unsupported_mechanism(text, plan)
    if mechanism_repair is not None:
        text, finding, repair = mechanism_repair
        findings.append(finding)
        repairs.append(repair)

    deduped = _dedupe_sentences(text)
    if deduped != text:
        findings.append(
            CritiqueFinding(
                code="duplicate_sentence",
                message="the draft repeated the same sentence",
                severity="info",
            )
        )
        repairs.append(
            RepairAction(
                kind="dedupe_sentence",
                reason="removed repeated sentence without changing facts",
            )
        )
        text = deduped

    return SelfCorrectionTrace(
        draft=draft,
        findings=tuple(findings),
        repairs=tuple(repairs),
        final_text=text.strip(),
    )


def _repair_unsupported_mechanism(
    text: str,
    plan: SpeechPlan,
) -> tuple[str, CritiqueFinding, RepairAction] | None:
    if plan.mechanism:
        return None
    if not plan.subject or "works by" not in text.lower():
        return None

    pattern = re.compile(
        rf"\b{re.escape(plan.subject)}\s+works\s+by\s+([^.!?]+)([.!?])",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if not match:
        return None

    repaired_clause = _purpose_like_clause(match.group(1).strip())
    replacement = (
        f"{plan.subject} {repaired_clause}, but I do not have the mechanism "
        "evidence yet."
    )
    repaired = text[: match.start()] + replacement + text[match.end() :]
    return (
        repaired,
        CritiqueFinding(
            code="unsupported_mechanism_wording",
            message="'works by' implies mechanism evidence, but no mechanism bucket is present",
            severity="error",
        ),
        RepairAction(
            kind="replace_unsupported_mechanism",
            reason="rewrote mechanism wording as supported non-mechanism wording plus gap",
        ),
    )


def _purpose_like_clause(fragment: str) -> str:
    low = fragment.lower()
    if low.startswith("providing "):
        return f"provides {fragment[len('providing '):]}"
    if low.startswith("using "):
        return f"uses {fragment[len('using '):]}"
    if low.startswith("operating "):
        return f"operates {fragment[len('operating '):]}"
    return fragment


def _dedupe_sentences(text: str) -> str:
    protected = _protect_decimal_periods(text)
    parts = re.findall(r"[^.!?]+[.!?]|[^.!?]+$", protected)
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        sentence = _restore_decimal_periods(" ".join(part.strip().split()))
        if not sentence:
            continue
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(sentence)
    return " ".join(out)


def _protect_decimal_periods(text: str) -> str:
    return re.sub(r"(?<=\d)\.(?=\d)", "<DECIMAL_DOT>", text)


def _restore_decimal_periods(text: str) -> str:
    return text.replace("<DECIMAL_DOT>", ".")
