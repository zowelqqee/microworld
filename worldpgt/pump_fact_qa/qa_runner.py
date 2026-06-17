"""Run generated prompts through the assistant surface and validate answers.

The runner is deterministic and local-only: it calls the in-process assistant
surface with ``overlay_mode='pump-dry-run'``. It never touches the network and
never mutates any overlay or memory file.
"""

from __future__ import annotations

from typing import Any, Callable

from worldpgt.pump_fact_qa.answer_validator import (
    validate_adversarial,
    validate_current,
    validate_positive,
)
from worldpgt.pump_fact_qa.types import (
    CATEGORY_ADVERSARIAL,
    CATEGORY_CURRENT,
    CATEGORY_POSITIVE,
    CRITICAL_CLASSIFICATIONS,
    QAPrompt,
    QAResult,
)

PUMP_OVERLAY_MODE = "pump-dry-run"

# Signature: (question, overlay_mode) -> object with .to_dict() or a dict.
AskFn = Callable[[str, str], Any]


def _answer_to_dict(answer: Any) -> dict[str, Any]:
    if isinstance(answer, dict):
        return answer
    if hasattr(answer, "to_dict"):
        return answer.to_dict()
    raise TypeError(f"Unsupported answer object: {type(answer)!r}")


def _validate(prompt: QAPrompt, answer: dict[str, Any]) -> tuple[bool, str]:
    if prompt.category == CATEGORY_POSITIVE:
        return validate_positive(prompt, answer)
    if prompt.category == CATEGORY_ADVERSARIAL:
        return validate_adversarial(prompt, answer)
    if prompt.category == CATEGORY_CURRENT:
        return validate_current(prompt, answer)
    raise ValueError(f"Unknown prompt category: {prompt.category}")


def run_prompts(prompts: list[QAPrompt], ask_fn: AskFn) -> list[QAResult]:
    """Run each prompt through ``ask_fn`` and validate the answer."""
    results: list[QAResult] = []
    for prompt in prompts:
        answer = _answer_to_dict(ask_fn(prompt.question, PUMP_OVERLAY_MODE))
        valid, classification = _validate(prompt, answer)
        results.append(
            QAResult(
                prompt=prompt,
                decision=str(answer.get("decision", "")),
                support_kind=str(answer.get("support_kind", "")),
                supported_by_context=bool(answer.get("supported_by_context")),
                overlay_mode=str(answer.get("overlay_mode", "")),
                answer_text=str(answer.get("answer_text", "")),
                risk_flags=list(answer.get("risk_flags", [])),
                valid=valid,
                classification=classification,
                is_critical_failure=classification in CRITICAL_CLASSIFICATIONS,
            )
        )
    return results
