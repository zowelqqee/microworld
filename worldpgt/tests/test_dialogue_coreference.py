"""Tests for in-memory dialogue coreference resolution."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from worldpgt.dialogue.conversation_context import ConversationContext, ConversationTurn
from worldpgt.dialogue.coreference_resolver import resolve_coreferences
from worldpgt.dialogue.followup_rewriter import rewrite_followup
from worldpgt.entity_qa.types import SemanticQuery
from worldpgt.relation_extraction_v2.entity_surface_index import EntitySurfaceIndex

_ROOT = Path(__file__).resolve().parent.parent.parent
_CLI = _ROOT / "worldpgt" / "experiments" / "ask_microworld_v1.py"
_ACCEPTED_OVERLAY_PATH = _ROOT / "worldpgt" / "experiments" / "accepted_wiki_memory_overlay_v1.json"
_PROMOTED_OVERLAY_PATH = (
    _ROOT
    / "worldpgt"
    / "experiments"
    / "self_ingestion_v1"
    / "promotion"
    / "promoted_wiki_memory_overlay_v1.json"
)
_SNAPSHOT_OVERLAY_PATH = (
    _ROOT / "worldpgt" / "experiments" / "wiki_snapshot_ingestion_v1" / "snapshot_dry_run_overlay.json"
)


def _index() -> EntitySurfaceIndex:
    return EntitySurfaceIndex(
        accepted_overlay_path=_ACCEPTED_OVERLAY_PATH,
        promoted_overlay_path=_PROMOTED_OVERLAY_PATH,
        snapshot_overlay_path=_SNAPSHOT_OVERLAY_PATH,
    )


def _semantic(
    entity_a: str | None,
    relation: str | None = None,
    entity_b: str | None = None,
) -> SemanticQuery:
    return SemanticQuery(
        entity_a=entity_a,
        entity_b=entity_b,
        relation_intent=relation,
        unknown_position="relation",
        query_type="lookup",
        confidence=0.9,
    )


def _turn(
    question: str,
    primary: str | None,
    mentioned: list[str],
    relation: str | None = None,
    decision: str = "answer",
) -> ConversationTurn:
    return ConversationTurn(
        question=question,
        semantic_query=_semantic(primary, relation),
        decision=decision,
        primary_entity=primary,
        mentioned_entities=mentioned,
        relation_type=relation,
    )


def test_pronoun_resolves_to_last_matching_person_in_context():
    context = ConversationContext(
        turns=[
            _turn(
                "Who founded SpaceX?",
                primary="SpaceX",
                mentioned=["SpaceX", "Elon Musk"],
                relation="founded_by",
            )
        ]
    )

    result = resolve_coreferences("What else did he found?", context, _index())

    assert result.resolved_question == "What else did Elon Musk found?"
    assert result.replacements[0].reference == "he"
    assert result.replacements[0].entities == ("Elon Musk",)


def test_descriptor_prefers_recent_primary_organization():
    context = ConversationContext(
        turns=[
            _turn("Who founded SpaceX?", "SpaceX", ["SpaceX", "Elon Musk"], "founded_by"),
            _turn("What else did he found?", "Elon Musk", ["Elon Musk", "SpaceX"]),
            _turn("Is he a businessman?", "Elon Musk", ["Elon Musk"], "is_a"),
        ]
    )

    result = resolve_coreferences("What does the company develop?", context, _index())

    assert result.resolved_question == "What does SpaceX develop?"
    assert result.replacements[0].reference == "the company"
    assert result.replacements[0].entities == ("SpaceX",)


def test_it_resolves_to_latest_primary_thing_entity():
    context = ConversationContext(
        turns=[
            _turn(
                "What does SpaceX develop?",
                primary="SpaceX",
                mentioned=["SpaceX", "Falcon 9"],
                relation="develops",
            )
        ]
    )

    result = resolve_coreferences("How is it connected to Falcon 9?", context, _index())

    assert result.resolved_question == "How is SpaceX connected to Falcon 9?"
    assert result.replacements[0].entities == ("SpaceX",)


def test_plural_reference_resolves_to_last_two_primary_entities():
    context = ConversationContext(
        turns=[
            _turn("Who founded SpaceX?", "SpaceX", ["SpaceX", "Elon Musk"], "founded_by"),
            _turn("Who founded Blue Origin?", "Blue Origin", ["Blue Origin", "Jeff Bezos"], "founded_by"),
        ]
    )

    result = resolve_coreferences("What do they have in common?", context, _index())

    assert result.resolved_question == "What do SpaceX and Blue Origin have in common?"
    assert result.replacements[0].entities == ("SpaceX", "Blue Origin")


def test_ambiguous_reference_is_left_unresolved():
    context = ConversationContext(
        turns=[
            _turn(
                "Who founded OpenAI?",
                primary="OpenAI",
                mentioned=["OpenAI", "Elon Musk", "Jeff Bezos"],
                relation="founded_by",
            )
        ]
    )

    result = resolve_coreferences("What did he found?", context, _index())

    assert result.resolved_question == "What did he found?"
    assert result.unresolved_reference == "he"
    assert result.replacements == ()


def test_followup_rewriter_expands_bare_entity_turn():
    context = ConversationContext(
        turns=[
            _turn(
                "Tell me about SpaceX.",
                primary="SpaceX",
                mentioned=["SpaceX", "Starlink"],
            )
        ]
    )

    result = rewrite_followup("а Starlink?", context, _index())

    assert result.resolved_question == "Tell me about Starlink."
    assert result.answer_style == "followup"
    assert result.reason == "bare_entity_followup"


def test_followup_rewriter_does_not_truncate_a_complete_new_question():
    """"Tell me about SpaceX" is a complete, self-contained question — it
    must not be misclassified as a short contextual continuation just
    because it is <=4 words and happens to name a known entity. Before the
    fix this forced answer_style="followup" (max_detail_units=1, no
    reasoning trailer), silently truncating an unrelated fresh question
    after any prior turn in the same session.
    """

    context = ConversationContext(
        turns=[
            _turn(
                "How does Starlink work?",
                primary="Starlink",
                mentioned=["Starlink", "SpaceX"],
            )
        ]
    )

    result = rewrite_followup("Tell me about SpaceX", context, _index())

    assert result.resolved_question == "Tell me about SpaceX"
    assert result.answer_style == "normal"
    assert result.reason is None


def test_followup_rewriter_supplies_last_subject_for_founding_question():
    context = ConversationContext(
        turns=[
            ConversationTurn(
                question="Tell me about Blue Origin.",
                semantic_query=_semantic("Blue Origin"),
                decision="answer",
                primary_entity="Jeff Bezos",
                mentioned_entities=["Blue Origin", "Jeff Bezos"],
                relation_type=None,
            )
        ]
    )

    result = rewrite_followup("кто основал?", context, _index())

    assert result.resolved_question == "Who founded Blue Origin?"
    assert result.answer_style == "followup"
    assert result.reason == "omitted_subject_founding"


def test_entities_in_window_skips_audit_turns():
    """Audit turns must not consume window slots; confirmed entities persist."""
    ctx = ConversationContext()
    ctx.turns = [
        _turn("What is SpaceX?", "SpaceX", ["SpaceX", "Elon Musk"]),
        _turn("audit 1", None, [], decision="audit"),
        _turn("audit 2", None, [], decision="audit"),
        _turn("audit 3", None, [], decision="audit"),
    ]
    window = ctx.entities_in_window(n=3)
    assert "SpaceX" in window
    assert "Elon Musk" in window


def test_entities_in_window_respects_n_confirmed_turns():
    """Window of n=2 should only include last 2 non-audit turns."""
    ctx = ConversationContext()
    ctx.turns = [
        _turn("turn 0", "Entity0", []),
        _turn("turn 1", "Entity1", []),
        _turn("turn 2", "Entity2", []),
    ]
    window = ctx.entities_in_window(n=2)
    assert "Entity2" in window
    assert "Entity1" in window
    assert "Entity0" not in window


def test_he_resolves_to_person_after_audit_turns():
    """'he' must resolve to Elon Musk even when audit turns follow Q2."""
    ctx = ConversationContext()
    ctx.turns = [
        _turn("What is SpaceX?", "SpaceX", ["SpaceX"]),
        # After "Who founded SpaceX?" → Elon Musk is primary + mentioned
        _turn("Who founded SpaceX?", "Elon Musk", ["Elon Musk", "SpaceX"], "founded_by"),
        _turn("audit 1", None, [], decision="audit"),
        _turn("audit 2", None, [], decision="audit"),
        _turn("audit 3", None, [], decision="audit"),
    ]

    result = resolve_coreferences("What else did he found?", ctx, _index())

    assert result.resolved_question == "What else did Elon Musk found?"
    assert result.replacements[0].reference == "he"
    assert result.replacements[0].entities == ("Elon Musk",)


def test_interactive_cli_reports_unresolved_reference_before_guessing():
    proc = subprocess.run(
        [sys.executable, str(_CLI), "--interactive"],
        input="What did he found?\nquit\n",
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    assert "unresolved_reference: could not determine what 'he' refers to" in proc.stdout
    assert "Decision: audit." in proc.stdout
