"""Integration tests for the unified answer router.

Verifies dispatch to every branch, that the safety screen runs first, and —
critically — that the five confidence levels stay architecturally DISTINCT
(speculative_extended is never collapsed into speculative_verified, and the
degenerate-bridge guard keeps co-attribution renders coherent).

Slow-ish (builds the orchestrator + GloVe centroids once, module-scoped).
"""

from __future__ import annotations

import pytest

from worldpgt.reasoning.integrated_answer_router import (
    CONFIDENCE_CREATIVE,
    CONFIDENCE_GROUNDED,
    CONFIDENCE_GROUNDED_GENERATION,
    CONFIDENCE_SPECULATIVE_EXTENDED,
    CONFIDENCE_SPECULATIVE_VERIFIED,
    IntegratedAnswerRouter,
)


@pytest.fixture(scope="module")
def router():
    return IntegratedAnswerRouter(overlay_mode="promoted")


def test_qa_branch_grounded(router):
    a = router.answer("Who founded SpaceX?")
    assert a.branch == "qa"
    assert a.confidence_level == CONFIDENCE_GROUNDED
    assert a.support_kind == "grounded"
    assert a.detail["branch_support_kind"] == "semi_stable_relation"
    assert "Musk" in a.answer_text


def test_reflective_fast_path_verified(router):
    a = router.answer("What if Elon Musk had not founded SpaceX?")
    assert a.confidence_level == CONFIDENCE_SPECULATIVE_VERIFIED
    assert a.support_kind == "speculative_inference"
    assert "leader_of" not in a.answer_text
    assert "developing rockets" in a.answer_text
    assert "reasoning_trace" in a.detail


def test_co_attribution_is_extended_not_verified(router):
    a = router.answer("Why might SpaceX and Blue Origin be related?")
    assert a.confidence_level == CONFIDENCE_SPECULATIVE_EXTENDED
    assert a.support_kind == "speculative_extended"
    assert a.caution is not None


def test_co_attribution_render_is_coherent(router):
    # Regression for the degenerate undirected-bridge artifact
    # ("X develops O, which develops O").
    a = router.answer("Why might NASA and SpaceX be related?")
    assert "not directly linked" in a.answer_text
    assert "which develops spacecraft" not in a.answer_text  # the degenerate form


def test_association_paraphrase_to_verified_bridge(router):
    a = router.answer("Why might Musk be linked to electric cars?")
    assert a.confidence_level == CONFIDENCE_SPECULATIVE_VERIFIED
    assert "Tesla" in a.answer_text


def test_constrained_creative_grounded_generation(router):
    a = router.answer("Write about SpaceX using only these facts")
    assert a.branch == "constrained_creative"
    assert a.confidence_level == CONFIDENCE_GROUNDED_GENERATION


def test_pure_creative(router):
    a = router.answer("Compose a poem about rockets")
    assert a.confidence_level == CONFIDENCE_CREATIVE


@pytest.mark.parametrize("question", (
    "Write an imaginative tale about Mars",
    "Tell a fictional story about a space company",
    "Write a creative piece about the ocean",
))
def test_unambiguous_pure_creative_paraphrases_do_not_fall_back_to_qa(router, question):
    a = router.answer(question)
    assert a.branch == "pure_creative"
    assert a.support_kind == "creative_generated"


def test_hard_safety_screened_first(router):
    a = router.answer("Who is the current president?")
    assert a.branch == "qa_safety"
    assert a.decision == "audit"


def test_force_branch_override(router):
    a = router.answer("Who founded SpaceX?", force_branch="pure_creative")
    assert a.route_method == "override"
    assert a.branch in ("pure_creative", "qa")  # creative, or safe fallback


def test_all_confidence_levels_distinct(router):
    # A small battery must produce distinct labels, none collapsed.
    labels = {
        router.answer("Who founded SpaceX?").confidence_level,
        router.answer("What if Elon Musk had not founded SpaceX?").confidence_level,
        router.answer("Why might SpaceX and Blue Origin be related?").confidence_level,
        router.answer("Write about SpaceX using only these facts").confidence_level,
        router.answer("Compose a poem about rockets").confidence_level,
    }
    assert CONFIDENCE_SPECULATIVE_VERIFIED in labels
    assert CONFIDENCE_SPECULATIVE_EXTENDED in labels
    assert CONFIDENCE_GROUNDED in labels
    # verified and extended must not have merged into one
    assert CONFIDENCE_SPECULATIVE_VERIFIED != CONFIDENCE_SPECULATIVE_EXTENDED


def test_qa_identical_to_direct_orchestrator(router):
    from worldpgt.assistant_surface.answer_orchestrator import AnswerOrchestrator
    orch = AnswerOrchestrator("promoted")
    for q in ("Who founded SpaceX?", "What does Tesla produce?"):
        direct = orch.answer(q)
        integ = router.answer(q)
        assert integ.answer_text == direct.answer_text
        assert integ.support_kind == "grounded"
        assert integ.detail["branch_support_kind"] == direct.support_kind
