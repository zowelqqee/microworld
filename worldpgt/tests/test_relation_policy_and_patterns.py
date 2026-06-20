"""Tests for relation_policy (Bug 2 + Bug 3) and volatile PatternSpec entries (Bug 1).

Covers:
  - relation_policy: CURRENT_SENSITIVE_PREDICATES expansion, RENDERING_HEDGE_PREDICATES,
    VOLATILE_STABILITY_PREDICATES, and the three helper functions
  - Volatile PatternSpec entries in relation_patterns: leader_of and valued_at
    patterns exist, carry stability="volatile", risk="high"
  - path_validator now imports from relation_policy (behaviour test: valued_at blocked)
  - entity_answer_renderer uses should_hedge_render, hedging fires for valued_at too
"""

from __future__ import annotations

import pytest

from worldpgt.relation_extraction_v2.relation_policy import (
    CURRENT_SENSITIVE_PREDICATES,
    RENDERING_HEDGE_PREDICATES,
    VOLATILE_STABILITY_PREDICATES,
    is_current_sensitive,
    is_volatile_predicate,
    freshness_window_days,
    should_hedge_render,
)
from worldpgt.relation_extraction_v2.relation_patterns import PATTERNS
from worldpgt.relation_extraction_v2.types import (
    ALLOWED_RELATIONS,
    ALLOWED_VOLATILE_RELATIONS,
)


# ---------------------------------------------------------------------------
# Bug 3 — CURRENT_SENSITIVE_PREDICATES expansion
# ---------------------------------------------------------------------------

class TestCurrentSensitivePredicates:
    def test_leader_of_is_current_sensitive(self):
        assert "leader_of" in CURRENT_SENSITIVE_PREDICATES

    def test_valued_at_is_current_sensitive(self):
        assert "valued_at" in CURRENT_SENSITIVE_PREDICATES

    def test_founded_by_not_current_sensitive(self):
        assert "founded_by" not in CURRENT_SENSITIVE_PREDICATES

    def test_is_a_not_current_sensitive(self):
        assert "is_a" not in CURRENT_SENSITIVE_PREDICATES

    def test_owned_by_not_current_sensitive(self):
        assert "owned_by" not in CURRENT_SENSITIVE_PREDICATES

    def test_helper_leader_of(self):
        assert is_current_sensitive("leader_of") is True

    def test_helper_valued_at(self):
        assert is_current_sensitive("valued_at") is True

    def test_helper_develops_false(self):
        assert is_current_sensitive("develops") is False


class TestFreshnessWindows:
    def test_estimated_net_worth_window_30_days(self):
        assert freshness_window_days("estimated_net_worth", "snapshot") == 30

    def test_population_window_365_days(self):
        assert freshness_window_days("population", "snapshot") == 365

    def test_ranking_window_90_days(self):
        assert freshness_window_days("ranking", "snapshot") == 90

    def test_owned_by_snapshot_window_180_days(self):
        assert freshness_window_days("owned_by", "snapshot") == 180

    def test_historical_never_stales(self):
        assert freshness_window_days("founded_by", "historical") == float("inf")


# ---------------------------------------------------------------------------
# Bug 2 — RENDERING_HEDGE_PREDICATES and helpers
# ---------------------------------------------------------------------------

class TestRenderingHedgePredicates:
    def test_leader_of_hedge(self):
        assert "leader_of" in RENDERING_HEDGE_PREDICATES

    def test_valued_at_hedge(self):
        assert "valued_at" in RENDERING_HEDGE_PREDICATES

    def test_founded_by_no_hedge(self):
        assert "founded_by" not in RENDERING_HEDGE_PREDICATES

    def test_helper_leader_of(self):
        assert should_hedge_render("leader_of") is True

    def test_helper_valued_at(self):
        assert should_hedge_render("valued_at") is True

    def test_helper_develops_false(self):
        assert should_hedge_render("develops") is False


# ---------------------------------------------------------------------------
# Bug 1 — VOLATILE_STABILITY_PREDICATES and ALLOWED_VOLATILE_RELATIONS
# ---------------------------------------------------------------------------

class TestVolatileStabilityPredicates:
    def test_leader_of_volatile(self):
        assert "leader_of" in VOLATILE_STABILITY_PREDICATES

    def test_valued_at_volatile(self):
        assert "valued_at" in VOLATILE_STABILITY_PREDICATES

    def test_helper_leader_of(self):
        assert is_volatile_predicate("leader_of") is True

    def test_helper_is_a_false(self):
        assert is_volatile_predicate("is_a") is False

    def test_allowed_volatile_contains_leader_of(self):
        assert "leader_of" in ALLOWED_VOLATILE_RELATIONS

    def test_allowed_volatile_contains_valued_at(self):
        assert "valued_at" in ALLOWED_VOLATILE_RELATIONS

    def test_allowed_relations_contains_volatile(self):
        assert "leader_of" in ALLOWED_RELATIONS
        assert "valued_at" in ALLOWED_RELATIONS


# ---------------------------------------------------------------------------
# Bug 1 — volatile PatternSpec entries exist in PATTERNS
# ---------------------------------------------------------------------------

class TestVolatilePatternSpecs:
    """leader_of and valued_at patterns must exist with volatile stability and high risk."""

    def _patterns_for(self, predicate: str):
        return [p for p in PATTERNS if p.relation == predicate]

    def test_leader_of_patterns_exist(self):
        assert self._patterns_for("leader_of"), "No PatternSpec for leader_of"

    def test_valued_at_patterns_exist(self):
        assert self._patterns_for("valued_at"), "No PatternSpec for valued_at"

    def test_leader_of_all_volatile(self):
        for p in self._patterns_for("leader_of"):
            assert p.stability == "volatile", f"{p.pattern_id} stability={p.stability}"

    def test_valued_at_all_volatile(self):
        for p in self._patterns_for("valued_at"):
            assert p.stability == "volatile", f"{p.pattern_id} stability={p.stability}"

    def test_leader_of_all_high_risk(self):
        for p in self._patterns_for("leader_of"):
            assert p.risk == "high", f"{p.pattern_id} risk={p.risk}"

    def test_valued_at_all_high_risk(self):
        for p in self._patterns_for("valued_at"):
            assert p.risk == "high", f"{p.pattern_id} risk={p.risk}"

    def test_is_ceo_of_pattern_exists(self):
        ids = {p.pattern_id for p in PATTERNS}
        assert "is_ceo_of" in ids

    def test_valued_at_passive_pattern_exists(self):
        ids = {p.pattern_id for p in PATTERNS}
        assert "valued_at_passive" in ids

    def test_leads_active_pattern_exists(self):
        ids = {p.pattern_id for p in PATTERNS}
        assert "leads_active" in ids

    def test_has_valuation_of_pattern_exists(self):
        ids = {p.pattern_id for p in PATTERNS}
        assert "has_valuation_of" in ids


# ---------------------------------------------------------------------------
# Bug 1 — volatile patterns actually match expected text
# ---------------------------------------------------------------------------

class TestVolatilePatternMatches:
    def _match(self, pattern_id: str, text: str):
        for p in PATTERNS:
            if p.pattern_id == pattern_id:
                m = p.regex.search(text)
                if m:
                    return m.groupdict()
        return None

    def test_is_ceo_of_matches_elon_tesla(self):
        m = self._match("is_ceo_of", "Elon Musk is CEO of Tesla")
        assert m is not None
        assert "Elon Musk" in m.get("A", "")

    def test_is_ceo_of_matches_president(self):
        m = self._match("is_ceo_of", "Gwynne Shotwell is president of SpaceX")
        assert m is not None

    def test_is_ceo_of_matches_chairman(self):
        m = self._match("is_ceo_of", "Elon Musk is chairman of X Corp")
        assert m is not None

    def test_leads_active_matches(self):
        m = self._match("leads_active", "Elon Musk leads SpaceX")
        assert m is not None

    def test_valued_at_passive_billions(self):
        m = self._match("valued_at_passive", "SpaceX is valued at $150 billion")
        assert m is not None
        assert "SpaceX" in m.get("A", "")

    def test_valued_at_passive_million(self):
        m = self._match("valued_at_passive", "Rivian was valued at $66 billion")
        assert m is not None

    def test_has_valuation_of_matches(self):
        m = self._match("has_valuation_of", "SpaceX has a valuation of $200 billion")
        assert m is not None

    def test_stable_pattern_still_stable(self):
        """Existing stable patterns must not have been accidentally changed."""
        for p in PATTERNS:
            if p.relation in ("is_a", "type_of"):
                assert p.stability == "stable", f"{p.pattern_id} became {p.stability}"


# ---------------------------------------------------------------------------
# Bug 2 — path_validator uses shared policy (valued_at now blocked)
# ---------------------------------------------------------------------------

class TestPathValidatorUsesPolicy:
    """Confirm path_validator blocks valued_at as current-sensitive."""

    def _make_hop(self, predicate: str, stability: str = "semi_stable", risk: str = "low"):
        from worldpgt.multihop_qa.types import HopEdge
        return HopEdge(
            subject="A",
            predicate=predicate,
            object="B",
            stability=stability,
            risk=risk,
        )

    def _make_question(self, chain_type: str = "connection"):
        from worldpgt.multihop_qa.types import MultihopQuestion
        return MultihopQuestion(
            question="How is A connected to B?",
            chain_type=chain_type,
            is_direct_relation=False,
        )

    def _make_path(self, hop1_pred: str, hop2_pred: str):
        from worldpgt.multihop_qa.types import MultihopPath
        return MultihopPath(
            hop1=self._make_hop(hop1_pred),
            hop2=self._make_hop(hop2_pred),
            via_node="B",
        )

    def test_valued_at_hop1_blocked(self):
        from worldpgt.multihop_qa.path_validator import validate_path
        path = self._make_path("valued_at", "founded_by")
        q = self._make_question()
        valid, reason = validate_path(path, q)
        assert not valid
        # volatile check fires before current_sensitive check
        assert "valued_at" in reason

    def test_valued_at_hop2_blocked(self):
        from worldpgt.multihop_qa.path_validator import validate_path
        path = self._make_path("founded_by", "valued_at")
        q = self._make_question()
        valid, reason = validate_path(path, q)
        assert not valid
        assert "valued_at" in reason

    def test_leader_of_still_blocked(self):
        from worldpgt.multihop_qa.path_validator import validate_path
        path = self._make_path("leader_of", "founded_by")
        q = self._make_question()
        valid, reason = validate_path(path, q)
        assert not valid
        assert "leader_of" in reason

    def test_safe_chain_still_passes(self):
        from worldpgt.multihop_qa.path_validator import validate_path
        path = self._make_path("founded_by", "develops")
        q = self._make_question()
        valid, reason = validate_path(path, q)
        assert valid
        assert reason is None


# ---------------------------------------------------------------------------
# Bug 2 — entity_answer_renderer uses should_hedge_render for valued_at too
# ---------------------------------------------------------------------------

class TestEntityAnswerRendererHedging:
    """Confirm hedged language fires for both leader_of and valued_at."""

    def _make_plan_with_relation(self, predicate: str, subject: str = "SpaceX", obj: str = "Elon Musk"):
        from worldpgt.entity_qa.types import (
            AnalyzedEntityQuestion,
            EntityQAEvidence,
            EntityQAPlan,
        )
        analyzed = AnalyzedEntityQuestion(
            question=f"What is the {predicate} of {subject}?",
            intent="relation_lookup",
            subject=subject,
            predicate_hint=predicate,
            secondary_entity=None,
            source_hint=None,
            is_current_query=False,
            is_unsupported=False,
        )
        return EntityQAPlan(
            analyzed=analyzed,
            decision="answer",
            audit_reason=None,
            evidence=EntityQAEvidence(),
            render_template="relation_lookup",
            render_args={
                "subject": subject,
                "predicate": predicate,
                "relations": [{"predicate": predicate, "object": obj}],
            },
            confidence=1.0,
        )

    def test_leader_of_renders_hedged(self):
        from worldpgt.entity_qa.entity_answer_renderer import render
        plan = self._make_plan_with_relation("leader_of", "SpaceX", "Elon Musk")
        result = render(plan)
        assert "linked" in result.lower() and "leadership" in result.lower()

    def test_valued_at_renders_hedged(self):
        from worldpgt.entity_qa.entity_answer_renderer import render
        plan = self._make_plan_with_relation("valued_at", "SpaceX", "$150 billion")
        result = render(plan)
        # should_hedge_render("valued_at") is True — hedged language applies
        assert "linked" in result.lower() and "leadership" in result.lower()

    def test_develops_not_hedged(self):
        from worldpgt.entity_qa.entity_answer_renderer import render
        plan = self._make_plan_with_relation("develops", "SpaceX", "Starship")
        result = render(plan)
        assert "leadership" not in result.lower()
        assert "develops" in result.lower()
