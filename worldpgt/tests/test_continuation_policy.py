from worldpgt.continuation.continuation_policy import ContinuationPolicy
from worldpgt.continuation.types import ContinuationPrompt


def _parsed(term="bank", senses=("financial_institution", "river_edge")):
    return ContinuationPrompt(prompt="p", ambiguous_term=term, candidate_senses=list(senses))


def test_strong_top_score_continues():
    policy = ContinuationPolicy()
    decision = policy.decide(_parsed(), {"financial_institution": 3.0, "river_edge": 0.0})
    assert decision.decision == "continue"
    assert decision.selected_sense == "financial_institution"
    assert 0.0 <= decision.confidence <= 1.0


def test_zero_scores_audit():
    policy = ContinuationPolicy()
    decision = policy.decide(_parsed(), {"financial_institution": 0.0, "river_edge": 0.0})
    assert decision.decision == "audit"
    assert decision.selected_sense is None


def test_close_scores_audit():
    policy = ContinuationPolicy(min_margin=1.0)
    decision = policy.decide(_parsed(), {"financial_institution": 2.0, "river_edge": 1.5})
    assert decision.decision == "audit"


def test_no_ambiguous_term_audits():
    policy = ContinuationPolicy()
    parsed = ContinuationPrompt(prompt="p", ambiguous_term=None, candidate_senses=[])
    decision = policy.decide(parsed, {})
    assert decision.decision == "audit"
    assert "no_ambiguous_term" in decision.reasons


def test_no_candidate_senses_audits():
    policy = ContinuationPolicy()
    parsed = ContinuationPrompt(prompt="p", ambiguous_term="bank", candidate_senses=[])
    decision = policy.decide(parsed, {})
    assert decision.decision == "audit"
    assert "no_candidate_senses" in decision.reasons


def test_top_score_below_min_score_audits():
    policy = ContinuationPolicy(min_score=2.0)
    decision = policy.decide(_parsed(), {"financial_institution": 1.0, "river_edge": 0.0})
    assert decision.decision == "audit"


def test_banned_pattern_suppresses():
    policy = ContinuationPolicy(banned_patterns=["loan"])
    continuations = {
        "financial_institution": ["ask for a loan", "open an account"],
        "river_edge": ["cast his line"],
    }
    decision = policy.decide(
        _parsed(),
        {"financial_institution": 3.0, "river_edge": 0.0},
        continuations,
    )
    assert decision.decision == "suppress"
    assert decision.selected_sense == "financial_institution"
    assert any("banned_pattern" in reason for reason in decision.reasons)


def test_confidence_formula_is_deterministic():
    policy = ContinuationPolicy()
    decision = policy.decide(_parsed(), {"financial_institution": 3.0, "river_edge": 1.0})
    assert decision.confidence == min(1.0, 3.0 / (3.0 + 1.0 + 1.0))
