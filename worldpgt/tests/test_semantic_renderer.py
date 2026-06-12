from __future__ import annotations

from worldpgt.continuation.semantic_frame import FrameCandidate, build_semantic_frame
from worldpgt.continuation.semantic_renderer import (
    generate_frame_candidates,
    rank_frame_candidates,
)


def _best(prompt: str, term: str, sense_id: str) -> FrameCandidate | None:
    frame = build_semantic_frame(prompt, term, sense_id, [])
    return rank_frame_candidates(prompt, generate_frame_candidates(prompt, frame))


def test_financial_to_prompt_produces_clean_transaction_continuation():
    prompt = "The customer reached the bank teller with cash to"
    best = _best(prompt, "bank", "financial_institution")
    assert best is not None
    assert best.text.endswith("to open an account")


def test_river_as_current_prompt_produces_shore_or_current_continuation():
    prompt = "The boat drifted toward the bank as the current"
    best = _best(prompt, "bank", "river_edge")
    assert best is not None
    assert best.text.endswith("carried it downstream")


def test_sports_bat_prompt_avoids_duplicated_hit():
    prompt = "The bat cracked when he hit it during the game"
    best = _best(prompt, "bat", "sports_equipment")
    assert best is not None
    # Must not append another "hit ... ball" after the prompt already used "hit".
    assert not best.text.endswith("hit the ball")


def test_animal_seal_prompt_avoids_to_swam():
    prompt = "The seal swam through ocean water chasing fish to"
    best = _best(prompt, "seal", "animal")
    assert best is not None
    assert "to swam" not in best.text
    assert best.text.endswith("to catch another fish")


def test_repeated_noun_candidate_loses_to_novel_candidate():
    prompt = "In April the spring rain warmed the garden and"
    best = _best(prompt, "spring", "season")
    assert best is not None
    # "the garden filled with flowers" repeats "garden"; the novel phrase wins.
    assert "garden" not in best.text[len(prompt):]


def test_candidate_with_story_drift_is_never_selected():
    frame = build_semantic_frame("The customer reached the bank to", "bank", "financial_institution", [])
    clean = FrameCandidate(
        text="The customer reached the bank to open an account",
        frame=frame,
        renderer_name="test",
        score=0.0,
        reasons=["connector_match=infinitive_after_to"],
    )
    drift = FrameCandidate(
        text='The customer reached the bank to ask his mother',
        frame=frame,
        renderer_name="test",
        score=0.0,
        reasons=["connector_match=infinitive_after_to", "drift", "drift_marker=mother"],
    )
    best = rank_frame_candidates("The customer reached the bank to", [drift, clean])
    assert best is clean


def test_all_unsafe_candidates_returns_none():
    frame = build_semantic_frame("The river bank", "bank", "river_edge", [])
    bad = FrameCandidate(
        text="The river bank current cast his line",
        frame=frame,
        renderer_name="test",
        score=0.0,
        reasons=["surface_invalid", "surface_pattern=current_cast_his_line"],
    )
    assert rank_frame_candidates("The river bank", [bad]) is None


def test_renderer_is_deterministic():
    prompt = "The boat drifted toward the bank as the current"
    frame = build_semantic_frame(prompt, "bank", "river_edge", [])
    first = rank_frame_candidates(prompt, generate_frame_candidates(prompt, frame))
    for _ in range(5):
        again = rank_frame_candidates(prompt, generate_frame_candidates(prompt, frame))
        assert again.text == first.text
