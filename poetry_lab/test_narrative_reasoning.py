from __future__ import annotations

import unittest

from poemcore.narrative_reasoning import event_hypothesis
from poemcore.plan_search import beam_search
from poemcore.world_state import StateFact, WorldState


class NarrativeHypothesisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = WorldState.from_initial_facts((
            StateFact("Pilat", "introduced", "scene", 0),
            StateFact("Pilat", "located_at", "palace", 0),
        ))

    def test_meaningful_valid_transition_beats_static_safe_continuation(self) -> None:
        static = event_hypothesis(name="static", subject="Pilat", action="", object_="")
        meaningful = event_hypothesis(name="meaningful", subject="Pilat", action="questions", object_="court")
        plan = beam_search(self.state, ((static, meaningful),), goal_terms=frozenset({"Pilat"}))

        self.assertEqual(plan.steps[0].name, "meaningful")
        self.assertEqual(plan.steps[0].status.value, "accepted")

    def test_invalid_entity_and_location_jumps_are_rejected(self) -> None:
        invalid_entity = event_hypothesis(name="behemoth", subject="Behemoth", action="acts", object_="laughs")
        invalid_location = event_hypothesis(name="jump", subject="Pilat", action="located_at", object_="Moscow")
        plan = beam_search(self.state, ((invalid_entity, invalid_location),))

        self.assertEqual(plan.steps, ())
        self.assertEqual(plan.audits, ("blocked_no_consistent_continuation_at_0",))

    def test_beam_and_greedy_are_deterministic_and_score_ablation_is_visible(self) -> None:
        first = event_hypothesis(name="first", subject="Pilat", action="questions", object_="court")
        second = event_hypothesis(name="second", subject="Pilat", action="answers", object_="court")
        greedy = beam_search(self.state, ((first,), (second,)), beam_width=1, disabled_scores=frozenset({"goal_relevance"}))
        beam = beam_search(self.state, ((first,), (second,)), beam_width=4, disabled_scores=frozenset({"goal_relevance"}))

        self.assertEqual(greedy.to_dict(), beam.to_dict())
        self.assertEqual(beam.steps[0].score.goal_relevance, 0.0)


if __name__ == "__main__":
    unittest.main()
