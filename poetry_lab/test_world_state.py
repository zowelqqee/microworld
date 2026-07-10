"""Focused Phase-1 tests for temporal narrative state."""

from __future__ import annotations

import unittest

from poemcore.world_state import StateDelta, StateFact, WorldState


def fact(subject: str, predicate: str, object_: str) -> StateFact:
    return StateFact(subject, predicate, object_, t=0)


class WorldStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.initial = WorldState.from_initial_facts((
            fact("Pilat", "introduced", "scene"),
            fact("Pilat", "located_at", "palace"),
            fact("Yeshua", "introduced", "scene"),
            fact("Yeshua", "located_at", "palace"),
        ))

    def test_frame_persistence_and_temporal_indexing(self) -> None:
        result = self.initial.apply(StateDelta(assertions=(fact("Pilat", "speaking_to", "Yeshua"),)))

        self.assertTrue(result.accepted)
        self.assertEqual(result.state.t, 1)
        self.assertIn(("Pilat", "located_at", "palace"), {item.triple for item in result.state.facts})
        self.assertTrue(all(item.t == 1 for item in result.state.facts))
        self.assertEqual(result.state.entities["pilat"].last_action_at, 1)

    def test_move_retracts_previous_location_and_records_proof(self) -> None:
        result = self.initial.apply(StateDelta(assertions=(fact("Pilat", "moves_to", "garden"),)))

        self.assertTrue(result.accepted)
        self.assertEqual(result.state.location_of("Pilat"), "garden")
        self.assertIn(("Pilat", "located_at", "palace"), {item.triple for item in result.retracted})
        self.assertIn("moves_to_implies_location", {item.rule for item in result.proof_steps})

    def test_entity_introduction_is_required_before_action(self) -> None:
        result = self.initial.apply(StateDelta(assertions=(fact("Behemoth", "acts", "laughs"),)))

        self.assertFalse(result.accepted)
        self.assertEqual(result.state, self.initial)
        self.assertEqual(result.violations[0].code, "unintroduced_entity")

    def test_bilocation_is_rejected(self) -> None:
        result = self.initial.apply(StateDelta(assertions=(
            fact("Pilat", "moves_to", "garden"),
            fact("Pilat", "moves_to", "ponds"),
        )))

        self.assertFalse(result.accepted)
        self.assertEqual({item.code for item in result.violations}, {"bilocation"})

    def test_direct_location_jump_is_rejected(self) -> None:
        result = self.initial.apply(StateDelta(assertions=(fact("Yeshua", "located_at", "Moscow"),)))

        self.assertFalse(result.accepted)
        self.assertEqual(result.violations[0].code, "location_change_without_move")

    def test_speaking_derives_colocation(self) -> None:
        result = self.initial.apply(StateDelta(assertions=(fact("Pilat", "speaking_to", "Yeshua"),)))

        self.assertIn(("Pilat", "co_located", "Yeshua"), {item.triple for item in result.inferred})
        self.assertEqual(result.proof_steps[-1].rule, "speaking_to_implies_co_located")

    def test_replay_is_deterministic(self) -> None:
        first = self.initial.apply(StateDelta(assertions=(fact("Pilat", "speaking_to", "Yeshua"),))).state
        second = first.apply(StateDelta(assertions=(fact("Pilat", "questions", "Yeshua"),))).state

        replayed = WorldState.replay(second.transitions, initial=self.initial)
        self.assertEqual(second.to_dict(), replayed.to_dict())

    def test_candidate_evaluation_does_not_mutate_parent(self) -> None:
        before = self.initial.to_dict()
        result = self.initial.apply(StateDelta(assertions=(fact("Pilat", "moves_to", "garden"),)))

        self.assertTrue(result.accepted)
        self.assertEqual(self.initial.to_dict(), before)
        self.assertEqual(self.initial.location_of("Pilat"), "palace")


if __name__ == "__main__":
    unittest.main()
