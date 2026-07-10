"""Focused checks for deterministic realization of explicit line decisions."""

from __future__ import annotations

import unittest

from poemcore.concept_graph import ConceptGraph
from poemcore.generator import Generator
from poemcore.line_plan import LineIntent, build_poem_intent, decision_role_hits
from poemcore.phrase_model import PhraseModel
from poemcore.planner import LinePlan, PoemPlan


class SurfaceRealizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.phrase = PhraseModel()
        self.phrase.learn_line("ветер тихо несет листья домой")
        # The novelty gate is unrelated to the realization test. Clear it so
        # this miniature corpus can serve as a valid transition fixture.
        self.phrase.seen_4grams.clear()
        self.graph = ConceptGraph(weight={"ветер": 2.0, "несет": 1.0, "листья": 2.0})
        self.graph.add_edge("ветер", "несет")
        self.graph.add_edge("несет", "листья")

    def test_slot_walk_realizes_ordered_roles_through_observed_bridge(self) -> None:
        forward = self.phrase.grow_forward_slots(
            "ветер", 10, "fixed", slots=("несет", "листья")
        )
        backward = self.phrase.grow_backward_slots(
            "домой", 10, "fixed", slots=("ветер", "несет", "листья")
        )

        self.assertEqual(forward, ["ветер", "тихо", "несет", "листья", "домой"])
        self.assertEqual(backward, ["ветер", "тихо", "несет", "листья", "домой"])

    def test_generator_records_exact_role_coverage(self) -> None:
        plan = PoemPlan(
            theme="осень", style_author="", rhyme_scheme="A", target_syllables=10,
            seed_concepts=["ветер", "листья"],
            lines=[LinePlan(0, "establish", "ветер", "A", 10, True)],
        )
        poem_intent = build_poem_intent(
            theme="осень", seed_concepts=["ветер", "листья"], graph=self.graph
        )
        intent = LineIntent(
            index=0, subject="ветер", action_wanted=True, action="несет", object="листья",
            modifier="осень", mood="sadness", relation_to_previous="introduce",
        )
        poem = Generator(self.phrase, self.graph, {}).render(
            plan, seed="fixed", poem_intent=poem_intent, line_intents=[intent]
        )

        self.assertEqual(poem.lines, ["Ветер тихо несет листья домой"])
        self.assertEqual(poem.realization[0].strategy, "role_anchored_forward")
        self.assertEqual(poem.realization[0].realized, ("subject", "action", "object"))
        self.assertEqual(decision_role_hits(poem.lines[0], intent), {
            "subject": True, "action": True, "object": True,
        })


if __name__ == "__main__":
    unittest.main()
