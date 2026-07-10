"""Focused checks for the explicit reasoning transfer boundary."""

from __future__ import annotations

from collections import Counter
import unittest

from poemcore.concept_graph import ConceptGraph
from poemcore.line_plan import build_line_intents, build_poem_intent
from poemcore.phrase_model import PhraseModel
from poemcore.planner import LinePlan, PoemPlan
from poemcore.planner import plan_poem
from poemcore.reasoning import reason_poem
from poemcore.reasoning import _choose_surface_roles, _is_action


class ReasoningPlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = ConceptGraph(weight={"осень": 2.0, "лист": 1.5, "движется": 1.0, "совет": 1.0})
        self.graph.add_edge("осень", "лист", 2.0)
        self.graph.add_edge("лист", "движется", 3.0)
        self.graph.add_edge("лист", "совет", 9.0)
        self.phrase = PhraseModel()
        self.phrase.learn_line("лист движется осень")
        self.phrase.unigram["совет"] = 99
        self.plan = PoemPlan(
            theme="autumn",
            style_author="",
            rhyme_scheme="AB",
            target_syllables=8,
            seed_concepts=["осень", "лист"],
            lines=[
                LinePlan(0, "establish", "осень", "A", 8, True),
                LinePlan(1, "develop", "лист", "B", 8, False),
                LinePlan(2, "turn", "лист", "A", 8, False),
                LinePlan(3, "closure", "осень", "B", 8, False),
            ],
        )

    def test_reasoning_is_complete_and_replayable(self) -> None:
        first = reason_poem(self.graph, self.phrase, self.plan, theme="autumn")
        second = reason_poem(self.graph, self.phrase, self.plan, theme="autumn")

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.goal.setting, "осень")
        self.assertEqual([s.purpose for s in first.stanzas], ["introduce", "turn_and_resolve"])
        self.assertEqual(len(first.lines), len(self.plan.lines))
        self.assertEqual(first.lines[1].subject, "лист")
        self.assertEqual(first.lines[1].action, "движется")
        self.assertEqual(first.lines[-1].subject, "я")

    def test_reasoning_decisions_become_line_intents(self) -> None:
        reasoning = reason_poem(self.graph, self.phrase, self.plan, theme="autumn")
        poem_intent = build_poem_intent(
            theme="autumn",
            seed_concepts=self.plan.seed_concepts,
            graph=self.graph,
            setting=reasoning.goal.setting,
            mood=reasoning.goal.mood,
        )
        intents = build_line_intents(poem_intent, self.plan.lines, reasoning.lines)

        self.assertEqual(
            [(intent.subject, intent.action, intent.object) for intent in intents],
            [(decision.subject, decision.action, decision.object) for decision in reasoning.lines],
        )
        self.assertEqual(intents[0].to_dict()["action"], reasoning.lines[0].action)

    def test_structural_plan_excludes_unconnected_high_frequency_nodes(self) -> None:
        graph = ConceptGraph(weight={"осень": 1.0, "лист": 1.0, "ветер": 1.0, "кто": 999.0})
        graph.add_edge("осень", "лист", 2.0)
        graph.add_edge("лист", "ветер", 2.0)

        plan = plan_poem(graph, ["осень"], theme="autumn", style_author="", stanzas=1)

        self.assertNotIn("кто", [line.focus for line in plan.lines])
        self.assertTrue(set(focus for focus, _score in plan.activated) <= {"осень", "лист", "ветер"})

    def test_structural_plan_develops_explicit_seed_commitments_first(self) -> None:
        plan = plan_poem(self.graph, ["осень", "лист"], theme="autumn", style_author="", stanzas=1)

        self.assertEqual(plan.lines[0].focus, "осень")
        self.assertEqual(plan.lines[1].focus, "лист")
        self.assertEqual(plan.lines[-1].focus, "осень")

    def test_reasoning_prefers_a_surface_realizable_clause(self) -> None:
        phrase = PhraseModel()
        phrase.learn_line("ветер тихо несет листья домой")
        graph = ConceptGraph(weight={"ветер": 2.0, "несет": 1.0, "листья": 2.0})
        graph.add_edge("ветер", "несет")
        graph.add_edge("несет", "листья")
        plan = PoemPlan(
            theme="осень", style_author="", rhyme_scheme="A", target_syllables=8,
            seed_concepts=["ветер", "листья"],
            lines=[LinePlan(0, "establish", "ветер", "A", 8, True)],
        )

        decision = reason_poem(graph, phrase, plan, theme="осень").lines[0]

        self.assertEqual((decision.subject, decision.action, decision.object), ("ветер", "несет", "листья"))

    def test_reasoning_rejects_false_actions_from_permissive_endings(self) -> None:
        phrase = PhraseModel(unigram=Counter({
            "отдала": 1, "узнают": 1, "если": 1, "золотистую": 1, "сердцем": 1, "честь": 1,
        }))

        self.assertTrue(_is_action("отдала", phrase))
        self.assertTrue(_is_action("узнают", phrase))
        for false_action in ("если", "золотистую", "сердцем", "честь"):
            self.assertFalse(_is_action(false_action, phrase), false_action)

    def test_surface_roles_can_anchor_on_the_planned_object(self) -> None:
        phrase = PhraseModel()
        phrase.learn_line("ветер тихо несет листья домой")
        graph = ConceptGraph(weight={"листья": 2.0})

        roles = _choose_surface_roles(graph, phrase, ("листья",), "fixed")

        self.assertEqual(roles, ("ветер", "несет", "листья"))


if __name__ == "__main__":
    unittest.main()
