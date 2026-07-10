from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from poemcore.ingest import build_narrative_artifacts
from poemcore.narrative import (
    SceneGoal, ScenePlan, SentencePlan, _compress_scene_plan, _event_fallback,
    _reasoned_sentence, _sentence_candidates, _trusted_svo,
)


class NarrativeReasoningIntegrationTest(unittest.TestCase):
    def test_world_facts_require_a_finite_verb(self) -> None:
        fragmentary = SentencePlan(0, "establish_scene", "пилат", "пилат", "пилат", "", "пилат", False, fragment=("пилат", "усмехнувшись", "но"))
        semantic = SentencePlan(1, "develop_action", "пилат", "пилат", "пилат", "", "пилат", False, fragment=("пилат", "велел", "постель"))

        self.assertIsNone(_reasoned_sentence(fragmentary))
        converted = _reasoned_sentence(semantic)
        self.assertIsNotNone(converted)
        self.assertEqual((converted.action, converted.object), ("велел", "постель"))

    def test_fragmentary_sentence_keeps_a_non_committing_surface_fallback(self) -> None:
        sentence = SentencePlan(0, "establish_scene", "пилат", "пилат", "пилат", "", "пилат", False, fragment=("пилат", "усмехнувшись", "но"))

        candidates = _sentence_candidates(sentence)
        self.assertEqual(candidates[0].required_preconditions, ("stateless_fallback",))

    def test_reasoning_candidates_include_an_active_subject_variant(self) -> None:
        sentence = SentencePlan(1, "develop_action", "маргарита", "берлиоз", "ответил", "ему", "воланд", False)

        candidates = _sentence_candidates(sentence, continuity_subject="воланд")
        self.assertTrue(any(item.involved_entities[:1] == ("воланд",) for item in candidates))

    def test_svo_gate_rejects_first_person_and_verb_as_object(self) -> None:
        self.assertFalse(_trusted_svo(("воланд", "имел", "я"), "воланд"))
        self.assertFalse(_trusted_svo(("маргарита", "вздохнула", "стала"), "маргарита"))
        self.assertTrue(_trusted_svo(("пилат", "задал", "вопрос"), "пилат"))
        self.assertFalse(_trusted_svo(("воланд", "пропали", "гости"), "воланд"))
        self.assertFalse(_trusted_svo(("пилат", "продолжал", "вообразите"), "пилат"))
        self.assertFalse(_trusted_svo(("воланд", "прошептал", "буфетчик"), "воланд", verb_lexicon={"dative_actions": ["прошептал"]}))
        self.assertFalse(_trusted_svo(("пилат", "полет", "невидима"), "пилат"))

    def test_event_fallback_is_a_plain_asserted_clause(self) -> None:
        plan = SentencePlan(0, "continue_action", "пилат", "пилат", "задал", "вопрос", "пилат", False)
        self.assertEqual(_event_fallback(plan), ["пилат", "задал", "вопрос"])

    def test_untrusted_action_is_not_forwarded_to_reasoning_surface(self) -> None:
        sentence = SentencePlan(0, "continue_action", "пилат", "пилат", "полет", "невидима", "пилат", False)

        candidates = _sentence_candidates(sentence, noun_like=frozenset({"полет"}))
        self.assertEqual(candidates[0].payload.action, "")

    def test_directory_ingest_learns_a_universal_speech_frame(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "one.txt").write_text("Иван протянул руку.", encoding="utf-8")
            (root / "two.txt").write_text("Петр протянул руку.", encoding="utf-8")
            artifact = build_narrative_artifacts(root)

        self.assertEqual(artifact["meta"]["sources"], ["one.txt", "two.txt"])
        self.assertIn(
            {"action": "протянул", "object": "руку", "weight": 2, "subject_count": 2},
            artifact["universal_speech_frames"],
        )

    def test_ingest_keeps_description_facts_separate_from_loose_graph_edges(self) -> None:
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "prose.txt"
            source.write_text(
                "Улица тянулась далеко. Улица тянулась далеко. Доктор вошел в комнату.",
                encoding="utf-8",
            )
            artifact = build_narrative_artifacts(source)

        facts = artifact["description_relations"]
        self.assertIn(
            {"predicate": "тянулась", "detail": "далеко", "kind": "property", "weight": 2},
            facts["улица"],
        )
        self.assertNotIn("доктор", facts)

    def test_ingest_drops_an_oblique_adjective_from_an_attribute_fact(self) -> None:
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "prose.txt"
            source.write_text(
                "В дверь постучали. В дверь постучали. Лакированная дверь открылась. Приоткрытую дверь закрыли.",
                encoding="utf-8",
            )
            artifact = build_narrative_artifacts(source)

        rows = artifact["description_relations"]["дверь"]
        self.assertIn({"predicate": "", "detail": "лакированная", "kind": "epithet", "weight": 1}, rows)
        self.assertFalse(any(row["detail"] == "приоткрытую" for row in rows))

    def test_ingest_rejects_a_reflexive_event_with_a_comparative_tail(self) -> None:
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "prose.txt"
            source.write_text("Вечер оделся потеплее. Вечер был прохладный.", encoding="utf-8")
            artifact = build_narrative_artifacts(source)

        rows = artifact["description_relations"]["вечер"]
        self.assertIn({"predicate": "был", "detail": "прохладный", "kind": "property", "weight": 1}, rows)
        self.assertFalse(any(row["predicate"] == "оделся" for row in rows))

    def test_reasoning_compresses_adjacent_events_into_clauses(self) -> None:
        goal = SceneGoal("иван", "scene", "иван", "", ("иван",), ("иван",))
        plans = (
            SentencePlan(0, "establish_scene", "иван", "иван", "", "", "иван", False),
            SentencePlan(1, "continue_action", "иван", "иван", "поднял", "руку", "иван", False, fragment=("иван", "поднял", "руку")),
            SentencePlan(2, "continue_action", "иван", "иван", "сделал", "попытку", "иван", False, fragment=("иван", "сделал", "попытку")),
            SentencePlan(3, "close_scene", "иван", "иван", "склонил", "голову", "иван", False, fragment=("иван", "склонил", "голову")),
        )
        compressed = _compress_scene_plan(ScenePlan(goal, tuple(item.purpose for item in plans), plans), noun_like=frozenset())

        self.assertEqual(len(compressed.sentences), 3)
        self.assertEqual(
            compressed.sentences[2].clause_fragments,
            (("иван", "сделал", "попытку"), ("иван", "склонил", "голову")),
        )


if __name__ == "__main__":
    unittest.main()
