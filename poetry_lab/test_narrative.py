"""Focused checks for the prose planning and realization transfer."""

from __future__ import annotations

import unittest

from poemcore.concept_graph import ConceptGraph
from poemcore.narrative import (
    NarrativeGenerator, SceneGoal, ScenePlan, SentencePlan, _complete_tokens,
    _description_fact_subject, _detail_subject, _scene_state, _transition_word, plan_scene,
)
from poemcore.novelty import check_line
from poemcore.phrase_model import PhraseModel
from poemcore.text import sentences


class NarrativeTransferTest(unittest.TestCase):
    def setUp(self) -> None:
        self.phrase = PhraseModel()
        self.phrase.learn_line("иван тихо сказал берлиозу правду")
        self.phrase.learn_line("иван спокойно сказал берлиозу правду")
        self.phrase.learn_line("он тихо сказал берлиозу правду")
        self.phrase.unigram["затем"] = 1
        self.graph = ConceptGraph(weight={"иван": 3.0, "берлиозу": 2.0, "сказал": 2.0})
        self.graph.add_edge("иван", "сказал")
        self.graph.add_edge("сказал", "берлиозу")

    def test_prose_segmenter_keeps_dialogue_as_sentences(self) -> None:
        self.assertEqual(sentences("– Да. Он ушел.\nИван ждал."), ["– Да.", "Он ушел.", "Иван ждал."])

    def test_scene_plan_is_replayable_and_keeps_dialogue_speaker(self) -> None:
        kwargs = dict(
            prompt="Continue this dialogue", context="Иван посмотрел на Берлиоза.",
            meta={"recurring_characters": ["иван", "берлиоз"], "recurring_locations": [], "recurring_objects": []},
            proper_names=frozenset({"иван", "берлиоз"}), sentence_count=3,
        )
        first = plan_scene(self.graph, self.phrase, **kwargs)
        second = plan_scene(self.graph, self.phrase, **kwargs)

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.goal.mode, "dialogue")
        self.assertTrue(all(item.speaker == first.goal.speaker for item in first.sentences))

    def test_description_has_no_default_character_actor(self) -> None:
        plan = plan_scene(
            self.graph, self.phrase, prompt="Describe a room.", context="",
            meta={"recurring_characters": ["иван"], "recurring_locations": [], "recurring_objects": []},
            proper_names=frozenset({"иван"}), sentence_count=2,
        )

        self.assertEqual(plan.goal.mode, "description")
        self.assertEqual(plan.goal.speaker, "")
        self.assertEqual(plan.goal.characters, ())
        self.assertIn(plan.goal.topic, plan.goal.active_field)

    def test_sentence_realizer_prevents_corpus_fourgram_echo(self) -> None:
        goal = SceneGoal("иван", "scene", "иван", "иван", ("иван",), ("иван", "берлиозу"))
        plan = ScenePlan(goal, ("establish_scene",), (
            SentencePlan(0, "establish_scene", "иван", "иван", "сказал", "берлиозу", "иван", False, "иван", "", 8),
        ))
        paragraph = NarrativeGenerator(self.phrase, self.graph, frozenset({"иван", "берлиозу"})).render(
            plan, seed="fixed"
        )

        self.assertTrue(check_line(self.phrase, paragraph.sentences[0]))
        self.assertIn("иван", paragraph.sentences[0].lower())
        self.assertIn("сказал", paragraph.sentences[0].lower())

    def test_completion_trims_dangling_surface_tail(self) -> None:
        self.assertEqual(_complete_tokens(["иван", "не", "ответил", "на"]), ["иван", "не", "ответил"])
        self.assertEqual(_complete_tokens(["иван", "ухватился", "за", "притолоку", "и", "долго"]),
                         ["иван", "ухватился", "за", "притолоку"])

    def test_scene_plan_expands_to_three_reasoned_beats(self) -> None:
        plan = plan_scene(
            self.graph, self.phrase, prompt="Describe a room.", context="",
            meta={"recurring_characters": [], "recurring_locations": [], "recurring_objects": []},
            proper_names=frozenset(), sentence_count=1,
        )

        self.assertEqual(len(plan.sentences), 3)
        self.assertEqual(plan.beats[1:], ("develop_field", "close_field"))

    def test_scene_plan_threads_a_continuity_anchor(self) -> None:
        plan = plan_scene(
            self.graph, self.phrase, prompt="Continue this dialogue", context="Иван посмотрел на Берлиоза.",
            meta={"recurring_characters": ["иван", "берлиоз"], "recurring_locations": [], "recurring_objects": []},
            proper_names=frozenset({"иван", "берлиоз"}), sentence_count=3,
        )

        self.assertEqual(plan.sentences[0].continuity_anchor, plan.goal.topic)
        self.assertTrue(plan.sentences[1].continuity_anchor)
        self.assertNotEqual(plan.sentences[1].transition, "")

    def test_description_activates_the_topic_field_not_a_global_character(self) -> None:
        graph = ConceptGraph(weight={"москва": 1.0, "вечер": 1.0, "иван": 100.0})
        graph.add_edge("москва", "вечер")
        plan = plan_scene(
            graph, self.phrase, prompt="Опиши Москву", context="",
            meta={"recurring_characters": ["иван"], "recurring_locations": [], "recurring_objects": []},
            proper_names=frozenset({"иван"}), sentence_count=3,
        )

        self.assertEqual(plan.goal.mode, "description")
        self.assertIn("москва", plan.goal.active_field)
        self.assertIn("вечер", plan.goal.active_field)
        self.assertNotIn("иван", plan.goal.active_field)
        self.assertEqual(plan.goal.speaker, "")

    def test_relevant_activation_does_not_start_from_global_frequency(self) -> None:
        graph = ConceptGraph(weight={"москва": 1.0, "вечер": 1.0, "иван": 1000.0})
        graph.add_edge("москва", "вечер")

        field = graph.activate_relevant(["москва"], rounds=2)

        self.assertIn("москва", field)
        self.assertIn("вечер", field)
        self.assertNotIn("иван", field)

    def test_description_renderer_realizes_a_place_relation_without_an_actor(self) -> None:
        phrase = PhraseModel()
        phrase.learn_line("наступил вечер")
        phrase.learn_line("улица тянулась далеко")
        graph = ConceptGraph(weight={"москве": 1.0, "вечер": 1.0, "улица": 1.0})
        goal = SceneGoal(
            "вечер", "description", "", "москве", (), ("вечер", "москве"),
            active_field=("вечер", "москве", "улица"), scene_objective="describe_grounded_field",
        )
        plan = ScenePlan(goal, ("establish_field", "develop_field"), (
            SentencePlan(0, "establish_field", "вечер", "вечер", "наступил", "москве", "", False, relation="located_at"),
            SentencePlan(1, "develop_field", "улица", "улица", "тянулась", "далеко", "", False, relation="property"),
        ))

        paragraph = NarrativeGenerator(phrase, graph, frozenset({"москве"})).render(plan, seed="fixed")

        self.assertEqual(paragraph.sentences, ["В Москве наступил вечер.", "Улица тянулась далеко."])
        self.assertEqual(paragraph.realization[0].realized_facts, ())

    def test_description_renderer_drops_an_ungrounded_field_term(self) -> None:
        goal = SceneGoal("москва", "description", "", "москве", (), ("москва", "москве"))
        plan = ScenePlan(goal, ("establish_field", "close_field"), (
            SentencePlan(0, "establish_field", "вечер", "вечер", "наступил", "москве", "", False, relation="located_at"),
            SentencePlan(1, "close_field", "город", "город", "", "", "", False, relation="observation"),
        ))
        paragraph = NarrativeGenerator(PhraseModel(), ConceptGraph(), frozenset({"москве"})).render(plan, seed="fixed")

        self.assertEqual(paragraph.sentences, ["В Москве наступил вечер."])
        self.assertEqual(len(paragraph.realization), 1)

    def test_description_renderer_keeps_location_and_property_in_one_sentence(self) -> None:
        goal = SceneGoal("вечер", "description", "", "москве", (), ("вечер", "москве"))
        plan = ScenePlan(goal, ("establish_field",), (
            SentencePlan(
                0, "establish_field", "вечер", "вечер", "был", "москве", "", False,
                relation="located_property", detail_fragment=("прохладный",),
            ),
        ))

        paragraph = NarrativeGenerator(PhraseModel(), ConceptGraph(), frozenset({"москве"})).render(plan, seed="fixed")

        self.assertEqual(paragraph.sentences, ["В Москве был прохладный вечер."])

    def test_description_renderer_realizes_a_fact_bundle_in_one_sentence(self) -> None:
        goal = SceneGoal("комната", "description", "", "", (), ("комната",))
        plan = ScenePlan(goal, ("establish_field",), (
            SentencePlan(
                0, "establish_field", "комната", "комната", "казалась", "мрачной", "", False,
                relation="property", epithet="особая", link=("в", "трактире"),
            ),
        ))

        paragraph = NarrativeGenerator(PhraseModel(), ConceptGraph(), frozenset()).render(plan, seed="fixed")

        self.assertEqual(paragraph.sentences, ["Особая комната в трактире казалась мрачной."])

    def test_description_renderer_places_an_inverted_event_before_the_subject(self) -> None:
        goal = SceneGoal("город", "description", "", "", (), ("город",))
        plan = ScenePlan(goal, ("establish_field",), (
            SentencePlan(
                0, "establish_field", "город", "город", "кончался", "за заводами", "", False,
                relation="event_place_inverted", epithet="изумительный",
            ),
        ))

        paragraph = NarrativeGenerator(PhraseModel(), ConceptGraph(), frozenset()).render(plan, seed="fixed")

        self.assertEqual(paragraph.sentences, ["За заводами кончался изумительный город."])

    def test_description_reasoning_bundles_compatible_facts_for_one_subject(self) -> None:
        graph = ConceptGraph(weight={"комната": 2.0})
        relations = {"комната": [
            {"predicate": "казалась", "detail": "мрачной", "kind": "property", "weight": 2},
            {"predicate": "", "detail": "особая", "kind": "epithet", "weight": 1},
            {"predicate": "в", "detail": "трактире", "kind": "object_link", "weight": 1},
        ]}

        plan = plan_scene(
            graph, self.phrase, prompt="Опиши комнату", context="",
            meta={"recurring_characters": [], "recurring_locations": [], "recurring_objects": []},
            proper_names=frozenset(), sentence_count=1,
            description_relations=relations,
        )
        first = plan.sentences[0]

        self.assertEqual(first.relation, "property")
        self.assertEqual((first.action, first.object), ("казалась", "мрачной"))
        self.assertEqual(first.epithet, "особая")
        self.assertEqual(first.link, ("в", "трактире"))

    def test_description_reasoning_never_promotes_a_link_to_the_primary_fact(self) -> None:
        graph = ConceptGraph(weight={"дверь": 2.0})
        relations = {"дверь": [
            {"predicate": "в", "detail": "коридор", "kind": "object_link", "weight": 9},
            {"predicate": "", "detail": "резная", "kind": "epithet", "weight": 1},
        ]}

        plan = plan_scene(
            graph, self.phrase, prompt="Опиши дверь", context="",
            meta={"recurring_characters": [], "recurring_locations": [], "recurring_objects": []},
            proper_names=frozenset(), sentence_count=1,
            description_relations=relations,
        )
        first = plan.sentences[0]

        self.assertEqual(first.relation, "epithet")
        self.assertEqual(first.object, "резная")
        self.assertEqual(first.link, ("в", "коридор"))

    def test_description_scene_state_exposes_field_and_objective(self) -> None:
        goal = SceneGoal(
            "москва", "description", "", "москве", (), ("москва", "москве"),
            active_field=("москва", "вечер", "улица"), scene_objective="describe_grounded_field",
        )
        state = _scene_state(ScenePlan(goal, (), ()))

        self.assertEqual(state.narrative_focus, "москва")
        self.assertEqual(state.scene_objective, "describe_grounded_field")
        self.assertEqual(state.active_field, ("москва", "вечер", "улица"))

    def test_description_case_resolution_does_not_merge_unrelated_words_by_prefix(self) -> None:
        relations = {"сад": [{"kind": "epithet"}], "садик": [{"kind": "epithet"}]}

        self.assertEqual(_description_fact_subject("саду", relations), "сад")
        self.assertEqual(_description_fact_subject("сад", {"садик": [{"kind": "epithet"}]}), "")

    def test_reasoning_selects_a_learned_event_fragment(self) -> None:
        plan = plan_scene(
            self.graph, self.phrase, prompt="Continue this dialogue", context="Иван посмотрел на Берлиоза.",
            meta={"recurring_characters": ["иван", "берлиоз"], "recurring_locations": [], "recurring_objects": []},
            proper_names=frozenset({"иван", "берлиоз"}), sentence_count=3,
        )

        fragments = [sentence.fragment for sentence in plan.sentences if sentence.fragment]
        self.assertTrue(fragments)
        for fragment in fragments:
            self.assertEqual(len(fragment), 3)
            self.assertIn(fragment[2], self.phrase.forward2[fragment[:2]])
            self.assertEqual(fragment[0], plan.goal.speaker)
        self.assertTrue(any(sentence.detail_fragment for sentence in plan.sentences))

    def test_two_word_clause_is_kept_as_a_complete_fallback(self) -> None:
        self.assertEqual(_complete_tokens(["иван", "сказал"]), ["иван", "сказал"])

    def test_detail_clause_uses_supported_pronoun_for_masculine_name(self) -> None:
        self.assertEqual(_detail_subject("иван", self.phrase), "он")
        self.assertEqual(_detail_subject("анна", self.phrase), "анна")

    def test_transition_seed_opens_multiple_learned_routes(self) -> None:
        self.phrase.unigram.update({"потом": 4, "тогда": 4, "позже": 4})
        values = {_transition_word(self.phrase, 2, seed) for seed in ("a", "b", "c", "d", "e")}
        self.assertGreater(len(values), 1)


if __name__ == "__main__":
    unittest.main()
