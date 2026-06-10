import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core import World, PredictionEngine, Prediction
from core.relations import Relation
from core.prediction import LIFECYCLE_RELS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _world_with(*observations) -> World:
    w = World()
    for text in observations:
        w.observe(text)
    return w


def _engine_from(*triples) -> PredictionEngine:
    rels = [Relation(source=s, relation_type=r, target=t) for s, r, t in triples]
    return PredictionEngine(rels)


def _lifecycle_world() -> World:
    """Two complete lifecycles + one partial."""
    return _world_with(
        "яблоня производит яблоко",
        "яблоко содержит семя",
        "семя вырастает в яблоню",
        "апельсин_дерево производит апельсин",
        "апельсин содержит семя",
        "семя вырастает в апельсин_дерево",
        "груша производит груша_плод",   # partial
    )


# ---------------------------------------------------------------------------
# find_chain_templates
# ---------------------------------------------------------------------------

class TestFindChainTemplates:
    def test_single_complete_lifecycle_creates_template(self):
        w = _world_with(
            "яблоня производит яблоко",
            "яблоко содержит семя",
            "семя вырастает в яблоню",
        )
        engine = PredictionEngine(w.get_relations())
        templates = engine.find_chain_templates()
        assert len(templates) == 1
        assert templates[0].relation_sequence == LIFECYCLE_RELS

    def test_two_complete_lifecycles_one_template_two_instances(self):
        w = _world_with(
            "яблоня производит яблоко",
            "яблоко содержит семя",
            "семя вырастает в яблоню",
            "апельсин_дерево производит апельсин",
            "апельсин содержит семя",
            "семя вырастает в апельсин_дерево",
        )
        engine = PredictionEngine(w.get_relations())
        templates = engine.find_chain_templates()
        assert len(templates) == 1
        assert len(templates[0].instances) == 2

    def test_no_template_when_cycle_incomplete(self):
        w = _world_with(
            "яблоня производит яблоко",
            "яблоко содержит семя",
            # missing: семя вырастает в яблоню
        )
        engine = PredictionEngine(w.get_relations())
        assert engine.find_chain_templates() == []

    def test_template_instance_slots_correct(self):
        w = _world_with(
            "яблоня производит яблоко",
            "яблоко содержит семя",
            "семя вырастает в яблоню",
        )
        engine = PredictionEngine(w.get_relations())
        inst = engine.find_chain_templates()[0].instances[0]
        assert inst[0] == "яблоня"
        assert inst[1] == "яблоко"
        assert inst[2] == "семя"

    def test_most_common_slot_value(self):
        w = _world_with(
            "яблоня производит яблоко",
            "яблоко содержит семя",
            "семя вырастает в яблоню",
            "апельсин_дерево производит апельсин",
            "апельсин содержит семя",
            "семя вырастает в апельсин_дерево",
        )
        engine = PredictionEngine(w.get_relations())
        template = engine.find_chain_templates()[0]
        assert template.most_common_slot_value(2) == "семя"


# ---------------------------------------------------------------------------
# predict_missing_links
# ---------------------------------------------------------------------------

class TestPredictMissingLinks:
    def test_predictions_generated_for_partial_match(self):
        w = _lifecycle_world()
        predictions = w.predict_missing_links()
        assert len(predictions) > 0

    def test_product_contains_connector_predicted(self):
        w = _lifecycle_world()
        preds = w.predict_missing_links()
        assert any(
            p.source == "груша_плод" and p.relation_type == "contains" and p.target == "семя"
            for p in preds
        )

    def test_connector_grows_into_entity_predicted(self):
        w = _lifecycle_world()
        preds = w.predict_missing_links()
        assert any(
            p.source == "семя" and p.relation_type == "grows_into" and p.target == "груша"
            for p in preds
        )

    def test_no_duplicate_predictions_for_existing_relations(self):
        w = _lifecycle_world()
        preds = w.predict_missing_links()
        existing = {(r.source, r.relation_type, r.target) for r in w.get_relations()}
        for p in preds:
            assert (p.source, p.relation_type, p.target) not in existing, (
                f"predicted an already-existing relation: {p}"
            )

    def test_complete_instance_not_re_predicted(self):
        w = _world_with(
            "яблоня производит яблоко",
            "яблоко содержит семя",
            "семя вырастает в яблоню",
            "груша производит груша_плод",   # partial
        )
        preds = w.predict_missing_links()
        # яблоня->яблоко is a complete instance — should not appear in predictions
        assert not any(p.source == "яблоко" and p.target == "семя" for p in preds)
        assert not any(p.source == "семя" and p.target == "яблоня" for p in preds)

    def test_multiple_partials_all_predicted(self):
        w = _world_with(
            "яблоня производит яблоко",
            "яблоко содержит семя",
            "семя вырастает в яблоню",
            "груша производит груша_плод",
            "вишня производит вишня_плод",
        )
        preds = w.predict_missing_links()
        targets = {p.target for p in preds}
        assert "груша" in targets
        assert "вишня" in targets

    def test_empty_world_no_predictions(self):
        w = World()
        assert w.predict_missing_links() == []

    def test_no_template_no_predictions(self):
        # Only produces_result relations, no complete cycle
        w = _world_with("яблоня производит яблоко", "груша производит груша_плод")
        assert w.predict_missing_links() == []


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_single_template_confidence_is_0_7(self):
        w = _world_with(
            "яблоня производит яблоко",
            "яблоко содержит семя",
            "семя вырастает в яблоню",
            "груша производит груша_плод",
        )
        preds = w.predict_missing_links()
        for p in preds:
            assert p.confidence == pytest.approx(0.7)

    def test_two_templates_confidence_is_0_8(self):
        w = _lifecycle_world()
        preds = w.predict_missing_links()
        for p in preds:
            assert p.confidence == pytest.approx(0.8)

    def test_confidence_capped_at_0_95(self):
        w = World()
        # Feed 10 complete lifecycles
        trees = ["дуб", "сосна", "ель", "береза", "клен", "ива", "тис", "кедр"]
        for tree in trees:
            product = f"{tree}_плод"
            w.observe(f"{tree} производит {product}")
            w.observe(f"{product} содержит семя")
            w.observe(f"семя вырастает в {tree}")
        w.observe("груша производит груша_плод")
        preds = w.predict_missing_links()
        for p in preds:
            assert p.confidence <= 0.95

    def test_confidence_increases_with_more_instances(self):
        def get_confidence(n_complete: int) -> float:
            w = World()
            trees = [f"дерево{i}" for i in range(n_complete)]
            for tree in trees:
                product = f"{tree}_плод"
                w.observe(f"{tree} производит {product}")
                w.observe(f"{product} содержит семя")
                w.observe(f"семя вырастает в {tree}")
            w.observe("груша производит груша_плод")
            preds = w.predict_missing_links()
            return preds[0].confidence if preds else 0.0

        assert get_confidence(1) < get_confidence(2) < get_confidence(3)


# ---------------------------------------------------------------------------
# explain_prediction
# ---------------------------------------------------------------------------

class TestExplainPrediction:
    def test_output_contains_relation(self):
        w = _lifecycle_world()
        preds = w.predict_missing_links()
        engine = PredictionEngine(w.get_relations())
        explanation = engine.explain_prediction(preds[0])
        assert preds[0].source in explanation
        assert preds[0].relation_type in explanation
        assert preds[0].target in explanation

    def test_output_contains_confidence(self):
        w = _lifecycle_world()
        preds = w.predict_missing_links()
        engine = PredictionEngine(w.get_relations())
        explanation = engine.explain_prediction(preds[0])
        assert "confidence" in explanation

    def test_output_contains_evidence(self):
        w = _lifecycle_world()
        preds = w.predict_missing_links()
        engine = PredictionEngine(w.get_relations())
        explanation = engine.explain_prediction(preds[0])
        assert "based on" in explanation or len(preds[0].evidence) == 0

    def test_prediction_has_reason(self):
        w = _lifecycle_world()
        preds = w.predict_missing_links()
        for p in preds:
            assert p.reason


# ---------------------------------------------------------------------------
# World integration
# ---------------------------------------------------------------------------

class TestWorldIntegration:
    def test_world_predict_missing_links_exists(self):
        w = World()
        assert hasattr(w, "predict_missing_links")
        assert callable(w.predict_missing_links)

    def test_prediction_dataclass_fields(self):
        w = _lifecycle_world()
        preds = w.predict_missing_links()
        assert preds
        p = preds[0]
        assert isinstance(p.source, str)
        assert isinstance(p.relation_type, str)
        assert isinstance(p.target, str)
        assert isinstance(p.confidence, float)
        assert isinstance(p.reason, str)
        assert isinstance(p.evidence, list)
