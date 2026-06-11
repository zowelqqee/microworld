import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.trust_learning import (
    TrustProfile,
    label_score,
    learn_trust_from_audits,
    parse_drift_type,
)


def _write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def test_label_scores():
    assert label_score("correct") == pytest.approx(1.0)
    assert label_score("plausible") == pytest.approx(0.7)
    assert label_score("wrong") == pytest.approx(0.0)
    assert label_score("unclear") == pytest.approx(0.4)
    assert label_score("plusable") == pytest.approx(0.7)
    assert label_score("") is None


def test_relation_trust_learned(tmp_path):
    path = _write_csv(tmp_path / "audit.csv", [
        {"relation_type": "made_of", "manual_label": "correct"},
        {"relation_type": "made_of", "manual_label": "plausible"},
        {"relation_type": "part_of", "manual_label": "wrong"},
    ])

    profile = learn_trust_from_audits([path])

    assert profile.relation_trust["made_of"] == pytest.approx(0.85)
    assert profile.relation_trust["part_of"] == pytest.approx(0.0)
    assert profile.counts["relation_trust"]["made_of"] == 2


def test_rule_trust_learned_when_rule_column_exists(tmp_path):
    path = _write_csv(tmp_path / "audit.csv", [
        {"relation_type": "made_of", "rule": "made_of->made_of", "manual_label": "correct"},
        {"relation_type": "made_of", "rule": "made_of->made_of", "manual_label": "wrong"},
        {"relation_type": "made_of", "rule": "part_of->made_of=>made_of", "manual_label": "plausible"},
    ])

    profile = learn_trust_from_audits([path])

    assert profile.rule_trust["made_of->made_of"] == pytest.approx(0.5)
    assert profile.rule_trust["part_of->made_of=>made_of"] == pytest.approx(0.7)


def test_drift_trust_parsed_from_reason(tmp_path):
    path = _write_csv(tmp_path / "audit.csv", [
        {
            "relation_type": "made_of",
            "reason": "transitive pattern (drift=atomic_component, drift_penalty=0.65)",
            "manual_label": "wrong",
        },
        {
            "relation_type": "made_of",
            "reason": "transitive pattern (drift=atomic_component, drift_penalty=0.65)",
            "manual_label": "plausible",
        },
        {
            "relation_type": "made_of",
            "reason": "transitive pattern (drift=raw_material, drift_penalty=0.85)",
            "manual_label": "correct",
        },
    ])

    profile = learn_trust_from_audits([path])

    assert parse_drift_type("x drift=atomic_component, y") == "atomic_component"
    assert profile.drift_trust["atomic_component"] == pytest.approx(0.35)
    assert profile.drift_trust["raw_material"] == pytest.approx(1.0)


def test_evidence_trust_learned(tmp_path):
    path = _write_csv(tmp_path / "audit.csv", [
        {"relation_type": "made_of", "evidence": "paper|wood", "manual_label": "correct"},
        {"relation_type": "made_of", "evidence": "paper", "manual_label": "wrong"},
    ])

    profile = learn_trust_from_audits([path])

    assert profile.evidence_trust["paper"] == pytest.approx(0.5)
    assert profile.evidence_trust["wood"] == pytest.approx(1.0)


def test_json_roundtrip(tmp_path):
    profile = TrustProfile(
        relation_trust={"made_of": 0.85},
        rule_trust={"r": 0.5},
        drift_trust={"atomic_component": 0.35},
        evidence_trust={"paper": 0.5},
        counts={"rows": 3},
    )
    path = tmp_path / "trust.json"

    profile.to_json(str(path))
    loaded = TrustProfile.from_json(str(path))

    assert loaded == profile


def test_unknown_labels_handled_safely(tmp_path):
    path = _write_csv(tmp_path / "audit.csv", [
        {"relation_type": "made_of", "manual_label": "mystery"},
        {"relation_type": "made_of", "manual_label": ""},
    ])

    profile = learn_trust_from_audits([path])

    assert profile.relation_trust["made_of"] == pytest.approx(0.4)
    assert profile.counts["used_rows"] == 1
    assert profile.counts["skipped_empty_label"] == 1
