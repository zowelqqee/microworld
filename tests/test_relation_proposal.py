import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.relations import Relation
from core.relation_proposal import RelationProposal, RelationProposalEngine
from examples.relation_proposal_audit_export import (
    COLUMNS,
    build_audit_rows,
    write_audit_csv,
)


def _rels(*triples) -> list[Relation]:
    return [Relation(s, r, t) for s, r, t in triples]


def _write_csv(tmp_path, triples) -> str:
    path = tmp_path / "relations.csv"
    lines = ["source,relation_type,target"]
    lines.extend(f"{s},{r},{t}" for s, r, t in triples)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _proposal_map(proposals):
    return {
        (p.source, p.proposed_relation, p.target): p
        for p in proposals
    }


class TestRelationRuleDiscovery:
    def test_learns_transitive_part_of_rule(self):
        rules = RelationProposalEngine(_rels(
            ("wheel", "part_of", "car"),
            ("car", "part_of", "garage"),
            ("wheel", "part_of", "garage"),
            ("engine", "part_of", "car"),
            ("engine", "part_of", "garage"),
            ("door", "part_of", "car"),
            ("door", "part_of", "garage"),
        )).discover_relation_rules(min_count=3, min_rule_total=1)

        assert ("part_of", "part_of") in rules
        assert ("part_of", 3, 3, pytest.approx(4 / 8)) in rules[("part_of", "part_of")]

    def test_learns_mixed_part_of_made_of_rule(self):
        rules = RelationProposalEngine(_rels(
            ("wheel", "part_of", "car"),
            ("car", "made_of", "metal"),
            ("wheel", "made_of", "metal"),
            ("door", "part_of", "car"),
            ("door", "made_of", "metal"),
            ("hood", "part_of", "car"),
            ("hood", "made_of", "metal"),
        )).discover_relation_rules(min_count=3, min_rule_total=1)

        assert ("part_of", "made_of") in rules
        assert ("made_of", 3, 3, pytest.approx(4 / 8)) in rules[("part_of", "made_of")]

    def test_supports_multiple_candidate_output_relations_for_same_chain_pattern(self):
        rules = RelationProposalEngine(_rels(
            ("wheel", "part_of", "car"),
            ("car", "made_of", "metal"),
            ("wheel", "made_of", "metal"),
            ("door", "part_of", "car"),
            ("door", "made_of", "metal"),
            ("country", "part_of", "world"),
            ("world", "made_of", "continents"),
            ("country", "contains", "continents"),
        )).discover_relation_rules(min_count=1, min_rule_total=1)

        candidates = {
            rel: (count, conf)
            for rel, count, _total, conf in rules[("part_of", "made_of")]
        }
        assert candidates["made_of"] == pytest.approx((2, 3 / 8))
        assert candidates["contains"] == pytest.approx((1, 2 / 8))

    def test_confidence_is_count_over_total_direct_edges_for_pattern(self):
        rules = RelationProposalEngine(_rels(
            ("a1", "part_of", "b"),
            ("b", "made_of", "c"),
            ("a1", "made_of", "c"),
            ("a2", "part_of", "b"),
            ("a2", "made_of", "c"),
            ("a3", "part_of", "b"),
            ("a3", "contains", "c"),
        )).discover_relation_rules(
            min_count=1,
            min_rule_total=1,
            rule_alpha=0.0,
            rule_beta=0.0,
        )

        made_of = next(
            item for item in rules[("part_of", "made_of")]
            if item[0] == "made_of"
        )
        assert made_of[1] == 2
        assert made_of[2] == 3
        assert made_of[3] == pytest.approx(2 / 3)

    def test_raw_seven_of_seven_no_longer_confidence_one_under_smoothing(self):
        triples = []
        for i in range(7):
            triples.extend([
                (f"a{i}", "at_location", f"b{i}"),
                (f"b{i}", "at_location", f"c{i}"),
                (f"a{i}", "at_location", f"c{i}"),
            ])

        rules = RelationProposalEngine(_rels(*triples)).discover_relation_rules(
            min_count=1,
            min_rule_total=1,
            include_disabled_relations=True,
        )

        rule = rules[("at_location", "at_location")][0]
        assert rule[1] == 7
        assert rule[2] == 7
        assert rule[3] == pytest.approx(8 / 12)

    def test_rules_below_min_rule_total_are_skipped(self):
        triples = []
        for i in range(7):
            triples.extend([
                (f"a{i}", "at_location", f"b{i}"),
                (f"b{i}", "at_location", f"c{i}"),
                (f"a{i}", "at_location", f"c{i}"),
            ])

        rules = RelationProposalEngine(_rels(*triples)).discover_relation_rules(
            min_count=1,
            min_rule_total=10,
            include_disabled_relations=True,
        )

        assert ("at_location", "at_location") not in rules

    def test_disabled_relation_rules_skipped_by_default(self):
        triples = []
        for i in range(10):
            triples.extend([
                (f"a{i}", "at_location", f"b{i}"),
                (f"b{i}", "at_location", f"c{i}"),
                (f"a{i}", "at_location", f"c{i}"),
            ])

        rules = RelationProposalEngine(_rels(*triples)).discover_relation_rules(
            min_count=1,
            min_rule_total=1,
        )

        assert rules == {}


class TestRelationProposals:
    def test_proposes_learned_relation_for_novel_chain(self):
        proposals = RelationProposalEngine(_rels(
            ("wheel", "part_of", "car"),
            ("car", "made_of", "metal"),
            ("wheel", "made_of", "metal"),
            ("door", "part_of", "car"),
            ("door", "made_of", "metal"),
            ("hood", "part_of", "car"),
            ("hood", "made_of", "metal"),
            ("seat", "part_of", "car"),
        )).propose_relations(
            min_count=3,
            min_confidence=0.4,
            min_rule_total=1,
        )

        proposal = _proposal_map(proposals)[("seat", "made_of", "metal")]
        assert isinstance(proposal, RelationProposal)
        assert proposal.confidence == pytest.approx(4 / 8)
        assert proposal.evidence == ["car"]
        assert proposal.original_relation == "made_of"
        assert "learned relation rule: part_of -> made_of => made_of" in proposal.reason
        assert "support=3/3" in proposal.reason
        assert "smoothed_conf=0.500" in proposal.reason
        assert "fanout=1" in proposal.reason

    def test_skips_existing_direct_edge(self):
        proposals = RelationProposalEngine(_rels(
            ("wheel", "part_of", "car"),
            ("car", "made_of", "metal"),
            ("wheel", "made_of", "metal"),
            ("door", "part_of", "car"),
            ("door", "made_of", "metal"),
            ("hood", "part_of", "car"),
            ("hood", "made_of", "metal"),
        )).propose_relations(min_count=3)

        assert ("wheel", "made_of", "metal") not in _proposal_map(proposals)
        assert proposals == []

    def test_hub_penalty_applies(self):
        triples = [
            ("wheel", "part_of", "car"),
            ("car", "made_of", "metal"),
            ("wheel", "made_of", "metal"),
            *[(f"leaf{i}", "part_of", "hub") for i in range(20)],
            ("hub", "made_of", "stuff"),
        ]
        proposals = RelationProposalEngine(_rels(*triples)).propose_relations(
            min_count=1,
            min_confidence=0.0,
            min_rule_total=1,
            rule_alpha=0.0,
            rule_beta=0.0,
        )

        proposal = _proposal_map(proposals)[("leaf0", "made_of", "stuff")]
        assert proposal.confidence == pytest.approx((10 / 21) ** 0.5)

    def test_relation_trust_applies_to_proposed_relation(self):
        proposals = RelationProposalEngine(_rels(
            ("wheel", "part_of", "car"),
            ("car", "made_of", "metal"),
            ("wheel", "made_of", "metal"),
            ("seat", "part_of", "car"),
        )).propose_relations(
            min_count=1,
            min_confidence=0.0,
            min_rule_total=1,
            rule_alpha=0.0,
            rule_beta=0.0,
            relation_trust={"made_of": 0.25},
        )

        proposal = _proposal_map(proposals)[("seat", "made_of", "metal")]
        assert proposal.confidence == pytest.approx(0.25)

    def test_node_quality_applies(self):
        noisy = "caf\u00e9"
        proposals = RelationProposalEngine(_rels(
            ("wheel", "part_of", "car"),
            ("car", "made_of", "metal"),
            ("wheel", "made_of", "metal"),
            (noisy, "part_of", "cup"),
            ("cup", "made_of", "clay"),
        )).propose_relations(
            min_count=1,
            min_confidence=0.0,
            min_rule_total=1,
            rule_alpha=0.0,
            rule_beta=0.0,
            use_node_quality=True,
            min_node_quality=0.0,
        )

        proposal = _proposal_map(proposals)[(noisy, "made_of", "clay")]
        assert proposal.confidence == pytest.approx(0.2)

    def test_node_quality_threshold_filters_low_quality_nodes(self):
        proposals = RelationProposalEngine(_rels(
            ("wheel", "part_of", "car"),
            ("car", "made_of", "metal"),
            ("wheel", "made_of", "metal"),
            ("epic_fail", "part_of", "cup"),
            ("cup", "made_of", "clay"),
        )).propose_relations(
            min_count=1,
            min_rule_total=1,
            use_node_quality=True,
            min_node_quality=0.3,
        )

        assert ("epic_fail", "made_of", "clay") not in _proposal_map(proposals)

    def test_high_fanout_intermediate_skipped(self):
        triples = [
            ("wheel", "part_of", "car"),
            ("car", "made_of", "metal"),
            ("wheel", "made_of", "metal"),
            ("seat", "part_of", "hub"),
            *[("hub", "made_of", f"stuff{i}") for i in range(5)],
        ]

        proposals = RelationProposalEngine(_rels(*triples)).propose_relations(
            min_count=1,
            min_rule_total=1,
            min_confidence=0.0,
            max_intermediate_relation_fanout=4,
        )

        assert not any(p.source == "seat" for p in proposals)

    def test_low_fanout_intermediate_survives(self):
        triples = [
            ("wheel", "part_of", "car"),
            ("car", "made_of", "metal"),
            ("wheel", "made_of", "metal"),
            ("seat", "part_of", "hub"),
            ("hub", "made_of", "stuff0"),
            ("hub", "made_of", "stuff1"),
        ]

        proposals = RelationProposalEngine(_rels(*triples)).propose_relations(
            min_count=1,
            min_rule_total=1,
            min_confidence=0.0,
            max_intermediate_relation_fanout=4,
        )

        assert any(p.source == "seat" for p in proposals)

    def test_disabled_relation_proposals_skipped_by_default(self):
        triples = []
        for i in range(10):
            triples.extend([
                (f"a{i}", "at_location", f"b{i}"),
                (f"b{i}", "at_location", f"c{i}"),
                (f"a{i}", "at_location", f"c{i}"),
            ])
        triples.extend([
            ("novel", "at_location", "bridge"),
            ("bridge", "at_location", "target"),
        ])

        proposals = RelationProposalEngine(_rels(*triples)).propose_relations(
            min_count=1,
            min_rule_total=1,
            min_confidence=0.0,
        )

        assert proposals == []


class TestRelationProposalAuditExport:
    def test_audit_export_writes_expected_columns(self, tmp_path):
        csv_path = _write_csv(tmp_path, [
            ("wheel", "part_of", "car"),
            ("car", "made_of", "metal"),
            ("wheel", "made_of", "metal"),
            ("seat", "part_of", "car"),
        ])
        rows = build_audit_rows(
            csv_path,
            min_count=1,
            min_confidence=0.0,
            min_rule_total=1,
            limit=20,
        )

        assert rows
        assert set(rows[0]) == set(COLUMNS)
        assert rows[0]["manual_label"] == ""
        assert rows[0]["notes"] == ""

        out = tmp_path / "audit.csv"
        written = write_audit_csv(rows, str(out))
        assert written == len(rows)
        with open(out, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert list(reader.fieldnames) == COLUMNS
            read_rows = list(reader)
        assert len(read_rows) == len(rows)
