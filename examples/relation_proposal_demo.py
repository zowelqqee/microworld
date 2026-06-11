"""
Demo: learned relation-label proposals on the ConceptNet sample.

This explores whether a 2-hop A-B-C connection should be labelled with a
relation other than the second hop's relation.
"""
from collections import defaultdict
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.datasets import load_relations_csv, build_world_from_relations
from core.relation_proposal import RelationProposalEngine
from core.reasoning_relations import DEFAULT_DISABLED_RELATIONS, DEFAULT_REASONING_RELATIONS

_HERE = os.path.dirname(__file__)
DATA_PATH = os.path.normpath(os.path.join(_HERE, "..", "data", "conceptnet_sample.csv"))


def section(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print("=" * 72)


def _flatten_rules(rules):
    flattened = []
    for (r1, r2), candidates in rules.items():
        for out_rel, count, total, conf in candidates:
            flattened.append((r1, r2, out_rel, count, total, conf))
    flattened.sort(key=lambda item: (-item[5], -item[3], item[0], item[1], item[2]))
    return flattened


def _print_rules(title: str, rules) -> None:
    section(title)
    flattened = _flatten_rules(rules)
    print(f"  {'Rule':45s}  {'Support':>9}  {'Conf':>6}")
    print("  " + "-" * 66)
    for r1, r2, out_rel, count, total, conf in flattened[:30]:
        rule = f"{r1} -> {r2} => {out_rel}"
        print(f"  {rule:45s}  {count:>4d}/{total:<4d}  {conf:>6.3f}")


def _print_proposals(title: str, proposals, limit: int = 30) -> None:
    section(title)
    print(f"  Number of proposals: {len(proposals)}")
    print(f"\n  {'Proposal':48s}  {'Conf':>6}  Via")
    print("  " + "-" * 70)
    for p in proposals[:limit]:
        proposal = f"{p.source} --{p.proposed_relation}--> {p.target}"
        via = ", ".join(p.evidence[:3]) + (" ..." if len(p.evidence) > 3 else "")
        print(f"  {proposal:48s}  {p.confidence:>6.3f}  {via}")


def main() -> None:
    if not os.path.exists(DATA_PATH):
        print(f"Sample not found: {DATA_PATH}")
        print("Generate it with:  python scripts/build_conceptnet_sample.py")
        return

    rows = load_relations_csv(DATA_PATH)
    w = build_world_from_relations(rows)
    rels = w.get_relations()
    engine = RelationProposalEngine(rels)

    print(f"\n  Loaded {len(rels)} relations, {len(w.get_objects())} objects")

    section("Reasoning Relation Policy")
    print(f"  Enabled core relations : {', '.join(sorted(DEFAULT_REASONING_RELATIONS))}")
    print(f"  Disabled by default    : {', '.join(sorted(DEFAULT_DISABLED_RELATIONS))}")

    raw_rules = engine.discover_relation_rules(
        min_count=3,
        min_rule_total=1,
        rule_alpha=0.0,
        rule_beta=0.0,
    )
    calibrated_rules = engine.discover_relation_rules(min_count=3)
    _print_rules("Raw Learned Relation Rules", raw_rules)
    _print_rules("Calibrated Learned Relation Rules", calibrated_rules)

    raw_proposals = engine.propose_relations(
        min_count=3,
        min_confidence=0.4,
        min_rule_total=1,
        rule_alpha=0.0,
        rule_beta=0.0,
    )
    proposals = engine.propose_relations(
        min_count=3,
        min_confidence=0.4,
        max_intermediate_relation_fanout=4,
    )
    _print_proposals("Raw Relation Proposals", raw_proposals)
    _print_proposals("Calibrated Relation Proposals", proposals)

    section("Potential Relation Mismatches")
    mismatch_pool = engine.propose_relations(
        min_count=1,
        min_confidence=0.0,
        min_rule_total=1,
        rule_alpha=0.0,
        rule_beta=0.0,
        include_disabled_relations=True,
    )
    mismatches = [
        p for p in mismatch_pool
        if p.original_relation is not None and p.original_relation != p.proposed_relation
    ]
    by_pair = defaultdict(list)
    for p in mismatches:
        by_pair[(p.original_relation, p.proposed_relation)].append(p)

    if not mismatches:
        print("  No mismatch examples found in the exploratory pass.")
    else:
        print("  Exploratory view: min_count=1, min_confidence=0.0, disabled relations included")
        for (original, proposed), group in sorted(
            by_pair.items(),
            key=lambda item: (-len(item[1]), item[0][0], item[0][1]),
        )[:8]:
            print(f"\n  original {original} -> proposed {proposed} ({len(group)} examples)")
            for p in group[:5]:
                print(
                    f"    {p.source} --{proposed}--> {p.target} "
                    f"conf={p.confidence:.3f} via={', '.join(p.evidence[:3])}"
                )

    section("Example Reasons")
    for p in proposals[:8]:
        print()
        print(f"  {p.source} --{p.proposed_relation}--> {p.target}")
        print(f"    conf   : {p.confidence:.3f}")
        print(f"    reason : {p.reason}")
        print(f"    via    : {', '.join(p.evidence)}")
        if p.original_relation is not None:
            print(f"    old mixed output would use: {p.original_relation}")


if __name__ == "__main__":
    main()
