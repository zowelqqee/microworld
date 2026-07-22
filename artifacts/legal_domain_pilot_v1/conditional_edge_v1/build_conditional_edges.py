"""Retrospective test: re-represent the 16 conditional candidates in the new schema.

Design artifact. No network, no model calls, no runtime imports. Reads only the
completed legal-pilot review data and writes this directory's results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from schema_prototype import (  # noqa: E402
    ConditionClause,
    ConditionalRelationEdge,
    content_tokens,
    coverage,
    verify_edge,
)

RUN = HERE.parents[1] / "legal_domain_pilot_v1" / "runs" / "usc35_chapter10_v1"
CANDIDATES = {c["candidate_id"]: c for c in json.loads((RUN / "manual_review_candidates.json").read_text())}
DECISIONS = {d["candidate_id"]: d for d in json.loads((RUN / "manual_review_decisions.json").read_text())}
UNITS = {u["id"]: u for u in json.loads((RUN / "source_units.json").read_text())}


def src(candidate_id: str) -> str:
    return CANDIDATES[candidate_id]["source_text"]


def statutory_body(candidate_id: str) -> str:
    """Provision text with the deterministic citation header removed."""
    return UNITS[CANDIDATES[candidate_id]["batch_id"]]["statutory_body_text"]


def C(text: str, span: str, kind: str = "factual") -> ConditionClause:
    return ConditionClause(text=text, evidence_span=span, kind=kind)


SCOPE_A1 = "under subsection (a)(1)"
SCOPE_A2 = "under subsection (a)(2)"
SCOPE_TITLE = "for the purposes of this title"

# Cross-provision assembly: 102(c)'s three conditions are joined by "and", so a
# faithful edge needs all three, which live in three separate source units.
C_UNITS = ["usc35c10-020:1", "usc35c10-021:1", "usc35c10-022:0"]
C_EVIDENCE = " ".join(src(i) for i in C_UNITS)

EDGES: list[tuple[str, ConditionalRelationEdge]] = [
    (
        "usc35c10-010:1",
        ConditionalRelationEdge(
            id="ce-010:1",
            subject="the effective filing date for a claimed invention in an application for reissue or reissued patent",
            predicate="determined_by_deeming",
            object="the claim to the invention to have been contained in the patent for which reissue was sought",
            evidence_sentence=src("usc35c10-010:1"),
            stated_in="35 U.S.C. §100(i)(2)",
            conditions=(C("the term is used in this title", "When used in this title", "scope"),),
            exceptions=(C("the context otherwise indicates", "unless the context otherwise indicates"),),
        ),
    ),
    (
        "usc35c10-012:1",
        ConditionalRelationEdge(
            id="ce-012:1",
            subject="whoever invents or discovers any new and useful process, machine, manufacture, or composition of matter, or any new and useful improvement thereof",
            predicate="may_obtain",
            object="a patent",
            evidence_sentence=src("usc35c10-012:1"),
            stated_in="35 U.S.C. §101",
            conditions=(
                C(
                    "the conditions and requirements of this title are satisfied",
                    "subject to the conditions and requirements of this title",
                ),
            ),
        ),
    ),
    (
        "usc35c10-013:1",
        ConditionalRelationEdge(
            id="ce-013:1",
            subject="a person",
            predicate="entitled_to",
            object="a patent",
            evidence_sentence=src("usc35c10-013:1"),
            stated_in="35 U.S.C. §102(a)(1)",
            polarity="negate",
            conditions=(
                C(
                    "the claimed invention was patented, described in a printed publication, or in public use, on sale, or otherwise available to the public before the effective filing date of the claimed invention",
                    "the claimed invention was patented, described in a printed publication, or in public use, on sale, or otherwise available to the public before the effective filing date of the claimed invention",
                ),
            ),
        ),
    ),
    (
        "usc35c10-014:2",
        ConditionalRelationEdge(
            id="ce-014:2",
            subject="a person",
            predicate="entitled_to",
            object="a patent",
            evidence_sentence=src("usc35c10-014:2"),
            stated_in="35 U.S.C. §102(a)(2)",
            polarity="negate",
            conditions=(
                C(
                    "the claimed invention was described in a patent issued under section 151, or in an application for patent published or deemed published under section 122(b), in which the patent or application names another inventor and was effectively filed before the effective filing date of the claimed invention",
                    "the claimed invention was described in a patent issued under section 151, or in an application for patent published or deemed published under section 122(b), in which the patent or application, as the case may be, names another inventor and was effectively filed before the effective filing date of the claimed invention",
                ),
            ),
        ),
    ),
    (
        "usc35c10-015:0",
        ConditionalRelationEdge(
            id="ce-015:0",
            subject="a disclosure made 1 year or less before the effective filing date of a claimed invention",
            predicate="is_prior_art_to",
            object="the claimed invention",
            evidence_sentence=src("usc35c10-015:0"),
            stated_in="35 U.S.C. §102(b)(1)(A)",
            polarity="negate",
            conditions=(
                C("the prior-art determination is made under subsection (a)(1)", SCOPE_A1, "scope"),
                C(
                    "the disclosure was made by the inventor or joint inventor or by another who obtained the subject matter disclosed directly or indirectly from the inventor or a joint inventor",
                    "the disclosure was made by the inventor or joint inventor or by another who obtained the subject matter disclosed directly or indirectly from the inventor or a joint inventor",
                ),
            ),
        ),
    ),
    (
        "usc35c10-016:2",
        ConditionalRelationEdge(
            id="ce-016:2",
            subject="a disclosure made 1 year or less before the effective filing date of a claimed invention",
            predicate="is_prior_art_to",
            object="the claimed invention",
            evidence_sentence=src("usc35c10-016:2"),
            stated_in="35 U.S.C. §102(b)(1)(B)",
            polarity="negate",
            conditions=(
                C("the prior-art determination is made under subsection (a)(1)", SCOPE_A1, "scope"),
                C(
                    "the subject matter disclosed had, before such disclosure, been publicly disclosed by the inventor or a joint inventor or another who obtained the subject matter disclosed directly or indirectly from the inventor or a joint inventor",
                    "the subject matter disclosed had, before such disclosure, been publicly disclosed by the inventor or a joint inventor or another who obtained the subject matter disclosed directly or indirectly from the inventor or a joint inventor",
                ),
            ),
        ),
    ),
    (
        "usc35c10-017:1",
        ConditionalRelationEdge(
            id="ce-017:1",
            subject="a disclosure",
            predicate="is_prior_art_to",
            object="a claimed invention",
            evidence_sentence=src("usc35c10-017:1"),
            stated_in="35 U.S.C. §102(b)(2)(A)",
            polarity="negate",
            conditions=(
                C("the prior-art determination is made under subsection (a)(2)", SCOPE_A2, "scope"),
                C(
                    "the subject matter disclosed was obtained directly or indirectly from the inventor or a joint inventor",
                    "the subject matter disclosed was obtained directly or indirectly from the inventor or a joint inventor",
                ),
            ),
        ),
    ),
    (
        "usc35c10-018:1",
        ConditionalRelationEdge(
            id="ce-018:1",
            subject="a disclosure",
            predicate="is_prior_art_to",
            object="a claimed invention",
            evidence_sentence=src("usc35c10-018:1"),
            stated_in="35 U.S.C. §102(b)(2)(B)",
            polarity="negate",
            conditions=(
                C("the prior-art determination is made under subsection (a)(2)", SCOPE_A2, "scope"),
                C(
                    "the subject matter disclosed had, before such subject matter was effectively filed under subsection (a)(2), been publicly disclosed by the inventor or a joint inventor or another who obtained the subject matter disclosed directly or indirectly from the inventor or a joint inventor",
                    "the subject matter disclosed had, before such subject matter was effectively filed under subsection (a)(2), been publicly disclosed by the inventor or a joint inventor or another who obtained the subject matter disclosed directly or indirectly from the inventor or a joint inventor",
                ),
            ),
        ),
    ),
    (
        "usc35c10-019:1",
        ConditionalRelationEdge(
            id="ce-019:1",
            subject="a disclosure",
            predicate="is_prior_art_to",
            object="a claimed invention",
            evidence_sentence=src("usc35c10-019:1"),
            stated_in="35 U.S.C. §102(b)(2)(C)",
            polarity="negate",
            conditions=(
                C("the prior-art determination is made under subsection (a)(2)", SCOPE_A2, "scope"),
                C(
                    "the subject matter disclosed and the claimed invention, not later than the effective filing date of the claimed invention, were owned by the same person or subject to an obligation of assignment to the same person",
                    "the subject matter disclosed and the claimed invention, not later than the effective filing date of the claimed invention, were owned by the same person or subject to an obligation of assignment to the same person",
                ),
            ),
        ),
    ),
    (
        "usc35c10-020:1",
        ConditionalRelationEdge(
            id="ce-020:1",
            subject="subject matter disclosed",
            predicate="deemed_commonly_owned_with",
            object="a claimed invention",
            evidence_sentence=C_EVIDENCE,
            stated_in="35 U.S.C. §102(c)(1)-(3)",
            conditions=(
                C("the deeming is applied in applying the provisions of subsection (b)(2)(C)",
                  "in applying the provisions of subsection (b)(2)(C)", "scope"),
                C(
                    "the subject matter disclosed was developed and the claimed invention was made by, or on behalf of, 1 or more parties to a joint research agreement that was in effect on or before the effective filing date of the claimed invention",
                    "the subject matter disclosed was developed and the claimed invention was made by, or on behalf of, 1 or more parties to a joint research agreement that was in effect on or before the effective filing date of the claimed invention",
                ),
                C(
                    "the claimed invention was made as a result of activities undertaken within the scope of the joint research agreement",
                    "the claimed invention was made as a result of activities undertaken within the scope of the joint research agreement",
                ),
                C(
                    "the application for patent for the claimed invention discloses or is amended to disclose the names of the parties to the joint research agreement",
                    "the application for patent for the claimed invention discloses or is amended to disclose the names of the parties to the joint research agreement",
                ),
            ),
        ),
    ),
    (
        "usc35c10-023:3",
        ConditionalRelationEdge(
            id="ce-023:3",
            subject="a patent or application for patent",
            predicate="effectively_filed_as_of",
            object="the actual filing date of the patent or the application for patent",
            evidence_sentence=src("usc35c10-023:3"),
            stated_in="35 U.S.C. §102(d)(1)",
            conditions=(
                C(
                    "the determination is whether the patent or application is prior art to a claimed invention under subsection (a)(2)",
                    "For purposes of determining whether a patent or application for patent is prior art to a claimed invention under subsection (a)(2)",
                    "scope",
                ),
                C(
                    "the determination is made with respect to subject matter described in the patent or application",
                    "with respect to any subject matter described in the patent or application",
                    "scope",
                ),
                C("paragraph (2) does not apply", "if paragraph (2) does not apply"),
            ),
        ),
    ),
    (
        "usc35c10-024:9",
        ConditionalRelationEdge(
            id="ce-024:9",
            subject="a patent or application for patent",
            predicate="effectively_filed_as_of",
            object="the filing date of the earliest such application that describes the subject matter",
            evidence_sentence=src("usc35c10-024:9"),
            stated_in="35 U.S.C. §102(d)(2)",
            conditions=(
                C(
                    "the determination is whether the patent or application is prior art to a claimed invention under subsection (a)(2)",
                    "For purposes of determining whether a patent or application for patent is prior art to a claimed invention under subsection (a)(2)",
                    "scope",
                ),
                C(
                    "the determination is made with respect to subject matter described in the patent or application",
                    "with respect to any subject matter described in the patent or application",
                    "scope",
                ),
                C(
                    "the patent or application for patent is entitled to claim a right of priority under section 119, 365(a), 365(b), 386(a), or 386(b), or to claim the benefit of an earlier filing date under section 120, 121, 365(c), or 386(c), based upon 1 or more prior filed applications for patent",
                    "if the patent or application for patent is entitled to claim a right of priority under section 119, 365(a), 365(b), 386(a), or 386(b), or to claim the benefit of an earlier filing date under section 120, 121, 365(c), or 386(c), based upon 1 or more prior filed applications for patent",
                ),
            ),
        ),
    ),
    (
        "usc35c10-025:2",
        ConditionalRelationEdge(
            id="ce-025:2",
            subject="a patent",
            predicate="may_be_obtained_for",
            object="a claimed invention",
            evidence_sentence=src("usc35c10-025:2"),
            stated_in="35 U.S.C. §103",
            polarity="negate",
            conditions=(
                C(
                    "the differences between the claimed invention and the prior art are such that the claimed invention as a whole would have been obvious before the effective filing date of the claimed invention to a person having ordinary skill in the art to which the claimed invention pertains",
                    "the differences between the claimed invention and the prior art are such that the claimed invention as a whole would have been obvious before the effective filing date of the claimed invention to a person having ordinary skill in the art to which the claimed invention pertains",
                ),
                C(
                    "this applies even where the claimed invention is not identically disclosed as set forth in section 102",
                    "notwithstanding that the claimed invention is not identically disclosed as set forth in section 102",
                    "scope",
                ),
            ),
        ),
    ),
    (
        "usc35c10-025:3",
        ConditionalRelationEdge(
            id="ce-025:3",
            subject="patentability",
            predicate="negated_by",
            object="the manner in which the invention was made",
            evidence_sentence=src("usc35c10-025:3"),
            stated_in="35 U.S.C. §103",
            polarity="negate",
        ),
    ),
    (
        "usc35c10-026:1",
        ConditionalRelationEdge(
            id="ce-026:1",
            subject="any invention made, used or sold in outer space on a space object or component thereof under the jurisdiction or control of the United States",
            predicate="considered_made_used_or_sold_in",
            object="the United States",
            evidence_sentence=src("usc35c10-026:1"),
            stated_in="35 U.S.C. §105(a)",
            conditions=(C("the determination is for the purposes of this title", SCOPE_TITLE, "scope"),),
            exceptions=(
                C(
                    "the space object or component thereof is specifically identified and otherwise provided for by an international agreement to which the United States is a party",
                    "except with respect to any space object or component thereof that is specifically identified and otherwise provided for by an international agreement to which the United States is a party",
                ),
                C(
                    "the space object or component thereof is carried on the registry of a foreign state in accordance with the Convention on Registration of Objects Launched into Outer Space",
                    "or with respect to any space object or component thereof that is carried on the registry of a foreign state in accordance with the Convention on Registration of Objects Launched into Outer Space",
                ),
            ),
        ),
    ),
    (
        "usc35c10-027:1",
        ConditionalRelationEdge(
            id="ce-027:1",
            subject="any invention made, used or sold in outer space on a space object or component thereof that is carried on the registry of a foreign state in accordance with the Convention on Registration of Objects Launched into Outer Space",
            predicate="considered_made_used_or_sold_in",
            object="the United States",
            evidence_sentence=src("usc35c10-027:1"),
            stated_in="35 U.S.C. §105(b)",
            conditions=(
                C("the determination is for the purposes of this title", SCOPE_TITLE, "scope"),
                C(
                    "it is specifically so agreed in an international agreement between the United States and the state of registry",
                    "if specifically so agreed in an international agreement between the United States and the state of registry",
                ),
            ),
        ),
    ),
]

# Supplementary case named in the task: the 100(i)(1) alternative-limbs pair.
# These are *definition* candidates, not part of the 16, and are scored apart.
SUPPLEMENTARY: list[tuple[str, ConditionalRelationEdge]] = [
    (
        "usc35c10-008:0",
        ConditionalRelationEdge(
            id="ce-008:0",
            subject="effective filing date",
            predicate="means",
            object="the actual filing date of the patent or the application for the patent containing a claim to the invention",
            evidence_sentence=src("usc35c10-008:0"),
            stated_in="35 U.S.C. §100(i)(1)(A)",
            conditions=(
                C("the term is used in this title", "When used in this title", "scope"),
                C("subparagraph (B) does not apply", "if subparagraph (B) does not apply"),
            ),
            exceptions=(C("the context otherwise indicates", "unless the context otherwise indicates"),),
        ),
    ),
    (
        "usc35c10-009:0",
        ConditionalRelationEdge(
            id="ce-009:0",
            subject="effective filing date",
            predicate="means",
            object="the filing date of the earliest application for which the patent or application is entitled, as to such invention, to a right of priority under section 119, 365(a), 365(b), 386(a), or 386(b) or to the benefit of an earlier filing date under section 120, 121, 365(c), or 386(c)",
            evidence_sentence=src("usc35c10-009:0"),
            stated_in="35 U.S.C. §100(i)(1)(B)",
            conditions=(C("the term is used in this title", "When used in this title", "scope"),),
            exceptions=(C("the context otherwise indicates", "unless the context otherwise indicates"),),
        ),
    ),
]


# Semantic verdicts. The mechanical checks in verify_edge() are automated and
# reproducible; these are reviewer judgements recorded per candidate with their
# reason, so a reader can disagree with a specific one without re-deriving the
# whole test. "FAIL" means the schema itself cannot hold the content.
SEMANTIC: dict[str, tuple[str, str]] = {
    "usc35c10-010:1": ("PASS", ""),
    "usc35c10-012:1": ("PASS", "unevaluable_condition: 'subject to the conditions and requirements of this title' is stored with a literal span but is an open-ended reference to an entire title — inspectable, not decidable"),
    "usc35c10-013:1": ("PASS", ""),
    "usc35c10-014:2": ("PASS", ""),
    "usc35c10-015:0": ("PASS", "original hallucination fixed: subject is now the disclosure, not the provision citation"),
    "usc35c10-016:2": ("PASS", ""),
    "usc35c10-017:1": ("PASS", ""),
    "usc35c10-018:1": ("PASS", ""),
    "usc35c10-019:1": ("PASS", ""),
    "usc35c10-020:1": ("FAIL", "disjunctive_consequence: 'deemed to have been owned by the same person OR subject to an obligation of assignment to the same person' is compressed into the predicate name; the schema has no representation for a disjunctive consequence. Also the only candidate requiring cross-provision assembly of conditions (c)(1)-(3)"),
    "usc35c10-023:3": ("PASS", "original hallucination fixed: predicate uses the statute's own term 'effectively filed', not the distinct defined term 'effective filing date'"),
    "usc35c10-024:9": ("PASS", ""),
    "usc35c10-025:2": ("PASS", "arity_coercion: 'a patent may not be obtained' is a unary proposition forced into a binary triple; truth preserved"),
    "usc35c10-025:3": ("PASS", ""),
    "usc35c10-026:1": ("PASS", "flagship exception case: both dropped exceptions restored as first-class clauses with literal spans"),
    "usc35c10-027:1": ("PASS", ""),
    "usc35c10-008:0": ("PASS", "alternative limb (A), guarded by 'if subparagraph (B) does not apply'"),
    "usc35c10-009:0": ("PASS", "alternative limb (B), the default limb"),
}


def flat_coverage(candidate_id: str, provision: str) -> float:
    """Content-word coverage of the ORIGINAL flat triple, for comparison."""
    d = DECISIONS[candidate_id]
    captured = content_tokens(" ".join([d["subject"], d["predicate"], d["object"]]))
    target = content_tokens(provision)
    return len(target & captured) / len(target) if target else 1.0


def contradiction_report(edges: list[tuple[str, ConditionalRelationEdge]]) -> list[dict]:
    """Find edge pairs that would collide if conditions were ignored.

    A pair collides when it shares subject+predicate but asserts a different
    object, or shares the whole triple with opposite polarity.  Under the flat
    schema such a pair is a bare contradiction.  Under this schema it is
    resolved iff at least one member carries a guarding condition.
    """

    findings: list[dict] = []
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            a, b = edges[i][1], edges[j][1]
            if (a.subject.lower(), a.predicate) != (b.subject.lower(), b.predicate):
                continue
            same_object = a.object.lower() == b.object.lower()
            if same_object and a.polarity == b.polarity:
                kind = "disjunctive_limbs"  # same claim, alternative sufficient conditions
            elif same_object:
                kind = "polarity_collision"
            else:
                kind = "alternative_limbs"
            findings.append(
                {
                    "kind": kind,
                    "a": a.stated_in,
                    "b": b.stated_in,
                    "a_conditions": len(a.conditions),
                    "b_conditions": len(b.conditions),
                    "resolved_by_conditions": bool(a.conditions or b.conditions),
                    "guard_text": [c.text for c in (a.conditions or b.conditions)][:1],
                }
            )
    return findings


def main() -> int:
    results = []
    for group, edges in (("conditional_consequence_16", EDGES), ("supplementary_alt_limbs", SUPPLEMENTARY)):
        for candidate_id, edge in edges:
            provision = statutory_body(candidate_id)
            check = verify_edge(edge)
            check.update(
                {
                    "group": group,
                    "candidate_id": candidate_id,
                    "citation": DECISIONS[candidate_id]["citation"],
                    "original_verdict": DECISIONS[candidate_id]["verdict"],
                    "original_defect": DECISIONS[candidate_id]["defect"],
                    "original_predicate_words": len(DECISIONS[candidate_id]["predicate"].split()),
                    "semantic_verdict": SEMANTIC[candidate_id][0],
                    "semantic_caveat": SEMANTIC[candidate_id][1],
                    "structured_coverage": round(coverage(edge, provision), 3),
                    "flat_coverage": round(flat_coverage(candidate_id, provision), 3),
                    "edge": edge.to_dict(),
                }
            )
            results.append(check)

    # Per-provision union coverage: a single edge understates fidelity when one
    # provision yields several edges, so measure what ALL its edges retain.
    by_citation: dict[str, list[ConditionalRelationEdge]] = {}
    provision_text: dict[str, str] = {}
    for candidate_id, edge in (*EDGES, *SUPPLEMENTARY):
        by_citation.setdefault(edge.stated_in, []).append(edge)
        provision_text.setdefault(edge.stated_in, statutory_body(candidate_id))
    union = {}
    for citation, group in by_citation.items():
        target = content_tokens(provision_text[citation])
        captured: set[str] = set()
        for edge in group:
            captured |= content_tokens(
                " ".join(
                    [edge.subject, edge.predicate.replace("_", " "), edge.object]
                    + [c.text for c in edge.conditions]
                    + [e.text for e in edge.exceptions]
                )
            )
        union[citation] = {
            "edges": len(group),
            "union_coverage": round(len(target & captured) / len(target), 3) if target else 1.0,
            "uncovered_content_words": sorted(target - captured),
        }

    collisions = contradiction_report([*EDGES, *SUPPLEMENTARY])
    (HERE / "verification.json").write_text(
        json.dumps(
            {
                "union_coverage_by_provision": union,
                "collision_analysis": collisions,
                "unresolved_collisions": [c for c in collisions if not c["resolved_by_conditions"]],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (HERE / "conditional_edges.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("\ncollision analysis (pairs that would contradict under the flat schema):")
    for c in collisions:
        mark = "resolved" if c["resolved_by_conditions"] else "UNRESOLVED"
        print(f"  {mark:10s} {c['kind']:18s} {c['a']} vs {c['b']}")
    print("\nunion coverage per provision:")
    for citation, stats in sorted(union.items()):
        print(f"  {citation:26s} {stats['edges']} edge(s)  {stats['union_coverage']:.2f}")

    main_set = [r for r in results if r["group"] == "conditional_consequence_16"]
    print(f"{len(main_set)} conditional-consequence edges rebuilt")
    passed = [r for r in main_set if r["passes"]]
    semantic_pass = [r for r in main_set if r["semantic_verdict"] == "PASS"]
    print(f"mechanical checks passed: {len(passed)}/{len(main_set)}")
    print(f"semantic verdict PASS:    {len(semantic_pass)}/{len(main_set)}  (gate threshold: 13)")
    for r in main_set:
        flag = "ok  " if r["semantic_verdict"] == "PASS" else "FAIL"
        print(
            f"  {flag} {r['citation']:26s} pred={r['predicate_words']}w "
            f"(was {r['original_predicate_words']}w)  cond={r['condition_count']} exc={r['exception_count']} "
            f"pol={r['polarity']:6s} cov {r['flat_coverage']:.2f}->{r['structured_coverage']:.2f}"
            + (f"  {r['failures']}" if r["failures"] else "")
        )
    print("\nsupplementary (alternative limbs):")
    for r in results:
        if r["group"] != "supplementary_alt_limbs":
            continue
        print(f"  {'ok ' if r['passes'] else 'FAIL'} {r['citation']:26s} cond={r['condition_count']} {r['failures']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
