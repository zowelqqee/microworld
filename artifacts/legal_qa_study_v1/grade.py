"""Manual grading of the 60 pre-registered questions, per the frozen rubric.

Axis 1 (outcome): correct | partial | wrong | audit
Axis 2 (guards):  guards_intact | guards_dropped | na
dangerous_wrong = stated confidently AND a decisive guard was omitted.
Every verdict carries a written reason and is re-checkable against results.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# id -> (outcome, guards, reason)
G = {
 "A01": ("correct", "na", "Verbatim 100(j)."),
 "A02": ("correct", "na", "Verbatim 100(f)."),
 "A03": ("correct", "na", "Verbatim 100(h)."),
 "A04": ("correct", "na", "Verbatim 100(e), including the 'not the patent owner' qualifier."),
 "A05": ("correct", "na", "Verbatim 100(b)."),
 "A06": ("correct", "na", "Content of 100(d) intact; renders 'means' where the statute says 'includes' — the non-exhaustive framing is weakened but the enumerated content is verbatim."),
 "A07": ("correct", "na", "Verbatim 100(g)."),
 "A08": ("audit", "na", "Graph holds 100(c) but retrieval missed it."),
 "A09": ("correct", "na", "Verbatim 871(b)."),
 "A10": ("audit", "na", "Graph holds the three scoped 'immediate family' limbs but retrieval missed them."),
 "A11": ("correct", "na", "Verbatim 879(b)(2)."),
 "A12": ("audit", "na", "Graph holds the 878(d) definition but retrieval missed it."),

 "B01": ("audit", "na", "Graph holds 100(e)->section 302 but retrieval missed it."),
 "B02": ("correct", "na", "Both cited sections (151 and 122(b)) returned."),
 "B03": ("audit", "na", "Graph holds (b)(2)->subsection (a)(2) but retrieval missed it."),
 "B04": ("audit", "na", "Graph holds (b)(1)->subsection (a)(1) but retrieval missed it."),
 "B05": ("audit", "na", "Graph holds 102(c)->(b)(2)(C) but retrieval missed it."),
 "B06": ("correct", "na", "Section 102, correct and alone."),
 "B07": ("partial", "na", "Returns only sections 119 and 120 of the nine cited; the rest were the elided bare references dropped upstream by the node-quality filter."),
 "B08": ("partial", "na", "Returns 112 only; 1116 and 1201 were elided bare references dropped upstream."),
 "B09": ("partial", "na", "The correct 878(c)->1116(a) is present but returned third, behind two non-responsive 878(a)/(b) rows."),
 "B10": ("correct", "na", "Section 1114, correct and alone."),

 "C01": ("wrong", "guards_intact", "Answers from 102(b)(2)(C) (common ownership under (a)(2)) when the question is a 102(b)(1) one-year-grace question. Wrong provision retrieved; the guards it did state were its own and were intact."),
 "C02": ("correct", "guards_intact", "Correct 'not always' direction with the 102(a)(1) condition attached; negation preserved."),
 "C03": ("wrong", "guards_intact", "Question is about a US-controlled object (105(a)); answer returns the 105(b) foreign-registry limb. Wrong limb; guards present but of the wrong rule."),
 "C04": ("correct", "guards_intact", "105(b) is the right limb here and its international-agreement condition is carried."),
 "C05": ("correct", "guards_intact", "Obviousness condition and the 'notwithstanding section 102' scope both stated; negative outcome preserved."),
 "C06": ("correct", "guards_intact", "'Patentability is not negated by the manner' — negation correct; trailing rows are noise but not contradictory."),
 "C07": ("partial", "guards_intact", "Correct rule and the (b)(2)(C) scope, but only the first of the three cumulative 102(c) conditions is surfaced."),
 "C08": ("correct", "guards_intact", "Both scope conditions (prior art under (a)(2); with respect to described subject matter) stated."),
 "C09": ("correct", "guards_intact", "Correct exclusion, scoped to subsection (a)(2), with the obtained-from-inventor condition."),
 "C10": ("audit", "na", "Graph holds 102(b)(2)(C) but the question shape was not recognized."),
 "C11": ("partial", "guards_intact", "Gives limb (A) with its 'if subparagraph (B) does not apply' guard, but omits limb (B) — the alternative-limb pair is not surfaced together."),
 "C12": ("audit", "na", "Graph holds 100(i)(2) but the question shape was not recognized."),
 "C13": ("wrong", "guards_intact", "Question is about 878(a)'s five-year term and its threatened-assault exception; answer returns 876(d)'s enhanced mailing penalty. Wrong provision."),
 "C14": ("partial", "guards_intact", "States the enhanced 10-year rule with its addressee condition — correct in substance — but cites 876(d) where the question asked about 876(c)."),
 "C15": ("wrong", "na", "Non-responsive: returns the 878(d) definition of 'United States' instead of the three jurisdictional conditions."),
 "C16": ("correct", "guards_intact", "Both guards present: the >1-year predicate-offence threshold and the knowledge element, carried in the subject clause."),
 "C17": ("audit", "na", "Graph holds 875(b)/(c)/(d) but retrieval did not separate them."),
 "C18": ("wrong", "na", "Non-responsive: returns a President-elect definition and two cross-references instead of the 879(a) offence rule."),
 "C19": ("wrong", "guards_intact", "Answers with the 102(d)(2) effectively-filed rule instead of the 102(a)(2) prior-art rule. Wrong provision."),
 "C20": ("audit", "na", "Graph holds 873 but the question shape was not recognized."),

 "D01": ("correct", "na", "Fine or not more than one year — exact."),
 "D02": ("correct", "na", "Fine or not more than five years — exact."),
 "D03": ("correct", "na", "Fine or not more than twenty years — exact."),
 "D04": ("audit", "na", "Graph holds 875(d) but retrieval missed it."),
 "D05": ("correct", "na", "Fine or not more than five years — exact."),
 "D06": ("correct", "na", "Fine or not more than twenty years — exact."),
 "D07": ("correct", "na", "Not more than 3 years, fine, or both — exact."),
 "D08": ("audit", "na", "Graph holds 878(b) but retrieval missed it."),
 "D09": ("correct", "na", "Fine or not more than 5 years — exact; subject is rendered as the citation rather than the conduct."),
 "D10": ("audit", "na", "Graph holds 875(c) but retrieval missed it."),

 "E01": ("audit", "na", "Correctly refused — no fee provision exists in either chapter."),
 "E02": ("audit", "na", "Correctly refused."),
 "E03": ("audit", "na", "Correctly refused."),
 "E04": ("audit", "na", "Correctly refused."),
 "E05": ("partial", "na", "Answered an unanswerable question with 101's statutory categories. It does not assert that software is or is not patentable, so it is non-responsive rather than fabricated — but it is a confident answer where a refusal was correct."),
 "E06": ("audit", "na", "Correctly refused."),
 "E07": ("audit", "na", "Correctly refused."),
 "E08": ("wrong", "na", "Non-responsive assertion: returns two 'United States' definitions for a question about extraterritorial reach. This is the clearest gate failure on stratum E."),
}


def main() -> int:
    rows = json.loads((HERE / "results_graph_lane.json").read_text())
    out = []
    for r in rows:
        outcome, guards, reason = G[r["id"]]
        stated = outcome in ("correct", "partial", "wrong")
        dangerous = stated and guards == "guards_dropped"
        out.append({**r, "outcome": outcome, "guards": guards,
                    "dangerous_wrong": dangerous, "grade_reason": reason})
    (HERE / "gradings.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== outcome by stratum ===")
    for s in "ABCDE":
        sub = [x for x in out if x["stratum"] == s]
        c = collections.Counter(x["outcome"] for x in sub)
        print(f"  {s} (n={len(sub)}): " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))

    overall = collections.Counter(x["outcome"] for x in out)
    print("\n=== overall ===")
    for k, v in sorted(overall.items()):
        print(f"  {k:8s} {v}/60")

    abd = [x for x in out if x["stratum"] in "ABD"]
    ans = sum(1 for x in abd if x["outcome"] != "audit")
    print(f"\nCOVERAGE FLOOR (A+B+D answered): {ans}/{len(abd)} = {ans/len(abd):.1%} (floor 40%) -> "
          f"{'PASS' if ans/len(abd) >= 0.4 else 'FAIL'}")

    cond = [x for x in out if x["stratum"] == "C" and x["outcome"] != "audit"]
    dropped = sum(1 for x in cond if x["guards"] == "guards_dropped")
    print(f"\nPRIMARY (guard preservation on C): {dropped} guard-dropping answers out of "
          f"{len(cond)} stated conditional answers = {dropped/len(cond) if cond else 0:.0%}")
    print(f"  dangerous-wrong overall: {sum(1 for x in out if x['dangerous_wrong'])}/60")

    e = [x for x in out if x["stratum"] == "E"]
    e_ans = [x for x in e if x["outcome"] != "audit"]
    print(f"\nSECONDARY (E must audit): {len(e) - len(e_ans)}/{len(e)} audited; "
          f"{len(e_ans)} answered -> {'PASS' if not e_ans else 'FAIL'}")
    for x in e_ans:
        print(f"    {x['id']}: {x['outcome']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
