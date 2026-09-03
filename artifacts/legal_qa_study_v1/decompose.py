"""Decompose every G failure into routing / retrieval / representation / rendering.

Pre-registered as mandatory: a score without this cannot distinguish "the
architecture cannot represent legal knowledge" from "the entity-QA analyzer
cannot parse a legal question". For each failed question this asks the only
decisive question — *is the answering knowledge actually in the graph?* — by
searching the overlay for an edge whose content covers the gold answer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

_STOP = set("""a an the of to in on for by with as is are was were be been being that which who whom
whose this these those such it its any no not shall may under does do did what when which where how
you your i we they he she or and but if then than there here about into over more most some all each
every under upon""".split())


def toks(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9()]+", (text or "").lower())
            if t not in _STOP and len(t) > 2}


def main() -> int:
    rows = json.loads((HERE / "results_graph.json").read_text())
    overlay = json.loads((HERE / "legal_overlay.json").read_text())

    # Every assertion the graph can make, as one searchable text blob per item.
    blobs = []
    for item in overlay:
        if item.get("overlay_type") == "overlay_relation":
            text = f"{item['subject']} {item['predicate']} {item['object']}"
            for c in (item.get("conditions") or []) + (item.get("exceptions") or []):
                text += " " + c.get("text", "")
        elif item.get("overlay_type") == "overlay_definition":
            text = f"{item['subject']} {item['definition']}"
        else:
            continue
        blobs.append((toks(text), item))

    out = []
    for r in rows:
        gold = toks(r["gold"])
        best, best_cov = None, 0.0
        for bt, item in blobs:
            if not gold:
                continue
            cov = len(gold & bt) / len(gold)
            if cov > best_cov:
                best_cov, best = cov, item
        # Knowledge is judged present when one stored item covers a substantial
        # share of the gold answer's content words.
        supported = best_cov >= 0.35
        reason = r["answer"][:45]
        if r["decision"] != "audit":
            layer = "answered"
        elif r["stratum"] == "E":
            layer = "correct_audit"
        elif not supported:
            layer = "representation"
        elif "not understood" in reason or "universal generalization" in reason:
            layer = "routing"
        elif "don't have a definition" in reason:
            layer = "routing"
        else:
            layer = "retrieval"
        out.append({**r, "graph_support_coverage": round(best_cov, 2),
                    "graph_supports_answer": supported,
                    "best_supporting_item": (
                        f"{best.get('subject','')} | {best.get('predicate','')} | "
                        f"{best.get('object', best.get('definition',''))}"[:130] if best else ""),
                    "failure_layer": layer})

    (HERE / "decomposition.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    import collections
    print("=== failure layer, all 60 ===")
    for layer, c in collections.Counter(x["failure_layer"] for x in out).most_common():
        print(f"  {layer:16s} {c}")
    print("\n=== by stratum ===")
    for s in "ABCDE":
        sub = [x for x in out if x["stratum"] == s]
        cnt = collections.Counter(x["failure_layer"] for x in sub)
        print(f"  {s}: " + ", ".join(f"{k}={v}" for k, v in cnt.most_common()))

    failed = [x for x in out if x["failure_layer"] not in ("answered", "correct_audit")]
    have = sum(1 for x in failed if x["graph_supports_answer"])
    print(f"\nOf {len(failed)} substantive failures, the graph ALREADY CONTAINS the "
          f"answering knowledge in {have} ({have/len(failed):.0%}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
