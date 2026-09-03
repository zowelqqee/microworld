"""Build a sandboxed serving overlay from the two legal pilots' verified relations.

Study artifact. Writes only into artifacts/legal_qa_study_v1/. It does not touch
serving memory, the promoted overlay, or any production path.

Every relation here was manually reviewed and ACCEPTed in pilot v1 or v2. The 16
conditional-consequence rules are emitted in the conditional-edge schema, so the
graph carries their conditions/exceptions/polarity as first-class fields rather
than welded into a predicate string.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "artifacts" / "legal_domain_pilot_v1" / "conditional_edge_v1"))

RUNS = {
    "v1": ROOT / "artifacts/legal_domain_pilot_v1/runs/usc35_chapter10_v1",
    "v2": ROOT / "artifacts/legal_domain_pilot_v2/runs/usc18_chapter41_v1",
}


def _headings() -> dict[str, str]:
    """Citation -> section heading, from the pilots' own segmentation output.

    A section heading is part of the enacted text and carries the name a reader
    actually uses ("Blackmail", "Kickbacks from public works employees") — the
    operative paragraphs never repeat it. Without it the graph stores the rule
    but not the word anyone would search for.
    """
    out: dict[str, str] = {}
    for run in RUNS.values():
        for unit in json.loads((run / "source_units.json").read_text()):
            out[unit["citation"]] = unit.get("section_heading", "")
    return out
SOURCE_PAGE = {"v1": "35 U.S.C. ch.10", "v2": "18 U.S.C. ch.41"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip())


def _predicate_slug(predicate: str) -> str:
    """Canonical snake_case predicate label; keeps the surface meaning."""
    text = _norm(predicate).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "relates_to"


def build() -> list[dict]:
    items: list[dict] = []
    entities: dict[str, dict] = {}
    headings = _headings()

    def add_entity(label: str, kind: str) -> None:
        label = _norm(label)
        if not label or label in entities:
            return
        entities[label] = {
            "overlay_type": "overlay_entity",
            "entity_id": f"legal:{re.sub(r'[^A-Za-z0-9]+', '_', label)[:60]}",
            "label": label,
            "aliases": [],
            "entity_type": kind,
            "source_page": "legal_pilot",
            "source_candidate_type": "legal_node",
            "trust": "overlay_candidate",
            "risk": "low",
        }

    # ---- 1. Plain accepted relations from both pilots ---------------------
    # A statutory definition is mapped onto the runtime's *native* definition
    # type rather than a generic relation: "the term X means Y" is semantically
    # the same shape the graph already models as ``overlay_definition``. This is
    # integration, not benchmark tuning — the question analyzer is left
    # untouched, and every other relation type stays an ``overlay_relation``.
    _DEFINITION_PREDICATES = {"means", "mean", "includes", "definition", "is_defined_as"}

    for tag, run in RUNS.items():
        decisions = json.loads((run / "manual_review_decisions.json").read_text())
        candidates = {c["candidate_id"]: c for c in
                      json.loads((run / "manual_review_candidates.json").read_text())}
        for d in decisions:
            if d["verdict"] != "ACCEPT":
                continue
            cand = candidates[d["candidate_id"]]
            subject, obj = _norm(d["subject"]), _norm(d["object"])
            if not subject or not obj:
                continue
            subject = subject.strip('"')
            slug = _predicate_slug(d["predicate"])
            is_definition = (
                slug in _DEFINITION_PREDICATES
                or d.get("relation_type") == "definition"
            )
            add_entity(subject, "legal_concept")
            common = {
                "section_heading": headings.get(d["citation"], ""),
                "source_page": SOURCE_PAGE[tag],
                "source_url": cand.get("source_url", ""),
                "evidence_text": _norm(cand.get("evidence_span") or cand.get("source_text", "")),
                "stated_in": d["citation"],
                "trust": "overlay_candidate",
                "risk": "medium",
                "stability": "stable",
            }
            if is_definition:
                items.append({
                    "overlay_type": "overlay_definition",
                    "subject": subject,
                    "definition": obj,
                    "predicate": "is_a",
                    **common,
                })
            else:
                add_entity(obj, "legal_concept")
                items.append({
                    "overlay_type": "overlay_relation",
                    "subject": subject,
                    "predicate": slug,
                    "object": obj,
                    **common,
                })

    # ---- 2. The 16 conditional rules, in the conditional-edge schema ------
    import build_conditional_edges as B  # noqa: E402

    for _cid, edge in B.EDGES:
        subject, obj = _norm(edge.subject), _norm(edge.object)
        add_entity(subject, "legal_class")
        add_entity(obj, "legal_concept")
        item = {
            "overlay_type": "overlay_relation",
            "subject": subject,
            "predicate": edge.predicate,
            "object": obj,
            "source_page": SOURCE_PAGE["v1"],
            "source_url": "",
            "evidence_text": _norm(edge.evidence_sentence)[:600],
            "stated_in": edge.stated_in,
            "section_heading": headings.get(edge.stated_in, ""),
            "trust": "overlay_candidate",
            "risk": "medium",
            "stability": "stable",
        }
        if edge.conditions:
            item["conditions"] = [
                {"text": c.text, "evidence_span": c.evidence_span, "kind": c.kind}
                for c in edge.conditions
            ]
        if edge.exceptions:
            item["exceptions"] = [
                {"text": c.text, "evidence_span": c.evidence_span, "kind": c.kind}
                for c in edge.exceptions
            ]
        if edge.polarity != "affirm":
            item["polarity"] = edge.polarity
        item["subject_kind"] = "class_subject"
        items.append(item)

    return list(entities.values()) + items


def main() -> int:
    items = build()
    out = HERE / "legal_overlay.json"
    out.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rel = [i for i in items if i["overlay_type"] == "overlay_relation"]
    cond = [i for i in rel if i.get("conditions") or i.get("exceptions")]
    print(f"entities={len(items) - len(rel)}  relations={len(rel)}  conditional={len(cond)}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
