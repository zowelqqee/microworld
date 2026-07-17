"""Conservative source-specific relation parsers for quarantined open-web lanes.

These parsers do not promote anything.  They re-derive a candidate from the
source family's own stored record and mark its provenance so the existing
open-web source gate may pass it *on to* the unchanged precision firewalls.
The caller supplies an allowlisted candidate set; no parser broadens a pump
frontier merely because it can read a record.
"""

from __future__ import annotations

import re
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

from worldpgt.relation_extraction_v2.sentence_splitter import split_sentences


_SUPPORTED_LANES = frozenset({"arxiv", "crossref", "openalex"})
_PREDICATE_CUES: dict[str, tuple[str, ...]] = {
    "uses": ("uses", "use"),
    "enables": ("enables", "enable"),
    "supports": ("supports", "support"),
    "provides": ("provides", "provide"),
    "works_by": ("works by", "works through"),
    "located_in": ("located in",),
}
_NONREFERENTIAL_SUBJECT_LEADS = frozenset({
    "a", "an", "the", "such", "this", "that", "these", "those", "one", "our", "we",
})
_SUBJECT_DISCOURSE_TOKENS = frozenset({"also", "however", "therefore", "moreover"})
_OBJECT_CLAUSE_MARKERS = frozenset({"but", "which", "that", "including"})
_GENERIC_SUBJECTS = frozenset({
    "abstract", "analysis", "approach", "background", "conclusion", "evaluation",
    "implementation", "introduction", "method", "methods", "research", "result",
    "results", "study", "system",
})


def _norm(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _canonical_url(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parts = urlsplit(raw)
    # arXiv records were historically stored under both HTTP and HTTPS.  The
    # source identity is host/path/query, not that transport spelling.
    return urlunsplit(("", parts.netloc.casefold(), parts.path.rstrip("/"), parts.query, ""))


def _phrase_in(text: str, phrase: str) -> bool:
    return bool(re.search(r"(?<!\\w)" + re.escape(_norm(phrase)) + r"(?!\\w)", _norm(text)))


def _atomic_graph_endpoints(candidate: dict[str, Any]) -> bool:
    """Reject discourse fragments and unbounded clause captures before gating."""

    subject_words = _norm(candidate.get("subject")).split()
    object_words = _norm(candidate.get("object")).split()
    if not subject_words or not object_words:
        return False
    if subject_words[0] in _NONREFERENTIAL_SUBJECT_LEADS:
        return False
    if len(subject_words) == 1 and subject_words[0] in _GENERIC_SUBJECTS:
        return False
    if _SUBJECT_DISCOURSE_TOKENS.intersection(subject_words):
        return False
    # The general precision firewall also screens clause objects.  Performing
    # the same bounded-span check here prevents a source-specific parser from
    # turning an explicit sentence into an overlong pseudo-object first.
    if len(object_words) > 6 or _OBJECT_CLAUSE_MARKERS.intersection(object_words):
        return False
    return True


def _supported_sentence(candidate: dict[str, Any], source_text: str) -> str | None:
    subject = str(candidate.get("subject") or "").strip()
    obj = str(candidate.get("object") or "").strip()
    cues = _PREDICATE_CUES.get(str(candidate.get("predicate") or ""), ())
    if not subject or not obj or not cues:
        return None
    for sentence in split_sentences(source_text):
        if not _phrase_in(sentence, subject) or not _phrase_in(sentence, obj):
            continue
        subject_at = _norm(sentence).find(_norm(subject))
        object_at = _norm(sentence).find(_norm(obj))
        if subject_at < 0 or object_at <= subject_at:
            continue
        between = _norm(sentence)[subject_at:object_at]
        if any(_phrase_in(between, cue) for cue in cues):
            return " ".join(sentence.split())
    return None


def extract_explicit_relations(
    lane: str,
    *,
    source_records: Iterable[dict[str, Any]],
    allowlisted_candidates: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Re-derive allowlisted relations from stored records of one source lane.

    Each returned item has an explicit sentence proving ``subject → predicate
    → object`` and is still only a proposal candidate.  The companion rejects
    preserve failures for the lane report; neither list writes accepted memory.
    """

    if lane not in _SUPPORTED_LANES:
        raise ValueError(f"unsupported source-specific lane: {lane}")
    records = {
        _canonical_url(record.get("source_url")): record
        for record in source_records
        if str(record.get("source_kind") or "") == lane and _canonical_url(record.get("source_url"))
    }
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for candidate in allowlisted_candidates:
        if str(candidate.get("source_kind") or "") != lane:
            continue
        key = (
            _norm(candidate.get("subject")),
            _norm(candidate.get("predicate")),
            _norm(candidate.get("object")),
            _canonical_url(candidate.get("source_url")),
        )
        if not all(key) or key in seen:
            continue
        seen.add(key)
        if not _atomic_graph_endpoints(candidate):
            rejected.append({"item": candidate, "reason": "source_specific_non_atomic_endpoint"})
            continue
        record = records.get(key[3])
        if record is None:
            rejected.append({"item": candidate, "reason": "source_specific_record_unavailable"})
            continue
        sentence = _supported_sentence(candidate, str(record.get("text") or ""))
        if sentence is None:
            rejected.append({"item": candidate, "reason": "source_specific_explicit_relation_not_found"})
            continue
        item = dict(candidate)
        pattern_id = f"{lane}_{candidate['predicate']}_explicit_relation_v1"
        item.update({
            "source_page": str(record.get("title") or candidate.get("source_page") or ""),
            "source_url": str(record.get("source_url") or candidate.get("source_url") or ""),
            "source_kind": lane,
            "evidence_text": sentence,
            "evidence_span": sentence,
            "candidate_source": f"{lane}_source_specific_relation_v1",
            "extraction_pattern": pattern_id,
            "v2_pattern_id": pattern_id,
            "confidence_label": "source_specific_explicit",
            "open_web_extraction": f"{lane}_explicit_relation_v1",
            "pump_source_kind": "open_web_proposal",
            "trust": "proposal_open_web",
            "safe_for_general_runtime": False,
            "requires_review": True,
        })
        accepted.append(item)
    return accepted, rejected
