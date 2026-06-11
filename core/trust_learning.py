"""
Audit-driven trust learning without neural backprop.

Human labels are mapped to numeric scores and averaged per interpretable
bucket: relation type, optional rule id, drift type, and evidence node.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

LABEL_SCORES: dict[str, float] = {
    "correct": 1.0,
    "plausible": 0.7,
    "unclear": 0.4,
    "wrong": 0.0,
}

_LABEL_ALIASES: dict[str, str] = {
    "plusable": "plausible",
    "plausable": "plausible",
    "posible": "plausible",
    "true": "correct",
    "yes": "correct",
    "false": "wrong",
    "no": "wrong",
}

_DRIFT_RE = re.compile(r"\bdrift=([a-zA-Z0-9_]+)")


@dataclass
class TrustProfile:
    relation_trust: dict[str, float] = field(default_factory=dict)
    rule_trust: dict[str, float] = field(default_factory=dict)
    drift_trust: dict[str, float] = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    evidence_trust: dict[str, float] = field(default_factory=dict)

    def to_json(self, path: str) -> None:
        """Write this trust profile to *path* as stable pretty JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)
            f.write("\n")

    @classmethod
    def from_json(cls, path: str) -> "TrustProfile":
        """Load a trust profile written by :meth:`to_json`."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            relation_trust=dict(data.get("relation_trust", {})),
            rule_trust=dict(data.get("rule_trust", {})),
            drift_trust=dict(data.get("drift_trust", {})),
            counts=dict(data.get("counts", {})),
            evidence_trust=dict(data.get("evidence_trust", {})),
        )


def learn_trust_from_audits(paths) -> TrustProfile:
    """Learn trust averages from one or more audit CSV files."""
    sums: dict[str, dict[str, float]] = {
        "relation_trust": {},
        "rule_trust": {},
        "drift_trust": {},
        "evidence_trust": {},
    }
    counts: dict[str, dict[str, int] | int] = {
        "relation_trust": {},
        "rule_trust": {},
        "drift_trust": {},
        "evidence_trust": {},
        "rows": 0,
        "used_rows": 0,
        "skipped_empty_label": 0,
        "missing_files": 0,
    }

    for path in paths:
        p = Path(path)
        if not p.exists():
            counts["missing_files"] = int(counts["missing_files"]) + 1
            continue
        for row in _read_rows(str(p)):
            counts["rows"] = int(counts["rows"]) + 1
            score = label_score(row.get("manual_label", ""))
            if score is None:
                counts["skipped_empty_label"] = int(counts["skipped_empty_label"]) + 1
                continue
            counts["used_rows"] = int(counts["used_rows"]) + 1

            relation = _relation_from_row(row)
            if relation:
                _add(sums, counts, "relation_trust", relation, score)

            rule = row.get("rule", "").strip()
            if rule:
                _add(sums, counts, "rule_trust", rule, score)

            drift_type = parse_drift_type(row.get("reason", ""))
            if drift_type:
                _add(sums, counts, "drift_trust", drift_type, score)

            for evidence in _evidence_nodes(row.get("evidence", "")):
                _add(sums, counts, "evidence_trust", evidence, score)

    return TrustProfile(
        relation_trust=_averages(sums["relation_trust"], counts["relation_trust"]),
        rule_trust=_averages(sums["rule_trust"], counts["rule_trust"]),
        drift_trust=_averages(sums["drift_trust"], counts["drift_trust"]),
        evidence_trust=_averages(sums["evidence_trust"], counts["evidence_trust"]),
        counts=counts,
    )


def label_score(raw_label: str) -> float | None:
    """Map an audit label to a score; empty labels are skipped."""
    label = raw_label.strip().lower()
    if not label:
        return None
    label = _LABEL_ALIASES.get(label, label)
    return LABEL_SCORES.get(label, LABEL_SCORES["unclear"])


def parse_drift_type(reason: str) -> str | None:
    """Extract drift type from a reason string, if present."""
    match = _DRIFT_RE.search(reason or "")
    if not match:
        return None
    drift_type = match.group(1)
    return None if drift_type == "none" else drift_type


def _read_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        sample = f.read(4096)
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = ","
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def _relation_from_row(row: dict) -> str:
    return (row.get("relation_type") or row.get("proposed_relation") or "").strip()


def _evidence_nodes(raw: str) -> list[str]:
    nodes: list[str] = []
    for chunk in (raw or "").replace(",", "|").split("|"):
        node = chunk.strip()
        if node:
            nodes.append(node)
    return nodes


def _add(
    sums: dict[str, dict[str, float]],
    counts: dict,
    table: str,
    key: str,
    score: float,
) -> None:
    table_counts = counts[table]
    sums[table][key] = sums[table].get(key, 0.0) + score
    table_counts[key] = table_counts.get(key, 0) + 1


def _averages(sums: dict[str, float], counts: dict[str, int]) -> dict[str, float]:
    return {
        key: round(total / counts[key], 6)
        for key, total in sorted(sums.items())
        if counts.get(key, 0) > 0
    }
