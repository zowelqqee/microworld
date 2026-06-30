"""JSON / JSONL IO for schema induction artifacts.

All artifacts are small, inspectable, and kept separate from accepted/promoted
memory. Writing is deterministic (stable key order).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from worldpgt.schema_induction.types import (
    ArgumentFrame,
    EntityMention,
    LocalType,
    PromotionDecision,
    RawClaim,
    RelationFamily,
    SchemaInductionResult,
)

RAW_CLAIMS = "raw_claims.jsonl"
ARGUMENT_FRAMES = "argument_frames.jsonl"
ENTITIES = "entities.generated.json"
RELATION_FAMILIES_GENERATED = "relation_families.generated.json"
RELATION_FAMILIES_PROMOTED = "relation_families.promoted.json"
LOCAL_TYPES_GENERATED = "local_types.generated.json"
DECISIONS = "promotion_decisions.json"
SUMMARY = "schema_induction_summary.json"


def _to_dict(obj) -> dict:
    return dataclasses.asdict(obj)


def _write_jsonl(path: Path, items: list) -> None:
    lines = [json.dumps(_to_dict(i), ensure_ascii=False, sort_keys=True) for i in items]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_json(path: Path, items: list) -> None:
    data = [_to_dict(i) for i in items]
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_result(result: SchemaInductionResult, output_dir: str | Path) -> dict[str, str]:
    """Write all artifacts for a result; return a map of artifact -> path."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    promoted = [f for f in result.families if f.promotion_status == "promoted"]

    paths = {
        RAW_CLAIMS: out / RAW_CLAIMS,
        ARGUMENT_FRAMES: out / ARGUMENT_FRAMES,
        ENTITIES: out / ENTITIES,
        RELATION_FAMILIES_GENERATED: out / RELATION_FAMILIES_GENERATED,
        RELATION_FAMILIES_PROMOTED: out / RELATION_FAMILIES_PROMOTED,
        LOCAL_TYPES_GENERATED: out / LOCAL_TYPES_GENERATED,
        DECISIONS: out / DECISIONS,
        SUMMARY: out / SUMMARY,
    }

    _write_jsonl(paths[RAW_CLAIMS], list(result.claims))
    _write_jsonl(paths[ARGUMENT_FRAMES], list(result.frames))
    _write_json(paths[ENTITIES], list(result.entities))
    _write_json(paths[RELATION_FAMILIES_GENERATED], list(result.families))
    _write_json(paths[RELATION_FAMILIES_PROMOTED], promoted)
    _write_json(paths[LOCAL_TYPES_GENERATED], list(result.local_types))
    _write_json(paths[DECISIONS], list(result.decisions))
    paths[SUMMARY].write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {k: str(v) for k, v in paths.items()}


def read_docs_jsonl(path: str | Path) -> list[dict]:
    """Read input docs from a JSONL file ({doc_id,title,url,text} per line)."""

    docs: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        docs.append(json.loads(line))
    return docs


def load_families(path: str | Path) -> list[RelationFamily]:
    """Load relation families from a generated/promoted JSON artifact."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[RelationFamily] = []
    for item in raw:
        item = dict(item)
        item["surface_forms"] = tuple(item.get("surface_forms", ()))
        item["roles"] = tuple(item.get("roles", ()))
        item["example_claim_ids"] = tuple(item.get("example_claim_ids", ()))
        item["frame_ids"] = tuple(item.get("frame_ids", ()))
        item["role_type_profile"] = {
            k: tuple(v) for k, v in (item.get("role_type_profile") or {}).items()
        }
        out.append(RelationFamily(**item))
    return out


def load_frames(path: str | Path) -> list[ArgumentFrame]:
    """Load argument frames from the JSONL artifact."""

    out: list[ArgumentFrame] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        item["claim_ids"] = tuple(item.get("claim_ids", ()))
        out.append(ArgumentFrame(**item))
    return out


def load_claims(path: str | Path) -> list[RawClaim]:
    """Load raw claims from the JSONL artifact."""

    out: list[RawClaim] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(RawClaim(**json.loads(line)))
    return out
