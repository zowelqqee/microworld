"""Group argument frames into relation families.

A relation family is a cluster of frames that share relation structure:
- a canonical (alias-normalized) trigger,
- the same set of generic roles,
- compatible role-type profiles.

The alias groups below are GENERIC verb synonyms (require/need/must show ->
"requires"), not domain predicate ids. Triggers only merge when their role
sets are compatible, so e.g. "migrate" and "move" only join when they share the
same roles. Everything is induced from observed frames; no family ids are
hardcoded.
"""

from __future__ import annotations

import hashlib

from worldpgt.schema_induction.types import ArgumentFrame, RelationFamily

# Generic surface synonyms -> canonical trigger label. These are language-level
# verb alternations, deliberately not domain specific.
_ALIAS_GROUPS: dict[str, str] = {
    # requirement family
    "requires": "requires", "require": "requires", "required": "requires",
    "needs": "requires", "need": "requires",
    "must show": "requires", "must provide": "requires", "must have": "requires",
    # permission family
    "allows": "allows", "allow": "allows", "permits": "allows", "permit": "allows",
    # prohibition family
    "prohibits": "prohibits", "prohibit": "prohibits",
    "forbids": "prohibits", "forbid": "prohibits", "bans": "prohibits",
    # agentive family
    "was founded by": "founded by", "were founded by": "founded by",
    "was created by": "founded by",
    "is operated by": "operated by", "are operated by": "operated by",
    "was operated by": "operated by", "is run by": "operated by",
    # movement family
    "migrates toward": "migrate", "migrate toward": "migrate",
    "migrates towards": "migrate", "migrate towards": "migrate",
    "migrates to": "migrate", "migrate to": "migrate",
    "migrates": "migrate", "migrate": "migrate",
    "moves toward": "move", "move toward": "move",
    "moves towards": "move", "move towards": "move",
    "moves to": "move", "move to": "move",
    "moves": "move", "move": "move",
    "travels to": "move", "travel to": "move",
    # dependency
    "depends on": "depends on", "depend on": "depends on",
    "feeds on": "feeds on", "feed on": "feeds on",
    # copula
    "is": "is", "are": "is", "was": "is", "were": "is",
    "is valid for": "valid for", "are valid for": "valid for",
}

# Movement canonical labels that share roles may be merged together.
_MERGEABLE_CANON = {"migrate", "move"}


def _canonical(trigger: str) -> str:
    return _ALIAS_GROUPS.get(trigger, trigger)


def _stable_id(prefix: str, key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _role_signature(frame: ArgumentFrame) -> tuple[str, ...]:
    return tuple(sorted(r for r in frame.roles if r != "subject"))


def build_relation_families(
    frames: list[ArgumentFrame],
    claim_doc_map: dict[str, str] | None = None,
) -> list[RelationFamily]:
    """Cluster frames into relation families.

    ``claim_doc_map`` maps claim_id -> source doc_id; when provided it yields an
    accurate ``source_doc_count`` per family.
    """

    claim_doc_map = claim_doc_map or {}

    # Group key: (canonical_trigger, role_signature). Mergeable movement
    # canonicals collapse to a shared key when role signatures match.
    groups: dict[tuple[str, tuple[str, ...]], list[ArgumentFrame]] = {}
    order: list[tuple[str, tuple[str, ...]]] = []

    for frame in frames:
        canon = _canonical(frame.trigger)
        sig = _role_signature(frame)
        merge_canon = "move/migrate" if canon in _MERGEABLE_CANON else canon
        key = (merge_canon, sig)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(frame)

    families: list[RelationFamily] = []
    for key in order:
        canon, sig = key
        members = groups[key]

        surface_forms = sorted({f.trigger for f in members})
        roles: list[str] = ["subject"]
        for r in sig:
            if r not in roles:
                roles.append(r)

        # Role -> observed type profile.
        profile: dict[str, list[str]] = {}
        example_claim_ids: list[str] = []
        frame_ids: list[str] = []
        source_docs: set[str] = set()
        for f in members:
            frame_ids.append(f.frame_id)
            example_claim_ids.extend(f.claim_ids)
            for role, rtype in f.role_types.items():
                profile.setdefault(role, [])
                if rtype not in profile[role]:
                    profile[role].append(rtype)
            for cid in f.claim_ids:
                doc = claim_doc_map.get(cid)
                if doc:
                    source_docs.add(doc)

        role_type_profile = {k: tuple(v) for k, v in profile.items()}
        evidence_count = len(members)
        source_doc_count = len(source_docs) if source_docs else 1

        confidence = round(
            sum(f.confidence for f in members) / max(1, len(members)), 4
        )
        canonical_label = "move/migrate" if canon == "move/migrate" else canon
        family_id = _stable_id("fam", f"{canonical_label}|{'+'.join(roles)}")

        families.append(
            RelationFamily(
                family_id=family_id,
                canonical_label=canonical_label,
                surface_forms=tuple(surface_forms),
                roles=tuple(roles),
                role_type_profile=role_type_profile,
                example_claim_ids=tuple(dict.fromkeys(example_claim_ids)),
                evidence_count=evidence_count,
                source_doc_count=source_doc_count,
                promotion_status="generated",
                confidence=confidence,
                frame_ids=tuple(frame_ids),
            )
        )
    return families
