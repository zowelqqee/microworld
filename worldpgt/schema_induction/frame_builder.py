"""Build argument frames from raw claims.

A frame lifts a surface claim into GENERIC semantic roles (subject, requirement,
permission, prohibition, agent, destination, cause, time, condition, ...). The
trigger -> primary-role mapping below is a small, generic *linguistic* lexicon
(what kind of complement the verb takes), NOT a domain ontology: it contains no
visa/animal/company specific knowledge and works the same for any corpus.
"""

from __future__ import annotations

import hashlib

from worldpgt.schema_induction.types import (
    ArgumentFrame,
    EntityMention,
    RawClaim,
)

# Generic mapping: normalized surface trigger -> role the object fills.
# This is verb-complement linguistics, not domain knowledge.
_TRIGGER_OBJECT_ROLE: dict[str, str] = {
    "requires": "requirement",
    "require": "requirement",
    "required": "requirement",
    "needs": "requirement",
    "need": "requirement",
    "must show": "requirement",
    "must provide": "requirement",
    "must have": "requirement",
    "allows": "permission",
    "allow": "permission",
    "permits": "permission",
    "permit": "permission",
    "prohibits": "prohibition",
    "prohibit": "prohibition",
    "forbids": "prohibition",
    "forbid": "prohibition",
    "bans": "prohibition",
    "was founded by": "agent",
    "were founded by": "agent",
    "was created by": "agent",
    "is operated by": "agent",
    "are operated by": "agent",
    "was operated by": "agent",
    "is run by": "agent",
    "depends on": "cause",
    "depend on": "cause",
    "feeds on": "patient",
    "feed on": "patient",
    "is valid for": "attribute",
    "are valid for": "attribute",
    "is": "attribute",
    "are": "attribute",
    "was": "attribute",
    "were": "attribute",
}

# Directional triggers route their object into a destination role.
_DIRECTIONAL = frozenset({
    "migrates toward", "migrate toward", "migrates towards", "migrate towards",
    "moves toward", "move toward", "moves towards", "move towards",
    "migrates to", "migrate to", "moves to", "move to",
    "travels to", "travel to", "migrates", "migrate", "moves", "move",
})


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _entity_type(surface: str, index: dict[str, EntityMention]) -> str:
    if not surface:
        return "unknown"
    ent = index.get(surface.strip().lower())
    if ent and ent.type_hints:
        return ent.type_hints[0]
    return "unknown"


def build_frame(
    claim: RawClaim, entity_index: dict[str, EntityMention] | None = None
) -> ArgumentFrame:
    """Build a single argument frame from a raw claim."""

    index = entity_index or {}
    roles: dict[str, str] = {"subject": claim.subject}
    role_types: dict[str, str] = {
        "subject": _entity_type(claim.subject, index),
    }

    surface = claim.relation_surface
    if claim.object:
        if surface in _DIRECTIONAL:
            role = "destination"
        else:
            role = _TRIGGER_OBJECT_ROLE.get(surface, "object")
        roles[role] = claim.object
        role_types[role] = _entity_type(claim.object, index)

    # Modifiers map straight onto generic roles.
    for mkey, mval in claim.modifiers.items():
        if mkey in ("cause", "condition", "time", "destination", "purpose"):
            if mkey not in roles:
                roles[mkey] = mval
                role_types[mkey] = _entity_type(mval, index)

    domain_hint = role_types.get("subject")
    if domain_hint == "unknown":
        domain_hint = None

    frame_id = _stable_id("frame", claim.claim_id)
    return ArgumentFrame(
        frame_id=frame_id,
        claim_ids=(claim.claim_id,),
        trigger=surface,
        roles=roles,
        role_types=role_types,
        domain_hint=domain_hint,
        confidence=claim.confidence,
    )


def build_frames(
    claims: list[RawClaim],
    entities: list[EntityMention] | None = None,
) -> list[ArgumentFrame]:
    """Build frames for every claim."""

    index: dict[str, EntityMention] = {}
    for ent in entities or []:
        index[ent.normalized] = ent
    return [build_frame(c, index) for c in claims]
