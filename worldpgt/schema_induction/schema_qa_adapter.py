"""Read-only QA adapter over generated / promoted relation families.

This is a NEW path that does not touch the curated ``entity_qa`` stack. It
answers:
- relation lookups ("What does X require?", "Куда мигрируют wildebeest?")
- open synthesis ("Tell me about X", "Расскажи про X")

entirely from induced relation families and their source-traced claims. Outcomes
are always one of: answer / audit / no. By default it prefers promoted families;
generated-only families answer only when ``allow_generated=True`` (the moral
equivalent of an explicit ``--allow-generated-schema`` flag).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from worldpgt.schema_induction.query_compiler import QueryPlan, compile_query
from worldpgt.schema_induction.types import (
    ArgumentFrame,
    RawClaim,
    RelationFamily,
    SchemaInductionResult,
)

# Family label + role -> sentence template for rendering.
_VERB_TEMPLATES: dict[tuple[str, str], str] = {
    ("requires", "requirement"): "{entity} requires {values}.",
    ("prohibits", "prohibition"): "{entity} prohibits {values}.",
    ("allows", "permission"): "{entity} allows {values}.",
    ("founded by", "agent"): "{entity} was founded by {values}.",
    ("operated by", "agent"): "{entity} is operated by {values}.",
    ("move/migrate", "destination"): "{entity} migrates toward {values}.",
    ("move/migrate", "cause"): "{entity} moves in search of {values}.",
    ("depends on", "cause"): "{entity} depends on {values}.",
    ("is", "attribute"): "{entity} is {values}.",
    ("valid for", "attribute"): "{entity} is valid for {values}.",
}

# Readable verb phrases for open-synthesis bullet lines.
_SYNTHESIS_VERBS: dict[str, str] = {
    "requires": "requires",
    "prohibits": "prohibits",
    "allows": "allows",
    "founded by": "was founded by",
    "operated by": "is operated by",
    "move/migrate": "moves in connection with",
    "depends on": "depends on",
    "is": "is described as",
    "valid for": "is valid for",
    "shift": "shifts",
}


@dataclass(frozen=True)
class SchemaAnswer:
    decision: str               # answer | audit | no
    text: str
    tier: str                   # VERIFIED | GENERATED | UNKNOWN
    sources: tuple[str, ...]
    evidence: tuple[dict, ...]
    plan: QueryPlan | None
    confidence: float = 0.0
    reason: str | None = None


def _join(values: list[str]) -> str:
    values = [v for v in values if v]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f" and {values[-1]}"


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _entity_match(entity: str, subject: str) -> bool:
    a, b = _norm(entity), _norm(subject)
    if not a or not b:
        return False
    return a == b or a in b or b in a


class SchemaQAAdapter:
    """Answer questions over induced relation families."""

    def __init__(
        self,
        families: list[RelationFamily],
        frames: list[ArgumentFrame],
        claims: list[RawClaim],
        allow_generated: bool = False,
    ) -> None:
        self.families = list(families)
        self.frames = list(frames)
        self.claims = list(claims)
        self.allow_generated = allow_generated

        self._frame_by_id = {f.frame_id: f for f in frames}
        self._claim_by_id = {c.claim_id: c for c in claims}
        self._frames_by_family: dict[str, list[ArgumentFrame]] = {}
        for fam in families:
            self._frames_by_family[fam.family_id] = [
                self._frame_by_id[fid]
                for fid in fam.frame_ids
                if fid in self._frame_by_id
            ]
        # Entity surfaces for the compiler: subjects (and objects) seen.
        surfaces: list[str] = []
        seen: set[str] = set()
        for c in claims:
            for s in (c.subject, c.object):
                if s and _norm(s) not in seen:
                    seen.add(_norm(s))
                    surfaces.append(s)
        # Longest first improves longest-match entity detection.
        self._entity_surfaces = sorted(surfaces, key=lambda s: -len(s))

    # -- construction helpers -------------------------------------------------

    @classmethod
    def from_result(
        cls, result: SchemaInductionResult, allow_generated: bool = False
    ) -> "SchemaQAAdapter":
        return cls(
            families=list(result.families),
            frames=list(result.frames),
            claims=list(result.claims),
            allow_generated=allow_generated,
        )

    # -- public API -----------------------------------------------------------

    def answer(self, question: str) -> SchemaAnswer:
        plan = compile_query(question, self.families, self._entity_surfaces)
        if plan.operation == "open_synthesis":
            return self._answer_synthesis(plan)
        if plan.operation == "find_role":
            return self._answer_find_role(plan)
        return self._audit(plan, plan.reason or "uncompilable_question")

    # -- find_role ------------------------------------------------------------

    def _candidate_families(self, label: str) -> tuple[list[RelationFamily], str]:
        """Return (families, tier) honoring promoted-first policy."""

        labelled = [f for f in self.families if f.canonical_label == label]
        promoted = [f for f in labelled if f.promotion_status == "promoted"]
        if promoted:
            return promoted, "VERIFIED"
        if labelled and self.allow_generated:
            return labelled, "GENERATED"
        return [], "UNKNOWN"

    def _answer_find_role(self, plan: QueryPlan) -> SchemaAnswer:
        families, tier = self._candidate_families(plan.family_label or "")
        if not families:
            return self._audit(
                plan,
                "relation_family_not_promoted"
                if any(f.canonical_label == plan.family_label for f in self.families)
                else "relation_family_not_found",
            )

        values: list[str] = []
        sources: list[str] = []
        evidence: list[dict] = []
        doc_ids: set[str] = set()
        for fam in families:
            for frame in self._frames_by_family.get(fam.family_id, []):
                # match_role is normally "subject" ("What does ENTITY require?")
                # but flips for reverse-direction questions ("Who won ENTITY?")
                # -- see QueryPlan.match_role / compile_query's direction check.
                matched = frame.roles.get(plan.match_role, "")
                if not _entity_match(plan.entity or "", matched):
                    continue
                val = frame.roles.get(plan.target_role or "")
                if not val:
                    continue
                if val not in values:
                    values.append(val)
                for cid in frame.claim_ids:
                    claim = self._claim_by_id.get(cid)
                    if claim:
                        if claim.source_sentence_id not in sources:
                            sources.append(claim.source_sentence_id)
                        doc_ids.add(claim.source_doc_id)
                        evidence.append({
                            "claim_id": claim.claim_id,
                            "sentence_id": claim.source_sentence_id,
                            "sentence": claim.sentence,
                        })

        if not values:
            return self._audit(plan, "no_supporting_claim")

        text = self._render_relation(
            entity=plan.entity or "",
            family_label=plan.family_label or "",
            target_role=plan.target_role or "",
            values=values,
            sources=sources,
            evidence_count=len(evidence),
            source_doc_count=len(doc_ids),
            tier=tier,
        )
        return SchemaAnswer(
            decision="answer",
            text=text,
            tier=tier,
            sources=tuple(sources),
            evidence=tuple(evidence),
            plan=plan,
            confidence=0.9 if tier == "VERIFIED" else 0.7,
        )

    def _render_relation(
        self,
        entity: str,
        family_label: str,
        target_role: str,
        values: list[str],
        sources: list[str],
        evidence_count: int,
        source_doc_count: int,
        tier: str,
    ) -> str:
        # Reverse-direction answers ("Who won ENTITY?" -> subject values) read
        # backwards with the entity-first fallback template ("ENTITY won
        # VALUES" when VALUES are actually the ones who won ENTITY) -- put
        # the retrieved values first instead when the answer IS the subject.
        default_template = (
            "{values} " + family_label + " {entity}."
            if target_role == "subject"
            else "{entity} " + family_label + " {values}."
        )
        template = _VERB_TEMPLATES.get((family_label, target_role), default_template)
        lead = template.format(entity=entity, values=_join(values))
        src_block = "\n".join(f"- {s}" for s in sources)
        tier_note = "" if tier == "VERIFIED" else " (generated schema, not yet promoted)"
        return (
            f"{lead}\n\n"
            f"Based on {evidence_count} claim(s) from {source_doc_count} "
            f"source(s){tier_note}.\n\n"
            f"Sources:\n{src_block}"
        )

    # -- open synthesis -------------------------------------------------------

    def _answer_synthesis(self, plan: QueryPlan) -> SchemaAnswer:
        entity = plan.entity
        if not entity:
            return self._audit(plan, "no_entity_detected")

        # Gather frames where the entity appears in ANY role.
        grouped: dict[str, list[tuple[ArgumentFrame, RelationFamily]]] = {}
        fam_by_frame: dict[str, RelationFamily] = {}
        for fam in self.families:
            for frame in self._frames_by_family.get(fam.family_id, []):
                fam_by_frame[frame.frame_id] = fam

        sources: list[str] = []
        evidence: list[dict] = []
        doc_ids: set[str] = set()
        promoted_only = not self.allow_generated
        lines: list[str] = []
        type_hint = None

        for frame in self.frames:
            if not any(_entity_match(entity, v) for v in frame.roles.values()):
                continue
            fam = fam_by_frame.get(frame.frame_id)
            if fam is None:
                continue
            if promoted_only and fam.promotion_status != "promoted":
                continue
            grouped.setdefault(fam.canonical_label, []).append((frame, fam))
            if type_hint is None and frame.domain_hint:
                type_hint = frame.domain_hint

        if not grouped:
            return self._audit(plan, "no_promoted_facts" if promoted_only else "no_facts")

        # Roles that are too peripheral to headline a synthesis sentence.
        skip_roles = {"subject", "time", "condition"}
        for label, items in grouped.items():
            values: list[str] = []
            for frame, fam in items:
                # pick the most informative non-subject role value
                for role in fam.roles:
                    if role in skip_roles:
                        continue
                    val = frame.roles.get(role)
                    if val and val not in values:
                        values.append(val)
                for cid in frame.claim_ids:
                    claim = self._claim_by_id.get(cid)
                    if claim:
                        if claim.source_sentence_id not in sources:
                            sources.append(claim.source_sentence_id)
                        doc_ids.add(claim.source_doc_id)
                        evidence.append({
                            "claim_id": claim.claim_id,
                            "sentence_id": claim.source_sentence_id,
                            "sentence": claim.sentence,
                        })
            if values:
                verb = _SYNTHESIS_VERBS.get(label, label)
                lines.append(f"It {verb} {_join(values)}.")

        # Only show a type hint when it is genuinely a *type*, not the entity
        # name echoed back (e.g. suppress "as a giraffes" for "giraffes").
        show_hint = type_hint and type_hint != _norm(entity)
        intro = (
            f"{entity} appears in the loaded corpus"
            + (f" as a {type_hint}." if show_hint else ".")
        )
        body = "\n".join(lines)
        tier = "VERIFIED" if promoted_only else "GENERATED"
        src_block = "\n".join(f"- {s}" for s in sources)
        text = (
            f"{intro}\n{body}\n\n"
            f"This is based on {len(evidence)} claim(s) from {len(doc_ids)} "
            f"source(s).\n\nSources:\n{src_block}"
        )
        return SchemaAnswer(
            decision="answer",
            text=text,
            tier=tier,
            sources=tuple(sources),
            evidence=tuple(evidence),
            plan=plan,
            confidence=0.85 if tier == "VERIFIED" else 0.7,
        )

    # -- audit ----------------------------------------------------------------

    def _audit(self, plan: QueryPlan, reason: str) -> SchemaAnswer:
        ent = plan.entity or "the requested entity"
        text = (
            f"I cannot answer that from the loaded corpus ({reason}). "
            f"No supported claim about {ent} matches this question."
        )
        return SchemaAnswer(
            decision="audit",
            text=text,
            tier="UNKNOWN",
            sources=(),
            evidence=(),
            plan=plan,
            confidence=0.0,
            reason=reason,
        )
