"""Context pack builder for Working Context Pack v1.

Reads the explicit wiki overlay (accepted or promoted) via the existing
read-only provider and assembles a :class:`WorkingContextPack`. It never writes
the overlay and never changes runtime behavior.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from worldpgt.knowledge.wiki_memory_overlay_provider import WikiMemoryOverlayProvider

from .entity_matcher import match_entities
from .types import (
    OVERLAY_ACCEPTED,
    OVERLAY_PROMOTED,
    STABLE_STABILITIES,
    ContextBlockedPath,
    ContextCandidatePath,
    ContextDefinition,
    ContextMissingKnowledgeHint,
    ContextRelation,
    ContextSourceFact,
    ContextWeakLink,
    MatchedEntity,
    WorkingContextPack,
)

# --------------------------------------------------------------------------- #
# Question safety screens (deterministic, regex-based).
# --------------------------------------------------------------------------- #
_CURRENT_LIVE_RE = re.compile(
    r"\b(current|currently|latest|today|right now|stock price|share price|"
    r"market cap|valuation|net worth|revenue|earnings)\b",
    re.I,
)
_PRIVATE_RE = re.compile(
    r"\b(phone number|home address|private email|date of birth|favorite|"
    r"favourite|social security|passport)\b",
    re.I,
)
_UNIVERSAL_RE = re.compile(
    r"\b(all .* are .* products|are all|is every|every .* is a)\b", re.I
)
_INVERSION_RE = re.compile(
    r"(did|does|is)\s+(.+?)\s+(found|founded|the founder of|lead|leads|leader of)\s+(.+?)[\?\.]*\s*$",
    re.I,
)
_ESTIMATE_RE = re.compile(r"\b(estimate|estimates|estimated|net worth|ranking)\b", re.I)


def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[''`]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


class _OverlayAccess:
    """Thin adapter so the matcher/builder can read overlay item lists."""

    def __init__(self, provider: WikiMemoryOverlayProvider) -> None:
        self._p = provider

    def entities(self) -> List[dict]:
        return self._p.all_entities()

    def definitions(self) -> List[dict]:
        return self._p.all_definitions()

    def relations(self) -> List[dict]:
        return self._p.all_relations()

    def context_links(self) -> List[dict]:
        return self._p.all_context_links()

    def source_facts(self) -> List[dict]:
        return self._p.all_source_facts()


class ContextPackBuilder:
    def __init__(
        self,
        overlay_json_path: str,
        overlay_mode: str = OVERLAY_ACCEPTED,
        knowledge_requests: Optional[List[dict]] = None,
    ) -> None:
        self.overlay_mode = overlay_mode
        self._provider = WikiMemoryOverlayProvider(overlay_json_path)
        self._overlay = _OverlayAccess(self._provider)
        self._knowledge_requests = knowledge_requests or []

    # ------------------------------------------------------------------ #
    def build(self, question: str) -> WorkingContextPack:
        matched = match_entities(question, self._overlay)
        matched_names = {_norm(m.name) for m in matched}
        matched_surfaces = {_norm(m.surface) for m in matched}
        all_matched = matched_names | matched_surfaces

        pack = WorkingContextPack(
            question=question,
            overlay_mode=self.overlay_mode,
            matched_entities=matched,
            safe_for_general_runtime=False,
        )

        pack.definitions = self._definitions_for(all_matched)
        pack.direct_relations = self._direct_relations(all_matched)
        neighbors = self._neighbors(pack.direct_relations, all_matched)
        pack.one_hop_relations = self._one_hop(neighbors, all_matched, pack.direct_relations)
        pack.weak_links = self._weak_links(all_matched)
        pack.source_facts = self._source_facts(all_matched)

        # Paths + blocks.
        self._build_paths(question, matched, all_matched, pack)

        # Safety screens.
        self._apply_safety_screens(question, matched, pack)

        # Missing knowledge hints (read-only, never applied).
        pack.missing_knowledge_hints = self._missing_hints(question)

        # Decide safe_for_answering.
        pack.safe_for_answering = self._decide_safe_for_answering(question, pack)
        pack.summary = self._summary(pack)
        return pack

    # ------------------------------------------------------------------ #
    def _definitions_for(self, matched: set) -> List[ContextDefinition]:
        out: List[ContextDefinition] = []
        for d in self._overlay.definitions():
            if _norm(d.get("subject", "")) in matched:
                out.append(
                    ContextDefinition(
                        subject=d.get("subject", ""),
                        definition=d.get("definition", ""),
                        predicate=d.get("predicate", ""),
                        stability=d.get("stability", ""),
                        temporal_class=d.get("temporal_class", ""),
                        evidence_text=d.get("evidence_text", ""),
                    )
                )
        return out

    def _rel_obj(self, r: dict) -> ContextRelation:
        return ContextRelation(
            subject=r.get("subject", ""),
            predicate=r.get("predicate", ""),
            object=r.get("object", ""),
            stability=r.get("stability", ""),
            risk=r.get("risk", ""),
            trust=r.get("trust", ""),
            temporal_class=r.get("temporal_class", ""),
            as_of=r.get("as_of", ""),
            evidence_text=r.get("evidence_text", ""),
            source_page=r.get("source_page", ""),
        )

    def _direct_relations(self, matched: set) -> List[ContextRelation]:
        out: List[ContextRelation] = []
        for r in self._overlay.relations():
            if _norm(r.get("subject", "")) in matched or _norm(r.get("object", "")) in matched:
                out.append(self._rel_obj(r))
        return out

    def _neighbors(self, direct: List[ContextRelation], matched: set) -> set:
        nb: set = set()
        for r in direct:
            for endpoint in (r.subject, r.object):
                if _norm(endpoint) not in matched:
                    nb.add(_norm(endpoint))
        return nb

    def _one_hop(
        self, neighbors: set, matched: set, direct: List[ContextRelation]
    ) -> List[ContextRelation]:
        direct_keys = {(r.subject, r.predicate, r.object) for r in direct}
        out: List[ContextRelation] = []
        for r in self._overlay.relations():
            key = (r.get("subject", ""), r.get("predicate", ""), r.get("object", ""))
            if key in direct_keys:
                continue
            if _norm(r.get("subject", "")) in neighbors or _norm(r.get("object", "")) in neighbors:
                out.append(self._rel_obj(r))
        return out

    def _weak_links(self, matched: set) -> List[ContextWeakLink]:
        out: List[ContextWeakLink] = []
        for c in self._overlay.context_links():
            blob = {
                _norm(c.get("source_page", "")),
                _norm(c.get("surface", "")),
                _norm(c.get("target", "")),
            }
            if blob & matched:
                out.append(
                    ContextWeakLink(
                        source_page=c.get("source_page", ""),
                        surface=c.get("surface", ""),
                        target=c.get("target", ""),
                        relation=c.get("relation", ""),
                        strength=c.get("strength", ""),
                        trust=c.get("trust", ""),
                    )
                )
        return out

    def _source_facts(self, matched: set) -> List[ContextSourceFact]:
        out: List[ContextSourceFact] = []
        for sf in self._overlay.source_facts():
            if _norm(sf.get("subject", "")) in matched or _norm(sf.get("object", "")) in matched:
                out.append(
                    ContextSourceFact(
                        subject=sf.get("subject", ""),
                        predicate=sf.get("predicate", ""),
                        object=sf.get("object", ""),
                        source_name=sf.get("source_name", ""),
                        as_of=sf.get("as_of", ""),
                        claim_type=sf.get("claim_type", ""),
                        temporal_class=sf.get("temporal_class", ""),
                        stability=sf.get("stability", ""),
                        risk=sf.get("risk", ""),
                        trust=sf.get("trust", ""),
                        requires_recheck=bool(sf.get("requires_recheck", True)),
                        evidence_text=sf.get("evidence_text", ""),
                    )
                )
        return out

    # ------------------------------------------------------------------ #
    def _stable_edges_from(self, node_norm: str):
        """Yield stable/semi-stable relations whose subject is node_norm."""
        for r in self._overlay.relations():
            if r.get("stability") not in STABLE_STABILITIES:
                continue
            if _norm(r.get("subject", "")) == node_norm:
                yield r

    def _build_paths(self, question, matched, matched_set, pack) -> None:
        # Anchor/target pairs from matched entities (ordered by appearance).
        names = [m.name for m in matched]
        if len(names) < 2:
            return
        anchor = names[0]
        targets = names[1:]

        for target in targets:
            a_n, t_n = _norm(anchor), _norm(target)
            if a_n == t_n:
                continue
            # 1-hop stable relation anchor->target (either direction).
            direct = self._find_stable_direct(a_n, t_n)
            if direct is not None:
                pack.candidate_paths.append(
                    ContextCandidatePath(
                        nodes=[direct.subject, direct.object],
                        relations=[f"{direct.subject} -{direct.predicate}-> {direct.object}"],
                        support=direct.stability,
                    )
                )
                continue
            # 2-hop stable path anchor -> X -> target.
            two = self._find_two_hop(a_n, t_n)
            if two is not None:
                pack.candidate_paths.append(two)
                continue
            # Only weak link connects them?
            if self._weak_only_connection(a_n, t_n):
                pack.blocked_paths.append(
                    ContextBlockedPath(
                        nodes=[anchor, target],
                        reason="weak_only_path",
                        detail="weak context link cannot prove a stable relation",
                    )
                )
            else:
                pack.blocked_paths.append(
                    ContextBlockedPath(
                        nodes=[anchor, target],
                        reason="missing_explicit_stable_path",
                        detail="no explicit stable/semi-stable path exists in the overlay",
                    )
                )

    def _find_stable_direct(self, a_n: str, t_n: str) -> Optional[ContextRelation]:
        for r in self._overlay.relations():
            if r.get("stability") not in STABLE_STABILITIES:
                continue
            s, o = _norm(r.get("subject", "")), _norm(r.get("object", ""))
            if {s, o} == {a_n, t_n}:
                return self._rel_obj(r)
        return None

    def _find_two_hop(self, a_n: str, t_n: str) -> Optional[ContextCandidatePath]:
        for r1 in self._stable_edges_from(a_n):
            mid = _norm(r1.get("object", ""))
            if mid == t_n:
                continue
            for r2 in self._stable_edges_from(mid):
                if _norm(r2.get("object", "")) == t_n:
                    return ContextCandidatePath(
                        nodes=[r1.get("subject"), r1.get("object"), r2.get("object")],
                        relations=[
                            f"{r1.get('subject')} -{r1.get('predicate')}-> {r1.get('object')}",
                            f"{r2.get('subject')} -{r2.get('predicate')}-> {r2.get('object')}",
                        ],
                        support="semi_stable",
                    )
        return None

    def _weak_only_connection(self, a_n: str, t_n: str) -> bool:
        """True if anchor and target are connected only via weak context links.

        Covers a direct shared weak link and a weak bridge (anchor weakly linked
        to some node that is also weakly linked to the target).
        """
        a_ends: set = set()
        t_ends: set = set()
        for c in self._overlay.context_links():
            ends = {
                _norm(c.get("source_page", "")),
                _norm(c.get("surface", "")),
                _norm(c.get("target", "")),
            }
            if a_n in ends and t_n in ends:
                return True  # direct shared weak link
            if a_n in ends:
                a_ends |= ends
            if t_n in ends:
                t_ends |= ends
        # Weak bridge through a shared intermediate node.
        return bool((a_ends & t_ends) - {a_n, t_n})

    # ------------------------------------------------------------------ #
    def _apply_safety_screens(self, question, matched, pack) -> None:
        # Current / live.
        if _CURRENT_LIVE_RE.search(question) and not self._has_supporting_source_fact(pack):
            pack.safety_notes.append(
                "current_or_live_question: requires explicit source-qualified fact "
                "with as_of; no stable context produced"
            )
        # Private / sensitive.
        if _PRIVATE_RE.search(question):
            pack.safety_notes.append(
                "private_or_sensitive_question: no factual context produced"
            )
            pack.direct_relations = []
            pack.definitions = []
            pack.candidate_paths = []
        # Unsupported universal.
        if _UNIVERSAL_RE.search(question):
            pack.blocked_paths.append(
                ContextBlockedPath(
                    nodes=[m.name for m in matched],
                    reason="unsupported_universal_claim",
                    detail="universal reversal is not supported by the overlay",
                )
            )
            pack.safety_notes.append(
                "unsupported_universal_claim: blocked; cannot generalize from "
                "specific overlay relations"
            )
        # Relation inversion.
        inv = self._detect_inversion(question)
        if inv is not None:
            subj, obj = inv
            pack.blocked_paths.append(
                ContextBlockedPath(
                    nodes=[subj, obj],
                    reason="relation_inversion",
                    detail=f"claim direction '{subj} -> {obj}' contradicts the overlay's "
                    f"stable relation direction",
                )
            )
            pack.safety_notes.append(
                "relation_inversion_suspicion: claimed direction is reversed vs overlay"
            )
            # Do not present any candidate path as supporting the inverted claim.
            pack.candidate_paths = []
        # Source-qualified volatile facts present.
        for sf in pack.source_facts:
            if sf.stability == "volatile" or sf.requires_recheck:
                pack.safety_notes.append(
                    f"source_qualified_volatile_fact: {sf.subject} {sf.predicate} "
                    f"(source={sf.source_name}, as_of={sf.as_of}) requires recheck; not stable"
                )
                break
        # Weak-only block note.
        if any(b.reason == "weak_only_path" for b in pack.blocked_paths):
            pack.safety_notes.append(
                "weak_only_path: weak context link cannot establish a stable relation"
            )

    def _detect_inversion(self, question: str):
        m = _INVERSION_RE.search(question)
        if not m:
            return None
        subj = m.group(2).strip()
        obj = m.group(4).strip()
        s_n, o_n = _norm(subj), _norm(obj)
        # Inversion if the overlay has a stable relation in the REVERSE direction
        # (obj is subject of a founded/leader_of relation whose object is subj).
        for r in self._overlay.relations():
            if r.get("predicate") not in ("founded", "leader_of", "known_for"):
                continue
            if _norm(r.get("subject", "")) == o_n and _norm(r.get("object", "")) == s_n:
                return subj, obj
        return None

    def _has_supporting_source_fact(self, pack) -> bool:
        return bool(pack.source_facts)

    # ------------------------------------------------------------------ #
    def _missing_hints(self, question: str) -> List[ContextMissingKnowledgeHint]:
        out: List[ContextMissingKnowledgeHint] = []
        q_norm = _norm(question)
        for req in self._knowledge_requests:
            rq = _norm(req.get("question", ""))
            subj = _norm(req.get("candidate_subject", ""))
            obj = _norm(req.get("candidate_object", ""))
            match = rq == q_norm
            if not match and subj and obj:
                match = subj in q_norm and obj in q_norm
            if match:
                out.append(
                    ContextMissingKnowledgeHint(
                        request_id=req.get("id", ""),
                        category=req.get("category", req.get("missing_knowledge_type", "")),
                        needed_evidence_type=req.get("needed_evidence_type", ""),
                        suggested_source_topic=req.get("suggested_source_topic", ""),
                        must_remain_audit_until_supported=bool(
                            req.get("must_remain_audit_until_supported", True)
                        ),
                    )
                )
        return out

    # ------------------------------------------------------------------ #
    def _decide_safe_for_answering(self, question: str, pack: WorkingContextPack) -> bool:
        # Hard safety blocks.
        if _PRIVATE_RE.search(question):
            return False
        if _UNIVERSAL_RE.search(question):
            return False
        if self._detect_inversion(question) is not None:
            return False
        if _CURRENT_LIVE_RE.search(question) and not pack.source_facts:
            return False
        if any(
            b.reason in ("relation_inversion", "unsupported_universal_claim")
            for b in pack.blocked_paths
        ):
            return False
        # An explicit stable/semi-stable candidate path always answers.
        if pack.candidate_paths:
            return True
        # For connection/path questions (a blocked anchor->target pair with no
        # candidate path), unrelated direct relations do NOT make it answerable.
        if pack.blocked_paths:
            return False
        # Otherwise (single-subject definitional / direct-relation question):
        # answerable only with an explicit stable/semi-stable direct relation
        # that is not a weak link or a volatile source-qualified fact.
        if any(
            r.stability in STABLE_STABILITIES and r.trust != "weak_context_only"
            for r in pack.direct_relations
        ):
            return True
        return False

    def _summary(self, pack: WorkingContextPack) -> str:
        ents = ", ".join(m.name for m in pack.matched_entities) or "none"
        if pack.safe_for_answering:
            verdict = "explicit stable/semi-stable support available"
        elif pack.blocked_paths:
            verdict = "no stable path; blocked/safety-noted"
        else:
            verdict = "insufficient explicit support; audit"
        return (
            f"entities=[{ents}] stable_paths={len(pack.candidate_paths)} "
            f"weak_links={len(pack.weak_links)} source_facts={len(pack.source_facts)} "
            f"blocked={len(pack.blocked_paths)} -> {verdict}"
        )


def load_knowledge_requests(path: Path) -> List[dict]:
    if not Path(path).is_file():
        return []
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
