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

from .entity_matcher import _get_pattern, _norm as _ematch_norm, build_surface_index, match_entities
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
        # Build all inverted indices once — O(n) amortized, O(1) per query.
        self._surface_index = build_surface_index(self._overlay)
        # Warm the pattern cache for all surfaces at init time so match_entities
        # never recompiles a regex during a live query.
        for surface, _kind, _meta in self._surface_index:
            _get_pattern(_ematch_norm(surface))
        self._build_indices()

    def _build_indices(self) -> None:
        """Construct per-type lookup dicts keyed by _norm(field)."""
        self._entity_equivalents_by_term: Dict[str, set[str]] = {}
        for e in self._overlay.entities():
            terms = {
                _norm(e.get("label", "")),
                _norm(e.get("source_page", "")),
                *{_norm(alias) for alias in (e.get("aliases") or [])},
            }
            terms = {term for term in terms if term}
            for term in terms:
                self._entity_equivalents_by_term.setdefault(term, set()).update(terms)

        self._defs_by_subj: Dict[str, List[dict]] = {}
        for d in self._overlay.definitions():
            self._defs_by_subj.setdefault(_norm(d.get("subject", "")), []).append(d)

        self._rels_by_subj: Dict[str, List[dict]] = {}
        self._rels_by_obj: Dict[str, List[dict]] = {}
        self._stable_rels_by_subj: Dict[str, List[dict]] = {}
        for r in self._overlay.relations():
            s = _norm(r.get("subject", ""))
            o = _norm(r.get("object", ""))
            self._rels_by_subj.setdefault(s, []).append(r)
            self._rels_by_obj.setdefault(o, []).append(r)
            if r.get("stability") in STABLE_STABILITIES:
                self._stable_rels_by_subj.setdefault(s, []).append(r)

        self._links_by_node: Dict[str, List[dict]] = {}
        for c in self._overlay.context_links():
            for field in ("source_page", "surface", "target"):
                k = _norm(c.get(field, ""))
                if k:
                    self._links_by_node.setdefault(k, []).append(c)

        self._sf_by_subj: Dict[str, List[dict]] = {}
        self._sf_by_obj: Dict[str, List[dict]] = {}
        for sf in self._overlay.source_facts():
            s = _norm(sf.get("subject", ""))
            o = _norm(sf.get("object", ""))
            if s:
                self._sf_by_subj.setdefault(s, []).append(sf)
            if o:
                self._sf_by_obj.setdefault(o, []).append(sf)

        # Inversion set: (subject_norm, object_norm) for relation-inversion detection.
        self._inversion_pairs: set = set()
        for r in self._overlay.relations():
            if r.get("predicate") in ("founded", "leader_of", "known_for"):
                s = _norm(r.get("subject", ""))
                o = _norm(r.get("object", ""))
                if s and o:
                    self._inversion_pairs.add((s, o))

    # ------------------------------------------------------------------ #
    def build(self, question: str) -> WorkingContextPack:
        matched = match_entities(question, self._overlay, prebuilt_index=self._surface_index)
        matched_names = {_norm(m.name) for m in matched}
        matched_surfaces = {_norm(m.surface) for m in matched}
        all_matched = matched_names | matched_surfaces
        all_matched = self._expand_entity_equivalents(all_matched)

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
    def _expand_entity_equivalents(self, matched: set) -> set:
        expanded = set(matched)
        for key in list(matched):
            expanded.update(self._entity_equivalents_by_term.get(key, set()))
        return expanded

    def _definitions_for(self, matched: set) -> List[ContextDefinition]:
        out: List[ContextDefinition] = []
        for k in matched:
            for d in self._defs_by_subj.get(k, []):
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
        seen: set = set()
        out: List[ContextRelation] = []
        for k in matched:
            for r in self._rels_by_subj.get(k, []) + self._rels_by_obj.get(k, []):
                rid = id(r)
                if rid not in seen:
                    seen.add(rid)
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
        seen: set = set()
        out: List[ContextRelation] = []
        for nb in neighbors:
            for r in self._rels_by_subj.get(nb, []) + self._rels_by_obj.get(nb, []):
                rid = id(r)
                if rid in seen:
                    continue
                seen.add(rid)
                key = (r.get("subject", ""), r.get("predicate", ""), r.get("object", ""))
                if key not in direct_keys:
                    out.append(self._rel_obj(r))
        return out

    def _weak_links(self, matched: set) -> List[ContextWeakLink]:
        seen: set = set()
        out: List[ContextWeakLink] = []
        for k in matched:
            for c in self._links_by_node.get(k, []):
                cid = id(c)
                if cid not in seen:
                    seen.add(cid)
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
        seen: set = set()
        out: List[ContextSourceFact] = []
        for k in matched:
            for sf in self._sf_by_subj.get(k, []) + self._sf_by_obj.get(k, []):
                sfid = id(sf)
                if sfid not in seen:
                    seen.add(sfid)
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
        yield from self._stable_rels_by_subj.get(node_norm, [])

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
        for r in self._stable_rels_by_subj.get(a_n, []) + self._stable_rels_by_subj.get(t_n, []):
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
        a_links = self._links_by_node.get(a_n, [])
        t_links = self._links_by_node.get(t_n, [])
        # Direct shared weak link: a link that mentions both a_n and t_n.
        for c in a_links:
            ends = {
                _norm(c.get("source_page", "")),
                _norm(c.get("surface", "")),
                _norm(c.get("target", "")),
            }
            if t_n in ends:
                return True
        # Weak bridge through a shared intermediate node.
        a_ends: set = set()
        for c in a_links:
            a_ends |= {
                _norm(c.get("source_page", "")),
                _norm(c.get("surface", "")),
                _norm(c.get("target", "")),
            }
        t_ends: set = set()
        for c in t_links:
            t_ends |= {
                _norm(c.get("source_page", "")),
                _norm(c.get("surface", "")),
                _norm(c.get("target", "")),
            }
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
        if (o_n, s_n) in self._inversion_pairs:
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
