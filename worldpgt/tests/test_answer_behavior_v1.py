"""Focused tests for the answer-behavior pattern layer.

All graphs below use deliberately meaningless entity and relation names —
the layer must behave identically for any vocabulary, so nothing here may
depend on real entities, domains, or known predicate strings.
"""

from __future__ import annotations

from array import array

from worldpgt.reasoning.answer_behavior import (
    PlanningInstrumentation,
    build_answer_plan,
    plan_is_expansion,
    prepare_evidence_edges,
    prepare_evidence_graph,
    prepare_persistent_evidence_graph,
)
from worldpgt.reasoning.answer_plan_renderer import render_answer_plan


def _edge(
    subject: str,
    predicate: str,
    obj: str,
    *,
    evidence: str | None = None,
    sources: tuple[str, ...] = ("https://example.test/src-a",),
    stability: str = "semi_stable",
    risk: str = "medium",
) -> dict:
    item = {
        "overlay_type": "overlay_relation",
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "stability": stability,
        "risk": risk,
        "trust": "proposal_open_web_exploratory",
    }
    if evidence is not None:
        item["evidence_text"] = evidence
    else:
        item["evidence_text"] = f"{subject} {predicate.replace('_', ' ')} {obj}."
    if sources:
        item["source_url"] = sources[0]
        item["supporting_sources"] = list(sources)
    return item


def _connected_graph() -> list[dict]:
    return [
        _edge(
            "veltrix array",
            "channels",
            "calibrated gates",
            sources=("https://example.test/src-a", "https://example.test/src-b"),
        ),
        # Exact chain: subject equals the first edge's object node.
        _edge(
            "calibrated gates",
            "regulates",
            "the compensator drift budget across four subrings",
        ),
        # Sibling facet on the target itself.
        _edge("veltrix array", "anchors", "a drift compensator lattice"),
        # Fully disconnected evidence sharing a question word ("channels").
        _edge("floop unit", "channels", "a quantum blender"),
    ]


_QUESTION = "What does the veltrix array do with its calibrated gates and channels?"


def test_connected_local_graph_builds_multi_block_plan():
    plan = build_answer_plan(_QUESTION, _connected_graph(), targets=["veltrix array"])

    assert plan is not None
    assert len(plan.blocks) >= 2
    assert plan_is_expansion(plan)
    assert plan.blocks[0].kind == "direct_claim"
    kinds = {block.kind for block in plan.blocks}
    assert kinds <= {
        "direct_claim",
        "explanatory_continuation",
        "sibling_elaboration",
        "uncertainty_note",
    }
    # The plan explains why it stopped and how blocks were ordered.
    assert plan.stop_reason
    assert len(plan.ordering_rationale) == len(plan.blocks)


def test_coherent_chain_is_preferred_over_random_enumeration():
    plan = build_answer_plan(_QUESTION, _connected_graph(), targets=["veltrix array"])

    assert plan is not None
    ids = plan.evidence_ids()
    # The exact-attachment chain continuation is part of the plan and it
    # attaches through the node introduced by the direct claim.
    chain_blocks = [
        block for block in plan.blocks
        if block.step.edge.subject == "calibrated gates"
    ]
    assert chain_blocks, ids
    assert chain_blocks[0].kind == "explanatory_continuation"
    assert chain_blocks[0].step.attach_node == "calibrated gates"
    assert chain_blocks[0].step.introduced_by_block == 0


def test_relation_diversity_prefers_a_new_relation_at_the_same_node():
    # Complementary ``enables`` objects occupy one relation slot.  The next
    # block must therefore prefer the structurally new ``uses`` relation over
    # flat edge-by-edge enumeration, and expose that reason in its score
    # breakdown.
    graph = [
        _edge("veltrix array", "enables", "calibrated gates"),
        _edge("veltrix array", "enables", "diagnostic channels"),
        _edge("veltrix array", "uses", "governance protocols"),
    ]
    plan = build_answer_plan(
        "What is veltrix array?", graph, targets=["veltrix array"], max_blocks=3
    )

    assert plan is not None
    assert [block.step.edge.predicate for block in plan.blocks[:2]] == [
        "enables", "uses",
    ]
    assert [edge.object for edge in plan.blocks[0].all_object_edges()] == [
        "calibrated gates", "diagnostic channels",
    ]
    assert plan.blocks[1].step.score.relation_diversity == 1.0
    assert "relation_diversity" in plan.to_dict()["blocks"][1]["step"]["score"]


def test_multi_object_relation_slot_keeps_all_exact_edges_and_renders_one_list():
    """A block limit constrains relation groups, never valid fan-out objects."""
    graph = [
        _edge("veltrix array", "developed_by", "orion works"),
        _edge("veltrix array", "developed_by", "meridian collective"),
        _edge("veltrix array", "developed_by", "cinder laboratory"),
        _edge("veltrix array", "runs_on", "the amber substrate"),
    ]
    plan = build_answer_plan(
        "Who developed veltrix array and what does it run on?",
        graph,
        targets=["veltrix array"],
        predicate_filter=frozenset({"developed_by", "runs_on"}),
        max_blocks=2,
    )

    assert plan is not None
    developed = next(block for block in plan.blocks if block.step.edge.predicate == "developed_by")
    assert [edge.object for edge in developed.all_object_edges()] == [
        "cinder laboratory", "meridian collective", "orion works",
    ]
    assert set(plan.evidence_ids()) == {
        "edge:veltrix array|developed_by|cinder laboratory",
        "edge:veltrix array|developed_by|meridian collective",
        "edge:veltrix array|developed_by|orion works",
        "edge:veltrix array|runs_on|the amber substrate",
    }
    assert [slot["object"] for slot in developed.to_dict()["object_slots"]] == [
        "cinder laboratory", "meridian collective", "orion works",
    ]
    text = render_answer_plan(plan)
    assert "cinder laboratory, meridian collective, and orion works" in text
    assert "the amber substrate" in text


def test_multi_object_slot_is_identical_on_prepared_and_persistent_retrieval(tmp_path):
    """Exact adjacency returns the full fan-out in every serving index mode."""
    graph = [
        _edge("veltrix array", "runs_on", "the amber substrate"),
        _edge("veltrix array", "runs_on", "the cobalt substrate"),
        _edge("veltrix array", "runs_on", "the jade substrate"),
        _edge("veltrix array", "developed_by", "orion works"),
    ]
    kwargs = {
        "targets": ["veltrix array"],
        "predicate_filter": frozenset({"runs_on", "developed_by"}),
        "max_blocks": 2,
    }
    legacy = build_answer_plan("What runs on and developed veltrix array?", graph, **kwargs)
    prepared = build_answer_plan(
        "What runs on and developed veltrix array?", [],
        prepared_edges=prepare_evidence_graph(graph), **kwargs,
    )
    persistent = prepare_persistent_evidence_graph(graph, tmp_path / "fanout.sqlite")
    disk = build_answer_plan(
        "What runs on and developed veltrix array?", [], prepared_edges=persistent, **kwargs,
    )

    assert legacy is not None and prepared is not None and disk is not None
    assert prepared.to_dict() == legacy.to_dict() == disk.to_dict()
    runs_on = next(block for block in disk.blocks if block.step.edge.predicate == "runs_on")
    assert len(runs_on.all_object_edges()) == 3


def test_explicit_predicate_intent_prevents_unrelated_diversity_blocks():
    graph = [
        _edge("veltrix array", "enables", "calibrated gates"),
        _edge("veltrix array", "supports", "diagnostic channels"),
        _edge("veltrix array", "uses", "governance protocols"),
    ]
    plan = build_answer_plan(
        "What does veltrix array enable?",
        graph,
        targets=["veltrix array"],
        predicate_filter="enables",
        max_blocks=3,
    )
    assert plan is not None
    assert [block.step.edge.predicate for block in plan.blocks] == ["enables"]


def test_explicit_multi_predicate_intent_keeps_each_requested_relation_only():
    graph = [
        _edge("veltrix array", "enables", "calibrated gates"),
        _edge("veltrix array", "supports", "diagnostic channels"),
        _edge("veltrix array", "uses", "governance protocols"),
    ]
    plan = build_answer_plan(
        "What does veltrix array enable and support?",
        graph,
        targets=["veltrix array"],
        predicate_filter=frozenset({"enables", "supports"}),
        max_blocks=3,
    )
    assert plan is not None
    assert {block.step.edge.predicate for block in plan.blocks} == {"enables", "supports"}


def test_explicit_multi_predicate_intent_keeps_distinct_relations_with_same_object():
    graph = [
        _edge("veltrix array", "developed_by", "orion works"),
        _edge("veltrix array", "product_of", "orion works"),
    ]
    plan = build_answer_plan(
        "Who developed and manufactured veltrix array?",
        graph,
        targets=["veltrix array"],
        predicate_filter=frozenset({"developed_by", "product_of"}),
    )

    assert plan is not None
    assert [block.step.edge.predicate for block in plan.blocks] == ["developed_by", "product_of"]


def test_implicit_two_fact_contract_selects_two_relation_groups_even_with_same_object():
    # The two claims have identical payload text, so ordinary novelty alone
    # would stop after one.  A structural cardinality request must preserve
    # both independently evidenced relation groups without naming either one.
    graph = [
        _edge("veltrix array", "developed_by", "orion works"),
        _edge("veltrix array", "owned_by", "orion works"),
        _edge("veltrix array", "uses", "calibration lattice"),
    ]
    plan = build_answer_plan(
        "Tell me two key relations about veltrix array.",
        graph,
        targets=["veltrix array"],
        required_distinct_predicates=2,
        max_blocks=2,
    )

    assert plan is not None
    assert [block.step.edge.predicate for block in plan.blocks] == ["developed_by", "owned_by"]
    assert plan_is_expansion(plan)


def test_focused_relation_lookup_rejects_contained_target_neighbours():
    # A lexical neighbour is valid graph evidence about a different entity,
    # but must not become an additional fact in an answer focused on the exact
    # target.  This protects support/provenance without subject-specific rules.
    graph = [
        _edge("north relay", "used_for", "signal routing"),
        _edge("north relay extended", "used_for", "diagnostic replay"),
    ]
    plan = build_answer_plan(
        "For what application is north relay employed?",
        graph,
        targets=["north relay"],
        predicate_filter="used_for",
        max_blocks=3,
    )

    assert plan is not None
    assert plan.evidence_ids() == ["edge:north relay|used_for|signal routing"]


def test_target_label_tokens_do_not_hide_a_distinct_object_entity():
    # ``adobe`` occurs inside the target name, but it is still a different
    # entity and must remain novel as the object of ``developed_by``.
    graph = [
        _edge("adobe golive", "developed_by", "adobe"),
        _edge("adobe golive", "used_for", "computer graphics"),
    ]
    plan = build_answer_plan(
        "By whom was Adobe GoLive engineered, and for what application is Adobe GoLive employed?",
        graph,
        targets=["adobe golive"],
        predicate_filter=frozenset({"developed_by", "used_for"}),
    )

    assert plan is not None
    assert {block.step.edge.predicate for block in plan.blocks} == {"developed_by", "used_for"}


def test_every_block_is_traceable_to_evidence_and_sources():
    plan = build_answer_plan(_QUESTION, _connected_graph(), targets=["veltrix array"])

    assert plan is not None
    for block in plan.blocks:
        edge = block.step.edge
        assert edge.evidence_id.startswith("edge:")
        assert edge.evidence_text
        assert edge.sources
        assert block.step.score.total > 0
    payload = plan.to_dict()
    assert payload["score_weights"]
    for block_payload in payload["blocks"]:
        assert block_payload["step"]["edge"]["evidence_text"]
        assert block_payload["step"]["edge"]["sources"]
    trace = plan.trace()
    assert len(trace["steps"]) == len(plan.blocks)
    assert all(step["evidence_id"] for step in trace["steps"])


def test_disconnected_evidence_never_enters_plan():
    plan = build_answer_plan(_QUESTION, _connected_graph(), targets=["veltrix array"])

    assert plan is not None
    assert "edge:floop unit|channels|a quantum blender" not in plan.evidence_ids()


def test_single_reliable_link_stays_single_block():
    graph = [
        _edge("veltrix array", "channels", "calibrated gates"),
        _edge("floop unit", "channels", "a quantum blender"),
    ]
    plan = build_answer_plan(_QUESTION, graph, targets=["veltrix array"])

    assert plan is not None
    assert len(plan.blocks) == 1
    assert not plan_is_expansion(plan)


def test_missing_evidence_creates_no_claims():
    no_evidence = [
        {
            "overlay_type": "overlay_relation",
            "subject": "veltrix array",
            "predicate": "channels",
            "object": "calibrated gates",
            "stability": "semi_stable",
            "risk": "medium",
            "source_url": "https://example.test/src-a",
        }
    ]
    no_source = [
        {
            "overlay_type": "overlay_relation",
            "subject": "veltrix array",
            "predicate": "channels",
            "object": "calibrated gates",
            "evidence_text": "veltrix array channels calibrated gates.",
            "stability": "semi_stable",
            "risk": "medium",
        }
    ]
    assert build_answer_plan(_QUESTION, no_evidence, targets=["veltrix array"]) is None
    assert build_answer_plan(_QUESTION, no_source, targets=["veltrix array"]) is None


def test_unknown_target_or_empty_graph_yields_no_plan():
    assert build_answer_plan(_QUESTION, [], targets=["veltrix array"]) is None
    assert build_answer_plan(_QUESTION, _connected_graph(), targets=[]) is None
    assert (
        build_answer_plan(_QUESTION, _connected_graph(), targets=["absent thing"])
        is None
    )


def test_conflicting_relation_becomes_uncertainty_not_confident_claim():
    graph = [
        # Same subject+relation, two evidence-backed but token-disjoint objects.
        _edge("veltrix array", "channels", "calibrated gates"),
        _edge("veltrix array", "channels", "an open plasma sluice"),
        # Elsewhere in the graph the relation is single-valued, so the pair
        # above reads as a real conflict rather than natural breadth.
        _edge("brumal disk", "channels", "one narrow feed"),
        _edge("quorv shell", "channels", "a single duct"),
    ]
    plan = build_answer_plan(_QUESTION, graph, targets=["veltrix array"])

    assert plan is not None
    first = plan.blocks[0]
    assert first.kind == "uncertainty_note"
    assert "conflicting_alternatives" in first.cautions
    assert first.alternatives

    text = render_answer_plan(plan)
    assert "neither reading is treated as settled" in text
    assert "calibrated gates" in text and "an open plasma sluice" in text


def test_multi_valued_relation_is_breadth_not_conflict():
    graph = [
        _edge("veltrix array", "channels", "calibrated gates"),
        _edge("veltrix array", "channels", "an open plasma sluice"),
        _edge("veltrix array", "channels", "the outer manifold ring"),
    ]
    plan = build_answer_plan(_QUESTION, graph, targets=["veltrix array"])

    assert plan is not None
    assert all(block.kind != "uncertainty_note" for block in plan.blocks)
    assert all(
        "conflicting_alternatives" not in block.cautions for block in plan.blocks
    )


def test_single_source_blocks_are_marked_and_rendered_with_provenance_note():
    plan = build_answer_plan(_QUESTION, _connected_graph(), targets=["veltrix array"])

    assert plan is not None
    single_source_blocks = [
        block for block in plan.blocks if "single_source" in block.cautions
    ]
    assert single_source_blocks  # graph above has single-source edges
    text = render_answer_plan(plan)
    assert "single source" in text
    assert "proposal-level" in text


def test_selection_is_name_and_predicate_agnostic():
    mapping: dict[str, str] = {}

    # Function words must survive the renaming unchanged: the point of the
    # test is that *content* vocabulary is irrelevant to selection, not that
    # grammar words are content.
    _function_words = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
        "its", "does", "do", "what", "how", "across",
    }

    def _map_word(word: str) -> str:
        low = word.lower()
        if len(low) <= 2 or low in _function_words:
            return word
        if low not in mapping:
            mapping[low] = f"tok{len(mapping)}zz"
        return mapping[low]

    def _map_text(text: str) -> str:
        return " ".join(
            _map_word(word.strip(".,?")) + word[len(word.rstrip('.,?')):]
            for word in text.split()
        )

    original_graph = _connected_graph()
    renamed_graph = []
    for item in original_graph:
        renamed = dict(item)
        for field in ("subject", "predicate", "object", "evidence_text"):
            renamed[field] = _map_text(str(item[field]))
        renamed_graph.append(renamed)

    original_plan = build_answer_plan(
        _QUESTION, original_graph, targets=["veltrix array"]
    )
    renamed_plan = build_answer_plan(
        _map_text(_QUESTION),
        renamed_graph,
        targets=[_map_text("veltrix array")],
    )

    assert original_plan is not None and renamed_plan is not None
    assert [b.kind for b in original_plan.blocks] == [b.kind for b in renamed_plan.blocks]
    assert [
        round(b.step.score.total, 6) for b in original_plan.blocks
    ] == [round(b.step.score.total, 6) for b in renamed_plan.blocks]
    assert len(original_plan.rejected) == len(renamed_plan.rejected)


def test_definition_items_do_not_participate_in_relation_only_layer():
    graph = [
        {
            "overlay_type": "overlay_definition",
            "subject": "veltrix array",
            "definition": "modular flux distribution rig",
            "predicate": "is_a",
            "evidence_text": "The veltrix array, a modular flux distribution rig, was described.",
            "source_url": "https://example.test/def",
            "stability": "semi_stable",
            "risk": "medium",
        },
        _edge("veltrix array", "channels", "calibrated gates"),
    ]
    plan = build_answer_plan(_QUESTION, graph, targets=["veltrix array"])

    assert plan is not None
    assert all(
        b.step.edge.object != "modular flux distribution rig" for b in plan.blocks
    )
    assert plan.evidence_ids() == ["edge:veltrix array|channels|calibrated gates"]


def test_definition_without_own_predicate_is_not_an_edge():
    graph = [
        {
            "overlay_type": "overlay_definition",
            "subject": "veltrix array",
            "definition": "modular flux distribution rig",
            "evidence_text": "The veltrix array, a modular flux distribution rig.",
            "source_url": "https://example.test/def",
        }
    ]
    assert build_answer_plan(_QUESTION, graph, targets=["veltrix array"]) is None


def test_plural_and_singular_forms_do_not_gain_morphological_relevance():
    graph = [_edge("veltrix array", "channels", "calibrated gates")]
    # Question uses singular forms; the edge uses plurals.
    plan = build_answer_plan(
        "Which gate does the veltrix array channel?",
        graph,
        targets=["veltrix array"],
    )
    assert plan is not None
    # The target name still overlaps, but ``gate``/``gates`` and
    # ``channel``/``channels`` remain distinct tokens.
    assert 0.0 < plan.blocks[0].step.score.direct_relevance <= 0.5


def test_sparse_relation_observations_do_not_claim_conflict():
    # Only the target subject uses this relation: too few observations to
    # call it single-valued, so two disjoint objects are breadth, not conflict.
    graph = [
        _edge("veltrix array", "channels", "calibrated gates"),
        _edge("veltrix array", "channels", "an open plasma sluice"),
    ]
    plan = build_answer_plan(_QUESTION, graph, targets=["veltrix array"])

    assert plan is not None
    assert all(block.kind != "uncertainty_note" for block in plan.blocks)


def test_containment_attachment_assembles_paraphrased_chain():
    graph = [
        _edge("veltrix array", "channels", "calibrated gates"),
        # Chain subject paraphrases the node: full name contained verbatim.
        _edge(
            "the calibrated gates of the outer manifold",
            "regulates",
            "compensator drift budgets across four subrings",
        ),
    ]
    plan = build_answer_plan(_QUESTION, graph, targets=["veltrix array"])

    assert plan is not None
    contained = [b for b in plan.blocks if b.step.attach_mode == "contained"]
    assert contained
    assert contained[0].step.attach_node == "calibrated gates"
    assert contained[0].kind == "explanatory_continuation"

    indexed = build_answer_plan(
        _QUESTION,
        [],
        targets=["veltrix array"],
        prepared_edges=prepare_evidence_graph(graph),
    )
    assert indexed is not None
    assert indexed.to_dict() == plan.to_dict()


def test_prepared_graph_preserves_conflict_detection():
    graph = [
        _edge("veltrix array", "channels", "calibrated gates"),
        _edge("veltrix array", "channels", "an open plasma sluice"),
        _edge("brumal disk", "channels", "one narrow feed"),
        _edge("quorv shell", "channels", "a single duct"),
    ]
    legacy = build_answer_plan(_QUESTION, graph, targets=["veltrix array"])
    indexed = build_answer_plan(
        _QUESTION,
        [],
        targets=["veltrix array"],
        prepared_edges=prepare_evidence_graph(graph),
    )

    assert legacy is not None and indexed is not None
    assert indexed.to_dict() == legacy.to_dict()


def test_shared_common_words_are_still_not_an_attachment():
    graph = [
        _edge("veltrix array", "channels", "natural flux processing"),
        # Shares two words with the node above but does not contain its name.
        _edge(
            "a natural approach",
            "handles",
            "flux for expressing deduction rules in processing pipelines",
        ),
    ]
    plan = build_answer_plan(_QUESTION, graph, targets=["veltrix array"])

    assert plan is not None
    assert "edge:a natural approach|handles|flux for expressing deduction rules in processing pipelines" not in plan.evidence_ids()


def test_two_edges_from_one_rich_span_collapse_to_a_single_block():
    # Both facts (the descriptor and the relation) are extracted from one
    # source sentence, so the second adds no new information once the first
    # is surfaced.  Span-aware coverage collapses them into one block whose
    # single quoted span conveys both facts — no redundant second sentence.
    shared_span = (
        "The report presents veltrix array, a modular flux distribution rig "
        "that channels calibrated gates."
    )
    graph = [
        _edge("veltrix array", "channels", "calibrated gates", evidence=shared_span),
        _edge(
            "veltrix array",
            "is_a",
            "modular flux distribution rig",
            evidence=shared_span,
        ),
    ]
    plan = build_answer_plan(_QUESTION, graph, targets=["veltrix array"])

    assert plan is not None
    assert len(plan.blocks) == 1
    assert not plan_is_expansion(plan)
    text = render_answer_plan(plan)
    assert text.count("The report presents veltrix array") == 1
    # The single span still carries both facts.
    assert "modular flux distribution rig" in text
    assert "channels calibrated gates" in text


def test_literal_predicate_label_is_not_counted_as_new_information():
    # A relation and a descriptor of the same target, drawn from one span:
    # the descriptor's only "new" token must not be the predicate label
    # itself (e.g. "is_a"), so the redundant descriptor collapses out.
    span = "Veltrix array, a modular flux rig, channels calibrated gates."
    graph = [
        _edge("veltrix array", "channels", "calibrated gates", evidence=span),
        _edge("veltrix array", "is_a", "modular flux rig", evidence=span),
    ]
    plan = build_answer_plan(_QUESTION, graph, targets=["veltrix array"])
    assert plan is not None
    assert len(plan.blocks) == 1


def test_non_referential_deictic_nodes_are_dropped():
    graph = [
        # Subject is a first-person deictic phrase: not a standalone referent.
        _edge("our approach", "channels", "calibrated gates"),
        # Object is a bare pronoun: pure noise.
        _edge("veltrix array", "produces", "it"),
        # A real edge so the target still resolves.
        _edge("veltrix array", "channels", "calibrated gates"),
    ]
    plan = build_answer_plan(_QUESTION, graph, targets=["veltrix array"])

    assert plan is not None
    ids = plan.evidence_ids()
    assert "edge:our approach|channels|calibrated gates" not in ids
    assert "edge:veltrix array|produces|it" not in ids
    assert "edge:veltrix array|channels|calibrated gates" in ids


def test_leading_definite_article_object_is_still_referential():
    # "the X" is a definite reference to a named thing, not deixis — it must
    # remain a usable node.
    graph = [
        _edge("veltrix array", "uses", "the calibrated gate protocol"),
        _edge(
            "the calibrated gate protocol",
            "regulates",
            "compensator drift budgets across four subrings",
        ),
    ]
    plan = build_answer_plan(_QUESTION, graph, targets=["veltrix array"])

    assert plan is not None
    assert "edge:veltrix array|uses|the calibrated gate protocol" in plan.evidence_ids()


def test_attribution_preamble_is_stripped_but_raw_span_preserved():
    from worldpgt.reasoning.answer_plan_renderer import _strip_attribution_preamble

    span = (
        "This paper introduces veltrix array, a modular flux distribution rig "
        "that channels calibrated gates."
    )
    rendered = _strip_attribution_preamble(span, "veltrix array")
    assert rendered.startswith("Veltrix array is a modular flux distribution rig")
    assert "This paper introduces" not in rendered

    # The plan/trace keeps the raw span verbatim (provenance is untouched).
    graph = [
        _edge("veltrix array", "is_a", "modular flux distribution rig", evidence=span),
    ]
    plan = build_answer_plan(_QUESTION, graph, targets=["veltrix array"])
    assert plan is not None
    assert plan.blocks[0].step.edge.evidence_text == span


def test_prepared_edges_produce_identical_plans():
    graph = _connected_graph()
    prepared = prepare_evidence_edges(graph)
    from_items = build_answer_plan(_QUESTION, graph, targets=["veltrix array"])
    from_prepared = build_answer_plan(
        _QUESTION, [], targets=["veltrix array"], prepared_edges=prepared
    )

    assert from_items is not None and from_prepared is not None
    assert from_items.to_dict() == from_prepared.to_dict()


def test_prepared_graph_keeps_plan_identical_without_scanning_disconnected_edges():
    local = _connected_graph()
    disconnected = [
        _edge(f"unrelated subject {index}", "relates_to", f"unrelated object {index}")
        for index in range(200)
    ]
    overlay = [*local, *disconnected]
    compatibility_plan = build_answer_plan(
        _QUESTION,
        [],
        targets=["veltrix array"],
        prepared_edges=prepare_evidence_edges(overlay),
    )
    instrumentation = PlanningInstrumentation()
    indexed_plan = build_answer_plan(
        _QUESTION,
        [],
        targets=["veltrix array"],
        prepared_edges=prepare_evidence_graph(overlay),
        instrumentation=instrumentation,
    )

    assert compatibility_plan is not None and indexed_plan is not None
    assert indexed_plan.to_dict() == compatibility_plan.to_dict()
    # The indexed path sees the target frontier and its introduced neighbour,
    # not the 200 disconnected edges in the store.
    assert instrumentation.edges_scanned < len(overlay)
    assert instrumentation.candidate_evaluations < len(overlay)


def test_prepared_graph_uses_compact_singleton_and_fanout_postings():
    graph = prepare_evidence_graph([
        _edge("single source", "relates_to", "single object"),
        _edge("shared source", "relates_to", "first object"),
        _edge("shared source", "relates_to", "second object"),
    ])

    assert isinstance(graph.adjacency["single source"], int)
    assert isinstance(graph.adjacency["shared source"], array)
    # Both sides need at least two content tokens before the fuzzy index stores
    # them; exact adjacency still covers the singleton side completely.
    assert "single" in graph.token_index


def test_persistent_graph_reopens_from_disk_with_identical_local_plan(tmp_path):
    overlay = _connected_graph()
    cache_path = tmp_path / "behavior-index.sqlite"
    legacy = build_answer_plan(_QUESTION, overlay, targets=["veltrix array"])
    persistent = prepare_persistent_evidence_graph(overlay, cache_path)
    indexed = build_answer_plan(
        _QUESTION,
        [],
        targets=["veltrix array"],
        prepared_edges=persistent,
    )
    reopened = prepare_persistent_evidence_graph(overlay, cache_path)

    assert cache_path.is_file()
    assert legacy is not None and indexed is not None
    assert indexed.to_dict() == legacy.to_dict()
    assert [edge.evidence_id for edge in reopened] == [
        edge.evidence_id for edge in persistent
    ]


def test_plan_is_deterministic():
    first = build_answer_plan(_QUESTION, _connected_graph(), targets=["veltrix array"])
    second = build_answer_plan(_QUESTION, _connected_graph(), targets=["veltrix array"])

    assert first is not None and second is not None
    assert first.to_dict() == second.to_dict()
    assert render_answer_plan(first) == render_answer_plan(second)
