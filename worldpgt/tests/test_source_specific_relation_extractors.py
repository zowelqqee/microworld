from __future__ import annotations

from worldpgt.knowledge_pump.open_web_pump import _open_web_source_gate
from worldpgt.knowledge_pump.source_specific_relation_extractors import extract_explicit_relations


# Real arXiv snapshot snippet: "Artificial Intelligence BlockCloud (AIBC)
# Technical Whitepaper", arXiv:1909.12063v1.  It is also used in the lane's
# manual spot-check report.
_DABFT_RECORD = {
    "source_kind": "arxiv",
    "source_url": "http://arxiv.org/abs/1909.12063v1",
    "title": "Artificial Intelligence BlockCloud (AIBC) Technical Whitepaper",
    "text": (
        "The DABFT uses deep learning techniques to predict and select the most "
        "suitable BFT algorithm in order to achieve the best balance of performance."
    ),
}
_DABFT_CANDIDATE = {
    "overlay_type": "overlay_relation",
    "subject": "DABFT",
    "predicate": "uses",
    "object": "deep learning techniques",
    "source_kind": "arxiv",
    "source_url": "https://arxiv.org/abs/1909.12063v1",
    "source_page": "Artificial Intelligence BlockCloud (AIBC) Technical Whitepaper",
    "risk": "low",
    "stability": "semi_stable",
}
_CYSEC_RECORD = {
    "source_kind": "arxiv",
    "source_url": "http://arxiv.org/abs/2308.04447v1",
    "title": "Automating the Communication of Cybersecurity Knowledge: Multi-Case Study",
    "text": (
        "CYSEC uses assessment questions and recommendations to communicate cybersecurity "
        "knowledge to the end-user SMBs and encourage self-motivated change."
    ),
}
_CYSEC_CANDIDATE = {
    "overlay_type": "overlay_relation",
    "subject": "CYSEC",
    "predicate": "uses",
    "object": "assessment questions and recommendations",
    "source_kind": "arxiv",
    "source_url": "http://arxiv.org/abs/2308.04447v1",
    "risk": "low",
    "stability": "semi_stable",
}


def test_arxiv_extractor_rederives_an_explicit_relation_from_a_real_snapshot():
    accepted, rejected = extract_explicit_relations(
        "arxiv", source_records=[_DABFT_RECORD], allowlisted_candidates=[_DABFT_CANDIDATE]
    )

    assert rejected == []
    assert len(accepted) == 1
    assert accepted[0]["evidence_text"].startswith("The DABFT uses")
    assert accepted[0]["open_web_extraction"] == "arxiv_explicit_relation_v1"
    assert _open_web_source_gate(accepted)["accepted"] == accepted


def test_arxiv_extractor_keeps_a_second_real_spot_check_relation():
    accepted, rejected = extract_explicit_relations(
        "arxiv", source_records=[_CYSEC_RECORD], allowlisted_candidates=[_CYSEC_CANDIDATE]
    )

    assert rejected == []
    assert accepted[0]["subject"] == "CYSEC"
    assert accepted[0]["object"] == "assessment questions and recommendations"


def test_arxiv_extractor_rejects_missing_source_record_without_guessing():
    accepted, rejected = extract_explicit_relations(
        "arxiv", source_records=[], allowlisted_candidates=[_DABFT_CANDIDATE]
    )

    assert accepted == []
    assert rejected[0]["reason"] == "source_specific_record_unavailable"


def test_arxiv_extractor_rejects_a_candidate_when_the_source_lacks_its_explicit_predicate():
    wrong = {**_DABFT_CANDIDATE, "predicate": "supports"}
    accepted, rejected = extract_explicit_relations(
        "arxiv", source_records=[_DABFT_RECORD], allowlisted_candidates=[wrong]
    )

    assert accepted == []
    assert rejected[0]["reason"] == "source_specific_explicit_relation_not_found"


def test_arxiv_extractor_rejects_a_real_snapshot_discourse_fragment_before_the_gate():
    # Real arXiv snapshot wording from "Quantum picturalism for topological
    # cluster-state computing": the source says "One implementation uses…",
    # but "One implementation" is not a stable graph referent.
    fragment_record = {
        "source_kind": "arxiv",
        "source_url": "http://arxiv.org/abs/1101.4722v3",
        "title": "Quantum picturalism for topological cluster-state computing",
        "text": (
            "Topological quantum computing is a way of allowing precise quantum computations "
            "to run on noisy and imperfect hardware. One implementation uses surface codes "
            "created by forming defects in a highly-entangled cluster state."
        ),
    }
    fragment = {
        **_DABFT_CANDIDATE,
        "subject": "One implementation",
        "object": "surface codes created by forming defects in a highly-entangled cluster state",
        "source_url": "http://arxiv.org/abs/1101.4722v3",
    }
    accepted, rejected = extract_explicit_relations(
        "arxiv",
        source_records=[fragment_record],
        allowlisted_candidates=[fragment],
    )

    assert accepted == []
    assert rejected[0]["reason"] == "source_specific_non_atomic_endpoint"


def test_extractor_rejects_a_generic_single_word_subject_before_the_gate():
    """A section/topic label is not a stable graph entity even in an explicit sentence."""

    record = {
        "source_kind": "crossref",
        "source_url": "https://doi.org/10.0000/example",
        "title": "Example",
        "text": "Evaluation uses quantitative and qualitative approaches.",
    }
    candidate = {
        **_DABFT_CANDIDATE,
        "subject": "Evaluation",
        "object": "quantitative and qualitative approaches",
        "source_kind": "crossref",
        "source_url": "https://doi.org/10.0000/example",
    }

    accepted, rejected = extract_explicit_relations(
        "crossref", source_records=[record], allowlisted_candidates=[candidate]
    )

    assert accepted == []
    assert rejected[0]["reason"] == "source_specific_non_atomic_endpoint"
