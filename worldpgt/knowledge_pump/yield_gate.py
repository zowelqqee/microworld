"""Yield safety gate for Knowledge Pump v1.9.

Decides — from incremental yield metrics — whether a blind network fetch is
worth running, or whether the pump should switch to a yield-ranked frontier.

This is pure selection/control. It never fetches, never mutates accepted
memory, and never weakens any existing safety or precision gate. It only
*recommends* and, when blind fetching is unproductive, *requires an explicit
force flag* to proceed with blind fetching anyway.
"""

from __future__ import annotations

from typing import Any

# Minimum new-answerable-facts per ready document for a blind fetch to be
# considered productive. Tuned well below historical cumulative yield so it
# only fires when recent incremental yield has genuinely collapsed.
DEFAULT_MIN_YIELD_PER_READY_DOC = 0.05


def evaluate_yield_gate(
    metrics: dict[str, Any],
    *,
    min_yield_per_ready_doc: float = DEFAULT_MIN_YIELD_PER_READY_DOC,
    yield_ranked_available: bool = True,
    frontier_policy: str = "default",
    force: bool = False,
) -> dict[str, Any]:
    """Evaluate the yield gate from recent-window metrics.

    Parameters
    ----------
    metrics:
        Output of :func:`incremental_yield_trace.recent_window_metrics`.
    min_yield_per_ready_doc:
        Threshold below which blind fetching is deemed unproductive.
    yield_ranked_available:
        Whether a non-empty yield-ranked frontier plan exists.
    frontier_policy:
        ``"default"`` (blind) or ``"yield-ranked"``.
    force:
        Whether ``--force-low-yield-fetch`` was supplied.
    """
    recent_facts = int(metrics.get("recent_new_answerable_facts", 0))
    recent_ready = int(metrics.get("recent_ready_docs", 0))
    recent_batches = int(metrics.get("recent_batches_analyzed", 0))
    zero_batches = int(metrics.get("zero_yield_batch_count", 0))
    negative = bool(metrics.get("negative_yield_detected", False))

    yield_per_ready = recent_facts / recent_ready if recent_ready else 0.0
    low_yield_detected = yield_per_ready < min_yield_per_ready_doc
    all_recent_zero = recent_batches > 0 and zero_batches >= recent_batches

    # Blind fetching is unproductive when yield has collapsed by any signal.
    blind_unproductive = bool(negative or all_recent_zero or low_yield_detected)

    blind_fetch_recommended = not blind_unproductive
    yield_ranked_fetch_recommended = bool(blind_unproductive and yield_ranked_available)
    force_required_for_blind_fetch = blind_unproductive

    # Whether a blind fetch is *allowed* to proceed this run.
    blind_fetch_allowed = (not blind_unproductive) or force

    warnings: list[str] = []
    if blind_unproductive:
        warnings.append(
            "Recent batches produced near-zero new answerable facts; blind "
            "network fetch is not recommended."
        )
        if frontier_policy == "default":
            warnings.append(
                "Use --frontier-policy yield-ranked to prefer high-yield "
                "titles, or --force-low-yield-fetch to override."
            )
        if not yield_ranked_available:
            warnings.append(
                "No yield-ranked candidates are available; verify the frontier "
                "plan was generated."
            )
    if blind_unproductive and force:
        warnings.append(
            "--force-low-yield-fetch supplied: proceeding with blind fetch "
            "despite low yield."
        )

    return {
        "min_yield_per_ready_doc_threshold": min_yield_per_ready_doc,
        "recent_answerable_yield_per_ready_doc": round(yield_per_ready, 6),
        "recent_new_answerable_facts": recent_facts,
        "recent_ready_docs": recent_ready,
        "recent_batches_analyzed": recent_batches,
        "zero_yield_batch_count": zero_batches,
        "negative_yield_detected": negative,
        "low_yield_detected": low_yield_detected,
        "all_recent_batches_zero_yield": all_recent_zero,
        "blind_fetch_unproductive": blind_unproductive,
        "blind_fetch_recommended": blind_fetch_recommended,
        "yield_ranked_fetch_recommended": yield_ranked_fetch_recommended,
        "force_required_for_blind_fetch": force_required_for_blind_fetch,
        "blind_fetch_allowed": blind_fetch_allowed,
        "yield_ranked_available": yield_ranked_available,
        "frontier_policy": frontier_policy,
        "forced": bool(force),
        "warnings": warnings,
    }
