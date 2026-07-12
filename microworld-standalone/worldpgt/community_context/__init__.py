"""Low-trust community-context ingestion and retrieval.

This package is intentionally separate from wiki overlays and accepted memory.
It can index Reddit-like exports for conversational context, but the produced
items are never factual support by themselves.
"""

from worldpgt.community_context.reddit_engine import (
    build_reddit_community_context,
    load_reddit_records,
    query_community_context,
    render_community_context,
)
from worldpgt.community_context.cognitive_pattern_pump import (
    build_cognitive_pattern_graph,
    extract_cognitive_pattern_events,
    plan_answer_with_cognitive_patterns,
    query_cognitive_patterns,
)

__all__ = [
    "build_cognitive_pattern_graph",
    "build_reddit_community_context",
    "extract_cognitive_pattern_events",
    "load_reddit_records",
    "plan_answer_with_cognitive_patterns",
    "query_cognitive_patterns",
    "query_community_context",
    "render_community_context",
]
