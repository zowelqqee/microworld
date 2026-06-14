"""Wikipedia Self-Ingestion v1.

Safe self-feeding / data-acquisition pipeline for Microworld. Ingests local
Wikipedia-like documents, extracts candidate knowledge (reusing the existing
ingestion v2 + overlay builder), classifies risk/stability/conflict, quarantines
suspicious items, builds an overlay *delta proposal*, dry-runs it against the
existing overlay, and verifies that the QA / adversarial / cross-page benchmarks
still pass.

This is NOT live web ingestion and NOT autonomous trusted-memory writing. Raw
documents never flow directly into accepted memory. Deterministic, rule-based,
offline only — no ML, no embeddings, no network. ``safe_for_general_runtime`` is
always False.
"""
