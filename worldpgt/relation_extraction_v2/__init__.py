"""Relation Extraction v2 for Wikipedia snapshot ingestion.

Narrow, deterministic regex-based extraction of explicit relations from local
Wikipedia snapshot text. No ML, no embeddings, no network, no auto-ingest,
no overlay mutation.

This package is probe-only: it extracts and validates relation candidates but
never writes to accepted memory, accepted overlay, promoted overlay, or the
snapshot dry-run overlay. All candidates require human review before promotion.
"""
