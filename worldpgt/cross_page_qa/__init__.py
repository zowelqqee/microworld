"""Cross-page Entity QA v1.

Controlled, deterministic graph-style QA over the isolated 50-page wiki memory
overlay. Answers only when a safe explicit path / source-qualified fact / weak-
link policy applies; otherwise audits.

Rule-based only. No ML. No embeddings. No network access. The overlay is NOT
general runtime memory; ``safe_for_general_runtime`` is always False.
"""
