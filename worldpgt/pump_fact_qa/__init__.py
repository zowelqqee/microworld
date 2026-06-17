"""Pump Fact QA Benchmark v1.

A deterministic, local-only benchmark that proves precision-filtered pump facts
are actually answerable through the assistant surface (``--overlay pump-dry-run``)
and that the surface still refuses false / reversed / current-live answers.

No network, no neural code, no training. It never mutates accepted memory,
accepted/promoted overlays, or the snapshot dry-run overlay. The pump overlay
stays a proposal/dry-run, not accepted memory.
"""
