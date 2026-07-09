"""Opt-in step timing for latency profiling.

``step()`` is a near-zero-cost no-op unless a ``capture()`` block is active
on the current context, so instrumenting the request path has no effect on
production latency. Used to get a per-step breakdown for the pipeline in
``answer_orchestrator.py`` / ``api/server.py`` without changing their return
types or public behavior.
"""

from __future__ import annotations

import contextvars
import time
from contextlib import contextmanager

_records: contextvars.ContextVar[list[tuple[str, float]] | None] = contextvars.ContextVar(
    "_perf_records", default=None
)


@contextmanager
def step(label: str):
    records = _records.get()
    if records is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        records.append((label, (time.perf_counter() - start) * 1000.0))


@contextmanager
def capture():
    """Collect (label, elapsed_ms) pairs for every ``step`` inside this block.

    Steps are recorded in call order, nested steps included (an outer step's
    elapsed time includes its inner steps' time), matching how the pipeline
    is actually nested.
    """
    token = _records.set([])
    try:
        yield _records.get()
    finally:
        _records.reset(token)
