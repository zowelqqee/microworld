"""Measure how legal retrieval scales, indexed vs the naive full scan.

The graph is replicated N times with distinct provision numbering to simulate a
larger corpus, then the same 60 pre-registered questions are answered against
each size. What matters is the *shape* of the curve: a full scan is linear in
corpus size, an inverted index bounded by the query's selective terms should be
close to flat.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from worldpgt.legal_qa.legal_answer_planner import _toks, retrieve
from worldpgt.legal_qa.legal_index import build_index
from worldpgt.legal_qa.legal_question_analyzer import analyze

HERE = Path(__file__).resolve().parent


def replicate(items: list[dict], factor: int) -> list[dict]:
    """Grow the corpus by cloning it under distinct section numbers."""
    out: list[dict] = []
    for copy in range(factor):
        for item in items:
            if item.get("overlay_type") not in ("overlay_relation", "overlay_definition"):
                continue
            clone = dict(item)
            if copy:
                # A distinct provision number per copy, so citations stay unique
                # and the index cannot collapse copies onto one posting.
                clone["stated_in"] = f"{item.get('stated_in','')} [c{copy}]"
                clone["subject"] = f"{item.get('subject','')} variant {copy}"
            out.append(clone)
    return out


def naive_retrieve(analyzed, items: list[dict]) -> int:
    """The original full-scan cost model: touch every item, tokenize it."""
    focus = _toks(analyzed.focus) or _toks(analyzed.question)
    seen = 0
    for item in items:
        text = " ".join([
            item.get("subject", ""),
            item.get("object", "") or item.get("definition", ""),
            item.get("section_heading", ""),
        ])
        if focus & _toks(text):
            seen += 1
    return seen


def main() -> int:
    base = json.loads((HERE / "legal_overlay.json").read_text())
    questions = json.loads((HERE / "questions.json").read_text())
    analyzed = [analyze(q["question"]) for q in questions]

    rows = []
    for factor in (1, 10, 50, 200):
        items = replicate(base, factor)

        t0 = time.perf_counter()
        index = build_index(items)
        build_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        for a in analyzed:
            retrieve(a, index)
        indexed_ms = (time.perf_counter() - t0) * 1000 / len(analyzed)

        t0 = time.perf_counter()
        for a in analyzed[:12]:
            naive_retrieve(a, items)
        naive_ms = (time.perf_counter() - t0) * 1000 / 12

        rows.append({
            "factor": factor, "items": len(items),
            "index_build_ms": round(build_ms, 1),
            "indexed_ms_per_question": round(indexed_ms, 3),
            "naive_ms_per_question": round(naive_ms, 3),
            "speedup": round(naive_ms / indexed_ms, 1) if indexed_ms else None,
        })
        print(f"items={len(items):>7}  build={build_ms:8.1f}ms  "
              f"indexed={indexed_ms:8.3f}ms/q  naive={naive_ms:9.3f}ms/q  "
              f"speedup={rows[-1]['speedup']}x", flush=True)

    (HERE / "scaling.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    first, last = rows[0], rows[-1]
    growth = last["items"] / first["items"]
    idx_growth = last["indexed_ms_per_question"] / max(first["indexed_ms_per_question"], 1e-9)
    naive_growth = last["naive_ms_per_question"] / max(first["naive_ms_per_question"], 1e-9)
    print(f"\ncorpus x{growth:.0f}  ->  indexed cost x{idx_growth:.1f}, "
          f"naive cost x{naive_growth:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
