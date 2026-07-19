#!/usr/bin/env python3
"""Branch router pilot (throwaway) — retrieval/centroid dispatch, NO training.

Reuses the proven predicate-centroid mechanism verbatim: GloVe glove-wiki-
gigaword-100 (static vectors, dict lookup + mean — not a model forward pass),
L2-normed phrase centroids, and the same margin-gated decision (absolute cosine
+ margin over runner-up). Only the example set changes: branch intents instead of
predicates.

Dispatch order (per the fixed architecture):
  1. Fast-path structural rules — the SAME explicit markers the branches already
     use (reflective what-if/why-might regexes; constrained-creative "using only
     these facts"). If matched, done — no embedding needed.
  2. Semantic centroid fallback — margin-gated cosine over branch centroids.
  3. Below threshold -> default to QA (conservative, self-auditing).

Safety/private/current-sensitive screen is assumed to run BEFORE this, unchanged.

Imports production code read-only; modifies nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# Reuse the proven infrastructure read-only.
from worldpgt.knowledge.predicate_centroid_index import _get_model, _tokenise
from worldpgt.reasoning.reflective_reasoning_v1 import _WHATIF_RE, _WHYMIGHT_RE

BRANCHES = ("qa", "reflective", "constrained_creative", "pure_creative")

# 10–15 representative examples per branch, all from existing material
# (QA question templates, today's reflective pilot forms, today's constrained
# build prompt, Creative-mode demo prompts).
BRANCH_EXAMPLES: dict[str, tuple[str, ...]] = {
    "qa": (
        "Who founded SpaceX?",
        "Where is Tesla headquartered?",
        "What does SpaceX produce?",
        "Who developed the Falcon 9?",
        "What is Neuralink used for?",
        "Who owns Louis Vuitton?",
        "Which company created this product?",
        "What does Blue Origin develop?",
        "Who published this book?",
        "Where is the company located?",
        "Which manufacturer made this vehicle?",
        "What technology does the system use?",
    ),
    "reflective": (
        "What if SpaceX had not been founded?",
        "What would happen if Tesla stopped making cars?",
        "Why might Musk be associated with rockets?",
        "Why might two companies be related?",
        "Suppose the founder had never created the company",
        "What if the leader had not led the firm?",
        "Why might one subject be connected to another?",
        "What might link these two subjects?",
        "How might these entities be related?",
        "What would the company be without its founder?",
    ),
    "constrained_creative": (
        "Write a short piece about SpaceX using only these facts",
        "Compose a paragraph about Tesla using exactly these facts",
        "Write about the company limited to the following facts",
        "Using only these facts, write about the subject",
        "Create a description from just these listed facts",
        "Write a grounded summary using only the given facts",
        "Compose using nothing but these listed facts",
        "Write about the topic staying strictly within these facts",
    ),
    "pure_creative": (
        "Compose a poem about rockets",
        "Write a story about a rocket",
        "Write a poem about autumn",
        "Write in the style of Pushkin",
        "Write a poem about space using classical imagery",
        "Write something imaginative about the ocean",
        "Compose a creative verse about time",
        "Tell a fictional tale about the stars",
        "Write a creative piece inspired by the sea",
        "Invent a short poem about winter",
    ),
}

# Fast-path: constrained-creative explicit grounding phrasing.
_CONSTRAINED_RE = re.compile(
    r"using (only|exactly|just|nothing but|strictly) (these|the following|the given)?\s*facts",
    re.IGNORECASE,
)

_DEFAULT_THRESHOLD = 0.85
_DEFAULT_MARGIN = 0.04


@dataclass
class RouteResult:
    branch: str
    method: str          # "fast_path" | "centroid" | "default_qa"
    similarity: float
    margin: float
    detail: str = ""


class BranchRouter:
    def __init__(self, threshold: float = _DEFAULT_THRESHOLD, margin: float = _DEFAULT_MARGIN):
        self.threshold = threshold
        self.margin = margin
        self._branches: list[str] = []
        self._centroids: np.ndarray | None = None
        self._built = False

    def _phrase_vec(self, model, phrase: str):
        vecs = [model[t] for t in _tokenise(phrase) if t in model]
        if not vecs:
            return None
        v = np.mean(vecs, axis=0)
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else None

    def build(self):
        model = _get_model()
        branches, centroids = [], []
        for branch in BRANCHES:
            pvs = [pv for p in BRANCH_EXAMPLES[branch] if (pv := self._phrase_vec(model, p)) is not None]
            if not pvs:
                continue
            c = np.mean(pvs, axis=0)
            n = np.linalg.norm(c)
            if n > 1e-9:
                branches.append(branch)
                centroids.append((c / n).astype(np.float32))
        self._branches = branches
        self._centroids = np.stack(centroids, axis=0)
        self._built = True

    def _encode(self, text: str):
        model = _get_model()
        return self._phrase_vec(model, text)

    def _fast_path(self, question: str) -> str | None:
        q = question or ""
        if _WHATIF_RE.match(q) or _WHYMIGHT_RE.match(q):
            return "reflective"
        if _CONSTRAINED_RE.search(q):
            return "constrained_creative"
        return None

    def route(self, question: str) -> RouteResult:
        if not self._built:
            self.build()
        fp = self._fast_path(question)
        if fp:
            return RouteResult(fp, "fast_path", 1.0, 1.0, "explicit marker")
        query = self._encode(question)
        if query is None:
            return RouteResult("qa", "default_qa", 0.0, 0.0, "no encodable tokens")
        sims = self._centroids @ query
        order = np.argsort(sims)[::-1]
        best_i = int(order[0])
        best = float(sims[best_i])
        runner = float(sims[int(order[1])]) if len(order) > 1 else 0.0
        margin = best - runner
        if best >= self.threshold and margin >= self.margin:
            return RouteResult(self._branches[best_i], "centroid", best, margin,
                               f"runner-up {self._branches[int(order[1])]}")
        return RouteResult("qa", "default_qa", best, margin, "below threshold/margin")

    def scores(self, question: str) -> list[tuple[str, float]]:
        if not self._built:
            self.build()
        q = self._encode(question)
        if q is None:
            return []
        sims = self._centroids @ q
        return sorted(zip(self._branches, (float(s) for s in sims)), key=lambda x: -x[1])
