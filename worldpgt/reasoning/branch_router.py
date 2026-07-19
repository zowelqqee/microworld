"""Branch router — retrieval/centroid dispatch, no training, no neural inference.

Production port of the routing_v1 pilot (see
``artifacts/routing_v1/design_and_pilot_report.md``). Reuses the proven
predicate-centroid mechanism: GloVe static vectors (dict lookup + mean — not a
model forward pass), L2-normed per-branch centroids, margin-gated cosine.

Calibrated on the pilot: the branch centroids sit in a tighter cone than the
predicate centroids, so the decisive margin is 0.02 (not the predicate default
0.04). Below margin -> the caller defaults to QA (conservative).

Dispatch precedence (applied by the integration layer, not here): explicit
fast-path markers first, then this centroid fallback, then QA default.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from worldpgt.knowledge.predicate_centroid_index import _get_model, _tokenise
from worldpgt.reasoning.reflective_reasoning_v1 import _WHATIF_RE, _WHYMIGHT_RE

BRANCHES = ("qa", "reflective", "constrained_creative", "pure_creative")

# Calibrated on the routing_v1 pilot (0% clear misroute, 7.7% ambiguous).
CALIBRATED_THRESHOLD = 0.85
CALIBRATED_MARGIN = 0.02

# Representative example phrases per branch (from existing material: QA templates,
# reflective pilot forms, constrained-creative build prompt, Creative-mode demos).
BRANCH_EXAMPLES: dict[str, tuple[str, ...]] = {
    "qa": (
        "Who founded SpaceX?", "Where is Tesla headquartered?", "What does SpaceX produce?",
        "Who developed the Falcon 9?", "What is Neuralink used for?", "Who owns Louis Vuitton?",
        "Which company created this product?", "What does Blue Origin develop?",
        "Who published this book?", "Where is the company located?",
        "Which manufacturer made this vehicle?", "What technology does the system use?",
    ),
    "reflective": (
        "What if SpaceX had not been founded?", "What would happen if Tesla stopped making cars?",
        "Why might Musk be associated with rockets?", "Why might two companies be related?",
        "Suppose the founder had never created the company", "What if the leader had not led the firm?",
        "Why might one subject be connected to another?", "What might link these two subjects?",
        "How might these entities be related?", "What would the company be without its founder?",
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
        "Compose a poem about rockets", "Write a story about a rocket", "Write a poem about autumn",
        "Write in the style of Pushkin", "Write a poem about space using classical imagery",
        "Write something imaginative about the ocean", "Compose a creative verse about time",
        "Tell a fictional tale about the stars", "Write a creative piece inspired by the sea",
        "Invent a short poem about winter",
    ),
}

CONSTRAINED_MARKER_RE = re.compile(
    r"using (only|exactly|just|nothing but|strictly) (these|the following|the given)?\s*facts",
    re.IGNORECASE,
)

# This intentionally recognizes only an unambiguous request for an invented
# literary form.  It does not treat ordinary "write about X" factual requests
# as creative, and is evaluated only after the hard-safety screen in the
# integration layer.
PURE_CREATIVE_MARKER_RE = re.compile(
    r"\b(?:compose|write|tell|invent|create)\b.*\b(?:poem|story|fiction|fictional|"
    r"tale|verse|creative|imaginative)\b|\b(?:poem|story|fiction|fictional|"
    r"tale|verse|creative|imaginative)\b.*\b(?:about|inspired by)\b",
    re.IGNORECASE,
)


@dataclass
class RouteResult:
    branch: str
    method: str          # "fast_path" | "centroid" | "default_qa"
    similarity: float
    margin: float
    detail: str = ""


class BranchRouter:
    def __init__(self, threshold: float = CALIBRATED_THRESHOLD, margin: float = CALIBRATED_MARGIN):
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

    def build(self) -> "BranchRouter":
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
        return self

    def _ensure_built(self):
        if not self._built:
            self.build()

    def fast_path(self, question: str) -> str | None:
        q = question or ""
        if _WHATIF_RE.match(q) or _WHYMIGHT_RE.match(q):
            return "reflective"
        if CONSTRAINED_MARKER_RE.search(q):
            return "constrained_creative"
        if PURE_CREATIVE_MARKER_RE.search(q):
            return "pure_creative"
        return None

    def route(self, question: str) -> RouteResult:
        self._ensure_built()
        fp = self.fast_path(question)
        if fp:
            return RouteResult(fp, "fast_path", 1.0, 1.0, "explicit marker")
        model = _get_model()
        query = self._phrase_vec(model, question)
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
