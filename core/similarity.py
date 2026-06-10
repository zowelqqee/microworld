from typing import Optional


class SimilarityGraph:
    """
    Undirected pairwise similarity scores between named entities.
    Scores are user-supplied floats in [0, 1].
    Normalizes names via EntityNormalizer if one is provided.
    """

    def __init__(self, normalizer=None) -> None:
        self._scores: dict[frozenset, float] = {}
        self._normalizer = normalizer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_similarity(self, a: str, b: str, score: float = 1.0) -> None:
        """Register similarity between a and b (undirected). Overwrites previous score."""
        a, b = self._norm(a), self._norm(b)
        if a == b:
            return
        self._scores[frozenset({a, b})] = float(score)

    def similarity(self, a: str, b: str) -> float:
        """Return stored score; 1.0 for identical names, 0.0 if unknown pair."""
        a, b = self._norm(a), self._norm(b)
        if a == b:
            return 1.0
        return self._scores.get(frozenset({a, b}), 0.0)

    def most_similar(self, entity: str, candidates: list[str]) -> tuple[str | None, float]:
        """
        Return (best_candidate, score) among candidates.
        Returns (None, 0.0) if no candidate has a positive score.
        """
        entity = self._norm(entity)
        best_name: str | None = None
        best_score = 0.0
        for c in candidates:
            s = self.similarity(entity, c)
            if s > best_score:
                best_score = s
                best_name = c
        return best_name, best_score

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _norm(self, name: str) -> str:
        return self._normalizer.normalize(name) if self._normalizer is not None else name
