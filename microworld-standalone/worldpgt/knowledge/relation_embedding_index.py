"""Embedding-based relation intent fallback for semantic question parsing.

When exact keyword matching in ``relation_intent_from_text`` returns None,
this module provides a cosine-similarity fallback over ``RELATION_KEYWORD_MAP``
using GloVe word embeddings (glove-wiki-gigaword-100, ~134 MB, loaded once).

Design:
- ``is_a`` and ``type_of`` are excluded from the embedding index.  Those
  question types are caught 100% by ``_IS_A_RE`` in ``parse_semantic_query``
  before the embedding fallback fires; their seed words ("is a", "is an",
  "type of", "kind of") contain function words that sit in a dense central
  region of the GloVe space and inflate similarity scores for every other
  relation, producing false positives.

- Matrix rows are individual *content words* extracted from each keyword
  phrase, not full-phrase means.  Stopwords (``by``, ``of``, ``the``,
  ``a``, ``an``, ``is``) and hyphen-split compound tokens are filtered so
  each row carries a single semantic concept.  Hyphenated compounds (e.g.
  "co-established") are kept intact and their GloVe vector is computed as
  the mean of their constituent parts when the full token is OOV.

- Query phrases are still encoded as the mean of their constituent word
  vectors (OOV tokens skipped; empty result → abstain).  Comparing a
  single-content-word query against single-content-word matrix rows avoids
  the signal dilution that plagued the previous phrase-mean approach.

- ``find_relation_intent`` uses a *per-relation max* aggregation: for each
  candidate phrase, it computes similarity against every keyword vector in the
  matrix, then takes the *maximum* similarity within each relation group.  The
  best-relation max is compared against the second-best-relation max; a *margin
  check* (≥ 0.05) rejects ambiguous phrases where two relations are nearly
  equidistant.  The system prefers to abstain (→ audit) rather than guess the
  wrong relation.

- ``top_matches`` retains the original per-keyword view for debugging.

Thresholds (instance defaults, overridable at call site):
- Cosine similarity: 0.65 (conservative)
- Margin: 0.05 (minimum gap between best-relation and runner-up relation)
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from worldpgt.relation_extraction_v2.relation_policy import RELATION_KEYWORD_MAP

_MODEL_NAME = "glove-wiki-gigaword-100"

# Relations excluded from the embedding index — handled by dedicated regex
# patterns upstream, and their seed words pollute the GloVe similarity space.
_EXCLUDED_EMBEDDING_RELATIONS: frozenset[str] = frozenset({"is_a", "type_of"})

# Minimum margin (best-relation sim − second-best-relation sim).
# Phrases where two relations are nearly equidistant → abstain rather than guess.
_MARGIN_THRESHOLD: float = 0.05

_NORM_RE = re.compile(r"[^a-z0-9\s-]")

# Stopwords filtered from keyword phrases when building the content-word matrix.
# Includes articles/prepositions AND common phrasal-verb particles (up, off, to,
# in, on, out).  Particles carry no standalone semantic content and cause
# cross-relation margin collapse when the same particle appears in keywords for
# different relations (e.g. "up" in "set up" → founded_by and "heads up" →
# leader_of).
_EMBEDDING_STOPWORDS: frozenset[str] = frozenset({
    "by", "of", "the", "a", "an", "is",  # articles / prepositions
    "up", "off", "to", "in", "on", "out",  # phrasal-verb particles
})

# ── Cache paths ───────────────────────────────────────────────────────────────

_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
_CACHE_MATRIX_PATH = _ARTIFACTS_DIR / "embedding_matrix_cache.npy"
_CACHE_VOCAB_PATH  = _ARTIFACTS_DIR / "embedding_vocab_cache.npy"
_CACHE_META_PATH   = _ARTIFACTS_DIR / "embedding_labels_cache.json"

# Top-N most-frequent GloVe words included in compact query vocabulary.
# 10 K covers essentially all common English verbs and nouns.
_COMPACT_VOCAB_TOP_N: int = 10_000


def _map_hash() -> str:
    """Short MD5 of RELATION_KEYWORD_MAP — used to invalidate stale caches."""
    content = "|".join(f"{k}:{v}" for k, v in sorted(RELATION_KEYWORD_MAP.items()))
    return hashlib.md5(content.encode()).hexdigest()[:16]


def _normalise(phrase: str) -> str:
    return _NORM_RE.sub("", phrase.lower().strip())


def _tokenise(phrase: str) -> list[str]:
    """Tokenise for query-phrase encoding: split on whitespace AND hyphens."""
    return [t for t in re.split(r"[\s\-]+", _normalise(phrase)) if t]


def _content_words(keyword: str) -> list[str]:
    """Return content words from a keyword phrase for matrix construction.

    Splits on whitespace only (preserves hyphenated compounds) then filters
    stopwords.  E.g. "founded by" → ["founded"], "co-established" → ["co-established"],
    "is a" → [] (all stopwords, caller skips).
    """
    return [w for w in keyword.lower().strip().split() if w not in _EMBEDDING_STOPWORDS]


# ── Singleton model loader ─────────────────────────────────────────────────────

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                import gensim.downloader as api
                _model = api.load(_MODEL_NAME)
    return _model


# ── Core class ────────────────────────────────────────────────────────────────

class RelationEmbeddingIndex:
    """Per-relation max cosine-similarity index over RELATION_KEYWORD_MAP.

    Build once and share as a module-level singleton — building is O(vocab)
    but cheap after the GloVe model is cached.

    Parameters
    ----------
    threshold:
        Minimum cosine similarity to accept a match (default 0.65).
    margin:
        Minimum gap between best-relation and runner-up relation similarity
        (default 0.05).  Ambiguous phrases (small margin) → abstain.
    """

    def __init__(
        self,
        threshold: float = 0.65,
        margin: float = _MARGIN_THRESHOLD,
    ) -> None:
        self.threshold = threshold
        self.margin = margin
        self._built = False
        self._build_lock = threading.Lock()
        self._keyword_labels: list[str] = []    # keyword strings
        self._relation_labels: list[str] = []   # corresponding relation types
        self._matrix: np.ndarray | None = None  # (N_keywords, D), L2-normed
        # Compact query vocabulary — avoids loading full GloVe at query time.
        self._vocab_words: list[str] = []
        self._vocab_vectors: np.ndarray | None = None  # (K, D) float32
        self._vocab_index: dict[str, int] = {}

    # ── Private: build ────────────────────────────────────────────────────────

    def _build(self) -> None:
        model = _get_model()
        dim = model.vector_size

        keywords: list[str] = []
        relations: list[str] = []
        vectors: list[np.ndarray] = []
        seen: set[tuple[str, str]] = set()  # dedup (content_word, relation)

        for keyword, relation in RELATION_KEYWORD_MAP.items():
            if relation in _EXCLUDED_EMBEDDING_RELATIONS:
                continue
            for word in _content_words(keyword):
                pair = (word, relation)
                if pair in seen:
                    continue
                seen.add(pair)
                vec = self._content_word_vector(word, model, dim)
                if vec is None:
                    continue
                keywords.append(word)
                relations.append(relation)
                vectors.append(vec)

        self._keyword_labels = keywords
        self._relation_labels = relations
        if vectors:
            mat = np.stack(vectors, axis=0).astype(np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms = np.where(norms < 1e-9, 1.0, norms)
            self._matrix = mat / norms
        else:
            self._matrix = np.empty((0, dim), dtype=np.float32)

        self._build_compact_vocab(model, dim)
        self._save_cache()

    @staticmethod
    def _content_word_vector(word: str, model, dim: int) -> Optional[np.ndarray]:
        """Return GloVe vector for a content word.

        For hyphenated compounds (e.g. "co-established"), averages constituent
        parts' vectors when the full token is OOV.
        """
        if word in model:
            return model[word].astype(np.float32)
        if "-" in word:
            parts = word.split("-")
            vecs = [model[p] for p in parts if p in model]
            if vecs:
                return np.mean(vecs, axis=0).astype(np.float32)
        return None

    @staticmethod
    def _phrase_vector(phrase: str, model, dim: int) -> Optional[np.ndarray]:
        """Encode a query phrase: mean of all constituent word vectors."""
        tokens = _tokenise(phrase)
        vecs = [model[t] for t in tokens if t in model]
        if not vecs:
            return None
        return np.mean(vecs, axis=0).astype(np.float32)

    # ── Cache I/O ─────────────────────────────────────────────────────────────

    def _try_load_cache(self) -> bool:
        """Load pre-built matrix and compact vocab from disk. Returns True on success."""
        if not (_CACHE_MATRIX_PATH.is_file() and _CACHE_VOCAB_PATH.is_file()
                and _CACHE_META_PATH.is_file()):
            return False
        try:
            meta = json.loads(_CACHE_META_PATH.read_text(encoding="utf-8"))
            if meta.get("map_hash") != _map_hash():
                return False
            self._matrix = np.load(_CACHE_MATRIX_PATH)
            self._vocab_vectors = np.load(_CACHE_VOCAB_PATH)
            self._keyword_labels = meta["keyword_labels"]
            self._relation_labels = meta["relation_labels"]
            self._vocab_words = meta["vocab_words"]
            self._vocab_index = {w: i for i, w in enumerate(self._vocab_words)}
            return True
        except Exception:
            return False

    def _save_cache(self) -> None:
        """Persist matrix and compact vocab to disk for fast future loads."""
        try:
            _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            assert self._matrix is not None
            assert self._vocab_vectors is not None
            np.save(_CACHE_MATRIX_PATH, self._matrix)
            np.save(_CACHE_VOCAB_PATH, self._vocab_vectors)
            meta = {
                "map_hash": _map_hash(),
                "keyword_labels": self._keyword_labels,
                "relation_labels": self._relation_labels,
                "vocab_words": self._vocab_words,
            }
            _CACHE_META_PATH.write_text(
                json.dumps(meta, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass  # cache save failure is non-fatal

    def _build_compact_vocab(self, model, dim: int) -> None:
        """Build compact query vocabulary: keyword tokens + top-N GloVe words."""
        words: list[str] = []
        vectors: list[np.ndarray] = []
        seen: set[str] = set()

        def _add(w: str) -> None:
            if w not in seen and w in model:
                seen.add(w)
                words.append(w)
                vectors.append(model[w].astype(np.float32))

        # All tokenised forms of every keyword phrase
        for keyword in RELATION_KEYWORD_MAP:
            for w in _tokenise(keyword):
                _add(w)

        # Top-N most-frequent GloVe words (covers common verbs like "steer", "helm")
        for w in model.index_to_key[:_COMPACT_VOCAB_TOP_N]:
            _add(w)

        self._vocab_words = words
        self._vocab_vectors = (
            np.stack(vectors, axis=0).astype(np.float32)
            if vectors
            else np.empty((0, dim), dtype=np.float32)
        )
        self._vocab_index = {w: i for i, w in enumerate(self._vocab_words)}

    def _phrase_vector_compact(self, phrase: str) -> Optional[np.ndarray]:
        """Encode a query phrase using the compact vocabulary — no GloVe load needed."""
        if self._vocab_vectors is None or self._vocab_vectors.shape[0] == 0:
            return None
        tokens = _tokenise(phrase)
        vecs = []
        for t in tokens:
            idx = self._vocab_index.get(t)
            if idx is not None:
                vecs.append(self._vocab_vectors[idx])
        if not vecs:
            return None
        return np.mean(vecs, axis=0).astype(np.float32)

    def _ensure_built(self) -> None:
        if not self._built:
            with self._build_lock:
                if not self._built:
                    if not self._try_load_cache():
                        self._build()
                    self._built = True

    # ── Private: per-relation max ─────────────────────────────────────────────

    def _per_relation_max(
        self,
        unit: np.ndarray,
    ) -> list[tuple[str, float]]:
        """Return ``[(relation, max_sim)]`` sorted by max_sim descending.

        For each relation, takes the *maximum* keyword similarity rather than
        comparing against a centroid.  This preserves the precision of
        individual keyword matching while enabling inter-relation margin checks.
        """
        assert self._matrix is not None
        sims = self._matrix @ unit  # (N_keywords,)
        per_rel: dict[str, float] = {}
        for i, rel in enumerate(self._relation_labels):
            s = float(sims[i])
            if rel not in per_rel or s > per_rel[rel]:
                per_rel[rel] = s
        return sorted(per_rel.items(), key=lambda x: -x[1])

    # ── Public API ────────────────────────────────────────────────────────────

    def find_relation_intent(
        self,
        verb_phrases: list[str],
        threshold: Optional[float] = None,
    ) -> tuple[Optional[str], float]:
        """Return ``(relation_type, similarity)`` or ``(None, rejected_sim)``.

        Tries each candidate in *verb_phrases*, returning the best match whose
        per-relation max similarity both exceeds *threshold* AND has a margin
        ≥ ``self.margin`` over the second-best relation.

        Parameters
        ----------
        verb_phrases:
            One or more verb/phrase candidates extracted from the question.
        threshold:
            Override the instance threshold for this call.

        Returns
        -------
        (relation, sim) on success; (None, best_rejected_sim) on abstain.
        ``best_rejected_sim`` is the highest per-relation max seen across all
        phrases (useful for threshold calibration), even if the match was
        rejected by the margin check.
        """
        if not verb_phrases:
            return None, 0.0

        self._ensure_built()
        assert self._matrix is not None

        if self._matrix.shape[0] == 0:
            return None, 0.0

        cutoff = threshold if threshold is not None else self.threshold

        best_relation: Optional[str] = None
        best_sim: float = 0.0          # best that passed BOTH checks
        best_rejected_sim: float = 0.0  # best seen regardless of checks

        for phrase in verb_phrases:
            vec = self._phrase_vector_compact(phrase)
            if vec is None:
                continue
            norm = np.linalg.norm(vec)
            if norm < 1e-9:
                continue
            unit = (vec / norm).astype(np.float32)

            ranked = self._per_relation_max(unit)
            rel1, sim1 = ranked[0]
            sim2 = ranked[1][1] if len(ranked) > 1 else 0.0
            margin = sim1 - sim2

            if sim1 > best_rejected_sim:
                best_rejected_sim = sim1

            if sim1 > best_sim and sim1 >= cutoff and margin >= self.margin:
                best_sim = sim1
                best_relation = rel1

        if best_relation is not None:
            return best_relation, best_sim
        return None, best_rejected_sim

    def top_matches(
        self,
        verb_phrase: str,
        n: int = 5,
    ) -> list[tuple[str, str, float]]:
        """Return top-*n* ``(keyword, relation_type, similarity)`` triples.

        Uses the keyword-level matrix for granular debugging.
        """
        self._ensure_built()
        assert self._matrix is not None

        vec = self._phrase_vector_compact(verb_phrase)
        if vec is None or self._matrix.shape[0] == 0:
            return []

        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            return []
        unit = (vec / norm).astype(np.float32)
        sims = self._matrix @ unit
        top_idx = np.argsort(sims)[::-1][:n]
        return [
            (self._keyword_labels[i], self._relation_labels[i], float(sims[i]))
            for i in top_idx
        ]

    def relation_scores(
        self,
        verb_phrase: str,
    ) -> list[tuple[str, float]]:
        """Return ``[(relation, per_relation_max_sim)]`` sorted descending.

        Diagnostic helper showing the per-relation max similarity for a phrase.
        Includes the margin between top-1 and top-2.
        """
        self._ensure_built()
        assert self._matrix is not None

        vec = self._phrase_vector_compact(verb_phrase)
        if vec is None or self._matrix.shape[0] == 0:
            return []
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            return []
        unit = (vec / norm).astype(np.float32)
        return self._per_relation_max(unit)


# ── Module-level singleton ────────────────────────────────────────────────────

_default_index: Optional[RelationEmbeddingIndex] = None
_index_lock = threading.Lock()


def get_default_index() -> RelationEmbeddingIndex:
    """Return the shared ``RelationEmbeddingIndex`` (built lazily on first call)."""
    global _default_index
    if _default_index is None:
        with _index_lock:
            if _default_index is None:
                _default_index = RelationEmbeddingIndex()
    return _default_index
