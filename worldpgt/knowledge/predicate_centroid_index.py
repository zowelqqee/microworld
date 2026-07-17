"""Phrase-centroid predicate fallback for semantic question parsing.

Complements :mod:`worldpgt.knowledge.relation_embedding_index` (per-verb-lemma
rows) with *whole-phrase* centroids built from the canonical question templates
that already define each predicate in the open-book QA dataset builder.

Why a second embedding view is needed:

- Verb lemmatisation erases grammatical voice.  "By whom was X engineered?"
  reduces to the lemma "engineer", which the keyword map binds to the *active*
  relation ``develops``; the passive question actually asks for the agent of
  ``developed_by``.  Full-phrase templates keep the WH-word and auxiliary
  ("who … developed" vs "what does … develop"), so the centroid carries the
  direction signal that the lemma loses.

- Nominal agents never surface as verbs.  "Which manufacturer made X?" has its
  strongest cue in the noun "manufacturer", which verb extraction cannot see.

- Noun cues in verb position mislead the per-verb index.  "Where does X
  maintain its headquarters?" matched ``founded_by`` through the verb
  "maintain" while the decisive token "headquarters" was discarded.

Architecture constraint (project invariant): **no neural inference at answer
time**.  Centroids are computed once, offline, from the GloVe model
(glove-wiki-gigaword-100) and persisted under ``worldpgt/artifacts``; the
query path encodes the incoming question through the same persisted compact
vocabulary (dict lookup + mean) and compares by cosine — deterministic
arithmetic over static vectors, never a model forward pass.

Matching is deliberately conservative: a candidate must clear an absolute
cosine threshold AND a margin over the runner-up predicate, otherwise the
caller falls through to the audit path.  A false "no match" costs one audit;
a false positive would silently answer the wrong relation.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

_MODEL_NAME = "glove-wiki-gigaword-100"

# ── Representative example phrases per predicate ─────────────────────────────
#
# Sources, in order:
# 1. The canonical question template per predicate from the dataset builder
#    (``worldpgt.benchmarks.open_book_qa.dataset._QUESTIONS``).
# 2. The known paraphrase templates from the same builder (``_PARAPHRASES``).
#    These are the *training-side* known patterns; held-out evaluation
#    templates are intentionally NOT included.
# 3. A few additional canonical formulations per predicate (marked ``extra``)
#    so a centroid is not defined by a single phrase.
#
# The ``{subject}`` placeholder is stripped before embedding: the centroid
# must describe the relation cue, not any entity vocabulary.
PREDICATE_EXAMPLE_PHRASES: dict[str, tuple[str, ...]] = {
    "uses": (
        "What does {subject} use?",
        "What technology does {subject} rely on?",
        "What does {subject} employ?",
        "What is used by {subject}?",
        # extra
        "Which tools does {subject} use?",
    ),
    "enables": (
        "What does {subject} enable?",
        "What does {subject} make possible?",
        "What capability does {subject} provide?",
        # extra
        "What functionality does {subject} enable?",
    ),
    "supports": (
        "What does {subject} support?",
        "What does {subject} help support?",
        "What outcome is supported by {subject}?",
        # extra
        "Which features does {subject} support?",
    ),
    "runs_on": (
        "What does {subject} run on?",
        "What platform does {subject} operate on?",
        # extra
        "On which platform does {subject} run?",
    ),
    "used_for": (
        "What is {subject} used for?",
        "What purpose does {subject} serve?",
        # extra
        "What is the purpose of {subject}?",
    ),
    "works_by": (
        "How does {subject} work?",
        "What mechanism does {subject} use?",
        # extra
        "By what mechanism does {subject} function?",
    ),
    "developed_by": (
        "Who developed {subject}?",
        "Who created {subject}?",
        # extra
        "Which company developed {subject}?",
        "Who was the developer of {subject}?",
    ),
    "founded_by": (
        "Who founded {subject}?",
        "Who established {subject}?",
        # extra
        "Who was the founder of {subject}?",
        "Which people founded {subject}?",
    ),
    "headquartered_in": (
        "Where is {subject} headquartered?",
        "Where does {subject} maintain its headquarters?",
        # extra
        "Where are the headquarters of {subject}?",
        "In which city is {subject} based?",
    ),
    "owned_by": (
        "Who owns {subject}?",
        "Which organisation controls {subject}?",
        # extra
        "Which company is the owner of {subject}?",
    ),
    "parent_company_of": (
        "What subsidiary belongs to {subject}?",
        "Which subsidiary is part of {subject}?",
        # extra
        "Which subsidiaries does {subject} have?",
    ),
    "produces": (
        "What does {subject} produce?",
        "Which products are made by {subject}?",
        # extra
        "What goods does {subject} produce?",
    ),
    "product_of": (
        "Who manufactured {subject}?",
        "Which manufacturer made {subject}?",
        # extra
        "Which company is the manufacturer of {subject}?",
    ),
    "created_by": (
        "Who created {subject}?",
        "Who is the creator of {subject}?",
        # extra
        "Which person created {subject}?",
    ),
    "published_by": (
        "Who published {subject}?",
        "Which publisher issued {subject}?",
        # extra
        "Which company published {subject}?",
    ),
}

# Predicates whose answer slot is the *agent* of the fact (the question names
# the graph subject and asks for the object of an inverse-style relation).
# Structural question shapes that grammatically request an agent ("by whom …",
# "which manufacturer …") may only accept predicates from this set.
AGENT_PREDICATES: frozenset[str] = frozenset({
    "developed_by", "founded_by", "created_by", "published_by",
    "owned_by", "product_of",
})

# Default decision thresholds.  Mean-pooled GloVe vectors of short questions
# live in a tight cone, so absolute cosine is weakly discriminative (observed
# range 0.79–0.98 for both true and false candidates); the *margin* between
# the best and runner-up predicate is the decisive signal.  Calibrated on
# 18 non-relational / out-of-schema probes (all margins ≤ 0.035, all must
# abstain) against the structural paraphrase forms this index exists to catch
# (margins ≥ 0.047); see tests/test_predicate_centroid_index.py.
_DEFAULT_THRESHOLD: float = 0.85
_DEFAULT_MARGIN: float = 0.04

# Tokens dropped before embedding.  Articles and pure fillers only — WH-words
# and auxiliaries are deliberately KEPT because they carry the voice/direction
# signal ("who … developed" vs "what does … develop").
_ARTICLE_STOPWORDS: frozenset[str] = frozenset({"a", "an", "the"})

_NORM_RE = re.compile(r"[^a-z0-9\s-]")
_PLACEHOLDER_RE = re.compile(r"\{subject\}")

# Top-N most-frequent GloVe words included in the compact query vocabulary
# (same budget as relation_embedding_index — covers common English words).
_COMPACT_VOCAB_TOP_N: int = 10_000

_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
_CACHE_VECTORS_PATH = _ARTIFACTS_DIR / "predicate_centroid_cache.npz"
_CACHE_META_PATH = _ARTIFACTS_DIR / "predicate_centroid_meta.json"


def _phrase_table_hash() -> str:
    content = "|".join(
        f"{predicate}:{';'.join(phrases)}"
        for predicate, phrases in sorted(PREDICATE_EXAMPLE_PHRASES.items())
    )
    return hashlib.md5(content.encode()).hexdigest()[:16]


def _tokenise(text: str) -> list[str]:
    cleaned = _NORM_RE.sub(" ", text.lower())
    return [
        token
        for token in re.split(r"[\s\-]+", cleaned)
        if token and token not in _ARTICLE_STOPWORDS
    ]


# ── Singleton GloVe loader (offline build only) ───────────────────────────────

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


class PredicateCentroidIndex:
    """Cosine matcher over per-predicate phrase centroids.

    Build once (offline GloVe pass, cached to ``worldpgt/artifacts``); query
    with plain dictionary lookups over the persisted compact vocabulary.
    """

    def __init__(
        self,
        threshold: float = _DEFAULT_THRESHOLD,
        margin: float = _DEFAULT_MARGIN,
    ) -> None:
        self.threshold = threshold
        self.margin = margin
        self._built = False
        self._build_lock = threading.Lock()
        self._predicates: list[str] = []
        self._centroids: np.ndarray | None = None  # (P, D), L2-normed rows
        self._vocab_words: list[str] = []
        self._vocab_vectors: np.ndarray | None = None
        self._vocab_index: dict[str, int] = {}

    # ── Build / cache ─────────────────────────────────────────────────────────

    def _build(self) -> None:
        model = _get_model()
        dim = model.vector_size

        predicates: list[str] = []
        centroids: list[np.ndarray] = []
        for predicate, phrases in sorted(PREDICATE_EXAMPLE_PHRASES.items()):
            phrase_vectors: list[np.ndarray] = []
            for phrase in phrases:
                tokens = _tokenise(_PLACEHOLDER_RE.sub(" ", phrase))
                vectors = [model[token] for token in tokens if token in model]
                if not vectors:
                    continue
                vector = np.mean(vectors, axis=0)
                norm = np.linalg.norm(vector)
                if norm > 1e-9:
                    phrase_vectors.append(vector / norm)
            if not phrase_vectors:
                continue
            centroid = np.mean(phrase_vectors, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 1e-9:
                predicates.append(predicate)
                centroids.append((centroid / norm).astype(np.float32))

        self._predicates = predicates
        self._centroids = (
            np.stack(centroids, axis=0)
            if centroids
            else np.empty((0, dim), dtype=np.float32)
        )
        self._build_compact_vocab(model, dim)
        self._save_cache()

    def _build_compact_vocab(self, model, dim: int) -> None:
        words: list[str] = []
        vectors: list[np.ndarray] = []
        seen: set[str] = set()

        def _add(word: str) -> None:
            if word not in seen and word in model:
                seen.add(word)
                words.append(word)
                vectors.append(model[word].astype(np.float32))

        for phrases in PREDICATE_EXAMPLE_PHRASES.values():
            for phrase in phrases:
                for token in _tokenise(_PLACEHOLDER_RE.sub(" ", phrase)):
                    _add(token)
        for word in model.index_to_key[:_COMPACT_VOCAB_TOP_N]:
            _add(word)

        self._vocab_words = words
        self._vocab_vectors = (
            np.stack(vectors, axis=0).astype(np.float32)
            if vectors
            else np.empty((0, dim), dtype=np.float32)
        )
        self._vocab_index = {word: i for i, word in enumerate(self._vocab_words)}

    def _try_load_cache(self) -> bool:
        if not (_CACHE_VECTORS_PATH.is_file() and _CACHE_META_PATH.is_file()):
            return False
        try:
            meta = json.loads(_CACHE_META_PATH.read_text(encoding="utf-8"))
            if meta.get("phrase_table_hash") != _phrase_table_hash():
                return False
            arrays = np.load(_CACHE_VECTORS_PATH)
            self._centroids = arrays["centroids"]
            self._vocab_vectors = arrays["vocab_vectors"]
            self._predicates = meta["predicates"]
            self._vocab_words = meta["vocab_words"]
            self._vocab_index = {w: i for i, w in enumerate(self._vocab_words)}
            return True
        except Exception:
            return False

    def _save_cache(self) -> None:
        try:
            _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            assert self._centroids is not None
            assert self._vocab_vectors is not None
            np.savez(
                _CACHE_VECTORS_PATH,
                centroids=self._centroids,
                vocab_vectors=self._vocab_vectors,
            )
            _CACHE_META_PATH.write_text(
                json.dumps(
                    {
                        "phrase_table_hash": _phrase_table_hash(),
                        "model": _MODEL_NAME,
                        "predicates": self._predicates,
                        "vocab_words": self._vocab_words,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass  # cache save failure is non-fatal

    def _ensure_built(self) -> None:
        if not self._built:
            with self._build_lock:
                if not self._built:
                    if not self._try_load_cache():
                        self._build()
                    self._built = True

    # ── Query ─────────────────────────────────────────────────────────────────

    def _encode(self, text: str) -> Optional[np.ndarray]:
        if self._vocab_vectors is None or self._vocab_vectors.shape[0] == 0:
            return None
        vectors = [
            self._vocab_vectors[self._vocab_index[token]]
            for token in _tokenise(text)
            if token in self._vocab_index
        ]
        if not vectors:
            return None
        vector = np.mean(vectors, axis=0)
        norm = np.linalg.norm(vector)
        if norm < 1e-9:
            return None
        return (vector / norm).astype(np.float32)

    def find_predicate(
        self,
        question: str,
        *,
        subject_span: str | None = None,
        allowed: Iterable[str] | None = None,
        threshold: Optional[float] = None,
        margin: Optional[float] = None,
    ) -> tuple[Optional[str], float]:
        """Return ``(predicate, similarity)`` or ``(None, best_rejected_sim)``.

        ``subject_span`` (the extracted entity text) is removed before
        encoding so entity vocabulary cannot pull the vector toward or away
        from any predicate.  ``allowed`` optionally restricts candidates to a
        structurally compatible set (e.g. :data:`AGENT_PREDICATES` for
        passive-agent question shapes); the margin is still computed against
        *all* predicates so an out-of-set near-tie keeps its ambiguity.
        """
        self._ensure_built()
        assert self._centroids is not None
        if self._centroids.shape[0] == 0:
            return None, 0.0

        text = question
        if subject_span:
            text = text.replace(subject_span, " ")
        query = self._encode(text)
        if query is None:
            return None, 0.0

        sims = self._centroids @ query
        order = np.argsort(sims)[::-1]
        allowed_set = frozenset(allowed) if allowed is not None else None

        best_idx = int(order[0])
        best_sim = float(sims[best_idx])
        runner_up = float(sims[int(order[1])]) if len(order) > 1 else 0.0

        cutoff = threshold if threshold is not None else self.threshold
        min_margin = margin if margin is not None else self.margin

        if allowed_set is not None and self._predicates[best_idx] not in allowed_set:
            # The globally best predicate is structurally impossible for this
            # question shape.  Falling back to the best *allowed* one would
            # promote a runner-up the margin check was designed to reject —
            # abstain instead.
            return None, best_sim

        if best_sim >= cutoff and (best_sim - runner_up) >= min_margin:
            return self._predicates[best_idx], best_sim
        return None, best_sim

    def predicate_scores(self, question: str, *, subject_span: str | None = None) -> list[tuple[str, float]]:
        """Diagnostic: every predicate with its cosine, sorted descending."""
        self._ensure_built()
        assert self._centroids is not None
        text = question
        if subject_span:
            text = text.replace(subject_span, " ")
        query = self._encode(text)
        if query is None or self._centroids.shape[0] == 0:
            return []
        sims = self._centroids @ query
        ranked = sorted(zip(self._predicates, sims), key=lambda item: -item[1])
        return [(predicate, float(sim)) for predicate, sim in ranked]


# ── Module-level singleton ────────────────────────────────────────────────────

_default_index: Optional[PredicateCentroidIndex] = None
_index_lock = threading.Lock()


def get_default_centroid_index() -> PredicateCentroidIndex:
    """Return the shared :class:`PredicateCentroidIndex` (built lazily)."""
    global _default_index
    if _default_index is None:
        with _index_lock:
            if _default_index is None:
                _default_index = PredicateCentroidIndex()
    return _default_index
