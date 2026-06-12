"""
Microworld-style name/surname generator: explicit character-transition graph.

A makemore-like character-level name generator implemented as an explicit,
inspectable graph of character transitions instead of a neural network.  There
are no weights and no backpropagation: generation is a weighted random walk over
counted transitions, and feedback is applied as a compact per-transition trust
multiplier (see examples/surname_trust_learn.py).

The generator is agnostic about name type — it works equally well with given
names, family names, or mixed personal-name datasets (one name per line).

Special tokens:
  START = "<START>"   padding before the first character
  END   = "<END>"     emitted to terminate a name

For an n-gram model of ``order`` n the context is the tuple of the previous n
emitted symbols, START-padded at the beginning of a name.  Transition keys are
stable strings of the form ``"<context>-><next_char>"`` (e.g. "ab->r"), which is
the same key format consumed by the trust profile.

Length-aware generation:
  ``generate()`` accepts ``min_length``, ``soft_max_length``, ``end_bias`` and
  ``length_end_bias`` to control how naturally names end.  Before ``min_length``
  the END token is suppressed; after ``soft_max_length`` it is boosted
  exponentially with each extra character.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

START = "<START>"
END = "<END>"

# Characters kept during normalization in addition to (unicode) alphabetic ones.
_ALLOWED_PUNCT = frozenset("'-")


# ── normalization / loading ───────────────────────────────────────────────────

def normalize_surname(raw: str, *, lowercase: bool = True) -> str:
    """Normalize a raw personal name string (given name, surname, or mixed).

    Strips whitespace, optionally lowercases, and keeps only (unicode) alphabetic
    characters plus apostrophe and hyphen.  Other characters are dropped.  Returns
    the cleaned string (which may be empty / too short — filtering is the caller's
    job).
    """
    s = raw.strip()
    if lowercase:
        s = s.lower()
    return "".join(c for c in s if c.isalpha() or c in _ALLOWED_PUNCT)


def load_surnames(
    path: str,
    *,
    column: int | None = None,
    lowercase: bool = True,
    min_length: int = 2,
    limit: int | None = None,
) -> list[str]:
    """Load and normalize personal names from a text or CSV file.

    Accepts one-name-per-line text files and CSV files alike — the dataset may
    contain given names, family names, or any mix of personal names.  A plain
    text file is handled as a single-column CSV, so the same code path covers
    both formats.  For CSV input the first non-empty column is used unless
    ``column`` (0-based) is given.

    Empty rows, rows whose chosen cell normalizes to empty, and names shorter than
    ``min_length`` characters after cleaning are skipped.
    """
    out: list[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            cell = _select_cell(row, column)
            if cell is None:
                continue
            name = normalize_surname(cell, lowercase=lowercase)
            if len(name) < min_length:
                continue
            out.append(name)
            if limit is not None and len(out) >= limit:
                break
    return out


def _select_cell(row: list[str], column: int | None) -> str | None:
    """Pick the relevant cell from a CSV row, or None if nothing usable."""
    if not row:
        return None
    if column is not None:
        if column < len(row) and row[column].strip():
            return row[column]
        return None
    for cell in row:
        if cell.strip():
            return cell
    return None


# ── transition keys ───────────────────────────────────────────────────────────

def context_to_str(context: tuple[str, ...]) -> str:
    """Render a context tuple to its stable string form (START tokens included)."""
    return "".join(context)


def transition_key(context: tuple[str, ...], next_char: str) -> str:
    """Stable trust-key string for a transition, e.g. ('a','b') + 'r' → 'ab->r'."""
    return f"{context_to_str(context)}->{next_char}"


def iter_transitions(name: str, order: int):
    """Yield ``(context, next_char)`` pairs for *name* at the given order.

    The sequence is START-padded and terminated with END, matching how the graph
    is built and walked, so it can be used to reconstruct the transitions a
    generated name went through (for trust learning).
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    context = (START,) * order
    for ch in list(name) + [END]:
        yield context, ch
        context = context[1:] + (ch,)


# ── trust profile ─────────────────────────────────────────────────────────────

def _get_transition_trust(trust_profile) -> dict[str, float]:
    """Extract the transition_trust mapping from a dict or object, or {}."""
    if trust_profile is None:
        return {}
    if isinstance(trust_profile, dict):
        return trust_profile.get("transition_trust", {}) or {}
    return getattr(trust_profile, "transition_trust", {}) or {}


def load_trust_profile(path: str) -> dict:
    """Load a name-generation trust profile JSON written by examples/surname_trust_learn.py."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── transition graph ──────────────────────────────────────────────────────────

class SurnameTransitionGraph:
    """Explicit character-transition graph for name/surname generation.

    Counts ``context -> next_char`` transitions over a corpus of personal names
    (given names, surnames, or mixed) and generates new names by a weighted
    random walk.  Optional per-transition trust multipliers bias the walk
    without changing the underlying counts.
    """

    def __init__(self, order: int = 1) -> None:
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        # context tuple -> {next_char/END -> count}
        self.transition_counts: dict[tuple[str, ...], dict[str, int]] = {}

    # -- construction --------------------------------------------------------

    def add_name(self, name: str) -> None:
        """Add the transitions of a single (already normalized) name."""
        for context, ch in iter_transitions(name, self.order):
            bucket = self.transition_counts.setdefault(context, {})
            bucket[ch] = bucket.get(ch, 0) + 1

    def build(self, names) -> "SurnameTransitionGraph":
        """Add every name in *names*; returns self for chaining."""
        for name in names:
            self.add_name(name)
        return self

    # -- probabilities -------------------------------------------------------

    def transition_probs(self, context: tuple[str, ...]) -> dict[str, float]:
        """Return the base probability distribution over next symbols for *context*.

        Probabilities sum to 1.0 for a known context, or {} for an unseen one.
        """
        counts = self.transition_counts.get(tuple(context))
        if not counts:
            return {}
        total = sum(counts.values())
        return {ch: c / total for ch, c in counts.items()}

    # -- walking -------------------------------------------------------------

    def next_char(
        self,
        context: tuple[str, ...],
        rng: random.Random,
        trust_profile=None,
        *,
        current_length: int = 0,
        min_length: int = 3,
        soft_max_length: int = 12,
        end_bias: float = 1.0,
        length_end_bias: float = 1.35,
    ) -> str:
        """Sample the next symbol for *context* (may be END).

        Weights are ``base_probability * trust_multiplier``; the multiplier
        defaults to 1.0 and comes from ``trust_profile``.  If every adjusted
        weight is zero, the walk falls back to the base probabilities.  Returns
        END when the context has no outgoing transitions.

        Length-aware END adjustment (applied after trust):
          - current_length < min_length: END weight is zeroed (suppressed).
            If no other transitions exist, END is returned anyway to avoid
            an infinite loop.
          - current_length >= soft_max_length: END weight is multiplied by
            ``length_end_bias ** (current_length - soft_max_length + 1)``.
          - otherwise: END weight is multiplied by ``end_bias`` (default 1.0
            = no change).
        """
        probs = self.transition_probs(context)
        if not probs:
            return END

        items = sorted(probs.items())  # deterministic ordering for seeded runs
        chars = [ch for ch, _ in items]
        base = [p for _, p in items]

        tt = _get_transition_trust(trust_profile)
        if tt:
            weights = [p * tt.get(transition_key(tuple(context), ch), 1.0)
                       for ch, p in items]
            if sum(weights) <= 0.0:
                weights = list(base)
        else:
            weights = list(base)

        # Apply length-aware END bias after trust adjustment.
        if END in chars:
            end_idx = chars.index(END)
            if current_length < min_length:
                weights[end_idx] = 0.0
                if sum(weights) <= 0.0:
                    # Only END was available; must terminate despite min_length.
                    return END
            elif current_length >= soft_max_length:
                boost = length_end_bias ** (current_length - soft_max_length + 1)
                weights[end_idx] *= boost
            else:
                weights[end_idx] *= end_bias

        return _weighted_choice(chars, weights, rng)

    def generate(
        self,
        max_length: int = 20,
        rng: random.Random | None = None,
        trust_profile=None,
        policy=None,
        *,
        min_length: int = 3,
        soft_max_length: int = 12,
        end_bias: float = 1.0,
        length_end_bias: float = 1.35,
    ) -> str:
        """Generate one name by a weighted graph walk.

        Stops at END or once ``max_length`` characters have been emitted.  The
        optional ``policy`` callable, if given, is applied to the final string and
        must return the (possibly unchanged) name to return; it is a hook for
        post-processing and does not affect the walk itself.

        Parameters
        ----------
        min_length:
            END is suppressed (weight set to 0) while the current length is
            below this value.  Default 3.
        soft_max_length:
            After this length, END's weight is boosted by
            ``length_end_bias ** (current_length - soft_max_length + 1)`` at
            each step, making termination increasingly likely.  Default 12.
        end_bias:
            Flat multiplier on END's weight in the normal range
            [min_length, soft_max_length).  1.0 = no change.
        length_end_bias:
            Per-step exponential boost applied to END once the name exceeds
            ``soft_max_length``.  Values > 1 push names toward shorter output.
            Default 1.35.
        """
        if rng is None:
            rng = random.Random()
        context = (START,) * self.order
        out: list[str] = []
        while len(out) < max_length:
            nxt = self.next_char(
                context,
                rng,
                trust_profile=trust_profile,
                current_length=len(out),
                min_length=min_length,
                soft_max_length=soft_max_length,
                end_bias=end_bias,
                length_end_bias=length_end_bias,
            )
            if nxt == END:
                break
            out.append(nxt)
            context = context[1:] + (nxt,)
        name = "".join(out)
        if policy is not None:
            name = policy(name)
        return name


def _weighted_choice(items: list[str], weights: list[float], rng: random.Random) -> str:
    """Pick an item proportional to its weight using *rng*."""
    total = sum(weights)
    if total <= 0.0:
        return items[-1]
    r = rng.random() * total
    upto = 0.0
    for item, w in zip(items, weights):
        upto += w
        if r <= upto:
            return item
    return items[-1]
