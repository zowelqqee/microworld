"""
Quality policy for generated personal names (given names, surnames, or mixed).

A small, inspectable rule set that scores a generated name in [0.0, 1.0] and
explains *why*.  It is intentionally not Anglo-centric: common Russian, Georgian,
Armenian and broader European endings are recognised as *positive signals*, and
unicode (e.g. Cyrillic) names are not penalised for failing ASCII-only vowel
checks.

The policy does *not* require a classic surname ending.  Input datasets may
contain given names, surnames, or a mixture — the rules apply equally to all
personal-name forms.

There are no learned weights here — every deduction is an explicit, named rule.

Scoring overview
----------------
Penalties (cumulative, clamped to [0.0, 1.0]):
  too short          length < 2                        -0.60
  very short         length == 2                       -0.20
  long_name          12 <= length < 15                 -0.12
  too_long           15 <= length < 18                 -0.30
  extremely_long     length >= 18                      -0.45
  invalid characters                                   -0.50
  starts/ends punct                                    -0.60
  too much punct     > 2 punct chars                   -0.30
  no vowels                                            -0.60
  too many vowels    ratio > 0.85 for len >= 4         -0.30
  repeated chars     run >= 3 same chars               -0.30
  weird consonant    run >= 4 consonants (ASCII only)  -0.40
  common_word_like                                      -0.30
  brand_like                                            -0.35
  too_fragmentary    weak length-3 form                 -0.18
  awkward_short_form weak length-3 readability          -0.15
  nickname_like      audited nickname/distortion        -0.25
  distorted_known_name                                  -0.25
  weird_q_usage      q not followed by u                -0.25
  medium_glued_name  joined medium-length chunks        -0.18
  poor_readability   awkward audited substrings         -0.18
  too_many_syllable_chunks  nuclei >= 7                -0.25
  too_many_syllable_chunks  nuclei >= 5 & len >= 10    -0.20
  possible_glued_name       nuclei >= 4 & len >= 12    -0.10
  doubled_vowel      same vowel twice adj, len >= 10   -0.08

Positive reasons (informational, do not raise score above 1.0):
  looks like a plausible name  (when no penalties)
  reasonable_length            3 <= length <= 11
  balanced_vowels              0.25 <= vowel_ratio <= 0.65
  common_name_ending           recognised suffix
"""
from __future__ import annotations

MIN_LENGTH = 2
MAX_LENGTH = 20
_VALID_THRESHOLD = 0.5

_PUNCT = frozenset("'-")

# Vowels across the scripts we expect (Latin + common accents + Cyrillic).
_VOWELS = frozenset(
    "aeiouy"
    "àáâäãåāèéêëēìíîïįıòóôöõøōùúûüūÿ"
    "аеёиоуыэюя"
)

# Endings that are well-known surname forms — used as a positive signal to
# relax the consonant-cluster and no-vowel checks.  Their absence is not
# penalised at all.
ALLOWED_ENDINGS: tuple[str, ...] = (
    "shvili", "dze", "ova", "eva", "ina", "sky", "ski", "son", "sen",
    "yan", "ian", "ich", "ov", "ev", "in", "ez",
)

COMMON_WORD_LIKE = frozenset({"all", "march", "loch"})
BRAND_LIKE = frozenset({"avito"})
NICKNAME_LIKE = frozenset({"appy", "lovelo", "kylez"})

# Short generated names are allowed, but only a small set should be treated as
# clean short personal names. Other length-3 forms remain possible but weak.
_CLEAN_SHORT_NAMES = frozenset(
    {
        "ada", "amy", "ann", "ava", "bea", "eli", "eva", "ian", "ida",
        "ira", "kai", "leo", "lia", "li", "mia", "nia", "noa", "ty",
        "zoe",
    }
)

_DISTORTED_KNOWN_NAMES = frozenset({"kylez", "jamess", "robee"})
_DISTORTED_SUFFIXES = (("kyle", "z"), ("james", "s"), ("rob", "ee"))

_MEDIUM_GLUED_EXAMPLES = frozenset(
    {
        "latalille",
        "brighteme",
        "nafranimi",
        "roaryonnia",
        "sauldenyx",
        "evreekahl",
        "qweslienna",
        "gateuillis",
        "journesten",
    }
)

_AWKWARD_SUBSTRINGS = (
    "qwe",
    "uille",
    "uill",
    "eigh",
    "ynn",
    "denyx",
    "lille",
    "franimi",
    "yonnia",
    "reekahl",
    "nesten",
    "ghteme",
)


def _is_vowel(ch: str) -> bool:
    return ch.lower() in _VOWELS


def _ends_allowed(name: str) -> bool:
    return any(name.endswith(suf) for suf in ALLOWED_ENDINGS)


def _longest_run(name: str, predicate) -> int:
    """Longest run of consecutive characters satisfying *predicate*."""
    best = run = 0
    for ch in name:
        if predicate(ch):
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def _longest_consonant_run(letters: list[str]) -> int:
    return _longest_run("".join(letters), lambda c: c.isalpha() and not _is_vowel(c))


def _longest_repeat(name: str) -> int:
    """Longest run of the same repeated character."""
    best = run = 0
    prev = None
    for ch in name:
        if ch == prev:
            run += 1
        else:
            run = 1
            prev = ch
        best = max(best, run)
    return best


def _count_vowel_nuclei(name: str) -> int:
    """Count maximal consecutive vowel groups (syllable nuclei) in *name*."""
    count = 0
    in_vowel = False
    for ch in name:
        if ch.isalpha() and _is_vowel(ch):
            if not in_vowel:
                count += 1
                in_vowel = True
        else:
            in_vowel = False
    return count


def _has_weird_q_usage(name: str) -> bool:
    """Return True for q that is not followed by u in otherwise Latin tokens."""
    for i, ch in enumerate(name):
        if ch.lower() != "q":
            continue
        nxt = name[i + 1].lower() if i + 1 < len(name) else ""
        if nxt != "u":
            return True
    return False


def _looks_distorted_known_name(name: str) -> bool:
    if name in _DISTORTED_KNOWN_NAMES:
        return True
    return any(name == base + suffix for base, suffix in _DISTORTED_SUFFIXES)


def _has_poor_readability(name: str) -> bool:
    return any(part in name for part in _AWKWARD_SUBSTRINGS)


def _looks_medium_glued(name: str, nuclei: int) -> bool:
    if name in _MEDIUM_GLUED_EXAMPLES:
        return True
    length = len(name)
    if not 7 <= length <= 12:
        return False
    if _has_poor_readability(name):
        return True
    # Four or more vowel nuclei in a medium token is suspicious only when paired
    # with additional awkwardness; otherwise names like "jakaria" stay clean.
    if nuclei >= 4 and (
        any(part in name for part in ("lill", "nini", "yonn", "uill"))
        or _has_weird_q_usage(name)
    ):
        return True
    return False


def _evaluate(name: str) -> tuple[float, list[str]]:
    """Return ``(score, reasons)`` for *name*.

    ``reasons`` combines penalty tags (which fired and lowered the score) with
    positive informational tags (reasonable_length, balanced_vowels, etc.).
    An empty penalty list produces "looks like a plausible name" as the first
    entry.
    """
    n = name.strip()
    penalties: list[str] = []

    if not n:
        return 0.0, ["empty name"]

    score = 1.0
    length = len(n)
    letters = [c for c in n if c.isalpha()]
    punct = [c for c in n if c in _PUNCT]
    invalid = [c for c in n if not c.isalpha() and c not in _PUNCT]
    has_nonascii = any(ord(c) > 127 for c in n)
    ends_ok = _ends_allowed(n)

    # ── Length ────────────────────────────────────────────────────────────────
    if length < MIN_LENGTH:
        penalties.append("too short")
        score -= 0.60
    elif length == MIN_LENGTH:
        penalties.append("very short")
        score -= 0.20
    elif 12 <= length < 15:
        penalties.append("long_name")
        score -= 0.25
    elif 15 <= length < 18:
        penalties.append("too_long")
        score -= 0.30
    elif length >= 18:
        penalties.append("extremely_long")
        score -= 0.45

    # ── Characters ────────────────────────────────────────────────────────────
    if invalid:
        penalties.append("invalid characters")
        score -= 0.50

    if n[0] in _PUNCT or n[-1] in _PUNCT:
        penalties.append("starts or ends with punctuation")
        score -= 0.60

    if len(punct) > 2:
        penalties.append("too much punctuation")
        score -= 0.30

    # ── Vowel / consonant structure ───────────────────────────────────────────
    vowels = [c for c in letters if _is_vowel(c)]
    vowel_ratio = len(vowels) / len(letters) if letters else 0.0

    # No-vowel garbage like "qzxqz".  Skipped for unicode names (different
    # vowel inventory) and for names with a recognised surname ending — the
    # ending acts as a positive signal that the form is intentional.
    if letters and not vowels and not has_nonascii and not ends_ok:
        penalties.append("no vowels")
        score -= 0.60

    if length >= 4 and vowel_ratio > 0.85:
        penalties.append("too many vowels")
        score -= 0.30

    if _longest_repeat(n) >= 3:
        penalties.append("too many repeated characters")
        score -= 0.30

    lowered = n.lower()

    # ── Audited token classes ────────────────────────────────────────────────
    if lowered in COMMON_WORD_LIKE:
        penalties.append("common_word_like")
        score -= 0.30

    if lowered in BRAND_LIKE:
        penalties.append("brand_like")
        score -= 0.35

    if lowered in NICKNAME_LIKE:
        penalties.append("nickname_like")
        score -= 0.25

    if _looks_distorted_known_name(lowered):
        penalties.append("distorted_known_name")
        score -= 0.25

    # Length-3 fragments are often readable-looking but not name-like enough to
    # deserve a perfect score. Known clean short names are explicitly spared.
    if length == 3 and lowered not in _CLEAN_SHORT_NAMES:
        penalties.append("too_fragmentary")
        score -= 0.18
        penalties.append("awkward_short_form")
        score -= 0.15

    # Unusual consonant clusters are penalised only for ASCII names without a
    # recognised surname-style ending (the ending is a positive signal here).
    if not has_nonascii and not ends_ok and _longest_consonant_run(letters) >= 4:
        penalties.append("weird consonant cluster")
        score -= 0.40

    # ── Syllable-nuclei / glued-name detection ────────────────────────────────
    nuclei = _count_vowel_nuclei(n)
    if not has_nonascii and _has_weird_q_usage(lowered):
        penalties.append("weird_q_usage")
        score -= 0.25

    if _looks_medium_glued(lowered, nuclei):
        penalties.append("medium_glued_name")
        score -= 0.18

    if _has_poor_readability(lowered):
        penalties.append("poor_readability")
        score -= 0.18

    if nuclei >= 7:
        penalties.append("too_many_syllable_chunks")
        score -= 0.25
    elif nuclei >= 5 and length >= 10:
        penalties.append("too_many_syllable_chunks")
        score -= 0.20
    elif nuclei >= 4 and length >= 12:
        penalties.append("too_many_syllable_chunks")
        score -= 0.20

    # Same vowel repeated consecutively (e.g. "aa", "ee", "yy") — a common
    # artefact of glued generated names; only penalised in longer strings.
    if length >= 10 and any(
        n[i] == n[i + 1] and _is_vowel(n[i]) for i in range(len(n) - 1)
    ):
        penalties.append("doubled_vowel")
        score -= 0.08

    # ── Positive informational signals ────────────────────────────────────────
    positive: list[str] = []
    if 3 <= length <= 11:
        positive.append("reasonable_length")
    if letters and 0.25 <= vowel_ratio <= 0.65:
        positive.append("balanced_vowels")
    if ends_ok:
        positive.append("common_name_ending")

    score = max(0.0, min(1.0, score))

    if not penalties:
        reasons = ["looks like a plausible name"] + positive
    else:
        reasons = penalties + positive

    return score, reasons


def surname_quality_score(name: str) -> float:
    """Return a quality score in [0.0, 1.0]; higher is more name-like."""
    return _evaluate(name)[0]


def explain_surname_quality(name: str) -> list[str]:
    """Return the list of quality reasons for *name*.

    The list always contains at least one entry.  For clean names it starts
    with "looks like a plausible name" followed by any positive signals
    (reasonable_length, balanced_vowels, common_name_ending).  For problematic
    names it contains the penalty tags that fired.
    """
    return _evaluate(name)[1]


def is_valid_generated_surname(
    name: str,
    *,
    source_names: set[str] | None = None,
    avoid_duplicates: bool = False,
    threshold: float = _VALID_THRESHOLD,
) -> bool:
    """Return True if *name* passes the quality policy.

    Works for given names, surnames, or mixed personal-name datasets.  When
    ``avoid_duplicates`` is set, a name that already exists in ``source_names``
    (compared case-insensitively) is rejected outright.
    """
    if avoid_duplicates and source_names is not None:
        if name.strip().lower() in source_names:
            return False
    return surname_quality_score(name) >= threshold
