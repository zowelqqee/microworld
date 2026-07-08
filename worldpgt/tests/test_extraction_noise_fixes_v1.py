"""Regression tests for two systemic extraction-noise bugs (2026-07-07).

Found while auditing the ~21% noise in the overlay's cleanest (hand-seeded)
relation tier. Both are deterministic parsing bugs, fixed at the source so a
future re-pump produces clean facts:

1. Abbreviation-restore corruption: split_sentences() turned every "U.S.",
   "Dr.", "St.", "Inc." etc. into doubled + NUL-suffixed garbage
   ("U.SU.S\x00", "DrDr\x00", ...). Measured blast radius: 279 / 8928
   overlay items carried this signature.
2. Anaphoric / malformed subject spans ("This precision", "These flights",
   "The user can") passed the entity screen and became relation subjects.
"""

from __future__ import annotations

from worldpgt.relation_extraction_v2.relation_candidate_extractor import _is_too_generic
from worldpgt.relation_extraction_v2.sentence_splitter import split_sentences


# --- Bug 1: abbreviation restore corruption --------------------------------

def test_us_abbreviation_is_not_corrupted():
    (sent, *_) = split_sentences("Capella provides imagery to the U.S. government today.")
    assert "U.SU.S" not in sent
    assert "\x00" not in sent
    assert "U.S. government" in sent


def test_common_abbreviations_restore_verbatim():
    text = "Dr. Smith joined Acme Inc. on St. Charles Ave. last year."
    (sent, *_) = split_sentences(text)
    for garbage in ("DrDr", "IncInc", "StSt", "\x00"):
        assert garbage not in sent
    assert "Dr. Smith" in sent
    assert "Acme Inc." in sent


def test_sentence_boundary_after_abbreviation_still_splits():
    """Fixing the corruption must NOT collapse a real sentence boundary that
    happens to follow an abbreviation-containing sentence."""
    sents = split_sentences("Capella serves the U.S. government. Next sentence here.")
    assert len(sents) == 2
    assert sents[0].endswith("government.")
    assert sents[1] == "Next sentence here."


def test_multiple_abbreviations_in_one_sentence():
    (sent, *_) = split_sentences("The firm operates in the U.S., the U.K., and the E.U. markets.")
    assert "\x00" not in sent
    for abbr in ("U.S.", "U.K.", "E.U."):
        assert abbr in sent


# --- Bug 2: anaphoric / malformed subject spans ----------------------------

def test_demonstrative_led_subject_is_rejected():
    for bad in ("This precision", "These flights", "That approach", "Those systems"):
        assert _is_too_generic(bad), bad


def test_subject_ending_in_auxiliary_is_rejected():
    for bad in ("The user can", "The system will", "The device is"):
        assert _is_too_generic(bad), bad


def test_real_the_led_entities_are_kept():
    for good in ("The Boring Company", "The New York Times", "The Washington Post"):
        assert not _is_too_generic(good), good


def test_plain_named_entities_are_kept():
    for good in ("SpaceX", "Elon Musk", "United States", "Bloomberg News"):
        assert not _is_too_generic(good), good
