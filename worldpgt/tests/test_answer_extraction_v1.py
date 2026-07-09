"""Tests for deterministic answer-sentence extraction (no network)."""

from __future__ import annotations

from worldpgt.web_search.answer_extraction import extract_answer


_CHADWICK = (
    "Sir James Chadwick was a British experimental physicist who received the "
    "Nobel Prize in Physics in 1935 for his discovery of the neutron. "
    "He was born in Bollington. He later worked on the Manhattan Project."
)

_RIHANNA = (
    "Robyn Rihanna Fenty is a Barbadian singer and businesswoman. "
    "She is regarded as one of the best-selling music artists. "
    "Career beginnings "
    "She was born and raised in Bridgetown, Barbados. "
    "She signed with Def Jam in 2005."
)


def test_definition_question_returns_intro_sentence():
    ans = extract_answer("who is james chadwick?", _CHADWICK)
    assert ans is not None
    assert "physicist" in ans.lower()


def test_relation_question_picks_relation_sentence():
    ans = extract_answer("what did chadwick discover?", _CHADWICK, subject="James Chadwick")
    assert ans is not None
    assert "neutron" in ans.lower()


def test_pronoun_led_sentence_counts_as_subject():
    # "She was born in ... Bridgetown" leads with a pronoun; must beat the
    # birth-date intro sentence.
    ans = extract_answer("where was rihanna born and raised?", _RIHANNA, subject="Rihanna")
    assert ans is not None
    assert "bridgetown" in ans.lower()


def test_off_subject_relation_keyword_is_ignored():
    text = (
        "Alan Smith is an English footballer. "
        "He played as a striker for many years. "
        "He scored over one hundred goals. "
        "He retired in 2010. "
        "He later became a pundit. "
        "He lives in London. "
        "His teammate Ryan Hite attended Denison University where he set records."
    )
    # The only 'university' sentence is about a named teammate (not the
    # subject), deep in the text (idx >= 6) -> must not be returned.
    ans = extract_answer("what school did alan smith go to?", text, subject="Alan Smith")
    assert ans is None or "denison" not in ans.lower()


def test_lead_boundary_uses_real_section_header_not_a_fixed_count():
    """A well-developed lead (a country, a company) can run past 6 sentences
    and still be entirely on-topic — the cutoff must follow the article's
    actual structure (the first section header), not a fixed sentence count.
    """

    text = (
        "Bhutan is a landlocked country in South Asia. "
        "It is located in the Eastern Himalayas. "
        "It borders China to the north. "
        "It borders India to the south. "
        "It has a population of over 800,000. "
        "It is a democratic constitutional monarchy. "
        "The Je Khenpo is the head of the state religion. "
        "The capital and largest city is Thimphu.\n\n"
        "== History ==\n\n"
        "Some unrelated historical sentence about a different topic here."
    )
    ans = extract_answer("what is the capital of bhutan?", text, subject="Bhutan")
    assert ans is not None
    assert "thimphu" in ans.lower()


def test_returns_none_on_empty_text():
    assert extract_answer("who is x?", "") is None


def test_no_relation_keyword_match_returns_none():
    text = (
        "The Blorptax is a fictional device. "
        "It hums quietly. It glows blue at night."
    )
    # A death question with no death keyword anywhere -> None (caller falls
    # back to intro).
    assert extract_answer("what did the blorptax die of?", text, subject="Blorptax") is None


def test_current_office_question_rejects_generic_office_intro():
    text = (
        "The president of Exampleland is the head of state and head of government. "
        "The office was created in 1900."
    )

    ans = extract_answer(
        "who is the current president of Exampleland?",
        text,
        subject="President of Exampleland",
    )

    assert ans is None


def test_current_office_question_accepts_officeholder_sentence():
    text = "The officeholder of President of Exampleland is Jane Example."

    ans = extract_answer(
        "who is the current president of Exampleland?",
        text,
        subject="President of Exampleland",
    )

    assert ans == text
