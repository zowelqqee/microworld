"""Deterministic phrase library, keyed by (term, sense_id) and connector type.

Small, explicit, hand-curated continuation phrases. No generation, no weights.
Each phrase is a short, clean clause/predicate appropriate for one sense under
one connector type. The renderer renders and ranks these; it never invents text.

Connector keys mirror the constants in ``realization`` (e.g. ``infinitive_after_to``,
``clause_after_and``, ``clause_after_as``, ``neutral_extension``).
"""

from __future__ import annotations

from worldpgt.continuation.realization import ENDING_NEUTRAL

# (term, sense_id) -> connector_type -> [phrases]
PHRASE_LIBRARY: dict[tuple[str, str], dict[str, list[str]]] = {
    ("bank", "financial_institution"): {
        "infinitive_after_to": ["open an account", "make a deposit", "speak with the teller"],
        "clause_after_and": ["the teller processed the request", "she signed the forms"],
        "clause_after_as": ["the teller checked the paperwork"],
        "clause_after_until_subject": ["the teller called her forward"],
        "clause_after_before_subject": ["the teller checked the paperwork"],
        "clause_after_when_subject": ["the teller called her forward"],
        "neutral_extension": ["and completed the transaction"],
    },
    ("bank", "river_edge"): {
        "infinitive_after_to": ["watch the current", "reach the water", "step onto the muddy edge"],
        "clause_after_and": ["the water moved past the reeds", "the current pulled at the reeds"],
        "clause_after_as": ["carried it downstream", "pushed it toward the shore"],
        "clause_after_until_subject": ["the boat touched the mud"],
        "clause_after_before_subject": ["the water reached the shore"],
        "clause_after_when_subject": ["the current slowed"],
        "neutral_extension": ["and watched the current"],
    },
    ("bat", "animal"): {
        "infinitive_after_to": ["search for insects", "find the night air"],
        "clause_after_and": ["it searched for insects", "it spread its wings"],
        "clause_after_as": ["it flew into the dark"],
        "clause_after_before_subject": ["it spread its wings"],
        "clause_after_when_subject": ["it flew into the dark"],
        "neutral_extension": ["and searched for insects", "and flew into the dark"],
    },
    ("bat", "sports_equipment"): {
        "infinitive_after_to": ["take a swing", "hit the ball", "prepare for the pitch"],
        "clause_after_and": ["the player took his stance", "the batter prepared to swing"],
        "clause_after_as": ["the pitcher released the ball"],
        "clause_after_before_subject": ["he steadied himself", "the player took his stance"],
        "clause_after_when_subject": ["the player took his stance"],
        "neutral_extension": ["and the player dropped the bat", "and hit the ball"],
    },
    ("seal", "animal"): {
        "infinitive_after_to": ["catch another fish", "look for more fish"],
        "clause_after_and": ["it surfaced for air", "it dove below the surface"],
        "clause_after_as": ["it followed the fish"],
        "clause_after_before_subject": ["it dove below the surface"],
        "clause_after_when_subject": ["it dove below the surface"],
        "neutral_extension": ["and dove below the surface", "and swam through the cold water"],
    },
    ("seal", "closure_stamp"): {
        "infinitive_after_to": ["close the envelope", "mark the document"],
        "clause_after_and": ["the wax hardened", "the mark held firm"],
        "clause_after_as": ["the wax cooled"],
        "clause_after_before_subject": ["the wax hardened"],
        "clause_after_when_subject": ["the wax hardened"],
        "neutral_extension": ["and marked the document", "and the wax hardened"],
    },
    ("crane", "bird"): {
        "infinitive_after_to": ["step through the reeds", "reach the lake"],
        "clause_after_and": ["it crossed the shallows", "it spread its wings"],
        "clause_after_as": ["it crossed the shallows"],
        "clause_after_before_subject": ["spread its wings"],
        "clause_after_when_subject": ["crossed the shallows"],
        "neutral_extension": ["and spread its wings", "and crossed the shallows"],
    },
    ("crane", "machine"): {
        "infinitive_after_to": ["lift the load", "raise the beam"],
        "clause_after_and": ["the operator raised the load"],
        "clause_after_as": ["the operator checked the cables"],
        "clause_after_until_subject": ["the operator lowered the hook"],
        "clause_after_before_subject": ["the operator checked the cables"],
        "clause_after_when_subject": ["the operator raised the load"],
        "clause_after_while_subject": ["the operator checked the cables"],
        "neutral_extension": ["and carried the load", "and the operator checked the cables"],
    },
    ("spring", "season"): {
        "infinitive_after_to": ["bring warmer days"],
        "clause_after_and": ["the flowers opened", "the garden filled with flowers"],
        "clause_after_as": ["the weather warmed"],
        "clause_after_until_subject": ["the rain softened the ground"],
        "clause_after_before_subject": ["the weather warmed"],
        "clause_after_when_subject": ["the weather warmed"],
        "neutral_extension": ["and brought warmer days"],
    },
    ("spring", "coil"): {
        "infinitive_after_to": ["store mechanical energy"],
        "clause_after_and": ["the mechanism snapped back", "the tension released"],
        "clause_after_as": ["the device compressed it"],
        "clause_after_until_subject": ["the latch released"],
        "clause_after_before_subject": ["the mechanism snapped back"],
        "clause_after_when_subject": ["the mechanism snapped back"],
        "clause_after_after_subject": ["the mechanism snapped back into place"],
        "neutral_extension": ["and snapped back into place"],
    },
    ("rock", "stone"): {
        "infinitive_after_to": ["roll down the hill"],
        "clause_after_and": ["it rolled downhill", "it scraped the ground"],
        "clause_after_as": ["it slid down the slope"],
        "clause_after_until_subject": ["it reached the ground"],
        "clause_after_before_subject": ["scraped the ground"],
        "clause_after_when_subject": ["scraped the ground"],
        "neutral_extension": ["and lay near the river"],
    },
    ("rock", "music"): {
        "infinitive_after_to": ["fill the room"],
        "clause_after_and": ["the band started playing"],
        "clause_after_as": ["the guitarist started the riff"],
        "clause_after_until_subject": ["the drummer counted in"],
        "clause_after_before_subject": ["the band started playing"],
        "clause_after_when_subject": ["the guitar started playing"],
        "neutral_extension": ["and filled the stadium"],
    },
}


def get_phrases(term: str, sense_id: str, connector_type: str) -> list[str]:
    """Phrases for one (term, sense, connector). Falls back to neutral, then []."""
    sense_phrases = PHRASE_LIBRARY.get((term.lower(), sense_id))
    if not sense_phrases:
        return []
    if connector_type in sense_phrases and sense_phrases[connector_type]:
        return list(sense_phrases[connector_type])
    if sense_phrases.get(ENDING_NEUTRAL):
        return list(sense_phrases[ENDING_NEUTRAL])
    return []
