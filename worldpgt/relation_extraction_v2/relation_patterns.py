"""Deterministic regex patterns for explicit relation extraction v2.

Each pattern is identified by a unique pattern_id string and produces:
- a relation type (from ALLOWED_RELATIONS)
- the direction: which capture group is subject vs object
- a confidence label
- a stability label
- a suggested risk label

No ML, no embeddings, no network. Conservative by design.

Patterns are ordered: more specific (higher confidence) first. The extractor
tries each pattern in order; once a pattern fires it yields a raw candidate.
The validator then filters/quarantines.
"""

from __future__ import annotations

import re
from typing import NamedTuple

# Named capture groups used in all patterns:
#   (?P<A>...) and (?P<B>...)
# Direction indicates which is the relation subject/object:
#   "A_subj_B_obj" means relation is A -> relation -> B
#   "B_subj_A_obj" means relation is B -> relation -> A


class PatternSpec(NamedTuple):
    pattern_id: str
    regex: re.Pattern
    relation: str
    direction: str         # "A_subj_B_obj" | "B_subj_A_obj"
    confidence: str        # "high" | "medium" | "low"
    stability: str         # "stable" | "semi_stable" | "volatile"
    risk: str              # "low" | "medium" | "high"
    description: str


# Maximum number of words between A and B in a single sentence match.
# Prevents runaway matches across clause boundaries.
_MAX_SPAN = 80   # characters

# Flexible entity chunk: letters, digits, spaces, commas, hyphens, periods — but
# no sentence-ending punctuation. Kept short to avoid crossing clause boundaries.
_E = r"[A-Za-z0-9][A-Za-z0-9 ,.\-'&]{0,60}?"   # entity-like phrase (non-greedy)


def _p(pattern: str, flags: int = re.IGNORECASE) -> re.Pattern:
    return re.compile(pattern, flags)


# ---- Safety exclusion screens (never extract if these match) ----------
# Checked BEFORE pattern matching on a per-sentence basis.
CURRENT_LIVE_SCREEN = _p(
    r"\b(current\s+(?:ceo|president|chairman|chief|leader|stock\s+price|price|valuation"
    r"|market\s+cap)|stock\s+price|share\s+price|market\s+cap(?:italization)?"
    r"|net\s+worth|latest\s+(?:revenue|ranking|quarterly)|quarterly\s+revenue"
    r"|right\s+now|as\s+of\s+today|currently\s+(?:serves?|is\s+the|holds?)"
    r"|today'?s|live\s+(?:market|price|data))\b"
)

PRIVATE_SCREEN = _p(
    r"\b(phone\s+number|home\s+address|private\s+email|date\s+of\s+birth"
    r"|social\s+security|passport\s+number)\b"
)

UNIVERSAL_SCREEN = _p(
    r"\b(all\s+\w+\s+are\b|is\s+every\b|are\s+all\b|every\s+\w+\s+is\b)\b"
)

# ---- Relation patterns (ordered: high confidence first) ---------------
PATTERNS: list[PatternSpec] = [

    # ------------------------------------------------------------------- #
    # FOUNDED_BY / FOUNDED — high confidence
    # ------------------------------------------------------------------- #
    PatternSpec(
        "founded_by_passive",
        _p(r"\b(?P<A>" + _E + r")\s+was\s+founded\s+by\s+(?P<B>" + _E + r")\b"),
        "founded_by", "A_subj_B_obj", "high", "semi_stable", "medium",
        "X was founded by Y",
    ),
    PatternSpec(
        "founded_by_passive_in_year",
        _p(r"\b(?P<A>" + _E + r")\s+was\s+founded\s+(?:in\s+\d{4}\s+)?by\s+(?P<B>" + _E + r")\b"),
        "founded_by", "A_subj_B_obj", "high", "semi_stable", "medium",
        "X was founded [in YYYY] by Y",
    ),
    PatternSpec(
        "founded_active",
        _p(r"\b(?P<B>" + _E + r")\s+founded\s+(?P<A>" + _E + r")\b"),
        "founded", "B_subj_A_obj", "high", "semi_stable", "medium",
        "Y founded X",
    ),
    PatternSpec(
        "is_founded_by",
        _p(r"\b(?P<A>" + _E + r")\s+is\s+founded\s+by\s+(?P<B>" + _E + r")\b"),
        "founded_by", "A_subj_B_obj", "high", "semi_stable", "medium",
        "X is founded by Y",
    ),
    PatternSpec(
        "founded_by_reduced_company_clause",
        _p(
            r"\b(?P<A>" + _E + r")(?:\s+\([^)]{0,40}\))?\s+is\s+(?:an?\s+|the\s+)"
            r"[^.]{0,160}?\b(?:company|corporation|manufacturer|provider|firm|startup|organization|"
            r"service|platform|conglomerate|publication|agency|enterprise|group)"
            r"\s+founded\s+by\s+(?P<B>" + _E + r")\b"
        ),
        "founded_by", "A_subj_B_obj", "high", "semi_stable", "medium",
        "X is a company/organization founded by Y",
    ),

    # ------------------------------------------------------------------- #
    # DEVELOPED_BY / DEVELOPS
    # ------------------------------------------------------------------- #
    PatternSpec(
        "developed_by_passive",
        _p(r"\b(?P<A>" + _E + r")\s+(?:was|is|were)\s+developed\s+by\s+(?P<B>" + _E + r")\b"),
        "developed_by", "A_subj_B_obj", "high", "semi_stable", "medium",
        "X was/is developed by Y",
    ),
    PatternSpec(
        "develops_active",
        _p(r"\b(?P<A>" + _E + r")\s+(?:develops?|developed)\s+(?P<B>" + _E + r")\b"),
        "develops", "A_subj_B_obj", "high", "semi_stable", "medium",
        "X develops/developed Y",
    ),
    PatternSpec(
        "designed_and_manufactured_by",
        _p(r"\b(?P<A>" + _E + r")\s+(?:was|is)\s+designed\s+and\s+manufactured\s+(?:in\s+[A-Za-z ]+\s+)?by\s+(?P<B>" + _E + r")\b"),
        "developed_by", "A_subj_B_obj", "high", "semi_stable", "medium",
        "X was designed and manufactured by Y",
    ),

    # ------------------------------------------------------------------- #
    # MANUFACTURES / MANUFACTURED_BY
    # ------------------------------------------------------------------- #
    PatternSpec(
        "manufactured_by_passive",
        _p(r"\b(?P<A>" + _E + r")\s+(?:was|is|are)\s+manufactured\s+by\s+(?P<B>" + _E + r")\b"),
        "developed_by", "A_subj_B_obj", "high", "semi_stable", "medium",
        "X is/was manufactured by Y",
    ),
    PatternSpec(
        "manufactures_active",
        _p(r"\b(?P<A>" + _E + r")\s+manufactures?\s+(?P<B>" + _E + r")\b"),
        "manufactures", "A_subj_B_obj", "high", "semi_stable", "medium",
        "X manufactures Y",
    ),
    PatternSpec(
        "designs_manufactures_sells",
        _p(r"\b(?P<A>" + _E + r")\s+(?:designs?,\s*)?manufactures?,?\s+(?:and\s+)?sells?\s+(?P<B>" + _E + r")\b"),
        "manufactures", "A_subj_B_obj", "high", "semi_stable", "medium",
        "X designs, manufactures, and sells Y",
    ),

    # ------------------------------------------------------------------- #
    # PRODUCES
    # ------------------------------------------------------------------- #
    PatternSpec(
        "produces_active",
        _p(r"\b(?P<A>" + _E + r")\s+produces?\s+(?P<B>" + _E + r")\b"),
        "produces", "A_subj_B_obj", "medium", "semi_stable", "medium",
        "X produces Y",
    ),

    # ------------------------------------------------------------------- #
    # PRODUCT_OF / SERVICE_OF / PLATFORM_OF
    # ------------------------------------------------------------------- #
    PatternSpec(
        "is_product_of",
        _p(r"\b(?P<A>" + _E + r")\s+is\s+a\s+(?:product|offering)\s+of\s+(?P<B>" + _E + r")\b"),
        "product_of", "A_subj_B_obj", "high", "semi_stable", "medium",
        "X is a product of Y",
    ),
    PatternSpec(
        "is_service_of",
        _p(r"\b(?P<A>" + _E + r")\s+is\s+a\s+(?:service|division)\s+of\s+(?P<B>" + _E + r")\b"),
        "service_of", "A_subj_B_obj", "high", "semi_stable", "medium",
        "X is a service of Y",
    ),
    PatternSpec(
        "is_platform_of",
        _p(r"\b(?P<A>" + _E + r")\s+is\s+a\s+platform\s+of\s+(?P<B>" + _E + r")\b"),
        "platform_of", "A_subj_B_obj", "high", "semi_stable", "medium",
        "X is a platform of Y",
    ),

    # ------------------------------------------------------------------- #
    # SUBSIDIARY_OF
    # ------------------------------------------------------------------- #
    PatternSpec(
        "is_subsidiary_of",
        _p(r"\b(?P<A>" + _E + r")\s+is\s+a\s+(?:wholly[\s-]owned\s+)?subsidiary\s+of\s+(?P<B>" + _E + r")\b"),
        "subsidiary_of", "A_subj_B_obj", "high", "semi_stable", "medium",
        "X is a subsidiary of Y",
    ),
    PatternSpec(
        "subsidiary_clause",
        _p(r"\b(?P<A>" + _E + r"),\s+a\s+(?:wholly[\s-]owned\s+)?subsidiary\s+of\s+(?P<B>" + _E + r"),"),
        "subsidiary_of", "A_subj_B_obj", "high", "semi_stable", "medium",
        "X, a subsidiary of Y,",
    ),

    # ------------------------------------------------------------------- #
    # OWNED_BY
    # ------------------------------------------------------------------- #
    PatternSpec(
        "is_owned_by",
        _p(r"\b(?P<A>" + _E + r")\s+is\s+owned\s+by\s+(?P<B>" + _E + r")\b"),
        "owned_by", "A_subj_B_obj", "high", "semi_stable", "medium",
        "X is owned by Y",
    ),
    PatternSpec(
        "acquired_by",
        _p(r"\b(?P<A>" + _E + r")\s+was\s+acquired\s+by\s+(?P<B>" + _E + r")\b"),
        "owned_by", "A_subj_B_obj", "medium", "semi_stable", "medium",
        "X was acquired by Y",
    ),

    # ------------------------------------------------------------------- #
    # PARENT_COMPANY_OF
    # ------------------------------------------------------------------- #
    PatternSpec(
        "parent_company_of",
        _p(r"\b(?P<A>" + _E + r")\s+is\s+the\s+parent\s+company\s+of\s+(?P<B>" + _E + r")\b"),
        "parent_company_of", "A_subj_B_obj", "high", "semi_stable", "medium",
        "X is the parent company of Y",
    ),

    # ------------------------------------------------------------------- #
    # HEADQUARTERED_IN
    # ------------------------------------------------------------------- #
    PatternSpec(
        "headquartered_in",
        _p(r"\b(?P<A>" + _E + r")\s+(?:is\s+)?headquartered\s+(?:in|at)\s+(?P<B>" + _E + r")\b"),
        "headquartered_in", "A_subj_B_obj", "high", "semi_stable", "low",
        "X is headquartered in Y",
    ),
    PatternSpec(
        "headquartered_in_city",
        _p(r"[Hh]eadquartered\s+in\s+(?P<B>[A-Z][A-Za-z ,]+?),\s+(?P<A>" + _E + r")\b"),
        "headquartered_in", "A_subj_B_obj", "medium", "semi_stable", "low",
        "Headquartered in Y, X",
    ),

    # ------------------------------------------------------------------- #
    # OPERATES / OPERATED_BY
    # ------------------------------------------------------------------- #
    PatternSpec(
        "is_operated_by",
        _p(r"\b(?P<A>" + _E + r")\s+(?:is\s+)?operated\s+by\s+(?P<B>" + _E + r")\b"),
        "owned_by", "A_subj_B_obj", "high", "semi_stable", "medium",
        "X is operated by Y",
    ),
    PatternSpec(
        "operates_active",
        _p(r"\b(?P<A>" + _E + r")\s+operates?\s+(?P<B>" + _E + r")\b"),
        "operates", "A_subj_B_obj", "medium", "semi_stable", "medium",
        "X operates Y",
    ),

    # ------------------------------------------------------------------- #
    # CREATED_BY
    # ------------------------------------------------------------------- #
    PatternSpec(
        "created_by_passive",
        _p(r"\b(?P<A>" + _E + r")\s+was\s+created\s+by\s+(?P<B>" + _E + r")\b"),
        "created_by", "A_subj_B_obj", "high", "semi_stable", "medium",
        "X was created by Y",
    ),

    # ------------------------------------------------------------------- #
    # PUBLISHED_BY
    # ------------------------------------------------------------------- #
    PatternSpec(
        "published_by_passive",
        _p(r"\b(?P<A>" + _E + r")\s+(?:was|is)\s+published\s+by\s+(?P<B>" + _E + r")\b"),
        "published_by", "A_subj_B_obj", "high", "semi_stable", "medium",
        "X is/was published by Y",
    ),

    # ------------------------------------------------------------------- #
    # PART_OF
    # ------------------------------------------------------------------- #
    PatternSpec(
        "is_part_of",
        _p(r"\b(?P<A>" + _E + r")\s+is\s+(?:a\s+)?part\s+of\s+(?P<B>" + _E + r")\b"),
        "part_of", "A_subj_B_obj", "medium", "semi_stable", "medium",
        "X is a part of Y",
    ),

    # ------------------------------------------------------------------- #
    # IS_A / TYPE_OF — only from strong definition sentence
    # Restricted: subject must be an org/product; predicate must be definition-
    # like ("is a/an <type>"). Very conservative match.
    # ------------------------------------------------------------------- #
    PatternSpec(
        "is_a_type",
        _p(
            r"\b(?P<A>" + _E + r"),?\s+(?:Inc\.?|Corp\.?|LLC\.?|Ltd\.?)?\s*"
            r"(?:\([^)]{0,40}\))?\s*"  # optional parenthetical
            r"is\s+(?:an?\s+|the\s+)(?P<B>"
            r"(?:American|international|global|multinational|private|public)?\s*"
            r"(?:aerospace|automotive|clean\s+energy|electric\s+vehicle|neurotechnology"
            r"|space\s+exploration|spaceflight|artificial\s+intelligence"
            r"|information\s+technology|satellite|telecommunications"
            r"|software|hardware|internet|broadband|media|news|financial"
            r"|investment|business\s+magazine|transport|infrastructure"
            r"|construction|tunneling|mining|energy|utility"
            r")\s*(?:company|corporation|manufacturer|provider|firm|startup|organization"
            r"|service|constellation|conglomerate|magazine|publication|platform"
            r"|agency|authority|enterprise|group"
            r")?)\b"
        ),
        "is_a", "A_subj_B_obj", "high", "stable", "low",
        "X is an <type> company/organization",
    ),

    # ------------------------------------------------------------------- #
    # LEADER_OF — volatile: leadership changes without notice
    #
    # stability = "volatile": the fact may be stale the moment it is read.
    # risk = "high": presenting outdated leadership as current is misleading.
    # These facts must not be auto-accepted and must never bridge a multi-hop
    # chain (path_validator blocks them via CURRENT_SENSITIVE_PREDICATES).
    # ------------------------------------------------------------------- #
    PatternSpec(
        "is_ceo_of",
        _p(
            r"\b(?P<A>" + _E + r")\s+is\s+(?:the\s+)?"
            r"(?:ceo|chief\s+executive\s+officer|president|chairman|chairperson"
            r"|head|director\s+general|managing\s+director)\s+of\s+(?P<B>" + _E + r")\b"
        ),
        "leader_of", "A_subj_B_obj", "high", "volatile", "high",
        "X is CEO/president/chairman/head of Y",
    ),
    PatternSpec(
        "serves_as_ceo_of",
        _p(
            r"\b(?P<A>" + _E + r")\s+(?:serves?|served)\s+as\s+(?:the\s+)?"
            r"(?:ceo|chief\s+executive\s+officer|president|chairman|head|director)\s+"
            r"of\s+(?P<B>" + _E + r")\b"
        ),
        "leader_of", "A_subj_B_obj", "high", "volatile", "high",
        "X serves/served as CEO/president/head of Y",
    ),
    PatternSpec(
        "leads_active",
        _p(
            r"\b(?P<A>" + _E + r")\s+(?:leads?|chairs?)\s+(?P<B>" + _E + r")\b"
        ),
        "leader_of", "A_subj_B_obj", "medium", "volatile", "high",
        "X leads/chairs Y",
    ),

    # ------------------------------------------------------------------- #
    # VALUED_AT — volatile: financial valuations change with market conditions
    #
    # stability = "volatile": valuation is a time-sensitive estimate.
    # risk = "high": stating an outdated valuation as fact is misleading.
    # These facts require source qualification and are never stable memory.
    # ------------------------------------------------------------------- #
    PatternSpec(
        "valued_at_passive",
        _p(
            r"\b(?P<A>" + _E + r")\s+(?:is|was|has\s+been)\s+valued\s+at\s+"
            r"(?P<B>(?:\$|USD\s*)?\d[\d.,]*\s*(?:billion|million|trillion|B|M|T)?)\b"
        ),
        "valued_at", "A_subj_B_obj", "high", "volatile", "high",
        "X is valued at $N billion/million",
    ),
    PatternSpec(
        "has_valuation_of",
        _p(
            r"\b(?P<A>" + _E + r")\s+(?:has|had)\s+a\s+valuation\s+of\s+"
            r"(?P<B>(?:\$|USD\s*)?\d[\d.,]*\s*(?:billion|million|trillion|B|M|T)?)\b"
        ),
        "valued_at", "A_subj_B_obj", "high", "volatile", "high",
        "X has a valuation of $N billion",
    ),
]
