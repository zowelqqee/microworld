"""Every tunable number in the dialogue context layer.

Single source of truth: no other module in ``worldpgt/dialogue`` may contain a
numeric literal that affects resolution behavior. The dialogue benchmark
imports these values to render expected traces, so changing a constant here
produces exactly one code diff plus a visible fixture diff.

All scores are integers. Salience is computed at read time from turn indices
stored in :mod:`worldpgt.dialogue.state`; nothing here is ever "trained".
"""

from __future__ import annotations

# ── Salience features (trace names match these identifiers verbatim) ────────
ACTIVE_TOPIC = 100
LAST_ANSWER_ENTITY = 40
LAST_QUESTION_SUBJECT = 30
USER_NAMED = 15
WAS_TOPIC = 10
ROLE_MATCH = 25
STICKY_REFERENT = 10
SAME_QUESTION_MENTION = 50
RECENCY_PENALTY_PER_TURN = 12
RECENCY_FLOOR = -60
MENTION_BONUS = 3
MENTION_BONUS_CAP = 5

# ── Decision thresholds ──────────────────────────────────────────────────────
# top1 - top2 must reach this margin, else the slot is ambiguous → audit.
RESOLVE_MARGIN = 25
# Entities at or below this salience are invisible to plural / selective /
# contrastive resolution (a sole type-gated survivor may still win a singular
# slot — being the only candidate of the right type is itself evidence).
ACTIVATION_THRESHOLD = 0
# "Which one ...?" with more active candidates than this audits instead of
# running an unboundedly wide filter.
SELECTIVE_MAX = 4
# Plural "they/their" resolves only when exactly this many active entities
# qualify (matches the v1 two-entity behavior).
PLURAL_SIZE = 2

# ── Lifecycle ────────────────────────────────────────────────────────────────
# Entities unmentioned for this many *confirmed* (non-audit) turns are evicted.
EVICT_AFTER_TURNS = 8
# Registry hard cap; beyond it the lowest-salience entities are evicted,
# ties broken by older introduced_turn first. Never by name, never randomly.
REGISTRY_CAP = 16
