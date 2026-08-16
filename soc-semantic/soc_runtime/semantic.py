"""The semantic layer: a typed entity/relation graph over SOC alerts.

Same shape as the two prior ports (AML, network intrusion detection): typed
entities, typed relations between them, and features that are properties of
those relations at the moment of the alert being scored - never a property
of the future.

Entities
    actor          resolved via `_resolve_actor` - who did it, or the best
                   available stand-in; see "Actor identity" below
    host           agent_name - where it happened
    actor_host     one actor's relationship with one host
    actor_technique     one actor's relationship with one technique, across hosts
    actor_host_technique     the full triple - has *this* combination fired before
    process_chain     (host, parent_image, image) - the launch context, when present

Causality
    Features for alert *i* are computed from alerts strictly before *i* (by
    timestamp, ties broken by input order), then alert *i* is folded into the
    state. Nothing reads `is_synthetic_anomaly` or `synthetic_anomaly_type` -
    those columns exist only for `modeling.py` to score against afterwards.

Identity
    The graph is keyed on actor names, host names and technique IDs, but only
    aggregates leave this module - no raw identity string is ever written to
    a `sem_*` column. This matters more here than in the network port: one of
    the two populations this dataset is designed to distinguish *is* a fixed,
    named service account, and a feature that leaked the account name would
    let a model memorise "svc-swinventory -> benign" instead of learning the
    frequency/novelty/timing pattern that makes it benign. `tests/test_semantic.py`
    asserts no raw identity column is ever produced.

Actor identity - a real finding from the first real run, not a hypothetical
    `subject_user_name`/`subject_domain_name` (`data.win.eventdata.subjectUserName`/
    `subjectDomainName`) is null on **99.48%** of real process-creation
    alerts on the T.Hunter cluster (10,426 of 10,480 in the first 5-day
    sample). That is not an edge case for a prototype whose every entity is
    keyed on "who did this" - it is close to the field never being usable at
    all. `_resolve_actor` falls back through three tiers rather than
    quietly keying everything on a mostly-empty string:

      1. `subject_user_name`/`subject_domain_name`, when present - the
         account the logging subsystem attributes the event to.
      2. `data.win.eventdata.user` (Sysmon's own `User` field: the execution
         context of the *new* process, not who caused the event to be
         logged - a different field with different semantics, not a second
         copy of the same one), parsed from its usual `"DOMAIN\\user"` shape,
         when tier 1 is empty.
      3. A host-keyed pseudo-identity, `"(host:<agent_name>)"`, when neither
         field is usable. "Who" degrades to "where" here - every alert from
         one host with no usable identity collapses onto one pseudo-actor,
         which is a real loss of resolution, not a cosmetic default. Whether
         `sem_novel_host_for_actor` and similar actor-keyed features still
         mean anything for these rows is weaker by construction, not just by
         data quality, since the "actor" and the "host" are the same value.

    Two binary features - `sem_actor_identity_missing_subject` and
    `sem_actor_identity_host_fallback` - record which tier fired, on the
    theory that whether `subject_user_name` was populated at all is itself
    informative on a field this sparse, not just a parsing nuisance to hide.
    **Tier 2's real fill rate has not been independently confirmed** - this
    codebase has not inspected the real documents directly (see the README's
    "Real data" section) - so tier 2 is a well-motivated implementation
    ready to be measured on the next real run, not a verified fix.

Multi-valued techniques - the second real finding that changed this module
    57.7% of real process-creation alerts carry more than one value in
    `rule.mitre.technique`. Because this module's state is keyed on
    (actor, host, technique) and every alert is *folded into* that state
    for the next alert to read, keeping only the first technique is not a
    reporting simplification - it corrupts the state: the dropped
    techniques' frequency counters never increment, their "seen before"
    sets never learn them (so a later alert is scored novel when it is
    not), and `sem_chain_tactic_diversity_15m` cannot observe two tactics
    arriving together, which is precisely what it exists to count.

    `build_features(frame, mode=...)` takes one of
    `config.MULTI_TECHNIQUE_MODES`:

      first_only          Read and fold the primary technique only. The
                          pre-fix behaviour, kept bit-exact - it is the
                          baseline every published v1 number was measured
                          under, and `_value_lists` guarantees
                          `list[0]` is the legacy scalar column's own
                          value so this really is the old code path.
      primary_plus_count  Fold every technique/tactic (the correctness
                          fix); read this alert's own features from the
                          primary technique, plus `MULTI_TECHNIQUE_FEATURE`.
      aggregate           Fold every technique/tactic; read every one and
                          collapse per-feature via `_AGGREGATION`.

    The full value lists arrive in the optional `rule_mitre_technique_all`
    / `rule_mitre_tactic_all` columns (real data and `synthetic_v2.py`);
    when absent, the scalar columns are used and all three modes agree by
    construction. See the README's multi-valued section for the measured
    comparison - the fix corrects the state demonstrably but does not
    improve any model metric on the corpora available.

Scope
    Computed over every `event_category == "process_creation"` alert,
    including the two escalation-only techniques (T1003, T1021) that never
    appear in `config.IN_SCOPE_TECHNIQUES` - the chain features need to see
    them to detect an escalation. `modeling.py` restricts the *scored*
    dataset to the six in-scope Discovery-tagged techniques; this module does
    not do that filtering itself.

Run:
    python -m soc_runtime.semantic
"""

from __future__ import annotations

import json
from collections import Counter, deque

import numpy as np
import pandas as pd

from soc_runtime import config, filters

# --------------------------------------------------------------------------
# Feature catalogue. One sentence per feature is the whole justification.
# --------------------------------------------------------------------------

FEATURES: tuple[tuple[str, str, str], ...] = (
    ("sem_freq_actor_host_technique_24h", "actor_host_technique",
     "How many times this exact actor ran this exact technique on this exact host in the "
     "preceding 24 hours - the signature of a repeating, already-explained pattern versus a "
     "first occurrence."),
    ("sem_freq_actor_technique_7d", "actor_technique",
     "How many times this actor has run this technique anywhere in the preceding 7 days - "
     "separates an actor whose role is discovery from one who essentially never touches it."),
    ("sem_novel_actor_host_technique", "actor_host_technique",
     "Whether this actor has ever run this exact technique on this exact host before, anywhere "
     "in history - the single strongest 'have we already explained this' signal."),
    ("sem_novel_host_for_actor", "actor_host",
     "Whether this actor has ever been seen on this host before, regardless of technique - a "
     "familiar operator on an unfamiliar machine is a different risk than the same operator on "
     "their usual box."),
    ("sem_novel_actor", "actor",
     "Whether this account has never appeared in the alert history before this moment - a "
     "brand-new identity running discovery commands is inherently less explainable than an "
     "established one."),
    ("sem_actor_time_typicality", "actor",
     "Share of this actor's historical alerts that fall in the same time-of-day/weekend bucket "
     "as this one, Laplace-smoothed - low share means this actor is acting at a time they "
     "essentially never have before."),
    ("sem_actor_never_active_this_bucket", "actor",
     "Whether this actor has zero prior alerts, before smoothing, in this time-of-day/weekend "
     "bucket - a sharper, binary companion to the smoothed share above."),
    ("sem_chain_tactic_diversity_15m", "actor_host",
     "How many distinct MITRE tactics this actor has triggered on this host in the trailing 15 "
     "minutes, including this alert - a lone Discovery alert and one inside a burst of three "
     "different tactics are not the same event."),
    ("sem_chain_discovery_to_escalation_15m", "actor_host",
     "Whether a Credential-Access- or Lateral-Movement-tactic alert followed a Discovery-tactic "
     "alert from this same actor and host within the trailing 15 minutes - the specific sequence "
     "an analyst would call an active intrusion rather than routine noise."),
    ("sem_process_chain_novel_for_host", "process_chain",
     "Whether this exact parent-process -> child-process pair has been seen on this host before "
     "- the same launcher running the same command every day is a different signal than a new "
     "parent spawning it for the first time."),
    ("sem_actor_identity_missing_subject", "actor",
     "Whether subjectUserName was empty on this alert and actor identity had to fall back to a "
     "different field - on real data this field is empty on the vast majority of alerts, so "
     "whether it's populated is itself informative, not just a parsing nuisance."),
    ("sem_actor_identity_host_fallback", "actor",
     "Whether even the fallback identity field was unusable and actor identity had to collapse "
     "to the host itself - every such alert from one host shares one pseudo-actor, a real loss "
     "of who-level resolution that downstream actor-keyed features inherit, not a cosmetic default."),
)

FEATURE_NAMES: tuple[str, ...] = tuple(name for name, _, _ in FEATURES)

#: The one extra feature the multi-technique modes add. Deliberately *not*
#: folded into `FEATURES` above: on a single-technique corpus it is a
#: constant column, and a constant column still perturbs a fitted
#: LogisticRegression (measured on v1: PR-AUC 0.975616 -> 0.975429), which
#: would silently restate every published v1 number. `feature_names(mode)`
#: is what callers should use; `FEATURE_NAMES` stays exactly what it was.
MULTI_TECHNIQUE_FEATURE: tuple[str, str, str] = (
    "sem_alert_technique_multiplicity", "alert",
    "How many distinct MITRE techniques this single alert was tagged with - on real data 57.7% "
    "of process-creation alerts carry more than one, and an alert that trips several techniques "
    "at once is a structurally different event from one that trips exactly one, which is worth "
    "letting the model see rather than discarding along with the extra technique values.",
)

#: How a technique-keyed feature collapses to one number when the alert
#: carries several techniques (mode "aggregate" only).
#:
#: `max` everywhere would be wrong, and the direction matters more than the
#: choice of function. These features do not share an orientation: a *high*
#: `sem_freq_*` means "this is a well-established, already-explained
#: pattern" (benign-leaning), while a *high* `sem_novel_*` means "never seen
#: before" (suspicious-leaning). Collapsing both with `max` would take the
#: most-explained technique's frequency - hiding a rare technique behind a
#: noisy one, exactly the information loss this whole change exists to fix.
#: Each feature is therefore aggregated toward its own least-explained end:
#: `min` for frequencies, `max` for novelty. Stated as a design decision
#: because it is one - "if any technique on this alert looks unexplained,
#: the alert is worth a look" is a policy, not a fact about the data.
_AGGREGATION = {
    "sem_freq_actor_host_technique_24h": min,
    "sem_freq_actor_technique_7d": min,
    "sem_novel_actor_host_technique": max,
}

#: Optional columns carrying the full multi-valued fields. Absent on v1's
#: corpus (`synthetic.py` is single-technique by construction and is not
#: modified), present on real data (`opensearch_client._hit_to_row`) and on
#: `synthetic_v2.py`'s calibrated corpus. Absence is handled, not an error.
TECHNIQUE_ALL_COLUMN = "rule_mitre_technique_all"
TACTIC_ALL_COLUMN = "rule_mitre_tactic_all"


def feature_names(mode: str | None = None) -> tuple[str, ...]:
    """The `sem_*` columns `build_features` produces under `mode`. Only the
    multi-technique modes carry `sem_alert_technique_multiplicity`; see that
    constant's note for why it is not unconditional."""
    mode = mode or config.MULTI_TECHNIQUE_MODE
    if mode not in config.MULTI_TECHNIQUE_MODES:
        raise ValueError(
            f"unknown multi-technique mode {mode!r}; expected one of {config.MULTI_TECHNIQUE_MODES}"
        )
    if mode == "first_only":
        return FEATURE_NAMES
    return FEATURE_NAMES + (MULTI_TECHNIQUE_FEATURE[0],)

_FREQ_WINDOW_24H_S = config.FREQ_SHORT_WINDOW_HOURS * 3600.0
_FREQ_WINDOW_7D_S = config.FREQ_LONG_WINDOW_DAYS * 86400.0
_CHAIN_WINDOW_S = config.CHAIN_WINDOW_MINUTES * 60.0
_PRIOR = config.TEMPORAL_PRIOR_COUNT
_N_BUCKETS = len(config.HOUR_BANDS) * 2  # x weekday/weekend


def _hour_bucket(ts: pd.Timestamp) -> int:
    hour = ts.hour
    band_index = 0
    for i, (start, end, _label) in enumerate(config.HOUR_BANDS):
        if start <= hour < end:
            band_index = i
            break
    weekend = 1 if ts.dayofweek >= 5 else 0
    return band_index * 2 + weekend


def _epoch_seconds(timestamps: pd.Series) -> np.ndarray:
    """Unix epoch seconds for `timestamps`, regardless of which of two
    shapes the column actually arrives in.

    The first real-data run raised `TypeError: int() argument must be a
    string, a bytes-like object or a real number, not 'Timestamp'` right
    here. The literal cause: a uniform `datetime64[ns]` column (what a
    homogeneous batch - every row `synthetic.py` produces - gives you) lets
    `.astype("int64")` run as a fast, vectorised C-level cast. A column that
    mixes timezone-*aware* and timezone-*naive* `pandas.Timestamp` objects
    cannot be represented as one `datetime64` dtype, so pandas leaves it as
    plain `object` dtype instead - and `.astype("int64")` on an `object`
    column falls back to calling Python's `int()` on every element one at a
    time, which is exactly what raised on a `Timestamp` instance. This is
    not hypothetical: different alert sources in the same real OpenSearch
    batch (`opensearch_client.hits_to_frame`, built from Sysmon, Suricata and
    KES documents together) format their `timestamp` field with inconsistent
    UTC-offset information, which is what actually produced the mixed
    column - not, as first suspected, a raw numeric-epoch value slipping in.
    That case is handled too, since it costs nothing extra to cover: a
    numeric `timestamps` column is assumed to already be epoch seconds and
    is returned as-is, no datetime parsing attempted.
    """
    if pd.api.types.is_numeric_dtype(timestamps):
        return timestamps.to_numpy(dtype="float64")
    return pd.to_datetime(timestamps, utc=True).astype("int64").to_numpy() / 1e9


def _is_usable(value: object) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _parse_domain_user(raw: str) -> tuple[str | None, str | None]:
    """Splits Sysmon's usual `"DOMAIN\\user"` `User` field shape into
    `(user, domain)`. Falls back to `(raw, None)` for the plain-username
    shape some events use (no backslash - e.g. local `SYSTEM`)."""
    if "\\" in raw:
        domain, _, user = raw.partition("\\")
        return (user or None), (domain or None)
    return raw, None


def _resolve_actor(
    subject_user: object, subject_domain: object, eventdata_user: object, host: str,
) -> tuple[tuple[str, str | None], bool, bool]:
    """`((user, domain), missing_subject, host_fallback)` - see the module
    docstring's "Actor identity" section for the three tiers and why they
    exist. `missing_subject` and `host_fallback` feed `sem_actor_identity_missing_subject`
    and `sem_actor_identity_host_fallback` directly; they are properties of
    this one alert, not aggregated history, so they need no causal state."""
    if _is_usable(subject_user):
        domain = subject_domain if _is_usable(subject_domain) else None
        return (subject_user, domain), False, False  # type: ignore[return-value]
    if _is_usable(eventdata_user):
        user, domain = _parse_domain_user(eventdata_user)  # type: ignore[arg-type]
        if _is_usable(user):
            return (user, domain), True, False
    return (f"(host:{host})", None), True, True


class _ActorState:
    __slots__ = ("hosts_ever", "time_hist", "time_total")

    def __init__(self) -> None:
        self.hosts_ever: set[str] = set()
        self.time_hist: Counter = Counter()
        self.time_total = 0


class SemanticState:
    """The entity graph, advanced one alert at a time.

    One instance covers one alert stream. `advance` writes the features for
    every process-creation row - computed from everything strictly before it
    - then folds that row into the state, so a second call continues exactly
    where the first left off.
    """

    def __init__(self) -> None:
        self.actor_state: dict[tuple[str, str], _ActorState] = {}
        self.actors_ever: set[tuple[str, str]] = set()
        self.aht_dq: dict[tuple, deque] = {}       # actor_host_technique -> deque[ts]
        self.aht_ever: set[tuple] = set()
        self.at_dq: dict[tuple, deque] = {}         # actor_technique -> deque[ts]
        self.chain_dq: dict[tuple, deque] = {}        # (actor,host) -> deque[(ts, tactic)]
        self.process_chain_ever: dict[tuple, set] = {}  # host -> set[(parent_image, image)]

    def _row_features(
        self, now: float, now_ts: pd.Timestamp, actor: tuple[str, str], host: str,
        techniques: list, tactics: list, parent_image: str | None, image: str,
        missing_subject: bool, host_fallback: bool, mode: str,
    ) -> dict[str, float]:
        """`techniques`/`tactics` are the alert's full value lists, always
        non-empty (a null field arrives as `[None]`), with the legacy
        single-value column's value guaranteed first - see `_value_lists`.

        `mode` selects two independent things: whether this alert's own
        secondary techniques contribute to *its own* feature values (the
        read), and whether they are folded into the causal state at all
        (the write). `first_only` does neither, reproducing the pre-fix
        behaviour bit-for-bit so it can serve as an honest baseline.
        """
        read_all = mode == "aggregate"
        fold_all = mode != "first_only"
        read_techniques = techniques if read_all else techniques[:1]
        read_tactics = tactics if read_all else tactics[:1]

        out: dict[str, float] = {}
        # A property of this alert's own fields, not of history - see
        # `_resolve_actor` and the module docstring's "Actor identity" section.
        out["sem_actor_identity_missing_subject"] = 1.0 if missing_subject else 0.0
        out["sem_actor_identity_host_fallback"] = 1.0 if host_fallback else 0.0
        if mode != "first_only":
            # The count is of the whole alert, not of `read_techniques` -
            # in "primary_plus_count" the secondary techniques deliberately
            # do not move any other feature, and this is the one place they
            # are visible to the model at all.
            out[MULTI_TECHNIQUE_FEATURE[0]] = float(len(techniques))

        # -- frequency + novelty, per technique, then collapsed -----------
        # Read every technique the mode asks for, then aggregate each
        # feature in its own direction (see `_AGGREGATION`). With one
        # technique this is exactly the old single-value computation.
        per_technique: list[dict[str, float]] = []
        for technique in read_techniques:
            key_aht = (actor, host, technique)
            dq = self.aht_dq.get(key_aht)
            if dq is not None:
                while dq and now - dq[0] > _FREQ_WINDOW_24H_S:
                    dq.popleft()
            key_at = (actor, technique)
            dq2 = self.at_dq.get(key_at)
            if dq2 is not None:
                while dq2 and now - dq2[0] > _FREQ_WINDOW_7D_S:
                    dq2.popleft()
            per_technique.append({
                "sem_freq_actor_host_technique_24h": float(len(dq)) if dq else 0.0,
                "sem_novel_actor_host_technique": 0.0 if key_aht in self.aht_ever else 1.0,
                "sem_freq_actor_technique_7d": float(len(dq2)) if dq2 else 0.0,
            })
        for name, collapse in _AGGREGATION.items():
            out[name] = collapse(values[name] for values in per_technique)

        # -- actor / actor_host novelty and timing -------------------------
        ast = self.actor_state.get(actor)
        out["sem_novel_actor"] = 0.0 if actor in self.actors_ever else 1.0
        out["sem_novel_host_for_actor"] = (
            0.0 if (ast is not None and host in ast.hosts_ever) else 1.0
        )
        bucket = _hour_bucket(now_ts)
        if ast is not None and ast.time_total > 0:
            raw_count = ast.time_hist.get(bucket, 0)
            out["sem_actor_time_typicality"] = (
                (raw_count + _PRIOR) / (ast.time_total + _PRIOR * _N_BUCKETS)
            )
            out["sem_actor_never_active_this_bucket"] = 0.0 if raw_count > 0 else 1.0
        else:
            out["sem_actor_time_typicality"] = 1.0 / _N_BUCKETS
            out["sem_actor_never_active_this_bucket"] = 1.0

        # -- chain: tactic diversity + discovery -> escalation, 15m --------
        key_ah = (actor, host)
        cdq = self.chain_dq.get(key_ah)
        if cdq is not None:
            while cdq and now - cdq[0][0] > _CHAIN_WINDOW_S:
                cdq.popleft()
        tactics_in_window = {t for _, t in cdq} if cdq else set()
        # This is the feature the single-technique reduction hurt most
        # directly: its whole purpose is counting distinct tactics, and
        # under `first_only` an alert tagged Discovery *and* Execution
        # contributes exactly one of them - the diversity it exists to
        # measure is destroyed before it is measured.
        out["sem_chain_tactic_diversity_15m"] = float(len(tactics_in_window | set(read_tactics)))
        # "Before" stays strictly prior: an alert carrying Discovery and an
        # escalation tactic *simultaneously* is not a Discovery-then-
        # escalation sequence, and counting it as one would quietly change
        # what this feature means rather than fix what it could not see.
        saw_discovery_before = any(
            t == "Discovery" for ts_, t in (cdq or []) if ts_ <= now
        )
        escalation_now = any(t in config.ESCALATION_TACTICS for t in read_tactics)
        out["sem_chain_discovery_to_escalation_15m"] = (
            1.0 if (saw_discovery_before and escalation_now) else 0.0
        )

        # -- process chain novelty, per host --------------------------------
        pc_key = (parent_image, image)
        seen_pairs = self.process_chain_ever.get(host)
        out["sem_process_chain_novel_for_host"] = (
            0.0 if (seen_pairs is not None and pc_key in seen_pairs) else 1.0
        )

        # -- fold this alert into the state ---------------------------------
        # Every technique the alert carried, not just the first: these
        # counters and sets are what the *next* alert reads, so dropping a
        # technique here is not a reporting simplification, it makes the
        # state wrong going forward. Under `first_only` only the primary is
        # folded, preserving the old (incorrect) behaviour as a baseline.
        for technique in (techniques if fold_all else techniques[:1]):
            key_aht = (actor, host, technique)
            dq = self.aht_dq.get(key_aht)
            if dq is None:
                dq = deque()
                self.aht_dq[key_aht] = dq
            dq.append(now)
            self.aht_ever.add(key_aht)

            key_at = (actor, technique)
            dq2 = self.at_dq.get(key_at)
            if dq2 is None:
                dq2 = deque()
                self.at_dq[key_at] = dq2
            dq2.append(now)

        if ast is None:
            ast = _ActorState()
            self.actor_state[actor] = ast
        ast.hosts_ever.add(host)
        ast.time_hist[bucket] += 1
        ast.time_total += 1
        self.actors_ever.add(actor)

        if cdq is None:
            cdq = deque()
            self.chain_dq[key_ah] = cdq
        for tactic in (tactics if fold_all else tactics[:1]):
            cdq.append((now, tactic))

        if seen_pairs is None:
            seen_pairs = set()
            self.process_chain_ever[host] = seen_pairs
        seen_pairs.add(pc_key)

        return out

    def advance(self, frame: pd.DataFrame, mode: str | None = None) -> pd.DataFrame:
        """`frame` must already be sorted by timestamp. Returns a copy of the
        process-creation rows with `sem_*` columns added."""
        mode = mode or config.MULTI_TECHNIQUE_MODE
        names = feature_names(mode)  # also validates `mode`
        work = frame.loc[frame["event_category"] == "process_creation"].copy()
        if work.empty:
            for name in names:
                work[name] = pd.Series(dtype="float64")
            return work

        ts_seconds = _epoch_seconds(work["timestamp"])
        hosts = work["agent_name"].tolist()
        resolved = [
            _resolve_actor(subject_user, subject_domain, eventdata_user, host)
            for subject_user, subject_domain, eventdata_user, host in zip(
                work["subject_user_name"], work["subject_domain_name"],
                work["eventdata_user"], hosts,
            )
        ]
        actors = [actor for actor, _, _ in resolved]
        missing_subject_flags = [missing for _, missing, _ in resolved]
        host_fallback_flags = [fallback for _, _, fallback in resolved]
        techniques = _value_lists(work, TECHNIQUE_ALL_COLUMN, "rule_mitre_technique")
        tactics = _value_lists(work, TACTIC_ALL_COLUMN, "rule_mitre_tactic")
        parent_images = work["parent_image"].tolist()
        # The alert's own `image` field (data.win.eventdata.image on real Sysmon
        # data), not derived from the technique - a lookup through
        # config.TECHNIQUE_CATALOG would KeyError on any real MITRE technique ID
        # outside the eight this prototype's synthetic generator invented, which
        # is every technique the real cluster tags that this prototype doesn't
        # score. Missing/null `image` (some rule configurations don't log it)
        # falls back to a constant placeholder rather than crashing or using
        # `None` as a dict key - repeated unknown-image launches on one host
        # still count as "seen before" for `sem_process_chain_novel_for_host`,
        # which is a reasonable degradation, not a silent wrong answer.
        images = work["image"].fillna("(unknown-image)").tolist()
        timestamps = work["timestamp"].tolist()

        rows_out: list[dict[str, float]] = []
        for i in range(len(work)):
            rows_out.append(self._row_features(
                now=float(ts_seconds[i]), now_ts=timestamps[i], actor=actors[i],
                host=hosts[i], techniques=techniques[i], tactics=tactics[i],
                parent_image=parent_images[i], image=images[i],
                missing_subject=missing_subject_flags[i], host_fallback=host_fallback_flags[i],
                mode=mode,
            ))

        feat_df = pd.DataFrame(rows_out, columns=list(names), index=work.index)
        return pd.concat([work, feat_df], axis=1)


def _value_lists(work: pd.DataFrame, all_column: str, scalar_column: str) -> list[list]:
    """One list of values per row, from the multi-valued `_all` column when
    the corpus carries it and from the legacy single-value column when it
    does not (v1's generator, and any real index predating the change).

    Two invariants the rest of this module relies on:

    * the list is never empty - a null field becomes `[None]`, so
      `techniques[0]` is always addressable and a null technique still keys
      the state exactly as it did before;
    * `list[0]` is the legacy scalar column's own value whenever that value
      is present, so `mode="first_only"` reading `list[0]` is bit-identical
      to the pre-change code reading the scalar column directly. Without
      this the baseline arm would not actually be the old behaviour, and
      every before/after comparison built on it would be measuring two
      changes at once.
    """
    scalars = work[scalar_column].tolist()
    if all_column not in work.columns:
        return [[scalar] for scalar in scalars]

    out: list[list] = []
    for values, scalar in zip(work[all_column].tolist(), scalars):
        items = (
            list(dict.fromkeys(v for v in values if v is not None))
            if isinstance(values, (list, tuple)) else []
        )
        if not items:
            out.append([scalar])
            continue
        if scalar in items:
            items.remove(scalar)
            items.insert(0, scalar)
        elif scalar is not None:
            items.insert(0, scalar)
        out.append(items)
    return out


def build_features(frame: pd.DataFrame, mode: str | None = None) -> pd.DataFrame:
    """One alert stream in, the process-creation subset with `sem_*` columns
    added out. `frame` must be sorted by `timestamp` ascending.

    `mode` selects the multi-valued-technique handling and defaults to
    `config.MULTI_TECHNIQUE_MODE` - see that constant for the three modes
    and why `first_only` is still the default."""
    filters.assert_no_excluded_nodes(frame)
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("semantic.build_features requires a frame sorted by timestamp")
    return SemanticState().advance(frame, mode=mode)


def main() -> int:
    from soc_runtime.synthetic import generate_alerts

    frame = generate_alerts()
    featured = build_features(frame)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    catalogue = {
        "feature_count": len(FEATURE_NAMES),
        # Which multi-valued-technique mode these statistics were computed
        # under. Recorded because three of the features below take
        # different values under a different mode (see this module's
        # docstring), so a bare number here would be ambiguous.
        "multi_technique_mode": config.MULTI_TECHNIQUE_MODE,
        "windows": {
            "freq_short_hours": config.FREQ_SHORT_WINDOW_HOURS,
            "freq_long_days": config.FREQ_LONG_WINDOW_DAYS,
            "chain_minutes": config.CHAIN_WINDOW_MINUTES,
        },
        "features": [
            {
                "name": name, "entity": entity, "meaning": meaning,
                "nonzero_share": round(float((featured[name] != 0).mean()), 6),
                "mean": round(float(featured[name].mean()), 6),
            }
            for name, entity, meaning in FEATURES
        ],
    }
    (config.ARTIFACTS_DIR / "semantic_catalogue.json").write_text(
        json.dumps(catalogue, indent=2), encoding="utf-8"
    )
    print(f"{len(featured):,} process-creation alerts featured, {len(FEATURE_NAMES)} sem_* columns")
    print(f"-> {config.ARTIFACTS_DIR / 'semantic_catalogue.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
