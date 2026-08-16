"""A second synthetic generator, calibrated against real aggregate statistics.

`synthetic.py` (v1) is **not touched by this file** and remains the
documented, reproducible baseline every number already published in the
README's Validation and Confidence-routing sections was measured against.
This module is a separate, additional generator - not a replacement -
built after real access confirmed several things about the real alert
stream's *shape* that v1 never modeled: `subject_user_name` null on
~99.57% of alerts (not ~0%, as v1 assumes), ~57.7% of alerts carrying more
than one MITRE technique tag (not 0%), ~12.36% with no MITRE tag at all,
and a technique-frequency distribution dominated by a small number of
extremely repetitive (actor, host, technique) triples whose 24h trailing
count sits in the hundreds, not single digits.

**This is still synthetic.** Calibrating shape parameters against real
aggregates makes the corpus's *statistical form* more defensible than v1's
arbitrarily-chosen parameters - it does not supply ground truth, and the
`is_synthetic_anomaly` labels here are exactly as fabricated as v1's. See
the README's "Calibration against real aggregates" section for exactly
which numbers came from where, and `config.REAL_CALIBRATION_TARGETS` for
the single source of truth this module calibrates against.

Every calibration input is an aggregated, non-personal statistic (a share,
a mean, a median, a count) - never a raw record, name, IP, or specific host.

Run:
    python -m soc_runtime.synthetic_v2
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from soc_runtime import config
from soc_runtime.opensearch_client import (
    all_of_multivalued, first_of_multivalued, translate_technique_name,
)
from soc_runtime.synthetic import COLUMNS, _guid, _host_ip, _host_node, _uid

_TARGETS = config.REAL_CALIBRATION_TARGETS

#: v1's schema plus the two optional multi-valued columns, in the same
#: shape `opensearch_client._hit_to_row` produces for real documents.
#: `synthetic.COLUMNS` itself is left alone - v1 is single-technique by
#: construction and must keep producing exactly the frame its published
#: numbers were measured on.
COLUMNS_V2: tuple[str, ...] = COLUMNS + ("rule_mitre_technique_all", "rule_mitre_tactic_all")

# --------------------------------------------------------------------------
# The 20 real technique names observed in the real 30-day window, each with
# its canonical MITRE tactic (not this SOC's own quirky per-alert tagging,
# which is only independently confirmed for the original six - see
# config.py's TECHNIQUE_CATALOG note on the T1059/T1105 quirk). "Tool" gets
# no tactic: it is not a real technique at all (see config.py), so
# assigning it a plausible-looking tactic would just be a second guess on
# top of the first.
#
# `in_scope`: True for the three names that translate to an ID inside
# `config.IN_SCOPE_TECHNIQUES` exactly (`Account Discovery` -> T1087,
# `Command and Scripting Interpreter` -> T1059, `System Owner/User
# Discovery` -> T1033). Sub-techniques (`PowerShell` -> T1059.001,
# `Windows Command Shell` -> T1059.003, etc.) do *not* count as in-scope
# here, on purpose - `modeling.restrict_to_scored_techniques` does an exact
# string match against `config.IN_SCOPE_TECHNIQUES`, so neither does the
# real pipeline. Mirroring that exactly, not "fixing" it, is the point.
#
# `weight`: calibrated only at the aggregate level. The three in_scope=True
# names get a *small* combined weight here on purpose: most of
# `in_scope_six_technique_share` (target 34.82%) is deliberately supplied
# by `DOMINANT_DRIVERS` below (a small number of extremely repetitive
# triples), not by this pool - see the worked accounting in this module's
# docstring-adjacent comment above `DOMINANT_DRIVERS`. Individual weights
# among the other 17 names are illustrative, not calibrated - the real
# report gave the 20 names with no frequencies ("без частот"), so anything
# beyond the in-scope aggregate share is an uncalibrated modelling choice,
# stated as exactly that.
# --------------------------------------------------------------------------

TECHNIQUE_POOL: tuple[dict, ...] = (
    {"name": "System Owner/User Discovery", "tactic": "Discovery", "in_scope": True, "weight": 2.8},
    {"name": "Command and Scripting Interpreter", "tactic": "Execution", "in_scope": True, "weight": 2.4},
    {"name": "Account Discovery", "tactic": "Discovery", "in_scope": True, "weight": 1.75},
    {"name": "PowerShell", "tactic": "Execution", "in_scope": False, "weight": 8.0},
    {"name": "Windows Command Shell", "tactic": "Execution", "in_scope": False, "weight": 6.0},
    {"name": "Remote System Discovery", "tactic": "Discovery", "in_scope": False, "weight": 6.0},
    {"name": "Network Share Discovery", "tactic": "Discovery", "in_scope": False, "weight": 5.0},
    {"name": "System Network Configuration Discovery", "tactic": "Discovery", "in_scope": False, "weight": 5.0},
    {"name": "Local Account", "tactic": "Discovery", "in_scope": False, "weight": 4.0},
    {"name": "Scheduled Task", "tactic": "Execution", "in_scope": False, "weight": 4.0},
    {"name": "Obfuscated Files or Information", "tactic": "Defense Evasion", "in_scope": False, "weight": 3.0},
    {"name": "Process Injection", "tactic": "Defense Evasion", "in_scope": False, "weight": 3.0},
    {"name": "Regsvr32", "tactic": "Defense Evasion", "in_scope": False, "weight": 2.5},
    {"name": "Hidden Window", "tactic": "Defense Evasion", "in_scope": False, "weight": 2.0},
    {"name": "File Deletion", "tactic": "Defense Evasion", "in_scope": False, "weight": 2.0},
    {"name": "Application Shimming", "tactic": "Persistence", "in_scope": False, "weight": 1.5},
    {"name": "Remote Services", "tactic": "Lateral Movement", "in_scope": False, "weight": 1.5},
    {"name": "Compile After Delivery", "tactic": "Defense Evasion", "in_scope": False, "weight": 1.0},
    {"name": "Exfiltration Over Web Service", "tactic": "Exfiltration", "in_scope": False, "weight": 1.0},
    {"name": "Tool", "tactic": None, "in_scope": False, "weight": 1.0},
)

_POOL_NAMES = [t["name"] for t in TECHNIQUE_POOL]
_POOL_WEIGHTS = np.array([t["weight"] for t in TECHNIQUE_POOL], dtype="float64")
_POOL_WEIGHTS = _POOL_WEIGHTS / _POOL_WEIGHTS.sum()
_POOL_TACTIC = {t["name"]: t["tactic"] for t in TECHNIQUE_POOL}

assert set(TECHNIQUE_POOL[i]["name"] for i in range(len(TECHNIQUE_POOL))) <= set(config.TECHNIQUE_NAME_TO_ID) | {"Tool"}, (
    "every name in TECHNIQUE_POOL must be a real observed name - either in "
    "config.TECHNIQUE_NAME_TO_ID or the deliberately-unmapped 'Tool'"
)


# --------------------------------------------------------------------------
# Population
#
# Two tiers, matched to the shape the real frequency figures imply (median
# 641 > mean 462 on the *scored* subset - see `measured_calibration`, which
# restricts these two stats to `modeling.restrict_to_scored_techniques`
# exactly as the real report's "Feature distributions (scored subset)"
# bullet did): a couple of "dominant" identities firing many times a day on
# one fixed host with one fixed in-scope technique, and a wide, much
# lower-rate "tail" of fixed (actor, host, technique) slots.
#
# Rough accounting behind the constants below (worked once, then verified
# empirically via `tests/test_synthetic_v2.py` rather than trusted blindly):
# target in-scope count over N=59,680 process-creation alerts is
# 0.3482*N ~ 20,780. With the two dominant drivers combined at ~660/day
# over 30 days (~19,800 raw rows, ~17,350 after the uniform 12.36% MITRE-
# null roll) supplying most of that, the remaining ~3,400 in-scope rows
# have to come from the tail's technique pool, which is why
# `TECHNIQUE_POOL`'s three in-scope names carry a *small* combined weight
# (~9.8%) above, not a large one - nearly all of the in-scope share is the
# dominant drivers, matching the real "small number of very repetitive
# triples" shape rather than "in-scope techniques are just common overall".
# --------------------------------------------------------------------------

V2_HOSTS: tuple[str, ...] = tuple(f"SOC-HOST-{i:03d}" for i in range(1, 21))

#: Extremely repetitive (actor, host, technique) driver - the real-world
#: analogue is something like a chatty monitoring/health-check process.
#: Rate is alerts/day, subject to the *same* uniform MITRE-null roll as
#: every other row (see `_process_alert_v2`) rather than an exemption -
#: some of this driver's own alerts land untagged too, which both keeps
#: the overall null rate honest and is itself realistic (not every alert
#: a given process trips necessarily carries a MITRE tag).
#:
#: Deliberately a *single* driver, not several at different rates: with
#: multiple drivers at different steady-state values, the low-rate driver's
#: rows sit between the tail's near-zero values and the high-rate driver's
#: values, which pulled the overall median toward the low driver instead of
#: the intended peak (measured directly while tuning this - see
#: `tests/test_synthetic_v2.py`). One driver keeps its rows a single tight
#: cluster, so as long as it's a majority of the scored subset the overall
#: median lands on its own steady-state value, close to the 641 target.
DOMINANT_DRIVERS: tuple[dict, ...] = (
    {"user": "svc-healthcheck", "domain": "CORP", "host": "SOC-HOST-001",
     "technique_name": "System Owner/User Discovery", "rate_per_day": 720.0},
)

#: Moderate-frequency background identities - legitimate but not
#: pattern-locked, the same role v1's ADMIN_ACCOUNTS played. Used for
#: anomaly injection (off_hours/new_host) below, not for tail volume -
#: tail volume comes from `TAIL_SLOTS`.
BACKGROUND_ACTORS: tuple[dict, ...] = tuple(
    {"user": f"admin.v2user{i:02d}", "domain": "CORP",
     "hosts": (V2_HOSTS[(2 * i) % len(V2_HOSTS)], V2_HOSTS[(2 * i + 1) % len(V2_HOSTS)])}
    for i in range(8)
)

#: The long tail: a moderate, fixed number of distinct (actor, host,
#: technique) triples, each firing at a low daily rate. Fixed rather than
#: drawn fresh per alert so the *number of distinct triples* stays bounded
#: (~176 here, plus the 2 dominant ones and a handful from anomaly
#: injection) - that bound is what keeps `sem_novel_actor_host_technique`
#: this close to zero on a corpus this size: nearly every alert repeats a
#: triple that has already fired earlier in the window, exactly as in
#: the real measurement. 14 distinct actors x (~13 hosts/techniques worth
#: of slots each) keeps `sem_novel_actor` bounded the same way.
_TAIL_ACTOR_NAMES: tuple[str, ...] = tuple(f"svc.v2tail{i:02d}" for i in range(14))
_TAIL_TECHNIQUE_NAMES: tuple[str, ...] = tuple(
    t["name"] for t in TECHNIQUE_POOL if t["name"] != "Tool"
)


_TAIL_IN_SCOPE_NAMES = {t["name"] for t in TECHNIQUE_POOL if t["in_scope"]}
#: Two separate rates so total tail volume (mostly non-in-scope slots) can
#: be tuned to match the real 59,680-alert total without disturbing the
#: already-calibrated scored-subset frequency stats, which only in-scope
#: slots feed into.
_TAIL_RATE_IN_SCOPE = 5.0
_TAIL_RATE_OTHER = 7.8


def _build_tail_slots() -> tuple[dict, ...]:
    slots: list[dict] = []
    for i, actor in enumerate(_TAIL_ACTOR_NAMES):
        for j in range(13):
            host = V2_HOSTS[(i * 13 + j) % len(V2_HOSTS)]
            technique_name = _TAIL_TECHNIQUE_NAMES[(i * 13 + j) % len(_TAIL_TECHNIQUE_NAMES)]
            rate = _TAIL_RATE_IN_SCOPE if technique_name in _TAIL_IN_SCOPE_NAMES else _TAIL_RATE_OTHER
            slots.append({
                "user": actor, "domain": "CORP", "host": host,
                "technique_name": technique_name, "rate_per_day": rate,
            })
    return tuple(slots)


TAIL_SLOTS: tuple[dict, ...] = _build_tail_slots()

#: Rare identities reserved for anomaly injection - kept small in number
#: (reused across several anomaly instances rather than one-off) so total
#: distinct-actor count stays low, which is what keeps
#: sem_novel_actor_mean this close to zero on a corpus this size.
RARE_ACTORS_V2: tuple[tuple[str, str], ...] = tuple(
    (f"contractor.v2temp{i:02d}", "CORP") for i in range(1, 9)
)
RARE_HOSTS_V2: tuple[str, ...] = tuple(f"SOC-LAP-{i:04d}" for i in range(9101, 9111))

PARENT_IMAGE = "C:\\Windows\\System32\\cmd.exe"


# --------------------------------------------------------------------------
# Technique drawing: name -> (possibly multi-valued) -> translated ID
# --------------------------------------------------------------------------

def _draw_technique_name(rng: np.random.Generator) -> str:
    return str(rng.choice(_POOL_NAMES, p=_POOL_WEIGHTS))


def _maybe_multivalued(name: str, rng: np.random.Generator) -> list[str]:
    """`[name]`, or `[name, other]` with probability
    `REAL_CALIBRATION_TARGETS["multi_valued_technique_share"]` - a pure,
    directly-testable function so the ~57.7% rate can be checked on its own
    terms (`tests/test_synthetic_v2.py`), since the final flat schema can
    only ever show the *first* element (see `first_of_multivalued`) and
    cannot itself reveal whether a row was originally multi-valued - the
    same real limitation `opensearch_client._hit_to_row` has.
    """
    if rng.random() >= _TARGETS["multi_valued_technique_share"]:
        return [name]
    other = name
    while other == name:
        other = str(rng.choice(_POOL_NAMES, p=_POOL_WEIGHTS))
    return [name, other]


def _resolve_technique(
    rng: np.random.Generator, forced_name: str | None = None,
) -> tuple[str | None, str | None, list, list]:
    """`(technique, tactic, technique_all, tactic_all)` for one alert.

    `None`s with probability `mitre_tag_null_share`, applied uniformly
    whether the name is drawn from `TECHNIQUE_POOL` or `forced_name` is
    given (a dominant-driver or anomaly row): every alert goes through the
    same null roll, matching the real pipeline's assumption that MITRE-tag
    absence is a per-alert property, not a per-source one.

    When tagged, the name is possibly expanded to a multi-valued list and
    reduced via the *same* `first_of_multivalued` + `translate_technique_name`
    functions the real ingestion path uses, not a reimplementation - and,
    since the real-data adapter now also keeps the whole list, so does
    this: the scalar columns carry the reduced first value exactly as
    before, the `_all` columns carry everything, and `semantic.py` decides
    which to use based on its mode. Without the `_all` columns here the
    calibrated corpus would model the 57.7% multi-valued rate in its
    generation and then throw it away before any feature could see it.
    """
    if rng.random() < _TARGETS["mitre_tag_null_share"]:
        return None, None, [], []
    name = forced_name if forced_name is not None else _draw_technique_name(rng)
    raw_list = _maybe_multivalued(name, rng)
    reduced_name = first_of_multivalued(raw_list)
    technique_all = [translate_technique_name(n) for n in all_of_multivalued(raw_list)]
    # "Tool" maps to no tactic at all (it is not a MITRE technique - see
    # config.py), so a tactic list can be shorter than its technique list.
    tactic_all = list(dict.fromkeys(
        t for t in (_POOL_TACTIC[n] for n in all_of_multivalued(raw_list)) if t is not None
    ))
    return translate_technique_name(reduced_name), _POOL_TACTIC[reduced_name], technique_all, tactic_all


# --------------------------------------------------------------------------
# Identity masking: subject_user_name/domain vs eventdata_user
# --------------------------------------------------------------------------

def _mask_identity(
    user: str, domain: str, rng: np.random.Generator, *, force_null: bool = False,
) -> tuple[str | None, str | None, str | None]:
    """`(subject_user_name, subject_domain_name, eventdata_user)` - null the
    subject fields with probability `subject_user_name_null_share` (or
    unconditionally when `force_null` is set - see the in-scope-technique
    call site) and populate `eventdata_user` in Sysmon's `"DOMAIN\\user"`
    shape instead, matching the real finding that every subject-missing
    alert in the measured window had eventdata_user populated
    (`eventdata_user_rescue_share_of_missing == 1.0`)."""
    if force_null or rng.random() < _TARGETS["subject_user_name_null_share"]:
        return None, None, f"{domain}\\{user}"
    return user, domain, None


# --------------------------------------------------------------------------
# Row builder
# --------------------------------------------------------------------------

def _process_alert_v2(
    *, ts: pd.Timestamp, host: str, user: str, domain: str,
    rng: np.random.Generator, is_anomaly: bool = False, anomaly_type: str | None = None,
    forced_technique_name: str | None = None,
) -> dict:
    technique_id, tactic, technique_all, tactic_all = _resolve_technique(
        rng, forced_name=forced_technique_name)

    # The real 30-day measurement found subject_user_name null on *every*
    # scored-technique alert (sem_actor_identity_missing_subject mean ==
    # 1.0 exactly), stronger than the ~99.57% population-wide null rate -
    # plausibly because the in-scope techniques here are dominated by
    # automated/service processes whose Sysmon events never populate
    # subject_user_name in the first place. Forcing the mask for in-scope
    # rows models that directly instead of leaving it to a 99.57% coin
    # flip that would occasionally (mis)populate it.
    force_null = technique_id is not None and technique_id in config.IN_SCOPE_TECHNIQUES
    subject_user, subject_domain, eventdata_user = _mask_identity(user, domain, rng, force_null=force_null)
    image = technique_id or "unknown.exe"
    guid = _guid(rng)
    return {
        "alert_uid": _uid(rng),
        "timestamp": ts,
        "cluster_node": _host_node(host),
        "agent_name": host,
        "agent_ip": _host_ip(host),
        "rule_id": abs(hash(technique_id)) % 100000 if technique_id else 90000,
        "rule_level": 5,
        "rule_description": f"{technique_id or 'untagged'} activity on {host}",
        "rule_mitre_tactic": tactic,
        "rule_mitre_technique": technique_id,
        "rule_mitre_technique_all": technique_all,
        "rule_mitre_tactic_all": tactic_all,
        "event_category": "process_creation",
        "subject_user_name": subject_user,
        "subject_domain_name": subject_domain,
        "eventdata_user": eventdata_user,
        "target_user_name": subject_user or user,
        "target_domain_name": subject_domain or domain,
        "command_line": f"{image} /calibrated-v2",
        "image": image,
        "parent_image": PARENT_IMAGE,
        "parent_command_line": "cmd.exe /k",
        "parent_process_guid": guid,
        "process_guid": _guid(rng),
        "logon_type": None,
        "authentication_package_name": None,
        "is_synthetic_anomaly": is_anomaly,
        "synthetic_anomaly_type": anomaly_type,
    }


# --------------------------------------------------------------------------
# Generators
# --------------------------------------------------------------------------

def _dominant_traffic(day: pd.Timestamp, rng: np.random.Generator) -> list[dict]:
    rows: list[dict] = []
    for driver in DOMINANT_DRIVERS:
        n = int(rng.poisson(driver["rate_per_day"]))
        minutes = rng.uniform(0, 24 * 60, size=n)
        minutes.sort()
        for m in minutes:
            ts = day + pd.Timedelta(minutes=float(m))
            rows.append(_process_alert_v2(
                ts=ts, host=driver["host"], user=driver["user"], domain=driver["domain"],
                rng=rng, forced_technique_name=driver["technique_name"],
            ))
    return rows


def _tail_traffic(day: pd.Timestamp, rng: np.random.Generator) -> list[dict]:
    """Volume from `TAIL_SLOTS` - a bounded set of fixed (actor, host,
    technique) triples firing at a low daily rate each. This is what
    supplies the bulk of total alert volume (v1's role for
    `_daily_auth_noise`), not the dominant drivers, which stay a small
    minority of rows despite dominating the *scored* frequency stats -
    see the accounting comment above `DOMINANT_DRIVERS`."""
    rows: list[dict] = []
    for slot in TAIL_SLOTS:
        n = int(rng.poisson(slot["rate_per_day"]))
        if n == 0:
            continue
        minutes = rng.uniform(0, 24 * 60, size=n)
        for m in minutes:
            ts = day + pd.Timedelta(minutes=float(m))
            rows.append(_process_alert_v2(
                ts=ts, host=slot["host"], user=slot["user"], domain=slot["domain"],
                rng=rng, forced_technique_name=slot["technique_name"],
            ))
    return rows


def _pick_business_ts(day: pd.Timestamp, rng: np.random.Generator) -> pd.Timestamp:
    return day + pd.Timedelta(hours=int(rng.integers(9, 17)), minutes=int(rng.integers(0, 60)))


def _pick_off_hours_ts(day: pd.Timestamp, rng: np.random.Generator) -> pd.Timestamp:
    hour = int(rng.choice([0, 1, 2, 3, 4, 23]))
    return day + pd.Timedelta(hours=hour, minutes=int(rng.integers(0, 60)))


def _inject_anomalies_v2(days: pd.DatetimeIndex, rng: np.random.Generator) -> list[dict]:
    """Four of v1's five anomaly types, scaled down for a 30-day window -
    `tactic_escalation` is deliberately omitted here, not forgotten: the
    real 30-day window measured `sem_chain_discovery_to_escalation_15m`
    at share_nonzero 0.0, i.e. that specific 15-minute Discovery-then-
    escalation sequence was never observed. Matching that absence in the
    default calibrated corpus is the honest choice; the escalation
    *mechanism* itself remains covered by v1's corpus and by
    `tests/test_semantic.py`'s direct unit tests, which do not depend on
    either generator.
    """
    rows: list[dict] = []
    usable_days = days[1:-1]

    for day in rng.choice(usable_days, size=10, replace=True):
        day = pd.Timestamp(day)
        actor = BACKGROUND_ACTORS[int(rng.integers(0, len(BACKGROUND_ACTORS)))]
        host = str(rng.choice(RARE_HOSTS_V2))
        rows.append(_process_alert_v2(
            ts=_pick_business_ts(day, rng), host=host, user=actor["user"], domain=actor["domain"],
            rng=rng, is_anomaly=True, anomaly_type="new_host",
            forced_technique_name="System Owner/User Discovery",
        ))

    for day in rng.choice(usable_days, size=10, replace=True):
        day = pd.Timestamp(day)
        actor = BACKGROUND_ACTORS[int(rng.integers(0, len(BACKGROUND_ACTORS)))]
        host = str(rng.choice(actor["hosts"]))
        rows.append(_process_alert_v2(
            ts=_pick_off_hours_ts(day, rng), host=host, user=actor["user"], domain=actor["domain"],
            rng=rng, is_anomaly=True, anomaly_type="off_hours",
            forced_technique_name="System Owner/User Discovery",
        ))

    for day in rng.choice(usable_days, size=10, replace=True):
        day = pd.Timestamp(day)
        user, domain = RARE_ACTORS_V2[int(rng.integers(0, len(RARE_ACTORS_V2)))]
        host = str(rng.choice(V2_HOSTS + RARE_HOSTS_V2))
        rows.append(_process_alert_v2(
            ts=_pick_business_ts(day, rng), host=host, user=user, domain=domain,
            rng=rng, is_anomaly=True, anomaly_type="new_actor",
            forced_technique_name="Account Discovery",
        ))

    for day in rng.choice(usable_days, size=8, replace=True):
        day = pd.Timestamp(day)
        if rng.random() < 0.6:
            user, domain = RARE_ACTORS_V2[int(rng.integers(0, len(RARE_ACTORS_V2)))]
        else:
            actor = BACKGROUND_ACTORS[int(rng.integers(0, len(BACKGROUND_ACTORS)))]
            user, domain = actor["user"], actor["domain"]
        host = str(rng.choice(RARE_HOSTS_V2 + V2_HOSTS))
        t = _pick_business_ts(day, rng) if rng.random() < 0.7 else _pick_off_hours_ts(day, rng)
        for tech_name in ("System Owner/User Discovery", "Account Discovery", "Local Account"):
            rows.append(_process_alert_v2(
                ts=t, host=host, user=user, domain=domain, rng=rng,
                is_anomaly=True, anomaly_type="manual_recon_chain",
                forced_technique_name=tech_name,
            ))
            t = t + pd.Timedelta(seconds=int(rng.integers(20, 90)))

    return rows


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def generate_alerts_v2(
    *, n_days: int = 30, end: str | pd.Timestamp | None = None, seed: int = config.SEED,
) -> pd.DataFrame:
    """Same flat schema as `synthetic.generate_alerts` (v1) - `COLUMNS` is
    shared, imported from `synthetic.py`, not redefined - but with
    generation parameters calibrated against `config.REAL_CALIBRATION_TARGETS`
    instead of chosen arbitrarily.

    Defaults to a 30-day window, not v1's 90: that is the exact window the
    real calibration aggregates were measured over, and stretching either
    corpus to match the other's window length would make the "same
    methodology, different scale" comparison in `run_pipeline.py`'s v1-vs-v2
    report less honest, not more.
    """
    rng = np.random.default_rng(seed)
    end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp(config.SYNTHETIC_END_DATE)
    days = pd.date_range(end=end_ts, periods=n_days, freq="D")

    rows: list[dict] = []
    for day in days:
        rows.extend(_dominant_traffic(day, rng))
        rows.extend(_tail_traffic(day, rng))
    rows.extend(_inject_anomalies_v2(days, rng))

    frame = pd.DataFrame(rows, columns=list(COLUMNS_V2))
    frame = frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    assert not (set(frame["cluster_node"].unique()) & config.EXCLUDED_CLUSTER_NODES), (
        "synthetic_v2 produced an excluded cluster node - a bug in _host_node, not real data."
    )
    return frame


def measured_calibration(frame: pd.DataFrame) -> dict:
    """The generated corpus's own statistics, in the same units as
    `config.REAL_CALIBRATION_TARGETS`, for honest side-by-side reporting -
    not asserted to match, measured and printed.

    The real report's "Feature distributions" numbers were explicitly
    measured on the *scored subset* (`modeling.restrict_to_scored_techniques`
    - i.e. alerts whose technique is one of the six in-scope IDs), not on
    every process-creation alert - so the sem_* stats here are computed on
    `scored`, matching that scope exactly. `subject_user_name_null_share`,
    `mitre_tag_null_share` and `in_scope_six_technique_share` are population-
    wide stats instead, matching how the real report presented those three.
    """
    from soc_runtime import baseline, modeling, semantic

    process = frame[frame["event_category"] == "process_creation"]
    featured = baseline.build_features(semantic.build_features(frame))
    scored = modeling.restrict_to_scored_techniques(featured)

    out = {
        "n_process_creation": int(len(process)),
        "subject_user_name_null_share": round(float(process["subject_user_name"].isna().mean()), 4),
        "mitre_tag_null_share": round(float(process["rule_mitre_technique"].isna().mean()), 4),
        "in_scope_six_technique_share": round(
            float(process["rule_mitre_technique"].isin(config.IN_SCOPE_TECHNIQUES).mean()), 4
        ),
        "n_scored": int(len(scored)),
    }
    if not scored.empty:
        out["sem_freq_actor_host_technique_24h_mean"] = round(float(scored["sem_freq_actor_host_technique_24h"].mean()), 2)
        out["sem_freq_actor_host_technique_24h_median"] = round(float(scored["sem_freq_actor_host_technique_24h"].median()), 2)
        out["sem_novel_actor_host_technique_mean"] = round(float(scored["sem_novel_actor_host_technique"].mean()), 4)
        out["sem_novel_actor_mean"] = round(float(scored["sem_novel_actor"].mean()), 4)
        out["sem_chain_discovery_to_escalation_15m_share_nonzero"] = round(
            float((scored["sem_chain_discovery_to_escalation_15m"] != 0).mean()), 4
        )
        out["sem_actor_identity_missing_subject_mean"] = round(float(scored["sem_actor_identity_missing_subject"].mean()), 4)
    return out


def main() -> int:
    frame = generate_alerts_v2()
    n_anom = int(frame["is_synthetic_anomaly"].sum())
    print(f"{len(frame):,} synthetic v2 alerts generated "
          f"({frame['event_category'].value_counts().to_dict()})")
    print(f"{n_anom:,} labeled synthetic anomalies "
          f"({frame.loc[frame['is_synthetic_anomaly'], 'synthetic_anomaly_type'].value_counts().to_dict()})")

    measured = measured_calibration(frame)
    print("\nmeasured vs real calibration targets:")
    pairs = (
        ("subject_user_name_null_share", "subject_user_name_null_share"),
        ("mitre_tag_null_share", "mitre_tag_null_share"),
        ("in_scope_six_technique_share", "in_scope_six_technique_share"),
        ("sem_freq_actor_host_technique_24h_mean", "sem_freq_actor_host_technique_24h_mean"),
        ("sem_freq_actor_host_technique_24h_median", "sem_freq_actor_host_technique_24h_median"),
        ("sem_novel_actor_host_technique_mean", "sem_novel_actor_host_technique_mean"),
        ("sem_novel_actor_mean", "sem_novel_actor_mean"),
        ("sem_chain_discovery_to_escalation_15m_share_nonzero", "sem_chain_discovery_to_escalation_15m_share_nonzero"),
        ("sem_actor_identity_missing_subject_mean", "sem_actor_identity_missing_subject_mean"),
    )
    for measured_key, target_key in pairs:
        m = measured.get(measured_key)
        t = _TARGETS.get(target_key)
        print(f"  {measured_key:<48} measured={m!s:<10} target={t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
