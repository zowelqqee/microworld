"""Synthetic Wazuh/OpenSearch alert generator.

There is no live OpenSearch access yet (see README), so this module stands in
for `opensearch_client.fetch_alerts` during development: same schema, same
column names, a fabricated 90-day alert stream instead of a real one.

**The anomaly labels this module attaches (`is_synthetic_anomaly`,
`synthetic_anomaly_type`) are fabricated ground truth for testing the
architecture. They are not a claim about what a real attack looks like.**
They exist so `soc_runtime.modeling` has something to score precision/recall
against before real, incident-derived labels are available. See the README's
Limitations section.

Column mapping (real Wazuh field -> flat column used here and by
`opensearch_client`, which is responsible for producing the same flattening
from a live cluster):

    agent.name                                  -> agent_name
    agent.ip                                     -> agent_ip
    rule.id                                       -> rule_id
    rule.level                                     -> rule_level
    rule.description                                -> rule_description
    rule.mitre.tactic                                -> rule_mitre_tactic
    rule.mitre.technique                              -> rule_mitre_technique
    data.win.eventdata.subjectUserName                 -> subject_user_name
    data.win.eventdata.subjectDomainName                -> subject_domain_name
    data.win.eventdata.user                              -> eventdata_user
    data.win.eventdata.targetUserName                    -> target_user_name
    data.win.eventdata.targetDomainName                   -> target_domain_name
    data.win.eventdata.commandLine                         -> command_line
    data.win.eventdata.image                                -> image
    data.win.eventdata.parentImage                          -> parent_image
    data.win.eventdata.parentCommandLine                     -> parent_command_line
    data.win.eventdata.parentProcessGuid                      -> parent_process_guid
    data.win.eventdata.processGuid                             -> process_guid
    data.win.eventdata.logonType                                -> logon_type
    data.win.eventdata.authenticationPackageName                 -> authentication_package_name
    timestamp                                                     -> timestamp
    cluster.node                                                   -> cluster_node

Plus columns that do not exist in the real index: `alert_uid` (a row key
convenient for joining chained alerts), `event_category`
(`"process_creation"` or `"authentication"`, used to scope the pipeline to
the alerts that carry a subject/host/technique triad), and the two synthetic
label columns above.

`eventdata_user` is a fallback identity field, not part of the original
design - added after the first real run found `subject_user_name` null on
99.48% of real process-creation alerts (this generator always populates it,
so the synthetic corpus does not exercise the fallback path at all; see
`semantic.py`'s actor-resolution note and the README's "Real data" section).
It is always `None` here.

Real Wazuh alerts also carry `rule.mitre.tactic`/`.technique` as *arrays* -
one correlation rule can tag several techniques on one alert. This generator
keeps it to one tactic and one technique per alert to keep the feature layer
simple; `opensearch_client` notes where real ingestion would need to explode
multi-valued alerts into one row per technique before this pipeline can
consume them.
"""

from __future__ import annotations

import hashlib
import uuid

import numpy as np
import pandas as pd

from soc_runtime import config

# --------------------------------------------------------------------------
# Fixed population: the hosts, actors and infrastructure the legitimate
# recurring pattern and the admins operate on. Anomalies deliberately reach
# outside this population (new host, new actor) or misuse it (odd time,
# odd sequence) - see `_inject_anomalies`.
# --------------------------------------------------------------------------

CORE_HOSTS: tuple[str, ...] = tuple(
    [f"CORP-WKS-{i:04d}" for i in range(1, 13)]
    + ["CORP-SRV-FILE01", "CORP-SRV-APP01", "CORP-SRV-DB01"]
)  # 15 hosts - the fleet the software-inventory agent sweeps daily.

SERVICE_ACCOUNT: tuple[str, str] = ("svc-swinventory", "CORP")
SERVICE_PARENT_IMAGE = "C:\\Program Files\\SWInventory\\sw-inventory-agent.exe"

ADMIN_ACCOUNTS: tuple[dict, ...] = (
    {"user": "admin.jsmith", "domain": "CORP", "hosts": ("CORP-WKS-0001", "CORP-SRV-APP01"), "hours": (8, 17)},
    {"user": "admin.pchen", "domain": "CORP", "hosts": ("CORP-WKS-0002", "CORP-SRV-DB01"), "hours": (9, 18)},
    {"user": "admin.rpatel", "domain": "CORP", "hosts": ("CORP-WKS-0003", "CORP-SRV-FILE01"), "hours": (7, 16)},
    {"user": "admin.kwong", "domain": "CORP", "hosts": ("CORP-WKS-0004",), "hours": (9, 17)},
    {"user": "admin.mgarcia", "domain": "CORP", "hosts": ("CORP-WKS-0005", "CORP-SRV-APP01"), "hours": (10, 19)},
)

#: Hosts reserved for anomaly injection - never touched by the legitimate
#: traffic generators, so "this host is new for this actor" is genuine.
RARE_HOSTS: tuple[str, ...] = tuple(
    [f"CORP-LAP-{i:04d}" for i in range(9001, 9021)]
    + [f"UNK-HOST-{i:04d}" for i in range(1, 11)]
)

#: Accounts reserved for anomaly injection - never appear in legitimate
#: traffic, so "this actor is new" is genuine.
RARE_ACTORS: tuple[tuple[str, str], ...] = tuple(
    ("contractor.temp%02d" % i, "CORP") for i in range(1, 16)
) + (
    ("j.doe", "AZUREAD"),  # a personal/cloud-domain account, never seen on-prem
)

#: A broader pool of ordinary employees, used only to pad realistic
#: authentication-log background noise. They never run the process-creation
#: alerts the semantic layer scores.
EMPLOYEE_ACCOUNTS: tuple[str, ...] = tuple(f"user.emp{i:03d}" for i in range(1, 61))

AUTH_PACKAGES = ("Negotiate", "NTLM", "Kerberos")
LOGON_TYPES = (2, 3, 10)  # interactive, network, remote-interactive

_CATALOG = config.TECHNIQUE_CATALOG


# --------------------------------------------------------------------------
# Deterministic helpers
# --------------------------------------------------------------------------

def _stable_hash(*parts: str) -> int:
    """A hash that is the same across processes and across runs.

    Python's built-in `hash()` is salted per-process for strings (hash
    randomisation, on by default) - two separate `python run_pipeline.py`
    invocations with the same `seed` would assign different hosts to
    different collector nodes and different sweep-time offsets, silently
    breaking the "deterministic given seed" guarantee `tests/test_synthetic.py`
    checks. That test only caught same-process calls, since a process's hash
    salt is fixed for its own lifetime. `hashlib` is not salted.
    """
    digest = hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _host_ip(host: str) -> str:
    h = _stable_hash("ip", host) % (250 * 250)
    return f"10.20.{(h // 250) + 1}.{(h % 250) + 2}"


def _host_node(host: str) -> str:
    """Every host reports to the same collector every time - a fixed
    assignment, not a per-alert coin flip, because that is how a real
    collector topology works."""
    return "office-collector" if _stable_hash("node", host) % 10 < 7 else "node01"


def _guid(rng: np.random.Generator) -> str:
    parts = [int(rng.integers(0, 2**32)) for _ in range(4)]
    return "{%08x-%04x-%04x-%04x-%04x%08x}" % (
        parts[0], parts[1] & 0xFFFF, parts[2] & 0xFFFF, parts[3] & 0xFFFF,
        parts[1] >> 16, parts[2],
    )


def _uid(rng: np.random.Generator) -> str:
    return uuid.UUID(int=int(rng.integers(0, 2**63)) << 64 | int(rng.integers(0, 2**63))).hex


# --------------------------------------------------------------------------
# Row builders
# --------------------------------------------------------------------------

def _process_alert(
    *, ts: pd.Timestamp, host: str, user: str, domain: str, technique: str,
    parent_image: str, parent_command_line: str, parent_process_guid: str,
    rng: np.random.Generator, is_anomaly: bool = False, anomaly_type: str | None = None,
) -> dict:
    meta = _CATALOG[technique]
    command_line = meta["command_template"].format(host=host)
    return {
        "alert_uid": _uid(rng),
        "timestamp": ts,
        "cluster_node": _host_node(host),
        "agent_name": host,
        "agent_ip": _host_ip(host),
        "rule_id": meta["rule_id"],
        "rule_level": meta["rule_level"],
        "rule_description": f"{meta['description']} ({meta['image']} executed)",
        "rule_mitre_tactic": meta["tactic"],
        "rule_mitre_technique": technique,
        "event_category": "process_creation",
        "subject_user_name": user,
        "subject_domain_name": domain,
        "eventdata_user": None,  # this generator never exercises the fallback path - see module docstring
        "target_user_name": user,
        "target_domain_name": domain,
        "command_line": command_line,
        "image": meta["image"],
        "parent_image": parent_image,
        "parent_command_line": parent_command_line,
        "parent_process_guid": parent_process_guid,
        "process_guid": _guid(rng),
        "logon_type": None,
        "authentication_package_name": None,
        "is_synthetic_anomaly": is_anomaly,
        "synthetic_anomaly_type": anomaly_type,
    }


def _auth_alert(*, ts: pd.Timestamp, host: str, user: str, domain: str, rng: np.random.Generator) -> dict:
    return {
        "alert_uid": _uid(rng),
        "timestamp": ts,
        "cluster_node": _host_node(host),
        "agent_name": host,
        "agent_ip": _host_ip(host),
        "rule_id": 60106,
        "rule_level": 3,
        "rule_description": "Windows Logon Success.",
        "rule_mitre_tactic": None,
        "rule_mitre_technique": None,
        "event_category": "authentication",
        "subject_user_name": user,
        "subject_domain_name": domain,
        "eventdata_user": None,
        "target_user_name": user,
        "target_domain_name": domain,
        "command_line": None,
        "image": None,
        "parent_image": None,
        "parent_command_line": None,
        "parent_process_guid": None,
        "process_guid": None,
        "logon_type": int(rng.choice(LOGON_TYPES, p=[0.7, 0.2, 0.1])),
        "authentication_package_name": str(rng.choice(AUTH_PACKAGES, p=[0.6, 0.1, 0.3])),
        "is_synthetic_anomaly": False,
        "synthetic_anomaly_type": None,
    }


# --------------------------------------------------------------------------
# Generators for each population
# --------------------------------------------------------------------------

def _daily_inventory_sweep(day: pd.Timestamp, rng: np.random.Generator) -> list[dict]:
    """The recurring legitimate pattern: one service account sweeps the same
    15 hosts every single day, close to the same time, running the same two
    commands. This is deliberately the loudest, most repetitive thing in the
    dataset - it is meant to be what a context-free technique count sees as
    "19,162 Discovery alerts" and a semantic layer sees as one actor."""
    rows: list[dict] = []
    hosts = list(CORE_HOSTS)
    rng.shuffle(hosts)
    for host in hosts:
        offset_min = _stable_hash("offset", host) % 55
        jitter_min = float(np.clip(rng.normal(0, 4), -10, 10))
        start = day + pd.Timedelta(hours=8, minutes=offset_min) + pd.Timedelta(minutes=jitter_min)
        parent_guid = _guid(rng)
        parent_cmd = f'"{SERVICE_PARENT_IMAGE}" --scan'
        t = start
        for technique in ("T1082", "T1033"):
            rows.append(_process_alert(
                ts=t, host=host, user=SERVICE_ACCOUNT[0], domain=SERVICE_ACCOUNT[1],
                technique=technique, parent_image=SERVICE_PARENT_IMAGE,
                parent_command_line=parent_cmd, parent_process_guid=parent_guid, rng=rng,
            ))
            t = t + pd.Timedelta(seconds=int(rng.integers(30, 90)))
    return rows


def _admin_manual_activity(day: pd.Timestamp, rng: np.random.Generator) -> list[dict]:
    """Ordinary admins occasionally run the same commands the inventory
    agent runs - legitimate, just not on a fixed schedule. This is what
    makes the legitimate-vs-suspicious boundary a matter of degree rather
    than "service account good, everyone else bad": these alerts are real
    noise the semantic layer has to not over-penalise."""
    rows: list[dict] = []
    if day.dayofweek >= 5:
        return rows
    for admin in ADMIN_ACCOUNTS:
        if rng.random() > 0.22:
            continue
        host = str(rng.choice(admin["hosts"]))
        h0, h1 = admin["hours"]
        hour = int(rng.integers(h0, h1))
        minute = int(rng.integers(0, 60))
        start = day + pd.Timedelta(hours=hour, minutes=minute)
        techniques = ["T1033", "T1082"]
        if rng.random() < 0.25:
            techniques.append("T1087")
        parent_guid = _guid(rng)
        t = start
        for technique in techniques:
            rows.append(_process_alert(
                ts=t, host=host, user=admin["user"], domain=admin["domain"],
                technique=technique, parent_image="C:\\Windows\\System32\\cmd.exe",
                parent_command_line="cmd.exe /k", parent_process_guid=parent_guid, rng=rng,
            ))
            t = t + pd.Timedelta(seconds=int(rng.integers(15, 60)))
        if rng.random() < 0.06:
            # occasional legitimate scripting / internal-tool-download noise
            extra_tech = "T1059" if rng.random() < 0.6 else "T1105"
            rows.append(_process_alert(
                ts=t + pd.Timedelta(minutes=int(rng.integers(1, 20))), host=host,
                user=admin["user"], domain=admin["domain"], technique=extra_tech,
                parent_image="C:\\Windows\\System32\\cmd.exe", parent_command_line="cmd.exe /k",
                parent_process_guid=parent_guid, rng=rng,
            ))
    return rows


def _daily_auth_noise(day: pd.Timestamp, rng: np.random.Generator, mean_events: float = 140.0) -> list[dict]:
    """Background authentication volume, unrelated to the Discovery-tactic
    problem this prototype targets. Present for index realism and to
    exercise `event_category` scoping; not fed to the semantic or baseline
    scorer."""
    rows: list[dict] = []
    n = int(rng.poisson(mean_events))
    all_hosts = list(CORE_HOSTS) + [h for a in ADMIN_ACCOUNTS for h in a["hosts"]]
    all_users = list(EMPLOYEE_ACCOUNTS) + [a["user"] for a in ADMIN_ACCOUNTS]
    is_weekday = day.dayofweek < 5
    for _ in range(n):
        host = str(rng.choice(all_hosts))
        user = str(rng.choice(all_users))
        if is_weekday and rng.random() < 0.9:
            hour = int(np.clip(rng.normal(13, 3), 6, 21))
        else:
            hour = int(rng.integers(0, 24))
        minute = int(rng.integers(0, 60))
        ts = day + pd.Timedelta(hours=hour, minutes=minute)
        rows.append(_auth_alert(ts=ts, host=host, user=user, domain="CORP", rng=rng))
    return rows


# --------------------------------------------------------------------------
# Anomaly injection
# --------------------------------------------------------------------------

def _pick_business_ts(day: pd.Timestamp, rng: np.random.Generator) -> pd.Timestamp:
    hour = int(rng.integers(9, 17))
    minute = int(rng.integers(0, 60))
    return day + pd.Timedelta(hours=hour, minutes=minute)


def _pick_off_hours_ts(day: pd.Timestamp, rng: np.random.Generator) -> pd.Timestamp:
    hour = int(rng.choice([0, 1, 2, 3, 4, 23]))
    minute = int(rng.integers(0, 60))
    return day + pd.Timedelta(hours=hour, minutes=minute)


def _inject_anomalies(days: pd.DatetimeIndex, rng: np.random.Generator) -> list[dict]:
    """Five synthetic anomaly types, each isolating roughly one semantic
    feature so the validation can say which signal caught which pattern
    instead of one undifferentiated "anomaly" bucket. See the README table.
    """
    rows: list[dict] = []
    usable_days = days[3:-1]  # keep the first few and last day clean of injected noise

    # 1. new_host: a *known* admin, on a host they have never touched.
    for day in rng.choice(usable_days, size=30, replace=True):
        day = pd.Timestamp(day)
        admin = ADMIN_ACCOUNTS[int(rng.integers(0, len(ADMIN_ACCOUNTS)))]
        host = str(rng.choice(RARE_HOSTS))
        ts = _pick_business_ts(day, rng)
        rows.append(_process_alert(
            ts=ts, host=host, user=admin["user"], domain=admin["domain"], technique="T1033",
            parent_image="C:\\Windows\\System32\\cmd.exe", parent_command_line="cmd.exe /k",
            parent_process_guid=_guid(rng), rng=rng, is_anomaly=True, anomaly_type="new_host",
        ))

    # 2. off_hours: a known actor+host pair, at a time that actor never uses.
    for day in rng.choice(usable_days, size=30, replace=True):
        day = pd.Timestamp(day)
        admin = ADMIN_ACCOUNTS[int(rng.integers(0, len(ADMIN_ACCOUNTS)))]
        host = str(rng.choice(admin["hosts"]))
        ts = _pick_off_hours_ts(day, rng)
        rows.append(_process_alert(
            ts=ts, host=host, user=admin["user"], domain=admin["domain"], technique="T1082",
            parent_image="C:\\Windows\\System32\\cmd.exe", parent_command_line="cmd.exe /k",
            parent_process_guid=_guid(rng), rng=rng, is_anomaly=True, anomaly_type="off_hours",
        ))

    # 3. new_actor: an account that appears nowhere else in the corpus.
    for day in rng.choice(usable_days, size=30, replace=True):
        day = pd.Timestamp(day)
        user, domain = RARE_ACTORS[int(rng.integers(0, len(RARE_ACTORS)))]
        host = str(rng.choice(list(CORE_HOSTS) + list(RARE_HOSTS)))
        ts = _pick_business_ts(day, rng)
        technique = str(rng.choice(["T1033", "T1082", "T1087"]))
        rows.append(_process_alert(
            ts=ts, host=host, user=user, domain=domain, technique=technique,
            parent_image="C:\\Windows\\System32\\cmd.exe", parent_command_line="cmd.exe /k",
            parent_process_guid=_guid(rng), rng=rng, is_anomaly=True, anomaly_type="new_actor",
        ))

    # 4. manual_recon_chain: whoami -> net user -> net group, tight window,
    #    one shell, on one host - the pattern named explicitly in the brief.
    for day in rng.choice(usable_days, size=25, replace=True):
        day = pd.Timestamp(day)
        if rng.random() < 0.6:
            user, domain = RARE_ACTORS[int(rng.integers(0, len(RARE_ACTORS)))]
        else:
            admin = ADMIN_ACCOUNTS[int(rng.integers(0, len(ADMIN_ACCOUNTS)))]
            user, domain = admin["user"], admin["domain"]
        host = str(rng.choice(list(RARE_HOSTS) + list(CORE_HOSTS)))
        parent_guid = _guid(rng)
        t = _pick_business_ts(day, rng) if rng.random() < 0.7 else _pick_off_hours_ts(day, rng)
        for technique in ("T1033", "T1087", "T1069"):
            rows.append(_process_alert(
                ts=t, host=host, user=user, domain=domain, technique=technique,
                parent_image="C:\\Windows\\System32\\cmd.exe",
                parent_command_line="cmd.exe /c manual recon session",
                parent_process_guid=parent_guid, rng=rng,
                is_anomaly=True, anomaly_type="manual_recon_chain",
            ))
            t = t + pd.Timedelta(seconds=int(rng.integers(20, 90)))

    # 5. tactic_escalation: Discovery -> Credential Access -> Lateral
    #    Movement inside ~10 minutes on the same actor/host.
    for day in rng.choice(usable_days, size=15, replace=True):
        day = pd.Timestamp(day)
        user, domain = RARE_ACTORS[int(rng.integers(0, len(RARE_ACTORS)))]
        host = str(rng.choice(list(RARE_HOSTS) + list(CORE_HOSTS)))
        parent_guid = _guid(rng)
        t = _pick_off_hours_ts(day, rng) if rng.random() < 0.5 else _pick_business_ts(day, rng)
        for technique in ("T1033", "T1003", "T1021"):
            rows.append(_process_alert(
                ts=t, host=host, user=user, domain=domain, technique=technique,
                parent_image="C:\\Windows\\System32\\cmd.exe",
                parent_command_line="cmd.exe /c escalation session",
                parent_process_guid=parent_guid, rng=rng,
                is_anomaly=True, anomaly_type="tactic_escalation",
            ))
            t = t + pd.Timedelta(minutes=float(rng.uniform(2, 4)))

    return rows


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

COLUMNS: tuple[str, ...] = (
    "alert_uid", "timestamp", "cluster_node", "agent_name", "agent_ip",
    "rule_id", "rule_level", "rule_description", "rule_mitre_tactic", "rule_mitre_technique",
    "event_category", "subject_user_name", "subject_domain_name", "eventdata_user",
    "target_user_name", "target_domain_name", "command_line", "image",
    "parent_image", "parent_command_line", "parent_process_guid", "process_guid",
    "logon_type", "authentication_package_name",
    "is_synthetic_anomaly", "synthetic_anomaly_type",
)


def generate_alerts(
    *, n_days: int = 90, end: str | pd.Timestamp | None = None, seed: int = config.SEED,
) -> pd.DataFrame:
    """Build a synthetic 90-day alert stream with the schema described above.

    `is_synthetic_anomaly` / `synthetic_anomaly_type` are fabricated labels
    for validating the pipeline, not a claim about real attacker behaviour -
    see the module docstring.
    """
    rng = np.random.default_rng(seed)
    end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp(config.SYNTHETIC_END_DATE)
    days = pd.date_range(end=end_ts, periods=n_days, freq="D")

    rows: list[dict] = []
    for day in days:
        rows.extend(_daily_inventory_sweep(day, rng))
        rows.extend(_admin_manual_activity(day, rng))
        rows.extend(_daily_auth_noise(day, rng))
    rows.extend(_inject_anomalies(days, rng))

    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    frame = frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    assert not (set(frame["cluster_node"].unique()) & config.EXCLUDED_CLUSTER_NODES), (
        "synthetic generator produced an excluded cluster node - this would be a bug in "
        "_host_node, not a real data issue."
    )
    return frame


def main() -> int:
    frame = generate_alerts()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.DATA_DIR / "synthetic_alerts_sample.csv"
    frame.head(1000).to_csv(out_path, index=False)
    n_anom = int(frame["is_synthetic_anomaly"].sum())
    print(f"{len(frame):,} synthetic alerts generated "
          f"({frame['event_category'].value_counts().to_dict()})")
    print(f"{n_anom:,} labeled synthetic anomalies "
          f"({frame.loc[frame['is_synthetic_anomaly'], 'synthetic_anomaly_type'].value_counts().to_dict()})")
    print(f"first 1,000 rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
