"""Read-only OpenSearch adapter for `wazuh-alerts-4.x-*`.

`config.DATA_SOURCE` defaults to `"synthetic"` and this module is still not
called by anything in this repository unless that is explicitly overridden -
switching sources is a one-line config change (`SOC_DATA_SOURCE=opensearch`
plus three environment variables) rather than a rewrite: `fetch_alerts`
returns a DataFrame with exactly the columns `synthetic.generate_alerts`
returns, so `semantic.py`, `baseline.py`, `modeling.py` and `pipeline.py` do
not know or care which one produced their input.

Real OpenSearch access has been confirmed working (index `wazuh-alerts-4.x-*`,
~1.2M documents over the last 30 days, history back to 2026-05-04, four
confirmed source shapes - see `classify_source` below and the README's "Real
data" section), and `real_data_check.py` has since run a first real fetch
successfully (189,034 documents over a 5-day window). This module was
written and unit-tested against fabricated documents that mimic the
confirmed structure before that happened, since credentials are supplied by
the operator via environment variables, not passed to whatever wrote this
code - the first real run is what found the `clear_scroll` permission gap
documented on `_scroll` below.

Real credentials, real permissions
    The confirmed OpenSearch user is read-only (`kibanauser`, `readall`
    backend roles) and does not have `indices:data/read/scroll/clear` -
    that is a separate, narrower permission than read access to the data
    itself, and it was never promised. `_scroll` treats a 403 on
    `clear_scroll` specifically as expected and non-fatal: the documents
    are already fully fetched by that point, and the server expires an
    unused scroll context on its own after its timeout regardless. Any
    other exception from `clear_scroll` - a real network problem, for
    instance - still propagates; only this one documented, expected
    permission gap is caught.

Import-time safety
    Importing this module never opens a socket. The `opensearchpy` import is
    wrapped so the module can be imported (and its tests can run) even where
    `opensearch-py` is not installed. A client is constructed only inside
    `_build_client`, called only from `fetch_alerts`, `count_alerts` and
    `discover_cluster_nodes`, and only after `OpenSearchSettings.from_env()`
    has confirmed the required environment variables are present - there is
    no default host to accidentally connect to.

Customer-data exclusion
    `res-engineering-collector` is excluded twice: once in the query itself
    (`_excluded_node_filter`, so the cluster never even returns those
    documents), and again in `filters.filter_customer_data` on the way out,
    so a change to the query cannot silently ship that node's data. The
    second check also now refuses to return a silently-empty frame if the
    node allow-list (which defaults to the *synthetic* topology) does not
    match what the real cluster actually sent - see `config.ALLOWED_CLUSTER_NODES`.

Read-only
    The only API surface used is `search`, `scroll`/`clear_scroll` (via
    `_scroll`) and `count`. Nothing in this module writes, updates, or deletes.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from soc_runtime import config, filters
from soc_runtime.synthetic import COLUMNS

try:
    from opensearchpy import OpenSearch
    from opensearchpy.exceptions import AuthorizationException
except ImportError:  # opensearch-py is only required when DATA_SOURCE="opensearch"
    OpenSearch = None  # type: ignore[assignment,misc]

    class AuthorizationException(Exception):  # type: ignore[no-redef]
        """Placeholder used only when opensearch-py is not installed.

        `_scroll`'s `except AuthorizationException` needs a real exception
        class to bind to in every environment - including this project's own
        test environment, which does not install opensearch-py (see
        requirements.txt) - so that the read-only `clear_scroll` 403 can be
        tested with a fabricated exception without the real dependency. A
        real `client` object can only exist when opensearch-py is actually
        installed (see `_build_client`), so this placeholder is never the
        thing a real server response raises; it exists so the `except`
        clause itself is always valid.
        """


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

_REQUIRED_ENV = ("SOC_OPENSEARCH_HOST", "SOC_OPENSEARCH_USER", "SOC_OPENSEARCH_PASSWORD")


@dataclass(frozen=True)
class OpenSearchSettings:
    host: str
    port: int
    username: str
    password: str
    index_pattern: str = field(default_factory=lambda: config.OPENSEARCH_INDEX_PATTERN)
    verify_certs: bool = False  # self-signed TLS on this cluster, per the brief

    @classmethod
    def from_env(cls) -> "OpenSearchSettings":
        missing = [name for name in _REQUIRED_ENV if not os.environ.get(name)]
        if missing:
            raise RuntimeError(
                "DATA_SOURCE='opensearch' but missing environment variable(s): "
                f"{', '.join(missing)}. Set them, or switch config.DATA_SOURCE (env var "
                "SOC_DATA_SOURCE) back to 'synthetic'. No default host is used - a missing "
                "credential must fail loudly, not connect somewhere unintended."
            )
        return cls(
            host=os.environ["SOC_OPENSEARCH_HOST"],
            port=int(os.environ.get("SOC_OPENSEARCH_PORT", "9200")),
            username=os.environ["SOC_OPENSEARCH_USER"],
            password=os.environ["SOC_OPENSEARCH_PASSWORD"],
        )


# --------------------------------------------------------------------------
# Query
# --------------------------------------------------------------------------

def _excluded_node_filter() -> list[dict]:
    return [{"term": {"cluster.node": node}} for node in config.EXCLUDED_CLUSTER_NODES]


def _build_query(start_iso: str, end_iso: str) -> dict[str, Any]:
    return {
        "query": {
            "bool": {
                "filter": [{"range": {"timestamp": {"gte": start_iso, "lt": end_iso}}}],
                "must_not": _excluded_node_filter(),
            }
        },
        "sort": [{"timestamp": "asc"}],
    }


# --------------------------------------------------------------------------
# Source classification
#
# The real cluster carries (at least) four kinds of documents, confirmed by
# structure: Sysmon/Windows Event Log (`data.win.*`), Suricata network alerts
# (`data.alert.*`, `data.src_ip`), Windows auth events (a subset of the same
# `data.win.eventdata.*` fields, no `commandLine`), and Kaspersky Endpoint
# Security (`data.KES.*`) - a fourth source the original synthetic design did
# not anticipate at all.
#
# Decision on KES, stated here rather than left implicit: this iteration
# classifies KES documents (and Suricata ones) explicitly and routes them to
# `event_category` values ("kes_endpoint", "network") that
# `semantic.build_features` and `modeling.restrict_to_scored_techniques`
# already ignore - they are *seen and counted*, never silently dropped, but
# not scored. The alternative (mapping KES fields onto the existing
# actor/host/technique schema) was rejected for this iteration: KES's own
# field names, event taxonomy and what counts as an "actor" or "technique" in
# its schema have not been reviewed, and guessing a mapping under real-data
# time pressure is how a semantic layer quietly starts scoring nonsense. See
# the README's "Real data" section for the full reasoning.
# --------------------------------------------------------------------------

SOURCE_TO_CATEGORY: dict[str, str] = {
    "sysmon_process": "process_creation",
    "sysmon_auth": "authentication",
    "suricata": "network",
    "kes": "kes_endpoint",
    "unknown": "unknown",
}


def classify_source(src: dict) -> str:
    """One of "sysmon_process", "sysmon_auth", "suricata", "kes", "unknown".

    Order matters: KES and Suricata are checked first because a document
    missing `data.win` entirely must never fall through to the Sysmon
    branches - that was the actual bug this function replaces (see
    `_hit_to_row`'s previous version, which classified anything without a
    Sysmon `commandLine` as `"authentication"`, silently mislabelling every
    Suricata and KES document as a Windows logon event).
    """
    if _get(src, "data", "KES") is not None:
        return "kes"
    if _get(src, "data", "alert") is not None or _get(src, "data", "src_ip") is not None:
        return "suricata"
    if _get(src, "data", "win") is not None:
        command_line = _get(src, "data", "win", "eventdata", "commandLine")
        return "sysmon_process" if command_line else "sysmon_auth"
    return "unknown"


# --------------------------------------------------------------------------
# Technique name -> MITRE ID translation
#
# Real finding: this SOC's `rule.mitre.technique` carries human-readable
# names ("System Owner/User Discovery"), not MITRE IDs ("T1033") - see
# `config.TECHNIQUE_NAME_TO_ID` for the full mapping, its sourcing, and the
# two names it deliberately does not resolve. Every downstream comparison
# against a technique ID - `config.IN_SCOPE_TECHNIQUES`,
# `modeling.restrict_to_scored_techniques`, `real_data_check.field_completeness`
# - depends on this running first, at the point a real document becomes a row.
# --------------------------------------------------------------------------

_TECHNIQUE_ID_PATTERN = re.compile(r"^T\d{4}(\.\d{3})?$")


def translate_technique_name(value: str | None) -> str | None:
    """`None` stays `None`. A value that already looks like a MITRE ID
    (`T1033`, `T1059.001`) passes through unchanged - defensive, not
    assumed: nothing here guarantees every real Wazuh installation sends
    names rather than IDs, and the pattern check is cheap. A known name
    translates via `config.TECHNIQUE_NAME_TO_ID`. Anything else - a name not
    in that dictionary - passes through **unchanged**, not silently dropped
    or nulled: it will not match `IN_SCOPE_TECHNIQUES` (so it is never
    scored, same as any other out-of-scope technique), but it stays visible
    in `real_data_check.py`'s field-completeness report exactly as it
    arrived, which is where an operator would notice and extend the
    dictionary.
    """
    if value is None:
        return None
    if _TECHNIQUE_ID_PATTERN.match(value):
        return value
    return config.TECHNIQUE_NAME_TO_ID.get(value, value)


# --------------------------------------------------------------------------
# Wazuh document -> flat row
#
# See the column-mapping table at the top of `synthetic.py` for the full
# dotted-path -> column-name correspondence this mirrors.
#
# Real `rule.mitre.tactic` / `.technique` are arrays; a correlation rule can
# tag several techniques on one alert. This adapter takes the first element
# of each and drops the rest, matching the single-technique-per-row schema
# `synthetic.py` and the rest of this pipeline assume. A production version
# would instead explode a multi-technique alert into one row per technique
# (keeping the shared `alert_uid` so `sem_*` counts do not double the same
# alert's severity) - flagged here rather than done silently, since it
# changes what a single "alert" means for counting purposes.
# `real_data_check.py` counts how often this actually happens on the real
# index rather than leaving it a hypothetical.
#
# The element that survives is then run through `translate_technique_name` -
# this SOC's `rule.mitre.technique` holds human-readable names, not MITRE
# IDs; see that function and `config.TECHNIQUE_NAME_TO_ID`.
# --------------------------------------------------------------------------

def _get(doc: dict, *path: str) -> Any:
    node = doc
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def first_of_multivalued(value: Any) -> Any:
    """`rule.mitre.tactic`/`.technique` arrive as arrays on real documents -
    one correlation rule can tag several techniques on one alert. This
    prototype's schema is single-technique-per-row (see `synthetic.py`'s
    module docstring), so only the first element survives; the rest is
    dropped, not exploded into extra rows. A non-list value (or an empty
    list) passes through unchanged.

    Factored out so `_hit_to_row` and `synthetic_v2.py` apply exactly the
    same reduction rule rather than two copies that could drift - the real
    30-day measurement of how often this actually matters (~57.7% of
    process-creation alerts) is precisely the number `synthetic_v2.py`
    calibrates its own multi-valued rate against, and it would be a
    meaningless comparison if the two rows used different reduction logic.
    """
    return value[0] if isinstance(value, list) and value else value


def all_of_multivalued(value: Any) -> list:
    """The *whole* multi-valued field, normalised to a list - the companion
    to `first_of_multivalued` above, added once it became clear that
    dropping the other values is not a harmless reporting simplification.

    `semantic.SemanticState` keys its causal state on
    (actor, host, technique) and folds every alert into it. Keeping only
    the first technique means the other techniques' frequency counters
    never increment and their "have I seen this triple before" sets never
    learn them, so a later alert can be scored as novel when the same
    triple already fired as a secondary value - and
    `sem_chain_tactic_diversity_15m`, whose entire job is counting
    distinct tactics, structurally cannot see two tactics arrive together.
    On real data that is not a corner case: 57.7% of process-creation
    alerts carry more than one technique.

    Ordering is preserved and duplicates are dropped (a rule can tag the
    same technique twice); `None` and empty values collapse to `[]`, and a
    non-list scalar becomes a one-element list, so callers get one shape
    to handle rather than three.
    """
    if isinstance(value, list):
        items = value
    elif value is None:
        return []
    else:
        items = [value]
    return list(dict.fromkeys(item for item in items if item is not None))


def _hit_to_row(hit: dict) -> dict:
    src = hit.get("_source", {})
    mitre_tactic = _get(src, "rule", "mitre", "tactic")
    mitre_technique = _get(src, "rule", "mitre", "technique")
    raw_technique = first_of_multivalued(mitre_technique)
    command_line = _get(src, "data", "win", "eventdata", "commandLine")
    source_kind = classify_source(src)
    return {
        "alert_uid": hit.get("_id"),
        "timestamp": pd.Timestamp(src.get("timestamp")),
        "cluster_node": _get(src, "cluster", "node"),
        "agent_name": _get(src, "agent", "name"),
        "agent_ip": _get(src, "agent", "ip"),
        "rule_id": _get(src, "rule", "id"),
        "rule_level": _get(src, "rule", "level"),
        "rule_description": _get(src, "rule", "description"),
        "rule_mitre_tactic": first_of_multivalued(mitre_tactic),
        "rule_mitre_technique": translate_technique_name(raw_technique),
        # The full multi-valued fields, kept alongside the single-value
        # columns above rather than replacing them: 57.7% of real
        # process-creation alerts carry several techniques, and collapsing
        # them corrupts the semantic layer's causal state (see
        # `all_of_multivalued`). The single-value columns stay because the
        # whole downstream schema - and every published v1 number - is
        # built on one-technique-per-row; `semantic.py` reads the `_all`
        # columns when present and falls back to the scalar ones when not,
        # so v1's generator needs no change to keep working.
        "rule_mitre_tactic_all": all_of_multivalued(mitre_tactic),
        "rule_mitre_technique_all": [
            translate_technique_name(name) for name in all_of_multivalued(mitre_technique)
        ],
        "event_category": SOURCE_TO_CATEGORY[source_kind],
        "subject_user_name": _get(src, "data", "win", "eventdata", "subjectUserName"),
        "subject_domain_name": _get(src, "data", "win", "eventdata", "subjectDomainName"),
        # Fallback identity field - see semantic.py's actor-resolution note.
        # Sysmon's own `User` field (the execution context of the *new*
        # process), a different Windows Event Log field from subjectUserName
        # (the account the logging subsystem attributes the event to) with
        # different semantics - added after the first real run found
        # subject_user_name null on 99.48% of real process-creation alerts.
        "eventdata_user": _get(src, "data", "win", "eventdata", "user"),
        "target_user_name": _get(src, "data", "win", "eventdata", "targetUserName"),
        "target_domain_name": _get(src, "data", "win", "eventdata", "targetDomainName"),
        "command_line": command_line,
        "image": _get(src, "data", "win", "eventdata", "image"),
        "parent_image": _get(src, "data", "win", "eventdata", "parentImage"),
        "parent_command_line": _get(src, "data", "win", "eventdata", "parentCommandLine"),
        "parent_process_guid": _get(src, "data", "win", "eventdata", "parentProcessGuid"),
        "process_guid": _get(src, "data", "win", "eventdata", "processGuid"),
        "logon_type": _get(src, "data", "win", "eventdata", "logonType"),
        "authentication_package_name": _get(src, "data", "win", "eventdata", "authenticationPackageName"),
        "is_synthetic_anomaly": False,  # real data carries no such label - see README
        "synthetic_anomaly_type": None,
    }


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------

def _scroll(client: "OpenSearch", index_pattern: str, query: dict, page_size: int) -> list[dict]:
    hits: list[dict] = []
    body = {**query, "size": page_size}
    resp = client.search(index=index_pattern, body=body, scroll="2m")
    scroll_id = resp.get("_scroll_id")
    page = resp["hits"]["hits"]
    try:
        while page:
            hits.extend(page)
            resp = client.scroll(scroll_id=scroll_id, scroll="2m")
            scroll_id = resp.get("_scroll_id")
            page = resp["hits"]["hits"]
    finally:
        if scroll_id:
            # Confirmed on the real cluster: a read-only user (backend roles
            # kibanauser/readall) has `indices:data/read/scroll/clear` denied
            # - a separate, narrower permission from read access to the data
            # itself, and one this project was never promised. By this point
            # every hit has already been collected above; clear_scroll is
            # only server-side cleanup, and OpenSearch expires an unused
            # scroll context on its own once its "2m" timeout lapses. Only
            # this one documented 403 is swallowed - any other exception
            # here (a real network problem, a different permission gap)
            # still propagates, same as it always did.
            try:
                client.clear_scroll(scroll_id=scroll_id)
            except AuthorizationException:
                print(
                    "note: clear_scroll skipped - read-only credentials lack "
                    "indices:data/read/scroll/clear. The server will expire this "
                    f"scroll context automatically; all {len(hits):,} already-fetched "
                    "documents are unaffected."
                )
    return hits


def _require_opensearch() -> None:
    if OpenSearch is None:
        raise ImportError(
            "opensearch-py is not installed. Run `pip install opensearch-py` to use "
            "DATA_SOURCE='opensearch'; the synthetic source needs no such dependency."
        )


def _build_client(settings: OpenSearchSettings) -> "OpenSearch":
    """The one place an `OpenSearch` client object is constructed. Every
    function below - `fetch_alerts`, `count_alerts`, `discover_cluster_nodes`
    - calls this, so there is exactly one place that ever opens a socket."""
    _require_opensearch()
    return OpenSearch(
        hosts=[{"host": settings.host, "port": settings.port}],
        http_auth=(settings.username, settings.password),
        use_ssl=True,
        verify_certs=settings.verify_certs,
    )


def count_alerts(start_iso: str, end_iso: str, *, settings: OpenSearchSettings | None = None) -> int:
    """A cheap `_count` call, meant to run *before* `fetch_alerts` on a new
    time window - the volume-safety check `real_data_check.py` uses to
    refuse an accidentally huge pull rather than scroll through it and find
    out the hard way. Does not exclude `res-engineering-collector` or apply
    the node allow-list; it is a raw document count for the window, used only
    to size the pull, not to decide what gets scored."""
    settings = settings or OpenSearchSettings.from_env()
    client = _build_client(settings)
    query = _build_query(start_iso, end_iso)
    resp = client.count(index=settings.index_pattern, body={"query": query["query"]})
    return int(resp["count"])


def discover_cluster_nodes(
    start_iso: str, end_iso: str, *, settings: OpenSearchSettings | None = None, size: int = 50,
) -> dict[str, int]:
    """`{cluster.node: doc_count}` for the window, via a terms aggregation.

    Deliberately does **not** exclude `res-engineering-collector` or apply
    the allow-list - the point of this call is to audit what node names
    actually exist on the real cluster, including confirming
    `res-engineering-collector` is really there (so its exclusion downstream
    is excluding something real, not a name that was never present), and to
    give the operator the real node names to put in
    `SOC_ALLOWED_CLUSTER_NODES` before pulling real data. This returns
    aggregate counts only, never document contents.

    `cluster.node` may be mapped as `text` (analyzed) or `keyword` on the
    real index - not confirmed from here, since it depends on the real
    index's mapping. Tries the bare field first, falls back to
    `cluster.node.keyword`, and raises with both underlying errors if
    neither aggregates.
    """
    settings = settings or OpenSearchSettings.from_env()
    client = _build_client(settings)
    time_filter = {"range": {"timestamp": {"gte": start_iso, "lt": end_iso}}}

    errors = []
    for field in ("cluster.node", "cluster.node.keyword"):
        body = {
            "size": 0,
            "query": time_filter,
            "aggs": {"nodes": {"terms": {"field": field, "size": size}}},
        }
        try:
            resp = client.search(index=settings.index_pattern, body=body)
            buckets = resp["aggregations"]["nodes"]["buckets"]
            return {b["key"]: int(b["doc_count"]) for b in buckets}
        except Exception as exc:  # noqa: BLE001 - re-raised below with both attempts' context
            errors.append((field, exc))
    raise RuntimeError(
        "could not aggregate on cluster.node under either mapping guess: "
        + "; ".join(f"{field}: {exc}" for field, exc in errors)
    )


def fetch_raw_hits(
    start_iso: str, end_iso: str, *, settings: OpenSearchSettings | None = None, page_size: int = 2_000,
) -> list[dict]:
    """The raw `_source` documents for `[start_iso, end_iso)`, before
    flattening. Excludes `config.EXCLUDED_CLUSTER_NODES` in the query itself
    (the allow-list is applied later, in `filters.filter_customer_data`, not
    here). Exists so `real_data_check.py` can run its diagnostics - source
    classification, multi-valued `rule.mitre.*` counts, field completeness -
    against exactly the documents `fetch_alerts` would flatten, from one
    query, instead of fetching the same window twice."""
    _require_opensearch()
    settings = settings or OpenSearchSettings.from_env()
    client = _build_client(settings)
    query = _build_query(start_iso, end_iso)
    return _scroll(client, settings.index_pattern, query, page_size)


def hits_to_frame(hits: list[dict]) -> pd.DataFrame:
    """Flatten + sort, without applying the customer-data filter - callers
    that need to inspect what was fetched before filtering (`real_data_check.py`'s
    node audit) use this directly; `fetch_alerts` filters on top of it.

    Normalises `timestamp` to a single, uniform naive `datetime64[ns]` dtype
    (interpreted as UTC) before returning. Without this, a batch of real
    documents whose `timestamp` field carries inconsistent UTC-offset
    information across sources (confirmed on the real cluster: some alert
    types include an explicit offset, some don't) leaves the column as
    `object` dtype holding a mix of timezone-aware and timezone-naive
    `pandas.Timestamp` instances - which is exactly what crashed
    `semantic.py`'s epoch-seconds conversion (`.astype("int64")` on an
    `object` column falls back to Python's `int()` per element) and would
    have gone on to crash `modeling.chronological_split` and
    `baseline.build_features` too, both of which need the `.dt` accessor and
    that only works on a real datetime64 column. Fixing it once here, at the
    one place a frame is assembled from raw hits, is more robust than
    patching each downstream `.dt`/`.astype` call site individually.
    """
    rows = [_hit_to_row(hit) for hit in hits]
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    if not frame.empty:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_localize(None)
        frame = frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return frame


def fetch_alerts(
    start_iso: str, end_iso: str, *, settings: OpenSearchSettings | None = None, page_size: int = 2_000,
) -> pd.DataFrame:
    """Read-only. Queries `[start_iso, end_iso)`, excludes
    `config.EXCLUDED_CLUSTER_NODES` in the query itself, and returns a
    DataFrame with the same columns `synthetic.generate_alerts` produces.

    Raises before opening any connection if `opensearch-py` is not installed
    or the required environment variables are not set.
    """
    hits = fetch_raw_hits(start_iso, end_iso, settings=settings, page_size=page_size)
    frame = hits_to_frame(hits)
    return filters.filter_customer_data(frame)
