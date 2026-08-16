"""This project's own test environment has no live OpenSearch cluster and
does not install opensearch-py (see requirements.txt) - a real cluster has
been reached once, manually, by the operator running real_data_check.py, not
by anything in this test suite. What these tests can and must guarantee
without one: importing the module performs no I/O, calling it without
credentials fails fast with a clear message rather than hanging on a
connection attempt, and the one confirmed real-world surprise so far - a
read-only user's `clear_scroll` 403 - is handled correctly."""

from __future__ import annotations

import pytest

from soc_runtime import opensearch_client as osc
from soc_runtime.synthetic import COLUMNS


def test_import_performs_no_network_io():
    """If this module had tried to connect at import time, it would already
    have failed or hung before this test body ever runs."""
    assert hasattr(osc, "fetch_alerts")
    assert hasattr(osc, "OpenSearchSettings")


def test_missing_env_vars_raise_before_any_connection(monkeypatch):
    for name in ("SOC_OPENSEARCH_HOST", "SOC_OPENSEARCH_USER", "SOC_OPENSEARCH_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="missing environment variable"):
        osc.OpenSearchSettings.from_env()


def test_fetch_alerts_raises_before_connecting_when_env_missing(monkeypatch):
    for name in ("SOC_OPENSEARCH_HOST", "SOC_OPENSEARCH_USER", "SOC_OPENSEARCH_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises((RuntimeError, ImportError)):
        osc.fetch_alerts("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z")


def test_settings_from_env_reads_all_fields(monkeypatch):
    monkeypatch.setenv("SOC_OPENSEARCH_HOST", "soc.example.internal")
    monkeypatch.setenv("SOC_OPENSEARCH_USER", "reader")
    monkeypatch.setenv("SOC_OPENSEARCH_PASSWORD", "secret")
    monkeypatch.setenv("SOC_OPENSEARCH_PORT", "9201")
    settings = osc.OpenSearchSettings.from_env()
    assert settings.host == "soc.example.internal"
    assert settings.port == 9201
    assert settings.username == "reader"
    assert settings.password == "secret"
    assert settings.verify_certs is False


def test_excluded_node_filter_names_the_customer_node():
    from soc_runtime import config

    query = osc._build_query("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z")
    must_not = query["query"]["bool"]["must_not"]
    excluded = {clause["term"]["cluster.node"] for clause in must_not}
    assert excluded == set(config.EXCLUDED_CLUSTER_NODES)


def test_hit_to_row_flattens_nested_wazuh_document():
    """A minimal fabricated Wazuh-shaped document, to check the dotted-path
    flattening independent of any live cluster."""
    hit = {
        "_id": "abc123",
        "_source": {
            "timestamp": "2026-06-01T08:00:00.000Z",
            "cluster": {"node": "office-collector"},
            "agent": {"name": "CORP-WKS-0001", "ip": "10.20.1.2"},
            "rule": {
                "id": 92033, "level": 3, "description": "System Owner/User Discovery",
                "mitre": {"tactic": ["Discovery"], "technique": ["T1033"]},
            },
            "data": {"win": {"eventdata": {
                "subjectUserName": "svc-swinventory", "subjectDomainName": "CORP",
                "commandLine": "whoami.exe /all", "image": "C:\\Windows\\System32\\whoami.exe",
                "parentImage": "sw-inventory-agent.exe",
                "parentProcessGuid": "{p}", "processGuid": "{c}",
            }}},
        },
    }
    row = osc._hit_to_row(hit)
    assert row["agent_name"] == "CORP-WKS-0001"
    assert row["rule_mitre_technique"] == "T1033"
    assert row["rule_mitre_tactic"] == "Discovery"
    assert row["subject_user_name"] == "svc-swinventory"
    assert row["event_category"] == "process_creation"
    assert row["image"] == "C:\\Windows\\System32\\whoami.exe"
    # Real rows carry the shared synthetic schema plus the two optional
    # multi-valued columns - `semantic._value_lists` reads those when
    # present and falls back to the scalar columns when they are not, which
    # is what keeps v1's generator (single-technique by construction, no
    # `_all` columns) working unchanged.
    assert set(row.keys()) == set(COLUMNS) | {
        "rule_mitre_technique_all", "rule_mitre_tactic_all",
    }


def test_hit_to_row_authentication_event_has_no_command_line():
    hit = {
        "_id": "def456",
        "_source": {
            "timestamp": "2026-06-01T08:00:00.000Z",
            "cluster": {"node": "node01"},
            "agent": {"name": "CORP-WKS-0002", "ip": "10.20.1.3"},
            "rule": {"id": 60106, "level": 3, "description": "Windows Logon Success."},
            "data": {"win": {"eventdata": {
                "subjectUserName": "user.emp001", "logonType": 2, "authenticationPackageName": "Negotiate",
            }}},
        },
    }
    row = osc._hit_to_row(hit)
    assert row["event_category"] == "authentication"
    assert row["command_line"] is None
    assert row["logon_type"] == 2


# --------------------------------------------------------------------------
# Source classification - the real cluster's four confirmed source shapes,
# fabricated to mimic their described structure since there is no live
# access to pull real examples from.
# --------------------------------------------------------------------------

def test_classify_source_sysmon_process():
    src = {"data": {"win": {"eventdata": {"commandLine": "whoami.exe /all", "image": "whoami.exe"}}}}
    assert osc.classify_source(src) == "sysmon_process"


def test_classify_source_sysmon_auth():
    src = {"data": {"win": {"eventdata": {"subjectUserName": "user1", "logonType": 2}}}}
    assert osc.classify_source(src) == "sysmon_auth"


def test_classify_source_suricata_via_alert_field():
    src = {"data": {"alert": {"signature": "ET SCAN Nmap"}, "src_ip": "10.0.0.5"}}
    assert osc.classify_source(src) == "suricata"


def test_classify_source_suricata_via_src_ip_only():
    src = {"data": {"src_ip": "10.0.0.5", "dest_ip": "10.0.0.9"}}
    assert osc.classify_source(src) == "suricata"


def test_classify_source_kes():
    src = {"data": {"KES": {"event": "Detected"}, "srcip": "10.0.0.5", "host": "CORP-WKS-0001"}}
    assert osc.classify_source(src) == "kes"


def test_classify_source_kes_checked_before_win_or_suricata():
    """If a document somehow carried both `data.KES` and `data.win` (or
    `data.alert`), KES must win - the classification order in
    `classify_source` is deliberate, not incidental."""
    src = {"data": {"KES": {"event": "Detected"}, "win": {"eventdata": {"commandLine": "x"}}}}
    assert osc.classify_source(src) == "kes"


def test_classify_source_unknown_shape():
    src = {"data": {"something_unrecognised": {"a": 1}}}
    assert osc.classify_source(src) == "unknown"


def test_first_of_multivalued_takes_first_element():
    assert osc.first_of_multivalued(["PowerShell", "Command and Scripting Interpreter"]) == "PowerShell"


def test_first_of_multivalued_passes_through_non_list():
    assert osc.first_of_multivalued("T1033") == "T1033"
    assert osc.first_of_multivalued(None) is None


def test_first_of_multivalued_passes_through_empty_list():
    assert osc.first_of_multivalued([]) == []


def test_hit_to_row_kes_document_is_tagged_kes_endpoint_not_authentication():
    """Regression test for the real bug this classification replaces: a
    document with no `data.win` used to fall through to `"authentication"`
    just because it had no Sysmon `commandLine`, silently mislabelling every
    KES and Suricata document as a Windows logon event."""
    hit = {
        "_id": "kes1",
        "_source": {
            "timestamp": "2026-08-05T10:00:00.000Z",
            "cluster": {"node": "office-collector"},
            "agent": {"name": "CORP-WKS-0009", "ip": "10.20.1.9"},
            "rule": {"id": 100500, "level": 7, "description": "Malware detected"},
            "data": {"KES": {"event": "Detected"}, "srcip": "10.20.1.9", "host": "CORP-WKS-0009"},
        },
    }
    row = osc._hit_to_row(hit)
    assert row["event_category"] == "kes_endpoint"
    assert row["event_category"] != "authentication"


def test_hit_to_row_suricata_document_is_tagged_network_not_authentication():
    hit = {
        "_id": "suri1",
        "_source": {
            "timestamp": "2026-08-05T10:00:00.000Z",
            "cluster": {"node": "office-collector"},
            "agent": {"name": "CORP-WKS-0010", "ip": "10.20.1.10"},
            "rule": {"id": 100600, "level": 6, "description": "Network scan detected"},
            "data": {"alert": {"signature": "ET SCAN Nmap"}, "src_ip": "10.20.1.10", "dest_ip": "10.20.1.1"},
        },
    }
    row = osc._hit_to_row(hit)
    assert row["event_category"] == "network"
    assert row["event_category"] != "authentication"


def test_kes_and_suricata_rows_are_excluded_from_semantic_scoring():
    """Both new categories must fall straight through
    `semantic.build_features`'s existing `event_category == "process_creation"`
    filter - unscored, but present in the frame, not dropped by this step."""
    from soc_runtime.semantic import build_features

    hits = [
        {
            "_id": "p1",
            "_source": {
                "timestamp": "2026-08-05T10:00:00.000Z", "cluster": {"node": "office-collector"},
                "agent": {"name": "H1", "ip": "10.0.0.1"},
                "rule": {"id": 92033, "level": 3, "mitre": {"tactic": ["Discovery"], "technique": ["T1033"]}},
                "data": {"win": {"eventdata": {
                    "subjectUserName": "u1", "subjectDomainName": "CORP",
                    "commandLine": "whoami.exe", "image": "whoami.exe",
                }}},
            },
        },
        {
            "_id": "k1",
            "_source": {
                "timestamp": "2026-08-05T10:05:00.000Z", "cluster": {"node": "office-collector"},
                "agent": {"name": "H1", "ip": "10.0.0.1"}, "rule": {"id": 5, "level": 5},
                "data": {"KES": {"event": "Detected"}},
            },
        },
    ]
    frame = osc.hits_to_frame(hits)
    featured = build_features(frame)
    assert len(featured) == 1  # only the sysmon_process row was scored
    assert (frame["event_category"] == "kes_endpoint").sum() == 1


def test_hits_to_frame_normalises_mixed_timezone_timestamps():
    """Regression test for the real failure: the first real run's `timestamp`
    field carried inconsistent UTC-offset information across alert sources
    within the same 5-day batch, which left the flattened `timestamp` column
    as `object` dtype and crashed `semantic.py`'s epoch-seconds conversion
    downstream. `hits_to_frame` must produce one uniform `datetime64` column
    regardless - here, one document with an explicit UTC offset (`Z`) and one
    without, exactly the two shapes that collided on the real cluster."""
    hits = [
        {
            "_id": "aware", "_source": {
                "timestamp": "2026-08-05T10:00:00.000Z", "cluster": {"node": "office-collector"},
                "agent": {"name": "H1"}, "rule": {"id": 1, "level": 3},
                "data": {"win": {"eventdata": {"commandLine": "whoami.exe", "image": "whoami.exe"}}},
            },
        },
        {
            "_id": "naive", "_source": {
                "timestamp": "2026-08-05T10:05:00", "cluster": {"node": "office-collector"},
                "agent": {"name": "H1"}, "rule": {"id": 1, "level": 3},
                "data": {"win": {"eventdata": {"commandLine": "systeminfo.exe", "image": "systeminfo.exe"}}},
            },
        },
    ]
    frame = osc.hits_to_frame(hits)
    assert frame["timestamp"].dtype.kind == "M"  # a real datetime64 dtype, not object
    assert frame["timestamp"].is_monotonic_increasing


def test_hits_to_frame_output_survives_build_features():
    """End-to-end version of the same regression, through the actual
    real-data entry point rather than a hand-built frame: `hits_to_frame`'s
    output must not crash `semantic.build_features` even when the source
    hits mixed timezone-aware and timezone-naive timestamps."""
    from soc_runtime.semantic import build_features

    hits = [
        {
            "_id": "aware", "_source": {
                "timestamp": "2026-08-05T10:00:00.000Z", "cluster": {"node": "office-collector"},
                "agent": {"name": "H1"}, "rule": {"id": 92033, "level": 3,
                "mitre": {"tactic": ["Discovery"], "technique": ["T1033"]}},
                "data": {"win": {"eventdata": {
                    "subjectUserName": "u1", "subjectDomainName": "CORP",
                    "commandLine": "whoami.exe", "image": "whoami.exe",
                }}},
            },
        },
        {
            "_id": "naive", "_source": {
                "timestamp": "2026-08-05T10:05:00", "cluster": {"node": "office-collector"},
                "agent": {"name": "H1"}, "rule": {"id": 92082, "level": 3,
                "mitre": {"tactic": ["Discovery"], "technique": ["T1082"]}},
                "data": {"win": {"eventdata": {
                    "subjectUserName": "u1", "subjectDomainName": "CORP",
                    "commandLine": "systeminfo.exe", "image": "systeminfo.exe",
                }}},
            },
        },
    ]
    frame = osc.hits_to_frame(hits)
    featured = build_features(frame)  # must not raise TypeError
    assert len(featured) == 2


# --------------------------------------------------------------------------
# Technique name -> MITRE ID translation
#
# Real finding: this SOC's rule.mitre.technique holds human-readable names
# ("System Owner/User Discovery"), not MITRE IDs ("T1033") - a 30-day real
# window reported share_technique_in_the_six_scored: 0.0 because of it, and
# modeling.restrict_to_scored_techniques compares against the same column,
# so this was a real scoring-pipeline bug, not just a reporting artifact.
# The 20 names below are exactly what that window observed; each mapping is
# checked against the official MITRE ATT&CK Enterprise matrix
# (https://attack.mitre.org/techniques/enterprise/), not guessed - see
# config.py's note above TECHNIQUE_NAME_TO_ID for sourcing and for the two
# names ("Local Account", "Tool") that could not be resolved with
# confidence.
# --------------------------------------------------------------------------

_CONFIRMED_NAME_TO_ID = {
    "Account Discovery": "T1087",
    "Application Shimming": "T1546.011",
    "Command and Scripting Interpreter": "T1059",
    "Compile After Delivery": "T1027.004",
    "Exfiltration Over Web Service": "T1567",
    "File Deletion": "T1070.004",
    "Hidden Window": "T1564.003",
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
}


@pytest.mark.parametrize("name,expected_id", sorted(_CONFIRMED_NAME_TO_ID.items()))
def test_translate_technique_name_matches_official_mitre_mapping(name, expected_id):
    assert osc.translate_technique_name(name) == expected_id


def test_translate_technique_name_local_account_is_the_documented_judgment_call():
    """"Local Account" is genuinely ambiguous - the same sub-technique name
    is reused under three different parent techniques (T1087.001, T1136.001,
    T1078.003) and the bare string cannot disambiguate which fired. This
    pins the documented choice (T1087.001, on Discovery-tactic context) so
    a change to it is a deliberate edit, not an accidental one."""
    assert osc.translate_technique_name("Local Account") == "T1087.001"


def test_translate_technique_name_leaves_tool_unmapped():
    """"Tool" is not a MITRE Enterprise technique name at all - MITRE
    separates Techniques from Software/Tools as distinct categories. Left
    out of the dictionary on purpose, so it must pass through unchanged
    (visible in reports) rather than disappear or get guessed at."""
    assert osc.translate_technique_name("Tool") == "Tool"


def test_translate_technique_name_passes_through_values_that_already_look_like_ids():
    assert osc.translate_technique_name("T1033") == "T1033"
    assert osc.translate_technique_name("T1059.001") == "T1059.001"


def test_translate_technique_name_passes_through_unknown_names_unchanged():
    """A name observed on some future real run that is in neither the
    dictionary nor ID-shaped must still not be dropped or raise - it stays
    exactly as it arrived, so an operator extending the dictionary later can
    find it in a field-completeness report."""
    assert osc.translate_technique_name("Some Brand New Technique Name") == "Some Brand New Technique Name"


def test_translate_technique_name_passes_through_none():
    assert osc.translate_technique_name(None) is None


def test_hit_to_row_translates_technique_name_to_id():
    hit = {
        "_id": "x", "_source": {
            "timestamp": "2026-08-05T10:00:00.000Z", "cluster": {"node": "office-collector"},
            "agent": {"name": "H1"},
            "rule": {"id": 1, "level": 3, "mitre": {"tactic": ["Discovery"],
                     "technique": ["System Owner/User Discovery"]}},
            "data": {"win": {"eventdata": {"commandLine": "whoami.exe", "image": "whoami.exe"}}},
        },
    }
    row = osc._hit_to_row(hit)
    assert row["rule_mitre_technique"] == "T1033"


def test_restrict_to_scored_techniques_recognises_translated_real_alerts():
    """The actual scoring-pipeline bug the finding uncovered, reproduced
    end to end: before this fix, a frame built from real-shaped documents
    using technique names would never match `config.IN_SCOPE_TECHNIQUES`,
    and every real alert would silently disappear from what gets scored."""
    from soc_runtime import config, modeling

    hits = [
        {
            "_id": "in-scope", "_source": {
                "timestamp": "2026-08-05T10:00:00.000Z", "cluster": {"node": "office-collector"},
                "agent": {"name": "H1"},
                "rule": {"id": 1, "level": 3, "mitre": {"tactic": ["Discovery"],
                         "technique": ["System Owner/User Discovery"]}},
                "data": {"win": {"eventdata": {"commandLine": "whoami.exe", "image": "whoami.exe"}}},
            },
        },
        {
            "_id": "out-of-scope", "_source": {
                "timestamp": "2026-08-05T10:05:00.000Z", "cluster": {"node": "office-collector"},
                "agent": {"name": "H1"},
                "rule": {"id": 2, "level": 5, "mitre": {"tactic": ["Persistence"],
                         "technique": ["Scheduled Task"]}},
                "data": {"win": {"eventdata": {"commandLine": "schtasks.exe", "image": "schtasks.exe"}}},
            },
        },
    ]
    frame = osc.hits_to_frame(hits)
    scored = modeling.restrict_to_scored_techniques(frame)
    assert len(scored) == 1
    assert scored["rule_mitre_technique"].iloc[0] == config.TECHNIQUE_NAME_TO_ID["System Owner/User Discovery"]


# --------------------------------------------------------------------------
# count_alerts / discover_cluster_nodes / fetch_raw_hits - same import/
# credential-safety guarantees as fetch_alerts, checked directly rather than
# assumed from fetch_alerts's tests.
# --------------------------------------------------------------------------

def test_count_alerts_raises_before_connecting_when_env_missing(monkeypatch):
    for name in ("SOC_OPENSEARCH_HOST", "SOC_OPENSEARCH_USER", "SOC_OPENSEARCH_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises((RuntimeError, ImportError)):
        osc.count_alerts("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z")


def test_discover_cluster_nodes_raises_before_connecting_when_env_missing(monkeypatch):
    for name in ("SOC_OPENSEARCH_HOST", "SOC_OPENSEARCH_USER", "SOC_OPENSEARCH_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises((RuntimeError, ImportError)):
        osc.discover_cluster_nodes("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z")


def test_fetch_raw_hits_raises_before_connecting_when_env_missing(monkeypatch):
    for name in ("SOC_OPENSEARCH_HOST", "SOC_OPENSEARCH_USER", "SOC_OPENSEARCH_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises((RuntimeError, ImportError)):
        osc.fetch_raw_hits("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z")


# --------------------------------------------------------------------------
# Node allow-list: the real cluster's node names are not the synthetic
# generator's, so the default allow-list must be overridable, and must not
# silently accept everything either.
# --------------------------------------------------------------------------

def test_allowed_cluster_nodes_overridable_via_env(monkeypatch):
    import importlib

    from soc_runtime import config as config_module

    monkeypatch.setenv("SOC_ALLOWED_CLUSTER_NODES", "sysmon-collector-01, suricata-sensor-02")
    try:
        importlib.reload(config_module)
        assert config_module.ALLOWED_CLUSTER_NODES == {"sysmon-collector-01", "suricata-sensor-02"}
    finally:
        monkeypatch.delenv("SOC_ALLOWED_CLUSTER_NODES", raising=False)
        importlib.reload(config_module)  # restore the synthetic default for every later test


def test_allowed_cluster_nodes_defaults_to_synthetic_topology_without_env():
    from soc_runtime import config as config_module

    assert config_module.ALLOWED_CLUSTER_NODES == {"office-collector", "node01"}


# --------------------------------------------------------------------------
# clear_scroll on read-only credentials
#
# Confirmed on the real cluster (first real fetch, 189,034 documents over a
# 5-day window): the read-only user (backend roles kibanauser, readall) has
# `indices:data/read/scroll/clear` denied, and the search/scroll itself
# succeeded completely before that 403. A fake client stands in for the real
# one - opensearch-py is not installed in this environment (nor does this
# project's test suite need it to be), so `osc.AuthorizationException` is
# whichever class the module bound: the real one if opensearch-py happens to
# be present, the module's own placeholder if not - either way a valid
# exception class `_scroll`'s `except` clause can bind to and these tests
# can raise.
# --------------------------------------------------------------------------

class _FakeScrollClient:
    """Just enough of the OpenSearch client surface for `_scroll` to run
    against: an initial `search`, repeated `scroll` calls until a page comes
    back empty, then one `clear_scroll` - optionally rigged to raise."""

    def __init__(self, pages: list[list[dict]], clear_scroll_effect: BaseException | None = None):
        self._pages = list(pages)
        self.clear_scroll_effect = clear_scroll_effect
        self.clear_scroll_called = False

    def _next_page(self) -> dict:
        page = self._pages.pop(0) if self._pages else []
        return {"_scroll_id": "fake-scroll-id", "hits": {"hits": page}}

    def search(self, index, body, scroll):
        return self._next_page()

    def scroll(self, scroll_id, scroll):
        return self._next_page()

    def clear_scroll(self, scroll_id):
        self.clear_scroll_called = True
        if self.clear_scroll_effect is not None:
            raise self.clear_scroll_effect


def test_scroll_survives_authorization_exception_on_clear_scroll(capsys):
    """The exact scenario found on the real cluster: read-only credentials,
    full fetch succeeds, clear_scroll 403s. Already-fetched hits must still
    come back, not be lost to an exception raised during cleanup."""
    pages = [[{"_id": "1"}, {"_id": "2"}], []]
    client = _FakeScrollClient(pages, clear_scroll_effect=osc.AuthorizationException("403 no perms"))

    hits = osc._scroll(client, "wazuh-alerts-4.x-*", {"query": {}}, page_size=2000)

    assert len(hits) == 2
    assert client.clear_scroll_called
    assert "clear_scroll skipped" in capsys.readouterr().out


def test_scroll_propagates_other_exceptions_from_clear_scroll():
    """Only the documented, expected AuthorizationException is swallowed - a
    real network problem (or anything else) on clear_scroll must still be
    visible, not silently absorbed alongside the expected case."""
    pages = [[{"_id": "1"}], []]
    client = _FakeScrollClient(pages, clear_scroll_effect=ConnectionError("network blip"))

    with pytest.raises(ConnectionError):
        osc._scroll(client, "wazuh-alerts-4.x-*", {"query": {}}, page_size=2000)


def test_scroll_calls_clear_scroll_normally_when_no_error(capsys):
    """The common case - full write-access or no permission problem - is
    unaffected: clear_scroll is still called, and nothing is printed about
    a permission gap that did not happen."""
    pages = [[{"_id": "1"}], []]
    client = _FakeScrollClient(pages)

    hits = osc._scroll(client, "wazuh-alerts-4.x-*", {"query": {}}, page_size=2000)

    assert len(hits) == 1
    assert client.clear_scroll_called
    assert "clear_scroll skipped" not in capsys.readouterr().out
