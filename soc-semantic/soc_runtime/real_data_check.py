"""Bounded, careful first contact with the real OpenSearch cluster.

This is not a validation run. There is no ground truth on real data - no
fabricated `is_synthetic_anomaly` label, nothing to compute a real
precision/recall against. What this script *can* honestly do:

1. Audit `cluster.node` before touching anything else, to confirm
   `res-engineering-collector` really is present and really would be
   excluded, and that the node allow-list matches the real topology (it
   defaults to the synthetic generator's node names, which almost certainly
   are not the real ones - see `config.ALLOWED_CLUSTER_NODES`).
2. Check the window's document volume *before* scrolling through it, and
   refuse an accidentally huge pull.
3. Fetch a small, recent, bounded window and report - honestly, including
   failures - how much of it is Sysmon process-creation (the only kind this
   prototype scores), how much is Suricata/KES/other (seen and counted, not
   silently dropped - see config.py's "Fourth source" section), and how
   complete the fields this prototype depends on actually are.
4. Run the semantic + baseline feature builders and report if they raise,
   with the actual exception, rather than swallowing it.
5. Report *descriptive* statistics on the resulting `sem_*` feature values -
   no fitted model needed for this, since novelty/frequency/typicality are
   computed directly from the data, not learned from a label. This is the
   closest honest thing to "does the shape of this look like the reported
   false-positive problem" available without real labels.
6. Optionally (--apply-synthetic-models) also score real features with the
   models `modeling.fit_arms` fits on *synthetic* data, clearly marked
   illustrative-only - see the loud caveat in that function.
7. Optionally (--incidents) cross-reference specific closed incidents you
   already know about (timestamp + host + rule id) against what the
   semantic layer would have scored them - the closest thing to a real check
   available without a cases API. This script ships no incident data of its
   own; you supply it.

Credentials are read from the environment
(`SOC_OPENSEARCH_HOST`/`_USER`/`_PASSWORD`) and never printed. Every number
this script writes to its report or stdout is an aggregate (a count, a
share, a mean) - never a raw per-row dump of sensitive field values, so
nothing here should require sending raw data anywhere to discuss the results.

Run:
    SOC_DATA_SOURCE=opensearch python -m soc_runtime.real_data_check
    SOC_DATA_SOURCE=opensearch python -m soc_runtime.real_data_check --days 3
    SOC_DATA_SOURCE=opensearch python -m soc_runtime.real_data_check --discover-nodes-only
    SOC_DATA_SOURCE=opensearch python -m soc_runtime.real_data_check --incidents incidents.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from soc_runtime import baseline, config, filters, modeling, semantic
from soc_runtime import opensearch_client as osc


class RealDataCheckError(RuntimeError):
    pass


def _require_opensearch_data_source() -> None:
    if config.DATA_SOURCE != "opensearch":
        raise RealDataCheckError(
            "real_data_check.py only runs against the live cluster. Set "
            "SOC_DATA_SOURCE=opensearch (and SOC_OPENSEARCH_HOST/_USER/_PASSWORD) "
            "in the environment before running this script - the synthetic default "
            "is deliberate and this script does not override it."
        )


def check_window_safety(days: int, allow_large_window: bool) -> None:
    if days > config.REAL_DATA_MAX_WINDOW_DAYS and not allow_large_window:
        raise RealDataCheckError(
            f"--days {days} exceeds REAL_DATA_MAX_WINDOW_DAYS={config.REAL_DATA_MAX_WINDOW_DAYS}. "
            "This is a deliberate guard against an accidentally huge first pull (the real index "
            "holds ~1.2M documents over 30 days). Pass --allow-large-window once you mean it."
        )


def check_volume_safety(count: int, allow_large_window: bool) -> None:
    if count > config.REAL_DATA_COUNT_SAFETY_LIMIT and not allow_large_window:
        raise RealDataCheckError(
            f"{count:,} documents in the requested window exceeds "
            f"REAL_DATA_COUNT_SAFETY_LIMIT={config.REAL_DATA_COUNT_SAFETY_LIMIT:,}. "
            "Narrow --days, or pass --allow-large-window to scroll through it anyway."
        )


# --------------------------------------------------------------------------
# 1. Node audit
# --------------------------------------------------------------------------

def node_audit(start_iso: str, end_iso: str) -> dict:
    counts = osc.discover_cluster_nodes(start_iso, end_iso)
    excluded = {n: c for n, c in counts.items() if n in config.EXCLUDED_CLUSTER_NODES}
    allowed = {n: c for n, c in counts.items() if n in config.ALLOWED_CLUSTER_NODES}
    unrecognised = {
        n: c for n, c in counts.items()
        if n not in config.EXCLUDED_CLUSTER_NODES and n not in config.ALLOWED_CLUSTER_NODES
    }
    return {
        "all_nodes": counts,
        "excluded_confirmed": excluded,
        "allowed": allowed,
        "unrecognised": unrecognised,
    }


# --------------------------------------------------------------------------
# 3. Source breakdown + field completeness
# --------------------------------------------------------------------------

def source_breakdown(hits: list[dict]) -> dict:
    counts: Counter = Counter()
    multi_technique = 0
    multi_tactic = 0
    for hit in hits:
        src = hit.get("_source", {})
        counts[osc.classify_source(src)] += 1
        technique = osc._get(src, "rule", "mitre", "technique")
        tactic = osc._get(src, "rule", "mitre", "tactic")
        if isinstance(technique, list) and len(technique) > 1:
            multi_technique += 1
        if isinstance(tactic, list) and len(tactic) > 1:
            multi_tactic += 1
    return {
        "by_source_kind": dict(counts),
        "multi_valued_technique_alerts": multi_technique,
        "multi_valued_tactic_alerts": multi_tactic,
        "note": (
            "multi-valued rule.mitre.technique/tactic alerts have only the first element kept "
            "(see opensearch_client.py) - a real gap if this count is large, since it means "
            "correlation alerts tagging several techniques are being undercounted per-technique."
        ),
    }


def unknown_source_sample(hits: list[dict], limit: int = 20) -> dict:
    """Characterises the `classify_source() == "unknown"` bucket without
    dumping raw document content - only the aggregate metadata needed to
    name what these documents actually are: `decoder.name`,
    `rule.description`, and the shape (just the key *names*, not values) of
    each document's top-level `data` dict. The first real run classified
    27,180 of 188,946 alerts (14.4%) as unknown; this exists so that bucket
    gets characterised on the next run instead of staying an unlabelled 14%
    of the corpus. Capped at `limit` distinct values per category so this
    cannot balloon into something close to a raw-data export.
    """
    decoders: Counter = Counter()
    rule_descriptions: Counter = Counter()
    data_key_shapes: Counter = Counter()
    n_unknown = 0
    for hit in hits:
        src = hit.get("_source", {})
        if osc.classify_source(src) != "unknown":
            continue
        n_unknown += 1
        decoder_name = osc._get(src, "decoder", "name")
        if decoder_name:
            decoders[decoder_name] += 1
        rule_description = osc._get(src, "rule", "description")
        if rule_description:
            rule_descriptions[rule_description] += 1
        data = src.get("data")
        if isinstance(data, dict):
            data_key_shapes[tuple(sorted(data.keys()))] += 1
    return {
        "n_unknown": n_unknown,
        "top_decoder_names": decoders.most_common(limit),
        "top_rule_descriptions": rule_descriptions.most_common(limit),
        "top_data_key_shapes": [
            {"keys": list(keys), "count": count}
            for keys, count in data_key_shapes.most_common(limit)
        ],
        "note": (
            "metadata only (decoder names, rule descriptions, data-key names) - never document "
            "content. Plausible explanations not yet confirmed: other native Wazuh alert types "
            "this prototype was never designed around (syscheck/FIM integrity events, rootcheck, "
            "agent connect/disconnect, vulnerability-detector, active-response) - the four "
            "confirmed source shapes (Sysmon, Suricata, KES, Windows auth) were never claimed to "
            "be the whole index, only the ones this prototype scores or explicitly tracks."
        ),
    }


def field_completeness(frame: pd.DataFrame) -> dict:
    """Null rates for every field the semantic/baseline layers actually
    depend on - not just `subject_user_name`, which is where the first real
    run's problem happened to be found. `host` (`agent_name`) and
    `technique`/`tactic` are checked with equal weight here, on the theory
    that a bad surprise should be found by looking, not by assuming the rest
    is fine because one field wasn't.
    """
    process = frame[frame["event_category"] == "process_creation"]
    if process.empty:
        return {"n_process_creation": 0}
    critical = (
        "subject_user_name", "subject_domain_name", "eventdata_user", "agent_name",
        "rule_mitre_technique", "rule_mitre_tactic", "image", "timestamp",
    )
    completeness = {
        field: {
            "null_count": int(process[field].isna().sum()),
            "null_share": round(float(process[field].isna().mean()), 4),
        }
        for field in critical
    }
    seen_techniques = set(process["rule_mitre_technique"].dropna().unique())
    # Two different things, not one: a value that never translated to an ID
    # at all (config.TECHNIQUE_NAME_TO_ID needs extending - a real gap) is
    # not the same finding as a correctly-translated ID this prototype just
    # doesn't happen to score (expected - a real environment tags far more
    # than the eight techniques in TECHNIQUE_CATALOG). Conflating them would
    # bury the actionable one in a long, mostly-expected list.
    unmapped_technique_names = sorted(
        t for t in seen_techniques if not osc._TECHNIQUE_ID_PATTERN.match(t)
    )
    unscored_technique_ids = sorted(
        t for t in seen_techniques
        if osc._TECHNIQUE_ID_PATTERN.match(t) and t not in config.TECHNIQUE_CATALOG
    )
    in_scope_share = float(process["rule_mitre_technique"].isin(config.IN_SCOPE_TECHNIQUES).mean())

    # The specific diagnostic finding 2 asks for: of the alerts where
    # subject_user_name is unusable, how many does the fallback field
    # actually rescue? This is the number that says whether the fallback
    # implemented in semantic.py is worth anything on this cluster, or
    # whether identity degrades to the host pseudo-actor for most of them.
    subject_missing = ~process["subject_user_name"].apply(
        lambda v: isinstance(v, str) and v.strip() != ""
    )
    eventdata_usable = process["eventdata_user"].apply(
        lambda v: isinstance(v, str) and v.strip() != ""
    )
    n_subject_missing = int(subject_missing.sum())
    n_rescued_by_fallback = int((subject_missing & eventdata_usable).sum())

    return {
        "n_process_creation": int(len(process)),
        "field_completeness": completeness,
        "distinct_techniques_seen": int(process["rule_mitre_technique"].nunique(dropna=True)),
        "unmapped_technique_names": unmapped_technique_names[:50],
        "n_unmapped_technique_names": len(unmapped_technique_names),
        "unmapped_technique_names_note": (
            "these values never translated to a MITRE ID at all - config.TECHNIQUE_NAME_TO_ID "
            "needs a new entry for each one (see opensearch_client.translate_technique_name). "
            "A non-empty list here is an actionable gap, unlike unscored_technique_ids below."
        ),
        "unscored_technique_ids": unscored_technique_ids[:50],
        "n_unscored_technique_ids": len(unscored_technique_ids),
        "share_technique_in_the_six_scored": round(in_scope_share, 4),
        "actor_identity_fallback_coverage": {
            "n_subject_user_name_missing": n_subject_missing,
            "n_rescued_by_eventdata_user": n_rescued_by_fallback,
            "rescued_share": (
                round(n_rescued_by_fallback / n_subject_missing, 4) if n_subject_missing else None
            ),
            "note": (
                "of the alerts subject_user_name could not identify, this is the share "
                "eventdata_user (data.win.eventdata.user) rescues instead of falling all the "
                "way back to the host pseudo-identity - see semantic.py's actor-resolution note."
            ),
        },
    }


# --------------------------------------------------------------------------
# 4-5. Feature building + descriptive stats
# --------------------------------------------------------------------------

def build_features_safely(
    frame: pd.DataFrame, mode: str | None = None,
) -> tuple[pd.DataFrame | None, str | None]:
    """Wraps feature building so a real-data parsing surprise produces a
    clear, reported message instead of a raw traceback and a dead script.

    `mode` defaults to `config.REAL_DATA_MULTI_TECHNIQUE_MODE`, not to
    `config.MULTI_TECHNIQUE_MODE`: the global default stays `first_only`
    only so v1's published numbers keep reproducing exactly, and v1 has no
    multi-technique alerts for the mode to matter on. Real data does - 57.7%
    of process-creation alerts - and running the real path under the mode
    that is known to corrupt the causal state would be choosing the wrong
    default where it actually costs something."""
    mode = mode or config.REAL_DATA_MULTI_TECHNIQUE_MODE
    try:
        featured = semantic.build_features(frame, mode=mode)
        featured = baseline.build_features(featured)
        return featured, None
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a diagnostic boundary
        return None, f"{type(exc).__name__}: {exc}"


def descriptive_semantic_stats(featured: pd.DataFrame) -> dict:
    """Model-free: these numbers come straight from the causal feature
    computation, not from any fitted classifier, so they need no ground
    truth and no synthetic-trained model to be honestly reportable."""
    scored = modeling.restrict_to_scored_techniques(featured)
    if scored.empty:
        return {"n_scored": 0}
    feature_stats = {
        name: {
            "mean": round(float(scored[name].mean()), 4),
            "median": round(float(scored[name].median()), 4),
            "share_nonzero": round(float((scored[name] != 0).mean()), 4),
        }
        for name in semantic.FEATURE_NAMES
    }
    return {
        "n_scored": int(len(scored)),
        "feature_stats": feature_stats,
        "note": (
            "descriptive only - no fitted model, no ground truth. A high novelty share is "
            "expected early in any short window (the whole history is new to the pipeline); "
            "these numbers are only informative once the pipeline has run against enough "
            "history that most legitimate actor/host/technique combinations have been seen "
            "before, which a 3-7 day first window will not yet show."
        ),
    }


# --------------------------------------------------------------------------
# 6. Illustrative synthetic-model scoring (heavily caveated)
# --------------------------------------------------------------------------

def illustrative_synthetic_model_scores(real_scored: pd.DataFrame) -> dict:
    """Applies the models `modeling.fit_arms` fits on SYNTHETIC data to REAL
    features. Explicitly illustrative, not a validated score - see the
    caveat in the returned dict, repeated in the README. Fits the synthetic
    arms fresh (fast - see modeling.py) rather than depending on any
    persisted model, and does so regardless of `config.DATA_SOURCE`, which
    will be "opensearch" while this script runs.
    """
    from soc_runtime.synthetic import generate_alerts

    synth = generate_alerts()
    synth = semantic.build_features(synth)
    synth = baseline.build_features(synth)
    synth_scored = modeling.restrict_to_scored_techniques(synth)
    train, test = modeling.chronological_split(synth_scored)
    fitted = modeling.fit_arms(train, test)

    out: dict = {
        "caveat": (
            "ILLUSTRATIVE ONLY. These models were fit on fabricated synthetic labels and have "
            "never been validated against anything real. This shows whether the flagged-share "
            "distribution on real data looks directionally plausible, not whether any individual "
            "score is accurate. Do not use these numbers to triage a real alert."
        ),
    }
    if real_scored.empty:
        out["error"] = "no in-scope real alerts to score"
        return out
    for arm_name in ("baseline", "semantic", "raw_plus_semantic"):
        feature_names = modeling.ARM_FEATURES[arm_name]
        if not set(feature_names) <= set(real_scored.columns):
            out[arm_name] = {"error": "required feature columns missing on real data"}
            continue
        X_real = real_scored[list(feature_names)].to_numpy("float64")
        proba = fitted[arm_name]["model"].predict_proba(X_real)[:, 1]
        flagged = proba >= config.DECISION_THRESHOLD
        out[arm_name] = {
            "n": int(len(real_scored)),
            "flagged_share": round(float(flagged.mean()), 4),
            "mean_score": round(float(proba.mean()), 4),
            "median_score": round(float(np.median(proba)), 4),
        }
    return out


# --------------------------------------------------------------------------
# 7. Known-incident cross-reference
#
# This prototype has no incident data of its own - no case IDs, no
# timestamps, nothing beyond the aggregate finding stated in the README
# ("several correlation rules closed as false positive"). This utility
# exists so that whoever *does* have specific closed-incident references
# (timestamp, host, rule id) can check what the semantic layer would have
# scored them, which is the closest thing to a real hypothesis check
# available without a /cases API. Supply incidents via --incidents, JSON:
#
#   [{"approx_timestamp": "2026-08-05T14:30:00Z", "agent_name": "HOST-01",
#     "rule_id": 92033, "note": "AD Enumeration Campaign, closed FP"}, ...]
#
# `rule_id` and `note` are optional; matching narrows by whichever fields
# are present.
# --------------------------------------------------------------------------

def match_known_incidents(
    frame: pd.DataFrame, incidents: list[dict], *, tolerance_minutes: int = 60,
) -> list[dict]:
    results = []
    for incident in incidents:
        ts = pd.Timestamp(incident["approx_timestamp"])
        window = frame[
            (frame["timestamp"] >= ts - pd.Timedelta(minutes=tolerance_minutes))
            & (frame["timestamp"] <= ts + pd.Timedelta(minutes=tolerance_minutes))
        ]
        if "agent_name" in incident:
            window = window[window["agent_name"] == incident["agent_name"]]
        if "rule_id" in incident:
            window = window[window["rule_id"] == incident["rule_id"]]

        matches = []
        for _, row in window.iterrows():
            entry = {
                "matched_timestamp": str(row["timestamp"]),
                "rule_id": row.get("rule_id"),
                "rule_mitre_technique": row.get("rule_mitre_technique"),
                "event_category": row.get("event_category"),
            }
            for name in semantic.FEATURE_NAMES:
                if name in row.index:
                    entry[name] = row[name]
            matches.append(entry)
        results.append({
            "incident": incident,
            "n_matches": len(matches),
            "matches": matches[:10],  # cap - this is a spot-check, not a dump
        })
    return results


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def _write_report(report: dict, out_path: Path | None) -> Path:
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = out_path or (config.ARTIFACTS_DIR / "real_data_check_report.json")
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=config.REAL_DATA_DEFAULT_WINDOW_DAYS)
    parser.add_argument("--end", type=str, default=None, help="ISO end timestamp, default now (UTC)")
    parser.add_argument("--allow-large-window", action="store_true")
    parser.add_argument("--discover-nodes-only", action="store_true")
    parser.add_argument("--apply-synthetic-models", action="store_true",
                         help="also score real features with the synthetic-fitted models (illustrative only)")
    parser.add_argument("--incidents", type=Path, default=None,
                         help="path to a JSON file of known closed-incident references, see module docstring")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    _require_opensearch_data_source()
    check_window_safety(args.days, args.allow_large_window)

    end_dt = dt.datetime.fromisoformat(args.end) if args.end else dt.datetime.now(dt.timezone.utc)
    start_dt = end_dt - dt.timedelta(days=args.days)
    start_iso, end_iso = start_dt.isoformat(), end_dt.isoformat()

    print("=" * 72)
    print("REAL DATA - first contact, bounded window. Not a validation run.")
    print("=" * 72)
    print("credentials: read from environment, not printed here")
    print(f"window: {start_iso} .. {end_iso}  ({args.days} day(s))")

    report: dict = {
        "disclaimer": "REAL DATA. Descriptive first-contact check only - see README. No ground truth available.",
        "window": {"start": start_iso, "end": end_iso, "days": args.days},
    }

    print("\n[1/6] node audit (unfiltered terms aggregation on cluster.node)")
    nodes = node_audit(start_iso, end_iso)
    report["node_audit"] = nodes
    for name, count in sorted(nodes["all_nodes"].items(), key=lambda kv: -kv[1]):
        if name in config.EXCLUDED_CLUSTER_NODES:
            tag = "EXCLUDED (res-engineering-collector, customer data)"
        elif name in config.ALLOWED_CLUSTER_NODES:
            tag = "allowed"
        else:
            tag = "UNRECOGNISED - not scored, add to SOC_ALLOWED_CLUSTER_NODES to include"
        print(f"    {name:<40} {count:>10,}  {tag}")
    if nodes["excluded_confirmed"]:
        print("    confirmed: res-engineering-collector is present in this window and will be "
              "excluded by the query filter and the customer-data filter.")
    elif not nodes["all_nodes"]:
        print("    no documents at all in this window (any source) - nothing to audit yet.")

    if args.discover_nodes_only:
        path = _write_report(report, args.out)
        print(f"\n--discover-nodes-only: stopping here. -> {path}")
        return 0

    print("\n[2/6] volume safety check")
    count = osc.count_alerts(start_iso, end_iso)
    report["raw_document_count"] = count
    print(f"    {count:,} documents in window (before any filtering)")
    check_volume_safety(count, args.allow_large_window)

    print("\n[3/6] fetching (single scroll pass)")
    hits = osc.fetch_raw_hits(start_iso, end_iso)
    print(f"    {len(hits):,} raw hits fetched")
    src_breakdown = source_breakdown(hits)
    report["source_breakdown"] = src_breakdown
    print(f"    by source: {src_breakdown['by_source_kind']}")
    if src_breakdown["multi_valued_technique_alerts"]:
        print(f"    {src_breakdown['multi_valued_technique_alerts']:,} alerts had multi-valued "
              "rule.mitre.technique (only the first is kept - see the note in the report)")

    n_unknown = src_breakdown["by_source_kind"].get("unknown", 0)
    if n_unknown:
        unknown_sample = unknown_source_sample(hits)
        report["unknown_source_sample"] = unknown_sample
        print(f"    {n_unknown:,} 'unknown' source alerts - top decoders: "
              f"{unknown_sample['top_decoder_names'][:5]}")

    frame_unfiltered = osc.hits_to_frame(hits)
    try:
        frame = filters.filter_customer_data(frame_unfiltered)
    except filters.CustomerDataError as exc:
        report["customer_data_filter_error"] = str(exc)
        path = _write_report(report, args.out)
        print(f"\n    CUSTOMER-DATA FILTER ERROR: {exc}\n    -> {path}")
        return 1
    print(f"    {len(frame):,} alerts kept after the cluster-node allow-list "
          f"({len(frame_unfiltered) - len(frame):,} dropped)")

    print("\n[4/6] field completeness (process-creation subset)")
    completeness = field_completeness(frame)
    report["field_completeness"] = completeness
    if completeness.get("n_process_creation"):
        print(f"    {completeness['n_process_creation']:,} process-creation alerts")
        for field, stats in completeness["field_completeness"].items():
            if stats["null_share"] > 0:
                print(f"      {field:<24} {stats['null_share']:.2%} null ({stats['null_count']:,})")
        if completeness["n_unscored_technique_ids"]:
            print(f"    {completeness['n_unscored_technique_ids']} distinct MITRE technique ID(s) seen "
                  "outside the six this prototype scores (expected - real environments tag many more)")
        if completeness["n_unmapped_technique_names"]:
            print(f"    WARNING: {completeness['n_unmapped_technique_names']} technique value(s) never "
                  f"translated to a MITRE ID - extend config.TECHNIQUE_NAME_TO_ID: "
                  f"{completeness['unmapped_technique_names'][:10]}")
        fallback = completeness["actor_identity_fallback_coverage"]
        if fallback["n_subject_user_name_missing"]:
            rescued_pct = f"{fallback['rescued_share']:.2%}" if fallback["rescued_share"] is not None else "n/a"
            print(f"    actor identity: {fallback['n_subject_user_name_missing']:,} alerts had no "
                  f"usable subject_user_name; eventdata_user rescued {fallback['n_rescued_by_eventdata_user']:,} "
                  f"of them ({rescued_pct}) - the rest fall back to host-level pseudo-identity")
    else:
        print("    no process-creation alerts in this window")

    print("\n[5/6] building semantic + baseline features")
    featured, error = build_features_safely(frame)
    if error:
        report["feature_build_error"] = error
        path = _write_report(report, args.out)
        print(f"    FAILED: {error}\n    -> {path}")
        return 1
    print(f"    OK - {len(featured):,} process-creation alerts featured")

    print("\n[6/6] descriptive stats (no ground truth, no model fit on real data)")
    desc_stats = descriptive_semantic_stats(featured)
    report["descriptive_semantic_stats"] = desc_stats
    print(f"    {desc_stats.get('n_scored', 0):,} alerts in the six scored techniques")

    if args.apply_synthetic_models:
        print("\n[optional] applying synthetic-fitted models to real features (ILLUSTRATIVE ONLY)")
        scored = modeling.restrict_to_scored_techniques(featured)
        illustrative = illustrative_synthetic_model_scores(scored)
        report["illustrative_synthetic_model_scores"] = illustrative
        for arm in ("baseline", "semantic", "raw_plus_semantic"):
            if arm in illustrative and "flagged_share" in illustrative[arm]:
                print(f"    {arm:<20} flagged_share={illustrative[arm]['flagged_share']:.2%}  "
                      f"mean_score={illustrative[arm]['mean_score']:.4f}")

    if args.incidents:
        incidents = json.loads(args.incidents.read_text())
        matches = match_known_incidents(frame, incidents)
        report["incident_matches"] = matches
        print(f"\n[incidents] {len(incidents)} incident(s) supplied, "
              f"{sum(m['n_matches'] for m in matches)} matching alert(s) found")

    path = _write_report(report, args.out)
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
