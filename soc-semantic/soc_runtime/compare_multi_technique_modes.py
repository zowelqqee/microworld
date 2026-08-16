"""How to handle a multi-valued `rule.mitre.technique`, measured three ways.

57.7% of real process-creation alerts (34,415 of 59,680 over 30 days) carry
more than one MITRE technique. Until now only the first survived
(`opensearch_client.first_of_multivalued`). This module runs the same
validation pipeline over the same corpus three times, changing only
`semantic.build_features`'s multi-technique mode, so the modes can be
compared on equal terms rather than argued about:

    first_only          read and fold the primary technique only - the
                        pre-fix behaviour, kept bit-exact as the baseline
    primary_plus_count  Option A: fold every technique into the causal
                        state, read features from the primary one, and add
                        `sem_alert_technique_multiplicity`
    aggregate           Option C: fold every technique, read every
                        technique, collapse per-feature toward the
                        least-explained value (see semantic._AGGREGATION)

Option B from the design discussion - exploding each multi-technique alert
into N rows - is deliberately **not** implemented. It would inflate the row
count by ~1.58x on this corpus purely by counting one real event several
times, which makes every count-based metric (`n_test`, flagged counts, the
router's traffic budget) incomparable with the numbers already published
without a correction factor that would itself need defending. It is a
legitimate design, just not one that can be compared honestly against the
existing baseline on a single run, so it is named and set aside rather than
half-measured.

**No mode is assumed to be better.** The comparison is run on the
calibrated v2 corpus (which actually contains multi-technique alerts) and,
for context, on v1 (which does not, so all three modes should agree - a
useful control that the mode plumbing changes nothing when there is nothing
to change).

Run:
    python -m soc_runtime.compare_multi_technique_modes

Writes artifacts/multi_technique_mode_comparison.json
"""

from __future__ import annotations

import json
from typing import Callable

import pandas as pd

from soc_runtime import baseline, config, filters, modeling, semantic

MODES = config.MULTI_TECHNIQUE_MODES
_METRICS = ("precision_at_threshold", "recall_at_threshold", "pr_auc")
_ARMS = ("baseline", "semantic", "raw_plus_semantic")


def run_mode(raw: pd.DataFrame, mode: str) -> dict:
    """One corpus, one mode, through the standard validation path."""
    featured = baseline.build_features(semantic.build_features(raw, mode=mode))
    scored = modeling.restrict_to_scored_techniques(featured)
    report = modeling.run_validation(scored, mode=mode)

    multiplicity = None
    name = semantic.MULTI_TECHNIQUE_FEATURE[0]
    if name in scored.columns:
        multiplicity = {
            "mean": round(float(scored[name].mean()), 4),
            "share_multi": round(float((scored[name] > 1).mean()), 4),
        }
    return {
        "mode": mode,
        "n_scored": int(len(scored)),
        "semantic_feature_count": len(semantic.feature_names(mode)),
        "alert_technique_multiplicity": multiplicity,
        "arms": {
            arm: {m: report["arms"][arm]["metrics"][m] for m in _METRICS}
            for arm in _ARMS
        },
        "chain_tactic_diversity_mean": round(
            float(scored["sem_chain_tactic_diversity_15m"].mean()), 4),
        "novel_actor_host_technique_mean": round(
            float(scored["sem_novel_actor_host_technique"].mean()), 6),
        "freq_actor_host_technique_24h_mean": round(
            float(scored["sem_freq_actor_host_technique_24h"].mean()), 2),
    }


def run_corpus(loader: Callable[[], pd.DataFrame], label: str) -> dict:
    raw = filters.filter_customer_data(loader())
    multi_share = None
    if semantic.TECHNIQUE_ALL_COLUMN in raw.columns:
        tagged = raw[raw["rule_mitre_technique"].notna()]
        if len(tagged):
            counts = tagged[semantic.TECHNIQUE_ALL_COLUMN].apply(len)
            multi_share = round(float((counts > 1).mean()), 4)
    return {
        "label": label,
        "multi_valued_share_among_tagged": multi_share,
        "modes": {mode: run_mode(raw, mode) for mode in MODES},
    }


def main() -> int:
    from soc_runtime.synthetic import generate_alerts
    from soc_runtime.synthetic_v2 import generate_alerts_v2

    print("running the three modes on the calibrated v2 corpus ...")
    v2 = run_corpus(generate_alerts_v2, "v2: calibrated, 57.7% multi-valued")
    print("running the three modes on v1 (control - no multi-valued alerts) ...")
    v1 = run_corpus(generate_alerts, "v1: original, single-technique by construction")

    results = {
        "disclaimer": (
            "SYNTHETIC DATA. is_synthetic_anomaly is a fabricated label in both corpora. "
            "This compares three ways of handling a real data-shape finding (57.7% of real "
            "process-creation alerts carry several MITRE techniques); it does not establish "
            "that any of them detects real attacks better. Option B (explode to N rows) is "
            "named in this module's docstring and deliberately not implemented - it would "
            "inflate row counts ~1.58x and make count-based metrics incomparable."
        ),
        "corpora": {"v2": v2, "v1": v1},
    }

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.ARTIFACTS_DIR / "multi_technique_mode_comparison.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    for corpus in (v2, v1):
        print("\n" + "=" * 78)
        print(corpus["label"])
        print(f"multi-valued share among MITRE-tagged alerts: {corpus['multi_valued_share_among_tagged']}")
        print("=" * 78)
        print(f"{'mode':<20}{'arm':<20}{'precision':>11}{'recall':>9}{'PR-AUC':>9}")
        for mode in MODES:
            entry = corpus["modes"][mode]
            for arm in _ARMS:
                m = entry["arms"][arm]
                pr = m["pr_auc"]
                print(f"{mode:<20}{arm:<20}{m['precision_at_threshold']:>11.4f}"
                      f"{m['recall_at_threshold']:>9.4f}{(pr if pr is not None else float('nan')):>9.4f}")
        print(f"\n{'mode':<20}{'tactic_div':>12}{'novel_aht':>12}{'freq_24h':>12}{'sem_feats':>11}")
        for mode in MODES:
            e = corpus["modes"][mode]
            print(f"{mode:<20}{e['chain_tactic_diversity_mean']:>12.4f}"
                  f"{e['novel_actor_host_technique_mean']:>12.6f}"
                  f"{e['freq_actor_host_technique_24h_mean']:>12.2f}"
                  f"{e['semantic_feature_count']:>11}")

    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
