"""A confidence router over the models `modeling.py` already fits.

Same premise as `ids-semantic/ids_runtime/router.py`, ported to this
domain's entities (actor/host/technique instead of host/target/service) and
to this project's model class (`LogisticRegression`, not CatBoost). The
mechanics are the same:

    raw_score outside the band  ->  final = raw_score        (semantic/union not consulted)
    raw_score inside  the band  ->  final = secondary_score   (semantic, or union)

The motivating finding, this time from a real T.Hunter source rather than a
synthetic one: a manual review of closed incidents found several different
correlation rules - *AD Enumeration Campaign, Lateral Movement Detection,
Suspicious PowerShell Activity, Malware Campaign, Defense Evasion Burst* -
consistently closed as false positive, for a boilerplate "autocorrelation
noise" reason. That reads as a structural problem with what gets escalated
in the first place, not a one-off. A router is one way to change what gets
escalated: keep the current alerting cheap and unchanged everywhere it is
already confident, and reserve the expensive second opinion - human or
model - for the alerts where the primary signal is genuinely ambiguous.

Two things this module is careful about, same as the network port.

**The raw model is reused, not retrained.** `modeling.fit_arms(train, test)`
is called exactly once; this module never constructs its own
`LogisticRegression` for the raw/semantic/union arms. There is no on-disk
model cache the way ids-semantic has one for CatBoost - fitting three
logistic regressions on a few thousand rows takes milliseconds, so caching
would add bookkeeping without saving anything real. "Not retrained" here
means "fit once, in `modeling.fit_arms`, and every router variant below
reuses that same call's output."

**The band is fitted without touching the held-out set.** Training-set
scores from the deployed raw model are in-sample and optimistic - the model
has already fit those rows. Instead the raw model is refit on expanding
chronological folds of the training set, each fold scored by a model that
only saw earlier folds, giving honest out-of-fold scores. The band is chosen
from those; the deployed raw model itself (the one scoring the held-out set)
is never touched by this process.

Run:
    python -m soc_runtime.router
"""

from __future__ import annotations

import json
import time
from typing import Callable

import numpy as np
import pandas as pd

from soc_runtime import config, modeling

MAIN_ARMS = ("baseline", "semantic", "raw_plus_semantic")
#: Router-facing names for the same three arms, matching the vocabulary the
#: brief and the ids-semantic port use ("raw" as the cheap primary, "union"
#: as the combined escalation target).
ARM_ALIAS = {"baseline": "raw", "semantic": "semantic", "raw_plus_semantic": "union"}


# --------------------------------------------------------------------------
# Out-of-fold raw scores on the training set
# --------------------------------------------------------------------------

def expanding_folds(n_rows: int, n_folds: int) -> np.ndarray:
    """Chronological fold index 0..n_folds-1 for `n_rows` rows already sorted
    by time. No per-capture grouping is needed here, unlike ids-semantic -
    this dataset is one continuous 90-day stream, not several independent
    packet captures, so a single positional cut is the right one."""
    edges = np.linspace(0, n_rows, n_folds + 1).astype(int)
    folds = np.zeros(n_rows, dtype="int8")
    for k in range(n_folds):
        folds[edges[k]:edges[k + 1]] = k
    return folds


def out_of_fold_raw_scores(
    train: pd.DataFrame, n_folds: int = config.ROUTER_OOF_FOLDS,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Honest raw-model scores on the training set.

    Fold *k* is scored by a raw model fit on folds 0..k-1 only. Fold 0 has no
    earlier data and is therefore never scored; it contributes training rows
    and nothing else. The result covers (n_folds-1)/n_folds of the training
    set with scores no model in this chain has seen.
    """
    if not train["timestamp"].is_monotonic_increasing:
        raise ValueError("out_of_fold_raw_scores requires a time-sorted training frame")

    feature_names = modeling.ARM_FEATURES["baseline"]
    X = train[list(feature_names)].to_numpy("float64")
    y = train[modeling.LABEL_COLUMN].to_numpy("int64")
    folds = expanding_folds(len(train), n_folds)

    scores_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    meta: dict = {"folds": n_folds, "per_fold": []}
    for k in range(1, n_folds):
        fit_rows = np.flatnonzero(folds < k)
        score_rows = np.flatnonzero(folds == k)
        started = time.time()
        model = modeling.fit_model(X[fit_rows], y[fit_rows])
        proba = model.predict_proba(X[score_rows])[:, 1]
        scores_parts.append(proba)
        labels_parts.append(y[score_rows])
        meta["per_fold"].append({
            "fold": k, "fitted_on": int(fit_rows.size), "scored": int(score_rows.size),
            "positives_scored": int(y[score_rows].sum()),
            "seconds": round(time.time() - started, 4),
        })

    scores = np.concatenate(scores_parts)
    labels = np.concatenate(labels_parts)
    return scores, labels, meta


# --------------------------------------------------------------------------
# Choosing the band
# --------------------------------------------------------------------------

def _cut_points(values: np.ndarray, threshold: float, grid_points: int) -> np.ndarray:
    """Candidate band edges, spread over both the mass and the range.

    Quantiles of the score distribution alone are not quite enough: even a
    logistic-regression score concentrates some mass near 0 and 1, so a
    quantile-only grid loses resolution exactly where a band edge might need
    to sit. Quantiles of the *distinct* values fix that; the union of the two
    keeps resolution where the mass is as well.
    """
    if values.size == 0:
        return np.array([threshold])
    by_mass = np.quantile(values, np.linspace(0, 1, grid_points))
    distinct = np.unique(values)
    picks = np.unique(np.linspace(0, distinct.size - 1, grid_points).astype(int))
    return np.unique(np.concatenate([by_mass, distinct[picks]]))


def band_curve(scores: np.ndarray, labels: np.ndarray, threshold: float, grid_points: int) -> pd.DataFrame:
    """Every candidate band, with what it would cost and what it would catch.

    An *error* is an alert the raw model gets wrong at the operating
    threshold. `error_capture` is the share of those errors whose score
    falls inside the band - the share of the raw model's mistakes the router
    would get a second opinion on. `routed_share` is the share of all
    training alerts that second opinion would have to be computed for.
    """
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    errors = ((scores >= threshold).astype("int8") != labels)[order]
    cum_errors = np.concatenate([[0], np.cumsum(errors)])
    total_errors = int(errors.sum())
    n = scores.size

    los = _cut_points(sorted_scores[sorted_scores < threshold], threshold, grid_points)
    his = _cut_points(sorted_scores[sorted_scores >= threshold], threshold, grid_points)

    rows = []
    for lo in los:
        start = int(np.searchsorted(sorted_scores, lo, side="left"))
        for hi in his:
            stop = int(np.searchsorted(sorted_scores, hi, side="right"))
            routed = stop - start
            captured = int(cum_errors[stop] - cum_errors[start])
            rows.append({
                "lo": float(lo), "hi": float(hi),
                "routed": routed, "routed_share": routed / n,
                "errors_captured": captured,
                "error_capture": captured / total_errors if total_errors else 0.0,
            })
    curve = pd.DataFrame(rows)
    curve.attrs["total_errors"] = total_errors
    curve.attrs["n"] = n
    return curve


def select_band(curve: pd.DataFrame, budget: float) -> dict:
    """Most errors caught per the stated traffic budget; narrower band breaks ties."""
    affordable = curve.loc[curve["routed_share"] <= budget].copy()
    if affordable.empty:
        raise ValueError(f"no band fits the traffic budget {budget}")
    affordable["width"] = affordable["hi"] - affordable["lo"]
    affordable = affordable.sort_values(
        ["errors_captured", "routed", "width"], ascending=[False, True, True])
    best = affordable.iloc[0]
    return {
        "lo": float(best["lo"]), "hi": float(best["hi"]),
        "train_routed_share": float(best["routed_share"]),
        "train_error_capture": float(best["error_capture"]),
        "train_errors_captured": int(best["errors_captured"]),
        "train_errors_total": int(curve.attrs["total_errors"]),
        "budget": budget,
    }


# --------------------------------------------------------------------------
# Applying the router
# --------------------------------------------------------------------------

def route(primary: np.ndarray, secondary: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    routed = (primary >= lo) & (primary <= hi)
    final = np.where(routed, secondary, primary)
    return final, routed


def flip_analysis(y: np.ndarray, primary: np.ndarray, final: np.ndarray, routed: np.ndarray, threshold: float) -> dict:
    """What the second opinion actually changed, split into help and harm."""
    raw_right = (primary >= threshold) == (y == 1)
    final_right = (final >= threshold) == (y == 1)
    fixed = routed & ~raw_right & final_right
    broken = routed & raw_right & ~final_right
    return {
        "routed": int(routed.sum()),
        "routed_share": round(float(routed.mean()), 6),
        "routed_anomalies": int((routed & (y == 1)).sum()),
        "routed_legit": int((routed & (y == 0)).sum()),
        "raw_errors_inside_band": int((routed & ~raw_right).sum()),
        "raw_errors_outside_band": int((~routed & ~raw_right).sum()),
        "fixed": int(fixed.sum()),
        "broken": int(broken.sum()),
        "net_corrections": int(fixed.sum()) - int(broken.sum()),
    }


# --------------------------------------------------------------------------
# Operational cost
#
# The routed share is not the compute saving it looks like, for the same
# structural reason ids-semantic documents: `semantic.SemanticState` is a
# causal streaming graph, and `sem_freq_actor_host_technique_24h` etc. for
# alert N depend on every alert before it having already been folded into
# that state. There is no way to know which alerts will be routed to the
# secondary model *before* computing semantic features for all of them - so
# routing cannot skip semantic feature generation, only the secondary
# model's own inference. This function measures that split honestly rather
# than asserting it, and says plainly that at this corpus's scale (a few
# thousand rows) the absolute timings are too small and too noisy to be a
# real benchmark - the structural argument is what should transfer to a
# production-scale deployment, not these specific microseconds.
# --------------------------------------------------------------------------

def measure_operational_cost(
    frame: pd.DataFrame, fitted: dict, n_repeats: int = 5,
    raw_alerts_loader: Callable[[], pd.DataFrame] | None = None,
    mode: str | None = None,
) -> dict:
    from soc_runtime import baseline as baseline_mod
    from soc_runtime import semantic as semantic_mod

    if raw_alerts_loader is None:
        from soc_runtime.synthetic import generate_alerts as raw_alerts_loader

    # This function rebuilds features itself (it is timing that build), so
    # it needs the same multi-valued-technique mode the caller fitted under
    # - otherwise the union arm's column list would not match the frame.
    arms = modeling.arm_features(mode)
    raw_alerts = raw_alerts_loader()  # unfeatured, same corpus semantic.build_features consumes
    process_alerts = raw_alerts[raw_alerts["event_category"] == "process_creation"]
    n = len(process_alerts)

    sem_times = []
    for _ in range(n_repeats):
        started = time.perf_counter()
        semantic_mod.build_features(raw_alerts, mode=mode)
        sem_times.append(time.perf_counter() - started)
    sem_seconds_per_alert = (sum(sem_times) / n_repeats) / n

    base_times = []
    semantic_only = semantic_mod.build_features(raw_alerts, mode=mode)
    featured = semantic_only
    for _ in range(n_repeats):
        started = time.perf_counter()
        featured = baseline_mod.build_features(semantic_only)
        base_times.append(time.perf_counter() - started)
    base_seconds_per_alert = (sum(base_times) / n_repeats) / n

    scored = modeling.restrict_to_scored_techniques(featured)
    raw_model = fitted["baseline"]["model"]
    union_model = fitted["raw_plus_semantic"]["model"]
    X_raw = scored[list(arms["baseline"])].to_numpy("float64")
    X_union = scored[list(arms["raw_plus_semantic"])].to_numpy("float64")

    infer_times_raw, infer_times_union = [], []
    for _ in range(n_repeats):
        started = time.perf_counter()
        raw_model.predict_proba(X_raw)
        infer_times_raw.append(time.perf_counter() - started)
        started = time.perf_counter()
        union_model.predict_proba(X_union)
        infer_times_union.append(time.perf_counter() - started)
    raw_infer_per_alert = (sum(infer_times_raw) / n_repeats) / len(scored)
    union_infer_per_alert = (sum(infer_times_union) / n_repeats) / len(scored)

    return {
        "note": (
            "sem/base feature generation must run for every alert regardless of routing - "
            "the streaming state (frequency counters, novelty sets, temporal histograms) has "
            "to be updated by every alert to be correct for the next one. Routing can only "
            "skip the secondary model's inference call, not feature generation. At this "
            "corpus's scale (a few thousand rows, well under a second end to end) these are "
            "microsecond-scale timings dominated by Python/pandas overhead, not a production "
            "benchmark - reported for the shape of the argument, not the specific numbers."
        ),
        "n_alerts_timed": n,
        "semantic_feature_build_seconds_per_alert": sem_seconds_per_alert,
        "baseline_feature_build_seconds_per_alert": base_seconds_per_alert,
        "raw_model_inference_seconds_per_alert": raw_infer_per_alert,
        "union_model_inference_seconds_per_alert": union_infer_per_alert,
        "semantic_build_vs_union_inference_ratio": (
            round(sem_seconds_per_alert / union_infer_per_alert, 1) if union_infer_per_alert else None
        ),
    }


# --------------------------------------------------------------------------
# Anomaly-type breakdown
# --------------------------------------------------------------------------

def routing_by_anomaly_type(test: pd.DataFrame, routed: np.ndarray, pred: np.ndarray) -> dict:
    """For each injected anomaly type: how many were routed to the secondary
    model, and how many were caught (flagged) by the final decision - so a
    reader can see which behavioural signal the router actually rescues and
    which it lets raw's confident scoring decide (correctly or not)."""
    out: dict[str, dict] = {}
    for atype in sorted(test["synthetic_anomaly_type"].dropna().unique()):
        mask = (test["synthetic_anomaly_type"] == atype).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue
        out[atype] = {
            "n": n,
            "routed": int(routed[mask].sum()),
            "routed_share": round(float(routed[mask].mean()), 4),
            "caught": int(pred[mask].sum()),
            "recall": round(float(pred[mask].mean()), 4),
        }
    return out


# --------------------------------------------------------------------------
# Full run
# --------------------------------------------------------------------------

def run_router(
    frame: pd.DataFrame, raw_alerts_loader: Callable[[], pd.DataFrame] | None = None,
    mode: str | None = None,
) -> dict:
    """`frame` must be the six-technique-restricted, fully featured frame -
    i.e. `modeling.restrict_to_scored_techniques(pipeline.load_features())`.

    `raw_alerts_loader` is passed through to `measure_operational_cost` for
    its own, separate raw-alert timing measurement - it defaults to v1's
    `synthetic.generate_alerts` (see `measure_operational_cost`), so callers
    running this against a different corpus (e.g. `synthetic_v2.generate_alerts_v2`)
    must pass the matching loader explicitly or the operational-cost section
    would silently time the wrong generator.
    """
    train, test = modeling.chronological_split(frame)
    # `mode` must match what `frame`'s sem_* columns were built under - see
    # `modeling.arm_features`.
    fitted = modeling.fit_arms(train, test, arms=modeling.arm_features(mode))

    y_test = test[modeling.LABEL_COLUMN].to_numpy("int64")
    raw_proba = fitted["baseline"]["proba"]
    semantic_proba = fitted["semantic"]["proba"]
    union_proba = fitted["raw_plus_semantic"]["proba"]

    oof_scores, oof_labels, oof_meta = out_of_fold_raw_scores(train)
    curve = band_curve(oof_scores, oof_labels, config.DECISION_THRESHOLD, config.ROUTER_GRID_POINTS)
    # A coarser corpus (fewer distinct OOF scores - e.g. a 30-day window
    # with only six baseline-technique categories, as v2's calibrated
    # corpus has) can make config.ROUTER_TRAFFIC_BUDGET (5%) unreachable:
    # the narrowest available band can still route more than that share of
    # training traffic. Rather than crash, fall back to the smallest
    # ROUTER_BUDGET_FRONTIER budget that *is* reachable and record that
    # this happened - a real, honest artifact of the corpus's score
    # granularity, not something to paper over silently.
    budget_used = config.ROUTER_TRAFFIC_BUDGET
    fallback_budget = None
    try:
        band = select_band(curve, budget_used)
    except ValueError:
        for candidate in sorted(config.ROUTER_BUDGET_FRONTIER):
            if candidate <= budget_used:
                continue
            try:
                band = select_band(curve, candidate)
            except ValueError:
                continue
            fallback_budget = candidate
            budget_used = candidate
            break
        else:
            raise

    frontier = []
    for budget in config.ROUTER_BUDGET_FRONTIER:
        try:
            point = select_band(curve, budget)
        except ValueError:
            continue
        frontier.append({"budget": budget, **{k: point[k] for k in
                        ("lo", "hi", "train_routed_share", "train_error_capture")}})

    results: dict = {
        "disclaimer": (
            "SYNTHETIC DATA. is_synthetic_anomaly is a fabricated label used only to check "
            "that the router behaves as designed. Not evidence about real SOC alert quality "
            "- see README Limitations."
        ),
        "threshold": config.DECISION_THRESHOLD,
        "configured_traffic_budget": config.ROUTER_TRAFFIC_BUDGET,
        "traffic_budget_fallback": (
            {"reason": "configured budget unreachable at this corpus's OOF score granularity",
             "used_budget": fallback_budget}
            if fallback_budget is not None else None
        ),
        "split": {
            "train_alerts": int(len(train)), "test_alerts": int(len(test)),
            "test_positive": int(y_test.sum()),
        },
        "oof": {
            **oof_meta,
            "scored_rows": int(oof_scores.size),
            "scored_positives": int(oof_labels.sum()),
            "raw_errors": int(curve.attrs["total_errors"]),
            "raw_error_rate": round(float(curve.attrs["total_errors"] / oof_scores.size), 6),
        },
        "band": band,
        "band_frontier": frontier,
        "naive_band": {"lo": config.ROUTER_NAIVE_BAND[0], "hi": config.ROUTER_NAIVE_BAND[1]},
        "arms": {},
        "routing": {},
    }

    # The three non-router arms, using ARM_ALIAS naming (raw/semantic/union)
    # for consistency with the router variants below.
    for arm_key, alias in ARM_ALIAS.items():
        result = fitted[arm_key]
        results["arms"][alias] = {
            "metrics": result["metrics"],
            "recurring_legit_false_positive_rate": modeling.recurring_legit_false_positive_rate(test, result["pred"]),
            "legit_false_positive_rate": modeling.legit_false_positive_rate(test, result["pred"]),
            "anomaly_recall_by_type": modeling.anomaly_recall_by_type(test, result["pred"]),
        }

    variants = {
        "router->semantic": (band["lo"], band["hi"], semantic_proba),
        "router->union": (band["lo"], band["hi"], union_proba),
        "router->union (naive band)": (*config.ROUTER_NAIVE_BAND, union_proba),
    }
    for name, (lo, hi, secondary_proba) in variants.items():
        final, routed = route(raw_proba, secondary_proba, lo, hi)
        pred = (final >= config.DECISION_THRESHOLD).astype("int64")
        results["arms"][name] = {
            "metrics": modeling.score_predictions(y_test, final, pred),
            "band": [lo, hi],
            "recurring_legit_false_positive_rate": modeling.recurring_legit_false_positive_rate(test, pred),
            "legit_false_positive_rate": modeling.legit_false_positive_rate(test, pred),
            "anomaly_recall_by_type": modeling.anomaly_recall_by_type(test, pred),
        }
        results["routing"][name] = flip_analysis(y_test, raw_proba, final, routed, config.DECISION_THRESHOLD)
        results["routing"][name]["by_anomaly_type"] = routing_by_anomaly_type(test, routed, pred)

    # Sanity check, stated as a verdict rather than left to the reader: a
    # router that loses to the best single arm on recall is a real result to
    # report, not a bug to quietly fix by picking a friendlier metric.
    best_single = {
        metric: max(results["arms"][alias]["metrics"][metric] for alias in ARM_ALIAS.values())
        for metric in ("recall_at_threshold", "pr_auc")
    }
    results["sanity_check"] = {
        "best_single_arm": best_single,
        "verdict": {
            name: {
                metric: {
                    "router": results["arms"][name]["metrics"][metric],
                    "best_single": best_single[metric],
                    "beats_or_matches": bool(
                        results["arms"][name]["metrics"][metric] >= best_single[metric] - 1e-12
                    ),
                }
                for metric in ("recall_at_threshold", "pr_auc")
            }
            for name in variants
        },
    }

    results["operational_cost"] = measure_operational_cost(
        frame, fitted, raw_alerts_loader=raw_alerts_loader, mode=mode)

    return results


def main() -> int:
    from soc_runtime.pipeline import load_features

    frame = load_features()
    scored = modeling.restrict_to_scored_techniques(frame)
    results = run_router(scored)

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.ARTIFACTS_DIR / "router_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )

    print("=" * 72)
    print("SYNTHETIC CONFIDENCE ROUTER - architecture check only, see README")
    print("=" * 72)
    print(f"train {results['split']['train_alerts']:,} / test {results['split']['test_alerts']:,} "
          f"({results['split']['test_positive']} labeled anomalies in test)")
    print(f"out-of-fold raw error rate on train: {results['oof']['raw_error_rate']:.4%} "
          f"({results['oof']['raw_errors']}/{results['oof']['scored_rows']})")
    b = results["band"]
    print(f"fitted band [{b['lo']:.4f}, {b['hi']:.4f}] at {b['budget']:.0%} traffic budget "
          f"-> captures {b['train_error_capture']:.2%} of train OOF raw errors "
          f"({b['train_errors_captured']}/{b['train_errors_total']})")

    print()
    for alias in ("raw", "semantic", "union"):
        m = results["arms"][alias]["metrics"]
        print(f"  {alias:<28} recall={m['recall_at_threshold']:.4f}  "
              f"precision={m['precision_at_threshold']:.4f}  PR-AUC={m['pr_auc']}")
    for name in ("router->semantic", "router->union", "router->union (naive band)"):
        m = results["arms"][name]["metrics"]
        r = results["routing"][name]
        print(f"  {name:<28} recall={m['recall_at_threshold']:.4f}  "
              f"precision={m['precision_at_threshold']:.4f}  PR-AUC={m['pr_auc']}  "
              f"routed={r['routed_share']:.2%}  fixed={r['fixed']}  broken={r['broken']}")

    print("\nsanity check (router recall/PR-AUC vs best single arm):")
    for name, verdict in results["sanity_check"]["verdict"].items():
        recall_v = verdict["recall_at_threshold"]
        print(f"  {name:<28} recall {recall_v['router']:.4f} vs best-single {recall_v['best_single']:.4f} "
              f"-> {'OK' if recall_v['beats_or_matches'] else 'WORSE'}")

    print(f"\n-> {config.ARTIFACTS_DIR / 'router_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
