"""Honest validation: baseline vs semantic, on a chronological split.

**Every number this module produces is measured on synthetic data with
fabricated labels.** See the README before quoting any of it as evidence the
architecture works on real alerts - it is evidence the architecture is
internally consistent and does what it was designed to do on a controlled
input, which is a narrower and much weaker claim.

Two arms, same split, same model class, same seed:
    baseline  - `soc_runtime.baseline` features only (what today's alert
                already carries: severity, technique, raw time-of-day)
    semantic  - `soc_runtime.semantic` features only (actor/host/technique
                history, novelty, timing typicality, chain escalation)

The split is chronological, not shuffled: the first `config.TRAIN_FRACTION`
of the 90-day range trains, the remainder is held out. Shuffling would let
the model see later occurrences of a repeating actor/host/technique triple
during training and be asked about earlier ones at test time, which is not
the direction time actually runs.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from soc_runtime import baseline, config, semantic

LABEL_COLUMN = "is_synthetic_anomaly"


def chronological_split(frame: pd.DataFrame, train_fraction: float = config.TRAIN_FRACTION) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Day-boundary split: every training alert's calendar day precedes every
    test alert's calendar day. Returns (train, test), both still sorted."""
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("chronological_split requires a frame sorted by timestamp")
    days = pd.Index(sorted(frame["timestamp"].dt.normalize().unique()))
    cutoff = days[int(len(days) * train_fraction)]
    train = frame[frame["timestamp"] < cutoff].reset_index(drop=True)
    test = frame[frame["timestamp"] >= cutoff].reset_index(drop=True)
    return train, test


def restrict_to_scored_techniques(frame: pd.DataFrame) -> pd.DataFrame:
    """The dataset baseline/semantic are actually compared on: the six
    Discovery-tagged techniques named in the 90-day MITRE coverage report.
    T1003/T1021 (escalation-only, see config.py) are excluded here - they
    exist to give the chain feature something to detect, not to be scored,
    since a raw severity baseline already handles them correctly."""
    return frame[frame["rule_mitre_technique"].isin(config.IN_SCOPE_TECHNIQUES)].reset_index(drop=True)


#: The three arms this module fits, and the one router.py reuses rather than
#: refitting: "baseline" is the router's cheap primary ("raw" in router.py's
#: vocabulary), "semantic" and "raw_plus_semantic" ("union") are its two
#: escalation candidates.
ARM_FEATURES: dict[str, tuple[str, ...]] = {
    "baseline": baseline.FEATURE_NAMES,
    "semantic": semantic.FEATURE_NAMES,
    "raw_plus_semantic": baseline.FEATURE_NAMES + semantic.FEATURE_NAMES,
}


def arm_features(mode: str | None = None) -> dict[str, tuple[str, ...]]:
    """`ARM_FEATURES` for one multi-valued-technique mode.

    The semantic arm gains one column (`sem_alert_technique_multiplicity`)
    outside `first_only`, so the arm definitions cannot be a single frozen
    constant once modes exist. `ARM_FEATURES` above stays as the default
    mode's value - unchanged, and still what every published number was
    measured with - rather than being replaced by this function, so no
    existing caller's behaviour shifts.
    """
    sem = semantic.feature_names(mode)
    return {
        "baseline": baseline.FEATURE_NAMES,
        "semantic": sem,
        "raw_plus_semantic": baseline.FEATURE_NAMES + sem,
    }


def fit_model(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """The one place `LogisticRegression(**config.LOGREG_PARAMS)` is
    constructed. `router.py`'s out-of-fold folds call this directly; every
    held-out arm goes through `_fit_arm` below, which also calls this."""
    model = LogisticRegression(**config.LOGREG_PARAMS)
    model.fit(X, y)
    return model


def score_predictions(y: np.ndarray, proba: np.ndarray, pred: np.ndarray) -> dict:
    """Metrics for one already-scored arm. Factored out so `router.py` can
    score its routed/blended decisions with the exact same definitions used
    here, instead of a second copy that could quietly drift."""
    n_pos = int(y.sum())
    return {
        "n_test": int(len(y)),
        "n_positive": n_pos,
        "precision_at_threshold": float(precision_score(y, pred, zero_division=0)),
        "recall_at_threshold": float(recall_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, proba)) if n_pos else None,
        "roc_auc": float(roc_auc_score(y, proba)) if n_pos and n_pos < len(y) else None,
        "n_flagged_at_threshold": int(pred.sum()),
    }


def _fit_arm(train: pd.DataFrame, test: pd.DataFrame, feature_names: tuple[str, ...]) -> dict:
    X_train = train[list(feature_names)].to_numpy("float64")
    y_train = train[LABEL_COLUMN].to_numpy("int64")
    X_test = test[list(feature_names)].to_numpy("float64")
    y_test = test[LABEL_COLUMN].to_numpy("int64")

    model = fit_model(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= config.DECISION_THRESHOLD).astype("int64")

    metrics = score_predictions(y_test, proba, pred)
    importances = sorted(
        zip(feature_names, model.coef_[0].tolist()), key=lambda kv: -abs(kv[1])
    )
    return {"metrics": metrics, "coefficients": importances, "proba": proba, "pred": pred, "model": model}


def fit_arms(
    train: pd.DataFrame, test: pd.DataFrame, arms: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, dict]:
    """Fits `arms` (default `ARM_FEATURES`) once and returns {arm: _fit_arm(...) result}.

    Both `run_validation` below and `router.py` call this - the router does
    not fit its own copy of the raw model. "Not retrained" here means
    exactly that: one `fit_arms` call per split, its output reused for every
    router variant. There is no on-disk model cache the way ids-semantic has
    one for CatBoost - a `LogisticRegression` on a few thousand rows fits in
    well under a second, so caching would add bookkeeping without saving
    anything real.
    """
    arms = ARM_FEATURES if arms is None else arms
    return {arm: _fit_arm(train, test, feature_names) for arm, feature_names in arms.items()}


def recurring_legit_false_positive_rate(test: pd.DataFrame, pred: np.ndarray) -> dict:
    """The metric the brief actually asks for: of the test-set alerts
    belonging to the known-legitimate recurring inventory pattern, what
    share does this arm flag at the decision threshold? This is the
    per-alert analogue of the correlation rules the user's SOC closes as
    false positive with a boilerplate 'autocorrelation noise' reason."""
    from soc_runtime.synthetic import SERVICE_ACCOUNT

    is_recurring = (test["subject_user_name"] == SERVICE_ACCOUNT[0]).to_numpy()
    n = int(is_recurring.sum())
    if n == 0:
        return {"n_recurring_legit_alerts": 0, "flagged_share": None}
    flagged = int(pred[is_recurring].sum())
    return {
        "n_recurring_legit_alerts": n,
        "flagged_count": flagged,
        "flagged_share": round(flagged / n, 4),
    }


def legit_false_positive_rate(test: pd.DataFrame, pred: np.ndarray) -> dict:
    """False-positive rate across *all* non-anomalous test alerts - the
    fixed-schedule inventory agent plus the ordinary, off-schedule admin
    manual activity from `synthetic.py`'s `_admin_manual_activity`. The
    recurring-only metric above saturates at 0% for every arm (see README):
    the inventory agent's narrow fixed morning window is separable from raw
    time-of-day alone, so it does not by itself distinguish the arms. This
    broader population - legitimate activity that is *not* on a fixed
    schedule - is where the arms actually separate."""
    is_legit = ~test["is_synthetic_anomaly"].to_numpy()
    n = int(is_legit.sum())
    if n == 0:
        return {"n_legit_alerts": 0, "flagged_share": None}
    flagged = int(pred[is_legit].sum())
    return {
        "n_legit_alerts": n,
        "flagged_count": flagged,
        "flagged_share": round(flagged / n, 4),
    }


def anomaly_recall_by_type(test: pd.DataFrame, pred: np.ndarray) -> dict:
    """Recall at the decision threshold, broken down by injected anomaly
    type, so a reader can see which behavioural signal is doing the work
    rather than one pooled recall number."""
    out: dict[str, dict] = {}
    anomaly_types = [t for t in test["synthetic_anomaly_type"].dropna().unique()]
    for atype in sorted(anomaly_types):
        mask = (test["synthetic_anomaly_type"] == atype).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue
        caught = int(pred[mask].sum())
        out[atype] = {"n": n, "caught": caught, "recall": round(caught / n, 4)}
    return out


def run_validation(frame: pd.DataFrame, mode: str | None = None) -> dict:
    """`frame` must already carry both `sem_*` and `base_*` columns (i.e. be
    the output of `semantic.build_features` followed by `baseline.build_features`),
    restricted to the six in-scope techniques.

    `mode` must match the multi-valued-technique mode `frame` was built
    under - outside `first_only` the semantic arm carries one extra column,
    and asking for the wrong arm definition would `KeyError` on a missing
    column rather than quietly score the wrong feature set."""
    arms = arm_features(mode)
    train, test = chronological_split(frame)

    report: dict = {
        "disclaimer": (
            "SYNTHETIC DATA. is_synthetic_anomaly is a fabricated label used only to "
            "check that this architecture behaves as designed. It is not evidence about "
            "real SOC alert quality - see README Limitations."
        ),
        "split": {
            "train_alerts": int(len(train)),
            "test_alerts": int(len(test)),
            "train_start": str(train["timestamp"].min()) if len(train) else None,
            "train_end": str(train["timestamp"].max()) if len(train) else None,
            "test_start": str(test["timestamp"].min()) if len(test) else None,
            "test_end": str(test["timestamp"].max()) if len(test) else None,
            "test_positive": int(test[LABEL_COLUMN].sum()),
        },
        "decision_threshold": config.DECISION_THRESHOLD,
        "arms": {},
    }

    fitted = fit_arms(train, test, arms=arms)
    for arm_name, result in fitted.items():
        report["arms"][arm_name] = {
            "feature_count": len(arms[arm_name]),
            "metrics": result["metrics"],
            "top_coefficients": result["coefficients"][:8],
            "recurring_legit_false_positive_rate": recurring_legit_false_positive_rate(test, result["pred"]),
            "legit_false_positive_rate": legit_false_positive_rate(test, result["pred"]),
            "anomaly_recall_by_type": anomaly_recall_by_type(test, result["pred"]),
        }

    return report


def main() -> int:
    from soc_runtime.pipeline import load_features

    frame = load_features()
    scored = restrict_to_scored_techniques(frame)
    report = run_validation(scored)

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.ARTIFACTS_DIR / "validation_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("=" * 72)
    print("SYNTHETIC VALIDATION - architecture check only, see README")
    print("=" * 72)
    print(f"train {report['split']['train_alerts']:,} alerts / "
          f"test {report['split']['test_alerts']:,} alerts "
          f"({report['split']['test_positive']} labeled anomalies in test)")
    for arm_name, arm in report["arms"].items():
        m = arm["metrics"]
        fp_recurring = arm["recurring_legit_false_positive_rate"]
        fp_legit = arm["legit_false_positive_rate"]
        print(f"\n[{arm_name}] ({arm['feature_count']} features)")
        print(f"  precision={m['precision_at_threshold']:.3f}  recall={m['recall_at_threshold']:.3f}  "
              f"PR-AUC={m['pr_auc']}  flagged={m['n_flagged_at_threshold']}/{m['n_test']}")
        print(f"  recurring-legit (inventory agent) flagged: "
              f"{fp_recurring.get('flagged_count')}/{fp_recurring.get('n_recurring_legit_alerts')} "
              f"({fp_recurring.get('flagged_share')})")
        print(f"  all legitimate alerts flagged: "
              f"{fp_legit.get('flagged_count')}/{fp_legit.get('n_legit_alerts')} "
              f"({fp_legit.get('flagged_share')})")
    print(f"\n-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
