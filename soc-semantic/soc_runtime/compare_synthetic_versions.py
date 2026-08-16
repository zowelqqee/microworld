"""Runs the *exact same* validation + router pipeline over both synthetic
corpora - v1 (`synthetic.py`, arbitrary parameters, documented baseline for
every existing published number) and v2 (`synthetic_v2.py`, calibrated
against real 30-day aggregate statistics) - and writes an honest
side-by-side comparison.

**This does not assume in advance which corpus produces "better" numbers.**
v2 is a step toward more realistic synthetic *shape* (null rates, technique
distribution, frequency skew, multi-valued tags) - it is not expected to
automatically produce higher recall or precision, and if it does not, that
is reported exactly as plainly as if it did. Neither corpus is evidence
about real SOC alert quality; both use the fabricated `is_synthetic_anomaly`
label described in `modeling.py` and the README's Limitations section.

Run:
    python -m soc_runtime.compare_synthetic_versions

Writes:
    artifacts/validation_report_v2.json   (v2's `modeling.run_validation` output)
    artifacts/router_results_v2.json      (v2's `router.run_router` output)
    artifacts/synthetic_v1_vs_v2_comparison.json
"""

from __future__ import annotations

import json
from typing import Callable

import pandas as pd

from soc_runtime import baseline, config, filters, modeling, router, semantic

_COMPARED_METRICS = ("recall_at_threshold", "precision_at_threshold", "pr_auc")
_COMPARED_ARMS = ("raw", "semantic", "union", "router->semantic", "router->union")


def _run_one(load_raw_alerts: Callable[[], pd.DataFrame], label: str) -> dict:
    raw = load_raw_alerts()
    filtered = filters.filter_customer_data(raw)
    featured = semantic.build_features(filtered)
    featured = baseline.build_features(featured)
    scored = modeling.restrict_to_scored_techniques(featured)

    validation = modeling.run_validation(scored)
    router_results = router.run_router(scored, raw_alerts_loader=load_raw_alerts)

    process = filtered[filtered["event_category"] == "process_creation"]
    return {
        "label": label,
        "n_process_creation": int(len(process)),
        "n_scored": int(len(scored)),
        "validation": validation,
        "router": router_results,
    }


def _arm_metrics(run: dict, arm: str) -> dict | None:
    """Metrics for `arm` from whichever report actually has it: the three
    base arms (raw/semantic/union) live in `validation.arms`, the two
    router variants live in `router.arms`."""
    if arm in run["validation"]["arms"]:
        return run["validation"]["arms"][arm]["metrics"]
    if arm in run["router"]["arms"]:
        return run["router"]["arms"][arm]["metrics"]
    return None


def build_delta(v1: dict, v2: dict) -> dict:
    delta: dict[str, dict] = {}
    for arm in _COMPARED_ARMS:
        m1, m2 = _arm_metrics(v1, arm), _arm_metrics(v2, arm)
        if m1 is None or m2 is None:
            continue
        delta[arm] = {}
        for metric in _COMPARED_METRICS:
            a, b = m1.get(metric), m2.get(metric)
            delta[arm][metric] = {
                "v1": a, "v2": b,
                "delta": round(b - a, 6) if (a is not None and b is not None) else None,
            }
    return delta


def main() -> int:
    from soc_runtime.synthetic import generate_alerts as v1_loader
    from soc_runtime.synthetic_v2 import generate_alerts_v2 as v2_loader, measured_calibration

    print("running v1 (synthetic.py) through validation + router ...")
    v1 = _run_one(v1_loader, "v1: synthetic.py, arbitrary parameters, 90-day window")

    print("running v2 (synthetic_v2.py) through validation + router ...")
    v2 = _run_one(v2_loader, "v2: synthetic_v2.py, calibrated to real 30-day aggregates")
    v2["calibration_vs_real_targets"] = {
        "targets": config.REAL_CALIBRATION_TARGETS,
        "measured": measured_calibration(v2_loader()),
    }

    delta = build_delta(v1, v2)

    comparison = {
        "disclaimer": (
            "SYNTHETIC DATA, both corpora. Neither v1 nor v2 is evidence about real SOC "
            "alert quality - is_synthetic_anomaly is a fabricated label in both. v2's only "
            "difference from v1 is that its generation parameters were calibrated toward "
            "real, non-personal, aggregated 30-day statistics (see "
            "config.REAL_CALIBRATION_TARGETS and the README's calibration section) - this is "
            "a step toward more defensible synthetic *shape*, not a claim that v2's "
            "validation numbers below are more 'correct' than v1's, and this comparison does "
            "not editorialize about which arm's numbers are better."
        ),
        "v1": {"label": v1["label"], "n_process_creation": v1["n_process_creation"], "n_scored": v1["n_scored"]},
        "v2": {"label": v2["label"], "n_process_creation": v2["n_process_creation"], "n_scored": v2["n_scored"]},
        "arm_metric_delta_v2_minus_v1": delta,
    }

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.ARTIFACTS_DIR / "validation_report_v2.json").write_text(
        json.dumps(v2["validation"], indent=2, default=str), encoding="utf-8"
    )
    (config.ARTIFACTS_DIR / "router_results_v2.json").write_text(
        json.dumps(v2["router"], indent=2, default=str), encoding="utf-8"
    )
    (config.ARTIFACTS_DIR / "synthetic_v1_vs_v2_comparison.json").write_text(
        json.dumps(comparison, indent=2, default=str), encoding="utf-8"
    )

    print("=" * 72)
    print("SYNTHETIC v1 vs v2 - honest comparison, not a claim v2 is 'better'")
    print("=" * 72)
    print(f"v1: {v1['n_process_creation']:,} process-creation / {v1['n_scored']:,} scored")
    print(f"v2: {v2['n_process_creation']:,} process-creation / {v2['n_scored']:,} scored")
    print()
    print(f"{'arm':<28}{'metric':<22}{'v1':>10}{'v2':>10}{'delta':>10}")
    for arm, metrics in delta.items():
        for metric, vals in metrics.items():
            v1_val = f"{vals['v1']:.4f}" if vals["v1"] is not None else "None"
            v2_val = f"{vals['v2']:.4f}" if vals["v2"] is not None else "None"
            d_val = f"{vals['delta']:+.4f}" if vals["delta"] is not None else "None"
            print(f"{arm:<28}{metric:<22}{v1_val:>10}{v2_val:>10}{d_val:>10}")

    print("\ncalibration achieved vs real targets (v2):")
    targets = config.REAL_CALIBRATION_TARGETS
    measured = v2["calibration_vs_real_targets"]["measured"]
    for key in measured:
        if key in targets:
            print(f"  {key:<48} measured={measured[key]!s:<10} target={targets[key]}")

    print(f"\n-> {config.ARTIFACTS_DIR / 'validation_report_v2.json'}")
    print(f"-> {config.ARTIFACTS_DIR / 'router_results_v2.json'}")
    print(f"-> {config.ARTIFACTS_DIR / 'synthetic_v1_vs_v2_comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
