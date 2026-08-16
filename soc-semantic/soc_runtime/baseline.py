"""The baseline: what today's context-free triage already sees.

No entities, no history, no relations - just the fields already sitting on
the alert: its declared severity, which technique fired, and the wall-clock
time it fired at. This is deliberately not a strawman: `rule.level` and
`rule.mitre.technique` are exactly the two fields the 90-day MITRE coverage
report in the README was built from, and a raw hour-of-day/weekend flag is
the kind of thing a correlation rule can and often does encode. If the
semantic layer's advantage evaporates against this, the honest conclusion is
that the advantage was not real.

One thing this baseline cannot see by construction: it has no memory. It
cannot know that the alert in front of it is the four-thousandth identical
alert from the same actor on the same host at the same time, because that
fact does not exist on the alert itself. That is exactly the gap the
semantic layer is built to close.
"""

from __future__ import annotations

import pandas as pd

from soc_runtime import config, semantic

FEATURES: tuple[tuple[str, str], ...] = (
    ("base_rule_level", "rule.level as shipped on the alert - the SOC's own static severity for this rule."),
    *[
        (f"base_technique_{t}", f"One-hot: whether this alert's technique is {t} "
                                  f"({config.TECHNIQUE_CATALOG[t]['description']}).")
        for t in config.IN_SCOPE_TECHNIQUES
    ],
    *[
        (f"base_hour_band_{label}", f"One-hot: whether this alert fired in the raw '{label}' wall-clock "
                                      "band, with no per-actor history behind it.")
        for _, _, label in config.HOUR_BANDS
    ],
    ("base_is_weekend", "Whether this alert fired on a Saturday or Sunday, read straight off the clock."),
)

FEATURE_NAMES: tuple[str, ...] = tuple(name for name, _ in FEATURES)


def _hour_band_label(ts: pd.Timestamp) -> str:
    for start, end, label in config.HOUR_BANDS:
        if start <= ts.hour < end:
            return label
    return config.HOUR_BANDS[-1][2]


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """`frame` should already be the process-creation subset (e.g. the output
    of `semantic.build_features`, or any frame with the same columns). Adds
    `base_*` columns and returns a copy; does not require time order, since
    every value here is read off the current alert alone."""
    work = frame.copy()
    work["base_rule_level"] = work["rule_level"].astype("float64")
    for t in config.IN_SCOPE_TECHNIQUES:
        work[f"base_technique_{t}"] = (work["rule_mitre_technique"] == t).astype("float64")
    hour_band = work["timestamp"].apply(_hour_band_label)
    for _, _, label in config.HOUR_BANDS:
        work[f"base_hour_band_{label}"] = (hour_band == label).astype("float64")
    work["base_is_weekend"] = (work["timestamp"].dt.dayofweek >= 5).astype("float64")
    return work


def main() -> int:
    from soc_runtime.synthetic import generate_alerts

    frame = generate_alerts()
    featured = semantic.build_features(frame)
    featured = build_features(featured)
    print(f"{len(featured):,} rows, {len(FEATURE_NAMES)} base_* columns: {FEATURE_NAMES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
