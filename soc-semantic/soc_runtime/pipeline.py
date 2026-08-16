"""Orchestration: load alerts -> filter -> semantic features -> baseline features.

This is the one module that knows about `config.DATA_SOURCE`. Everything
downstream (`semantic.py`, `baseline.py`, `modeling.py`) takes a DataFrame
with the schema documented in `synthetic.py` and does not know or care
whether it came from the generator or from a live OpenSearch cluster - that
is the seam the brief asked for, so swapping the source is a config change
here, not a rewrite there.
"""

from __future__ import annotations

import pandas as pd

from soc_runtime import baseline, config, filters, semantic


def load_alerts() -> pd.DataFrame:
    """Dispatches on `config.DATA_SOURCE`. Never touches the network unless
    `DATA_SOURCE == "opensearch"`, and even then only inside `fetch_alerts`
    - importing this module or calling it with the default source performs
    no I/O beyond in-memory generation."""
    if config.DATA_SOURCE == "synthetic":
        from soc_runtime.synthetic import generate_alerts

        return generate_alerts()
    if config.DATA_SOURCE == "opensearch":
        from soc_runtime.opensearch_client import fetch_alerts

        import datetime as _dt

        end = _dt.datetime.now(_dt.timezone.utc)
        start = end - _dt.timedelta(days=90)
        return fetch_alerts(start.isoformat(), end.isoformat())
    raise ValueError(
        f"unknown config.DATA_SOURCE={config.DATA_SOURCE!r}; expected 'synthetic' or 'opensearch'"
    )


def load_features_v2() -> pd.DataFrame:
    """Same as `load_features()`, but sourced from `synthetic_v2.py`'s
    real-aggregate-calibrated generator instead of `config.DATA_SOURCE`.
    Not wired into `config.DATA_SOURCE` - v2 is an additional, explicitly
    opt-in corpus for comparison (see `compare_synthetic_versions.py`), not
    a replacement default; `load_alerts`/`load_features` above are
    untouched and still serve v1 or a real OpenSearch cluster exactly as
    before."""
    from soc_runtime.synthetic_v2 import generate_alerts_v2

    frame = generate_alerts_v2()
    frame = filters.filter_customer_data(frame)
    featured = semantic.build_features(frame)
    featured = baseline.build_features(featured)
    return featured


def load_features() -> pd.DataFrame:
    """Full feature-build: load -> exclude customer data -> sem_* -> base_*.
    Returns the process-creation subset (all in-catalogue techniques,
    in-scope and escalation-only alike) with both feature families attached.
    `modeling.restrict_to_scored_techniques` narrows this further to the six
    techniques actually compared in the validation report."""
    frame = load_alerts()
    frame = filters.filter_customer_data(frame)
    featured = semantic.build_features(frame)
    featured = baseline.build_features(featured)
    return featured


def main() -> int:
    frame = load_features()
    print(f"data source: {config.DATA_SOURCE}")
    print(f"{len(frame):,} process-creation alerts featured "
          f"({len(semantic.FEATURE_NAMES)} sem_* + {len(baseline.FEATURE_NAMES)} base_* columns)")
    if "is_synthetic_anomaly" in frame.columns:
        n_anom = int(frame["is_synthetic_anomaly"].sum())
        print(f"{n_anom:,} labeled synthetic anomalies (label is fabricated - see README)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
