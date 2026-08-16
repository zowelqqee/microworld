#!/usr/bin/env python3
"""Full reproduction, synthetic data source, start to finish.

    python run_pipeline.py

Generates the synthetic alert stream, builds the semantic and baseline
feature layers, runs the chronological-split validation, fits the
confidence router on out-of-fold training scores, and writes
artifacts/semantic_catalogue.json, artifacts/validation_report.json and
artifacts/router_results.json.

To point this at a real OpenSearch cluster instead, set
`SOC_DATA_SOURCE=opensearch` plus `SOC_OPENSEARCH_HOST` / `_USER` /
`_PASSWORD` in the environment - see soc_runtime/opensearch_client.py and
the README before doing that.
"""

from __future__ import annotations

from soc_runtime import semantic, modeling, router


def main() -> int:
    semantic.main()
    print()
    modeling.main()
    print()
    return router.main()


if __name__ == "__main__":
    raise SystemExit(main())
