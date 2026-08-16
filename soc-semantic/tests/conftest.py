"""Session-wide test safety: no test may reach a live network, no matter
what the shell running pytest happens to have set.

The incident this guards against actually happened: a developer's shell
still had `SOC_DATA_SOURCE=opensearch` (and real `SOC_OPENSEARCH_HOST`/
`_USER`/`_PASSWORD`) set from an earlier `real_data_check.py` run.
`config.DATA_SOURCE` reads that environment variable at import time, and
several tests call `pipeline.load_alerts()`/`load_features()` trusting the
default was `"synthetic"` without forcing it. With the real values still in
the environment, those tests silently took the `"opensearch"` branch and ran
an *unbounded* 90-day scroll fetch against the real cluster instead of
generating synthetic data in memory - `pipeline.load_alerts()` has none of
`real_data_check.py`'s volume/window safety caps. Multi-minute (or longer)
hangs pulling from a real environment's data, not a bug in test logic, and
not something a test suite should ever be able to do by accident.

**Why this is module-level code, not a fixture.** A first version of this
used an `autouse=True`, function-scoped `monkeypatch` fixture. That is too
late: `tests/test_router.py`'s `scored_frame` fixture is `scope="module"`
and calls `pipeline.load_features()` once, the first time any test in that
module needs it - which happens *before* a function-scoped fixture for that
first test has run, by pytest's own scope-nesting order (broader-scoped
fixtures are set up "outside" narrower-scoped ones). A module-scoped
fixture reading the real ambient environment slipped through the
function-scoped version of this fix in exactly the same way the original
bug did. Plain module-level code in `conftest.py` runs at import time -
before pytest sets up any fixture of any scope for any test - which is
early enough.

Individual tests that want a specific `SOC_OPENSEARCH_*` value or a
non-`"synthetic"` `config.DATA_SOURCE` still set it themselves via their own
`monkeypatch` calls; `monkeypatch` snapshots whatever value is current when
it's called (which is the safe state this module already established) and
correctly restores exactly that at teardown, so this composes with the
existing tests in `tests/test_opensearch_client.py` and
`tests/test_real_data_check.py` without any special-casing.
"""

from __future__ import annotations

import os

from soc_runtime import config

_OPENSEARCH_ENV_VARS = (
    "SOC_DATA_SOURCE",
    "SOC_OPENSEARCH_HOST",
    "SOC_OPENSEARCH_PORT",
    "SOC_OPENSEARCH_USER",
    "SOC_OPENSEARCH_PASSWORD",
    "SOC_OPENSEARCH_INDEX",
    "SOC_ALLOWED_CLUSTER_NODES",
)

for _name in _OPENSEARCH_ENV_VARS:
    os.environ.pop(_name, None)
config.DATA_SOURCE = "synthetic"
