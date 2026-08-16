"""The one job this module has: keep res-engineering-collector out.

That node belongs to a customer's own collector, mixed into the same
OpenSearch cluster for operational reasons. It is not part of this task and
must never be read, scored, or shipped in an artifact. Every entry point that
can produce a DataFrame of alerts - the synthetic generator and the
OpenSearch client - routes its output through `filter_customer_data` before
anything else touches it, so the exclusion happens twice: once in the
OpenSearch query itself, and again here as a belt-and-suspenders check that
raises instead of silently passing through if it ever fails.
"""

from __future__ import annotations

import pandas as pd

from soc_runtime import config


class CustomerDataError(RuntimeError):
    """Raised if excluded-customer data is found where it must never be."""


def filter_customer_data(frame: pd.DataFrame, *, node_column: str = "cluster_node") -> pd.DataFrame:
    """Drop any row whose cluster node is not on the explicit allow-list.

    Uses `config.ALLOWED_CLUSTER_NODES` (an allow-list) rather than only
    `EXCLUDED_CLUSTER_NODES` (a block-list): a node that appears in the
    cluster after this file was written is excluded by default, not included
    by default.
    """
    if node_column not in frame.columns:
        raise CustomerDataError(
            f"expected a '{node_column}' column to filter on; got {list(frame.columns)}"
        )
    seen_nodes = set(frame[node_column].dropna().unique())
    kept = frame[frame[node_column].isin(config.ALLOWED_CLUSTER_NODES)].reset_index(drop=True)
    assert_no_excluded_nodes(kept, node_column=node_column)

    if len(frame) > 0 and len(kept) == 0:
        # Every row was dropped. On synthetic data this cannot happen (the
        # generator only ever produces allow-listed nodes). On real data
        # this is almost always a configuration problem, not an empty
        # window - raise with the actual node names seen, rather than
        # silently handing back a frame that looks like "no alerts" when
        # the real answer is "wrong allow-list". See config.ALLOWED_CLUSTER_NODES.
        unrecognised = seen_nodes - config.EXCLUDED_CLUSTER_NODES
        raise CustomerDataError(
            f"filter_customer_data dropped all {len(frame)} row(s) - none of the cluster.node "
            f"values present {sorted(seen_nodes)} are on the allow-list "
            f"{sorted(config.ALLOWED_CLUSTER_NODES)}. "
            + (
                f"{sorted(unrecognised)} are not yet recognised - "
                "set SOC_ALLOWED_CLUSTER_NODES (comma-separated) to the real node names you "
                "intend to read from before pulling real data; the default allow-list is the "
                "synthetic generator's topology and will not match a real cluster."
                if unrecognised
                else "every row present was on the excluded list."
            )
        )
    return kept


def assert_no_excluded_nodes(frame: pd.DataFrame, *, node_column: str = "cluster_node") -> None:
    """Hard guard: raises if any excluded node slipped through.

    Called at the end of `filter_customer_data` and again right before the
    semantic layer or OpenSearch client hand off a frame, so a bug in one
    filter does not silently ship customer data past the other.
    """
    if node_column not in frame.columns:
        return
    present = set(frame[node_column].unique()) & config.EXCLUDED_CLUSTER_NODES
    if present:
        raise CustomerDataError(
            f"excluded cluster node(s) {sorted(present)} present in a frame that should "
            "never contain them - this is a bug, not a data issue. Stopping rather than "
            "scoring or persisting anything from this frame."
        )
