"""The customer-data exclusion is the one hard requirement in the brief that
must never regress silently, so it gets its own direct tests independent of
the synthetic generator (which already excludes the node by construction)."""

from __future__ import annotations

import pandas as pd
import pytest

from soc_runtime.filters import CustomerDataError, assert_no_excluded_nodes, filter_customer_data


def _frame(nodes: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"cluster_node": nodes, "value": range(len(nodes))})


def test_drops_excluded_node():
    frame = _frame(["office-collector", "res-engineering-collector", "node01"])
    out = filter_customer_data(frame)
    assert "res-engineering-collector" not in set(out["cluster_node"])
    assert len(out) == 2


def test_keeps_only_allowed_nodes_even_if_not_explicitly_excluded():
    """An allow-list is stricter than a block-list: an unrecognised node name
    (neither on the allow-list nor the exclusion list) must still be dropped."""
    frame = _frame(["office-collector", "some-new-collector-nobody-configured"])
    out = filter_customer_data(frame)
    assert list(out["cluster_node"]) == ["office-collector"]


def test_assert_raises_if_excluded_node_present():
    frame = _frame(["office-collector", "res-engineering-collector"])
    with pytest.raises(CustomerDataError):
        assert_no_excluded_nodes(frame)


def test_assert_passes_on_clean_frame():
    frame = _frame(["office-collector", "node01"])
    assert_no_excluded_nodes(frame)  # must not raise


def test_missing_node_column_is_an_error_not_a_silent_pass():
    frame = pd.DataFrame({"value": [1, 2, 3]})
    with pytest.raises(CustomerDataError):
        filter_customer_data(frame)


def test_dropping_every_row_raises_instead_of_returning_silently_empty():
    """Regression guard for the real-data failure mode this was added for:
    the allow-list defaults to the *synthetic* generator's two node names,
    which will not match a real cluster's actual node names. Without this
    check, pulling real data with the default allow-list would silently
    return an empty frame that looks exactly like "no alerts in this
    window" - the wrong diagnosis for "wrong allow-list"."""
    frame = _frame(["prod-collector-01", "prod-collector-02"])  # neither allowed nor excluded
    with pytest.raises(CustomerDataError, match="SOC_ALLOWED_CLUSTER_NODES"):
        filter_customer_data(frame)


def test_dropping_every_row_because_all_were_excluded_gets_a_different_message():
    """If every row present actually was the excluded customer node, that is
    a real (if unusual) outcome, not a misconfigured allow-list - the error
    message should say so rather than pointing at SOC_ALLOWED_CLUSTER_NODES."""
    frame = _frame(["res-engineering-collector", "res-engineering-collector"])
    with pytest.raises(CustomerDataError, match="excluded list"):
        filter_customer_data(frame)


def test_empty_input_frame_does_not_raise():
    """An empty window (genuinely no documents) is not the same failure as
    "all documents were dropped by the filter" - only the latter should raise."""
    frame = _frame([])
    out = filter_customer_data(frame)
    assert out.empty
