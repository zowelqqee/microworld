"""CLI integration tests for optional Multi-hop QA v1.

The ``--enable-multihop`` flag is opt-in. Default assistant CLI behavior stays
unchanged, and direct single-hop facts still win before the multi-hop layer.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
_ASK = _REPO / "worldpgt" / "experiments" / "ask_microworld_v1.py"


def _run_cli(question: str, *, multihop: bool = False) -> str:
    cmd = [
        sys.executable,
        str(_ASK),
        "--overlay",
        "pump-dry-run",
    ]
    if multihop:
        cmd.append("--enable-multihop")
    cmd.append(question)
    proc = subprocess.run(
        cmd,
        cwd=_REPO,
        check=True,
        text=True,
        capture_output=True,
    )
    return proc.stdout


def test_without_flag_connection_question_does_not_get_multihop_answer() -> None:
    out = _run_cli("How is Starlink connected to Falcon 9?")
    assert "Decision: answer" in out
    assert "Support: explicit connection path." in out
    assert "explicit multi-hop relation chain" not in out


def test_with_flag_starlink_falcon9_connection_answers_multihop() -> None:
    out = _run_cli("How is Starlink connected to Falcon 9?", multihop=True)
    assert "Starlink" in out
    assert "SpaceX" in out
    assert "Falcon 9" in out
    assert "owned by" in out
    assert "develops" in out
    assert "Support: explicit multi-hop relation chain." in out
    assert "Decision: answer" in out


def test_with_flag_through_node_question_answers() -> None:
    out = _run_cli("Is Starlink connected to Falcon 9 through SpaceX?", multihop=True)
    assert "Starlink is connected to Falcon 9 through SpaceX" in out
    assert "Support: explicit multi-hop relation chain." in out
    assert "Decision: answer" in out


def test_with_flag_solarcity_chain_answers() -> None:
    out = _run_cli("How is SolarCity connected to electric cars?", multihop=True)
    assert "SolarCity" in out
    assert "Tesla" in out
    assert "electric cars" in out
    assert "owned by" in out
    assert "produces" in out
    assert "Decision: answer" in out


def test_with_flag_current_sensitive_leader_chain_audits() -> None:
    out = _run_cli("How is Elon Musk connected to Falcon 9?", multihop=True)
    assert "Decision: audit" in out
    assert "path_blocked" in out
    assert "current_sensitive_relation:leader_of" in out


def test_direct_founder_question_still_audits_with_flag() -> None:
    out = _run_cli("Who founded Starlink?", multihop=True)
    assert "Decision: audit" in out
    assert "explicit multi-hop relation chain" not in out


def test_direct_ownership_question_uses_existing_single_hop_when_supported() -> None:
    out = _run_cli("Who owns SolarCity?", multihop=True)
    assert "SolarCity is owned by Tesla." in out
    assert "Support: semi-stable relation." in out
    assert "explicit multi-hop relation chain" not in out
    assert "Decision: answer" in out


def test_existing_single_hop_xm_founder_unchanged_with_flag() -> None:
    out = _run_cli("Who founded XM?", multihop=True)
    assert "Lon Levin" in out
    assert "Gary Parsons" in out
    assert "Support: semi-stable relation." in out
    assert "Decision: answer" in out


def test_existing_single_hop_june_definition_unchanged_with_flag() -> None:
    out = _run_cli("What is June?", multihop=True)
    assert "June is" in out
    assert "sixth month" in out
    assert "Support: stable definition." in out
    assert "Decision: answer" in out


def test_existing_single_hop_solarcity_owner_unchanged_without_flag() -> None:
    out = _run_cli("Who owns SolarCity?")
    assert "SolarCity is owned by Tesla." in out
    assert "Support: semi-stable relation." in out
    assert "Decision: answer" in out
