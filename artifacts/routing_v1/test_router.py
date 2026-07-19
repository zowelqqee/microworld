"""Smoke tests for the branch router pilot. Run with:
    PYTHONPATH=<repo> python3 -m pytest artifacts/routing_v1/test_router.py
Pins the calibrated behaviour (thr=0.85, margin=0.02): fast-path markers, clear
centroid routing, and the safe QA default on below-margin cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from router import BranchRouter  # noqa: E402


def _router():
    r = BranchRouter(threshold=0.85, margin=0.02)
    r.build()
    return r


def test_fast_path_reflective_what_if():
    r = _router()
    res = r.route("What if Elon Musk had not founded SpaceX?")
    assert res.branch == "reflective" and res.method == "fast_path"


def test_fast_path_reflective_why_might():
    r = _router()
    res = r.route("Why might Gwynne Shotwell be associated with rockets?")
    assert res.branch == "reflective" and res.method == "fast_path"


def test_fast_path_constrained_creative():
    r = _router()
    res = r.route("Write about Neuralink using only these facts")
    assert res.branch == "constrained_creative" and res.method == "fast_path"


def test_centroid_routes_clear_qa():
    r = _router()
    assert r.route("Who founded Blue Origin?").branch == "qa"


def test_centroid_routes_clear_pure_creative():
    r = _router()
    assert r.route("Compose a poem about rockets").branch == "pure_creative"


def test_underspecified_defaults_to_qa():
    r = _router()
    res = r.route("Tell me about SpaceX")
    assert res.branch == "qa"


def test_below_margin_fails_safe_to_qa():
    # The known near-tie must not commit to a creative/speculative branch.
    r = _router()
    res = r.route("Tell a short story about a rocket company like SpaceX")
    assert res.branch == "qa"


def test_clear_cases_all_correct():
    import json
    r = _router()
    cases = json.loads((Path(__file__).resolve().parent / "pilot_cases.json").read_text())["cases"]
    clear = [c for c in cases if c["boundary"] == "clear"]
    wrong = [c for c in clear if r.route(c["q"]).branch != c["gold"]]
    assert wrong == [], f"clear cases must route perfectly, got misroutes: {wrong}"
