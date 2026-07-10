"""Manual Pilat/Yeshua Phase-1 trace for inspectable state transitions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poemcore.world_state import StateDelta, StateFact, WorldState


def _fact(subject: str, predicate: str, object_: str) -> StateFact:
    return StateFact(subject, predicate, object_, t=0)


def _print(label: str, result) -> None:
    print(f"\n=== {label} ===")
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    state = WorldState.from_initial_facts((
        _fact("Pilat", "introduced", "scene"),
        _fact("Pilat", "located_at", "palace"),
        _fact("Yeshua", "introduced", "scene"),
        _fact("Yeshua", "located_at", "palace"),
    ))
    speaking = state.apply(StateDelta(assertions=(_fact("Pilat", "speaking_to", "Yeshua"),), label="t1 Pilat speaks"))
    _print("t1 accepted: Pilat speaks to Yeshua", speaking)
    questioning = speaking.state.apply(StateDelta(assertions=(_fact("Pilat", "questions", "Yeshua"),), label="t2 Pilat questions"))
    _print("t2 accepted: Pilat questions Yeshua", questioning)

    for label, delta in (
        ("invalid: Behemoth acts without introduction", StateDelta(assertions=(_fact("Behemoth", "acts", "laughs"),))),
        ("invalid: Yeshua appears in Moscow without moving", StateDelta(assertions=(_fact("Yeshua", "located_at", "Moscow"),))),
        ("invalid: Pilat occupies palace and Patriarch's Ponds", StateDelta(assertions=(_fact("Pilat", "moves_to", "Patriarch's Ponds"), _fact("Pilat", "moves_to", "palace")))),
    ):
        _print(label, questioning.state.apply(delta))


if __name__ == "__main__":
    main()
