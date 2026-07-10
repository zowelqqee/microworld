"""Run every evaluation script in sequence and print a combined report."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

EVALS = [
    "eval_novelty",
    "eval_meter",
    "eval_rhyme",
    "eval_coherence",
    "eval_continuity",
    "eval_line_semantics",
    "eval_seeded_generation",
    "eval_reasoning",
    "eval_style",
]


def main() -> None:
    for name in EVALS:
        print("\n" + "#" * 60)
        print(f"# {name}")
        print("#" * 60)
        runpy.run_path(str(_HERE / f"{name}.py"), run_name="__main__")


if __name__ == "__main__":
    main()
