"""Compare Microworld continuation v1.1 outputs against v1.2 outputs.

Usage:
    python3 -m worldpgt.experiments.compare_v1_1_to_v1_2 \
        --before worldpgt/experiments/microworld_continuation_v1_1_outputs.csv \
        --after worldpgt/experiments/microworld_continuation_v1_2_outputs.csv \
        --output worldpgt/experiments/v1_1_vs_v1_2_comparison.json
"""

from __future__ import annotations

from worldpgt.experiments.compare_v1_to_v1_1 import compare, main

__all__ = ["compare", "main"]


if __name__ == "__main__":
    main()
