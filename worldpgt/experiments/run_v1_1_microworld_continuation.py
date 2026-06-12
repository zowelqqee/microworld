"""Run the v1.1 Microworld controlled continuation benchmark.

Usage:
    python3 -m worldpgt.experiments.run_v1_1_microworld_continuation \
        --input worldpgt/experiments/continuation_prompts_v1.csv \
        --output worldpgt/experiments/microworld_continuation_v1_1_outputs.csv
"""

from __future__ import annotations

from worldpgt.experiments.run_v1_microworld_continuation import main, run

__all__ = ["run", "main"]


if __name__ == "__main__":
    main()
