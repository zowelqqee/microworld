"""Meter metric — does the language layer hold a syllable target?

Rhythm is one of the structural properties the brief asks the architecture to
carry. Each line has a syllable target from the plan; this reports how tightly
generated lines cluster around it (mean absolute deviation and the share of
lines within ±1 syllable).
"""

from __future__ import annotations

from statistics import mean

from _harness import generated_lines, run_battery
from poemcore.text import syllables


def main() -> None:
    results = run_battery()
    deviations: list[int] = []
    within1 = 0
    counts: list[int] = []
    for result in results:
        target = result.plan.target_syllables
        for line in generated_lines(result):
            n = syllables(line)
            counts.append(n)
            dev = abs(n - target)
            deviations.append(dev)
            if dev <= 1:
                within1 += 1
    total = len(deviations)
    print("=== meter ===")
    print(f"lines measured        : {total}")
    print(f"mean syllables/line   : {mean(counts):.2f}")
    print(f"mean abs deviation    : {mean(deviations):.2f} syllables from target")
    print(f"within ±1 syllable    : {within1}/{total} ({100*within1/total:.0f}%)")


if __name__ == "__main__":
    main()
