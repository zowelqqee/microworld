"""Rhyme metric — does the generator satisfy the plan's rhyme scheme?

For each stanza the plan assigns rhyme labels (ABAB): lines sharing a label are
supposed to end on matching rhyme keys. This checks, per label-pair, whether the
line endings actually share a ``rhyme_key`` — the constraint the generator plans
around by choosing the rhyme word before growing the line.
"""

from __future__ import annotations

from collections import defaultdict

from _harness import run_battery
from poemcore.text import last_word, rhyme_key


def main() -> None:
    results = run_battery()
    pairs_total, pairs_ok = 0, 0
    for result in results:
        scheme = result.plan.rhyme_scheme
        per_stanza = len(scheme)
        # walk poem lines excluding blanks/separators, regroup into stanzas of
        # per_stanza content lines matching the plan
        content = [ln for ln in result.poem.lines if ln.strip() and set(ln.strip()) - {"—", "-", " "}]
        given = set(result.request.given_verse.splitlines())
        content = [ln for ln in content if ln not in given]
        for s in range(0, len(content) - per_stanza + 1, per_stanza):
            stanza = content[s : s + per_stanza]
            by_label: dict[str, list[str]] = defaultdict(list)
            for i, line in enumerate(stanza):
                by_label[scheme[i % len(scheme)]].append(line)
            for _label, group in by_label.items():
                if len(group) < 2:
                    continue
                keys = {rhyme_key(last_word(g)) for g in group}
                pairs_total += 1
                if len(keys) == 1:
                    pairs_ok += 1
    rate = pairs_ok / pairs_total if pairs_total else 0.0
    print("=== rhyme ===")
    print(f"rhyme groups checked : {pairs_total}")
    print(f"groups that rhyme    : {pairs_ok}")
    print(f"rhyme success rate   : {rate:.3f}")


if __name__ == "__main__":
    main()
