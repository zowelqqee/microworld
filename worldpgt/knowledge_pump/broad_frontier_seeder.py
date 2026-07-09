"""Seed the proposal-only pump frontier with broad encyclopedia topics.

This does not fetch data and does not touch accepted memory. It only adds
high-level Wikipedia page titles to the dynamic frontier artifact so the next
bounded Knowledge Pump network batch can branch into more general domains.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from worldpgt.knowledge_pump.title_ranker import normalize_title


BROAD_WIKIPEDIA_SEEDS: tuple[str, ...] = (
    "World history",
    "Ancient history",
    "Middle Ages",
    "Renaissance",
    "Industrial Revolution",
    "World War I",
    "World War II",
    "Cold War",
    "History of science",
    "Geography",
    "Africa",
    "Asia",
    "Europe",
    "North America",
    "South America",
    "Oceania",
    "United States",
    "China",
    "India",
    "Brazil",
    "Russia",
    "Japan",
    "France",
    "Germany",
    "United Kingdom",
    "Mathematics",
    "Physics",
    "Chemistry",
    "Biology",
    "Astronomy",
    "Earth science",
    "Medicine",
    "Computer science",
    "Artificial intelligence",
    "Internet",
    "Engineering",
    "Economics",
    "Political science",
    "Law",
    "Philosophy",
    "Religion",
    "Islam",
    "Christianity",
    "Buddhism",
    "Literature",
    "Music",
    "Film",
    "Sport",
    "Association football",
    "Olympic Games",
    "Human language",
    "English language",
    "Spanish language",
    "Arabic",
    "Culture",
    "Education",
    "Psychology",
    "Sociology",
    "Climate change",
    "Energy",
    "Agriculture",
    "Transportation",
)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [row for row in data if isinstance(row, dict)]


def _write_rows(rows: list[dict[str, Any]], json_path: Path, csv_path: Path) -> None:
    rows = sorted(rows, key=lambda row: (-int(row.get("weight") or 0), str(row.get("title") or "").casefold()))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["title", "source", "reason", "weight"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "title": row.get("title", ""),
                "source": row.get("source", ""),
                "reason": row.get("reason", ""),
                "weight": row.get("weight", 0),
            })


def seed_frontier(
    *,
    frontier_json: Path,
    frontier_csv: Path,
    weight: int = 120,
) -> dict[str, Any]:
    rows = _read_rows(frontier_json)
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        title = normalize_title(str(row.get("title") or ""))
        if title:
            row = dict(row)
            row["title"] = title
            by_key[title.casefold()] = row

    added: list[str] = []
    boosted: list[str] = []
    for raw_title in BROAD_WIKIPEDIA_SEEDS:
        title = normalize_title(raw_title)
        key = title.casefold()
        if key in by_key:
            old_weight = int(by_key[key].get("weight") or 0)
            if old_weight < weight:
                by_key[key]["weight"] = weight
                boosted.append(title)
            source = str(by_key[key].get("source") or "")
            if "broad_seed" not in source.split("|"):
                by_key[key]["source"] = f"{source}|broad_seed" if source else "broad_seed"
            continue
        by_key[key] = {
            "title": title,
            "source": "broad_seed",
            "reason": "broad encyclopedic seed for general coverage",
            "weight": weight,
        }
        added.append(title)

    out_rows = list(by_key.values())
    _write_rows(out_rows, frontier_json, frontier_csv)
    return {
        "frontier_json": str(frontier_json),
        "frontier_csv": str(frontier_csv),
        "seed_count": len(BROAD_WIKIPEDIA_SEEDS),
        "added_count": len(added),
        "boosted_count": len(boosted),
        "total_count": len(out_rows),
        "added_sample": added[:20],
        "boosted_sample": boosted[:20],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed dynamic frontier with broad Wikipedia topics")
    parser.add_argument("--frontier-json", required=True)
    parser.add_argument("--frontier-csv", required=True)
    parser.add_argument("--weight", type=int, default=120)
    args = parser.parse_args(argv)

    result = seed_frontier(
        frontier_json=Path(args.frontier_json),
        frontier_csv=Path(args.frontier_csv),
        weight=args.weight,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
