"""Allowlisted Wikipedia page title loading.

The default cap is intentionally small and deterministic. Titles are source
collection inputs only; loading them does not imply any trust in fetched pages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

MAX_DEFAULT_TITLES = 100

DEFAULT_PAGE_TITLES = [
    "Elon Musk",
    "Tesla, Inc.",
    "SpaceX",
    "Starlink",
    "Falcon 1",
    "Falcon 9",
    "Neuralink",
    "The Boring Company",
    "OpenAI",
    "SolarCity",
    "Larry Ellison",
    "Oracle Corporation",
    "Jeff Bezos",
    "Amazon",
    "Blue Origin",
    "Bernard Arnault",
    "LVMH",
    "Michael Bloomberg",
    "Bloomberg L.P.",
    "Forbes",
    "Bloomberg News",
    "Electric vehicle",
    "Battery electric vehicle",
    "Lithium-ion battery",
    "Gigafactory",
    "Rocket",
    "Reusable launch system",
    "Satellite internet",
    "Spacecraft",
    "Clean energy",
    "Solar power",
    "Net worth",
    "Stock price",
    "Market capitalization",
    "Revenue",
    "Chief executive officer",
    "Founder",
    "Subsidiary",
    "Product",
    "Organization",
    "Leadership",
    "Business magnate",
    "Entrepreneurship",
    "Automotive industry",
    "Aerospace manufacturer",
    "Artificial intelligence",
    "Brain-computer interface",
    "Hyperloop",
    "Social media",
    "X Corp.",
    "Twitter",
    "PayPal",
    "Zip2",
    "X.com",
    "Space tourism",
    "Satellite constellation",
    "Internet service provider",
    "Electric battery",
    "Energy storage",
    "Tesla Model S",
    "Tesla Model 3",
    "Tesla Model X",
    "Tesla Model Y",
    "Cybertruck",
    "Roadster",
    "Supercharger",
    "Autopilot",
    "Full Self-Driving",
    "SpaceX Starship",
    "Dragon 2",
    "International Space Station",
    "NASA",
    "Commercial Crew Program",
    "Launch vehicle",
    "Merlin rocket engine",
    "Raptor rocket engine",
    "Cape Canaveral Space Force Station",
    "Kennedy Space Center",
    "Vandenberg Space Force Base",
    "Low Earth orbit",
    "Mars",
    "Mars colonization",
    "Private spaceflight",
    "Space industry",
    "Electric car",
    "Renewable energy",
    "Energy industry",
    "Financial estimate",
    "Billionaire",
    "The World's Billionaires",
    "Fortune 500",
    "U.S. Securities and Exchange Commission",
    "Public company",
    "Privately held company",
    "Initial public offering",
    "Stock exchange",
    "Nasdaq",
    "New York Stock Exchange",
    "Economy of the United States",
    "Technology company",
    "Software company",
    "Cloud computing",
    "Satellite",
    "Telecommunications",
    "Internet access",
]


def dedupe_titles(titles: Iterable[str], limit: int | None = MAX_DEFAULT_TITLES) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw_title in titles:
        title = str(raw_title).strip()
        if not title:
            continue
        key = " ".join(title.split()).casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(title)
        if limit is not None and len(result) >= limit:
            break
    return result


def _coerce_titles(raw: object) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, dict) and isinstance(raw.get("titles"), list):
        return [str(item) for item in raw["titles"]]
    raise ValueError("page allowlist must be a JSON list or an object with a titles list")


def load_page_titles(path: str | Path, limit: int | None = MAX_DEFAULT_TITLES) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return dedupe_titles(_coerce_titles(data), limit=limit)


def write_default_allowlist(path: str | Path, overwrite: bool = False) -> list[str]:
    out = Path(path)
    titles = dedupe_titles(DEFAULT_PAGE_TITLES, limit=MAX_DEFAULT_TITLES)
    if overwrite or not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(titles, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return titles

