"""Collect and finalize a diagnostic-only audit of Wikidata exact-QID misses."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import sleep
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from worldpgt.knowledge_pump.wikidata_resolution_gap import classify_manual_sample, seeded_original_failure_sample


_API = "https://www.wikidata.org/w/api.php"


class _Client:
    def __init__(self, *, user_agent: str, delay_seconds: float) -> None:
        self.user_agent = user_agent
        self.delay_seconds = delay_seconds
        self.calls = 0

    def search(self, label: str) -> list[dict[str, Any]]:
        if self.calls:
            sleep(self.delay_seconds)
        params = {
            "action": "wbsearchentities", "search": label, "language": "en",
            "strictlanguage": "true", "format": "json", "limit": "10", "type": "item",
        }
        request = Request(_API + "?" + urlencode(params), headers={"User-Agent": self.user_agent})
        with urlopen(request, timeout=45) as response:  # nosec B310: official HTTPS API
            self.calls += 1
            payload = json.loads(response.read().decode("utf-8"))
        return [
            {
                "qid": str(item.get("id") or ""),
                "label": str(item.get("label") or ""),
                "description": str(item.get("description") or ""),
                "match_type": str((item.get("match") or {}).get("type") or ""),
                "match_text": str((item.get("match") or {}).get("text") or ""),
            }
            for item in payload.get("search", []) if isinstance(item, dict)
        ]


def _resolution_manifest() -> list[dict[str, Any]]:
    return json.loads(Path("artifacts/open_book_qa/wikidata_density_recon/resolution_manifest.json").read_text(encoding="utf-8"))


def _summary(rows: list[dict[str, Any]], *, population: int, seed: int, source_counts: dict[str, Any]) -> dict[str, Any]:
    verdicts = Counter(str(row["verdict"]) for row in rows)
    failure_types = Counter(str(row.get("failure_type") or "unspecified") for row in rows if row["verdict"] == "matching_gap")
    absent_types = Counter(str(row.get("failure_type") or "unspecified") for row in rows if row["verdict"] == "genuinely_absent")
    n = len(rows)
    # Normal approximation is intentionally labelled approximate; n=30 is a
    # diagnostic sample, not a population census.
    rate = verdicts["matching_gap"] / n
    margin = 1.96 * ((rate * (1 - rate) / n) ** 0.5) if n else 0.0
    recommendation = (
        "repair matching logic in a separate follow-up" if verdicts["matching_gap"] > verdicts["genuinely_absent"]
        else "treat the current exact-QID coverage as a structural ceiling for this source cohort"
    )
    return {
        "version": "wikidata_resolution_gap_v1",
        "diagnostic_only": True,
        "production_resolver_modified": False,
        "extraction_performed": False,
        "precision_gate_run": False,
        "accepted_memory_modified": False,
        "serving_overlay_modified": False,
        "source": "official Wikidata Action API wbsearchentities; reviewed outside the production resolver",
        "original_331_failure_population": population,
        "sample_size": n,
        "sample_seed": seed,
        "sample_verdict_counts": dict(sorted(verdicts.items())),
        "matching_gap_rate": rate,
        "genuinely_absent_rate": verdicts["genuinely_absent"] / n,
        "matching_gap_rate_approx_95pct_margin": margin,
        "extrapolation_note": "Based on a seeded n=30 sample; the margin is an approximate normal 95% interval and does not remove classification uncertainty.",
        "matching_gap_failure_types": dict(sorted(failure_types.items())),
        "genuinely_absent_sample_subtypes": dict(sorted(absent_types.items())),
        "matching_gap_examples": [
            {
                "subject": row["subject"],
                "correct_wikidata_qid": row["correct_wikidata_qid"],
                "correct_wikidata_label": row["correct_wikidata_label"],
                "why_current_resolver_missed": row["failure_type"],
            }
            for row in rows if row["verdict"] == "matching_gap"
        ][:3],
        "original_failure_interpretation": "The reviewed failures are predominantly source-prose fragments or narrow/new technical methods and platforms, not mainstream entities that an exact resolver should normally identify.",
        "cohort_resolution_context": source_counts,
        "recommendation": recommendation,
        "crossref_openalex_note": "Crossref works are DOI-level academic publications, a category Wikidata covers less consistently than people, organisations, or broad concepts; this is a plausible structural contributor, not a resolver diagnosis. OpenAlex n=2 is too small for inference.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnostic-only Wikidata exact-resolution gap audit")
    parser.add_argument("--output-dir", default="artifacts/open_book_qa/wikidata_resolution_gap_v1")
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay-seconds", type=float, default=0.25)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--finalize", action="store_true", help="read reviewed_classifications.json and write summary.json")
    args = parser.parse_args()
    if args.sample_size != 30:
        raise SystemExit("this diagnostic is intentionally frozen at a 30-subject sample")
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = _resolution_manifest()
    sample = seeded_original_failure_sample(manifest, size=args.sample_size, seed=args.seed)
    failures = [row for row in manifest if "original_331" in (row.get("cohorts") or ()) and not row.get("canonical_qid")]
    root.joinpath("sample_manifest.json").write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not args.finalize:
        if not args.allow_network:
            raise SystemExit("refusing to fetch without --allow-network")
        user_agent = os.environ.get("MICROWORLD_WIKI_USER_AGENT", "")
        if not user_agent:
            raise SystemExit("MICROWORLD_WIKI_USER_AGENT is required")
        client = _Client(user_agent=user_agent, delay_seconds=args.delay_seconds)
        cache_path = root / "diagnostic_search_results.json"
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
        for row in sample:
            subject = str(row["subject"])
            if subject not in cache:
                cache[subject] = client.search(subject)
                cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"sample_size": len(sample), "searches_cached": len(cache), "network_calls": client.calls}, ensure_ascii=False))
        return 0

    reviewed_path = root / "reviewed_classifications.json"
    reviews = json.loads(reviewed_path.read_text(encoding="utf-8"))
    classified = classify_manual_sample(sample, reviews)
    root.joinpath("sample_classifications.json").write_text(json.dumps(classified, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    source_counts = {
        "original_331": "41/331 exact QID",
        "crossref_promoted_46": "2/46 exact QID",
        "openalex_accepted_2": "0/2 exact QID",
    }
    summary = _summary(classified, population=len(failures), seed=args.seed, source_counts=source_counts)
    root.joinpath("summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
