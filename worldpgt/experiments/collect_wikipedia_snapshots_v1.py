"""Wikipedia Snapshot Collector v1.

Controlled source-collection layer only:

- writes local raw snapshots, normalized docs, manifest, readiness, and reports
- never writes accepted memory, accepted overlay, or promoted overlay
- never runs ingestion, promotion, regression, QA, planner, or context packing
- never calls the network unless ``--allow-network`` is explicitly passed

Usage::

    python3 worldpgt/experiments/collect_wikipedia_snapshots_v1.py --dry-run
    python3 worldpgt/experiments/collect_wikipedia_snapshots_v1.py --allow-network --limit 25
    python3 worldpgt/experiments/collect_wikipedia_snapshots_v1.py --validate-existing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

# Allow running directly as a script.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from worldpgt.wiki_snapshots.mediawiki_client import MediaWikiClient
from worldpgt.wiki_snapshots.page_title_loader import (
    MAX_DEFAULT_TITLES,
    load_page_titles,
    write_default_allowlist,
)
from worldpgt.wiki_snapshots.snapshot_manifest import build_manifest_row, write_manifest
from worldpgt.wiki_snapshots.snapshot_normalizer import safe_title_filename, write_normalized_doc
from worldpgt.wiki_snapshots.snapshot_readiness import evaluate_snapshot_readiness
from worldpgt.wiki_snapshots.snapshot_report import (
    build_report,
    build_summary,
    count_files,
    write_json,
)
from worldpgt.wiki_snapshots.types import ManifestRow, PageSnapshot, ReadinessResult

_EXPERIMENTS = Path(__file__).resolve().parent
_DEFAULT_OUTDIR = _EXPERIMENTS / "wiki_snapshots_v1"
_DEFAULT_ALLOWLIST = _DEFAULT_OUTDIR / "page_allowlist.json"
_DEFAULT_USER_AGENT = "MicroworldResearchBot/0.1 (local research; contact: YOUR_EMAIL)"

ClientFactory = Callable[..., MediaWikiClient]


def _read_snapshot(path: Path) -> PageSnapshot:
    data = json.loads(path.read_text(encoding="utf-8"))
    return PageSnapshot(
        title=data.get("title", ""),
        normalized_title=data.get("normalized_title", data.get("title", "")),
        pageid=data.get("pageid"),
        revision_id=data.get("revision_id"),
        timestamp=data.get("timestamp", ""),
        source_url=data.get("source_url", ""),
        api_url=data.get("api_url", ""),
        retrieved_at=data.get("retrieved_at", ""),
        raw_text=data.get("raw_text", ""),
        raw_text_sha256=data.get("raw_text_sha256", ""),
        license_note=data.get("license_note", ""),
        fetch_status=data.get("fetch_status", "error"),
        error=data.get("error", ""),
        links=list(data.get("links") or []),
    )


def _write_raw_snapshot(snapshot: PageSnapshot, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{safe_title_filename(snapshot.normalized_title or snapshot.title)}.json"
    path.write_text(
        json.dumps(snapshot.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _ensure_scaffold(out_dir: Path, allowlist_path: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_snapshots").mkdir(parents=True, exist_ok=True)
    (out_dir / "normalized_docs").mkdir(parents=True, exist_ok=True)
    write_default_allowlist(allowlist_path, overwrite=False)


def _build_existing_rows(out_dir: Path) -> tuple[list[ManifestRow], list[ReadinessResult], list[str]]:
    raw_dir = out_dir / "raw_snapshots"
    normalized_dir = out_dir / "normalized_docs"
    rows: list[ManifestRow] = []
    readiness: list[ReadinessResult] = []
    failed_pages: list[str] = []
    for raw_path in sorted(raw_dir.glob("*.json")):
        snapshot = _read_snapshot(raw_path)
        doc_path = normalized_dir / f"{safe_title_filename(snapshot.normalized_title or snapshot.title)}.md"
        result = evaluate_snapshot_readiness(snapshot, doc_path)
        readiness.append(result)
        rows.append(build_manifest_row(snapshot, raw_path, doc_path))
        if snapshot.fetch_status != "success":
            failed_pages.append(snapshot.normalized_title or snapshot.title)
    return rows, readiness, failed_pages


def run_collection(
    out_dir: str | Path = _DEFAULT_OUTDIR,
    titles_file: str | Path | None = None,
    limit: int = MAX_DEFAULT_TITLES,
    allow_network: bool = False,
    dry_run: bool = False,
    validate_existing: bool = False,
    user_agent: str = _DEFAULT_USER_AGENT,
    delay_sec: float = 0.5,
    timeout_sec: float = 20.0,
    client_factory: ClientFactory = MediaWikiClient,
) -> dict:
    out = Path(out_dir)
    allowlist_path = Path(titles_file) if titles_file else out / "page_allowlist.json"
    requested_limit = min(limit, MAX_DEFAULT_TITLES)
    _ensure_scaffold(out, allowlist_path)

    titles = load_page_titles(allowlist_path, limit=requested_limit)
    raw_dir = out / "raw_snapshots"
    normalized_dir = out / "normalized_docs"
    manifest_json = out / "snapshot_manifest.json"
    manifest_csv = out / "snapshot_manifest.csv"
    readiness_json = out / "snapshot_readiness_report.json"
    summary_json = out / "snapshot_collection_summary.json"
    report_json = out / "snapshot_collection_report.json"

    rows: list[ManifestRow] = []
    readiness: list[ReadinessResult] = []
    errors: list[dict] = []
    warnings: list[str] = []
    failed_pages: list[str] = []
    skipped_pages: list[str] = []
    fetched_count = 0
    network_calls = 0

    if allow_network:
        client = client_factory(
            allowed_titles=titles,
            allow_network=True,
            user_agent=user_agent,
            delay_sec=delay_sec,
            timeout_sec=timeout_sec,
        )
        for title in titles[:requested_limit]:
            try:
                snapshot = client.fetch_page(title)
                fetched_count += 1
            except Exception as exc:
                errors.append({"title": title, "reason": str(exc)})
                failed_pages.append(title)
                continue
            raw_path = _write_raw_snapshot(snapshot, raw_dir)
            doc_path = write_normalized_doc(snapshot, normalized_dir)
            row = build_manifest_row(snapshot, raw_path, doc_path)
            result = evaluate_snapshot_readiness(snapshot, doc_path)
            rows.append(row)
            readiness.append(result)
            if snapshot.fetch_status != "success":
                failed_pages.append(snapshot.normalized_title or snapshot.title)
        network_calls = getattr(client, "network_calls", fetched_count)
        skipped_pages = titles[requested_limit:]
    else:
        if dry_run and not validate_existing:
            warnings.append("dry_run_no_network_no_pages_fetched")
            skipped_pages = titles[:requested_limit]
        rows, readiness, failed_existing = _build_existing_rows(out)
        if validate_existing:
            failed_pages.extend(failed_existing)
        if not rows:
            write_manifest([], manifest_json, manifest_csv)

    if rows:
        write_manifest(rows, manifest_json, manifest_csv)
    elif not manifest_json.exists() or not manifest_csv.exists():
        write_manifest([], manifest_json, manifest_csv)

    write_json(readiness_json, [item.to_dict() for item in readiness])

    raw_total = count_files(raw_dir, ".json")
    normalized_total = count_files(normalized_dir, ".md")
    summary = build_summary(
        allowlist_total=len(titles),
        requested_limit=requested_limit,
        fetched_count=fetched_count,
        skipped_count=len(skipped_pages),
        rows=rows,
        raw_snapshots_total=raw_total,
        normalized_docs_total=normalized_total,
        network_calls=network_calls,
        allow_network=allow_network,
    )
    artifacts_written = [
        str(allowlist_path),
        str(manifest_json),
        str(manifest_csv),
        str(readiness_json),
        str(summary_json),
        str(report_json),
    ]
    if fetched_count:
        artifacts_written.extend([str(raw_dir), str(normalized_dir)])
    report = build_report(
        summary=summary,
        errors=errors,
        warnings=warnings,
        failed_pages=failed_pages,
        skipped_pages=skipped_pages,
        artifacts_written=artifacts_written,
        readiness=readiness,
    )
    write_json(summary_json, summary)
    write_json(report_json, report)
    return {
        "titles": titles,
        "summary": summary,
        "report": report,
        "rows": rows,
        "readiness": readiness,
        "out_dir": out,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wikipedia Snapshot Collector v1")
    parser.add_argument("--dry-run", action="store_true", help="do not fetch; write scaffold/report")
    parser.add_argument("--allow-network", action="store_true", help="explicitly allow MediaWiki API fetches")
    parser.add_argument("--limit", type=int, default=MAX_DEFAULT_TITLES)
    parser.add_argument("--titles-file", default=None)
    parser.add_argument("--validate-existing", action="store_true", help="validate local snapshots only")
    parser.add_argument("--out-dir", default=str(_DEFAULT_OUTDIR))
    parser.add_argument("--user-agent", default=_DEFAULT_USER_AGENT)
    parser.add_argument("--delay-sec", type=float, default=0.5)
    parser.add_argument("--timeout-sec", type=float, default=20.0)
    args = parser.parse_args(argv)

    result = run_collection(
        out_dir=args.out_dir,
        titles_file=args.titles_file,
        limit=args.limit,
        allow_network=args.allow_network,
        dry_run=args.dry_run or not args.allow_network,
        validate_existing=args.validate_existing,
        user_agent=args.user_agent,
        delay_sec=args.delay_sec,
        timeout_sec=args.timeout_sec,
    )
    summary = result["summary"]
    mode = "real-fetch" if args.allow_network else "dry-run/validate-existing"
    print("Wikipedia Snapshot Collector v1")
    print(f"  mode                         : {mode}")
    print(f"  allowlist_total              : {summary['allowlist_total']}")
    print(f"  requested_limit              : {summary['requested_limit']}")
    print(f"  fetched_count                : {summary['fetched_count']}")
    print(f"  success_count                : {summary['success_count']}")
    print(f"  failed_count                 : {summary['failed_count']}")
    print(f"  ready_for_self_ingestion     : {summary['ready_for_self_ingestion_count']}")
    print(f"  network_calls                : {summary['network_calls']}")
    print(f"  artifacts written to         : {result['out_dir']}")
    print(f"  auto_ingest                  : {summary['auto_ingest']}")
    print(f"  auto_promote                 : {summary['auto_promote']}")
    print(f"  trusted_memory_modified      : {summary['trusted_memory_modified']}")
    print(f"  accepted_overlay_modified    : {summary['accepted_overlay_modified']}")
    print(f"  promoted_overlay_modified    : {summary['promoted_overlay_modified']}")
    print(f"  safe_for_general_runtime     : {summary['safe_for_general_runtime']}")
    if not args.allow_network:
        print("  would_fetch_titles           :")
        for title in result["titles"][:summary["requested_limit"]]:
            print(f"    - {title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
