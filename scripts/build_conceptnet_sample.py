"""
Convert the ConceptNet assertions dump to a small CSV sample.

Usage:
    python scripts/build_conceptnet_sample.py \\
        --input  data/conceptnet-assertions-5.7.0.csv.gz \\
        --output data/conceptnet_sample.csv \\
        --lang   en \\
        --max-rows 5000

Input format (TSV, 5 fields):
    uri \\t relation \\t source \\t target \\t metadata_json

Output:
    CSV with header: source,relation_type,target
"""
import argparse
import csv
import gzip
import re
import sys

RELATION_MAP: dict[str, str] = {
    "/r/IsA":         "is_a",
    "/r/PartOf":      "part_of",
    "/r/HasA":        "has_a",
    "/r/UsedFor":     "used_for",
    "/r/CapableOf":   "capable_of",
    "/r/Causes":      "causes",
    "/r/HasProperty": "has_property",
    "/r/AtLocation":  "at_location",
    "/r/MadeOf":      "made_of",
}

# Allowed node form: only lowercase letters, digits, underscores; 1-40 chars.
# No spaces, no dashes, no slashes, no uppercase.
_CLEAN = re.compile(r"^[a-z][a-z0-9_]{0,38}[a-z0-9]$|^[a-z]{1,40}$")


def normalize_node(uri: str) -> str | None:
    """
    /c/en/apple/n         -> apple
    /c/en/fire_engine     -> fire_engine
    /c/en/bad-node        -> None  (hyphen rejected)
    /c/en/too long string -> None  (space rejected)

    Returns None if the node should be skipped.
    """
    parts = uri.split("/")
    # expected: ['', 'c', lang, concept, ...]
    if len(parts) < 4:
        return None
    concept = parts[3].lower()
    if not concept:
        return None
    if len(concept) > 40:
        return None
    if not _CLEAN.match(concept):
        return None
    return concept


def build_sample(
    input_path: str,
    output_path: str,
    lang: str = "en",
    max_rows: int = 5000,
    per_relation: int | None = None,
) -> int:
    """Stream the .gz dump, filter, normalise, deduplicate, write CSV.

    Never loads the full file into memory.

    per_relation: max rows per relation type (default: max_rows // n_relations).
      Without this cap the output is dominated by whichever relation type
      appears first in the dump.

    Returns the number of rows written.
    """
    prefix = f"/c/{lang}/"
    seen: set[tuple[str, str, str]] = set()
    per_rel_count: dict[str, int] = {}
    written = 0

    with (
        gzip.open(input_path, "rt", encoding="utf-8") as fin,
        open(output_path, "w", newline="", encoding="utf-8") as fout,
    ):
        writer = csv.writer(fout)
        writer.writerow(["source", "relation_type", "target"])

        for line in fin:
            if written >= max_rows:
                break
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                continue

            _uri, relation, src_uri, tgt_uri = fields[0], fields[1], fields[2], fields[3]

            rel_name = RELATION_MAP.get(relation)
            if rel_name is None:
                continue
            if per_relation is not None and per_rel_count.get(rel_name, 0) >= per_relation:
                continue
            if not src_uri.startswith(prefix):
                continue
            if not tgt_uri.startswith(prefix):
                continue

            src = normalize_node(src_uri)
            tgt = normalize_node(tgt_uri)
            if src is None or tgt is None:
                continue
            if src == tgt:
                continue

            key = (src, rel_name, tgt)
            if key in seen:
                continue
            seen.add(key)

            writer.writerow([src, rel_name, tgt])
            per_rel_count[rel_name] = per_rel_count.get(rel_name, 0) + 1
            written += 1

    return written


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a small ConceptNet sample CSV from the full dump."
    )
    ap.add_argument(
        "--input",
        default="data/conceptnet-assertions-5.7.0.csv.gz",
        help="Path to the .csv.gz assertions file",
    )
    ap.add_argument(
        "--output",
        default="data/conceptnet_sample.csv",
        help="Path for the output CSV",
    )
    ap.add_argument("--lang",     default="en",   help="Language code to keep")
    ap.add_argument("--max-rows", type=int, default=5000, dest="max_rows",
                    help="Maximum rows to emit")
    ap.add_argument("--per-relation", type=int, default=None, dest="per_relation",
                    help="Max rows per relation type (default: max_rows // 9)")
    args = ap.parse_args()

    per_relation = args.per_relation
    if per_relation is None:
        per_relation = max(1, args.max_rows // len(RELATION_MAP))

    n = build_sample(
        args.input, args.output,
        lang=args.lang, max_rows=args.max_rows, per_relation=per_relation,
    )
    print(f"wrote {n} rows → {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
