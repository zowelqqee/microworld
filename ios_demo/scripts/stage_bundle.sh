#!/usr/bin/env bash
#
# stage_bundle.sh — copy the real MicroWorld engine into the iOS app resources.
#
# Produces:  MicroWorldDemo/MicroWorldDemo/Python/app_packages/worldpgt
#            MicroWorldDemo/MicroWorldDemo/Python/app_packages/mw_ios.py
#
# What it does:
#   * copies the stdlib-only `worldpgt` package + `mw_ios.py` adapter
#   * strips build/test cruft and the numpy-only embedding caches (*.npy),
#     which the answer path never touches (proven in TECHNICAL_DECISION.md)
#
# The staged app_packages/ is git-ignored: it is generated, not source.
#
# Usage:  ios_demo/scripts/stage_bundle.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IOS_DEMO="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$IOS_DEMO/.." && pwd)"

# Source engine: prefer the phone bundle folder; fall back to the repo package.
if [ -d "$REPO_ROOT/microworld_cli/worldpgt" ]; then
  SRC_ENGINE="$REPO_ROOT/microworld_cli/worldpgt"
elif [ -d "$REPO_ROOT/worldpgt" ]; then
  SRC_ENGINE="$REPO_ROOT/worldpgt"
else
  echo "error: could not find a worldpgt/ package under $REPO_ROOT" >&2
  exit 1
fi

DEST="$IOS_DEMO/MicroWorldDemo/MicroWorldDemo/Python/app_packages"
ADAPTER="$IOS_DEMO/MicroWorldDemo/MicroWorldDemo/Python/mw_ios.py"

echo "Source engine : $SRC_ENGINE"
echo "Destination   : $DEST"

rm -rf "$DEST"
mkdir -p "$DEST"

# Copy the package, excluding generated / heavy / non-runtime artifacts.
rsync -a \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '*.pyc' \
  --exclude '*.npy' \
  --exclude 'benchmarks/' \
  --exclude 'tests/' \
  --exclude '*_test.py' \
  --exclude 'test_*.py' \
  "$SRC_ENGINE/" "$DEST/worldpgt/"

# Creative mode uses the proven poetry_lab narrative surface, not the factual
# QA word graph.  Stage its reasoning runtime and the prebuilt mixed-corpus
# artifact as a separate package boundary.
#
# On-device we ship the SLIM artifact (narrative_model.phone.json): the full
# one parses to ~1 GB of Python dicts and jetsam-kills the app on iPhone 11.
# The slim copy (~380 MB resident, same generation output) is produced by
# `python3 poetry_lab/cli.py slim-narrative`. It is staged under the runtime's
# expected name `narrative_model.json` so mw_ios.py needs no change.
POETRY_SRC="$REPO_ROOT/poetry_lab"
POETRY_DEST="$DEST/poetry_lab"
PHONE_ARTIFACT="$POETRY_SRC/artifacts/narrative_model.phone.json"
FULL_ARTIFACT="$POETRY_SRC/artifacts/narrative_model.json"
if [ -f "$PHONE_ARTIFACT" ]; then
  STAGE_ARTIFACT="$PHONE_ARTIFACT"
elif [ -f "$FULL_ARTIFACT" ]; then
  echo "warning: no slim phone artifact; staging the FULL one (may OOM on device)." >&2
  echo "         run 'python3 poetry_lab/cli.py slim-narrative' first." >&2
  STAGE_ARTIFACT="$FULL_ARTIFACT"
else
  echo "error: missing poetry_lab narrative artifact; run 'python3 poetry_lab/cli.py ingest-narrative'" >&2
  exit 1
fi
mkdir -p "$POETRY_DEST"
rsync -a --exclude '__pycache__/' --exclude '*.pyc' "$POETRY_SRC/poemcore/" "$POETRY_DEST/poemcore/"
mkdir -p "$POETRY_DEST/artifacts"
cp "$STAGE_ARTIFACT" "$POETRY_DEST/artifacts/narrative_model.json"
echo "Staged narrative artifact: $(basename "$STAGE_ARTIFACT") ($(du -h "$STAGE_ARTIFACT" | cut -f1))"

cp "$ADAPTER" "$DEST/mw_ios.py"

BYTES=$(du -sh "$DEST" | cut -f1)
NPY_LEFT=$(find "$DEST" -name '*.npy' | wc -l | tr -d ' ')
echo "Staged bundle size: $BYTES  (.npy files remaining: $NPY_LEFT)"
echo "Done. app_packages/ is ready to be bundled by Xcode as a folder reference."
