#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --disable-pip-version-check -r requirements.txt

# The main runtime intentionally ignores generated experiment data.  Render
# builds from a clean clone, so restore only the three read-only support
# artifacts needed by the custom-overlay surface index from the tracked
# portable bundle.  The public app still filters the promoted source down to
# its hard allowlist before constructing AnswerOrchestrator.
mkdir -p worldpgt/experiments/self_ingestion_v1/promotion
mkdir -p worldpgt/experiments/wiki_snapshot_ingestion_v1
cp microworld-standalone/worldpgt/experiments/accepted_wiki_memory_overlay_v1.json \
  worldpgt/experiments/accepted_wiki_memory_overlay_v1.json
cp microworld-standalone/worldpgt/experiments/self_ingestion_v1/promotion/promoted_wiki_memory_overlay_v1.json \
  worldpgt/experiments/self_ingestion_v1/promotion/promoted_wiki_memory_overlay_v1.json
cp microworld-standalone/worldpgt/experiments/wiki_snapshot_ingestion_v1/snapshot_dry_run_overlay.json \
  worldpgt/experiments/wiki_snapshot_ingestion_v1/snapshot_dry_run_overlay.json

python3 -c "from worldpgt.public_demo.app import app; assert app is not None"
