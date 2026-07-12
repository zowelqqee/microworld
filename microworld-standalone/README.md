# Microworld — standalone runtime

This is a self-contained local version of Microworld, an experimental semantic
AI runtime built around explicit memory, deterministic reasoning, conservative
dialogue state, and a controlled language layer.

The repository includes the code and the read-only demo artifacts required for
the default runtime. It works offline after installation: no model download,
API key, database, or external service is required.

## Run locally

Requires Python 3.11 or newer.

```bash
git clone <YOUR-REPOSITORY-URL> microworld
cd microworld
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install .
microworld "Who founded SpaceX?" --overlay pump-dry-run
```

For an interactive dialogue session:

```bash
microworld --overlay pump-dry-run --interactive
```

Run the local web interface and API:

```bash
microworld-api --overlay pump-dry-run --port 8000
```

Then open <http://127.0.0.1:8000>.

## Included data and boundaries

`worldpgt/experiments/` contains a compact checked-in runtime dataset:

- accepted, promoted, snapshot, and pump dry-run overlays;
- the read-only ontology layer;
- low-trust community language/cognitive patterns; and
- the phrase-learning artifacts used by the controlled renderer.

The default `pump-dry-run` overlay is deliberately labelled as a proposal. The
runtime reads it but never promotes it into accepted memory. It answers only
when explicit support is present and returns an audit for unsupported,
ambiguous, private, or current-sensitive requests.

This is a bounded research runtime, not an open-domain assistant or a source
of current facts. It runs with local artifacts only unless an optional
live-search path is explicitly enabled and configured by the caller.

## Quick verification

```bash
microworld "What is Starlink?" --overlay pump-dry-run --json
python3 -m worldpgt.experiments.benchmark_speech_quality_v1 --suite large --no-save
```

The included benchmark checks the deterministic speech/reasoning surface; it
does not establish open-domain factual coverage.

## Project layout

```text
worldpgt/
  api/                 FastAPI server and local web UI
  assistant_surface/   routing, support checks, and response orchestration
  cognition/           explicit reasoning and controlled rendering
  dialogue/            local session state and conservative coreference
  entity_qa/           semantic parsing, planning, and synthesis
  experiments/         checked-in local runtime data and CLI entry points
```

For the source license, see [LICENSE](LICENSE).
