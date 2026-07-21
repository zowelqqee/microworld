# MicroWorld public demo

This is a thin public FastAPI wrapper over the existing
`AnswerOrchestrator`. It does not reimplement the reasoning core and does not
enable web search, the live pump, community context, or proposal overlays.

At runtime the service reads the tracked promoted overlay from the portable
bundle, applies a fixed entity/literal allowlist, rejects high-risk and
volatile items, and writes a small ephemeral overlay for the orchestrator.
Only that bounded public subset is used for answers or graph context.

## Local run

```bash
python3 -m uvicorn worldpgt.public_demo.app:app \
  --host 127.0.0.1 \
  --port 8000
```

Open `http://127.0.0.1:8000`. The engine initializes lazily on the first
`POST /ask`, so the landing page can show an honest warm-up state.

Run the focused tests:

```bash
python3 -m pytest worldpgt/tests/public_demo/test_public_demo.py -q
```

## Public API

`POST /ask`

```json
{"question": "What does SpaceX develop?"}
```

The required response fields are `answer`, `support_kind`, `edges_used`, and
`latency_ms`. The UI also consumes additive `decision` and `context_edges`
fields. `edges_used` contains only the exact relations selected by the same
deterministic entity/path planners used by the answer route. Audits return an
empty `edges_used` list.

`GET /health` always returns HTTP 200 while the web process is healthy. Its
`engine_status` is `cold`, `ready`, or `error`; Render can therefore health
check the lightweight service without forcing the reasoning core to warm.

## Configuration

- `MICROWORLD_RATE_LIMIT_PER_MINUTE` — per-process, per-IP request limit;
  defaults to `12`.
- `MICROWORLD_CORS_ORIGINS` — comma-separated additional frontend origins.
- `MICROWORLD_DEMO_OVERLAY_SOURCE` — optional path to another promoted source
  artifact. The fixed public allowlist is still applied and cannot be widened
  with an environment variable.

No API key or other secret is needed by this offline demo.

## Render

The repository-root `render.yaml` defines a one-worker Free web service, its
health check, exact Python version, build command, start command, and safe
configuration defaults. The build script restores the main runtime's three
read-only surface-index artifacts from the tracked portable bundle before
import validation.

Render Free web services currently spin down after 15 minutes without inbound
traffic and take about one minute to return. The page keeps the request open
and changes its loading copy as a cold start becomes likely. This is a research
demo, not a production uptime service.

