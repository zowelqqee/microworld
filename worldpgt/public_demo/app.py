"""Thin public FastAPI surface over the existing MicroWorld runtime.

The public demo deliberately builds a small allow-listed overlay from the
tracked promoted artifact.  It never serves the full pump/proposal graph and
never enables web search, community context, or cognitive-pattern inputs.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import is_dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from worldpgt.assistant_surface.answer_orchestrator import AnswerOrchestrator
from worldpgt.cross_page_qa.cross_page_question_analyzer import analyze as analyze_cross_page
from worldpgt.entity_qa.entity_question_analyzer import analyze as analyze_entity


_REPO_ROOT = Path(__file__).resolve().parents[2]
_STATIC_DIR = Path(__file__).resolve().parent / "static"
_DEFAULT_PROMOTED_SOURCE = (
    _REPO_ROOT
    / "microworld-standalone"
    / "worldpgt"
    / "experiments"
    / "self_ingestion_v1"
    / "promotion"
    / "promoted_wiki_memory_overlay_v1.json"
)

# Public boundary: every entity and literal that may leave the service is
# declared here.  Changing an environment variable cannot widen this set.
_PUBLIC_NODES = frozenset(
    {
        "Elon Musk",
        "SpaceX",
        "Tesla",
        "Starlink",
        "Falcon 9",
        "Dragon spacecraft",
        "Blue Origin",
        "Jeff Bezos",
        "rockets",
        "spacecraft",
        "Falcon rockets",
        "electric cars",
        "battery energy storage",
        "Starbase, Texas",
        "Kent, Washington",
    }
)
_PUBLIC_STABILITIES = frozenset({"stable", "semi_stable"})


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)


class GraphEdge(BaseModel):
    subject: str
    predicate: str
    object: str
    evidence_id: str


class AskResponse(BaseModel):
    answer: str
    support_kind: str
    edges_used: list[GraphEdge]
    latency_ms: float
    # Additive fields used by the demo UI.  The required contract above stays
    # stable for API consumers.
    decision: str
    context_edges: list[GraphEdge] = Field(default_factory=list)


class _SlidingWindowLimiter:
    """Small process-local IP limiter suitable for one free-tier worker."""

    def __init__(self, limit: int, window_seconds: int = 60) -> None:
        self.limit = max(1, limit)
        self.window_seconds = max(1, window_seconds)
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> int | None:
        current = time.monotonic() if now is None else now
        cutoff = current - self.window_seconds
        with self._lock:
            if len(self._events) > 5000:
                stale = [ip for ip, rows in self._events.items() if not rows or rows[-1] <= cutoff]
                for ip in stale:
                    self._events.pop(ip, None)
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                return max(1, int(self.window_seconds - (current - events[0])) + 1)
            events.append(current)
            return None


class _DemoEngine:
    def __init__(self, overlay_path: Path, items: list[dict[str, Any]]) -> None:
        self.overlay_path = overlay_path
        self.items = items
        self.orchestrator = AnswerOrchestrator("promoted", overlay_path=str(overlay_path))
        self.edges = _overlay_edges(items)
        self.edge_index = {
            _edge_key(edge["subject"], edge["predicate"], edge["object"]): edge
            for edge in self.edges
        }
        self.answer_lock = threading.RLock()

    def ask(self, question: str) -> tuple[Any, list[dict[str, str]], list[dict[str, str]]]:
        with self.answer_lock:
            answer = self.orchestrator.answer(
                question,
                web_search_enabled=False,
                community_context_enabled=False,
                cognitive_patterns_enabled=False,
            )
            used = self._used_edges(question, answer)
            context = self._context_edges(answer, used)
        return answer, used, context

    def _used_edges(self, question: str, answer: Any) -> list[dict[str, str]]:
        if answer.decision not in {"answer", "no"}:
            return []

        selected: list[dict[str, str]] = []
        route_intent = getattr(getattr(answer, "trace", None), "route_intent", "")

        if route_intent == "connection_path":
            plan = self.orchestrator._cross_page_planner.plan(analyze_cross_page(question))
            if plan.decision == "answer":
                for raw in plan.render_args.get("edges", []):
                    self._append_matching(selected, raw)
        else:
            analyzed = analyze_entity(question, index=self.orchestrator._surface_index)
            plan = self.orchestrator._entity_planner.plan(analyzed)
            if plan.decision in {"answer", "no"}:
                self._collect_entity_plan_edges(selected, plan, analyzed)

        return _dedupe_edges(selected)

    def _collect_entity_plan_edges(self, selected: list[dict[str, str]], plan: Any, analyzed: Any) -> None:
        args = plan.render_args or {}
        for key in ("relations", "common_pairs", "path", "edges"):
            for raw in args.get(key, []) or []:
                self._append_matching(selected, raw)

        synthesis = args.get("synthesis")
        if synthesis is not None:
            subject = str(getattr(synthesis, "subject", "") or getattr(analyzed, "subject", ""))
            definition = str(getattr(synthesis, "definition", "") or "")
            if subject and definition:
                self._append_key(selected, subject, "is_a", definition)
            for group in getattr(synthesis, "groups", ()) or ():
                predicate = str(getattr(group, "predicate", "") or "")
                objects = getattr(group, "objects", ()) or ()
                for obj in objects:
                    if getattr(group, "kind", "") == "inverse_relation":
                        self._append_key(selected, str(obj), predicate, subject)
                    else:
                        self._append_key(selected, subject, predicate, str(obj))

        if not selected:
            subject = str(args.get("subject") or getattr(analyzed, "subject", "") or "")
            definition = args.get("definition")
            if isinstance(definition, dict):
                definition = definition.get("definition")
            if subject and definition:
                self._append_key(selected, subject, "is_a", str(definition))

    def _append_matching(self, selected: list[dict[str, str]], raw: Any) -> None:
        if isinstance(raw, dict):
            subject = raw.get("subject")
            predicate = raw.get("predicate")
            obj = raw.get("object")
        elif is_dataclass(raw) or all(hasattr(raw, key) for key in ("subject", "predicate", "object")):
            subject = getattr(raw, "subject", None)
            predicate = getattr(raw, "predicate", None)
            obj = getattr(raw, "object", None)
        else:
            return
        if subject and predicate and obj:
            self._append_key(selected, str(subject), str(predicate), str(obj))

    def _append_key(self, selected: list[dict[str, str]], subject: str, predicate: str, obj: str) -> None:
        match = self.edge_index.get(_edge_key(subject, predicate, obj))
        if match is not None:
            selected.append(match)

    def _context_edges(
        self,
        answer: Any,
        used: list[dict[str, str]],
        limit: int = 10,
    ) -> list[dict[str, str]]:
        used_ids = {edge["evidence_id"] for edge in used}
        active_nodes = {
            value.casefold()
            for edge in used
            for value in (edge["subject"], edge["object"])
        }
        if not active_nodes:
            summary = getattr(getattr(answer, "trace", None), "context_summary", None) or {}
            active_nodes.update(str(item).casefold() for item in summary.get("matched_entities", []))
        if not active_nodes:
            return []

        context = [
            edge
            for edge in self.edges
            if edge["evidence_id"] not in used_ids
            and (
                edge["subject"].casefold() in active_nodes
                or edge["object"].casefold() in active_nodes
            )
        ]
        return context[:limit]


_engine: _DemoEngine | None = None
_engine_error: str | None = None
_engine_lock = threading.Lock()
_limiter = _SlidingWindowLimiter(int(os.environ.get("MICROWORLD_RATE_LIMIT_PER_MINUTE", "12")))


def _source_overlay_path() -> Path:
    configured = os.environ.get("MICROWORLD_DEMO_OVERLAY_SOURCE")
    if not configured:
        return _DEFAULT_PROMOTED_SOURCE
    path = Path(configured)
    return path if path.is_absolute() else (_REPO_ROOT / path).resolve()


def _build_public_overlay() -> tuple[Path, list[dict[str, Any]]]:
    source_path = _source_overlay_path()
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise RuntimeError("promoted overlay must be a JSON list")

    selected: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        kind = item.get("overlay_type")
        if str(item.get("risk") or "").casefold() == "high":
            continue
        if kind == "overlay_entity" and item.get("label") in _PUBLIC_NODES:
            selected.append(item)
        elif (
            kind == "overlay_definition"
            and item.get("subject") in _PUBLIC_NODES
            and str(item.get("stability") or "stable") in _PUBLIC_STABILITIES
        ):
            selected.append(item)
        elif (
            kind == "overlay_relation"
            and item.get("subject") in _PUBLIC_NODES
            and item.get("object") in _PUBLIC_NODES
            and str(item.get("stability") or "") in _PUBLIC_STABILITIES
        ):
            selected.append(item)

    if not selected:
        raise RuntimeError("bounded public overlay is empty")

    target = Path(tempfile.gettempdir()) / "microworld-public-demo-overlay.json"
    target.write_text(json.dumps(selected, ensure_ascii=False), encoding="utf-8")
    return target, selected


def _get_engine() -> _DemoEngine:
    global _engine, _engine_error
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is not None:
            return _engine
        try:
            overlay_path, items = _build_public_overlay()
            _engine = _DemoEngine(overlay_path, items)
            _engine_error = None
        except Exception as exc:
            _engine_error = exc.__class__.__name__
            raise
    return _engine


def _edge_key(subject: str, predicate: str, obj: str) -> tuple[str, str, str]:
    return (
        " ".join(subject.casefold().split()),
        " ".join(predicate.casefold().split()),
        " ".join(obj.casefold().split()),
    )


def _evidence_id(subject: str, predicate: str, obj: str, evidence_text: str) -> str:
    payload = "\0".join((subject, predicate, obj, evidence_text)).encode("utf-8")
    return "promoted:" + sha256(payload).hexdigest()[:16]


def _overlay_edges(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    edges: list[dict[str, str]] = []
    for item in items:
        kind = item.get("overlay_type")
        if kind == "overlay_relation":
            subject = str(item.get("subject") or "")
            predicate = str(item.get("predicate") or "")
            obj = str(item.get("object") or "")
        elif kind == "overlay_definition":
            subject = str(item.get("subject") or "")
            predicate = "is_a"
            obj = str(item.get("definition") or "")
        else:
            continue
        if not (subject and predicate and obj):
            continue
        evidence_text = str(item.get("evidence_text") or "")
        edges.append(
            {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "evidence_id": _evidence_id(subject, predicate, obj, evidence_text),
            }
        )
    return _dedupe_edges(edges)


def _dedupe_edges(edges: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    result: list[dict[str, str]] = []
    for edge in edges:
        key = edge["evidence_id"]
        if key not in seen:
            seen.add(key)
            result.append(edge)
    return result


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    return request.client.host if request.client else "unknown"


def _cors_origins() -> list[str]:
    configured = os.environ.get("MICROWORLD_CORS_ORIGINS", "")
    origins = [item.strip() for item in configured.split(",") if item.strip()]
    render_url = os.environ.get("RENDER_EXTERNAL_URL")
    if render_url:
        origins.append(render_url.rstrip("/"))
    render_hostname = os.environ.get("RENDER_EXTERNAL_HOSTNAME")
    if render_hostname:
        origins.append(f"https://{render_hostname.strip().strip('/')}")
    origins.extend(["http://127.0.0.1:8000", "http://localhost:8000"])
    return list(dict.fromkeys(origins))


app = FastAPI(
    title="MicroWorld public demo",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    max_age=600,
)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/", include_in_schema=False)
def landing() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/health")
def health(response: Response) -> dict[str, Any]:
    response.headers["Cache-Control"] = "no-store"
    return {
        "status": "ok",
        "engine_status": "ready" if _engine is not None else ("error" if _engine_error else "cold"),
        "overlay_scope": "bounded_promoted_public_subset",
        "overlay_items": len(_engine.items) if _engine is not None else None,
        "graph_edges": len(_engine.edges) if _engine is not None else None,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request, response: Response) -> AskResponse:
    retry_after = _limiter.check(_client_ip(request))
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail="rate_limit_exceeded",
            headers={"Retry-After": str(retry_after)},
        )

    question = " ".join(req.question.split())
    if len(question) < 2:
        raise HTTPException(status_code=422, detail="question_too_short")

    started = time.perf_counter()
    try:
        engine = _get_engine()
        answer, used, context = engine.ask(question)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail="demo_engine_unavailable") from exc

    response.headers["Cache-Control"] = "no-store"
    return AskResponse(
        answer=answer.answer_text,
        support_kind=answer.support_kind,
        edges_used=[GraphEdge(**edge) for edge in used],
        context_edges=[GraphEdge(**edge) for edge in context],
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
        decision=answer.decision,
    )
