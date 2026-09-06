"""
Lumen Prediction API — FastAPI wrapper over the existing pre-mortem engine.

Implements openapi/lumen-api.yaml (../frontend/openapi/lumen-api.yaml):
  POST /v1/predictions
  GET  /v1/diagnostics/model

Run: uvicorn api:app --port 8787 --reload
"""
from __future__ import annotations

import asyncio
import base64
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Literal, Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from src.agents import response_mapping as M
from src.ingestion.script_parser import ScriptParser, validate_screenplay
from src.utils.env_helper import load_env_file
from src.utils.event_bus import get_event_bus
from src.premortem.premortem_agent import PreMortemAgent
from src.quant.benchmarks import get_benchmark_movie, list_benchmark_movies
from src.quant.quant_agent import QuantAgent

load_env_file()  # PARALLEL_API_KEY / GEMINI_API_KEY, else the engine runs on mock search

MAX_MATERIAL_BYTES = 20 * 1024 * 1024

# ---------------------------------------------------------------- schemas

class Material(BaseModel):
    kind: Literal["script", "frame", "cast", "concept_art", "trailer", "other"]
    uri: str  # contract says format: uri, which admits data: alongside http(s):
    mimeType: str
    sha256: Optional[str] = None

    @field_validator("uri")
    @classmethod
    def _known_scheme(cls, v: str) -> str:
        if v.split(":", 1)[0].lower() not in ("http", "https", "data"):
            raise ValueError("uri scheme must be http, https or data")
        return v


class PredictionRequest(BaseModel):
    story: str = Field(min_length=20, max_length=2000)
    medium: Literal["Feature film", "TV series", "Limited series", "Documentary"]
    targetGeography: Literal[
        "North America", "United Kingdom", "Nordics", "Western Europe", "Global streaming"
    ]
    classifications: Dict[str, str] = Field(default_factory=dict)
    materials: List[Material] = Field(default_factory=list, max_length=30)


class AgentResult(BaseModel):
    agentId: Literal["story", "audience", "market", "visual"]
    status: Literal["complete", "partial", "failed"]
    finding: str
    score: Optional[int] = None
    sources: List[str] = Field(default_factory=list)


class Evidence(BaseModel):
    title: str
    statement: str
    sourceType: Literal["submission", "parallel_search", "model_signal"]
    sourceUrl: Optional[str] = None


class PredictionResponse(BaseModel):
    predictionId: uuid.UUID
    outcome: Literal["hit", "miss", "inconclusive"]
    score: int
    confidence: float
    summary: Optional[str] = None
    agents: List[AgentResult]
    evidence: List[Evidence]
    model: Optional[Dict[str, Optional[str]]] = None


class Evaluation(BaseModel):
    hitPrecision: Optional[float] = None
    missPrecision: Optional[float] = None
    calibrationError: Optional[float] = None


class ModelDiagnostics(BaseModel):
    trainerStatus: Literal["healthy", "running", "degraded", "failed"]
    activeModel: str
    lastTrainingRun: datetime
    featureCount: Optional[int] = None
    evaluation: Optional[Evaluation] = None


# ---------------------------------------------------------------- app

app = FastAPI(title="Lumen Prediction API", version="0.1.0")
# ponytail: wide-open CORS, tighten to the deployed frontend origin before prod
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)



@app.exception_handler(RequestValidationError)
async def _invalid_submission(request: Request, exc: RequestValidationError) -> JSONResponse:
    """The contract calls invalid submissions 400, not FastAPI's default 422."""
    return JSONResponse(status_code=400, content={"detail": jsonable_encoder(exc.errors())})


_agent = PreMortemAgent()
_quant = QuantAgent()


def _fetch_materials(materials: List[Material], workdir: Path):
    """Download materials. Returns (script_text, keyframes_dir)."""
    script_text = None
    images = workdir / "keyframes"
    images.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(materials):
        if m.kind not in ("script", "frame", "concept_art"):
            continue
        try:
            if m.uri.startswith("data:"):
                # ponytail: inline decode; the studio posts picked files as base64 data: URIs
                header, _, payload = m.uri.partition(",")
                if not header.endswith(";base64"):
                    raise ValueError("only base64 data: URIs are supported")
                body = base64.b64decode(payload, validate=True)
            else:
                r = requests.get(m.uri, timeout=20, stream=True)
                r.raise_for_status()
                body = r.raw.read(MAX_MATERIAL_BYTES + 1)
        except Exception:
            continue  # a broken material degrades the run, it doesn't fail it
        if len(body) > MAX_MATERIAL_BYTES:
            raise HTTPException(413, f"Material {m.uri[:80]} exceeds {MAX_MATERIAL_BYTES} bytes")
        if m.kind == "script":
            if script_text is None:
                script_text = body.decode("utf-8", errors="replace")
        else:
            suffix = {"image/png": ".png", "image/webp": ".webp"}.get(m.mimeType, ".jpg")
            (images / f"frame_{i:03d}{suffix}").write_bytes(body)
    return script_text, (str(images) if any(images.iterdir()) else None)


def run_engine(
    req: PredictionRequest,
    on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Fetch the materials, gate the screenplay, run the orchestrated pre-mortem."""
    with tempfile.TemporaryDirectory() as tmp:
        script_text, keyframes = _fetch_materials(req.materials, Path(tmp))
        title = req.classifications.get("title", "Untitled Submission")
        if script_text:
            metrics = ScriptParser().parse_script_text(script_text, title=title)
            reasons = validate_screenplay(metrics)
            if reasons:
                raise HTTPException(400, {"error": "not_a_screenplay", "reasons": reasons})
        # ponytail: the orchestrator re-parses the text; it takes text, not metrics
        return _agent.run_premortem(
            story=req.story,
            script_text=script_text,
            keyframes_dir=keyframes,
            genre=req.classifications.get("genre"),
            title=title,
            medium=req.medium,
            target_geography=req.targetGeography,
            on_event=on_event,
        )


def _to_response(report: Dict[str, Any], req: PredictionRequest) -> PredictionResponse:
    """Map the orchestrator's report onto the contract. No arithmetic beyond copying."""
    prediction = report["prediction"]
    cv_mae = prediction.get("cv_mae")
    claims = report.get("claims") or []
    degraded = bool(report.get("degraded"))
    has_screenplay = bool(report.get("has_screenplay"))
    # Without a screenplay `expected_rating` is the corpus median film in this
    # genre, so the genre prior is the only number that describes the submission.
    score = M.score_from_rating(
        prediction["expected_rating"] if has_screenplay
        else prediction["genre_baseline_rating"]
    )

    findings = M.slot_findings(report)
    agents = [
        AgentResult(
            agentId=slot,
            status=M.slot_status(slot, report),
            finding=findings[slot],
            score=score if slot == "story" else None,
            sources=sorted({c["source_url"] for c in M.claims_for(slot, report)})
            if slot in ("audience", "market") else [],
        )
        for slot in ("story", "audience", "market", "visual")
    ]

    evidence = [
        Evidence(title=title, statement=statement, sourceType="model_signal")
        for title, statement in M.model_signals(report)
    ]
    evidence += [
        Evidence(
            title=c.get("category", "Research claim"),
            statement=f"{c['claim']} Quoted: “{c['evidence']}”",
            sourceType="parallel_search",
            sourceUrl=c["source_url"],
        )
        for c in claims
    ]
    evidence.append(Evidence(
        title="Submission",
        statement=f"{req.medium} for {req.targetGeography}: {req.story[:200]}",
        sourceType="submission",
    ))

    model = _quant.champion_model
    return PredictionResponse(
        predictionId=uuid.uuid4(),
        outcome=M.outcome_for(score, cv_mae, has_screenplay, degraded),
        score=score,
        confidence=M.confidence_for(len(claims), cv_mae, has_screenplay, degraded),
        summary=M.summary_for(report),
        agents=agents,
        evidence=evidence,
        model={"modelId": getattr(model, "model_type", None), "version": "champion"},
    )



async def _to_thread(func, /, *args, **kwargs):
    """Compatible with Python 3.8+ (asyncio.to_thread was added in 3.9)."""
    if hasattr(asyncio, "to_thread"):
        return await asyncio.to_thread(func, *args, **kwargs)
    import functools
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))


@app.post("/v1/predictions", response_model=PredictionResponse)
async def create_prediction(req: PredictionRequest) -> PredictionResponse:
    # ponytail: engine is sync, so one thread per request; add a queue if it saturates
    report = await _to_thread(run_engine, req)
    return _to_response(report, req)


async def _execute_streaming(req: PredictionRequest, prediction_id: uuid.UUID) -> None:
    event_bus = get_event_bus()
    pid = str(prediction_id)

    def _bus_emit(event_type: str, data: Dict[str, Any]) -> None:
        event_bus.publish(pid, event_type, data)

    try:
        _bus_emit("stage", {"stage": "started", "label": "Initiating pre-mortem investigation...", "progress": 0.05})
        report = await _to_thread(run_engine, req, _bus_emit)
        response = _to_response(report, req)
        response.predictionId = prediction_id
        response_dict = jsonable_encoder(response)
        _bus_emit("complete", response_dict)
    except HTTPException as exc:
        _bus_emit("error", {"status_code": exc.status_code, "detail": exc.detail})
    except Exception as exc:
        _bus_emit("error", {"status_code": 500, "detail": str(exc)})


async def _stream_generator(req: PredictionRequest, prediction_id: uuid.UUID) -> AsyncGenerator[str, None]:
    event_bus = get_event_bus()
    pid = str(prediction_id)
    exec_task = asyncio.create_task(_execute_streaming(req, prediction_id))

    async for chunk in event_bus.sse_generator(pid, timeout_sec=120.0):
        yield chunk

    try:
        await exec_task
    except Exception:
        pass


@app.post("/v1/predictions/stream")
async def create_prediction_stream(req: PredictionRequest) -> StreamingResponse:
    """Execute pre-mortem and stream real-time events over SSE, ending with the complete report."""
    prediction_id = uuid.uuid4()
    return StreamingResponse(
        _stream_generator(req, prediction_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Prediction-ID": str(prediction_id),
        },
    )


@app.get("/v1/predictions/{prediction_id}/events")
async def stream_prediction_events(prediction_id: uuid.UUID) -> StreamingResponse:
    """Stream live events or replay event history for a prediction ID."""
    pid = str(prediction_id)
    event_bus = get_event_bus()
    return StreamingResponse(
        event_bus.sse_generator(pid, timeout_sec=120.0),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Prediction-ID": pid,
        },
    )


@app.get("/v1/diagnostics/model", response_model=ModelDiagnostics)
def model_diagnostics() -> ModelDiagnostics:
    model = _quant.champion_model
    path = Path(_quant.data_root) / "models" / "champion_model.joblib"
    if model is None:
        return ModelDiagnostics(
            trainerStatus="failed",
            activeModel="none",
            lastTrainingRun=datetime.now(timezone.utc),
        )
    m = model.metrics or {}
    return ModelDiagnostics(
        trainerStatus="healthy" if m.get("cv_r2", 0.0) >= 0.3 else "degraded",
        activeModel=model.model_type,
        lastTrainingRun=datetime.fromtimestamp(
            path.stat().st_mtime if path.exists() else 0, tz=timezone.utc
        ),
        featureCount=len(model.feature_names),
        evaluation=Evaluation(
            hitPrecision=m.get("cv_r2"),
            missPrecision=m.get("fit_r2"),
            calibrationError=m.get("cv_rmse"),
        ),
    )


@app.get("/v1/benchmarks")
def get_benchmarks(featured: bool = False) -> Dict[str, Any]:
    """List 100 genuine cinema benchmarks with pre-extracted craft features."""
    movies = list_benchmark_movies(featured_only=featured)
    return {"count": len(movies), "movies": movies}


@app.get("/v1/benchmarks/{slug}")
def get_benchmark_detail(slug: str) -> Dict[str, Any]:
    """Get full pre-mortem evaluation, feature attributions, and 15-point counterfactual sweep for a benchmark movie."""
    movie = get_benchmark_movie(slug)
    if not movie:
        raise HTTPException(status_code=404, detail=f"Benchmark movie '{slug}' not found.")
    return movie
