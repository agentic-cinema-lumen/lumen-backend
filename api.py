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
from typing import Any, Dict, List, Literal, Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.agents import response_mapping as M
from src.ingestion.script_parser import ScriptParser, validate_screenplay
from src.utils.env_helper import load_env_file
from src.premortem.premortem_agent import PreMortemAgent
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


def run_engine(req: PredictionRequest) -> Dict[str, Any]:
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
        )


def _to_response(report: Dict[str, Any], req: PredictionRequest) -> PredictionResponse:
    """Map the orchestrator's report onto the contract. No arithmetic beyond copying."""
    prediction = report["prediction"]
    cv_mae = prediction.get("cv_mae")
    score = M.score_from_rating(prediction["expected_rating"])
    claims = report.get("claims") or []
    degraded = bool(report.get("degraded"))

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
        outcome=M.outcome_for(score, cv_mae),
        score=score,
        confidence=M.confidence_for(
            len(claims), cv_mae, report.get("has_screenplay", False), degraded
        ),
        summary=M.summary_for(report),
        agents=agents,
        evidence=evidence,
        model={"modelId": getattr(model, "model_type", None), "version": "champion"},
    )



@app.post("/v1/predictions", response_model=PredictionResponse)
async def create_prediction(req: PredictionRequest) -> PredictionResponse:
    # ponytail: engine is sync, so one thread per request; add a queue if it saturates
    report = await asyncio.to_thread(run_engine, req)
    return _to_response(report, req)


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
