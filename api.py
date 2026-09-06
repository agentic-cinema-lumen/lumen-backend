"""
Lumen Prediction API — FastAPI wrapper over the existing pre-mortem engine.

Implements openapi/lumen-api.yaml (../frontend/openapi/lumen-api.yaml):
  POST /v1/predictions
  GET  /v1/diagnostics/model

Run: uvicorn api:app --port 8787 --reload
"""
from __future__ import annotations

import asyncio
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl

from src.utils.env_helper import load_env_file
from src.premortem.premortem_agent import PreMortemAgent
from src.quant.quant_agent import QuantAgent

load_env_file()  # PARALLEL_API_KEY / GEMINI_API_KEY, else the engine runs on mock search

MAX_MATERIAL_BYTES = 20 * 1024 * 1024

# ---------------------------------------------------------------- schemas

class Material(BaseModel):
    kind: Literal["script", "frame", "cast", "concept_art", "trailer", "other"]
    uri: HttpUrl
    mimeType: str
    sha256: Optional[str] = None


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
            r = requests.get(str(m.uri), timeout=20, stream=True)
            r.raise_for_status()
            body = r.raw.read(MAX_MATERIAL_BYTES + 1)
        except Exception:
            continue  # a broken material degrades the run, it doesn't fail it
        if len(body) > MAX_MATERIAL_BYTES:
            raise HTTPException(413, f"Material {m.uri} exceeds {MAX_MATERIAL_BYTES} bytes")
        if m.kind == "script":
            if script_text is None:
                script_text = body.decode("utf-8", errors="replace")
        else:
            suffix = {"image/png": ".png", "image/webp": ".webp"}.get(m.mimeType, ".jpg")
            (images / f"frame_{i:03d}{suffix}").write_bytes(body)
    return script_text, (str(images) if any(images.iterdir()) else None)


def run_engine(req: PredictionRequest) -> Dict[str, Any]:
    """Adapter boundary: swap this body for the ADK agent, keep the report shape."""
    with tempfile.TemporaryDirectory() as tmp:
        script_text, keyframes = _fetch_materials(req.materials, Path(tmp))
        title = req.classifications.get("title", "Untitled Submission")
        if script_text:
            return _agent.run_script_premortem(
                script_path_or_text=script_text, keyframes_path_or_dir=keyframes, title=title
            )
        return _agent.run_premise_premortem(
            logline=req.story,
            keyframes_path_or_dir=keyframes,
            title=title,
            genre=req.classifications.get("genre"),
        )


def _to_response(report: Dict[str, Any], req: PredictionRequest) -> PredictionResponse:
    if report["mode"] == "script_premortem":
        rating = report["projected_baseline_rating"]
        baseline = report["historical_show_mean"]
        cm = report["craft_metrics"]
        story_finding = (
            f"Runtime {cm['estimated_runtime_min']} min across {cm['total_scenes']} scenes, "
            f"{cm['words_per_minute']} WPM, climax acceleration {cm['climax_acceleration']}x."
        )
        visual_finding = report["vision_summary"]
        market = report["audience_trope_intelligence"]
        flaws = report["detected_craft_flaws"]
        audience_finding = "; ".join(f["finding"] for f in flaws) or "No craft flaws above threshold."
        notes = report["recommendations"]
    else:
        rating = report["projected_rating_potential"]["median_target"]
        baseline = report["empirical_genre_baseline"]
        band = report["projected_rating_potential"]
        story_finding = (
            f"{report['genre']} premise, tone '{report['detected_tone']}'. "
            f"Rating band {band['floor']}-{band['ceiling']}."
        )
        coh = report["style_premise_cohesion"]
        visual_finding = f"{coh['rating']}: {coh['evaluation']}"
        market = report["audience_fatigue_radar"]
        audience_finding = report["greenlight_verdict"]
        notes = report["make_or_break_dependencies"]

    score = max(0, min(100, round(rating * 10)))
    outcome = "hit" if score >= 70 else "miss" if score <= 50 else "inconclusive"

    claims = market.get("audience_consensus_claims", [])
    queries = market.get("queries_executed", [])
    query_count = len(queries) if isinstance(queries, list) else int(queries or 0)
    sources = [c["source_url"] for c in claims if c.get("source_url")]
    # confidence: distance from the inconclusive band, tempered by evidence volume
    confidence = round(min(0.95, 0.4 + abs(score - 60) / 100 + 0.05 * len(claims)), 2)

    agents = [
        AgentResult(agentId="story", status="complete", finding=story_finding, score=score),
        AgentResult(agentId="audience", status="complete", finding=audience_finding),
        AgentResult(
            agentId="market",
            status="complete" if claims else "partial",
            finding=f"{query_count} Parallel Search queries; {len(claims)} audience claims extracted.",
            sources=sources,
        ),
        AgentResult(agentId="visual", status="complete", finding=visual_finding),
    ]

    evidence = [
        Evidence(
            title="Submission",
            statement=f"{req.medium} for {req.targetGeography}: {req.story[:200]}",
            sourceType="submission",
        ),
        Evidence(
            title="Model signal",
            statement=f"Projected rating {rating}/10 against baseline {baseline}/10.",
            sourceType="model_signal",
        ),
    ] + [
        Evidence(
            title=c.get("category", "Audience claim"),
            statement=c.get("claim", ""),
            sourceType="parallel_search",
            sourceUrl=c.get("source_url"),
        )
        for c in claims
    ]
    for n in notes[:3]:
        evidence.append(Evidence(title="Recommendation", statement=n, sourceType="model_signal"))

    model = _quant.champion_model
    return PredictionResponse(
        predictionId=uuid.uuid4(),
        outcome=outcome,
        score=score,
        confidence=confidence,
        summary=f"{outcome.upper()} at {score}/100 - projected {rating}/10 vs baseline {baseline}/10.",
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
