"""Bite.ai — FastAPI Backend (Ollama + Rule-based Hybrid)

Run locally:
    python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Then access from phone on same WiFi:
    http://<YOUR-PC-IP>:8000/app/
"""

import os
import re
import time
import uuid
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.models.schemas import QueryRequest, QueryResponse, Verdict
from app.models.session import ConversationSession
from app.services.intent import RulesLayer, KeywordRouter, FollowUpResolver
from app.services.barcode import OFFClient
from app.services.resolver import FoodResolver
from app.services.analyzer import HybridAnalyzer, RuleBasedAnalyzer
from app.services.response import ResponseFormatter
from app.services.response_agent import ResponseAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== SERVICES (initialized at startup) =====
rules_layer = RulesLayer()
keyword_router = KeywordRouter()
follow_up_resolver = FollowUpResolver()
off_client = OFFClient()
food_resolver: FoodResolver = None
analyzer: HybridAnalyzer = None
response_agent: ResponseAgent = None
response_formatter = ResponseFormatter()

# ===== SESSION STORE =====
sessions: dict[str, ConversationSession] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown."""
    global food_resolver, analyzer, response_agent

    logger.info("Starting Bite.ai...")

    # Project root (for resolving data paths)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load INDB food database
    excel_path = os.path.join(project_root, "data", "Anuvaad_INDB_2024.11.xlsx")
    food_resolver = FoodResolver(excel_path)
    logger.info(f"Food resolver: {food_resolver.get_food_count()} foods loaded")

    # Agent 1: Hybrid analyzer (rule-based primary + Ollama SLM fallback)
    analyzer = HybridAnalyzer(slm_model="qwen:1.8b")
    logger.info("Agent 1 (Analysis): Rule-based + Ollama SLM fallback ready")

    # Agent 2: Response generation SLM via Ollama
    response_agent = ResponseAgent(model="qwen:1.8b", timeout=8.0)
    if await response_agent.is_available():
        logger.info("Agent 2 (Response): Ollama qwen:1.8b ready")
    else:
        logger.warning("Agent 2 (Response): Ollama not available — using templates")

    # Frontend static files (absolute path)
    frontend_dir = os.path.join(project_root, "frontend")
    try:
        app.mount("/app", StaticFiles(directory=frontend_dir, html=True), name="frontend")
        logger.info(f"Frontend mounted from {frontend_dir}")
    except Exception:
        logger.warning("Frontend directory not found — API-only mode")

    yield

    # Cleanup
    await off_client.close()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Bite.ai",
    version="2.0.0",
    description="Voice food assistant — Hybrid Rule-based + SLM",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== GLOBAL ERROR HANDLER =====
@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=200,
        content={
            "session_id": "error",
            "intent": "error",
            "product_name": None,
            "verdict": None,
            "response_text": response_formatter.error_response(),
            "latency_ms": 0,
        },
    )


# ===== ROUTES =====
@app.get("/health")
async def health():
    ollama_ok = await response_agent.is_available() if response_agent else False
    return {
        "status": "ok",
        "foods_loaded": food_resolver.get_food_count() if food_resolver else 0,
        "ollama": ollama_ok,
        "agents": {
            "analysis": "rule-based + SLM fallback",
            "response": "ollama" if ollama_ok else "templates",
        },
    }


@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    start = time.time()

    # 1. Session
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = ConversationSession(session_id=session_id)
    session = sessions[session_id]

    if request.user_profile:
        session.user_profile = request.user_profile.model_dump()

    session.add_turn("user", request.text)

    # 2. Intent classification
    intent, meta = rules_layer.classify(request.text)
    if intent is None:
        intent = keyword_router.classify(request.text)
        meta = {"raw_text": request.text}

    # 3. Follow-up check
    if follow_up_resolver.is_follow_up(request.text, session):
        intent = "follow_up"

    # 4. Route
    product = None
    verdict = None
    response_text = ""

    if intent == "barcode":
        barcode = meta.get("barcode", "")
        product = await off_client.get_product(barcode)
        if product:
            verdict = await analyzer.analyze(product, session.user_profile)
            session.set_product(product, verdict)
            response_text = await _generate_response(product, verdict, session.user_profile)
        else:
            response_text = f"I couldn't find barcode {barcode}. Double-check and try again!"

    elif intent == "food_query":
        raw = meta.get("raw_text", request.text)
        food_name = _extract_food_name(raw)
        product = await food_resolver.resolve(food_name)

        if product:
            verdict = await analyzer.analyze(product, session.user_profile)
            session.set_product(product, verdict)
            response_text = await _generate_response(product, verdict, session.user_profile)
        else:
            response_text = response_formatter.no_product_found(food_name)

    elif intent == "follow_up":
        if session.has_product_context():
            verdict = await analyzer.analyze(session.current_product, session.user_profile)
            response_text = await _generate_response(
                session.current_product, verdict, session.user_profile
            )
        else:
            response_text = (
                "I don't have a food in context. "
                "Try asking about a food first!"
            )

    elif intent == "casual":
        subtype = meta.get("subtype", "generic") if meta else "generic"
        response_text = response_formatter.casual_response(subtype)

    elif intent == "diet_advice":
        response_text = (
            "I can help you analyze specific foods! "
            "Try 'is idli good for weight loss?' or scan a barcode."
        )

    else:
        # Treat unknown as food query
        food_name = _extract_food_name(request.text)
        product = await food_resolver.resolve(food_name)
        if product:
            verdict = await analyzer.analyze(product, session.user_profile)
            session.set_product(product, verdict)
            response_text = await _generate_response(product, verdict, session.user_profile)
        else:
            response_text = response_formatter.no_product_found(request.text)
        intent = "food_query"

    session.add_turn("assistant", response_text)
    latency = (time.time() - start) * 1000

    return QueryResponse(
        session_id=session_id,
        intent=intent,
        product_name=product.get("product_name") if product else None,
        verdict=Verdict(**{k: v for k, v in verdict.items() if k != "source"}) if verdict else None,
        response_text=response_text,
        latency_ms=round(latency, 1),
    )

async def _generate_response(product: dict, verdict: dict, user_profile: dict) -> str:
    """Templates primary, SLM as fallback enhancement."""
    from app.services.response_agent import generate_template_response

    product_name = product.get("product_name", "Unknown")

    # Try SLM first — if it gives a good response, use it
    if response_agent:
        try:
            slm_resp = await response_agent._try_slm(product_name, verdict, user_profile)
            if slm_resp:
                return slm_resp
        except Exception:
            pass

    # Fallback: template (instant, always reliable)
    return generate_template_response(product_name, verdict, user_profile)


def _extract_food_name(text: str) -> str:
    result = text.strip()

    # Strip leading question prefixes
    prefix_patterns = [
        r"^(is|are|was)\s+",
        r"^(can|should|do)\s+(i|we|diabetics|people)\s+(eat|have|consume)\s+",
        r"^(tell\s+me\s+about|what\s+about|how\s+about|how\s+healthy\s+is)\s+",
        r"^(what\s+is|what\s+are|analyze|check)\s+",
        r"^(i\s+want\s+to\s+eat|i\s+ate|i\s+had)\s+",
    ]
    for p in prefix_patterns:
        result = re.sub(p, "", result, flags=re.IGNORECASE).strip()

    # Strip trailing qualifiers
    suffix_patterns = [
        r"\s*(healthy|good|bad|safe|ok|okay|fine|harmful|unhealthy)\s*\??$",
        r"\s*(for\s+me|for\s+health|for\s+weight\s+loss|for\s+diabetes|for\s+diabetics)\s*\??$",
        r"\s*(good\s+for\s+\w+|bad\s+for\s+\w+|safe\s+for\s+\w+)\s*\??$",
        r"\s*\?+$",
    ]
    for p in suffix_patterns:
        result = re.sub(p, "", result, flags=re.IGNORECASE).strip()

    # Remove stray articles
    result = re.sub(r"^(a|an|the|some)\s+", "", result, flags=re.IGNORECASE).strip()

    return result or text

