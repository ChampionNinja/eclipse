"""NutriAssist — FastAPI Backend

Run:
    uvicorn app.main:app --reload --port 8000
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
from app.services.analyzer import SLMAnalyzer, RuleBasedAnalyzer
from app.services.response import ResponseFormatter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== SERVICES (initialized at startup) =====
rules_layer = RulesLayer()
keyword_router = KeywordRouter()
follow_up_resolver = FollowUpResolver()
off_client = OFFClient()
food_resolver: FoodResolver = None  # lazy init
analyzer: SLMAnalyzer = None
response_formatter = ResponseFormatter()

# ===== SESSION STORE =====
sessions: dict[str, ConversationSession] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown."""
    global food_resolver, analyzer

    logger.info("Starting NutriAssist...")

    # Load INDB food database
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    excel_path = os.path.join(project_root, "data", "Anuvaad_INDB_2024.11.xlsx")
    food_resolver = FoodResolver(excel_path)
    logger.info(f"Food resolver: {food_resolver.get_food_count()} foods loaded")

    # Load SLM analyzer (falls back to rules if model not found)
    adapter_path = os.path.join(
        project_root, "nutriassist-adapter", "content", "nutriassist-model"
    )
    if os.path.isdir(adapter_path):
        analyzer = SLMAnalyzer(model_path=adapter_path)
        logger.info(f"Analyzer ready (SLM mode: {adapter_path})")
    else:
        analyzer = SLMAnalyzer(model_path=None)
        logger.info("Analyzer ready (rule-based fallback — adapter not found)")

    yield

    # Cleanup
    await off_client.close()
    logger.info("Shutdown complete")


app = FastAPI(
    title="NutriAssist",
    version="1.0.0",
    description="Voice food assistant API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
try:
    app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")
except Exception:
    logger.warning("Frontend directory not found — API-only mode")


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
    return {
        "status": "ok",
        "foods_loaded": food_resolver.get_food_count() if food_resolver else 0,
        "model_loaded": analyzer.model is not None if analyzer else False,
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
            verdict = analyzer.analyze(product, session.user_profile)
            session.set_product(product, verdict)
            response_text = response_formatter.format_verdict(
                product["product_name"], verdict
            )
        else:
            response_text = f"I couldn't find barcode {barcode}. Double-check and try again!"

    elif intent == "food_query":
        raw = meta.get("raw_text", request.text)
        food_name = _extract_food_name(raw)
        product = await food_resolver.resolve(food_name)

        if product:
            verdict = analyzer.analyze(product, session.user_profile)
            session.set_product(product, verdict)
            response_text = response_formatter.format_verdict(
                product["product_name"], verdict
            )
        else:
            response_text = response_formatter.no_product_found(food_name)

    elif intent == "follow_up":
        if session.has_product_context():
            verdict = analyzer.analyze(session.current_product, session.user_profile)
            response_text = response_formatter.format_followup(
                session.current_product["product_name"], request.text, verdict
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
            verdict = analyzer.analyze(product, session.user_profile)
            session.set_product(product, verdict)
            response_text = response_formatter.format_verdict(
                product["product_name"], verdict
            )
        else:
            response_text = response_formatter.no_product_found(request.text)
        intent = "food_query"

    session.add_turn("assistant", response_text)
    latency = (time.time() - start) * 1000

    return QueryResponse(
        session_id=session_id,
        intent=intent,
        product_name=product.get("product_name") if product else None,
        verdict=Verdict(**verdict) if verdict else None,
        response_text=response_text,
        latency_ms=round(latency, 1),
    )


def _extract_food_name(text: str) -> str:
    patterns = [
        r"^(is|are|can i eat|should i eat|tell me about|how healthy is|what about)\s+",
        r"\s*(healthy|good|bad|safe|ok)\s*\??$",
        r"\s*(for me|for health|for weight loss|for diabetes)\s*\??$",
    ]
    result = text.strip()
    for pattern in patterns:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE).strip()
    return result or text
