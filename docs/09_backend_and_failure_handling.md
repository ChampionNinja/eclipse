# STEP 12: Backend Implementation (FastAPI)

## Project Structure

```
nutriassist2/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app + routes
│   ├── config.py             # Settings
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py        # Pydantic models (UnifiedProduct, etc.)
│   │   └── session.py        # ConversationSession
│   ├── services/
│   │   ├── __init__.py
│   │   ├── stt.py            # Whisper STT
│   │   ├── tts.py            # edge-tts
│   │   ├── intent.py         # RulesLayer + KeywordRouter
│   │   ├── barcode.py        # OFFClient
│   │   ├── resolver.py       # FoodResolver
│   │   ├── analyzer.py       # BiteAIInference (SLM)
│   │   ├── smart_router.py   # SLM skip logic
│   │   └── response.py       # Response formatter
│   └── data/
│       ├── local_foods.json  # Curated food database
│       └── templates.json    # Response templates
├── data/
│   └── training_data.jsonl   # Fine-tuning dataset
├── scripts/
│   ├── train.py              # Fine-tuning script
│   ├── generate_dataset.py   # Dataset generation
│   └── convert_gguf.py       # Model conversion
├── requirements.txt
└── docs/                     # These blueprint files
```

## Main Application (app/main.py)

```python
import asyncio
import uuid
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from app.services.intent import RulesLayer, KeywordRouter
from app.services.barcode import OFFClient
from app.services.resolver import FoodResolver
from app.services.analyzer import BiteAIInference
from app.services.smart_router import SmartRouter
from app.services.response import ResponseFormatter
from app.models.session import ConversationSession

app = FastAPI(title="Bite.ai", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize Services ---
rules_layer = RulesLayer()
keyword_router = KeywordRouter()
off_client = OFFClient()
food_resolver = FoodResolver()
analyzer = BiteAIInference(
    base_model="Qwen/Qwen2.5-0.5B-Instruct",
    adapter_path="./nutriassist-model-final"
)
smart_router = SmartRouter()
response_formatter = ResponseFormatter()

# --- Session Store (in-memory, use Redis for production) ---
sessions: dict[str, ConversationSession] = {}


# --- Request/Response Models ---
class QueryRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    user_profile: Optional[dict] = None

class QueryResponse(BaseModel):
    session_id: str
    intent: str
    product_name: Optional[str] = None
    verdict: Optional[dict] = None
    response_text: str
    latency_ms: float


# --- Main Endpoint ---
@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    import time
    start = time.time()
    
    # 1. Get or create session
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = ConversationSession(session_id=session_id)
    session = sessions[session_id]
    
    # Update user profile if provided
    if request.user_profile:
        session.user_profile.update(request.user_profile)
    
    # 2. Add user turn
    session.add_turn("user", request.text)
    
    # 3. Intent Classification
    intent, meta = rules_layer.classify(request.text)
    if intent is None:
        intent = keyword_router.classify(request.text)
        meta = {"raw_text": request.text}
    
    # 4. Check for follow-up
    from app.services.intent import FollowUpResolver
    follow_up_resolver = FollowUpResolver()
    if follow_up_resolver.is_follow_up(request.text, session):
        intent = "follow_up"
    
    # 5. Route by intent
    product = None
    verdict = None
    response_text = ""
    
    if intent == "barcode":
        barcode = meta.get("barcode", "")
        product = await off_client.get_product(barcode)
        if product:
            product["user_profile"] = session.user_profile
            
            # Check if SLM can be skipped
            can_skip, skip_verdict = smart_router.can_skip_slm(
                product, session.user_profile
            )
            if can_skip:
                verdict = skip_verdict
            else:
                verdict = analyzer.analyze(product, session.user_profile)
            
            session.set_product(product, verdict)
            response_text = response_formatter.format_verdict(
                product["product_name"], verdict
            )
        else:
            response_text = f"Sorry, I couldn't find a product with barcode {barcode}."
    
    elif intent == "food_query":
        raw_text = meta.get("raw_text", request.text)
        food_name = _extract_food_name(raw_text)
        product = await food_resolver.resolve(food_name)
        
        if product:
            product["user_profile"] = session.user_profile
            
            can_skip, skip_verdict = smart_router.can_skip_slm(
                product, session.user_profile
            )
            if can_skip:
                verdict = skip_verdict
            else:
                verdict = analyzer.analyze(product, session.user_profile)
            
            session.set_product(product, verdict)
            response_text = response_formatter.format_verdict(
                product["product_name"], verdict
            )
        else:
            response_text = f"I couldn't find nutrition data for that food."
    
    elif intent == "follow_up":
        if session.has_product_context():
            # Reuse current product, answer the specific question
            verdict = analyzer.analyze(
                session.current_product, session.user_profile
            )
            response_text = response_formatter.format_followup(
                session.current_product["product_name"],
                request.text,
                verdict
            )
        else:
            response_text = "I don't have a product in context. Try scanning or asking about a food first."
    
    elif intent == "casual":
        response_text = response_formatter.casual_response(
            meta.get("subtype", "generic")
        )
    
    elif intent == "diet_advice":
        response_text = ("I can help you analyze specific foods! "
                        "Try asking 'is oatmeal good for weight loss?' "
                        "or scan a barcode.")
    
    # 6. Add assistant turn
    session.add_turn("assistant", response_text)
    
    latency = (time.time() - start) * 1000
    
    return QueryResponse(
        session_id=session_id,
        intent=intent,
        product_name=product["product_name"] if product else None,
        verdict=verdict,
        response_text=response_text,
        latency_ms=round(latency, 1),
    )


def _extract_food_name(text: str) -> str:
    """Simple food name extraction from query text."""
    import re
    # Remove common question patterns to isolate the food name
    patterns = [
        r'^(is|are|can i eat|should i eat|tell me about|how healthy is|what about)\s+',
        r'\s*(healthy|good|bad|safe|ok)\s*\??$',
        r'\s*(for me|for health|for weight loss|for diabetes)\s*\??$',
    ]
    result = text.lower().strip()
    for pattern in patterns:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE).strip()
    return result or text


# --- Health Check ---
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": analyzer.model is not None}


# --- Startup ---
@app.on_event("startup")
async def startup():
    print("Bite.ai API starting...")
    # Pre-warm the model with a dummy inference
    analyzer.analyze(
        {"product_name": "test", "nutriments": {}, "source_type": "inferred",
         "data_confidence": "low", "ingredients": ""},
        {"allergies": [], "conditions": []}
    )
    print("Model warmed up. Ready to serve.")
```

## Response Formatter (app/services/response.py)

```python
class ResponseFormatter:
    VERDICT_TEMPLATES = {
        "eat": "{name}: Go ahead! {reason}. Want to know more?",
        "avoid": "{name}: I'd skip this one. {reason}. Ask me about alternatives!",
        "sometimes": "{name}: It's okay in moderation. {reason}. Want details?",
    }
    
    CASUAL_RESPONSES = {
        "greeting": "Hey! I'm Bite.ai. Tell me a food or scan a barcode, and I'll analyze it for you!",
        "thanks": "You're welcome! Ask me about any other food anytime.",
        "generic": "I help you make smart food choices! Try asking about a food or scanning a barcode.",
    }
    
    def format_verdict(self, product_name: str, verdict: dict) -> str:
        template = self.VERDICT_TEMPLATES.get(verdict["verdict"], self.VERDICT_TEMPLATES["sometimes"])
        return template.format(name=product_name, reason=verdict["reason"])
    
    def format_followup(self, product_name: str, question: str, verdict: dict) -> str:
        return f"About {product_name}: {verdict['reason']}."
    
    def casual_response(self, subtype: str) -> str:
        return self.CASUAL_RESPONSES.get(subtype, self.CASUAL_RESPONSES["generic"])
```

## Requirements (requirements.txt)

```
fastapi==0.115.0
uvicorn==0.30.0
httpx==0.27.0
pydantic==2.9.0
torch==2.4.0
transformers==4.45.0
peft==0.13.0
trl==0.11.0
bitsandbytes==0.44.0
datasets==3.0.0
rapidfuzz==3.9.0
edge-tts==6.1.0
llama-cpp-python==0.3.0
```

---

# STEP 13: Failure Handling

## Failure Scenarios & Responses

### 1. No Product Found (Barcode)

```python
# In barcode flow
if product is None:
    responses = [
        f"I couldn't find barcode {barcode} in my database. Try another product!",
        f"That barcode ({barcode}) didn't match any product. Can you double-check?",
    ]
    return random.choice(responses)
```

### 2. No Product Found (Food Query)

```python
# In food resolver flow
if product["data_confidence"] == "low" and not product["nutriments"]:
    return (f"I don't have detailed data for '{food_name}'. "
            f"I can still give a general assessment, or try being more specific!")
```

### 3. Ambiguous Query

```python
# Multiple fuzzy matches with close scores
def handle_ambiguity(matches: list) -> str:
    if len(matches) > 1 and matches[0][1] - matches[1][1] < 5:
        options = ", ".join(m[0].title() for m in matches[:3])
        return f"I found multiple matches: {options}. Which one did you mean?"
    return None
```

### 4. Missing Nutrition Data

```python
# Product found but nutriments mostly empty
def assess_data_quality(nutriments: dict) -> str:
    filled = sum(1 for v in nutriments.values() if v is not None)
    total = len(nutriments)
    
    if filled == 0:
        return "no_data"      # → use general knowledge only
    elif filled / total < 0.5:
        return "partial_data"  # → analyze with caveats
    else:
        return "sufficient"    # → full analysis

# Response with caveat
if quality == "partial_data":
    response = f"Based on limited data: {verdict['reason']}. " \
               f"For a complete analysis, try scanning the barcode."
```

### 5. API Timeout / Network Error

```python
# OFF API unreachable
async def get_product_with_fallback(barcode: str) -> dict:
    try:
        return await off_client.get_product(barcode)
    except Exception:
        # Check local cache
        cached = barcode_cache.get(barcode)
        if cached:
            return cached
        return None

# Response
if product is None:
    return ("I'm having trouble connecting to the food database right now. "
            "Try again in a moment, or ask about a common food by name!")
```

### 6. SLM Output Parse Failure

```python
# Already handled in VerdictParser with cascading fallbacks:
# 1. JSON parse → 2. Regex extract → 3. Default safe response
# The system NEVER crashes on bad model output

DEFAULT_VERDICT = {
    "verdict": "sometimes",
    "reason": "Unable to complete analysis, please try again",
    "confidence": 0.3
}
```

### 7. Full Error Handling Middleware

```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    traceback.print_exc()
    
    return JSONResponse(
        status_code=200,  # Return 200 to keep voice flow smooth
        content={
            "session_id": "error",
            "intent": "error",
            "product_name": None,
            "verdict": None,
            "response_text": "Something went wrong on my end. Could you try that again?",
            "latency_ms": 0,
        }
    )
```

## Error Recovery Matrix

| Error | Detection | Recovery | User Message |
|-------|-----------|----------|--------------|
| Barcode not found | API returns status 0 | Suggest re-scan | "Barcode not found, try again" |
| API timeout | httpx.TimeoutException | Check cache → fallback | "Connection issue, trying..." |
| Ambiguous food | Multiple fuzzy matches | Present options | "Did you mean X or Y?" |
| No nutrition data | Empty nutriments dict | General assessment + caveat | "Limited data: ..." |
| SLM parse failure | JSON decode fails | Regex → default | "Let me try analyzing again" |
| Unknown intent | All classifiers fail | Default to casual | "Could you rephrase that?" |
| Session expired | Session not in store | Create new session | Seamless (invisible) |
