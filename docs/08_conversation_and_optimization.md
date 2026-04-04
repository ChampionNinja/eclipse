# STEP 10: Conversation Loop Design

## Session Schema

```python
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

@dataclass
class ConversationSession:
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    
    # Current product context
    current_product: Optional[dict] = None
    current_verdict: Optional[dict] = None
    
    # History (last 5 products)
    product_history: list = field(default_factory=list)
    
    # Conversation turns (last 10)
    conversation_history: list = field(default_factory=list)
    
    # User profile (persisted per user)
    user_profile: dict = field(default_factory=lambda: {
        "allergies": [],
        "conditions": [],
        "diet_type": None,
        "goal": None,
    })
    
    def set_product(self, product: dict, verdict: dict):
        """Update current product context."""
        if self.current_product:
            self.product_history.append(self.current_product)
            if len(self.product_history) > 5:
                self.product_history.pop(0)
        self.current_product = product
        self.current_verdict = verdict
    
    def add_turn(self, role: str, content: str):
        """Add conversation turn."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.conversation_history) > 10:
            self.conversation_history.pop(0)
    
    def has_product_context(self) -> bool:
        return self.current_product is not None
```

## Follow-Up Query Resolution

```python
class FollowUpResolver:
    """Resolves follow-up queries using session context."""
    
    # Patterns that indicate follow-up (references to previous product)
    FOLLOW_UP_PATTERNS = [
        "it", "this", "that", "the product", "this product",
        "how much", "what about", "does it", "is it", "can i",
        "more about", "tell me more",
    ]
    
    def is_follow_up(self, text: str, session: ConversationSession) -> bool:
        """Check if query references existing product context."""
        if not session.has_product_context():
            return False
        text_lower = text.lower()
        return any(p in text_lower for p in self.FOLLOW_UP_PATTERNS)
    
    def resolve(self, query: str, session: ConversationSession) -> dict:
        """Build context-enriched query for the SLM."""
        product = session.current_product
        verdict = session.current_verdict
        
        return {
            "query": query,
            "product_context": product,
            "previous_verdict": verdict,
            "user_profile": session.user_profile,
        }
```

## How Context is Reused

```
Turn 1: "Scan 8901234567890"
  → Barcode flow → product stored in session
  → SLM: "Maggi Noodles: avoid. High sodium and MSG."

Turn 2: "Why should I avoid it?"
  → FollowUpResolver detects "it" + session has product
  → Reuses session.current_product (no re-fetch)
  → SLM gets full product context + the question
  → Response: "Maggi has 2.9g salt per 100g and contains MSG, risky for blood pressure."

Turn 3: "What if I only eat half a packet?"
  → Still a follow-up on same product
  → SLM gets product + portion context
  → Response: "Half packet reduces salt to ~1.4g, still moderate. Limit to occasionally."

Turn 4: "Is brown rice better?"
  → New food query (not a follow-up)
  → Food Resolver: "brown rice" → local DB
  → New product replaces session.current_product
  → Old product pushed to product_history
```

## Product Memory Update Flow

```
New product identified
  → session.product_history.append(session.current_product)
  → session.current_product = new_product
  → session.current_verdict = new_verdict

User asks "compare with the previous one"
  → Pull from session.product_history[-1]
  → Compare with session.current_product
```

---

# STEP 11: Latency Optimization

## 1. Caching Strategies

```python
from functools import lru_cache
from collections import OrderedDict
import time

class TTLCache:
    """Time-based LRU cache for API responses."""
    
    def __init__(self, maxsize: int = 500, ttl: int = 3600):
        self._cache = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl
    
    def get(self, key: str):
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry['time'] < self._ttl:
                self._cache.move_to_end(key)
                return entry['value']
            else:
                del self._cache[key]
        return None
    
    def set(self, key: str, value):
        if len(self._cache) >= self._maxsize:
            self._cache.popitem(last=False)
        self._cache[key] = {'value': value, 'time': time.time()}

# Cache layers:
barcode_cache = TTLCache(maxsize=1000, ttl=3600)    # 1 hour
food_cache = TTLCache(maxsize=500, ttl=7200)         # 2 hours
verdict_cache = TTLCache(maxsize=500, ttl=1800)      # 30 min
```

## 2. Skipping Unnecessary Model Calls

```python
class SmartRouter:
    """Skip SLM when deterministic rules suffice."""
    
    SKIP_SLM_CONDITIONS = {
        # Allergen match → instant avoid, no SLM needed
        "allergen_match": lambda product, profile: (
            any(a.lower() in product.get("ingredients", "").lower()
                for a in profile.get("allergies", []))
        ),
        
        # NutriScore A/B → instant eat
        "nutriscore_good": lambda product, _: (
            product.get("nutriscore") in ("a", "b")
        ),
        
        # NutriScore E → instant avoid
        "nutriscore_bad": lambda product, _: (
            product.get("nutriscore") == "e"
        ),
    }
    
    def can_skip_slm(self, product: dict, profile: dict) -> tuple:
        """Returns (can_skip, verdict_dict) or (False, None)."""
        for name, check in self.SKIP_SLM_CONDITIONS.items():
            if check(product, profile):
                if name == "allergen_match":
                    return True, {
                        "verdict": "avoid",
                        "reason": "Contains ingredients matching your allergies",
                        "confidence": 0.99
                    }
                elif name == "nutriscore_good":
                    return True, {
                        "verdict": "eat",
                        "reason": f"NutriScore {product['nutriscore'].upper()} - well rated",
                        "confidence": 0.85
                    }
                elif name == "nutriscore_bad":
                    return True, {
                        "verdict": "avoid",
                        "reason": "NutriScore E - poorly rated nutritionally",
                        "confidence": 0.85
                    }
        return False, None
```

## 3. Parallel Execution

```python
import asyncio

async def process_query_parallel(text: str, session):
    """Run independent steps in parallel."""
    
    # Step 1: Classify intent (fast)
    intent, meta = rules_layer.classify(text)
    
    if intent == 'barcode':
        # Parallel: fetch product + pre-warm SLM
        product_task = asyncio.create_task(off_client.get_product(meta['barcode']))
        warmup_task = asyncio.create_task(slm_warmup())
        
        product = await product_task
        await warmup_task  # SLM ready when product arrives
        
        # Now analyze (SLM is warm)
        verdict = engine.analyze(product, session.user_profile)
        
    elif intent == 'food_query':
        # Parallel: resolve food + pre-warm SLM
        product_task = asyncio.create_task(resolver.resolve(meta['raw_text']))
        warmup_task = asyncio.create_task(slm_warmup())
        
        product = await product_task
        await warmup_task
        
        verdict = engine.analyze(product, session.user_profile)
    
    return verdict
```

## 4. Quantization

```python
# GGUF quantization for CPU inference (fastest option)
# Convert after fine-tuning:

# Step 1: Merge LoRA adapter
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(base, "./nutriassist-model-final")
merged = model.merge_and_unload()
merged.save_pretrained("./nutriassist-merged")

# Step 2: Convert to GGUF (using llama.cpp)
# python convert_hf_to_gguf.py ./nutriassist-merged --outtype q4_k_m

# Step 3: Use with llama-cpp-python
from llama_cpp import Llama

llm = Llama(
    model_path="./nutriassist-merged.Q4_K_M.gguf",
    n_ctx=512,       # Context window
    n_threads=4,     # CPU threads
    n_batch=64,      # Batch size
    verbose=False,
)

# Inference: ~150-300ms on CPU
output = llm(prompt, max_tokens=80, temperature=0.1, stop=["###"])
```

## Latency Budget Summary

| Component | Without Opt | With Optimization |
|-----------|------------|-------------------|
| STT | 300ms | 150ms (ONNX Whisper) |
| Intent Classification | 50ms | 5ms (rules fast-path) |
| Data Retrieval | 500ms | 50ms (cached) / 300ms (API) |
| SLM Inference | 500ms | 200ms (GGUF Q4) |
| Response + TTS | 300ms | 200ms (edge-tts streaming) |
| **Total** | **1650ms** | **605ms–905ms** |
