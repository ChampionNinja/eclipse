"""Agent 2 — Response Generation.

Generates natural, conversational responses from structured verdicts.
Uses smart templates with randomized variations for instant, reliable output.
Optionally enhances with Ollama SLM if available and response passes validation.
"""

import random
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

# ===== TEMPLATE-BASED RESPONSES (primary, instant, reliable) =====

EAT_TEMPLATES = [
    "{name} is a great choice! {reason}. Go for it! 🥗",
    "Good news — {name} looks healthy! {reason}.",
    "{name}? Absolutely! {reason}. Enjoy!",
    "Thumbs up for {name}! {reason}.",
    "{name} is a solid pick — {reason}. Eat up!",
    "Yes! {name} is good for you. {reason}.",
]

AVOID_TEMPLATES = [
    "I'd skip {name} — {reason}. Try something lighter!",
    "Heads up — {name} isn't great. {reason}.",
    "{name}? Better to avoid. {reason}.",
    "Not the best choice — {name} has issues. {reason}.",
    "I'd pass on {name}. {reason}. Want alternatives?",
    "Watch out with {name} — {reason}.",
]

SOMETIMES_TEMPLATES = [
    "{name} is okay in moderation. {reason}.",
    "{name}? Fine sometimes, just don't overdo it. {reason}.",
    "You can have {name} occasionally. {reason}.",
    "{name} in small portions is alright. {reason}.",
    "Moderation is key with {name}. {reason}.",
]

ALLERGY_TEMPLATES = [
    "⚠️ Stop — {name} is not safe for you! {reason}.",
    "🚫 Avoid {name}! {reason}. This is an allergy risk.",
    "Warning — {name} contains allergens! {reason}.",
]

CONDITION_TEMPLATES = [
    "Be careful with {name} — {reason}. Not ideal for your {condition}.",
    "{name} isn't recommended given your {condition}. {reason}.",
    "With your {condition}, I'd avoid {name}. {reason}.",
]


def _pick_template(verdict: str, reason: str, conditions: list) -> list:
    """Select the right template pool based on verdict and context."""
    reason_lower = reason.lower()

    # Allergy match
    if "allergy" in reason_lower or "allergen" in reason_lower:
        return ALLERGY_TEMPLATES

    # Condition-specific avoid
    if verdict == "avoid" and conditions and any(
        c in reason_lower for c in ["diabetes", "hypertension", "cholesterol"]
    ):
        return CONDITION_TEMPLATES

    if verdict == "eat":
        return EAT_TEMPLATES
    elif verdict == "avoid":
        return AVOID_TEMPLATES
    else:
        return SOMETIMES_TEMPLATES


def generate_template_response(
    product_name: str, verdict: dict, user_profile: dict
) -> str:
    """Generate a natural response using templates. Instant and reliable."""
    v = verdict.get("verdict", "sometimes")
    reason = verdict.get("reason", "moderate nutritional profile")
    conditions = user_profile.get("conditions", [])

    # Clean reason: capitalize first letter, remove trailing period
    reason = reason.rstrip(".")
    if reason:
        reason = reason[0].lower() + reason[1:]

    templates = _pick_template(v, reason, conditions)
    template = random.choice(templates)

    # Build response
    response = template.format(
        name=product_name,
        reason=reason,
        condition=conditions[0].replace("_", " ") if conditions else "condition",
    )

    return response


class ResponseAgent:
    """Response generator: templates (primary) + optional SLM enhancement."""

    def __init__(self, model: str = "qwen:1.8b", timeout: float = 8.0):
        self.model = model
        self.timeout = timeout
        self._available: Optional[bool] = None

    async def is_available(self) -> bool:
        """Check if Ollama is reachable."""
        if self._available is not None:
            return self._available
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get("http://localhost:11434/api/tags")
                self._available = r.status_code == 200
        except Exception:
            self._available = False
        return self._available

    async def generate(
        self,
        product_name: str,
        verdict: dict,
        user_profile: dict,
    ) -> str:
        """Generate response. Always returns a valid string (never None)."""
        # Always generate a template response first (instant, reliable)
        template_response = generate_template_response(
            product_name, verdict, user_profile
        )

        # Optionally try SLM for a more natural response
        if await self.is_available():
            slm_response = await self._try_slm(product_name, verdict, user_profile)
            if slm_response:
                return slm_response

        return template_response

    async def _try_slm(
        self, product_name: str, verdict: dict, user_profile: dict
    ) -> Optional[str]:
        """Try SLM enhancement. Returns None if response is bad."""
        v = verdict.get("verdict", "sometimes")
        reason = verdict.get("reason", "")

        messages = [
            {
                "role": "system",
                "content": (
                    "Rephrase the food verdict below as ONE short friendly spoken sentence (under 20 words). "
                    "Match the verdict tone exactly. Output ONLY the sentence."
                ),
            },
            {
                "role": "user",
                "content": f"{product_name}: {v.upper()}. {reason}.",
            },
        ]

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(
                    OLLAMA_CHAT_URL,
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "options": {"temperature": 0.3, "num_predict": 40},
                    },
                )
                if r.status_code == 200:
                    text = r.json().get("message", {}).get("content", "").strip()
                    text = text.strip('"\'').strip()

                    if self._is_valid(text, product_name, v):
                        return text

        except Exception as e:
            logger.warning(f"SLM error: {e}")

        return None

    def _is_valid(self, text: str, product_name: str, verdict: str) -> bool:
        """Validate SLM response isn't generic garbage."""
        if not text or len(text) < 8 or len(text) > 200:
            return False

        text_lower = text.lower()

        # Reject generic chatbot responses
        bad_phrases = [
            "how can i help", "is there anything else", "you're welcome",
            "let me know", "i'm here to help", "let's just enjoy",
            "no need to be concerned", "i see what you mean",
            "the short friendly sentence", "rephrase", "verdict",
        ]
        if any(p in text_lower for p in bad_phrases):
            return False

        # Reject if it contradicts the verdict
        if verdict == "avoid" and any(w in text_lower for w in ["great choice", "go for it", "enjoy", "healthy choice"]):
            return False
        if verdict == "eat" and any(w in text_lower for w in ["skip", "avoid", "stay away", "not safe"]):
            return False

        return True
