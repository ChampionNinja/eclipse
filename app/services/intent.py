"""Intent classification — Rules layer + keyword router."""

import re
from typing import Optional, Tuple


class RulesLayer:
    """Deterministic regex intent classifier. Runs first, ~5ms."""

    BARCODE_PATTERN = re.compile(r"\b(\d{8,13})\b")
    GREETING_PATTERNS = re.compile(
        r"^(hi|hello|hey|good morning|good evening|thanks|thank you|bye|goodbye)\b",
        re.IGNORECASE,
    )
    FOOD_INDICATORS = [
        "is ", "can i eat", "should i eat", "how healthy", "tell me about",
        "what about", "nutrition", "calories in", "good for", "bad for",
        "how much", "healthy", "unhealthy",
    ]

    def classify(self, text: str) -> Tuple[Optional[str], Optional[dict]]:
        text = text.strip()

        barcode_match = self.BARCODE_PATTERN.search(text)
        if barcode_match:
            return "barcode", {"barcode": barcode_match.group(1)}

        if self.GREETING_PATTERNS.match(text):
            return "casual", {"subtype": "greeting"}

        text_lower = text.lower()
        for indicator in self.FOOD_INDICATORS:
            if indicator in text_lower:
                return "food_query", {"raw_text": text}

        return None, None


class KeywordRouter:
    """Fallback keyword scorer when RulesLayer returns None."""

    INTENT_KEYWORDS = {
        "food_query": [
            "eat", "food", "healthy", "nutrition", "calories",
            "protein", "sugar", "fat", "carbs", "diet", "meal",
            "snack", "drink", "fruit", "vegetable",
        ],
        "diet_advice": [
            "weight loss", "diabetes", "cholesterol", "blood pressure",
            "diet plan", "what should i eat", "suggest", "recommend",
        ],
        "casual": ["who are you", "what can you do", "help", "how does this work"],
    }

    def classify(self, text: str) -> str:
        text_lower = text.lower()
        scores = {
            intent: sum(1 for kw in keywords if kw in text_lower)
            for intent, keywords in self.INTENT_KEYWORDS.items()
        }
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "food_query"


class FollowUpResolver:
    """Detects follow-up queries that reference the current product."""

    FOLLOW_UP_PATTERNS = [
        "it", "this", "that", "the product", "this product",
        "how much", "what about", "does it", "is it", "can i",
        "more about", "tell me more", "why", "explain",
    ]

    def is_follow_up(self, text: str, session) -> bool:
        if not session.has_product_context():
            return False
        text_lower = text.lower()
        return any(p in text_lower for p in self.FOLLOW_UP_PATTERNS)
