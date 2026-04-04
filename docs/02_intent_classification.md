# STEP 2: Intent Classification Design

## Intent Categories

| Intent | Trigger Pattern | Example |
|--------|----------------|---------|
| `barcode` | Numeric 8-13 digits | "scan 8901234567890" |
| `food_query` | Food name + question | "is paneer healthy?" |
| `diet_advice` | Diet/health keywords | "what should I eat for diabetes?" |
| `casual` | Greetings, meta-questions | "hello", "what can you do?" |
| `follow_up` | Pronoun/context ref | "what about sugar in it?" |

## Rules Layer (Fast Path — ~5ms)

Deterministic regex/keyword classification. Runs BEFORE any model call.

```python
import re
from typing import Optional, Tuple

class RulesLayer:
    BARCODE_PATTERN = re.compile(r'\b(\d{8,13})\b')
    GREETING_PATTERNS = re.compile(
        r'^(hi|hello|hey|good morning|good evening|thanks|thank you|bye)\b',
        re.IGNORECASE
    )
    FOOD_INDICATORS = [
        'is', 'can i eat', 'should i eat', 'how healthy', 'tell me about',
        'what about', 'nutrition', 'calories in', 'good for', 'bad for'
    ]

    def classify(self, text: str) -> Tuple[Optional[str], Optional[dict]]:
        text = text.strip()

        # Barcode (highest priority)
        barcode_match = self.BARCODE_PATTERN.search(text)
        if barcode_match:
            return 'barcode', {'barcode': barcode_match.group(1)}

        # Greeting/casual
        if self.GREETING_PATTERNS.match(text):
            return 'casual', {'subtype': 'greeting'}

        # Food query (keyword check)
        text_lower = text.lower()
        for indicator in self.FOOD_INDICATORS:
            if indicator in text_lower:
                return 'food_query', {'raw_text': text}

        return None, None  # Fall through to Router
```

## Router (Fallback — ~50ms)

Used ONLY when Rules Layer returns None.

```python
class KeywordRouter:
    INTENT_KEYWORDS = {
        'food_query': ['eat', 'food', 'healthy', 'nutrition', 'calories',
                       'protein', 'sugar', 'fat', 'carbs', 'diet', 'meal'],
        'diet_advice': ['weight loss', 'diabetes', 'cholesterol', 'blood pressure',
                        'diet plan', 'what should i eat', 'suggest', 'recommend'],
        'casual': ['who are you', 'what can you do', 'help'],
    }

    def classify(self, text: str) -> str:
        text_lower = text.lower()
        scores = {
            intent: sum(1 for kw in keywords if kw in text_lower)
            for intent, keywords in self.INTENT_KEYWORDS.items()
        }
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else 'casual'
```

## Latency Strategy

1. Rules first — 90%+ queries classified in <5ms
2. Skip router when rules are confident
3. Preload models at startup
4. Parallel: while classifying, pre-warm SLM context
