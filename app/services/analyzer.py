"""Agent 1 — Product Analysis: Rule-based primary + SLM fallback via Ollama."""

import json
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"

ANALYSIS_SYSTEM_PROMPT = (
    "You are a nutrition analysis engine. Given product data and user profile, "
    "output ONLY a valid JSON verdict.\nRules:\n"
    '- verdict: "eat" (healthy), "avoid" (unhealthy), "sometimes" (moderate)\n'
    "- reason: max 15 words, cite specific nutrients\n"
    "- confidence: 0.0-1.0 based on data completeness\n"
    '- If user has allergies matching ingredients, verdict MUST be "avoid"\n'
    '- If user is vegetarian/vegan and product contains meat/fish, verdict MUST be "avoid"\n'
    "- Condition-specific flags (diabetes, hypertension, cholesterol) "
    "must lower verdict if relevant nutrients are high"
)


class RuleBasedAnalyzer:
    """Deterministic rule-based analyzer — no model needed. Primary engine."""

    ALLERGEN_KEYWORDS = {
        "milk": ["milk", "dairy", "cream", "butter", "cheese", "whey",
                 "paneer", "curd", "yogurt", "ghee", "buttermilk", "khoya", "dahi"],
        "lactose": ["milk", "dairy", "cream", "butter", "cheese", "whey",
                    "paneer", "curd", "yogurt", "buttermilk", "lassi"],
        "gluten": ["wheat", "flour", "maida", "semolina", "suji", "rava",
                   "barley", "rye", "bread", "biscuit", "noodle", "pasta"],
        "peanuts": ["peanut", "groundnut"],
        "tree nuts": ["almond", "cashew", "walnut", "pistachio", "hazelnut"],
        "eggs": ["egg", "omelette", "bhurji"],
        "soy": ["soy", "soybean", "soya", "tofu"],
        "shellfish": ["shrimp", "prawn", "crab", "lobster"],
        "fish": ["fish", "tuna", "salmon", "sardine", "mackerel"],
    }

    # Non-veg keywords for diet filtering
    NON_VEG_KEYWORDS = [
        "chicken", "mutton", "lamb", "pork", "beef", "meat", "fish",
        "prawn", "shrimp", "crab", "lobster", "egg", "bacon", "ham",
        "salami", "sausage", "pepperoni", "lard", "gelatin", "tallow",
    ]

    def analyze(self, product: dict, user_profile: dict) -> dict:
        nut = product.get("nutriments", {})
        ingredients = (product.get("ingredients", "") or "").lower()
        product_name_lower = product.get("product_name", "").lower()
        searchable = f"{product_name_lower} {ingredients}"

        # 1. Allergen check
        for allergy in user_profile.get("allergies", []):
            keywords = self.ALLERGEN_KEYWORDS.get(allergy.lower(), [allergy.lower()])
            for kw in keywords:
                if kw in searchable:
                    return {
                        "verdict": "avoid",
                        "reason": f"Contains {kw} — matches your {allergy} allergy",
                        "confidence": 0.98,
                    }

        # 2. Diet check (vegetarian/vegan)
        diet = user_profile.get("diet_type", "").lower()
        if diet in ("vegetarian", "vegan"):
            for kw in self.NON_VEG_KEYWORDS:
                if kw in searchable:
                    return {
                        "verdict": "avoid",
                        "reason": f"Contains {kw} — not suitable for {diet} diet",
                        "confidence": 0.97,
                    }

        # 3. Score nutrients
        sugar = nut.get("sugars_100g")
        fat = nut.get("fat_100g")
        sat_fat = nut.get("saturated_fat_100g")
        salt = nut.get("salt_100g")
        protein = nut.get("proteins_100g")
        fiber = nut.get("fiber_100g")
        calories = nut.get("energy_kcal_100g")

        available = sum(1 for v in [sugar, fat, sat_fat, salt, protein, fiber, calories]
                        if v is not None)
        if available == 0:
            conf = product.get("data_confidence", "low")
            return {
                "verdict": "sometimes",
                "reason": "Insufficient nutrition data for definitive assessment",
                "confidence": 0.30 if conf == "low" else 0.45,
            }

        score = 0
        reasons = []

        if sugar is not None:
            if sugar > 20: score -= 3; reasons.append(f"very high sugar ({sugar}g)")
            elif sugar > 12: score -= 2; reasons.append(f"high sugar ({sugar}g)")
            elif sugar > 8: score -= 1
            elif sugar < 3: score += 1; reasons.append("low sugar")

        if fat is not None:
            if fat > 25: score -= 2; reasons.append(f"very high fat ({fat}g)")
            elif fat > 15: score -= 1; reasons.append(f"high fat ({fat}g)")
            elif fat < 3: score += 1; reasons.append("low fat")

        if sat_fat is not None:
            if sat_fat > 10: score -= 2; reasons.append("high saturated fat")
            elif sat_fat > 5: score -= 1

        if salt is not None:
            if salt > 1.5: score -= 2; reasons.append(f"high salt ({salt}g)")
            elif salt > 1.0: score -= 1

        if protein is not None:
            if protein > 15: score += 2; reasons.append(f"high protein ({protein}g)")
            elif protein > 8: score += 1; reasons.append(f"good protein ({protein}g)")

        if fiber is not None:
            if fiber > 5: score += 2; reasons.append(f"high fiber ({fiber}g)")
            elif fiber > 3: score += 1; reasons.append("good fiber")

        if calories is not None and calories < 80:
            score += 1; reasons.append("low calorie")

        # Condition adjustments
        conditions = user_profile.get("conditions", [])
        goal = user_profile.get("goal", "")

        if "diabetes" in conditions and sugar is not None and sugar > 8:
            score -= 2; reasons.append("risky for diabetes")
        if "hypertension" in conditions and salt is not None and salt > 1.0:
            score -= 2; reasons.append("risky for hypertension")
        if "cholesterol" in conditions and sat_fat is not None and sat_fat > 5:
            score -= 2; reasons.append("bad for cholesterol")
        if "high_cholesterol" in conditions and sat_fat is not None and sat_fat > 5:
            score -= 2; reasons.append("bad for cholesterol")
        if goal == "weight_loss" and calories is not None and calories > 350:
            score -= 1

        # Determine verdict
        if score >= 2: verdict = "eat"
        elif score <= -2: verdict = "avoid"
        else: verdict = "sometimes"

        reason = ", ".join(reasons[:3]) if reasons else "Moderate nutritional profile"
        reason = reason[0].upper() + reason[1:]
        base_conf = min(0.95, 0.6 + (available / 7) * 0.35)

        return {"verdict": verdict, "reason": reason, "confidence": round(base_conf, 2)}


class OllamaSLMAnalyzer:
    """SLM fallback analyzer via Ollama — used when rule-based confidence is low."""

    def __init__(self, model: str = "qwen:0.5b", timeout: float = 8.0):
        self.model = model
        self.timeout = timeout
        self._available: Optional[bool] = None

    async def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get("http://localhost:11434/api/tags")
                self._available = r.status_code == 200
        except Exception:
            self._available = False
        return self._available

    async def analyze(self, product: dict, user_profile: dict) -> Optional[dict]:
        """SLM-based analysis via Ollama. Returns None if unavailable."""
        if not await self.is_available():
            return None

        nut = product.get("nutriments", {})

        def fmt(key):
            v = nut.get(key)
            return str(round(v, 1)) if v is not None else "N/A"

        allergies = ", ".join(user_profile.get("allergies", [])) or "none"
        conditions = ", ".join(user_profile.get("conditions", [])) or "none"
        diet = user_profile.get("diet_type", "none") or "none"

        prompt = f"""[PRODUCT]
Name: {product.get('product_name', 'Unknown')}
Nutrients per 100g: Cal={fmt('energy_kcal_100g')}, Fat={fmt('fat_100g')}, SatFat={fmt('saturated_fat_100g')}, Sugar={fmt('sugars_100g')}, Salt={fmt('salt_100g')}, Protein={fmt('proteins_100g')}, Fiber={fmt('fiber_100g')}
Ingredients: {product.get('ingredients', 'N/A') or 'N/A'}

[USER] Allergies: {allergies} | Conditions: {conditions} | Diet: {diet}

[OUTPUT] Valid JSON only:"""

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(
                    OLLAMA_URL,
                    json={
                        "model": self.model,
                        "system": ANALYSIS_SYSTEM_PROMPT,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "num_predict": 80,
                        },
                    },
                )
                if r.status_code == 200:
                    raw = r.json().get("response", "")
                    return self._parse(raw)
        except Exception as e:
            logger.warning(f"Ollama SLM analysis error: {e}")

        return None

    def _parse(self, raw: str) -> Optional[dict]:
        import re
        try:
            match = re.search(r'\{[^{}]+\}', raw)
            if match:
                result = json.loads(match.group())
                if all(k in result for k in ["verdict", "reason", "confidence"]):
                    return {
                        "verdict": result["verdict"].lower(),
                        "reason": str(result["reason"]),
                        "confidence": min(max(float(result["confidence"]), 0.0), 1.0),
                    }
        except (json.JSONDecodeError, ValueError):
            pass
        return None


class HybridAnalyzer:
    """Main analyzer: Rule-based primary, SLM fallback for low-confidence cases."""

    def __init__(self, slm_model: str = "qwen:0.5b"):
        self.rules = RuleBasedAnalyzer()
        self.slm = OllamaSLMAnalyzer(model=slm_model)

    async def analyze(self, product: dict, user_profile: dict) -> dict:
        # Always run rule-based first (instant)
        verdict = self.rules.analyze(product, user_profile)

        # If rule-based is confident enough, use it directly
        if verdict["confidence"] >= 0.7:
            verdict["source"] = "rules"
            return verdict

        # Low confidence → try SLM fallback
        slm_result = await self.slm.analyze(product, user_profile)
        if slm_result:
            slm_result["source"] = "slm"
            return slm_result

        # SLM failed → use rule-based anyway
        verdict["source"] = "rules"
        return verdict
