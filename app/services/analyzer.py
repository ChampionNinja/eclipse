"""Product Analysis — SLM inference + rule-based fallback."""

import json
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Try importing torch/transformers — graceful fallback if not available
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("torch/transformers not installed — using rule-based analysis only")


SYSTEM_PROMPT = (
    "You are a nutrition analysis engine. Given product data and user profile, "
    "output ONLY a valid JSON verdict.\nRules:\n"
    '- verdict: "eat" (healthy), "avoid" (unhealthy), "sometimes" (moderate)\n'
    "- reason: max 15 words, cite specific nutrients\n"
    "- confidence: 0.0-1.0 based on data completeness\n"
    '- If user has allergies matching ingredients, verdict MUST be "avoid"\n'
    '- If data_confidence is "low", confidence must be <= 0.5'
)


class RuleBasedAnalyzer:
    """Deterministic rule-based analyzer — no model needed."""

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
    }

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

        # 2. Score nutrients
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
            elif fiber > 3: score += 1; reasons.append(f"good fiber")

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


class SLMAnalyzer:
    """Fine-tuned SLM analyzer. Falls back to rules if model unavailable."""

    def __init__(self, model_path: Optional[str] = None, base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.rule_analyzer = RuleBasedAnalyzer()
        self.model = None
        self.tokenizer = None
        self._json_pattern = re.compile(r"\{[^{}]+\}")

        if model_path and HAS_TORCH:
            self._load_model(base_model, model_path)

    def _load_model(self, base_model: str, adapter_path: str):
        try:
            logger.info(f"Loading SLM: {base_model} + {adapter_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model.eval()
            logger.info("SLM loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SLM: {e}")
            self.model = None
            self.tokenizer = None

    def analyze(self, product: dict, user_profile: dict) -> dict:
        # If no model loaded, fall back to rules
        if self.model is None or self.tokenizer is None:
            return self.rule_analyzer.analyze(product, user_profile)

        try:
            return self._slm_analyze(product, user_profile)
        except Exception as e:
            logger.warning(f"SLM inference failed: {e}, falling back to rules")
            return self.rule_analyzer.analyze(product, user_profile)

    def _slm_analyze(self, product: dict, user_profile: dict) -> dict:
        prompt = self._build_prompt(product, user_profile)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=450)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.1,
                do_sample=False,
                repetition_penalty=1.1,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(generated, skip_special_tokens=True)

        return self._parse(raw)

    def _build_prompt(self, product: dict, profile: dict) -> str:
        nut = product.get("nutriments", {})

        def fmt(key):
            v = nut.get(key)
            return str(round(v, 1)) if v is not None else "not available"

        allergies = ", ".join(profile.get("allergies", [])) or "none"
        conditions = ", ".join(profile.get("conditions", [])) or "none"

        return f"""[PRODUCT]
Name: {product.get('product_name', 'Unknown')}
Source: {product.get('source_type', 'inferred')}
Data Confidence: {product.get('data_confidence', 'low')}
Nutrients (per 100g):
  Calories: {fmt('energy_kcal_100g')}
  Fat: {fmt('fat_100g')}
  Saturated Fat: {fmt('saturated_fat_100g')}
  Sugar: {fmt('sugars_100g')}
  Salt: {fmt('salt_100g')}
  Protein: {fmt('proteins_100g')}
  Fiber: {fmt('fiber_100g')}
Ingredients: {product.get('ingredients', 'not available') or 'not available'}

[USER PROFILE]
Allergies: {allergies}
Conditions: {conditions}
Diet: {profile.get('diet_type') or 'none'}
Goal: {profile.get('goal') or 'none'}

[OUTPUT] Respond with ONLY valid JSON:"""

    def _parse(self, raw: str) -> dict:
        try:
            match = self._json_pattern.search(raw)
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
        return {"verdict": "sometimes", "reason": "Analysis unavailable", "confidence": 0.3}
