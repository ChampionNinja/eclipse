"""
NutriAssist — Dataset Generator from INDB Excel
Generates Oumi-compatible training JSONL from the Indian Nutritional Database.
NO external API calls — purely local, clean data.

Usage:
    python generate_from_indb.py
    python generate_from_indb.py --output data/training_data_indb.jsonl
"""

import json
import random
import argparse
import math
from pathlib import Path
import openpyxl


# ===== SYSTEM PROMPT =====
SYSTEM_PROMPT = (
    "You are a nutrition analysis engine. Given product data and user profile, "
    "output ONLY a valid JSON verdict.\nRules:\n"
    '- verdict: "eat" (healthy), "avoid" (unhealthy), "sometimes" (moderate)\n'
    "- reason: max 15 words, cite specific nutrients\n"
    "- confidence: 0.0-1.0 based on data completeness\n"
    '- If user has allergies matching ingredients, verdict MUST be "avoid"\n'
    '- If data_confidence is "low", confidence must be <= 0.5'
)

# ===== USER PROFILES =====
PROFILES = [
    {"allergies": [], "conditions": [], "diet_type": "", "goal": "weight_loss"},
    {"allergies": [], "conditions": [], "diet_type": "", "goal": "muscle_gain"},
    {"allergies": [], "conditions": ["diabetes"], "diet_type": "", "goal": ""},
    {"allergies": [], "conditions": ["hypertension"], "diet_type": "", "goal": ""},
    {"allergies": [], "conditions": ["cholesterol"], "diet_type": "", "goal": "weight_loss"},
    {"allergies": [], "conditions": [], "diet_type": "vegetarian", "goal": ""},
    {"allergies": [], "conditions": [], "diet_type": "vegan", "goal": "muscle_gain"},
    {"allergies": [], "conditions": [], "diet_type": "keto", "goal": ""},
    {"allergies": ["lactose", "milk"], "conditions": [], "diet_type": "", "goal": ""},
    {"allergies": ["gluten"], "conditions": [], "diet_type": "", "goal": ""},
    {"allergies": ["peanuts"], "conditions": [], "diet_type": "", "goal": ""},
    {"allergies": ["tree nuts"], "conditions": [], "diet_type": "", "goal": ""},
    {"allergies": ["eggs"], "conditions": [], "diet_type": "", "goal": ""},
    {"allergies": [], "conditions": ["diabetes"], "diet_type": "vegetarian", "goal": "weight_loss"},
    {"allergies": ["gluten"], "conditions": ["diabetes"], "diet_type": "", "goal": ""},
    {"allergies": [], "conditions": [], "diet_type": "", "goal": ""},
    {"allergies": [], "conditions": [], "diet_type": "", "goal": "maintenance"},
]

# ===== INGREDIENT INFERENCE (food_name → likely ingredients for allergen matching) =====
INGREDIENT_HINTS = {
    # Dairy
    "paneer": "paneer, milk, citric acid",
    "curd": "milk, live cultures",
    "raita": "curd, milk, cucumber, spices",
    "lassi": "curd, milk, sugar, water",
    "chaas": "buttermilk, milk, salt, cumin",
    "kheer": "milk, rice, sugar, cardamom",
    "rabri": "milk, sugar, cardamom",
    "kulfi": "milk, sugar, cardamom, nuts",
    "dahi": "milk, live cultures",
    "ghee": "clarified butter, milk fat",
    "butter": "cream, milk, salt",
    "cheese": "milk, rennet, salt",
    "milk": "whole milk",
    "cream": "milk cream",
    # Gluten
    "roti": "whole wheat flour, water, salt",
    "chapati": "whole wheat flour, water, salt",
    "paratha": "whole wheat flour, ghee, salt",
    "naan": "maida, yeast, milk, ghee",
    "puri": "whole wheat flour, oil, salt",
    "bhatura": "maida, yogurt, oil",
    "bread": "wheat flour, yeast, salt, sugar",
    "biscuit": "wheat flour, sugar, butter",
    "pasta": "wheat flour, water",
    "noodle": "wheat flour, salt",
    "maggi": "wheat flour, palm oil, salt, spices",
    "samosa": "maida, potato, peas, oil, spices",
    "kachori": "maida, moong dal, oil, spices",
    "upma": "semolina, onion, mustard seeds, oil, salt",
    "suji": "semolina, wheat",
    "rava": "semolina, wheat",
    "halwa": "semolina, ghee, sugar",
    "cake": "wheat flour, sugar, eggs, butter",
    "toast": "wheat flour, butter, salt",
    # Nuts
    "almond": "almonds",
    "cashew": "cashew nuts",
    "walnut": "walnuts",
    "pistachio": "pistachios",
    "peanut": "peanuts, groundnuts",
    "groundnut": "peanuts, groundnuts",
    # Eggs
    "egg": "egg",
    "omelette": "egg, oil, onion, spices",
    "bhurji": "egg, onion, tomato, oil, spices",
    "anda": "egg",
    # Non-veg
    "chicken": "chicken, spices, oil",
    "mutton": "mutton, spices, oil",
    "fish": "fish, spices, oil",
    "prawn": "prawns, spices, oil",
    "shrimp": "shrimp, spices, oil",
    "meat": "meat, spices, oil",
    "keema": "minced meat, spices, oil, onion",
}


def safe_float(val):
    """Convert value to float, return None if invalid."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else round(f, 2)
    except (ValueError, TypeError):
        return None


def infer_ingredients(food_name: str) -> str:
    """Infer likely ingredients from food name for allergen matching."""
    name_lower = food_name.lower()
    for keyword, ingredients in INGREDIENT_HINTS.items():
        if keyword in name_lower:
            return ingredients
    # Generic fallback
    return "not available"


def compute_verdict(nutrients: dict, ingredients: str, profile: dict) -> dict:
    """Deterministic verdict computation from nutrient values."""
    ingredients_lower = (ingredients or "").lower()

    # 1. ALLERGEN CHECK (highest priority)
    allergen_keywords = {
        "milk": ["milk", "dairy", "cream", "butter", "cheese", "whey", "casein",
                 "lactose", "paneer", "curd", "yogurt", "ghee", "buttermilk", "khoya"],
        "lactose": ["milk", "dairy", "cream", "butter", "cheese", "whey", "casein",
                    "lactose", "paneer", "curd", "yogurt", "buttermilk"],
        "gluten": ["wheat", "flour", "maida", "semolina", "suji", "rava", "barley",
                   "rye", "bread", "biscuit", "noodle", "pasta"],
        "peanuts": ["peanut", "groundnut"],
        "tree nuts": ["almond", "cashew", "walnut", "pistachio", "hazelnut", "pecan"],
        "eggs": ["egg", "albumin", "ovalbumin", "omelette", "bhurji"],
        "soy": ["soy", "soybean", "soya", "tofu"],
        "shellfish": ["shrimp", "prawn", "crab", "lobster"],
    }

    for allergy in profile.get("allergies", []):
        allergy_lower = allergy.lower()
        keywords = allergen_keywords.get(allergy_lower, [allergy_lower])
        for kw in keywords:
            if kw in ingredients_lower:
                return {
                    "verdict": "avoid",
                    "reason": f"Contains {kw} — matches your {allergy} allergy",
                    "confidence": 0.98,
                }

    # 2. EXTRACT VALUES
    sugar = nutrients.get("sugars_100g")
    fat = nutrients.get("fat_100g")
    sat_fat = nutrients.get("saturated_fat_100g")
    sodium = nutrients.get("sodium_mg")  # INDB uses mg, not g
    protein = nutrients.get("proteins_100g")
    fiber = nutrients.get("fiber_100g")
    calories = nutrients.get("energy_kcal_100g")

    # Convert sodium mg to salt g (salt ≈ sodium × 2.5 / 1000)
    salt_g = round(sodium * 2.5 / 1000, 2) if sodium is not None else None

    available = sum(1 for v in [sugar, fat, sat_fat, salt_g, protein, fiber, calories]
                    if v is not None)
    base_confidence = min(0.95, 0.6 + (available / 7) * 0.35)

    # 3. SCORING
    score = 0
    reasons = []

    if sugar is not None:
        if sugar > 20:
            score -= 3; reasons.append(f"very high sugar ({sugar}g)")
        elif sugar > 12:
            score -= 2; reasons.append(f"high sugar ({sugar}g)")
        elif sugar > 8:
            score -= 1; reasons.append(f"moderate sugar ({sugar}g)")
        elif sugar < 3:
            score += 1; reasons.append("low sugar")

    if fat is not None:
        if fat > 25:
            score -= 2; reasons.append(f"very high fat ({fat}g)")
        elif fat > 15:
            score -= 1; reasons.append(f"high fat ({fat}g)")
        elif fat < 3:
            score += 1; reasons.append("low fat")

    if sat_fat is not None:
        # INDB has SFA in mg, convert to g
        sat_fat_g = sat_fat / 1000 if sat_fat > 100 else sat_fat  # detect mg vs g
        if sat_fat_g > 10:
            score -= 2; reasons.append(f"high saturated fat")
        elif sat_fat_g > 5:
            score -= 1

    if salt_g is not None:
        if salt_g > 1.5:
            score -= 2; reasons.append(f"high salt ({salt_g}g)")
        elif salt_g > 1.0:
            score -= 1; reasons.append(f"moderate salt")

    if protein is not None:
        if protein > 15:
            score += 2; reasons.append(f"high protein ({protein}g)")
        elif protein > 8:
            score += 1; reasons.append(f"good protein ({protein}g)")

    if fiber is not None:
        if fiber > 5:
            score += 2; reasons.append(f"high fiber ({fiber}g)")
        elif fiber > 3:
            score += 1; reasons.append(f"good fiber ({fiber}g)")

    if calories is not None:
        if calories < 80:
            score += 1; reasons.append("low calorie")
        elif calories > 400:
            score -= 1; reasons.append(f"high calorie ({int(calories)} kcal)")

    # 4. CONDITION ADJUSTMENTS
    conditions = profile.get("conditions", [])
    goal = profile.get("goal", "")

    if "diabetes" in conditions:
        if sugar is not None and sugar > 8:
            score -= 2; reasons.append("risky for diabetes")
        elif sugar is not None and sugar < 3:
            reasons.append("safe sugar level for diabetes")

    if "hypertension" in conditions:
        if salt_g is not None and salt_g > 1.0:
            score -= 2; reasons.append("risky for hypertension")
        elif sodium is not None and sodium > 400:
            score -= 1; reasons.append("elevated sodium")

    if "cholesterol" in conditions and sat_fat is not None:
        sat_fat_g = sat_fat / 1000 if sat_fat > 100 else sat_fat
        if sat_fat_g > 5:
            score -= 2; reasons.append("bad for cholesterol")

    if goal == "weight_loss":
        if calories is not None and calories > 300:
            score -= 1
            if "high calorie" not in " ".join(reasons):
                reasons.append("calorie-dense for weight loss")
        if fat is not None and fat > 15:
            if "high fat" not in " ".join(reasons):
                reasons.append("high fat for weight loss")

    if goal == "muscle_gain" and protein is not None and protein > 10:
        score += 1
        if "protein" not in " ".join(reasons):
            reasons.append("good for muscle gain")

    # 5. VERDICT
    if score >= 2:
        verdict = "eat"
    elif score <= -2:
        verdict = "avoid"
    else:
        verdict = "sometimes"

    reason = ", ".join(reasons[:3]) if reasons else "Moderate nutritional profile"
    words = reason.split()
    if len(words) > 15:
        reason = " ".join(words[:15])
    reason = reason[0].upper() + reason[1:] if reason else "Moderate nutritional profile"

    return {"verdict": verdict, "reason": reason, "confidence": round(base_confidence, 2)}


def format_val(val):
    """Format a nutrient value for the prompt."""
    if val is None:
        return "not available"
    return round(val, 1) if isinstance(val, float) else val


def build_sample(food_name, nutrients, ingredients, profile, verdict):
    """Build one Oumi conversations-format sample."""

    # Format salt from sodium
    sodium = nutrients.get("sodium_mg")
    salt_g = round(sodium * 2.5 / 1000, 2) if sodium is not None else None

    user_content = f"""[PRODUCT]
Name: {food_name}
Source: inferred
Data Confidence: high
Nutrients (per 100g):
  Calories: {format_val(nutrients.get('energy_kcal_100g'))}
  Fat: {format_val(nutrients.get('fat_100g'))}
  Saturated Fat: {format_val(nutrients.get('saturated_fat_100g'))}
  Sugar: {format_val(nutrients.get('sugars_100g'))}
  Salt: {format_val(salt_g)}
  Protein: {format_val(nutrients.get('proteins_100g'))}
  Fiber: {format_val(nutrients.get('fiber_100g'))}
Ingredients: {ingredients}

[USER PROFILE]
Allergies: {', '.join(profile.get('allergies', [])) or 'none'}
Conditions: {', '.join(profile.get('conditions', [])) or 'none'}
Diet: {profile.get('diet_type') or 'none'}
Goal: {profile.get('goal') or 'none'}

[OUTPUT] Respond with ONLY valid JSON:"""

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps(verdict)},
        ]
    }


def load_indb(excel_path: str) -> list:
    """Load INDB Excel into list of dicts."""
    wb = openpyxl.load_workbook(excel_path, read_only=True)
    ws = wb["Sheet1"]

    foods = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        food_code = row[0]
        food_name = row[1]
        if not food_name:
            continue

        nutrients = {
            "energy_kcal_100g": safe_float(row[4]),
            "carbs_100g": safe_float(row[5]),
            "proteins_100g": safe_float(row[6]),
            "fat_100g": safe_float(row[7]),
            "sugars_100g": safe_float(row[8]),
            "fiber_100g": safe_float(row[9]),
            "saturated_fat_100g": safe_float(row[10]),  # SFA in mg in INDB
            "sodium_mg": safe_float(row[17]),
            "cholesterol_mg": safe_float(row[13]),
            "calcium_mg": safe_float(row[14]),
            "iron_mg": safe_float(row[19]),
        }

        foods.append({
            "code": food_code,
            "name": str(food_name).strip(),
            "nutrients": nutrients,
        })

    wb.close()
    return foods


def main():
    parser = argparse.ArgumentParser(description="Generate training data from INDB Excel")
    parser.add_argument("--excel", default="data/Anuvaad_INDB_2024.11.xlsx")
    parser.add_argument("--output", default="data/training_data_indb.jsonl")
    parser.add_argument("--seed-data", default="data/training_data.jsonl",
                        help="Hand-curated seed data to merge")
    parser.add_argument("--profiles-per-food", type=int, default=2,
                        help="Number of user profiles to pair with each food")
    args = parser.parse_args()

    # 1. Load seed data
    all_samples = []
    seed_path = Path(args.seed_data)
    if seed_path.exists():
        with open(seed_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_samples.append(json.loads(line))
        print(f"Loaded {len(all_samples)} seed samples")

    # 2. Load INDB
    foods = load_indb(args.excel)
    print(f"Loaded {len(foods)} foods from INDB")

    # 3. Generate samples
    generated = 0
    for food in foods:
        name = food["name"]
        nutrients = food["nutrients"]
        ingredients = infer_ingredients(name)

        # Pick N random profiles
        n = args.profiles_per_food
        selected = random.sample(PROFILES, min(n, len(PROFILES)))

        for profile in selected:
            verdict = compute_verdict(nutrients, ingredients, profile)
            sample = build_sample(name, nutrients, ingredients, profile, verdict)
            all_samples.append(sample)
            generated += 1

    print(f"Generated {generated} samples from INDB")

    # 4. Shuffle
    random.shuffle(all_samples)

    # 5. Write
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(all_samples)} samples → {output_path}")

    # Stats
    verdicts = {"eat": 0, "avoid": 0, "sometimes": 0}
    for s in all_samples:
        for msg in s["messages"]:
            if msg["role"] == "assistant":
                try:
                    v = json.loads(msg["content"])
                    verdicts[v["verdict"]] = verdicts.get(v["verdict"], 0) + 1
                except:
                    pass
    print(f"Distribution: {verdicts}")


if __name__ == "__main__":
    main()
