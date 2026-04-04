"""
NutriAssist — Dataset Augmentation Script
Generates training samples from Open Food Facts API.
Applies deterministic labeling rules to create verdicts.
Output: Oumi-compatible JSONL (conversations format).

Usage:
    python generate_dataset.py --count 500 --output data/training_data_augmented.jsonl
"""

import json
import random
import argparse
import time
from pathlib import Path
from typing import Optional

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    import urllib.request
    HAS_HTTPX = False


# ===== SYSTEM PROMPT (same across all samples) =====
SYSTEM_PROMPT = (
    "You are a nutrition analysis engine. Given product data and user profile, "
    "output ONLY a valid JSON verdict.\nRules:\n"
    "- verdict: \"eat\" (healthy), \"avoid\" (unhealthy), \"sometimes\" (moderate)\n"
    "- reason: max 15 words, cite specific nutrients\n"
    "- confidence: 0.0-1.0 based on data completeness\n"
    "- If user has allergies matching ingredients, verdict MUST be \"avoid\"\n"
    "- If data_confidence is \"low\", confidence must be <= 0.5"
)

# ===== USER PROFILE TEMPLATES =====
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
    {"allergies": [], "conditions": [], "diet_type": "", "goal": ""},  # no constraints
    {"allergies": [], "conditions": [], "diet_type": "", "goal": "maintenance"},
]

# ===== ALLERGEN KEYWORDS (for matching) =====
ALLERGEN_KEYWORDS = {
    "milk": ["milk", "dairy", "cream", "butter", "cheese", "whey", "casein", "lactose", "paneer", "curd", "yogurt", "ghee"],
    "lactose": ["milk", "dairy", "cream", "butter", "cheese", "whey", "casein", "lactose", "paneer", "curd", "yogurt"],
    "gluten": ["wheat", "flour", "maida", "semolina", "suji", "rava", "barley", "rye", "bread", "biscuit"],
    "peanuts": ["peanut", "groundnut"],
    "tree nuts": ["almond", "cashew", "walnut", "pistachio", "hazelnut", "pecan", "macadamia"],
    "eggs": ["egg", "albumin", "ovalbumin"],
    "soy": ["soy", "soybean", "soya"],
    "shellfish": ["shrimp", "prawn", "crab", "lobster", "crayfish"],
}


def fetch_products_from_off(page: int = 1, page_size: int = 50) -> list:
    """Fetch products from Open Food Facts API."""
    url = "https://world.openfoodfacts.org/cgi/search.pl"
    params = {
        "action": "process",
        "json": "1",
        "page": page,
        "page_size": page_size,
        "fields": "product_name,nutriments,ingredients_text,nutriscore_grade,nova_group,allergens_tags",
        "sort_by": "unique_scans_n",  # popular products first
    }

    if HAS_HTTPX:
        resp = httpx.get(url, params=params, timeout=10)
        data = resp.json()
    else:
        query = "&".join(f"{k}={v}" for k, v in params.items())
        full_url = f"{url}?{query}"
        with urllib.request.urlopen(full_url, timeout=10) as resp:
            data = json.loads(resp.read().decode())

    return data.get("products", [])


def extract_nutriments(product: dict) -> dict:
    """Extract and normalize nutriment values."""
    nut = product.get("nutriments", {})
    return {
        "energy_kcal_100g": nut.get("energy-kcal_100g"),
        "fat_100g": nut.get("fat_100g"),
        "saturated_fat_100g": nut.get("saturated-fat_100g"),
        "sugars_100g": nut.get("sugars_100g"),
        "salt_100g": nut.get("salt_100g"),
        "proteins_100g": nut.get("proteins_100g"),
        "fiber_100g": nut.get("fiber_100g"),
    }


def compute_verdict(nutriments: dict, ingredients: str, profile: dict) -> Optional[dict]:
    """Apply deterministic rules to generate verdict."""
    ingredients_lower = (ingredients or "").lower()

    # 1. CHECK ALLERGENS FIRST (highest priority)
    for allergy in profile.get("allergies", []):
        allergy_lower = allergy.lower()
        keywords = ALLERGEN_KEYWORDS.get(allergy_lower, [allergy_lower])
        for kw in keywords:
            if kw in ingredients_lower:
                return {
                    "verdict": "avoid",
                    "reason": f"Contains {kw} — matches your {allergy} allergy",
                    "confidence": 0.98,
                }

    # 2. GET NUTRIENT VALUES (with defaults)
    sugar = nutriments.get("sugars_100g")
    fat = nutriments.get("fat_100g")
    sat_fat = nutriments.get("saturated_fat_100g")
    salt = nutriments.get("salt_100g")
    protein = nutriments.get("proteins_100g")
    fiber = nutriments.get("fiber_100g")
    calories = nutriments.get("energy_kcal_100g")

    # Count available fields for confidence
    available = sum(1 for v in [sugar, fat, sat_fat, salt, protein, fiber, calories] if v is not None)
    if available == 0:
        return None  # skip — no data at all

    base_confidence = min(0.95, 0.5 + (available / 7) * 0.45)

    # 3. SCORING SYSTEM
    score = 0  # positive = healthy, negative = unhealthy
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
        if sat_fat > 10:
            score -= 2; reasons.append(f"high saturated fat ({sat_fat}g)")
        elif sat_fat > 5:
            score -= 1

    if salt is not None:
        if salt > 1.5:
            score -= 2; reasons.append(f"high salt ({salt}g)")
        elif salt > 1.0:
            score -= 1; reasons.append(f"moderate salt ({salt}g)")

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

    # 4. CONDITION-SPECIFIC ADJUSTMENTS
    conditions = profile.get("conditions", [])
    goal = profile.get("goal", "")

    if "diabetes" in conditions and sugar is not None and sugar > 8:
        score -= 2
        reasons.append("risky for diabetes")

    if "hypertension" in conditions and salt is not None and salt > 1.0:
        score -= 2
        reasons.append("risky for hypertension")

    if "cholesterol" in conditions and sat_fat is not None and sat_fat > 5:
        score -= 2
        reasons.append("bad for cholesterol")

    if goal == "weight_loss" and calories is not None and calories > 400:
        score -= 1
        reasons.append("high calorie for weight loss")

    if goal == "muscle_gain" and protein is not None and protein > 10:
        score += 1
        reasons.append("good for muscle gain")

    # 5. DETERMINE VERDICT
    if score >= 2:
        verdict = "eat"
    elif score <= -2:
        verdict = "avoid"
    else:
        verdict = "sometimes"

    # Build reason (max 15 words, pick top 2-3)
    reason = ", ".join(reasons[:3]) if reasons else "moderate nutritional profile"
    # Truncate to ~15 words
    words = reason.split()
    if len(words) > 15:
        reason = " ".join(words[:15])

    # Capitalize first letter
    reason = reason[0].upper() + reason[1:] if reason else "Moderate nutritional profile"

    return {
        "verdict": verdict,
        "reason": reason,
        "confidence": round(base_confidence, 2),
    }


def format_nutrient_line(value, label):
    """Format a single nutrient line."""
    if value is not None:
        return f"  {label}: {value}"
    return f"  {label}: not available"


def build_sample(product_name: str, nutriments: dict, ingredients: str,
                 source_type: str, data_confidence: str, profile: dict,
                 verdict: dict) -> dict:
    """Build a single Oumi conversations-format sample."""

    user_content = f"""[PRODUCT]
Name: {product_name}
Source: {source_type}
Data Confidence: {data_confidence}
Nutrients (per 100g):
{format_nutrient_line(nutriments.get('energy_kcal_100g'), 'Calories')}
{format_nutrient_line(nutriments.get('fat_100g'), 'Fat')}
{format_nutrient_line(nutriments.get('saturated_fat_100g'), 'Saturated Fat')}
{format_nutrient_line(nutriments.get('sugars_100g'), 'Sugar')}
{format_nutrient_line(nutriments.get('salt_100g'), 'Salt')}
{format_nutrient_line(nutriments.get('proteins_100g'), 'Protein')}
{format_nutrient_line(nutriments.get('fiber_100g'), 'Fiber')}
Ingredients: {ingredients or 'not available'}

[USER PROFILE]
Allergies: {', '.join(profile.get('allergies', [])) or 'none'}
Conditions: {', '.join(profile.get('conditions', [])) or 'none'}
Diet: {profile.get('diet_type') or 'none'}
Goal: {profile.get('goal') or 'none'}

[OUTPUT] Respond with ONLY valid JSON:"""

    assistant_content = json.dumps(verdict)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def generate_from_off(target_count: int) -> list:
    """Generate samples by fetching from OFF API."""
    samples = []
    page = 1
    seen_names = set()

    print(f"Generating {target_count} samples from Open Food Facts...")

    while len(samples) < target_count:
        try:
            products = fetch_products_from_off(page=page, page_size=50)
            if not products:
                print(f"No more products at page {page}")
                break

            for product in products:
                if len(samples) >= target_count:
                    break

                name = product.get("product_name", "").strip()
                if not name or name in seen_names or len(name) < 3:
                    continue
                seen_names.add(name)

                nutriments = extract_nutriments(product)
                ingredients = product.get("ingredients_text", "") or ""

                # Generate 1-3 samples per product with different profiles
                num_profiles = random.randint(1, 3)
                selected_profiles = random.sample(PROFILES, min(num_profiles, len(PROFILES)))

                for profile in selected_profiles:
                    if len(samples) >= target_count:
                        break

                    verdict = compute_verdict(nutriments, ingredients, profile)
                    if verdict is None:
                        continue

                    source = "barcode"
                    confidence = "high"

                    sample = build_sample(
                        product_name=name,
                        nutriments=nutriments,
                        ingredients=ingredients[:300],  # truncate long ingredients
                        source_type=source,
                        data_confidence=confidence,
                        profile=profile,
                        verdict=verdict,
                    )
                    samples.append(sample)

            page += 1
            time.sleep(0.5)  # rate limiting
            print(f"  Page {page-1}: {len(samples)}/{target_count} samples")

        except Exception as e:
            print(f"  Error on page {page}: {e}")
            page += 1
            time.sleep(2)
            continue

    return samples


def generate_partial_data_samples(count: int = 50) -> list:
    """Generate samples with missing nutrient data (low/medium confidence)."""
    partial_foods = [
        "Street Vada Pav", "Homemade Khichdi", "Local Jalebi", "Pav Bhaji",
        "Aloo Paratha", "Chole Bhature", "Puri", "Misal Pav", "Dhokla",
        "Kachori", "Dabeli", "Bhel Puri", "Sev Puri", "Ragda Pattice",
        "Egg Bhurji", "Butter Chicken", "Dal Makhani", "Aloo Gobi",
        "Baingan Bharta", "Paneer Tikka", "Tandoori Chicken", "Fish Curry",
        "Prawn Masala", "Mutton Curry", "Chicken Shawarma", "Frankie Roll",
        "Mysore Pak", "Barfi", "Rasgulla", "Kheer",
        "Lassi (Sweet)", "Chaas (Buttermilk)", "Nimbu Pani", "Sugarcane Juice",
        "Filter Coffee with Milk", "Masala Chai", "Green Tea",
        "Sambar", "Rasam", "Kadhi", "Undhiyu", "Ven Pongal",
        "Medu Vada", "Bonda", "Pakora", "Onion Bhaji",
        "Gajar Halwa", "Sooji Halwa", "Rava Ladoo", "Modak",
    ]

    samples = []
    for i in range(min(count, len(partial_foods))):
        food = partial_foods[i]
        profile = random.choice(PROFILES)

        # Randomly null out most fields
        nutriments = {
            "energy_kcal_100g": random.choice([None, random.randint(80, 500)]),
            "fat_100g": random.choice([None, None, round(random.uniform(1, 30), 1)]),
            "saturated_fat_100g": None,
            "sugars_100g": random.choice([None, round(random.uniform(0, 40), 1)]),
            "salt_100g": None,
            "proteins_100g": random.choice([None, round(random.uniform(1, 20), 1)]),
            "fiber_100g": None,
        }

        verdict = compute_verdict(nutriments, "", profile)
        if verdict is None:
            # Force a low-confidence response
            verdict = {
                "verdict": "sometimes",
                "reason": "Insufficient nutrition data for definitive assessment",
                "confidence": 0.35,
            }

        # Low confidence for partial data
        verdict["confidence"] = min(verdict["confidence"], 0.50)

        sample = build_sample(
            product_name=food,
            nutriments=nutriments,
            ingredients="not available",
            source_type="inferred",
            data_confidence="low",
            profile=profile,
            verdict=verdict,
        )
        samples.append(sample)

    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate NutriAssist training data")
    parser.add_argument("--count", type=int, default=500, help="Target number of OFF samples")
    parser.add_argument("--output", type=str, default="data/training_data_augmented.jsonl")
    parser.add_argument("--merge-seed", type=str, default="data/training_data.jsonl",
                        help="Path to seed dataset to merge with")
    args = parser.parse_args()

    all_samples = []

    # 1. Load seed data (hand-curated)
    seed_path = Path(args.merge_seed)
    if seed_path.exists():
        with open(seed_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_samples.append(json.loads(line))
        print(f"Loaded {len(all_samples)} seed samples from {seed_path}")

    # 2. Generate from OFF API
    off_samples = generate_from_off(args.count)
    all_samples.extend(off_samples)
    print(f"Generated {len(off_samples)} samples from OFF API")

    # 3. Generate partial-data samples
    partial_samples = generate_partial_data_samples(50)
    all_samples.extend(partial_samples)
    print(f"Generated {len(partial_samples)} partial-data samples")

    # 4. Shuffle
    random.shuffle(all_samples)

    # 5. Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(all_samples)} samples written to {output_path}")

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
