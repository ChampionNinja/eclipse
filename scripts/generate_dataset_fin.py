"""
NutriAssist — Dataset Generation Script
Uses a hardcoded real-world packaged food database (no API calls).
Applies hard constraints (allergies, diet, conditions) + nutrient scoring.
Output: Oumi-compatible JSONL (conversations format).

Usage:
    python generate_dataset.py --count 500 --output data/training_data.jsonl
"""

import json
import random
import argparse
from pathlib import Path


# ===== SYSTEM PROMPT =====
SYSTEM_PROMPT = (
    "You are a nutrition analysis engine. Given product data and user profile, "
    "output ONLY a valid JSON verdict.\nRules:\n"
    "- verdict: \"eat\" (healthy), \"avoid\" (unhealthy), \"sometimes\" (moderate)\n"
    "- reason: max 15 words, cite specific nutrients\n"
    "- confidence: 0.0-1.0 based on data completeness\n"
    "- If user has allergies matching ingredients, verdict MUST be \"avoid\"\n"
    "- If user is vegetarian/vegan and product contains meat/fish, verdict MUST be \"avoid\"\n"
    "- Condition-specific flags (diabetes, hypertension, cholesterol, heart disease) "
    "must lower verdict if relevant nutrients are high"
)


# ===== ALLERGEN INFERENCE =====
ALLERGEN_KEYWORDS = {
    "milk":      ["milk", "dairy", "cream", "butter", "cheese", "whey",
                  "casein", "lactose", "paneer", "curd", "yogurt", "ghee"],
    "lactose":   ["milk", "dairy", "cream", "butter", "cheese", "whey",
                  "casein", "lactose", "paneer", "curd", "yogurt"],
    "gluten":    ["wheat", "flour", "maida", "semolina", "suji", "rava",
                  "barley", "rye", "bread", "biscuit", "oat"],
    "peanuts":   ["peanut", "groundnut"],
    "tree nuts": ["almond", "cashew", "walnut", "pistachio", "hazelnut",
                  "pecan", "macadamia"],
    "eggs":      ["egg", "albumin", "ovalbumin"],
    "soy":       ["soy", "soybean", "soya"],
    "shellfish": ["shrimp", "prawn", "crab", "lobster", "crayfish"],
    "fish":      ["fish", "tuna", "salmon", "sardine", "anchovy", "cod"],
    "sesame":    ["sesame", "tahini"],
}

NON_VEG_KEYWORDS = [
    "chicken", "beef", "pork", "lamb", "mutton", "fish", "tuna", "salmon",
    "sardine", "anchovy", "shrimp", "prawn", "crab", "lobster", "gelatin",
    "lard", "rennet", "meat",
]

ANIMAL_KEYWORDS = NON_VEG_KEYWORDS + [
    "milk", "dairy", "cream", "butter", "cheese", "whey", "casein",
    "lactose", "egg", "albumin", "honey", "beeswax",
]


def infer_allergens(ingredients: str) -> list:
    """Infer allergen tags from ingredient text."""
    ing = ingredients.lower()
    found = []
    for allergen, keywords in ALLERGEN_KEYWORDS.items():
        for kw in keywords:
            if kw in ing:
                tag = f"en:{allergen}"
                if tag not in found:
                    found.append(tag)
                break
    return found


def generate_barcode() -> str:
    """Generate a realistic 13-digit EAN barcode."""
    digits = [random.randint(0, 9) for _ in range(12)]
    # Compute check digit
    odd_sum = sum(digits[i] for i in range(0, 12, 2))
    even_sum = sum(digits[i] for i in range(1, 12, 2))
    check = (10 - ((odd_sum + even_sum * 3) % 10)) % 10
    digits.append(check)
    return "".join(str(d) for d in digits)


def compute_nutriscore(nutriments: dict) -> str:
    """Compute a simple Nutri-Score (A–E) based on nutrient profile."""
    sugar   = nutriments.get("energy_kcal_100g", 0) or 0
    fat     = nutriments.get("fat_100g", 0) or 0
    sat_fat = nutriments.get("saturated_fat_100g", 0) or 0
    sugars  = nutriments.get("sugars_100g", 0) or 0
    salt    = nutriments.get("salt_100g", 0) or 0
    protein = nutriments.get("proteins_100g", 0) or 0
    fiber   = nutriments.get("fiber_100g", 0) or 0
    kcal    = nutriments.get("energy_kcal_100g", 0) or 0

    negative = 0
    if kcal > 335: negative += 1
    if kcal > 670: negative += 1
    if sugars > 4.5: negative += 1
    if sugars > 9: negative += 1
    if sugars > 13.5: negative += 1
    if sat_fat > 1: negative += 1
    if sat_fat > 2: negative += 1
    if sat_fat > 4: negative += 1
    if salt > 0.45: negative += 1
    if salt > 0.9: negative += 1
    if salt > 1.35: negative += 1

    positive = 0
    if protein > 1.6: positive += 1
    if protein > 3.2: positive += 1
    if protein > 4.8: positive += 1
    if fiber > 0.7: positive += 1
    if fiber > 1.4: positive += 1
    if fiber > 2.8: positive += 1

    score = negative - positive

    if score <= -1: return "a"
    elif score <= 2: return "b"
    elif score <= 10: return "c"
    elif score <= 18: return "d"
    else: return "e"


def compute_nova_group(ingredients: str) -> int:
    """Estimate NOVA group based on ingredient complexity."""
    ing = ingredients.lower()
    ultra_markers = [
        "modified starch", "hydrolyzed", "isolate", "concentrate",
        "maltodextrin", "high fructose", "artificial flavor", "natural flavor",
        "emulsifier", "stabilizer", "preservative", "color", "dye",
        "carrageenan", "xanthan", "sucralose", "aspartame", "saccharin",
        "msg", "monosodium glutamate", "sodium benzoate", "citric acid",
        "taurine", "invert syrup", "glucose syrup", "corn syrup",
        "palm oil", "hydrogenated", "trans fat", "seasoning", "flavor",
    ]
    processed_markers = [
        "salt", "sugar", "vinegar", "oil", "yeast", "sodium",
    ]
    count_ultra = sum(1 for m in ultra_markers if m in ing)
    count_proc = sum(1 for m in processed_markers if m in ing)
    ingredient_count = len([x for x in ing.split(",") if x.strip()])

    if count_ultra >= 3 or ingredient_count >= 8:
        return 4
    elif count_ultra >= 1 or ingredient_count >= 5:
        return 3
    elif count_proc >= 2:
        return 2
    else:
        return 1


def compute_data_confidence(nutriments: dict) -> float:
    """Compute data confidence based on nutriment completeness."""
    fields = ["energy_kcal_100g", "fat_100g", "saturated_fat_100g",
              "sugars_100g", "salt_100g", "proteins_100g", "fiber_100g"]
    available = sum(1 for f in fields if nutriments.get(f) is not None)
    return round(min(0.95, 0.50 + (available / len(fields)) * 0.45), 2)


def build_product_record(raw: dict) -> dict:
    """Build a structured product record from raw food data."""
    nut = raw["nutriments"]
    ingredients = raw.get("ingredients", "")
    allergens = infer_allergens(ingredients)
    barcode = generate_barcode()
    nutriscore = compute_nutriscore(nut)
    nova_group = compute_nova_group(ingredients)
    data_confidence = compute_data_confidence(nut)

    return {
        "product_name": raw.get("product_name", "Unknown Product"),
        "barcode": barcode,
        "nutriments": {
            "energy_kcal_100g": nut.get("energy_kcal_100g"),
            "fat_100g":         nut.get("fat_100g"),
            "saturated_fat_100g": nut.get("saturated_fat_100g"),
            "sugars_100g":      nut.get("sugars_100g"),
            "salt_100g":        nut.get("salt_100g"),
            "proteins_100g":    nut.get("proteins_100g"),
            "fiber_100g":       nut.get("fiber_100g"),
        },
        "ingredients": ingredients,
        "nutriscore": nutriscore,
        "nova_group": nova_group,
        "allergens": allergens,
        "source_type": "barcode",
        "data_confidence": data_confidence,
    }


# ===== PACKAGED FOODS DATABASE =====
# Only industrial/packaged products. Nutrients per 100g.
_RAW_FOODS = [
    # ----- Beverages -----
    # low sugar + high salt
    {
        "product_name": "Coca-Cola Classic",
        "ingredients": "carbonated water, sugar, caramel color, phosphoric acid, natural flavor, caffeine",
        "nutriments": {"energy_kcal_100g": 42, "fat_100g": 0.0, "saturated_fat_100g": 0.0, "sugars_100g": 10.6, "salt_100g": 0.01, "proteins_100g": 0.0, "fiber_100g": 0.0},
    },
    {
        "product_name": "Coca-Cola Zero Sugar",
        "ingredients": "carbonated water, caramel color, phosphoric acid, aspartame, potassium benzoate, natural flavor, acesulfame potassium, caffeine",
        "nutriments": {"energy_kcal_100g": 1, "fat_100g": 0.0, "saturated_fat_100g": 0.0, "sugars_100g": 0.0, "salt_100g": 0.04, "proteins_100g": 0.0, "fiber_100g": 0.0},
    },
    {
        "product_name": "Pepsi Cola",
        "ingredients": "carbonated water, high fructose corn syrup, caramel color, phosphoric acid, natural flavor, caffeine, citric acid",
        "nutriments": {"energy_kcal_100g": 41, "fat_100g": 0.0, "saturated_fat_100g": 0.0, "sugars_100g": 10.8, "salt_100g": 0.01, "proteins_100g": 0.0, "fiber_100g": 0.0},
    },
    {
        "product_name": "Red Bull Energy Drink",
        "ingredients": "carbonated water, sucrose, glucose, citric acid, taurine, sodium bicarbonate, magnesium carbonate, caffeine, niacin, vitamin B6, pantothenic acid, vitamin B12",
        "nutriments": {"energy_kcal_100g": 45, "fat_100g": 0.0, "saturated_fat_100g": 0.0, "sugars_100g": 11.0, "salt_100g": 0.1, "proteins_100g": 0.0, "fiber_100g": 0.0},
    },
    {
        "product_name": "Monster Energy Original",
        "ingredients": "carbonated water, sucrose, glucose, citric acid, taurine, sodium citrate, color, natural flavor, caffeine, maltodextrin, niacin, B vitamins",
        "nutriments": {"energy_kcal_100g": 46, "fat_100g": 0.0, "saturated_fat_100g": 0.0, "sugars_100g": 11.0, "salt_100g": 0.17, "proteins_100g": 0.0, "fiber_100g": 0.0},
    },
    {
        "product_name": "Tropicana Orange Juice Tetra",
        "ingredients": "orange juice concentrate, water, sugar, citric acid, vitamin C, natural flavor",
        "nutriments": {"energy_kcal_100g": 45, "fat_100g": 0.1, "saturated_fat_100g": 0.0, "sugars_100g": 9.0, "salt_100g": 0.01, "proteins_100g": 0.7, "fiber_100g": 0.2},
    },
    {
        "product_name": "Minute Maid Apple Juice",
        "ingredients": "apple juice concentrate, water, citric acid, ascorbic acid, natural flavor",
        "nutriments": {"energy_kcal_100g": 48, "fat_100g": 0.0, "saturated_fat_100g": 0.0, "sugars_100g": 11.5, "salt_100g": 0.01, "proteins_100g": 0.1, "fiber_100g": 0.0},
    },
    {
        "product_name": "Lipton Iced Tea Lemon",
        "ingredients": "water, sugar, black tea, citric acid, natural lemon flavor, sodium benzoate",
        "nutriments": {"energy_kcal_100g": 30, "fat_100g": 0.0, "saturated_fat_100g": 0.0, "sugars_100g": 7.5, "salt_100g": 0.02, "proteins_100g": 0.0, "fiber_100g": 0.0},
    },
    {
        "product_name": "Gatorade Lemon-Lime",
        "ingredients": "water, sucrose, dextrose, citric acid, sodium chloride, sodium citrate, monopotassium phosphate, natural flavor, modified food starch, color",
        "nutriments": {"energy_kcal_100g": 26, "fat_100g": 0.0, "saturated_fat_100g": 0.0, "sugars_100g": 6.0, "salt_100g": 0.45, "proteins_100g": 0.0, "fiber_100g": 0.0},
    },
    # high sugar + low fat
    {
        "product_name": "Fanta Orange",
        "ingredients": "carbonated water, high fructose corn syrup, citric acid, sodium benzoate, natural flavor, modified food starch, glycerol ester of rosin, yellow 6",
        "nutriments": {"energy_kcal_100g": 51, "fat_100g": 0.0, "saturated_fat_100g": 0.0, "sugars_100g": 13.0, "salt_100g": 0.02, "proteins_100g": 0.0, "fiber_100g": 0.0},
    },
    {
        "product_name": "Sprite",
        "ingredients": "carbonated water, high fructose corn syrup, citric acid, natural flavor, sodium benzoate",
        "nutriments": {"energy_kcal_100g": 41, "fat_100g": 0.0, "saturated_fat_100g": 0.0, "sugars_100g": 10.4, "salt_100g": 0.03, "proteins_100g": 0.1, "fiber_100g": 0.0},
    },
    {
        "product_name": "Amul Lassi Sweet Carton",
        "ingredients": "milk, sugar, curd cultures, sodium citrate, stabilizer",
        "nutriments": {"energy_kcal_100g": 80, "fat_100g": 2.5, "saturated_fat_100g": 1.5, "sugars_100g": 13.0, "salt_100g": 0.1, "proteins_100g": 3.0, "fiber_100g": 0.0},
    },

    # ----- Chips & Snacks -----
    # high fat + low sugar
    {
        "product_name": "Lay's Classic Salted Chips",
        "ingredients": "potatoes, vegetable oil (sunflower, corn, or canola oil), salt",
        "nutriments": {"energy_kcal_100g": 536, "fat_100g": 35.0, "saturated_fat_100g": 10.0, "sugars_100g": 0.5, "salt_100g": 0.6, "proteins_100g": 6.0, "fiber_100g": 3.0},
    },
    {
        "product_name": "Lay's Sour Cream & Onion Chips",
        "ingredients": "potatoes, vegetable oil, sour cream powder, onion powder, salt, sugar, natural flavor, maltodextrin, artificial color",
        "nutriments": {"energy_kcal_100g": 540, "fat_100g": 34.0, "saturated_fat_100g": 9.5, "sugars_100g": 2.5, "salt_100g": 0.8, "proteins_100g": 6.0, "fiber_100g": 2.5},
    },
    {
        "product_name": "Pringles Original",
        "ingredients": "dried potatoes, vegetable oil, degerminated yellow corn flour, cornstarch, rice flour, maltodextrin, mono and diglycerides, salt",
        "nutriments": {"energy_kcal_100g": 524, "fat_100g": 33.0, "saturated_fat_100g": 9.0, "sugars_100g": 0.7, "salt_100g": 0.5, "proteins_100g": 5.0, "fiber_100g": 3.5},
    },
    {
        "product_name": "Haldiram's Bhujia Sev",
        "ingredients": "moth bean flour, gram flour, vegetable oil, spices, salt, citric acid",
        "nutriments": {"energy_kcal_100g": 537, "fat_100g": 32.0, "saturated_fat_100g": 9.0, "sugars_100g": 3.0, "salt_100g": 1.8, "proteins_100g": 12.0, "fiber_100g": 5.0},
    },
    {
        "product_name": "Kurkure Masala Munch",
        "ingredients": "rice meal, corn meal, vegetable oil, spices, salt, sugar, maltodextrin, artificial flavor, color",
        "nutriments": {"energy_kcal_100g": 520, "fat_100g": 28.0, "saturated_fat_100g": 7.0, "sugars_100g": 2.0, "salt_100g": 1.5, "proteins_100g": 6.0, "fiber_100g": 2.0},
    },
    # low sugar + high salt
    {
        "product_name": "Doritos Nacho Cheese",
        "ingredients": "corn flour, vegetable oil, cheddar cheese powder, salt, whey, maltodextrin, artificial flavor, monosodium glutamate, color",
        "nutriments": {"energy_kcal_100g": 486, "fat_100g": 23.0, "saturated_fat_100g": 5.5, "sugars_100g": 1.2, "salt_100g": 1.7, "proteins_100g": 7.5, "fiber_100g": 3.0},
    },
    {
        "product_name": "Cheetos Crunchy",
        "ingredients": "corn flour, vegetable oil, cheddar cheese, salt, monosodium glutamate, whey protein concentrate, artificial color, natural flavor",
        "nutriments": {"energy_kcal_100g": 540, "fat_100g": 33.0, "saturated_fat_100g": 6.5, "sugars_100g": 1.5, "salt_100g": 2.0, "proteins_100g": 6.0, "fiber_100g": 1.0},
    },
    {
        "product_name": "Ruffles Cheddar & Sour Cream",
        "ingredients": "potatoes, vegetable oil, cheddar cheese powder, sour cream, salt, maltodextrin, natural flavor, artificial color",
        "nutriments": {"energy_kcal_100g": 530, "fat_100g": 32.0, "saturated_fat_100g": 9.0, "sugars_100g": 2.0, "salt_100g": 1.9, "proteins_100g": 6.5, "fiber_100g": 2.5},
    },
    {
        "product_name": "PopCorners Flex Protein Chips",
        "ingredients": "corn, pea protein, soy protein isolate, vegetable oil, salt, natural flavor",
        "nutriments": {"energy_kcal_100g": 410, "fat_100g": 12.0, "saturated_fat_100g": 2.0, "sugars_100g": 1.0, "salt_100g": 1.3, "proteins_100g": 20.0, "fiber_100g": 4.0},
    },
    {
        "product_name": "Baked Lays Original",
        "ingredients": "potatoes, cornstarch, sugar, salt, soy lecithin",
        "nutriments": {"energy_kcal_100g": 390, "fat_100g": 6.0, "saturated_fat_100g": 1.5, "sugars_100g": 2.5, "salt_100g": 0.7, "proteins_100g": 5.5, "fiber_100g": 3.5},
    },

    # ----- Biscuits & Crackers -----
    {
        "product_name": "Britannia Marie Gold Biscuits",
        "ingredients": "wheat flour, sugar, vegetable oil, invert syrup, salt, sodium bicarbonate, artificial flavor",
        "nutriments": {"energy_kcal_100g": 450, "fat_100g": 12.0, "saturated_fat_100g": 5.0, "sugars_100g": 25.0, "salt_100g": 0.5, "proteins_100g": 6.0, "fiber_100g": 2.0},
    },
    {
        "product_name": "Parle-G Glucose Biscuits",
        "ingredients": "wheat flour, sugar, vegetable oil, glucose, milk solids, salt, sodium bicarbonate, artificial flavor",
        "nutriments": {"energy_kcal_100g": 447, "fat_100g": 12.0, "saturated_fat_100g": 6.0, "sugars_100g": 28.0, "salt_100g": 0.5, "proteins_100g": 7.0, "fiber_100g": 1.5},
    },
    {
        "product_name": "Oreo Original Cookies",
        "ingredients": "sugar, enriched flour, palm and canola oil, cocoa, high fructose corn syrup, leavening, cornstarch, salt, soy lecithin, natural flavor",
        "nutriments": {"energy_kcal_100g": 473, "fat_100g": 20.0, "saturated_fat_100g": 6.0, "sugars_100g": 40.0, "salt_100g": 0.65, "proteins_100g": 5.0, "fiber_100g": 2.0},
    },
    {
        "product_name": "Ritz Crackers",
        "ingredients": "enriched flour, vegetable oil, sugar, leavening, salt, high fructose corn syrup, soy lecithin",
        "nutriments": {"energy_kcal_100g": 480, "fat_100g": 23.0, "saturated_fat_100g": 5.5, "sugars_100g": 8.0, "salt_100g": 1.2, "proteins_100g": 6.0, "fiber_100g": 2.0},
    },
    {
        "product_name": "Digestive Biscuits (McVitie's)",
        "ingredients": "whole wheat flour, sugar, vegetable oil, glucose syrup, salt, sodium bicarbonate, natural flavor",
        "nutriments": {"energy_kcal_100g": 471, "fat_100g": 20.0, "saturated_fat_100g": 9.5, "sugars_100g": 17.0, "salt_100g": 0.6, "proteins_100g": 7.5, "fiber_100g": 5.5},
    },
    {
        "product_name": "Good Day Cashew Cookies (Britannia)",
        "ingredients": "wheat flour, sugar, vegetable oil, cashew, invert syrup, milk solids, salt, raising agent, dough conditioner, artificial flavor",
        "nutriments": {"energy_kcal_100g": 500, "fat_100g": 22.0, "saturated_fat_100g": 10.0, "sugars_100g": 26.0, "salt_100g": 0.4, "proteins_100g": 7.0, "fiber_100g": 1.5},
    },
    {
        "product_name": "Wheat Thins Crackers",
        "ingredients": "whole grain wheat flour, enriched flour, sugar, soybean oil, malt syrup, salt, soy lecithin, natural flavor",
        "nutriments": {"energy_kcal_100g": 400, "fat_100g": 10.0, "saturated_fat_100g": 2.0, "sugars_100g": 8.0, "salt_100g": 0.9, "proteins_100g": 7.0, "fiber_100g": 4.0},
    },
    # low sugar + high fat
    {
        "product_name": "Cream Crackers (Jacob's)",
        "ingredients": "wheat flour, vegetable oil, salt, sodium bicarbonate, calcium phosphate",
        "nutriments": {"energy_kcal_100g": 440, "fat_100g": 16.0, "saturated_fat_100g": 7.0, "sugars_100g": 3.5, "salt_100g": 1.5, "proteins_100g": 8.5, "fiber_100g": 2.5},
    },

    # ----- Instant Noodles & Pasta -----
    {
        "product_name": "Maggi 2-Minute Masala Noodles",
        "ingredients": "wheat flour, palm oil, salt, sugar, spices, onion powder, flavor enhancer, acidity regulator, color",
        "nutriments": {"energy_kcal_100g": 445, "fat_100g": 17.0, "saturated_fat_100g": 8.0, "sugars_100g": 3.0, "salt_100g": 2.4, "proteins_100g": 8.0, "fiber_100g": 2.0},
    },
    {
        "product_name": "Nissin Top Ramen Chicken",
        "ingredients": "wheat flour, palm oil, salt, chicken powder, soy sauce, sugar, monosodium glutamate, artificial flavor, color",
        "nutriments": {"energy_kcal_100g": 450, "fat_100g": 18.0, "saturated_fat_100g": 9.0, "sugars_100g": 2.5, "salt_100g": 2.8, "proteins_100g": 9.0, "fiber_100g": 1.5},
    },
    {
        "product_name": "Indomie Mi Goreng Instant Noodles",
        "ingredients": "wheat flour, palm oil, salt, sugar, soy sauce, garlic, onion, artificial flavor, color, monosodium glutamate",
        "nutriments": {"energy_kcal_100g": 460, "fat_100g": 20.0, "saturated_fat_100g": 10.0, "sugars_100g": 4.0, "salt_100g": 2.5, "proteins_100g": 8.5, "fiber_100g": 1.8},
    },
    {
        "product_name": "Cup Noodles Seafood",
        "ingredients": "wheat flour, palm oil, salt, seafood powder, sugar, soy sauce, monosodium glutamate, starch, artificial flavor, color",
        "nutriments": {"energy_kcal_100g": 430, "fat_100g": 16.0, "saturated_fat_100g": 7.5, "sugars_100g": 3.5, "salt_100g": 3.0, "proteins_100g": 8.0, "fiber_100g": 1.0},
    },
    {
        "product_name": "Knorr Soupy Noodles",
        "ingredients": "wheat flour, tapioca starch, palm oil, salt, sugar, soy sauce, flavor enhancer, artificial flavor, color",
        "nutriments": {"energy_kcal_100g": 420, "fat_100g": 14.0, "saturated_fat_100g": 6.5, "sugars_100g": 5.0, "salt_100g": 2.6, "proteins_100g": 7.5, "fiber_100g": 2.0},
    },
    {
        "product_name": "Barilla Pasta Penne",
        "ingredients": "semolina, wheat flour, water",
        "nutriments": {"energy_kcal_100g": 352, "fat_100g": 1.5, "saturated_fat_100g": 0.3, "sugars_100g": 3.5, "salt_100g": 0.01, "proteins_100g": 12.5, "fiber_100g": 3.0},
    },

    # ----- Chocolates & Sweets -----
    # high sugar + high fat
    {
        "product_name": "Cadbury Dairy Milk",
        "ingredients": "milk solids, sugar, cocoa butter, cocoa mass, vegetable fat, emulsifier (soy lecithin), artificial flavor",
        "nutriments": {"energy_kcal_100g": 534, "fat_100g": 30.0, "saturated_fat_100g": 18.0, "sugars_100g": 56.0, "salt_100g": 0.2, "proteins_100g": 7.0, "fiber_100g": 2.0},
    },
    {
        "product_name": "KitKat 4-Finger",
        "ingredients": "sugar, wheat flour, cocoa butter, skim milk powder, cocoa mass, vegetable fat, lactose, soy lecithin, artificial flavor",
        "nutriments": {"energy_kcal_100g": 518, "fat_100g": 27.0, "saturated_fat_100g": 15.0, "sugars_100g": 55.0, "salt_100g": 0.1, "proteins_100g": 6.5, "fiber_100g": 1.5},
    },
    {
        "product_name": "Snickers Bar",
        "ingredients": "milk chocolate, sugar, peanuts, corn syrup, skim milk powder, butter, palm oil, salt, egg whites, artificial flavor",
        "nutriments": {"energy_kcal_100g": 488, "fat_100g": 24.0, "saturated_fat_100g": 9.0, "sugars_100g": 48.0, "salt_100g": 0.35, "proteins_100g": 9.0, "fiber_100g": 1.5},
    },
    # high fat + lower sugar (dark chocolate)
    {
        "product_name": "Lindt Dark Chocolate 70%",
        "ingredients": "cocoa mass, sugar, cocoa butter, vanilla extract",
        "nutriments": {"energy_kcal_100g": 598, "fat_100g": 42.0, "saturated_fat_100g": 24.0, "sugars_100g": 24.0, "salt_100g": 0.01, "proteins_100g": 8.0, "fiber_100g": 11.0},
    },
    {
        "product_name": "Twix Chocolate Bar",
        "ingredients": "milk chocolate, sugar, enriched flour, palm oil, corn syrup, soy lecithin, salt, artificial flavor",
        "nutriments": {"energy_kcal_100g": 495, "fat_100g": 24.5, "saturated_fat_100g": 13.0, "sugars_100g": 54.0, "salt_100g": 0.3, "proteins_100g": 4.5, "fiber_100g": 1.0},
    },
    {
        "product_name": "M&Ms Milk Chocolate",
        "ingredients": "milk chocolate, sugar, corn starch, corn syrup, vegetable oil, soy lecithin, artificial color, natural flavor",
        "nutriments": {"energy_kcal_100g": 480, "fat_100g": 20.0, "saturated_fat_100g": 12.0, "sugars_100g": 67.0, "salt_100g": 0.15, "proteins_100g": 4.5, "fiber_100g": 1.5},
    },
    {
        "product_name": "Ferrero Rocher",
        "ingredients": "milk chocolate, sugar, hazelnuts, skim milk powder, palm oil, wheat flour, salt, soy lecithin, artificial flavor, cocoa powder",
        "nutriments": {"energy_kcal_100g": 578, "fat_100g": 39.0, "saturated_fat_100g": 14.0, "sugars_100g": 45.0, "salt_100g": 0.12, "proteins_100g": 8.5, "fiber_100g": 3.5},
    },

    # ----- Breakfast Cereals -----
    {
        "product_name": "Kellogg's Corn Flakes",
        "ingredients": "corn, sugar, malt extract, salt, vitamins, minerals",
        "nutriments": {"energy_kcal_100g": 357, "fat_100g": 0.4, "saturated_fat_100g": 0.1, "sugars_100g": 8.0, "salt_100g": 1.0, "proteins_100g": 7.0, "fiber_100g": 3.0},
    },
    # high sugar + low fat
    {
        "product_name": "Kellogg's Froot Loops",
        "ingredients": "sugar, corn flour, wheat flour, oat flour, salt, corn syrup, hydrogenated vegetable oil, maltodextrin, artificial color, natural flavor, vitamins, minerals",
        "nutriments": {"energy_kcal_100g": 380, "fat_100g": 4.0, "saturated_fat_100g": 0.5, "sugars_100g": 38.0, "salt_100g": 0.8, "proteins_100g": 5.0, "fiber_100g": 3.0},
    },
    {
        "product_name": "Quaker Instant Oatmeal Maple Brown Sugar",
        "ingredients": "whole grain rolled oats, sugar, brown sugar, natural flavor, salt, calcium carbonate, niacinamide, vitamin A palmitate",
        "nutriments": {"energy_kcal_100g": 370, "fat_100g": 6.0, "saturated_fat_100g": 1.0, "sugars_100g": 20.0, "salt_100g": 0.6, "proteins_100g": 11.0, "fiber_100g": 7.0},
    },
    {
        "product_name": "Honey Bunches of Oats",
        "ingredients": "corn, whole grain rolled oats, sugar, rice flour, corn syrup, honey, salt, palm oil, soy lecithin, malt flavoring, vitamins, minerals",
        "nutriments": {"energy_kcal_100g": 380, "fat_100g": 5.0, "saturated_fat_100g": 1.0, "sugars_100g": 15.0, "salt_100g": 0.75, "proteins_100g": 6.5, "fiber_100g": 4.5},
    },
    {
        "product_name": "Kellogg's Special K Protein",
        "ingredients": "rice, wheat gluten, sugar, wheat bran, soy protein isolate, salt, malt flavor, vitamins, minerals",
        "nutriments": {"energy_kcal_100g": 370, "fat_100g": 3.0, "saturated_fat_100g": 0.5, "sugars_100g": 14.0, "salt_100g": 0.9, "proteins_100g": 20.0, "fiber_100g": 5.0},
    },
    # high protein + low fat
    {
        "product_name": "Kashi GoLean Crisp Cereal",
        "ingredients": "whole grain wheat, soy protein isolate, oat fiber, soy grits, brown rice syrup, salt, natural flavor",
        "nutriments": {"energy_kcal_100g": 340, "fat_100g": 3.5, "saturated_fat_100g": 0.5, "sugars_100g": 10.0, "salt_100g": 0.5, "proteins_100g": 17.0, "fiber_100g": 10.0},
    },

    # ----- Frozen Foods -----
    {
        "product_name": "Frozen Cheese Pizza",
        "ingredients": "wheat flour, cheese, tomato sauce, vegetable oil, processed meat (pepperoni), yeast, sugar, salt, modified starch",
        "nutriments": {"energy_kcal_100g": 266, "fat_100g": 22.0, "saturated_fat_100g": 10.0, "sugars_100g": 4.0, "salt_100g": 1.8, "proteins_100g": 11.0, "fiber_100g": 2.0},
    },
    {
        "product_name": "McCain French Fries Frozen",
        "ingredients": "potatoes, vegetable oil (canola, sunflower), dextrose, sodium acid pyrophosphate, salt",
        "nutriments": {"energy_kcal_100g": 160, "fat_100g": 6.0, "saturated_fat_100g": 1.0, "sugars_100g": 0.5, "salt_100g": 0.3, "proteins_100g": 2.5, "fiber_100g": 2.5},
    },
    {
        "product_name": "Tyson Chicken Nuggets Frozen",
        "ingredients": "chicken, water, wheat flour, modified food starch, salt, sodium phosphate, vegetable oil, spices, artificial flavor",
        "nutriments": {"energy_kcal_100g": 296, "fat_100g": 20.0, "saturated_fat_100g": 5.0, "sugars_100g": 1.0, "salt_100g": 1.5, "proteins_100g": 15.0, "fiber_100g": 1.0},
    },
    {
        "product_name": "Lean Cuisine Chicken Frozen Meal",
        "ingredients": "chicken breast, water, rice, vegetables, sauce (soy sauce, salt, modified corn starch, sugar, natural flavor)",
        "nutriments": {"energy_kcal_100g": 105, "fat_100g": 2.5, "saturated_fat_100g": 0.5, "sugars_100g": 3.0, "salt_100g": 0.7, "proteins_100g": 10.0, "fiber_100g": 2.0},
    },
    {
        "product_name": "Amy's Veggie Burrito Frozen",
        "ingredients": "organic whole wheat tortilla, organic beans, organic cheese, organic salsa, organic vegetables, water, salt",
        "nutriments": {"energy_kcal_100g": 170, "fat_100g": 5.0, "saturated_fat_100g": 2.0, "sugars_100g": 2.0, "salt_100g": 0.6, "proteins_100g": 7.0, "fiber_100g": 4.0},
    },
    {
        "product_name": "Dr. Oetker Ristorante Margherita Pizza",
        "ingredients": "wheat flour, tomato puree, mozzarella cheese, vegetable oil, sugar, salt, yeast, modified starch, spices",
        "nutriments": {"energy_kcal_100g": 250, "fat_100g": 10.0, "saturated_fat_100g": 5.0, "sugars_100g": 3.5, "salt_100g": 1.4, "proteins_100g": 10.0, "fiber_100g": 2.0},
    },

    # ----- Protein Bars & Health Snacks -----
    # high protein + low fat
    {
        "product_name": "Quest Protein Bar Chocolate Chip",
        "ingredients": "protein blend (whey protein isolate, milk protein isolate), almonds, water, erythritol, soluble corn fiber, natural flavor, cocoa, sea salt, stevia",
        "nutriments": {"energy_kcal_100g": 371, "fat_100g": 14.0, "saturated_fat_100g": 4.0, "sugars_100g": 3.0, "salt_100g": 0.75, "proteins_100g": 42.0, "fiber_100g": 14.0},
    },
    {
        "product_name": "Clif Bar Energy Bar Chocolate Chip",
        "ingredients": "organic rolled oats, organic brown rice syrup, soy protein isolate, organic roasted soybeans, organic cane syrup, organic grain dextrin, organic soy flour, cocoa, salt",
        "nutriments": {"energy_kcal_100g": 380, "fat_100g": 7.0, "saturated_fat_100g": 1.5, "sugars_100g": 22.0, "salt_100g": 0.3, "proteins_100g": 10.0, "fiber_100g": 5.0},
    },
    {
        "product_name": "KIND Dark Chocolate Nut Bar",
        "ingredients": "almonds, peanuts, chicory root fiber, honey, dark chocolate (cocoa mass, sugar), rice flour, sea salt, soy lecithin",
        "nutriments": {"energy_kcal_100g": 460, "fat_100g": 30.0, "saturated_fat_100g": 5.0, "sugars_100g": 15.0, "salt_100g": 0.3, "proteins_100g": 12.0, "fiber_100g": 7.0},
    },
    {
        "product_name": "RXBar Chocolate Sea Salt",
        "ingredients": "dates, egg whites, cashews, almonds, cocoa, sea salt",
        "nutriments": {"energy_kcal_100g": 350, "fat_100g": 9.0, "saturated_fat_100g": 1.5, "sugars_100g": 22.0, "salt_100g": 0.4, "proteins_100g": 12.0, "fiber_100g": 5.0},
    },
    {
        "product_name": "Luna Bar Chocolate Dipped Coconut",
        "ingredients": "soy rice crisps, organic rolled oats, organic brown rice syrup, soy protein isolate, organic cane syrup, palm kernel oil, coconut, chocolate, salt, vitamins",
        "nutriments": {"energy_kcal_100g": 380, "fat_100g": 10.0, "saturated_fat_100g": 5.0, "sugars_100g": 20.0, "salt_100g": 0.25, "proteins_100g": 8.0, "fiber_100g": 3.0},
    },
    {
        "product_name": "Fulfil Chocolate Peanut Butter Protein Bar",
        "ingredients": "hydrolysed collagen, whey protein concentrate, peanuts, dark chocolate, sugar, peanut butter, cocoa, salt, stevia",
        "nutriments": {"energy_kcal_100g": 370, "fat_100g": 16.0, "saturated_fat_100g": 5.5, "sugars_100g": 14.0, "salt_100g": 0.5, "proteins_100g": 30.0, "fiber_100g": 8.0},
    },

    # ----- Spreads & Condiments (packaged) -----
    # high fat + low sugar
    {
        "product_name": "Nutella Hazelnut Spread",
        "ingredients": "sugar, palm oil, hazelnuts, cocoa, skim milk powder, soy lecithin, vanillin",
        "nutriments": {"energy_kcal_100g": 539, "fat_100g": 30.9, "saturated_fat_100g": 10.6, "sugars_100g": 57.5, "salt_100g": 0.107, "proteins_100g": 6.3, "fiber_100g": 3.4},
    },
    {
        "product_name": "Jif Creamy Peanut Butter",
        "ingredients": "roasted peanuts, sugar, hydrogenated vegetable oil, fully hydrogenated rapeseed oil, salt",
        "nutriments": {"energy_kcal_100g": 598, "fat_100g": 50.0, "saturated_fat_100g": 9.5, "sugars_100g": 8.0, "salt_100g": 0.6, "proteins_100g": 25.0, "fiber_100g": 5.0},
    },
    {
        "product_name": "Philadelphia Cream Cheese",
        "ingredients": "cream cheese (pasteurized milk and cream, whey protein concentrate, salt, carob bean gum, natamycin)",
        "nutriments": {"energy_kcal_100g": 342, "fat_100g": 34.0, "saturated_fat_100g": 21.0, "sugars_100g": 2.5, "salt_100g": 0.7, "proteins_100g": 6.0, "fiber_100g": 0.0},
    },
    # low sugar + high salt
    {
        "product_name": "Heinz Tomato Ketchup",
        "ingredients": "tomato concentrate, spirit vinegar, sugar, salt, spice and herb extracts, onion powder",
        "nutriments": {"energy_kcal_100g": 101, "fat_100g": 0.1, "saturated_fat_100g": 0.0, "sugars_100g": 22.0, "salt_100g": 2.0, "proteins_100g": 1.0, "fiber_100g": 0.8},
    },
    {
        "product_name": "Maggi Hot & Sweet Tomato Chilli Sauce",
        "ingredients": "tomato pulp, sugar, vinegar, red chilli, salt, modified starch, sodium benzoate, citric acid",
        "nutriments": {"energy_kcal_100g": 170, "fat_100g": 0.2, "saturated_fat_100g": 0.0, "sugars_100g": 38.0, "salt_100g": 1.9, "proteins_100g": 1.0, "fiber_100g": 1.0},
    },

    # ----- Canned & Packaged Ready-to-Eat -----
    {
        "product_name": "Campbell's Tomato Soup Can",
        "ingredients": "tomato puree, water, high fructose corn syrup, salt, citric acid, modified food starch, ascorbic acid, natural flavor",
        "nutriments": {"energy_kcal_100g": 50, "fat_100g": 0.2, "saturated_fat_100g": 0.0, "sugars_100g": 9.0, "salt_100g": 0.8, "proteins_100g": 1.5, "fiber_100g": 0.6},
    },
    {
        "product_name": "Heinz Baked Beans Can",
        "ingredients": "beans in tomato sauce (navy beans, water, tomato puree, sugar, modified starch, salt, spices, herb extract)",
        "nutriments": {"energy_kcal_100g": 73, "fat_100g": 0.2, "saturated_fat_100g": 0.0, "sugars_100g": 5.0, "salt_100g": 0.54, "proteins_100g": 4.7, "fiber_100g": 3.7},
    },
    {
        "product_name": "Starkist Tuna Canned in Water",
        "ingredients": "light tuna, water, vegetable broth, salt",
        "nutriments": {"energy_kcal_100g": 90, "fat_100g": 1.0, "saturated_fat_100g": 0.0, "sugars_100g": 0.0, "salt_100g": 0.85, "proteins_100g": 20.0, "fiber_100g": 0.0},
    },
    {
        "product_name": "Del Monte Diced Tomatoes Can",
        "ingredients": "tomatoes, tomato juice, calcium chloride, citric acid, salt",
        "nutriments": {"energy_kcal_100g": 20, "fat_100g": 0.0, "saturated_fat_100g": 0.0, "sugars_100g": 2.5, "salt_100g": 0.3, "proteins_100g": 1.0, "fiber_100g": 1.2},
    },
    {
        "product_name": "Libby's Pumpkin Canned",
        "ingredients": "pumpkin",
        "nutriments": {"energy_kcal_100g": 35, "fat_100g": 0.1, "saturated_fat_100g": 0.0, "sugars_100g": 3.5, "salt_100g": 0.0, "proteins_100g": 1.5, "fiber_100g": 2.5},
    },
    # ultra-processed
    {
        "product_name": "Spam Classic Canned Meat",
        "ingredients": "pork, water, salt, modified potato starch, sugar, sodium nitrite",
        "nutriments": {"energy_kcal_100g": 310, "fat_100g": 27.0, "saturated_fat_100g": 10.0, "sugars_100g": 0.5, "salt_100g": 3.3, "proteins_100g": 13.0, "fiber_100g": 0.0},
    },

    # ----- Ice Cream & Frozen Desserts -----
    {
        "product_name": "Häagen-Dazs Vanilla Ice Cream",
        "ingredients": "cream, skim milk, sugar, egg yolks, vanilla extract",
        "nutriments": {"energy_kcal_100g": 260, "fat_100g": 17.0, "saturated_fat_100g": 10.0, "sugars_100g": 24.0, "salt_100g": 0.1, "proteins_100g": 4.0, "fiber_100g": 0.0},
    },
    {
        "product_name": "Ben & Jerry's Chocolate Fudge Brownie",
        "ingredients": "cream, skim milk, liquid sugar, brownie pieces (wheat flour, sugar, cocoa, eggs), egg yolks, cocoa, guar gum, soy lecithin",
        "nutriments": {"energy_kcal_100g": 270, "fat_100g": 14.0, "saturated_fat_100g": 8.5, "sugars_100g": 28.0, "salt_100g": 0.15, "proteins_100g": 4.5, "fiber_100g": 1.5},
    },
    {
        "product_name": "Magnum Classic Ice Cream Bar",
        "ingredients": "cream, skim milk, sugar, cocoa butter, chocolate liquor, coconut oil, whey, soy lecithin, vanilla extract, carob bean gum",
        "nutriments": {"energy_kcal_100g": 290, "fat_100g": 19.0, "saturated_fat_100g": 13.0, "sugars_100g": 25.0, "salt_100g": 0.1, "proteins_100g": 4.0, "fiber_100g": 1.0},
    },

    # ----- Packaged Dairy -----
    # high protein + low fat
    {
        "product_name": "Chobani Plain Greek Yogurt 0%",
        "ingredients": "cultured nonfat milk (pasteurized skim milk, active yogurt cultures)",
        "nutriments": {"energy_kcal_100g": 59, "fat_100g": 0.4, "saturated_fat_100g": 0.0, "sugars_100g": 3.2, "salt_100g": 0.05, "proteins_100g": 10.0, "fiber_100g": 0.0},
    },
    {
        "product_name": "Yoplait Strawberry Yogurt",
        "ingredients": "cultured pasteurized low fat milk, sugar, strawberries, modified corn starch, natural flavor, gelatin, tricalcium phosphate, color",
        "nutriments": {"energy_kcal_100g": 100, "fat_100g": 1.5, "saturated_fat_100g": 1.0, "sugars_100g": 15.0, "salt_100g": 0.09, "proteins_100g": 4.5, "fiber_100g": 0.0},
    },
    {
        "product_name": "Kraft Singles American Cheese",
        "ingredients": "milk, water, whey, milk protein concentrate, milk fat, sodium citrate, calcium phosphate, salt, lactic acid, sorbic acid, artificial color, soy lecithin",
        "nutriments": {"energy_kcal_100g": 270, "fat_100g": 21.0, "saturated_fat_100g": 12.0, "sugars_100g": 2.0, "salt_100g": 1.7, "proteins_100g": 14.0, "fiber_100g": 0.0},
    },
    {
        "product_name": "Amul Processed Cheese Spread",
        "ingredients": "cheese, water, skim milk powder, salt, emulsifier, acidity regulator, artificial flavor",
        "nutriments": {"energy_kcal_100g": 275, "fat_100g": 22.0, "saturated_fat_100g": 14.0, "sugars_100g": 1.5, "salt_100g": 1.6, "proteins_100g": 12.0, "fiber_100g": 0.0},
    },
    {
        "product_name": "Yakult Probiotic Drink",
        "ingredients": "water, skim milk powder, sugar, glucose, flavoring, Lactobacillus casei Shirota",
        "nutriments": {"energy_kcal_100g": 67, "fat_100g": 0.0, "saturated_fat_100g": 0.0, "sugars_100g": 14.5, "salt_100g": 0.03, "proteins_100g": 1.3, "fiber_100g": 0.0},
    },

    # ----- Packaged Bread -----
    # moderate nutrition
    {
        "product_name": "Pepperidge Farm Whole Wheat Bread",
        "ingredients": "whole wheat flour, water, sugar, yeast, contains 2% or less of wheat gluten, soybean oil, salt, molasses, calcium propionate, soy lecithin, monoglycerides",
        "nutriments": {"energy_kcal_100g": 247, "fat_100g": 2.5, "saturated_fat_100g": 0.5, "sugars_100g": 5.0, "salt_100g": 0.8, "proteins_100g": 9.5, "fiber_100g": 6.5},
    },
    {
        "product_name": "Wonder Classic White Bread",
        "ingredients": "enriched flour, water, sugar, yeast, soybean oil, salt, dough conditioners, calcium sulfate, soy flour, DATEM, ethoxylated mono- and diglycerides, ascorbic acid",
        "nutriments": {"energy_kcal_100g": 265, "fat_100g": 3.2, "saturated_fat_100g": 0.7, "sugars_100g": 5.0, "salt_100g": 1.0, "proteins_100g": 8.0, "fiber_100g": 2.0},
    },

    # ----- Packaged Savory Snacks / Diverse -----
    # high protein + moderate fat
    {
        "product_name": "Jack Link's Beef Jerky Original",
        "ingredients": "beef, water, sugar, salt, corn syrup solids, hydrolyzed corn protein, sodium erythorbate, sodium nitrite, natural flavor",
        "nutriments": {"energy_kcal_100g": 286, "fat_100g": 7.0, "saturated_fat_100g": 2.5, "sugars_100g": 14.0, "salt_100g": 3.5, "proteins_100g": 44.0, "fiber_100g": 0.0},
    },
    # low sugar + low fat but high salt
    {
        "product_name": "Kettle Brand Sea Salt Popcorn",
        "ingredients": "popcorn, sunflower oil, sea salt",
        "nutriments": {"energy_kcal_100g": 440, "fat_100g": 20.0, "saturated_fat_100g": 2.0, "sugars_100g": 0.5, "salt_100g": 1.1, "proteins_100g": 9.0, "fiber_100g": 12.0},
    },
    {
        "product_name": "SkinnyPop Original Popcorn",
        "ingredients": "popcorn, sunflower oil, salt",
        "nutriments": {"energy_kcal_100g": 480, "fat_100g": 28.0, "saturated_fat_100g": 3.5, "sugars_100g": 0.5, "salt_100g": 0.8, "proteins_100g": 9.0, "fiber_100g": 15.0},
    },
    {
        "product_name": "Nature Valley Crunchy Oats & Honey Bar",
        "ingredients": "whole grain rolled oats, sugar, canola oil, yellow corn flour, honey, soy flour, salt, baking soda, soy lecithin",
        "nutriments": {"energy_kcal_100g": 430, "fat_100g": 12.0, "saturated_fat_100g": 1.0, "sugars_100g": 19.0, "salt_100g": 0.45, "proteins_100g": 8.0, "fiber_100g": 4.0},
    },
    {
        "product_name": "Quaker Chewy Granola Bar Chocolate Chip",
        "ingredients": "granola (whole grain rolled oats, brown sugar, crisp rice, corn syrup, partially hydrogenated soybean oil, salt), semisweet chocolate chips, soy lecithin",
        "nutriments": {"energy_kcal_100g": 380, "fat_100g": 9.0, "saturated_fat_100g": 2.5, "sugars_100g": 27.0, "salt_100g": 0.4, "proteins_100g": 5.0, "fiber_100g": 3.0},
    },
    {
        "product_name": "Rice Cakes Original (Quaker)",
        "ingredients": "whole grain brown rice, salt",
        "nutriments": {"energy_kcal_100g": 390, "fat_100g": 3.0, "saturated_fat_100g": 0.5, "sugars_100g": 0.5, "salt_100g": 0.35, "proteins_100g": 9.0, "fiber_100g": 3.5},
    },

    # ----- Diversity boosters: ultra-processed vs minimally processed -----
    # Ultra-processed: high everything
    {
        "product_name": "Hot Pockets Pepperoni Pizza",
        "ingredients": "enriched flour, pepperoni (pork, beef, salt, water, corn syrup, dextrose, spices, sodium nitrate), mozzarella cheese, tomato paste, vegetable oil, modified food starch, salt, sugar, artificial flavor",
        "nutriments": {"energy_kcal_100g": 280, "fat_100g": 12.0, "saturated_fat_100g": 5.5, "sugars_100g": 4.5, "salt_100g": 1.6, "proteins_100g": 10.0, "fiber_100g": 1.5},
    },
    {
        "product_name": "Lunchables Turkey & Cheddar",
        "ingredients": "turkey (water, salt, sodium phosphate, sodium erythorbate, sodium nitrite, dextrose), cheddar cheese (milk, salt, enzymes, artificial color), crackers (enriched flour, soybean oil, salt, high fructose corn syrup, artificial flavor)",
        "nutriments": {"energy_kcal_100g": 230, "fat_100g": 11.0, "saturated_fat_100g": 5.0, "sugars_100g": 5.0, "salt_100g": 2.0, "proteins_100g": 11.0, "fiber_100g": 1.0},
    },
    # Minimally processed: low everything
    {
        "product_name": "Unsalted Mixed Nuts (packaged)",
        "ingredients": "almonds, cashews, walnuts, peanuts",
        "nutriments": {"energy_kcal_100g": 607, "fat_100g": 55.0, "saturated_fat_100g": 7.0, "sugars_100g": 4.0, "salt_100g": 0.01, "proteins_100g": 20.0, "fiber_100g": 8.5},
    },
    {
        "product_name": "Kirkland Salted Mixed Nuts",
        "ingredients": "cashews, almonds, macadamia nuts, pecans, brazil nuts, salt",
        "nutriments": {"energy_kcal_100g": 620, "fat_100g": 58.0, "saturated_fat_100g": 8.5, "sugars_100g": 4.5, "salt_100g": 0.55, "proteins_100g": 18.0, "fiber_100g": 7.0},
    },
    # high sugar + very high fat (ultra-processed dessert)
    {
        "product_name": "Entenmann's Rich Frosted Donuts",
        "ingredients": "enriched flour, sugar, palm oil, eggs, water, yeast, corn syrup, skim milk, salt, dextrose, soy lecithin, artificial flavor, color",
        "nutriments": {"energy_kcal_100g": 420, "fat_100g": 22.0, "saturated_fat_100g": 10.0, "sugars_100g": 32.0, "salt_100g": 0.6, "proteins_100g": 5.0, "fiber_100g": 1.0},
    },
    # low fat + high sugar + high salt
    {
        "product_name": "Kellogg's Pop-Tarts Brown Sugar Cinnamon",
        "ingredients": "enriched flour, sugar, high fructose corn syrup, palm oil, dextrose, fructose, modified corn starch, salt, cinnamon, soy lecithin, artificial flavor, color",
        "nutriments": {"energy_kcal_100g": 387, "fat_100g": 9.0, "saturated_fat_100g": 2.5, "sugars_100g": 32.0, "salt_100g": 0.5, "proteins_100g": 4.5, "fiber_100g": 1.5},
    },
    # High protein + high fat
    {
        "product_name": "Optimum Nutrition Gold Standard Whey Protein",
        "ingredients": "whey protein isolate, whey protein concentrate, whey peptides, cocoa, lecithin, natural flavor, acesulfame potassium, aminogen",
        "nutriments": {"energy_kcal_100g": 373, "fat_100g": 7.0, "saturated_fat_100g": 2.0, "sugars_100g": 5.0, "salt_100g": 0.4, "proteins_100g": 75.0, "fiber_100g": 2.0},
    },
    # Very high salt + low sugar
    {
        "product_name": "Nissin Demae Ramen Spicy",
        "ingredients": "wheat flour, palm oil, salt, chili, soy sauce, garlic, sugar, monosodium glutamate, artificial flavor",
        "nutriments": {"energy_kcal_100g": 440, "fat_100g": 16.0, "saturated_fat_100g": 7.0, "sugars_100g": 2.0, "salt_100g": 3.5, "proteins_100g": 9.5, "fiber_100g": 1.5},
    },
    # Low calorie, balanced
    {
        "product_name": "Belvita Breakfast Biscuits Blueberry",
        "ingredients": "whole grain wheat flour, canola oil, brown sugar, sugar, blueberry filling, oat fiber, salt, baking soda, natural flavor",
        "nutriments": {"energy_kcal_100g": 430, "fat_100g": 12.0, "saturated_fat_100g": 1.5, "sugars_100g": 17.0, "salt_100g": 0.55, "proteins_100g": 8.5, "fiber_100g": 5.0},
    },
    # High sugar protein shake
    {
        "product_name": "Ensure Original Nutrition Shake Vanilla",
        "ingredients": "water, maltodextrin, sugar, milk protein concentrate, soy oil, soy protein isolate, canola oil, corn oil, vitamins, minerals, artificial flavor",
        "nutriments": {"energy_kcal_100g": 71, "fat_100g": 2.0, "saturated_fat_100g": 0.3, "sugars_100g": 9.0, "salt_100g": 0.13, "proteins_100g": 3.0, "fiber_100g": 0.0},
    },
]

# Build the structured FOODS list
FOODS = [build_product_record(raw) for raw in _RAW_FOODS]


# ===== USER PROFILES =====
PROFILES = [
    {"allergies": [],            "diet": "veg",     "conditions": []},
    {"allergies": [],            "diet": "veg",     "conditions": ["diabetes"]},
    {"allergies": [],            "diet": "veg",     "conditions": ["hypertension"]},
    {"allergies": [],            "diet": "veg",     "conditions": ["cholesterol"]},
    {"allergies": [],            "diet": "non-veg", "conditions": []},
    {"allergies": [],            "diet": "non-veg", "conditions": ["heart disease"]},
    {"allergies": [],            "diet": "non-veg", "conditions": ["diabetes"]},
    {"allergies": ["milk"],      "diet": "veg",     "conditions": []},
    {"allergies": ["gluten"],    "diet": "veg",     "conditions": []},
    {"allergies": ["peanuts"],   "diet": "veg",     "conditions": []},
    {"allergies": ["tree nuts"], "diet": "veg",     "conditions": []},
    {"allergies": ["eggs"],      "diet": "veg",     "conditions": []},
    {"allergies": ["soy"],       "diet": "non-veg", "conditions": []},
    {"allergies": ["gluten"],    "diet": "veg",     "conditions": ["diabetes"]},
    {"allergies": ["milk"],      "diet": "veg",     "conditions": ["cholesterol"]},
    {"allergies": [],            "diet": "vegan",   "conditions": []},
    {"allergies": [],            "diet": "vegan",   "conditions": ["diabetes"]},
    {"allergies": [],            "diet": "veg",     "conditions": ["PCOS"]},
    {"allergies": [],            "diet": "veg",     "conditions": ["menopause"]},
    {"allergies": [],            "diet": "non-veg", "conditions": []},
]


# ===== HARD CONSTRAINT CHECK =====
def check_hard_constraints(ingredients: str, profile: dict):
    """Returns (verdict, reason) if violated, else (None, None)."""
    ing = ingredients.lower()

    for allergy in profile.get("allergies", []):
        keywords = ALLERGEN_KEYWORDS.get(allergy.lower(), [allergy.lower()])
        for kw in keywords:
            if kw in ing:
                return "avoid", f"Contains {kw} — matches your {allergy} allergy"

    diet = profile.get("diet", "").lower()
    if diet in ("veg", "vegetarian", "vegan"):
        for kw in NON_VEG_KEYWORDS:
            if kw in ing:
                return "avoid", f"Contains {kw} — not suitable for {diet} diet"

    if diet == "vegan":
        for kw in ANIMAL_KEYWORDS:
            if kw in ing:
                return "avoid", f"Contains {kw} — not suitable for vegan diet"

    return None, None


# ===== NUTRIENT SCORING =====
def compute_nutrient_score(nutriments: dict, profile: dict):
    """Returns (score, reasons_list)."""
    sugar   = nutriments.get("sugars_100g")
    fat     = nutriments.get("fat_100g")
    sat_fat = nutriments.get("saturated_fat_100g")
    salt    = nutriments.get("salt_100g")
    protein = nutriments.get("proteins_100g")
    fiber   = nutriments.get("fiber_100g")

    score = 0
    reasons = []

    if sugar is not None:
        if sugar > 20:   score -= 3; reasons.append(f"very high sugar ({sugar}g)")
        elif sugar > 10: score -= 2; reasons.append(f"high sugar ({sugar}g)")
        elif sugar > 5:  score -= 1; reasons.append(f"moderate sugar ({sugar}g)")
        elif sugar < 3:  score += 1; reasons.append("low sugar")

    if fat is not None:
        if fat > 25:   score -= 2; reasons.append(f"very high fat ({fat}g)")
        elif fat > 15: score -= 1; reasons.append(f"high fat ({fat}g)")
        elif fat < 3:  score += 1; reasons.append("low fat")

    if sat_fat is not None:
        if sat_fat > 10: score -= 2; reasons.append(f"high saturated fat ({sat_fat}g)")
        elif sat_fat > 5: score -= 1

    if salt is not None:
        if salt > 1.5:   score -= 2; reasons.append(f"high salt ({salt}g)")
        elif salt > 1.0: score -= 1; reasons.append(f"moderate salt ({salt}g)")

    if protein is not None:
        if protein > 15: score += 2; reasons.append(f"high protein ({protein}g)")
        elif protein > 8: score += 1; reasons.append(f"good protein ({protein}g)")

    if fiber is not None:
        if fiber > 5:   score += 2; reasons.append(f"high fiber ({fiber}g)")
        elif fiber > 3: score += 1; reasons.append(f"good fiber ({fiber}g)")

    conditions = [c.lower() for c in profile.get("conditions", [])]

    if "diabetes" in conditions and sugar is not None and sugar > 8:
        score -= 2; reasons.append("risky for diabetes")

    if "hypertension" in conditions and salt is not None and salt > 1.0:
        score -= 2; reasons.append("risky for hypertension")

    if any(c in conditions for c in ("cholesterol", "heart disease")):
        if sat_fat is not None and sat_fat > 5:
            score -= 2; reasons.append("bad for cholesterol/heart")
        if fat is not None and fat > 20:
            score -= 1

    if "heart disease" in conditions and salt is not None and salt > 1.0:
        score -= 1; reasons.append("high salt risky for heart")

    if any(c in conditions for c in ("pcos", "menopause")):
        if sugar is not None and sugar > 10:
            score -= 1; reasons.append("high sugar not ideal for hormonal conditions")
        if fiber is not None and fiber > 3:
            score += 1

    return score, reasons


# ===== FULL VERDICT =====
def compute_verdict(product: dict, profile: dict):
    """Compute verdict for a structured product record."""
    nutriments = product.get("nutriments", {})
    ingredients = product.get("ingredients", "")

    if not any(v is not None for v in nutriments.values()):
        return None

    hard_verdict, hard_reason = check_hard_constraints(ingredients, profile)
    if hard_verdict:
        return {"verdict": hard_verdict, "reason": hard_reason, "confidence": 0.98}

    score, reasons = compute_nutrient_score(nutriments, profile)
    confidence = product.get("data_confidence", 0.7)

    if score >= 2:    verdict = "eat"
    elif score <= -2: verdict = "avoid"
    else:             verdict = "sometimes"

    reason = ", ".join(reasons[:3]) if reasons else "moderate nutritional profile"
    words = reason.split()
    if len(words) > 15:
        reason = " ".join(words[:15])
    reason = reason[0].upper() + reason[1:] if reason else "Moderate nutritional profile"

    return {"verdict": verdict, "reason": reason, "confidence": confidence}


# ===== SAMPLE BUILDER =====
def build_sample(product: dict, profile: dict, verdict: dict) -> dict:
    """Build a training sample using structured product JSON."""
    product_section = json.dumps(product, ensure_ascii=False, indent=2)

    user_content = f"""[PRODUCT]
{product_section}

[USER PROFILE]
Allergies:  {', '.join(profile.get('allergies', [])) or 'none'}
Conditions: {', '.join(profile.get('conditions', [])) or 'none'}
Diet:       {profile.get('diet') or 'none'}

[OUTPUT] Respond with ONLY valid JSON:"""

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": json.dumps(verdict)},
        ]
    }


# ===== MAIN =====
def main():
    parser = argparse.ArgumentParser(description="Generate NutriAssist training data")
    parser.add_argument("--count",      type=int, default=500,
                        help="Target number of samples to generate")
    parser.add_argument("--output",     type=str, default="data/training_data.jsonl")
    parser.add_argument("--merge-seed", type=str, default="data/training_data_seed.jsonl",
                        help="Optional path to hand-curated seed dataset to merge")
    args = parser.parse_args()

    all_samples = []

    # 1. Load optional seed data
    seed_path = Path(args.merge_seed)
    if seed_path.exists():
        with open(seed_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_samples.append(json.loads(line))
        print(f"Loaded {len(all_samples)} seed samples from {seed_path}")
    else:
        print(f"No seed file at {seed_path}, skipping.")

    # 2. Generate samples by randomly pairing foods x profiles
    print(f"Generating {args.count} samples from local packaged food database "
          f"({len(FOODS)} products x {len(PROFILES)} profiles)...")
    generated = 0
    attempts  = 0
    max_attempts = args.count * 10

    while generated < args.count and attempts < max_attempts:
        attempts += 1
        product = random.choice(FOODS)
        profile = random.choice(PROFILES)

        verdict = compute_verdict(product, profile)
        if verdict is None:
            continue

        all_samples.append(build_sample(product, profile, verdict))
        generated += 1

    print(f"Generated {generated} samples.")

    # 3. Shuffle
    random.shuffle(all_samples)

    # 4. Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Total: {len(all_samples)} samples written to {output_path}")

    # 5. Distribution stats
    verdicts = {"eat": 0, "avoid": 0, "sometimes": 0}
    nova_dist = {1: 0, 2: 0, 3: 0, 4: 0}
    nutriscore_dist = {"a": 0, "b": 0, "c": 0, "d": 0, "e": 0}

    for s in all_samples:
        for msg in s["messages"]:
            if msg["role"] == "assistant":
                try:
                    v = json.loads(msg["content"])
                    verdicts[v["verdict"]] = verdicts.get(v["verdict"], 0) + 1
                except Exception:
                    pass
            if msg["role"] == "user":
                try:
                    # Extract product JSON from user content
                    content = msg["content"]
                    start = content.index("{")
                    end = content.rindex("}") + 1
                    product_data = json.loads(content[start:end])
                    nova = product_data.get("nova_group")
                    ns = product_data.get("nutriscore")
                    if nova in nova_dist:
                        nova_dist[nova] += 1
                    if ns in nutriscore_dist:
                        nutriscore_dist[ns] += 1
                except Exception:
                    pass

    print(f"Verdict distribution:    {verdicts}")
    print(f"NOVA group distribution: {nova_dist}")
    print(f"Nutri-Score distribution:{nutriscore_dist}")


if __name__ == "__main__":
    main()