import json
import random

# ===== BASE FOODS (EXPAND THIS OVER TIME) =====
FOODS = [
    {
        "name": "Apple",
        "ingredients": "apple",
        "nutrients": {"sugar": 10, "fat": 0.2, "salt": 0.01, "protein": 0.3, "fiber": 2.4},
    },
    {
        "name": "Potato Chips",
        "ingredients": "potatoes, vegetable oil, salt",
        "nutrients": {"sugar": 0.5, "fat": 35, "salt": 0.6, "protein": 6, "fiber": 3},
    },
    {
        "name": "Greek Yogurt",
        "ingredients": "milk, cultures",
        "nutrients": {"sugar": 4, "fat": 0.4, "salt": 0.05, "protein": 10, "fiber": 0},
    },
    {
        "name": "Chicken Breast",
        "ingredients": "chicken",
        "nutrients": {"sugar": 0, "fat": 3.5, "salt": 0.07, "protein": 31, "fiber": 0},
    },
    {
        "name": "Chocolate Bar",
        "ingredients": "sugar, cocoa butter, milk",
        "nutrients": {"sugar": 50, "fat": 30, "salt": 0.2, "protein": 5, "fiber": 2},
    },
]

# ===== PROFILES =====
PROFILES = [
    {"allergies": [], "diet": "veg", "conditions": []},
    {"allergies": [], "diet": "non-veg", "conditions": ["diabetes"]},
    {"allergies": ["dairy"], "diet": "veg", "conditions": []},
    {"allergies": [], "diet": "veg", "conditions": ["pcos"]},
    {"allergies": [], "diet": "non-veg", "conditions": ["heart disease"]},
]

# ===== HARD RULES =====
def violates_rules(ingredients, profile):
    ing = ingredients.lower()

    if "dairy" in profile["allergies"] and "milk" in ing:
        return True

    if profile["diet"] == "veg" and "chicken" in ing:
        return True

    return False


# ===== SCORING =====
def compute_score(n):
    score = 0

    if n["sugar"] > 20: score -= 3
    elif n["sugar"] > 10: score -= 2
    elif n["sugar"] < 5: score += 1

    if n["fat"] > 25: score -= 2
    elif n["fat"] < 5: score += 1

    if n["salt"] > 1.5: score -= 2

    if n["protein"] > 10: score += 2

    if n["fiber"] > 3: score += 1

    return score


# ===== CONDITION MODIFIERS =====
def adjust(score, n, profile):
    if "diabetes" in profile["conditions"] and n["sugar"] > 8:
        score -= 2

    if "heart disease" in profile["conditions"]:
        if n["fat"] > 20: score -= 2
        if n["salt"] > 1: score -= 2

    if "pcos" in profile["conditions"] and n["sugar"] > 10:
        score -= 2

    return score


def verdict(score):
    if score >= 2: return "eat"
    if score <= -2: return "avoid"
    return "sometimes"


def confidence(n):
    return 0.9  # always high (clean data)


# ===== VARIATION =====
def vary(n):
    return {
        k: round(v * random.uniform(0.9, 1.1), 2)
        for k, v in n.items()
    }


# ===== GENERATE =====
def generate(count=500, output="hybrid_clean.jsonl"):
    data = []

    for _ in range(count):
        food = random.choice(FOODS)
        profile = random.choice(PROFILES)

        nutrients = vary(food["nutrients"])

        if violates_rules(food["ingredients"], profile):
            v = "avoid"
        else:
            s = compute_score(nutrients)
            s = adjust(s, nutrients, profile)
            v = verdict(s)

        sample = {
            "messages": [
                {"role": "system", "content": "Output JSON"},
                {"role": "user", "content": json.dumps({
                    "product": food["name"],
                    "nutrients": nutrients,
                    "ingredients": food["ingredients"],
                    "profile": profile,
                })},
                {"role": "assistant", "content": json.dumps({
                    "verdict": v,
                    "reason": "Based on nutrient profile",
                    "confidence": confidence(nutrients),
                })}
            ]
        }

        data.append(sample)

    with open(output, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

    print(f"Generated {len(data)} samples")


if __name__ == "__main__":
    generate()