# STEP 7: Dataset Design

## Training Data Structure

Each sample is an instruction-response pair:

```json
{
  "instruction": "[PRODUCT]\nName: ...\nNutrients: ...\n[USER PROFILE]\n...\n[OUTPUT]",
  "response": "{\"verdict\": \"...\", \"reason\": \"...\", \"confidence\": ...}"
}
```

## 15 High-Quality Training Examples

### Example 1: Coca-Cola (Avoid — high sugar)
```json
{
  "instruction": "[PRODUCT]\nName: Coca-Cola Classic\nSource: barcode\nData Confidence: high\nNutrients (per 100g):\n  Calories: 42\n  Fat: 0\n  Saturated Fat: 0\n  Sugar: 10.6\n  Salt: 0\n  Protein: 0\n  Fiber: 0\nIngredients: carbonated water, sugar, colour (caramel E150d), phosphoric acid, natural flavourings including caffeine\n\n[USER PROFILE]\nAllergies: none\nConditions: none\nDiet: none\nGoal: weight_loss\n\n[OUTPUT]",
  "response": "{\"verdict\": \"avoid\", \"reason\": \"Very high sugar (10.6g/100ml), zero nutrition, bad for weight loss\", \"confidence\": 0.95}"
}
```

### Example 2: Greek Yogurt (Eat — high protein)
```json
{
  "instruction": "[PRODUCT]\nName: Greek Yogurt Plain\nSource: barcode\nData Confidence: high\nNutrients (per 100g):\n  Calories: 97\n  Fat: 5.0\n  Saturated Fat: 3.2\n  Sugar: 3.6\n  Salt: 0.1\n  Protein: 9.0\n  Fiber: 0\nIngredients: pasteurised milk, live cultures\n\n[USER PROFILE]\nAllergies: none\nConditions: none\nDiet: none\nGoal: muscle_gain\n\n[OUTPUT]",
  "response": "{\"verdict\": \"eat\", \"reason\": \"High protein (9g), low sugar, excellent for muscle gain\", \"confidence\": 0.93}"
}
```

### Example 3: Maggi Instant Noodles (Avoid — high sodium)
```json
{
  "instruction": "[PRODUCT]\nName: Maggi 2-Minute Noodles Masala\nSource: barcode\nData Confidence: high\nNutrients (per 100g):\n  Calories: 390\n  Fat: 15.0\n  Saturated Fat: 6.7\n  Sugar: 2.1\n  Salt: 2.9\n  Protein: 8.2\n  Fiber: 1.8\nIngredients: wheat flour, palm oil, salt, sugar, spices, MSG\n\n[USER PROFILE]\nAllergies: none\nConditions: hypertension\nDiet: none\nGoal: none\n\n[OUTPUT]",
  "response": "{\"verdict\": \"avoid\", \"reason\": \"Very high salt (2.9g), contains MSG, dangerous for hypertension\", \"confidence\": 0.94}"
}
```

### Example 4: Oats (Eat — healthy)
```json
{
  "instruction": "[PRODUCT]\nName: Rolled Oats\nSource: inferred\nData Confidence: high\nNutrients (per 100g):\n  Calories: 379\n  Fat: 6.5\n  Saturated Fat: 1.1\n  Sugar: 0.99\n  Salt: 0.01\n  Protein: 13.2\n  Fiber: 10.1\nIngredients: 100% whole grain oats\n\n[USER PROFILE]\nAllergies: none\nConditions: diabetes\nDiet: none\nGoal: weight_loss\n\n[OUTPUT]",
  "response": "{\"verdict\": \"eat\", \"reason\": \"High fiber (10.1g), high protein, very low sugar, great for diabetes\", \"confidence\": 0.95}"
}
```

### Example 5: Nutella (Sometimes — mixed signals)
```json
{
  "instruction": "[PRODUCT]\nName: Nutella\nSource: barcode\nData Confidence: high\nNutrients (per 100g):\n  Calories: 539\n  Fat: 30.9\n  Saturated Fat: 10.6\n  Sugar: 56.3\n  Salt: 0.107\n  Protein: 6.3\n  Fiber: 0\nIngredients: sugar, palm oil, hazelnuts 13%, cocoa, skimmed milk powder, lecithin, vanillin\n\n[USER PROFILE]\nAllergies: none\nConditions: none\nDiet: none\nGoal: none\n\n[OUTPUT]",
  "response": "{\"verdict\": \"avoid\", \"reason\": \"Extremely high sugar (56.3g), high saturated fat, mostly sugar and oil\", \"confidence\": 0.96}"
}
```

### Example 6: Paneer (Eat — allergen-aware)
```json
{
  "instruction": "[PRODUCT]\nName: Paneer\nSource: inferred\nData Confidence: high\nNutrients (per 100g):\n  Calories: 265\n  Fat: 20.8\n  Saturated Fat: 13.0\n  Sugar: 1.2\n  Salt: 0.02\n  Protein: 18.3\n  Fiber: 0\nIngredients: milk, citric acid\n\n[USER PROFILE]\nAllergies: none\nConditions: none\nDiet: vegetarian\nGoal: muscle_gain\n\n[OUTPUT]",
  "response": "{\"verdict\": \"eat\", \"reason\": \"High protein (18.3g), low sugar, great vegetarian protein source\", \"confidence\": 0.92}"
}
```

### Example 7: Paneer (AVOID — lactose allergy)
```json
{
  "instruction": "[PRODUCT]\nName: Paneer\nSource: inferred\nData Confidence: high\nNutrients (per 100g):\n  Calories: 265\n  Fat: 20.8\n  Saturated Fat: 13.0\n  Sugar: 1.2\n  Salt: 0.02\n  Protein: 18.3\n  Fiber: 0\nIngredients: milk, citric acid\n\n[USER PROFILE]\nAllergies: lactose, milk\nConditions: none\nDiet: none\nGoal: none\n\n[OUTPUT]",
  "response": "{\"verdict\": \"avoid\", \"reason\": \"Contains milk - matches your lactose/milk allergy\", \"confidence\": 0.98}"
}
```

### Example 8: Unknown Food — Low confidence
```json
{
  "instruction": "[PRODUCT]\nName: Local Street Samosa\nSource: inferred\nData Confidence: low\nNutrients (per 100g):\n  Calories: not available\n  Fat: not available\n  Saturated Fat: not available\n  Sugar: not available\n  Salt: not available\n  Protein: not available\n  Fiber: not available\nIngredients: not available\n\n[USER PROFILE]\nAllergies: none\nConditions: none\nDiet: none\nGoal: weight_loss\n\n[OUTPUT]",
  "response": "{\"verdict\": \"sometimes\", \"reason\": \"Insufficient data; samosas are typically deep-fried, limit if losing weight\", \"confidence\": 0.35}"
}
```

### Example 9: Brown Rice (Eat — diabetic-friendly)
```json
{
  "instruction": "[PRODUCT]\nName: Brown Rice\nSource: inferred\nData Confidence: high\nNutrients (per 100g):\n  Calories: 112\n  Fat: 0.83\n  Saturated Fat: 0.17\n  Sugar: 0.35\n  Salt: 0.005\n  Protein: 2.32\n  Fiber: 1.8\nIngredients: brown rice\n\n[USER PROFILE]\nAllergies: none\nConditions: diabetes\nDiet: none\nGoal: none\n\n[OUTPUT]",
  "response": "{\"verdict\": \"eat\", \"reason\": \"Low sugar, good fiber, low GI grain suitable for diabetes\", \"confidence\": 0.90}"
}
```

### Example 10: Lay's Chips (Avoid)
```json
{
  "instruction": "[PRODUCT]\nName: Lay's Classic Salted Chips\nSource: barcode\nData Confidence: high\nNutrients (per 100g):\n  Calories: 536\n  Fat: 35.0\n  Saturated Fat: 3.0\n  Sugar: 0.5\n  Salt: 1.8\n  Protein: 6.0\n  Fiber: 4.4\nIngredients: potatoes, vegetable oil, salt\n\n[USER PROFILE]\nAllergies: none\nConditions: hypertension\nDiet: none\nGoal: weight_loss\n\n[OUTPUT]",
  "response": "{\"verdict\": \"avoid\", \"reason\": \"Very high fat (35g), high salt (1.8g), bad for hypertension and weight loss\", \"confidence\": 0.94}"
}
```

### Example 11: Almonds (Eat)
```json
{
  "instruction": "[PRODUCT]\nName: Raw Almonds\nSource: inferred\nData Confidence: high\nNutrients (per 100g):\n  Calories: 579\n  Fat: 49.9\n  Saturated Fat: 3.7\n  Sugar: 4.4\n  Salt: 0.01\n  Protein: 21.2\n  Fiber: 12.5\nIngredients: almonds\n\n[USER PROFILE]\nAllergies: none\nConditions: none\nDiet: keto\nGoal: none\n\n[OUTPUT]",
  "response": "{\"verdict\": \"eat\", \"reason\": \"High protein (21g), excellent fiber (12.5g), healthy fats, perfect for keto\", \"confidence\": 0.94}"
}
```

### Example 12: Almonds (AVOID — tree nut allergy)
```json
{
  "instruction": "[PRODUCT]\nName: Raw Almonds\nSource: inferred\nData Confidence: high\nNutrients (per 100g):\n  Calories: 579\n  Fat: 49.9\n  Saturated Fat: 3.7\n  Sugar: 4.4\n  Salt: 0.01\n  Protein: 21.2\n  Fiber: 12.5\nIngredients: almonds\n\n[USER PROFILE]\nAllergies: tree nuts\nConditions: none\nDiet: none\nGoal: none\n\n[OUTPUT]",
  "response": "{\"verdict\": \"avoid\", \"reason\": \"Almonds are tree nuts - matches your tree nut allergy\", \"confidence\": 0.99}"
}
```

### Example 13: White Bread (Sometimes)
```json
{
  "instruction": "[PRODUCT]\nName: White Bread\nSource: inferred\nData Confidence: medium\nNutrients (per 100g):\n  Calories: 265\n  Fat: 3.2\n  Saturated Fat: 0.7\n  Sugar: 5.0\n  Salt: 1.1\n  Protein: 9.0\n  Fiber: 2.7\nIngredients: wheat flour, water, yeast, salt, sugar\n\n[USER PROFILE]\nAllergies: none\nConditions: diabetes\nDiet: none\nGoal: weight_loss\n\n[OUTPUT]",
  "response": "{\"verdict\": \"sometimes\", \"reason\": \"High GI, moderate salt; limit portions for diabetes and weight loss\", \"confidence\": 0.82}"
}
```

### Example 14: Protein Bar (Sometimes — partial data)
```json
{
  "instruction": "[PRODUCT]\nName: Unknown Protein Bar\nSource: inferred\nData Confidence: medium\nNutrients (per 100g):\n  Calories: 350\n  Fat: not available\n  Saturated Fat: not available\n  Sugar: 18.0\n  Salt: not available\n  Protein: 20.0\n  Fiber: not available\nIngredients: not available\n\n[USER PROFILE]\nAllergies: none\nConditions: none\nDiet: none\nGoal: muscle_gain\n\n[OUTPUT]",
  "response": "{\"verdict\": \"sometimes\", \"reason\": \"Good protein (20g) but high sugar (18g); check full ingredients\", \"confidence\": 0.60}"
}
```

### Example 15: Dal (Lentils — Eat)
```json
{
  "instruction": "[PRODUCT]\nName: Moong Dal (Yellow Lentils)\nSource: inferred\nData Confidence: high\nNutrients (per 100g):\n  Calories: 348\n  Fat: 1.2\n  Saturated Fat: 0.3\n  Sugar: 0\n  Salt: 0.01\n  Protein: 24.5\n  Fiber: 16.3\nIngredients: moong dal\n\n[USER PROFILE]\nAllergies: none\nConditions: diabetes\nDiet: vegetarian\nGoal: weight_loss\n\n[OUTPUT]",
  "response": "{\"verdict\": \"eat\", \"reason\": \"Excellent protein (24.5g), very high fiber, zero sugar, ideal for all goals\", \"confidence\": 0.96}"
}
```

## Labeling Logic (Decision Rules)

| Condition | Verdict |
|-----------|---------|
| Sugar > 15g/100g | avoid |
| Sugar > 8g AND < 15g | sometimes |
| Sugar < 5g | positive signal |
| Protein > 15g/100g | eat signal |
| Fat > 20g/100g | negative signal |
| Saturated Fat > 10g/100g | avoid signal |
| Salt > 1.5g/100g | avoid signal |
| Fiber > 5g/100g | eat signal |
| Allergen match with user | always avoid |
| Condition conflict (e.g., diabetes + sugar) | always avoid |
| data_confidence = low | confidence ≤ 0.5 |
| Missing > 50% fields | confidence ≤ 0.6 |

## Dataset Size Recommendations

| Phase | Samples | Quality |
|-------|---------|---------|
| **MVP** | 500-1000 | Hand-curated, diverse |
| **Production** | 3000-5000 | Includes edge cases |
| **Advanced** | 10000+ | Augmented + validated |

### Generation Strategy

1. **Seed**: 200 hand-written examples (like above)
2. **Augment**: Use OFF database to pull 2000+ real products, apply labeling rules programmatically
3. **Validate**: Human review of 10% random samples
4. **Edge cases**: Deliberately create partial-data, allergy-conflict, and ambiguous examples (20% of dataset)
