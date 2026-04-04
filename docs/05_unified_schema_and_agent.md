# STEP 5: Unified Data Format + STEP 6: Product Analysis Agent

## Unified Schema (Pydantic)

```python
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class SourceType(str, Enum):
    BARCODE = "barcode"
    INFERRED = "inferred"

class DataConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class Nutriments(BaseModel):
    energy_kcal_100g: Optional[float] = None
    fat_100g: Optional[float] = None
    saturated_fat_100g: Optional[float] = None
    sugars_100g: Optional[float] = None
    salt_100g: Optional[float] = None
    proteins_100g: Optional[float] = None
    fiber_100g: Optional[float] = None

class UserProfile(BaseModel):
    allergies: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    diet_type: Optional[str] = None  # vegetarian, vegan, keto
    goal: Optional[str] = None  # weight_loss, muscle_gain

class UnifiedProduct(BaseModel):
    product_name: str
    nutriments: Nutriments = Field(default_factory=Nutriments)
    ingredients: str = ""
    source_type: SourceType = SourceType.INFERRED
    data_confidence: DataConfidence = DataConfidence.LOW
    nutriscore: Optional[str] = None
    nova_group: Optional[int] = None
    allergens: list[str] = Field(default_factory=list)
    user_profile: UserProfile = Field(default_factory=UserProfile)
```

## Flow Mapping Table

| Flow | source_type | data_confidence | Fields Filled |
|------|------------|----------------|---------------|
| Barcode (OFF hit) | barcode | high | 90%+ |
| Local DB match | inferred | high | 80%+ curated |
| OFF search match | inferred | medium | Variable |
| Fallback/unknown | inferred | low | Mostly empty |

## Missing Fields Handling

- `None` values rendered as "not available" in prompt text
- Model trained to output lower confidence when data is sparse
- Response templates add caveats: "Based on limited data..."

---

## STEP 6: Product Analysis Agent

### Contract

**Input**: UnifiedProduct as structured text
**Output**: JSON verdict

```json
{"verdict": "eat", "reason": "High protein, low sugar, good fiber content", "confidence": 0.92}
```

### What the Model Learns

1. **Nutrient thresholds**: sugar > 15g = bad, protein > 10g = good, etc.
2. **Allergen matching**: user allergies vs product allergens/ingredients
3. **Condition awareness**: diabetes + high sugar = avoid
4. **Confidence calibration**: less data = lower confidence score
5. **Consistent format**: always output valid JSON with exact 3 fields

### Handling Incomplete Data

- Missing all nutriments + low confidence → `{"verdict": "sometimes", "reason": "Insufficient data for definitive assessment", "confidence": 0.3}`
- Missing some fields → model uses available fields, notes gaps in reason
- The model NEVER hallucinates nutrient values; it works only with what's given

### Consistency Enforcement

1. **Constrained output format**: prompt demands JSON only, response parsed with `json.loads()`
2. **Regex fallback parser**: if JSON parse fails, extract verdict/reason via regex
3. **Temperature = 0**: deterministic outputs during inference
4. **Validation layer**: Pydantic model validates SLM output before returning

### Prompt Construction Code

```python
def build_analysis_prompt(product: dict, user_profile: dict) -> str:
    nut = product.get("nutriments", {})
    
    nutrient_lines = []
    for key, label in [
        ("energy_kcal_100g", "Calories"),
        ("fat_100g", "Fat"),
        ("saturated_fat_100g", "Saturated Fat"),
        ("sugars_100g", "Sugar"),
        ("salt_100g", "Salt"),
        ("proteins_100g", "Protein"),
        ("fiber_100g", "Fiber"),
    ]:
        val = nut.get(key)
        nutrient_lines.append(f"  {label}: {val if val is not None else 'not available'}")
    
    nutrients_str = "\n".join(nutrient_lines)
    
    allergies = ", ".join(user_profile.get("allergies", [])) or "none"
    conditions = ", ".join(user_profile.get("conditions", [])) or "none"
    
    prompt = f"""[SYSTEM] You are a nutrition analysis engine. Output ONLY valid JSON.
Rules:
- verdict: "eat" (healthy), "avoid" (unhealthy), "sometimes" (moderate)
- reason: max 15 words, cite specific nutrients
- confidence: 0.0-1.0 based on data completeness
- If user has allergies matching ingredients, verdict MUST be "avoid"
- If data_confidence is "low", confidence must be <= 0.5

[PRODUCT]
Name: {product['product_name']}
Source: {product.get('source_type', 'inferred')}
Data Confidence: {product.get('data_confidence', 'low')}
Nutrients (per 100g):
{nutrients_str}
Ingredients: {product.get('ingredients', 'not available')}

[USER PROFILE]
Allergies: {allergies}
Conditions: {conditions}
Diet: {user_profile.get('diet_type', 'none')}
Goal: {user_profile.get('goal', 'none')}

[OUTPUT] Respond with ONLY valid JSON:"""
    
    return prompt
```

### Output Parser

```python
import json
import re

class VerdictParser:
    VERDICT_REGEX = re.compile(
        r'"verdict"\s*:\s*"(eat|avoid|sometimes)".*'
        r'"reason"\s*:\s*"([^"]+)".*'
        r'"confidence"\s*:\s*([\d.]+)',
        re.DOTALL
    )
    
    def parse(self, raw_output: str) -> dict:
        # Try JSON parse first
        try:
            # Extract JSON from output (handle model preamble)
            json_match = re.search(r'\{[^{}]+\}', raw_output)
            if json_match:
                result = json.loads(json_match.group())
                if all(k in result for k in ['verdict', 'reason', 'confidence']):
                    result['verdict'] = result['verdict'].lower()
                    result['confidence'] = float(result['confidence'])
                    return result
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Regex fallback
        match = self.VERDICT_REGEX.search(raw_output)
        if match:
            return {
                "verdict": match.group(1),
                "reason": match.group(2),
                "confidence": float(match.group(3))
            }
        
        # Ultimate fallback
        return {
            "verdict": "sometimes",
            "reason": "Unable to analyze - please try again",
            "confidence": 0.3
        }
```
