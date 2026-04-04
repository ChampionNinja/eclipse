# STEP 4: Non-Barcode Flow — Food Resolver

## Pipeline

```
User: "is maggi healthy?"
  │
  ▼  Extract food entity: "maggi"
  │
  ▼  1. Exact match in local_foods.json → confidence="high"
  │
  ▼  2. Fuzzy match (rapidfuzz, score > 80) → confidence="medium"
  │
  ▼  3. OFF Search API → confidence="medium"
  │
  ▼  4. Fallback: generic estimate → confidence="low"
  │
  ▼  Normalize → UnifiedProduct schema (source_type="inferred")
```

## Food Resolver Code

```python
import json
from rapidfuzz import fuzz, process
from typing import Optional, Dict, List

class FoodResolver:
    def __init__(self, local_db_path: str = "data/local_foods.json"):
        with open(local_db_path) as f:
            self._local_db: List[dict] = json.load(f)
        self._name_index = {item["name"].lower(): item for item in self._local_db}
        self._names = list(self._name_index.keys())

    async def resolve(self, food_text: str) -> Dict:
        food_text = food_text.lower().strip()

        # 1. Exact match
        if food_text in self._name_index:
            return self._to_schema(self._name_index[food_text], "high")

        # 2. Fuzzy match
        match = process.extractOne(food_text, self._names, scorer=fuzz.ratio)
        if match and match[1] > 80:
            return self._to_schema(self._name_index[match[0]], "medium")

        # 3. OFF Search
        off_result = await self._search_off(food_text)
        if off_result:
            return off_result

        # 4. Fallback
        return {
            "product_name": food_text.title(),
            "nutriments": {},
            "ingredients": "",
            "source_type": "inferred",
            "data_confidence": "low",
        }

    async def _search_off(self, query: str) -> Optional[Dict]:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(
                    "https://world.openfoodfacts.org/cgi/search.pl",
                    params={"search_terms": query, "search_simple": 1,
                            "action": "process", "json": 1, "page_size": 1}
                )
                products = resp.json().get("products", [])
                if products:
                    p = products[0]
                    nut = p.get("nutriments", {})
                    return {
                        "product_name": p.get("product_name", query.title()),
                        "nutriments": {
                            "energy_kcal_100g": nut.get("energy-kcal_100g"),
                            "sugars_100g": nut.get("sugars_100g"),
                            "fat_100g": nut.get("fat_100g"),
                            "proteins_100g": nut.get("proteins_100g"),
                            "fiber_100g": nut.get("fiber_100g"),
                            "salt_100g": nut.get("salt_100g"),
                            "saturated_fat_100g": nut.get("saturated-fat_100g"),
                        },
                        "ingredients": p.get("ingredients_text", ""),
                        "source_type": "inferred",
                        "data_confidence": "medium",
                    }
        except Exception:
            pass
        return None

    def _to_schema(self, item: dict, confidence: str) -> dict:
        return {
            "product_name": item["name"],
            "nutriments": item.get("nutriments", {}),
            "ingredients": item.get("ingredients", ""),
            "source_type": "inferred",
            "data_confidence": confidence,
        }
```

## Ambiguity Handling

| Scenario | Strategy |
|----------|----------|
| Multiple fuzzy matches | Pick highest score; tie → ask user |
| "Milk" (ambiguous type) | Ask: "Which type — dairy, almond, or oat?" |
| Missing nutrition data | Set confidence="low", add caveats |
| Unknown food | Fallback response: "I don't have data for that" |
