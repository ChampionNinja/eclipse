# STEP 3: Barcode Flow

## Pipeline

```
User: "scan 8901058847116"
  │
  ▼  Rules Layer extracts barcode
  │
  ▼  Cache check (in-memory dict)
  │
  ▼  [miss] OFF API: GET /api/v2/product/{barcode}
  │         Timeout: 2s, Retry: 1x
  │
  ▼  Extract: product_name, nutriments, ingredients,
  │           nutriscore, nova_group, allergens
  │
  ▼  Normalize → UnifiedProduct schema
  │  source_type="barcode", confidence="high"
  │
  ▼  Store in session.current_product + cache (TTL 1h)
  │
  ▼  SLM Analysis → verdict + reason
  │
  ▼  Format response → TTS
```

## OFF API Client

```python
import httpx
from typing import Optional, Dict, Any

class OFFClient:
    BASE_URL = "https://world.openfoodfacts.org/api/v2/product"

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=2.0)
        self._cache: Dict[str, dict] = {}

    async def get_product(self, barcode: str) -> Optional[Dict[str, Any]]:
        if barcode in self._cache:
            return self._cache[barcode]

        try:
            resp = await self._client.get(
                f"{self.BASE_URL}/{barcode}",
                params={"fields": "product_name,nutriments,ingredients_text,"
                                  "nutriscore_grade,nova_group,allergens_tags"}
            )
            data = resp.json()
            if data.get("status") != 1:
                return None

            product = data["product"]
            normalized = self._normalize(product, barcode)
            self._cache[barcode] = normalized
            return normalized
        except (httpx.TimeoutException, httpx.HTTPError):
            return None

    def _normalize(self, product: dict, barcode: str) -> dict:
        nut = product.get("nutriments", {})
        return {
            "product_name": product.get("product_name", "Unknown Product"),
            "barcode": barcode,
            "nutriments": {
                "energy_kcal_100g": nut.get("energy-kcal_100g"),
                "fat_100g": nut.get("fat_100g"),
                "saturated_fat_100g": nut.get("saturated-fat_100g"),
                "sugars_100g": nut.get("sugars_100g"),
                "salt_100g": nut.get("salt_100g"),
                "proteins_100g": nut.get("proteins_100g"),
                "fiber_100g": nut.get("fiber_100g"),
            },
            "ingredients": product.get("ingredients_text", ""),
            "nutriscore": product.get("nutriscore_grade"),
            "nova_group": product.get("nova_group"),
            "allergens": product.get("allergens_tags", []),
            "source_type": "barcode",
            "data_confidence": "high",
        }
```

## Follow-Up Queries

Once stored in session, queries like "how much sugar?" or "is it good for diabetics?"
resolve against `session.current_product` without re-fetching data.
