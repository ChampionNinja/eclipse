"""Barcode flow — Open Food Facts API client."""

import httpx
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class OFFClient:
    BASE_URL = "https://world.openfoodfacts.org/api/v2/product"

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        self._cache: Dict[str, dict] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=3.0)
        return self._client

    async def get_product(self, barcode: str) -> Optional[Dict[str, Any]]:
        # Cache check
        if barcode in self._cache:
            return self._cache[barcode]

        try:
            client = await self._get_client()
            resp = await client.get(
                f"{self.BASE_URL}/{barcode}",
                params={
                    "fields": "product_name,nutriments,ingredients_text,"
                              "nutriscore_grade,nova_group,allergens_tags"
                },
            )
            data = resp.json()
            if data.get("status") != 1:
                return None

            product = data["product"]
            normalized = self._normalize(product, barcode)
            self._cache[barcode] = normalized
            return normalized

        except Exception as e:
            logger.warning(f"OFF API error for {barcode}: {e}")
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
            "ingredients": product.get("ingredients_text", "") or "",
            "nutriscore": product.get("nutriscore_grade"),
            "nova_group": product.get("nova_group"),
            "allergens": product.get("allergens_tags", []),
            "source_type": "barcode",
            "data_confidence": "high",
        }

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
