"""Food resolver — resolves food names using the local INDB database."""

import math
import logging
from pathlib import Path
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


def _safe_float(val):
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else round(f, 2)
    except (ValueError, TypeError):
        return None


class FoodResolver:
    """Resolves food names against the local INDB Excel database."""

    def __init__(self, excel_path: str = "data/Anuvaad_INDB_2024.11.xlsx"):
        self._foods: List[dict] = []
        self._name_index: Dict[str, dict] = {}
        self._names: List[str] = []

        self._load_indb(excel_path)

    def _load_indb(self, excel_path: str):
        path = Path(excel_path)
        if not path.exists():
            logger.warning(f"INDB file not found: {excel_path}")
            return

        try:
            import openpyxl
        except ImportError:
            logger.warning("openpyxl not installed — cannot load INDB")
            return

        wb = openpyxl.load_workbook(excel_path, read_only=True)
        ws = wb["Sheet1"]

        for row in ws.iter_rows(min_row=2, values_only=True):
            name = row[1]
            if not name:
                continue
            name = str(name).strip()

            sodium_mg = _safe_float(row[17])
            salt_g = round(sodium_mg * 2.5 / 1000, 2) if sodium_mg is not None else None

            # SFA is in mg in INDB — convert to g
            sfa_mg = _safe_float(row[10])
            sfa_g = round(sfa_mg / 1000, 2) if sfa_mg is not None else None

            food = {
                "product_name": name,
                "nutriments": {
                    "energy_kcal_100g": _safe_float(row[4]),
                    "fat_100g": _safe_float(row[7]),
                    "saturated_fat_100g": sfa_g,
                    "sugars_100g": _safe_float(row[8]),
                    "salt_100g": salt_g,
                    "proteins_100g": _safe_float(row[6]),
                    "fiber_100g": _safe_float(row[9]),
                },
                "ingredients": "",
                "source_type": "inferred",
                "data_confidence": "high",
            }

            self._foods.append(food)
            self._name_index[name.lower()] = food

        self._names = list(self._name_index.keys())
        wb.close()
        logger.info(f"Loaded {len(self._foods)} foods from INDB")

    async def resolve(self, food_text: str) -> Optional[Dict]:
        food_text = food_text.lower().strip()

        # 1. Exact match
        if food_text in self._name_index:
            return self._name_index[food_text].copy()

        # 2. Substring match (check if query is contained in any name)
        for name, food in self._name_index.items():
            if food_text in name or name in food_text:
                result = food.copy()
                result["data_confidence"] = "high"
                return result

        # 3. Fuzzy match
        try:
            from rapidfuzz import fuzz, process
            if self._names:
                match = process.extractOne(food_text, self._names, scorer=fuzz.ratio)
                if match and match[1] > 65:
                    result = self._name_index[match[0]].copy()
                    result["data_confidence"] = "high" if match[1] > 85 else "medium"
                    return result
        except ImportError:
            pass

        # 4. No match — return minimal fallback
        return {
            "product_name": food_text.title(),
            "nutriments": {
                "energy_kcal_100g": None,
                "fat_100g": None,
                "saturated_fat_100g": None,
                "sugars_100g": None,
                "salt_100g": None,
                "proteins_100g": None,
                "fiber_100g": None,
            },
            "ingredients": "",
            "source_type": "inferred",
            "data_confidence": "low",
        }

    def get_food_count(self) -> int:
        return len(self._foods)
