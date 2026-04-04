"""Pydantic schemas — single source of truth for data structures."""

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
    diet_type: Optional[str] = None
    goal: Optional[str] = None


class UnifiedProduct(BaseModel):
    product_name: str
    nutriments: Nutriments = Field(default_factory=Nutriments)
    ingredients: str = ""
    source_type: SourceType = SourceType.INFERRED
    data_confidence: DataConfidence = DataConfidence.LOW
    nutriscore: Optional[str] = None
    nova_group: Optional[int] = None
    allergens: list[str] = Field(default_factory=list)


class Verdict(BaseModel):
    verdict: str = "sometimes"
    reason: str = "Unable to analyze"
    confidence: float = 0.3


class QueryRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    user_profile: Optional[UserProfile] = None


class QueryResponse(BaseModel):
    session_id: str
    intent: str
    product_name: Optional[str] = None
    verdict: Optional[Verdict] = None
    response_text: str
    latency_ms: float
