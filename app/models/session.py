"""Session management — in-memory conversation state."""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class ConversationSession:
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)

    current_product: Optional[dict] = None
    current_verdict: Optional[dict] = None

    product_history: list = field(default_factory=list)
    conversation_history: list = field(default_factory=list)

    user_profile: dict = field(default_factory=lambda: {
        "allergies": [],
        "conditions": [],
        "diet_type": None,
        "goal": None,
    })

    def set_product(self, product: dict, verdict: dict):
        if self.current_product:
            self.product_history.append(self.current_product)
            if len(self.product_history) > 5:
                self.product_history.pop(0)
        self.current_product = product
        self.current_verdict = verdict

    def add_turn(self, role: str, content: str):
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        if len(self.conversation_history) > 10:
            self.conversation_history.pop(0)

    def has_product_context(self) -> bool:
        return self.current_product is not None
