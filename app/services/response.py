"""Response formatter — converts verdicts to natural language."""


class ResponseFormatter:
    VERDICT_TEMPLATES = {
        "eat": "{name}: Go ahead! {reason}. Want to know more?",
        "avoid": "{name}: I'd skip this one. {reason}. Ask me about alternatives!",
        "sometimes": "{name}: Okay in moderation. {reason}. Want details?",
    }

    CASUAL_RESPONSES = {
        "greeting": (
            "Hey! I'm NutriAssist. Tell me a food or scan a barcode, "
            "and I'll tell you if it's good for you!"
        ),
        "thanks": "You're welcome! Ask me about any food anytime.",
        "generic": (
            "I help you make smart food choices! "
            "Try asking about a food or scanning a barcode."
        ),
    }

    def format_verdict(self, product_name: str, verdict: dict) -> str:
        template = self.VERDICT_TEMPLATES.get(
            verdict["verdict"], self.VERDICT_TEMPLATES["sometimes"]
        )
        return template.format(name=product_name, reason=verdict["reason"])

    def format_followup(self, product_name: str, question: str, verdict: dict) -> str:
        return f"About {product_name}: {verdict['reason']}."

    def casual_response(self, subtype: str) -> str:
        return self.CASUAL_RESPONSES.get(subtype, self.CASUAL_RESPONSES["generic"])

    def no_product_found(self, query: str) -> str:
        return f"I couldn't find nutrition data for '{query}'. Try being more specific!"

    def error_response(self) -> str:
        return "Something went wrong on my end. Could you try that again?"
