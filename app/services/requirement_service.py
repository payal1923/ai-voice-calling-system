import json
import re

from app.services.llm_service import generate_response

STRUCTURED_KEYS = [
    "requirements",
    "products_or_services",
    "quantity",
    "budget",
    "deadline",
    "important_details",
]

EXTRACTION_PROMPT = (
    "You are a precise information extraction engine for customer phone calls.\n"
    "Extract the customer's requirements from the transcript below.\n\n"
    "Return ONLY a valid JSON object (no markdown, no commentary) with exactly these keys:\n"
    '{{"requirements": string or null, "products_or_services": string or null, '
    '"quantity": string or null, "budget": string or null, '
    '"deadline": string or null, "important_details": string or null}}\n\n'
    "Rules:\n"
    "- Use null for any field that was not mentioned.\n"
    "- Do NOT invent or assume values.\n"
    "- Keep values short and factual.\n\n"
    "Transcript:\n{transcript}\n"
)

_KNOWN_PRODUCTS = [
    "laptop", "laptops", "desktop", "desktops", "computer", "computers",
    "server", "servers", "phone", "phones", "printer", "printers",
    "monitor", "monitors", "tablet", "tablets", "camera", "cameras",
    "website", "web development", "web application", "mobile app",
    "application", "software", "license", "licenses", "crm", "erp",
    "cloud", "migration", "hosting", "maintenance", "support",
    "consulting", "marketing", "seo", "integration", "training",
]

_QUANTITY_RE = re.compile(
    r"(\d[\d,]*)\s*(?:units?|pieces?|pcs?|boxes?|licenses?|seats?|items?|"
    r"sets?|laptops?|desktops?|computers?|servers?|phones?|monitors?|tablets?|users?)",
    re.IGNORECASE,
)

_BUDGET_RE = re.compile(
    r"([\d][\d,]*(?:\.\d+)?)\s*(dollars?|usd|rupees?|inr|rs\.?|euros?|eur|pounds?|gbp)",
    re.IGNORECASE,
)

_BUDGET_LABEL_RE = re.compile(
    r"budget(?:\s+is)?[^\d]{0,25}([\d][\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)

_DEADLINE_RE = re.compile(
    r"\b(next month|next week|next year|this month|this week|this year|"
    r"tomorrow|today|asap|by [a-z]+|within \d+ (?:days?|weeks?|months?)|"
    r"in \d+ (?:days?|weeks?|months?)|end of (?:the )?(?:month|week|year))\b",
    re.IGNORECASE,
)

_DETAIL_RE = re.compile(
    r"(\d+\s*gb\s*ram|\d+\s*tb|\d+\s*gb\s*ssd|16\s*gb|32\s*gb|8\s*gb|"
    r"i[357]\b|ryzen\s*\d|core\s*i\d|warranty|colour|color|branded?)",
    re.IGNORECASE,
)


class RequirementService:
    """
    Extracts structured customer requirements from free-form transcripts using
    the shared LLM service, with a deterministic regex fallback to guarantee
    populated fields even when the model output is malformed.
    """

    def __init__(self, llm_generator=None):
        self.llm_generator = llm_generator if llm_generator is not None else generate_response

    async def _call_llm(self, prompt: str, num_predict: int) -> str:
        """Invokes the LLM generator, tolerating generators without num_predict."""
        try:
            return await self.llm_generator(prompt, num_predict=num_predict)
        except TypeError:
            return await self.llm_generator(prompt)

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Best-effort extraction of the first JSON object found in a string."""
        if not raw:
            return {}
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        candidate = raw[start : end + 1]
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _normalize(data: dict) -> dict:
        """Normalizes model output into a fixed schema with null defaults."""
        result = {key: None for key in STRUCTURED_KEYS}
        if not isinstance(data, dict):
            return result
        for key in STRUCTURED_KEYS:
            value = data.get(key)
            if value is not None:
                text = str(value).strip()
                if text and text.lower() not in ["null", "none", "n/a", "unknown"]:
                    result[key] = text
        return result

    @staticmethod
    def _heuristic_extract(text: str) -> dict:
        """Deterministic fallback extraction based on pattern matching."""
        found = {key: None for key in STRUCTURED_KEYS}
        if not text or not text.strip():
            return found

        qty_match = _QUANTITY_RE.search(text)
        if qty_match:
            found["quantity"] = qty_match.group(1).replace(",", "").strip()

        budget_match = _BUDGET_RE.search(text)
        if budget_match:
            found["budget"] = f"{budget_match.group(1)} {budget_match.group(2)}".strip()
        else:
            label_match = _BUDGET_LABEL_RE.search(text)
            if label_match:
                found["budget"] = label_match.group(1).replace(",", "").strip()

        deadline_match = _DEADLINE_RE.search(text)
        if deadline_match:
            found["deadline"] = deadline_match.group(0).strip()

        lower = text.lower()
        products = [p for p in _KNOWN_PRODUCTS if re.search(rf"\b{re.escape(p)}\b", lower)]
        # Deduplicate while keeping a stable order.
        seen = set()
        unique_products = []
        for p in products:
            if p not in seen:
                seen.add(p)
                unique_products.append(p)
        if unique_products:
            found["products_or_services"] = ", ".join(unique_products)

        detail_match = _DETAIL_RE.search(text)
        if detail_match:
            found["important_details"] = detail_match.group(0).strip()

        if not any(found[k] for k in STRUCTURED_KEYS):
            found["requirements"] = text.strip()

        return found

    async def extract_requirements(self, full_text: str, is_resolved: bool = False) -> dict:
        """
        Extracts and returns structured requirements as a dictionary containing
        the standard schema plus a raw_response field for traceability.
        During active turns (is_resolved=False), uses deterministic pattern extraction
        to avoid locking the shared LLM service and delaying real-time customer voice turns.
        """
        # Use fast, deterministic heuristic extraction to avoid background LLM lock contention
        # and CPU saturation during active call sequences.
        result = self._normalize({})

        # Fill any gaps using the deterministic fallback so fields are never lost.
        heuristic = self._heuristic_extract(full_text or "")
        for key in STRUCTURED_KEYS:
            if not result.get(key) and heuristic.get(key):
                result[key] = heuristic[key]

        result["raw_response"] = "FAST_HEURISTIC_EXTRACTION"
        return result
