"""
app/domains/hospitality.py
Hospitality domain extension for hotels and resorts.
"""

from typing import Any


class HospitalityDomain:
    """Domain extension for hotel & hospitality businesses (Pullman reference)."""
    domain_name = "hospitality"

    @staticmethod
    def is_hospitality(profile_or_industry: Any) -> bool:
        if not profile_or_industry:
            return False
        if isinstance(profile_or_industry, str):
            return profile_or_industry.strip().lower() in ("hospitality", "hotel", "resort")
        ind = getattr(getattr(profile_or_industry, "metadata", None), "industry", "")
        mod = getattr(getattr(profile_or_industry, "domain_extensions", None), "domain_module", "")
        return (ind or "").strip().lower() == "hospitality" or (mod or "").strip().lower() == "hospitality"

