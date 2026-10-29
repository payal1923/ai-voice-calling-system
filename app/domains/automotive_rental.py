"""
app/domains/automotive_rental.py
Automotive rental domain extension for vehicle hire, car rental, and mobility providers.
"""

import re
from typing import Any, Optional


class AutomotiveRentalDomain:
    """Domain extension for car rental and automotive mobility businesses."""
    domain_name = "automotive_rental"

    @staticmethod
    def is_automotive_rental(profile_or_industry: Any) -> bool:
        if not profile_or_industry:
            return False
        if isinstance(profile_or_industry, str):
            return profile_or_industry.strip().lower() in ("automotive_rental", "car_rental", "mobility")
        ind = getattr(getattr(profile_or_industry, "metadata", None), "industry", "")
        mod = getattr(getattr(profile_or_industry, "domain_extensions", None), "domain_module", "")
        return (ind or "").strip().lower() in ("automotive_rental", "car_rental") or (mod or "").strip().lower() in ("automotive_rental", "car_rental")

    @classmethod
    def get_fast_path_response(cls, transcript: str, knowledge_context: str = "") -> Optional[str]:
        """Provides deterministic, instant grounded responses for automotive rental queries."""
        if not transcript or not transcript.strip():
            return None
        t_lower = transcript.strip().lower()

        # 1. Fleet overview
        if any(w in t_lower for w in ["what cars", "which cars", "fleet", "vehicle options", "what models", "car options", "what vehicles", "cars do you have", "cars available"]):
            return (
                "Apex Wheels offers Compact Economy hatchbacks (Maruti Swift and Hyundai i20), "
                "Executive Sedans (Honda City and Skoda Slavia), and Premium SUVs (Toyota Innova Crysta "
                "and Mahindra XUV700). All vehicles include air conditioning and GPS navigation."
            )

        # 2. Driver requirements & license
        if any(w in t_lower for w in ["what documents", "driver requirement", "driving license", "driver licence", "minimum age", "how old", "license required", "id requirement"]):
            return (
                "All primary drivers must be at least 21 years old and hold a valid original driving license "
                "for at least one year. International travellers must present a valid passport and "
                "International Driving Permit."
            )

        # 3. Fuel policy
        if any(w in t_lower for w in ["fuel policy", "petrol policy", "diesel policy", "full to full", "refuel", "gas policy"]):
            return (
                "Our rentals operate on a transparent full-to-full fuel policy. You receive the vehicle with a "
                "full tank of fuel and must return it full to avoid refueling service surcharges."
            )

        # 4. Pickup / drop-off location
        if any(w in t_lower for w in [
            "where to pickup", "where to pick up", "pickup location", "where to collect",
            "where do i collect", "collect the car", "collect car", "collect",
            "where is the counter", "mlcp", "terminal 3 counter", "terminal 3"
        ]):
            return (
                "Airport vehicle pickups occur at Terminal 3 Multi-Level Car Parking (MLCP) Level 2, "
                "accessible via the covered skywalk directly from the arrival hall."
            )

        # 5. Security deposit
        if any(w in t_lower for w in ["security deposit", "how much deposit", "deposit amount", "deposit refund", "refundable deposit"]):
            return (
                "A refundable security deposit of 5,000 to 10,000 rupees depending on vehicle category is "
                "pre-authorized on a credit card upon vehicle pickup and released within 48 hours of vehicle return inspection."
            )

        # 6. Pet policy in car
        if any(w in t_lower for w in ["pet", "pets", "dog", "dogs", "cat", "cats", "animal", "animals"]):
            return (
                "Pets are permitted in rental vehicles provided they are transported in suitable carriers or crates. "
                "A deep-cleaning fee applies only if pet hair or excessive soiling requires professional detailing."
            )

        return None
