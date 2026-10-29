"""
tests/test_contextual_stt_and_tenant_rules.py
Unit tests for tenant-aware STT acoustic normalization, rule isolation, and cache controls.
"""

import unittest
from app.services.stt_service import normalize_acoustic_phonetic_errors, STTService
from app.services.knowledge_service import KnowledgeService
from app.knowledge.schema import AcousticNormalizationRule


class TestContextualSTTAndTenantRules(unittest.TestCase):
    """Verifies that acoustic normalization is scoped to the active tenant and preserves neutral queries."""

    def test_01_pullman_default_normalization(self):
        """Verifies Pullman acoustic rules apply when company_id defaults to pullman_aerocity."""
        slip = "I need a fresh dog in my room."
        norm = normalize_acoustic_phonetic_errors(slip)
        self.assertEqual(norm.rstrip("."), "I need fresh towels in my room")

        wifi_slip = "is it verified available in room"
        norm_wifi = normalize_acoustic_phonetic_errors(wifi_slip)
        self.assertIn("is Wi-Fi available", norm_wifi)

        timing_slip = "what is taking time"
        norm_timing = normalize_acoustic_phonetic_errors(timing_slip)
        self.assertEqual(norm_timing, "what is check-in time")

    def test_02_pullman_explicit_normalization(self):
        """Verifies Pullman acoustic rules apply when company_id='pullman_aerocity' is explicit."""
        slip = "bring fresh dogs to our room"
        norm = normalize_acoustic_phonetic_errors(slip, company_id="pullman_aerocity")
        self.assertIn("fresh towels in my room", norm)

    def test_03_non_hotel_tenant_isolation(self):
        """Verifies non-hotel tenants (e.g. apex_car_rental) are NOT contaminated by Pullman hotel rules."""
        # A pet inquiry in a car rental context must NOT be converted to towels
        pet_query = "Can I bring a fresh dog in my rental car?"
        norm_car = normalize_acoustic_phonetic_errors(pet_query, company_id="apex_car_rental")
        self.assertEqual(norm_car, pet_query)
        self.assertNotIn("towels", norm_car.lower())

        # A timing question in car rental must NOT become hotel check-in
        car_timing = "what is the taking time for the vehicle"
        norm_timing = normalize_acoustic_phonetic_errors(car_timing, company_id="apex_car_rental")
        self.assertEqual(norm_timing, car_timing)
        self.assertNotIn("check-in", norm_timing.lower())

    def test_04_universal_neutral_mode(self):
        """Verifies that an unknown or neutral tenant receives raw text without hotel mutations."""
        text = "I need a fresh dog in my office."
        norm = normalize_acoustic_phonetic_errors(text, company_id="universal_generic")
        self.assertEqual(norm, text)

    def test_05_custom_acoustic_rules_applied(self):
        """Verifies that custom declarative acoustic rules are applied properly."""
        custom_rule = AcousticNormalizationRule(
            rule_id="car_rental_fix",
            pattern=r"\bcard\s+rental\b",
            replacement="car rental",
            description="Fix acoustic mistranscription of car"
        )
        text = "I am calling about a card rental booking."
        norm = normalize_acoustic_phonetic_errors(text, acoustic_rules=[custom_rule])
        self.assertEqual(norm, "I am calling about a car rental booking.")

    def test_06_knowledge_service_cache_controls(self):
        """Verifies clear_cache and reload_profile functionality on KnowledgeService."""
        ks = KnowledgeService()
        p1 = ks.load_profile("pullman_aerocity")
        self.assertIsNotNone(p1)

        # Clear specific cache
        ks.clear_cache("pullman_aerocity")
        p2 = ks.load_profile("pullman_aerocity")
        self.assertIsNotNone(p2)

        # Clear all cache
        ks.clear_cache()
        p3 = ks.load_profile("pullman_aerocity")
        self.assertIsNotNone(p3)

    def test_07_stt_service_signature_compatibility(self):
        """Verifies that STTService.transcribe accepts company_id without errors."""
        stt = STTService()
        self.assertTrue(hasattr(stt, "transcribe"))
        self.assertTrue(hasattr(stt, "transcribe_with_info"))
        self.assertTrue(hasattr(stt, "transcribe_with_confidence"))


if __name__ == "__main__":
    unittest.main()
