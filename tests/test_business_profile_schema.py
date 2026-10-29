"""
tests/test_business_profile_schema.py
Unit tests for the Universal Multi-Business Profile Schema and KnowledgeService loader.
"""

import unittest
import json
import os
from pydantic import ValidationError
from app.knowledge.schema import (
    BusinessProfile,
    BusinessMetadata,
    LocationConfig,
    OperatingHoursConfig,
    VoiceLanguageConfig,
    ConversationalPreferences,
    AcousticNormalizationRule,
    SpeechRecognitionConfig,
    KnowledgeFact,
    FAQItem,
    KnowledgeBaseConfig,
    BusinessPolicies,
    DomainExtensionConfig,
)
from app.services.knowledge_service import KnowledgeService


class TestBusinessProfileSchema(unittest.TestCase):
    """Verifies schema validation, constraint enforcement, and legacy adaptations."""

    def setUp(self):
        self.ks = KnowledgeService()

    def test_01_load_pullman_profile_as_model(self):
        """Verifies that existing pullman_aerocity.json loads and validates as BusinessProfile."""
        profile = self.ks.load_profile("pullman_aerocity")
        self.assertIsNotNone(profile)
        self.assertIsInstance(profile, BusinessProfile)
        self.assertEqual(profile.metadata.company_id, "pullman_aerocity")
        self.assertEqual(profile.metadata.company_name, "Pullman New Delhi Aerocity")
        self.assertEqual(profile.metadata.industry, "hospitality")
        self.assertIn("Accor", profile.metadata.brand or "")
        self.assertTrue(len(profile.knowledge_base.facts) >= 20)
        self.assertIn("check_in_out", [f.id for f in profile.knowledge_base.facts])

    def test_02_load_company_profile_backward_compatibility(self):
        """Verifies that load_company_profile returns a dict with legacy keys."""
        profile = self.ks.load_company_profile("pullman_aerocity")
        self.assertIsNotNone(profile)
        self.assertIsInstance(profile, dict)
        self.assertEqual(profile["company_name"], "Pullman New Delhi Aerocity")
        self.assertIn("facts", profile)
        self.assertIsInstance(profile["facts"], list)
        self.assertIn("greeting_message", profile)
        self.assertIn("stt_prompt", profile)

    def test_03_slug_validation(self):
        """Verifies that company_id validation rejects invalid or malicious slugs."""
        # Valid slugs
        valid_meta = BusinessMetadata(company_id="valid_slug-123", company_name="Valid Co", industry="tech")
        self.assertEqual(valid_meta.company_id, "valid_slug-123")

        # Invalid slugs
        for bad_id in ["../etc/passwd", "bad slug", "bad/path", "bad\\slash", "a", "UPPER!CASE"]:
            with self.assertRaises(ValidationError):
                BusinessMetadata(company_id=bad_id, company_name="Bad Co", industry="tech")

    def test_04_regex_validation_in_acoustic_rules(self):
        """Verifies that invalid regex patterns in acoustic rules are rejected at validation time."""
        # Valid pattern
        rule = AcousticNormalizationRule(
            rule_id="r1",
            pattern=r"\bcard\s+rental\b",
            replacement="car rental",
            description="Fix acoustic mistranscription"
        )
        self.assertEqual(rule.pattern, r"\bcard\s+rental\b")

        # Broken regex pattern (unclosed parenthesis)
        with self.assertRaises(ValidationError):
            AcousticNormalizationRule(
                rule_id="r2",
                pattern=r"(unclosed_bracket",
                replacement="fix"
            )

    def test_05_duplicate_fact_ids_rejected(self):
        """Verifies that duplicate fact IDs within a profile are rejected."""
        facts = [
            KnowledgeFact(id="wifi_fact", category="amenities", keywords=["wifi"], fact="Free Wi-Fi is provided."),
            KnowledgeFact(id="wifi_fact", category="amenities", keywords=["internet"], fact="High-speed internet."),
        ]
        with self.assertRaises(ValidationError):
            KnowledgeBaseConfig(facts=facts)

    def test_06_round_trip_serialization(self):
        """Verifies that BusinessProfile can be dumped to dict and re-validated without loss."""
        profile = self.ks.load_profile("pullman_aerocity")
        self.assertIsNotNone(profile)
        dumped = profile.model_dump()
        reloaded = BusinessProfile.from_dict(dumped)
        self.assertEqual(profile.metadata.company_id, reloaded.metadata.company_id)
        self.assertEqual(len(profile.knowledge_base.facts), len(reloaded.knowledge_base.facts))

    def test_07_second_industry_profile_instantiation(self):
        """Verifies that a car rental business profile satisfies the schema cleanly."""
        rental_data = {
            "metadata": {
                "company_id": "apex_car_rental",
                "company_name": "Apex Wheels Car Rental",
                "industry": "automotive_rental",
                "brand": "Apex Mobility"
            },
            "location": {
                "address": "Terminal 3, IGI Airport, New Delhi",
                "proximity": "Multi-Level Car Parking Level 2"
            },
            "operating_hours": {
                "is_24_7": True
            },
            "voice_and_language": {
                "default_language": "en-US",
                "tts_voice_id": "en_US-lessac-medium"
            },
            "conversational_preferences": {
                "greeting_message": "Hello, thank you for calling Apex Wheels. How can I assist you with your car rental today?",
                "system_instruction": "You are the AI assistant for Apex Wheels Car Rental.",
                "fallback_message": "I apologize, could you please repeat your car rental question?",
                "clarification_message": "Could you clarify which vehicle model you need?"
            },
            "speech_recognition": {
                "stt_prompt": "Apex Wheels Car Rental: SUV, sedan, airport pickup, driver license.",
                "vocabulary_boost": ["Apex", "SUV", "sedan"],
                "acoustic_normalization_rules": [
                    {
                        "rule_id": "car_mishearing",
                        "pattern": r"\bcard\s+rental\b",
                        "replacement": "car rental"
                    }
                ]
            },
            "knowledge_base": {
                "facts": [
                    {
                        "id": "fleet_overview",
                        "category": "fleet",
                        "keywords": ["car", "cars", "suv", "sedan", "fleet"],
                        "fact": "We offer economy sedans, compact hatchbacks, and 7-seater SUVs.",
                        "priority": 10
                    }
                ]
            },
            "policies": {
                "cancellation_policy": "Free cancellation up to 12 hours before pickup.",
                "custom_policies": {
                    "fuel_policy": "Full-to-full fuel policy applies."
                }
            },
            "domain_extensions": {
                "domain_module": "automotive_rental",
                "deterministic_handlers_enabled": True
            }
        }
        profile = BusinessProfile.from_dict(rental_data)
        self.assertEqual(profile.metadata.company_id, "apex_car_rental")
        self.assertEqual(profile.metadata.industry, "automotive_rental")
        self.assertEqual(profile.knowledge_base.facts[0].id, "fleet_overview")

    def test_08_path_traversal_protection(self):
        """Verifies that KnowledgeService blocks path traversal attempts."""
        self.assertIsNone(self.ks.load_profile("../../etc/passwd"))
        self.assertIsNone(self.ks.load_company_profile("../../etc/passwd"))
        self.assertIsNone(self.ks.load_profile("pullman/aerocity"))
        self.assertIsNone(self.ks.load_profile("pullman\\aerocity"))


if __name__ == "__main__":
    unittest.main()

