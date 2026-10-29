"""
tests/test_multi_business_platform_proof.py
Verification suite for Universal Multi-Business Voice Agent Platform.
Proves simultaneous, conflict-free multi-business operation across diverse industries:
1. Pullman New Delhi Aerocity (Hospitality reference tenant)
2. Apex Wheels Car Rental (Automotive rental second tenant)
"""

import sys
import unittest
import asyncio
from unittest.mock import patch, MagicMock

# Ensure project root is in path
sys.path.insert(0, ".")


class TestMultiBusinessPlatformProof(unittest.TestCase):
    """Proves domain registry, tenant routing, and domain isolation."""

    def setUp(self):
        from app.services.knowledge_service import KnowledgeService
        from app.services.response_service import ResponseService
        from app.domains.registry import DomainRegistry
        from app.domains.hospitality import HospitalityDomain
        from app.domains.automotive_rental import AutomotiveRentalDomain

        self.ks = KnowledgeService()
        self.rs = ResponseService()
        self.DomainRegistry = DomainRegistry
        self.HospitalityDomain = HospitalityDomain
        self.AutomotiveRentalDomain = AutomotiveRentalDomain

    def test_01_domain_registry_and_resolution(self):
        """Domain registry properly maps hospitality and automotive rental domains."""
        self.assertIsNotNone(self.DomainRegistry.get("hospitality"))
        self.assertIsNotNone(self.DomainRegistry.get("automotive_rental"))
        self.assertIsNone(self.DomainRegistry.get("unknown_domain"))

        # Verify tenant industry resolution
        self.assertTrue(self.rs._is_hospitality_domain("pullman_aerocity"))
        self.assertTrue(self.rs._is_hospitality_domain(None))
        self.assertTrue(self.rs._is_hospitality_domain(""))
        self.assertFalse(self.rs._is_hospitality_domain("apex_car_rental"))

    def test_02_pullman_hospitality_preserved(self):
        """Pullman hotel inquiries continue to execute specialized hospitality fast paths."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Check-in timing
            kb_checkin = self.ks.get_relevant_knowledge("pullman_aerocity", "What are your check-in timings?")
            resp1 = loop.run_until_complete(
                self.rs.generate_response(
                    "What are your check-in timings?",
                    knowledge_context=kb_checkin,
                    company_id="pullman_aerocity",
                )
            )
            self.assertIn("3:00 PM", resp1)
            self.assertIn("Pullman New Delhi Aerocity", resp1)

            # Check-out timing
            kb_checkout = self.ks.get_relevant_knowledge("pullman_aerocity", "What time is check out?")
            resp1_out = loop.run_until_complete(
                self.rs.generate_response(
                    "What time is check out?",
                    knowledge_context=kb_checkout,
                    company_id="pullman_aerocity",
                )
            )
            self.assertIn("12:00 PM", resp1_out)

            # Dining overview
            kb_dining = self.ks.get_relevant_knowledge("pullman_aerocity", "Where can I eat dinner at the hotel?")
            resp2 = loop.run_until_complete(
                self.rs.generate_response(
                    "Where can I eat dinner at the hotel?",
                    knowledge_context=kb_dining,
                    company_id="pullman_aerocity",
                )
            )
            self.assertTrue(any(rest in resp2 for rest in ["Farmer's Basket", "Honk", "Pluck"]))

            # Swimming pool
            kb_pool = self.ks.get_relevant_knowledge("pullman_aerocity", "What time does the swimming pool open?")
            resp3 = loop.run_until_complete(
                self.rs.generate_response(
                    "What time does the swimming pool open?",
                    knowledge_context=kb_pool,
                    company_id="pullman_aerocity",
                )
            )
            self.assertIn("swimming pool", resp3.lower())
            self.assertIn("heated outdoor", resp3.lower())
        finally:
            loop.close()

    def test_03_apex_car_rental_fast_paths(self):
        """Apex Wheels car rental inquiries resolve through automotive rental fast paths."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Fleet options
            resp1 = loop.run_until_complete(
                self.rs.generate_response(
                    "What cars do you have in your fleet?",
                    company_id="apex_car_rental",
                )
            )
            self.assertIn("Apex Wheels", resp1)
            self.assertIn("Swift", resp1)
            self.assertIn("Innova Crysta", resp1)
            # Must NOT contain hotel facts
            self.assertNotIn("hotel", resp1.lower())
            self.assertNotIn("check-in", resp1.lower())

            # Fuel policy
            resp2 = loop.run_until_complete(
                self.rs.generate_response(
                    "What is your fuel policy?",
                    company_id="apex_car_rental",
                )
            )
            self.assertIn("full-to-full", resp2.lower())
            self.assertIn("refueling", resp2.lower())

            # Driver license and age requirements
            resp3 = loop.run_until_complete(
                self.rs.generate_response(
                    "What documents do I need to rent a vehicle?",
                    company_id="apex_car_rental",
                )
            )
            self.assertIn("21 years old", resp3)
            self.assertIn("driving license", resp3.lower())

            # Security deposit
            resp4 = loop.run_until_complete(
                self.rs.generate_response(
                    "How much is the security deposit?",
                    company_id="apex_car_rental",
                )
            )
            self.assertIn("5,000", resp4)
            self.assertIn("deposit", resp4.lower())

            # Pet policy in car (MUST return crate/carrier policy, NEVER towels!)
            resp5 = loop.run_until_complete(
                self.rs.generate_response(
                    "Can I bring a dog in my rental car?",
                    company_id="apex_car_rental",
                )
            )
            self.assertIn("carrier", resp5.lower())
            self.assertNotIn("towel", resp5.lower())
            self.assertNotIn("room", resp5.lower())
        finally:
            loop.close()

    def test_04_streaming_sentences_both_tenants(self):
        """Streaming sentence generation works cleanly for both Pullman and Apex."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Pullman streaming
            async def get_pullman():
                sentences = []
                async for s in self.rs.generate_response_stream_sentences(
                    "What time is check out?",
                    company_id="pullman_aerocity",
                ):
                    sentences.append(s)
                return " ".join(sentences)

            pullman_text = loop.run_until_complete(get_pullman())
            self.assertIn("12:00 PM", pullman_text)

            # Apex streaming
            async def get_apex():
                sentences = []
                async for s in self.rs.generate_response_stream_sentences(
                    "Where do I collect the car at Delhi Airport?",
                    company_id="apex_car_rental",
                ) :
                    sentences.append(s)
                return " ".join(sentences)

            apex_text = loop.run_until_complete(get_apex())
            self.assertIn("Terminal 3", apex_text)
            self.assertIn("MLCP", apex_text)
        finally:
            loop.close()

    def test_05_cross_tenant_isolation_and_no_leakage(self):
        """Strict isolation: queries for one tenant never cross-contaminate the other."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Car rental query sent to Pullman must NOT return Apex fleet
            pullman_car_resp = loop.run_until_complete(
                self.rs.generate_response(
                    "What cars do you have?",
                    company_id="pullman_aerocity",
                )
            )
            self.assertNotIn("Apex Wheels", pullman_car_resp)
            self.assertNotIn("Innova Crysta", pullman_car_resp)

            # Hotel query sent to Apex must NOT return Pullman check-in times
            apex_hotel_resp = loop.run_until_complete(
                self.rs.generate_response(
                    "What are your check-in timings?",
                    company_id="apex_car_rental",
                )
            )
            self.assertNotIn("Pullman", apex_hotel_resp)
        finally:
            loop.close()

    def test_06_acoustic_normalization_isolation(self):
        """Acoustic normalization is tenant-isolated between hotel and car rental."""
        from app.services.stt_service import normalize_acoustic_phonetic_errors

        # Pullman tenant maps hotel acoustic distortions
        pullman_text = normalize_acoustic_phonetic_errors("I need a fresh dog in my room", company_id="pullman_aerocity")
        self.assertIn("fresh towels", pullman_text)

        # Apex tenant maps automotive distortions and does NOT touch pets
        apex_rules = self.ks.get_acoustic_rules("apex_car_rental")
        apex_text = normalize_acoustic_phonetic_errors("I want a card rental for a su v", company_id="apex_car_rental", acoustic_rules=apex_rules)
        self.assertIn("car rental", apex_text)
        self.assertIn("SUV", apex_text)

        # Pet inquiry for Apex is completely untouched by hotel rules
        apex_pet = normalize_acoustic_phonetic_errors("Can I bring a fresh dog in my rental car?", company_id="apex_car_rental", acoustic_rules=apex_rules)
        self.assertEqual(apex_pet, "Can I bring a fresh dog in my rental car?")
        self.assertNotIn("towel", apex_pet.lower())


if __name__ == "__main__":
    unittest.main()
