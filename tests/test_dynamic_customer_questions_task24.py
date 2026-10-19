import unittest
import asyncio
from app.services.knowledge_service import KnowledgeService
from app.services.response_service import ResponseService

class TestDynamicCustomerQuestionsTask24(unittest.IsolatedAsyncioTestCase):
    """
    Test suite verifying dynamic, unscripted customer question handling:
    1. Dynamic question understanding without fixed order or forced dining/check-in routing.
    2. Grounded answers across hotel facilities (luggage, accessibility, parking, metro, ID policy).
    3. Natural follow-ups and pronoun resolution across diverse non-dining entities.
    4. Honest limitations for unlisted amenities and policies without hallucination.
    """

    def setUp(self):
        self.company_id = "pullman_aerocity"
        self.ks = KnowledgeService()
        self.rs = ResponseService()

    async def test_01_luggage_storage_unscripted_inquiry(self):
        """1. Customer asks about holding bags before check-in without mentioning standard keywords."""
        transcript = "Hi, my flight lands early in the morning, can I drop my suitcases somewhere before my room is ready?"
        knowledge = self.ks.get_relevant_knowledge(
            company_id=self.company_id,
            query=transcript,
            conversation_context=""
        )
        self.assertIn("luggage", knowledge.lower())
        self.assertIn("concierge", knowledge.lower())

        response = await self.rs.generate_response(
            transcript=transcript,
            knowledge_context=knowledge,
            conversation_context=""
        )
        # Should provide luggage storage info, NOT check-in timing (3 PM)
        self.assertTrue(
            any(w in response.lower() for w in ["luggage", "concierge", "bag", "storage"]),
            f"Expected luggage information, got: {response}"
        )
        self.assertNotIn("check-in at pullman new delhi aerocity is at 3:00 pm", response.lower())

    async def test_02_accessibility_and_wheelchair_facilities(self):
        """2. Customer asks in natural language about accessibility for an elderly or disabled relative."""
        transcript = "I am traveling with my grandmother who uses a wheelchair. Is the property accessible for her?"
        knowledge = self.ks.get_relevant_knowledge(
            company_id=self.company_id,
            query=transcript,
            conversation_context=""
        )
        self.assertIn("wheelchair", knowledge.lower())
        self.assertIn("accessible", knowledge.lower())

        response = await self.rs.generate_response(
            transcript=transcript,
            knowledge_context=knowledge,
            conversation_context=""
        )
        self.assertTrue(
            any(w in response.lower() for w in ["wheelchair", "accessible", "accessibility", "ramp", "ramps", "elevator"]),
            f"Expected accessibility details, got: {response}"
        )

    async def test_03_parking_and_valet_facilities(self):
        """3. Customer asks about bringing their car and parking charges."""
        transcript = "Do you guys have parking on-site and is valet service available?"
        knowledge = self.ks.get_relevant_knowledge(
            company_id=self.company_id,
            query=transcript,
            conversation_context=""
        )
        self.assertIn("parking", knowledge.lower())

        response = await self.rs.generate_response(
            transcript=transcript,
            knowledge_context=knowledge,
            conversation_context=""
        )
        self.assertTrue(
            any(w in response.lower() for w in ["parking", "valet"]),
            f"Expected parking information, got: {response}"
        )

    async def test_04_metro_station_public_transport_query(self):
        """4. Customer asks about public transit / Delhi Metro access."""
        transcript = "Is the hotel close to the metro station or public transport?"
        knowledge = self.ks.get_relevant_knowledge(
            company_id=self.company_id,
            query=transcript,
            conversation_context=""
        )
        self.assertTrue(
            any(w in knowledge.lower() for w in ["metro", "aerocity metro", "transport", "airport"]),
            f"Expected metro/transport knowledge, got: {knowledge}"
        )

        response = await self.rs.generate_response(
            transcript=transcript,
            knowledge_context=knowledge,
            conversation_context=""
        )
        self.assertTrue(
            any(w in response.lower() for w in ["metro", "aerocity", "transport", "station"]),
            f"Expected metro or transit answer, got: {response}"
        )

    async def test_05_checkin_pan_card_rejection_policy(self):
        """5. Customer asks if PAN card is acceptable for check-in; must answer ID policy, not 3 PM time."""
        transcript = "Can I just show my PAN card when checking in tomorrow?"
        knowledge = self.ks.get_relevant_knowledge(
            company_id=self.company_id,
            query=transcript,
            conversation_context=""
        )
        self.assertIn("pan", knowledge.lower())
        self.assertIn("government-issued photo id", knowledge.lower())

        response = await self.rs.generate_response(
            transcript=transcript,
            knowledge_context=knowledge,
            conversation_context=""
        )
        # PAN cards are strictly not accepted as address proof per policy
        self.assertTrue(
            any(w in response.lower() for w in ["pan card is not accepted", "pan card", "not accepted", "photo id", "passport", "aadhaar"]),
            f"Expected ID policy stating PAN card requirement, got: {response}"
        )
        # Check-in time (3:00 PM) must NOT hijack the response
        self.assertNotIn("check-in at pullman new delhi aerocity is at 3:00 pm", response.lower())

    async def test_06_unlisted_amenity_business_center_handled_honestly(self):
        """6. Customer asks for an unlisted facility (business center); agent must not hallucinate."""
        transcript = "Do you have a dedicated business center with printing and computers?"
        knowledge = self.ks.get_relevant_knowledge(
            company_id=self.company_id,
            query=transcript,
            conversation_context=""
        )
        # Business center is not a separate listed amenity
        response = await self.rs.generate_response(
            transcript=transcript,
            knowledge_context=knowledge,
            conversation_context=""
        )
        # Must decline honestly and offer front desk / available amenities, not claim 24/7 or invent services
        self.assertTrue(
            any(w in response.lower() for w in ["not offer", "front desk", "do not have", "available amenities", "not available"]),
            f"Expected honest unlisted facility response, got: {response}"
        )
        self.assertNotIn("24-hour", response.lower())

    async def test_07_cancellation_policy_honest_limitation(self):
        """7. Customer asks about cancelling their reservation and getting a refund."""
        transcript = "What is your cancellation policy if I need to cancel my room?"
        knowledge = self.ks.get_relevant_knowledge(
            company_id=self.company_id,
            query=transcript,
            conversation_context=""
        )
        response = await self.rs.generate_response(
            transcript=transcript,
            knowledge_context=knowledge,
            conversation_context=""
        )
        # Should refer to front desk / reservations rather than hallucinating specific penalty fees
        self.assertTrue(
            any(w in response.lower() for w in ["cancellation", "front desk", "reservations", "policy"]),
            f"Expected cancellation policy handling, got: {response}"
        )

    async def test_08_natural_followup_pronoun_resolution_gym(self):
        """8. Cross-entity follow-up: gym followed by 'Is it open 24 hours?' resolves to gym, not dining."""
        conv_context = (
            "Customer: Does the hotel have a fitness center?\n"
            "Assistant: Yes, our Fit Lounge fitness center offers contemporary equipment and dedicated wellness spaces."
        )

        followup_transcript = "Is it open 24 hours?"
        knowledge = self.ks.get_relevant_knowledge(
            company_id=self.company_id,
            query=followup_transcript,
            conversation_context=conv_context
        )
        # Knowledge must retrieve Fit Lounge facts
        self.assertIn("fit lounge", knowledge.lower())
        self.assertIn("24/7", knowledge.lower())

        response = await self.rs.generate_response(
            transcript=followup_transcript,
            knowledge_context=knowledge,
            conversation_context=conv_context
        )
        # Must confirm 24-hour access for Fit Lounge and NOT mention dining/restaurants
        self.assertTrue(
            any(w in response.lower() for w in ["24-hour", "24 hours", "24/7", "fit lounge", "yes"]),
            f"Expected 24-hour gym confirmation, got: {response}"
        )
        self.assertNotIn("bar", response.lower())
        self.assertNotIn("honk", response.lower())

    async def test_09_natural_followup_pool_hours_unconfirmed(self):
        """9. Customer inquires about pool, then asks 'What time does it close?'; handles honestly without hallucinating."""
        conv_context = (
            "Customer: Tell me about your swimming pool.\n"
            "Assistant: The hotel offers a heated outdoor swimming pool for relaxation."
        )

        followup_transcript = "What time does it close?"
        knowledge = self.ks.get_relevant_knowledge(
            company_id=self.company_id,
            query=followup_transcript,
            conversation_context=conv_context
        )
        # Knowledge should maintain pool context
        self.assertTrue("swimming pool" in knowledge.lower() or "pool" in knowledge.lower())

        response = await self.rs.generate_response(
            transcript=followup_transcript,
            knowledge_context=knowledge,
            conversation_context=conv_context
        )
        # Pool operating hours are unconfirmed in KB; agent must state limitation and refer to front desk
        self.assertTrue(
            any(w in response.lower() for w in ["not available", "front desk", "timings", "operating hours"]),
            f"Expected unconfirmed pool hours response, got: {response}"
        )
        self.assertNotIn("check-out", response.lower())

    async def test_10_topic_switching_without_predefined_order(self):
        """10. Customer rapidly switches topics: dining -> check-in time -> parking -> luggage."""
        # Step 1: Ask about dining
        t1 = "What food options do you have?"
        k1 = self.ks.get_relevant_knowledge(self.company_id, t1, conversation_context="")
        r1 = await self.rs.generate_response(t1, knowledge_context=k1, conversation_context="")
        self.assertTrue(any(v in r1.lower() for v in ["pluck", "honk", "café", "cafe", "bar"]))

        # Step 2: Switch to check-in time
        t2 = "Okay, and when is standard check-in?"
        ctx2 = f"Customer: {t1}\nAssistant: {r1}"
        k2 = self.ks.get_relevant_knowledge(self.company_id, t2, conversation_context=ctx2)
        r2 = await self.rs.generate_response(t2, knowledge_context=k2, conversation_context=ctx2)
        self.assertIn("3:00 pm", r2.lower())

        # Step 3: Switch to parking
        t3 = "Do you have valet parking if I drive there?"
        ctx3 = ctx2 + f"\nCustomer: {t2}\nAssistant: {r2}"
        k3 = self.ks.get_relevant_knowledge(self.company_id, t3, conversation_context=ctx3)
        r3 = await self.rs.generate_response(t3, knowledge_context=k3, conversation_context=ctx3)
        self.assertIn("parking", r3.lower())
        self.assertNotIn("3:00 pm", r3.lower())

        # Step 4: Switch to luggage storage
        t4 = "Can the concierge hold our bags after check-out?"
        ctx4 = ctx3 + f"\nCustomer: {t3}\nAssistant: {r3}"
        k4 = self.ks.get_relevant_knowledge(self.company_id, t4, conversation_context=ctx4)
        r4 = await self.rs.generate_response(t4, knowledge_context=k4, conversation_context=ctx4)
        self.assertTrue(any(w in r4.lower() for w in ["luggage", "concierge", "bag", "storage"]))

if __name__ == "__main__":
    unittest.main()
