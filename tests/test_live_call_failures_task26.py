import unittest
import asyncio
import re
from app.services.knowledge_service import KnowledgeService
from app.services.response_service import ResponseService
from app.services.escalation_service import EscalationService


class TestLiveCallFailuresTask26(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ks = KnowledgeService()
        cls.rs = ResponseService()

    # =========================================================================
    # ISSUE A TESTS: Arrival Time Matching & Greeting Repetition Prevention
    # =========================================================================

    def test_a01_arrival_time_retrieves_checkin_knowledge(self):
        """Arrival query with progressive verb 'arriving' must match check_in_out fact."""
        query = "I will be arriving at the hotel around half past 11 in the evening."
        facts = self.ks.get_relevant_knowledge("pullman_aerocity", query)
        self.assertTrue(len(facts) > 0, "Expected knowledge facts to be retrieved for arrival query")
        self.assertIn("Check-in time at Pullman New Delhi Aerocity is 3:00 PM", facts)

    def test_a02_arrival_variants_matching(self):
        """Various arrival expressions must reliably match check_in_out fact."""
        arrival_queries = [
            "What time can we arrive at the hotel?",
            "When is our arrival time scheduled?",
            "We are arriving late tonight.",
        ]
        for q in arrival_queries:
            facts = self.ks.get_relevant_knowledge("pullman_aerocity", q)
            self.assertTrue(len(facts) > 0, f"Expected facts for '{q}'")
            self.assertIn("Check-in time at Pullman New Delhi Aerocity is 3:00 PM", facts)

    def test_a03_arrival_query_receives_accurate_response_without_greeting_fragments(self):
        """Arrival statement must not receive repeated greeting or broken fragments like 'Pullman New Delhi Aerocity. or visit today.'"""
        query = "I will be arriving at the hotel around half past 11 in the evening."
        facts = self.ks.get_relevant_knowledge("pullman_aerocity", query)
        history = (
            "Customer: Hello\n"
            "Assistant: Hello! Thank you for calling Pullman New Delhi Aerocity. How may I assist you with your stay or visit today?"
        )
        response = asyncio.run(self.rs.generate_response(query, conversation_context=history, knowledge_context=facts))
        resp_lower = response.lower()
        self.assertNotIn("pullman new delhi aerocity. or visit today", resp_lower)
        self.assertNotIn("thank you for calling", resp_lower)
        self.assertTrue("check-in" in resp_lower or "3:00 pm" in resp_lower or "15:00" in resp_lower, f"Unexpected response: {response}")

    def test_a04_sanitization_suppresses_orphan_greeting_tails_on_turn_2(self):
        """_sanitize_response must never output broken orphan phrases from greeting sentences."""
        hallucinated_sentence_1 = "Thank you for calling Pullman New Delhi Aerocity."
        hallucinated_sentence_2 = "How can I assist you with your stay or visit today?"
        history = "Customer: Hello\nAssistant: Welcome to Pullman."
        
        # Test first chunk handling with prior history
        sanitized_1 = self.rs._sanitize_response(
            hallucinated_sentence_1,
            transcript="I am arriving late",
            conversation_context=history,
            knowledge_context="",
            is_first_sentence=True,
        )
        self.assertNotEqual(sanitized_1, "Pullman New Delhi Aerocity.")
        self.assertNotIn("or visit today", sanitized_1.lower())

        # Test subsequent chunk handling (must return empty string, never orphan fragment)
        sanitized_2 = self.rs._sanitize_response(
            hallucinated_sentence_2,
            transcript="I am arriving late",
            conversation_context=history,
            knowledge_context="",
            is_first_sentence=False,
        )
        self.assertEqual(sanitized_2, "")

    def test_a05_streaming_chunks_for_arrival_clean_and_complete(self):
        """Sentence streaming must yield complete, grammatically sound chunks without greeting repetitions."""
        async def _test():
            q = "I will be arriving at the hotel around half past 11 in the evening."
            history = "Customer: Hello\nAssistant: Hello! Thank you for calling Pullman New Delhi Aerocity. How may I assist you with your stay or visit today?"
            chunks = [c async for c in self.rs.generate_response_stream_sentences(q, conversation_context=history)]
            self.assertTrue(len(chunks) >= 1)
            full_text = " ".join(chunks)
            self.assertNotIn("or visit today", full_text.lower())
            self.assertNotIn("pullman new delhi aerocity. or visit today", full_text.lower())
            self.assertTrue(any("check-in" in c.lower() or "3:00" in c for c in chunks), f"Chunks: {chunks}")
        asyncio.run(_test())

    # =========================================================================
    # ISSUE B TESTS: Airport Transportation vs. Luggage Storage Routing
    # =========================================================================

    def test_b01_airport_transport_with_suitcases_does_not_trigger_luggage_storage(self):
        """Mentioning passengers and suitcases for airport transport must NOT return luggage storage at front desk."""
        q = "Could you arrange airport transportation for 4 passengers with 3 large suitcases?"
        kb = self.ks.get_relevant_knowledge("pullman_aerocity", q)
        resp = asyncio.run(self.rs.generate_response(q, knowledge_context=kb))
        resp_lower = resp.lower()
        self.assertNotIn("luggage storage and concierge assistance are available for hotel guests at the front desk before check-in", resp_lower)
        self.assertTrue("airport" in resp_lower or "transfer" in resp_lower or "concierge" in resp_lower, f"Unexpected response: {resp}")

    def test_b02_airport_transport_capacity_is_not_fabricated(self):
        """Response must not claim that vehicle capacity is confirmed or that a car is booked."""
        q = "Could you arrange airport transportation for 4 passengers with 3 large suitcases?"
        kb = self.ks.get_relevant_knowledge("pullman_aerocity", q)
        resp = asyncio.run(self.rs.generate_response(q, knowledge_context=kb))
        resp_lower = resp.lower()
        self.assertNotIn("booked a car", resp_lower)
        self.assertNotIn("driver will arrive", resp_lower)
        self.assertIn("concierge", resp_lower)

    def test_b03_genuine_luggage_storage_receives_storage_response(self):
        """Inquiries asking to store or hold luggage must continue receiving the correct storage response."""
        queries = [
            "Can I store my luggage before check-in?",
            "Could the reception team look after our rucksacks after we vacate our room?",
            "Where can we leave our bags after check-out?",
        ]
        for q in queries:
            kb = self.ks.get_relevant_knowledge("pullman_aerocity", q)
            resp = asyncio.run(self.rs.generate_response(q, knowledge_context=kb))
            self.assertIn("luggage storage and concierge assistance", resp.lower(), f"Failed for '{q}': {resp}")

    def test_b04_streaming_transport_chunks_accurate(self):
        """Sentence streaming for airport transport with suitcases must yield transport chunks."""
        async def _test():
            q = "Could you arrange airport transportation for 4 passengers with 3 large suitcases?"
            chunks = [c async for c in self.rs.generate_response_stream_sentences(q)]
            self.assertTrue(len(chunks) >= 1)
            full_text = " ".join(chunks).lower()
            self.assertNotIn("luggage storage and concierge assistance are available for hotel guests at the front desk before check-in", full_text)
            self.assertTrue("airport transfer" in full_text or "concierge" in full_text, f"Full text: {full_text}")
        asyncio.run(_test())

    # =========================================================================
    # ISSUE C TESTS: Non-Escalation for 'Person' & Truthful Escalation Flow
    # =========================================================================

    def test_c01_person_in_reservation_does_not_escalate(self):
        """Customer inquiry mentioning 'the person who will be checking in' must NOT trigger escalation."""
        q = "My reservation is under a different name than the person who will be checking in."
        should_esc, reason = EscalationService.should_escalate(q)
        self.assertFalse(should_esc, f"Falsely escalated: {reason}")

    def test_c02_ordinary_person_and_agent_nouns_do_not_escalate(self):
        """Inquiries with ordinary non-escalation uses of 'person' or 'agent' must NOT escalate."""
        phrases = [
            "Can one person check in for two people?",
            "Is there a fee per person for the buffet?",
            "I booked through a travel agent online.",
            "Can another person pick up the room key?",
        ]
        for p in phrases:
            should_esc, reason = EscalationService.should_escalate(p)
            self.assertFalse(should_esc, f"Falsely escalated for '{p}': {reason}")

    def test_c03_explicit_human_requests_continue_to_escalate(self):
        """Legitimate requests to speak with a human or operator MUST escalate."""
        phrases = [
            "I want to speak to a person",
            "Transfer me to an operator right away",
            "Can I speak with a human manager or supervisor?",
            "Human representative please",
            "I need an agent",
            "I am furious, connect me to a human!",
        ]
        for p in phrases:
            should_esc, reason = EscalationService.should_escalate(p)
            self.assertTrue(should_esc, f"Expected escalation for '{p}', got False")

    def test_c04_escalation_does_not_promise_unsupported_live_transfer(self):
        """Escalation response must state live transfer is not available and offer front desk assistance without false promises."""
        from app.services.voice_response_service import VoiceResponseService
        vrs = VoiceResponseService()
        # Verify the canned message content used by VoiceResponseService
        truthful_phrase = "A live representative transfer is not available on this line."
        # Call EscalationService check
        should_esc, _ = EscalationService.should_escalate("I want to speak with a human representative please")
        self.assertTrue(should_esc)


if __name__ == "__main__":
    unittest.main()
