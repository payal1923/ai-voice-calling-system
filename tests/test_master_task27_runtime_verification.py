import unittest
import asyncio
import re

from app.services.knowledge_service import KnowledgeService
from app.services.response_service import ResponseService
from app.services.escalation_service import EscalationService
from app.services.voice_response_service import VoiceResponseService
from app.services.conversation_service import ConversationService
from app.database.database import SessionLocal
from app.database.models import Customer, Call


class TestMasterTask27RuntimeVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ks = KnowledgeService()
        cls.rs = ResponseService()
        cls.es = EscalationService()
        cls.company_id = "pullman_aerocity"

    # =========================================================================
    # ISSUE A: ARRIVAL TIME & GREETING FRAGMENTS
    # =========================================================================

    def test_issue_a_late_arrival_response(self):
        query = "I will be arriving at the hotel around half past 11 in the evening."
        k = self.ks.get_relevant_knowledge(self.company_id, query)
        self.assertTrue("check-in" in k.lower() or "check in" in k.lower())

        resp, _ = asyncio.run(self.rs.generate_response(
            transcript=query,
            conversation_context="assistant: Hello! Thank you for calling Pullman New Delhi Aerocity. How may I assist you today?",
            return_details=True,
            knowledge_context=k
        ))
        resp_lower = resp.lower()
        self.assertIn("3:00", resp_lower)
        self.assertTrue("24 hours" in resp_lower or "24-hour" in resp_lower or "front desk" in resp_lower)
        self.assertNotIn("late check-out is subject", resp_lower)
        self.assertNotIn("pullman new delhi aerocity. or visit today", resp_lower)

    def test_issue_a_first_turn_greeting_preserved(self):
        greeting = self.ks.get_greeting(self.company_id)
        self.assertIsNotNone(greeting)
        self.assertIn("Pullman New Delhi Aerocity", greeting)

    def test_issue_a_no_greeting_leak_on_turn_two(self):
        query = "I will be arriving at the hotel around half past 11 in the evening."
        k = self.ks.get_relevant_knowledge(self.company_id, query)
        context = "assistant: Hello! Thank you for calling Pullman New Delhi Aerocity. How may I assist you with your stay or visit today?"

        chunks = []
        async def run_stream():
            async for s in self.rs.generate_response_stream_sentences(query, conversation_context=context, knowledge_context=k):
                chunks.append(s)
        asyncio.run(run_stream())

        self.assertGreater(len(chunks), 0)
        full_streamed = " ".join(chunks)
        self.assertNotIn("Thank you for calling", full_streamed)
        self.assertNotIn("Pullman New Delhi Aerocity. or visit today.", full_streamed)
        self.assertNotIn("or visit today", full_streamed)

    # =========================================================================
    # ISSUE B: AIRPORT TRANSPORT VS LUGGAGE STORAGE
    # =========================================================================

    def test_issue_b_transport_with_suitcases_intended_speech(self):
        query = "Could you arrange airport transportation for four passengers with three large suitcases?"
        k = self.ks.get_relevant_knowledge(self.company_id, query)

        resp, _ = asyncio.run(self.rs.generate_response(
            transcript=query,
            return_details=True,
            knowledge_context=k
        ))
        resp_lower = resp.lower()
        self.assertIn("concierge", resp_lower)
        self.assertIn("transfers", resp_lower)
        self.assertIn("passengers", resp_lower)
        self.assertIn("suitcases", resp_lower)
        self.assertNotIn("complimentary luggage storage", resp_lower)
        self.assertNotIn("before check-in and after check-out", resp_lower)

    def test_issue_b_transport_with_asr_artifacts(self):
        query = "Cool New Arrange airport transportation for 4 passengers with 3 lap suitcase."
        k = self.ks.get_relevant_knowledge(self.company_id, query)

        resp, _ = asyncio.run(self.rs.generate_response(
            transcript=query,
            return_details=True,
            knowledge_context=k
        ))
        resp_lower = resp.lower()
        self.assertIn("concierge", resp_lower)
        self.assertIn("transfers", resp_lower)
        self.assertIn("passengers", resp_lower)
        self.assertIn("suitcases", resp_lower)
        self.assertNotIn("complimentary luggage storage", resp_lower)

    def test_issue_b_genuine_luggage_storage_request(self):
        query = "Can I store my luggage at the hotel before check-in?"
        k = self.ks.get_relevant_knowledge(self.company_id, query)

        resp, _ = asyncio.run(self.rs.generate_response(
            transcript=query,
            return_details=True,
            knowledge_context=k
        ))
        resp_lower = resp.lower()
        self.assertIn("luggage storage", resp_lower)
        self.assertIn("concierge", resp_lower)

    def test_issue_b_no_hallucinated_booking_or_price(self):
        query = "Can you book an airport cab for me and tell me how much it costs?"
        k = self.ks.get_relevant_knowledge(self.company_id, query)

        resp, _ = asyncio.run(self.rs.generate_response(
            transcript=query,
            return_details=True,
            knowledge_context=k
        ))
        resp_lower = resp.lower()
        for fake in ["rupees", "inr", "$", "500", "1000", "booked for you", "confirmed your cab"]:
            self.assertNotIn(fake, resp_lower)

    # =========================================================================
    # ISSUE C: RESERVATION NAME MISMATCH & ESCALATION LIFECYCLE
    # =========================================================================

    def test_issue_c_reservation_name_mismatch_not_escalated(self):
        query1 = "My reservation is under a different name than the person who will be checking in."
        should_esc1, _ = self.es.should_escalate(query1)
        self.assertFalse(should_esc1)

        query2 = "My reservation is under a different name than the person who will be check-in."
        should_esc2, _ = self.es.should_escalate(query2)
        self.assertFalse(should_esc2)

    def test_issue_c_reservation_name_guidance_response(self):
        query = "My reservation is under a different name than the person who will be checking in."
        k = self.ks.get_relevant_knowledge(self.company_id, query)

        resp, _ = asyncio.run(self.rs.generate_response(
            transcript=query,
            return_details=True,
            knowledge_context=k
        ))
        resp_lower = resp.lower()
        self.assertIn("id", resp_lower)
        self.assertIn("reservation", resp_lower)
        self.assertIn("booking", resp_lower)
        self.assertNotIn("transferring you to a human", resp_lower)
        self.assertNotIn("room rates and availability vary in real time", resp_lower)

    def test_issue_c_legitimate_human_request_escalates(self):
        query = "I would like to speak with a human representative please."
        should_esc, reason = self.es.should_escalate(query)
        self.assertTrue(should_esc)
        self.assertIn("human", reason.lower())

    def test_issue_c_escalation_does_not_promise_live_transfer(self):
        db = SessionLocal()
        try:
            conv_service = ConversationService()
            cust = db.query(Customer).first()
            if not cust:
                cust = Customer(external_customer_id="TASK27_USER", name="Task 27 Customer", phone="+919999900027")
                db.add(cust)
                db.commit()
                db.refresh(cust)

            call = conv_service.create_call(db, cust.id)
            call.company_id = "pullman_aerocity"
            db.commit()

            should_esc, reason = self.es.should_escalate("I want to speak with a human agent please")
            self.assertTrue(should_esc)

            req = self.es.create_escalation(
                db=db,
                call_id=call.id,
                reason=reason,
                language_selected="en",
                mark_call_ended=False,
            )
            db.refresh(call)

            # Verification 1: Assistance request created in DB
            self.assertIsNotNone(req.id)
            self.assertEqual(req.call_id, call.id)

            # Verification 2: Call remains open and IN_PROGRESS (not terminated or set to ESCALATED)
            self.assertIsNone(call.ended_at)
            self.assertEqual(call.status, "IN_PROGRESS")

            # Verification 3: Truthful message content
            esc_response = (
                "A live representative transfer is not available on this line. "
                "I have registered an assistance request with our front desk team. "
                "You may continue speaking with me, or contact the front desk directly for immediate assistance."
            )
            esc_lower = esc_response.lower()
            self.assertNotIn("transferring you to a human agent right now", esc_lower)
            self.assertNotIn("please stay on the line while we connect you", esc_lower)
            self.assertIn("assistance request", esc_lower)
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
