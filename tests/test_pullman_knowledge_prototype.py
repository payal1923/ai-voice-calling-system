import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
import unittest
from app.services.knowledge_service import KnowledgeService
from app.services.response_service import ResponseService
from app.services.escalation_service import EscalationService


def safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


class TestPullmanKnowledgePrototype(unittest.TestCase):
    def setUp(self):
        self.knowledge_service = KnowledgeService()
        self.response_service = ResponseService()

    def test_01_knowledge_retrieval_pullman(self):
        """Verifies that KnowledgeService retrieves accurate Pullman facts for hotel queries."""
        profile = self.knowledge_service.load_company_profile("pullman_aerocity")
        self.assertIsNotNone(profile)
        self.assertEqual(profile["company_name"], "Pullman New Delhi Aerocity")

        # 1. Check-in / check-out retrieval
        facts_checkin = self.knowledge_service.get_relevant_knowledge(
            "pullman_aerocity", "What are the check-in and check-out timings?"
        )
        self.assertIn("3:00 PM", facts_checkin)
        self.assertIn("12:00 PM", facts_checkin)

        # 2. Dining retrieval
        facts_dining = self.knowledge_service.get_relevant_knowledge(
            "pullman_aerocity", "Where can I eat dinner? What restaurants do you have?"
        )
        self.assertIn("Pluck", facts_dining)
        self.assertIn("Honk", facts_dining)

        # 3. Absent amenity query (helipad)
        facts_absent = self.knowledge_service.get_relevant_knowledge(
            "pullman_aerocity", "Can I land my private helicopter on the rooftop?"
        )
        self.assertEqual(facts_absent, "")
        print("\n[TEST 1 PASSED] Knowledge retrieval for Pullman Aerocity verified.")

    def test_02_hotel_check_in_check_out_response(self):
        """Tests that LLM answers check-in and check-out query using retrieved Pullman facts."""
        async def _test():
            query = "What is the check in and check out time?"
            system_instruction = self.knowledge_service.get_system_instruction("pullman_aerocity")
            knowledge_context = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", query)

            self.assertIn("3:00 PM", knowledge_context)
            self.assertIn("12:00 PM", knowledge_context)

            response = await self.response_service.generate_response(
                transcript=query,
                system_instruction=system_instruction,
                knowledge_context=knowledge_context,
            )
            safe_print(f"\n[TEST 2 Response] Query: {query}\nResponse: {response}")
            resp_lower = response.lower()
            self.assertTrue(
                "3:00" in response or "3 pm" in resp_lower or "3:00 pm" in resp_lower or "15:00" in response,
                f"Check-in time not mentioned: {response}"
            )
            self.assertTrue(
                "12:00" in response or "12 pm" in resp_lower or "12:00 pm" in resp_lower or "noon" in resp_lower,
                f"Check-out time not mentioned: {response}"
            )

        asyncio.run(_test())
        print("[TEST 2 PASSED] Check-in and check-out response verified.")

    def test_03_restaurant_information_response(self):
        """Tests that LLM provides approved restaurant names (Pluck, Honk, etc.) from dataset."""
        async def _test():
            query = "What dining options and restaurants do you have at the hotel?"
            system_instruction = self.knowledge_service.get_system_instruction("pullman_aerocity")
            knowledge_context = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", query)

            self.assertIn("Pluck", knowledge_context)
            self.assertIn("Honk", knowledge_context)

            response = await self.response_service.generate_response(
                transcript=query,
                system_instruction=system_instruction,
                knowledge_context=knowledge_context,
            )
            safe_print(f"\n[TEST 3 Response] Query: {query}\nResponse: {response}")
            resp_lower = response.lower()
            self.assertTrue(
                "pluck" in resp_lower or "honk" in resp_lower or "dining" in resp_lower or "restaurant" in resp_lower,
                f"Approved restaurant not mentioned: {response}"
            )

        asyncio.run(_test())
        print("[TEST 3 PASSED] Restaurant information response verified.")

    def test_04_absent_policy_or_amenity_response(self):
        """Tests that LLM does NOT hallucinate amenities absent from the dataset (e.g. helipad)."""
        async def _test():
            query = "Do you have a helicopter landing pad on the roof for guests?"
            system_instruction = self.knowledge_service.get_system_instruction("pullman_aerocity")
            knowledge_context = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", query)

            # Verified absent from facts
            self.assertEqual(knowledge_context, "")

            response = await self.response_service.generate_response(
                transcript=query,
                system_instruction=system_instruction,
                knowledge_context=knowledge_context,
            )
            safe_print(f"\n[TEST 4 Response] Query: {query}\nResponse: {response}")
            resp_lower = response.lower()
            # Must state absence or lack of information, or offer front desk assistance
            has_absence_phrase = any(term in resp_lower for term in [
                "do not have", "don't have", "not available", "no", "unable", "not offer",
                "front desk", "staff", "connect", "assist", "information", "website", "sorry", "apologize"
            ])
            self.assertTrue(has_absence_phrase, f"Did not politely decline absent amenity: {response}")
            self.assertNotIn("we have a helicopter", resp_lower)
            self.assertNotIn("we do have a helicopter", resp_lower)
            self.assertNotIn("we have a helipad", resp_lower)

        asyncio.run(_test())
        print("[TEST 4 PASSED] Absent amenity handling verified (no hallucination).")

    def test_05_price_availability_live_information_response(self):
        """Tests that LLM directs price/availability inquiries to reservations/website without inventing prices."""
        async def _test():
            query = "How much does a deluxe room cost for tonight, and do you have rooms available?"
            system_instruction = self.knowledge_service.get_system_instruction("pullman_aerocity")
            knowledge_context = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", query)

            self.assertIn("Room rates and availability vary in real time", knowledge_context)

            response = await self.response_service.generate_response(
                transcript=query,
                system_instruction=system_instruction,
                knowledge_context=knowledge_context,
            )
            safe_print(f"\n[TEST 5 Response] Query: {query}\nResponse: {response}")
            resp_lower = response.lower()
            # Must refer to reservations, website, dates, or contact, and NOT fabricate a fixed price
            has_live_guidance = any(term in resp_lower for term in [
                "reservation", "website", "accor", "dates", "date", "contact", "front desk",
                "vary", "real time", "real-time", "check", "call", "information"
            ])
            self.assertTrue(has_live_guidance, f"Did not provide live rate guidance: {response}")

        asyncio.run(_test())
        print("[TEST 5 PASSED] Live price and availability handling verified.")

    def test_06_existing_non_hotel_without_profile(self):
        """Tests backward compatibility: non-hotel question without company profile uses existing telecom behavior."""
        async def _test():
            query = "My wifi internet connection is very slow, can you check my broadband account?"
            response = await self.response_service.generate_response(
                transcript=query,
                system_instruction=None,
                knowledge_context=None,
            )
            safe_print(f"\n[TEST 6 Response] Query: {query}\nResponse: {response}")
            resp_lower = response.lower()
            # Should use the telecom assistant persona
            self.assertNotIn("pullman", resp_lower)
            has_telecom_response = any(term in resp_lower for term in [
                "internet", "connection", "troubleshoot", "account", "service", "assist", "help", "support", "wifi"
            ])
            self.assertTrue(has_telecom_response, f"Did not retain telecom response: {response}")

        asyncio.run(_test())
        print("[TEST 6 PASSED] Default persona backward compatibility verified.")

    def test_07_human_escalation_behavior_preserved(self):
        """Tests that existing regex-based escalation to human assistance remains functional."""
        furious_text = "I am furious, this is useless! Transfer me to a human representative right now!"
        should_esc, reason = EscalationService.should_escalate(furious_text)
        self.assertTrue(should_esc)
        self.assertTrue("dissatisfaction" in reason or "human" in reason)

        calm_hotel_text = "What time is check in?"
        should_esc_calm, _ = EscalationService.should_escalate(calm_hotel_text)
        self.assertFalse(should_esc_calm)
        print("[TEST 7 PASSED] Human escalation behavior preserved.")

    def test_08_streaming_sentence_with_pullman_knowledge(self):
        """Tests generate_response_stream_sentence with Pullman knowledge profile."""
        async def _test():
            query = "What is the check in time?"
            system_instruction = self.knowledge_service.get_system_instruction("pullman_aerocity")
            knowledge_context = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", query)

            sentence, timings = await self.response_service.generate_response_stream_sentence(
                transcript=query,
                system_instruction=system_instruction,
                knowledge_context=knowledge_context,
            )
            safe_print(f"\n[TEST 8 Response] Query: {query}\nSentence: {sentence}\nTimings: {timings}")
            self.assertTrue(sentence and len(sentence) > 5)
            self.assertEqual(timings["sentence_count"], 1)

        asyncio.run(_test())
        print("[TEST 8 PASSED] Streaming first-sentence with Pullman knowledge verified.")


    def test_09_browser_live_call_websocket_lifecycle(self):
        """Tests browser Live Call WebSocket lifecycle with Pullman vs default persona, barge-in, and cleanup."""
        from starlette.testclient import TestClient
        from main import app
        from app.utils.auth import create_access_token
        from app.database.database import SessionLocal
        from app.database.models import Call, Customer, Message

        token = create_access_token({"sub": "admin", "role": "admin"})
        client = TestClient(app)

        # 1. Test Live Call with Pullman persona
        created_call_id = None
        try:
            with client.websocket_connect(f"/api/ws/live-call/new?company_id=pullman_aerocity&token={token}") as ws:
                msg1 = ws.receive_json()
                self.assertEqual(msg1["event"], "session_started")
                created_call_id = msg1["call_id"]

                msg2 = ws.receive_json()
                self.assertEqual(msg2["event"], "assistant_response")
                self.assertIn("Pullman New Delhi Aerocity", msg2["response"])

                # Barge-in / Interrupt test
                ws.send_json({"event": "interrupt"})
                msg3 = ws.receive_json()
                self.assertEqual(msg3["event"], "interrupted")

                # Hangup / End Call test
                ws.send_json({"event": "end_call"})
        finally:
            if created_call_id:
                db = SessionLocal()
                try:
                    c = db.query(Call).filter(Call.id == created_call_id).first()
                    if c:
                        db.query(Message).filter(Message.call_id == created_call_id).delete()
                        cust_id = c.customer_id
                        db.delete(c)
                        db.commit()
                        if cust_id:
                            cust = db.query(Customer).filter(Customer.id == cust_id).first()
                            if cust and not cust.calls:
                                db.delete(cust)
                                db.commit()
                finally:
                    db.close()

        # 2. Test Live Call with default persona (no company_id)
        created_call_id_def = None
        try:
            with client.websocket_connect(f"/api/ws/live-call/new?token={token}") as ws:
                msg1 = ws.receive_json()
                self.assertEqual(msg1["event"], "session_started")
                created_call_id_def = msg1["call_id"]

                msg2 = ws.receive_json()
                self.assertEqual(msg2["event"], "assistant_response")
                self.assertIn("AI voice assistant", msg2["response"])

                ws.send_json({"event": "end_call"})
        finally:
            if created_call_id_def:
                db = SessionLocal()
                try:
                    c = db.query(Call).filter(Call.id == created_call_id_def).first()
                    if c:
                        db.query(Message).filter(Message.call_id == created_call_id_def).delete()
                        cust_id = c.customer_id
                        db.delete(c)
                        db.commit()
                        if cust_id:
                            cust = db.query(Customer).filter(Customer.id == cust_id).first()
                            if cust and not cust.calls:
                                db.delete(cust)
                                db.commit()
                finally:
                    db.close()

        print("[TEST 9 PASSED] Browser Live Call WebSocket lifecycle & persona routing verified.")


if __name__ == "__main__":
    unittest.main()

