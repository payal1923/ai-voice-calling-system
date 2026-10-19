import os
import sys
import re
import asyncio
import unittest
from datetime import datetime
from unittest.mock import AsyncMock

sys.path.insert(0, os.path.abspath("."))

from app.services.knowledge_service import KnowledgeService
from app.services.response_service import ResponseService
from app.database.models import Call, Customer, Message, SystemUser
from app.database.database import Base, engine, SessionLocal
from app.utils.auth import create_access_token


class TestLiveVoiceFixesTask12(unittest.TestCase):
    """
    Regression test suite for Task 12:
    1. Complete dining response output without truncation (all 4 venues, complete sentences).
    2. Clear check-in questions return 3:00 PM (including 'checking' STT variation).
    3. Clear check-out questions return 12:00 PM.
    4. Ambiguous timing questions return clarification instead of unrelated answers.
    5. Session-ended events and repeated messages handled safely without 500 error.
    6. Frontend session-ended and audio capture cessation verification.
    """

    def setUp(self):
        self.ks = KnowledgeService()
        self.rs = ResponseService(llm_generator=AsyncMock(return_value="The hotel offers a variety of amenities."))
        self.company_id = "pullman_aerocity"
        self.sys_instruction = self.ks.get_system_instruction(self.company_id)

    # ========================================================================
    # 1. Complete Dining Response Output Without Truncation
    # ========================================================================

    def test_01_dining_overview_completeness(self):
        """Dining overview answers must end with complete sentences and list all 4 venues."""
        queries = [
            "What type of restaurant do you have?",
            "What type of food options do you have?",
            "What dining options do you have?",
            "What restaurants do you have at the hotel?",
        ]
        for q in queries:
            facts_text = self.ks.get_relevant_knowledge(self.company_id, q)
            resp = asyncio.run(self.rs.generate_response(q, knowledge_context=facts_text))
            print(f"\n[Dining Query]: '{q}' -> Response: '{resp}'")

            # Must end with full punctuation
            self.assertTrue(resp.endswith((".", "!", "?")), f"Truncated response for '{q}': {resp}")
            # Must not contain unclosed parenthesis
            self.assertEqual(resp.count("("), resp.count(")"), f"Unmatched parenthesis in '{q}': {resp}")
            # Must not end with the truncated observed fragment
            self.assertFalse(resp.endswith("(curated artisanal."), f"Ended with truncated fragment in '{q}': {resp}")
            # Must include all 4 venues
            for venue in ["Pluck", "Honk", "Café Pluck", "The Chidya Ghar Bar"]:
                self.assertIn(venue, resp, f"Missing venue '{venue}' in '{q}': {resp}")
            # For dining options / type queries, must describe cuisine without cutting off
            if "dining options" in q.lower() or "type of" in q.lower():
                self.assertIn("curated artisanal cocktails and wines", resp)
        print("[PASS] Test 1: Complete dining response output without truncation.")

    def test_02_sanitize_response_parenthesis_truncation_repair(self):
        """Sanitizer repairs dangling parenthetical fragments caused by LLM token limits."""
        truncated_llm_output = (
            "The hotel offers a variety of dining options, including Pluck (seasonal modern European and Indian cuisine), "
            "Honk (pan-Asian street food and dim sum), Café Pluck (24-hour delicatessen and coffee bar), and "
            "The Chidya Ghar Bar (curated artisanal."
        )
        repaired = ResponseService._sanitize_response(
            truncated_llm_output,
            transcript="What type of restaurant do you have?",
            knowledge_context="The hotel offers dining options including Pluck, Honk, Cafe Pluck, and The Chidya Ghar Bar (curated artisanal cocktails and wines)."
        )
        print(f"\n[Repaired Output]: '{repaired}'")
        self.assertTrue(repaired.endswith("."), f"Repaired text does not end with period: {repaired}")
        self.assertEqual(repaired.count("("), repaired.count(")"), f"Unmatched parentheses: {repaired}")
        self.assertIn("(curated artisanal cocktails and wines).", repaired)
        print("[PASS] Test 2: Sanitizer parenthetical repair verified.")

    # ========================================================================
    # 2. Clear Check-In and Check-Out Questions
    # ========================================================================

    def test_03_clear_checkin_questions(self):
        """Clearly recognized check-in questions must return 3:00 PM."""
        checkin_queries = [
            "What time is it checking?",
            "What time is check in?",
            "What is the check-in time?",
            "I want to know about check-in time.",
            "When can I check in?",
        ]
        for q in checkin_queries:
            facts_text = self.ks.get_relevant_knowledge(self.company_id, q)
            resp = asyncio.run(self.rs.generate_response(q, knowledge_context=facts_text))
            print(f"\n[Check-in Query]: '{q}' -> Response: '{resp}'")

            self.assertTrue("3:00 PM" in resp or "15:00" in resp, f"Failed to return 3:00 PM for '{q}': {resp}")
            self.assertTrue("Check-in" in resp or "check-in" in resp, f"Failed to mention check-in for '{q}': {resp}")
            self.assertNotIn("12:00", resp, f"Leaked check-out time into check-in query '{q}': {resp}")
        print("[PASS] Test 3: Clear check-in questions correctly return 3:00 PM.")

    def test_04_clear_checkout_questions(self):
        """Clearly recognized check-out questions must return 12:00 PM."""
        checkout_queries = [
            "And what about check-out time?",
            "What is the check-out time?",
            "When is check out?",
            "What time do I need to check out?",
        ]
        for q in checkout_queries:
            facts_text = self.ks.get_relevant_knowledge(self.company_id, q)
            resp = asyncio.run(self.rs.generate_response(q, knowledge_context=facts_text))
            print(f"\n[Check-out Query]: '{q}' -> Response: '{resp}'")

            self.assertTrue("12:00 PM" in resp or "12:00" in resp, f"Failed to return 12:00 PM for '{q}': {resp}")
            self.assertTrue("Check-out" in resp or "check-out" in resp, f"Failed to mention check-out for '{q}': {resp}")
            self.assertNotIn("3:00", resp, f"Leaked check-in time into check-out query '{q}': {resp}")
        print("[PASS] Test 4: Clear check-out questions correctly return 12:00 PM.")

    # ========================================================================
    # 3. Ambiguous Timing Questions Returning Clarification
    # ========================================================================

    def test_05_ambiguous_timing_and_garbled_queries(self):
        """Ambiguous or garbled timing questions must return clarification instead of unrelated answers."""
        # 1. Ambiguous timing query from actual STT mistranscription
        q_chicken = "What time of the chicken?"
        facts_chicken_text = self.ks.get_relevant_knowledge(self.company_id, q_chicken)
        resp_chicken = asyncio.run(self.rs.generate_response(q_chicken, knowledge_context=facts_chicken_text))
        print(f"\n[Ambiguous Query 1]: '{q_chicken}' -> Response: '{resp_chicken}'")

        self.assertTrue(
            "clarify" in resp_chicken.lower() or "repeat" in resp_chicken.lower(),
            f"Did not clarify for '{q_chicken}': {resp_chicken}"
        )
        self.assertNotIn("3:00", resp_chicken)
        self.assertNotIn("12:00", resp_chicken)

        # 2. Ambiguous room query
        q_room = "Time for a special room for me."
        facts_room_text = self.ks.get_relevant_knowledge(self.company_id, q_room)
        # Crucial check: check_in_out fact must NOT match on 'time' + 'room'
        self.assertNotIn("15:00", (facts_room_text or ""))
        resp_room = asyncio.run(self.rs.generate_response(q_room, knowledge_context=facts_room_text))
        print(f"\n[Ambiguous Query 2]: '{q_room}' -> Response: '{resp_room}'")
        self.assertNotIn("3:00", resp_room)
        self.assertNotIn("12:00", resp_room)
        self.assertTrue(
            "clarify" in resp_room.lower() or "repeat" in resp_room.lower() or "not offer" in resp_room.lower(),
            f"Failed to ask for clarification on '{q_room}': {resp_room}"
        )

        # 3. Garbled non-timing speech turns
        garbled_queries = [
            "Do your hats in, Markle?",
            "Don't give hands for me, go.",
        ]
        for gq in garbled_queries:
            facts_text = self.ks.get_relevant_knowledge(self.company_id, gq)
            resp = asyncio.run(self.rs.generate_response(gq, knowledge_context=facts_text))
            print(f"\n[Garbled Query]: '{gq}' -> Response: '{resp}'")
            self.assertTrue(
                "repeat" in resp.lower() or "didn't" in resp.lower() or "clarify" in resp.lower(),
                f"Did not ask to repeat for '{gq}': {resp}"
            )
        print("[PASS] Test 5: Ambiguous timing and garbled speech correctly return clarification.")

    # ========================================================================
    # 4. Session-Ended Events & Repeated Messages Handled Safely
    # ========================================================================

    def test_06_add_turn_on_already_ended_call_handled_safely(self):
        """Verify that attempting to process a turn on an ended or escalated call is handled safely."""
        from fastapi.testclient import TestClient
        from main import app

        client = TestClient(app)
        db = SessionLocal()
        try:
            customer = Customer(
                name="Ended Call Test Customer",
                phone="+919999999901",
                external_customer_id=f"TASK12_ENDED_TEST_{datetime.utcnow().timestamp()}",
            )
            db.add(customer)
            db.flush()

            ended_call = Call(
                customer_id=customer.id,
                call_type="INBOUND",
                status="RESOLVED",
                started_at=datetime.utcnow(),
                ended_at=datetime.utcnow(),
            )
            db.add(ended_call)
            db.commit()
            db.refresh(ended_call)

            token = create_access_token(data={"sub": "admin", "role": "admin", "uid": 1})
            headers = {"Authorization": f"Bearer {token}"}

            dummy_wav = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
            files = {"file": ("test.wav", dummy_wav, "audio/wav")}

            res = client.post(f"/api/calls/{ended_call.id}/turn", headers=headers, files=files)
            # Must return 400 with a clear message, never a 500 crash
            self.assertEqual(res.status_code, 400)
            data = res.json()
            self.assertIn("already ended", data.get("detail", "").lower())
            print(f"[PASS] Add turn to ended call returned 400 safely: {data.get('detail')}")

        finally:
            db.close()

    def test_07_frontend_ended_call_handlers_present(self):
        """Verify frontend/index.html includes safe session-ended and escalation termination logic."""
        with open("frontend/index.html", "r", encoding="utf-8") as f:
            content = f.read()

        # 1. Session ended event handling in frontend
        self.assertIn("data.event === 'session_ended'", content)
        # 2. Dedicated ended call cleanup
        self.assertIn("handleLiveCallEnded", content)
        # 3. Cease recording when escalated
        self.assertIn("isLiveCallActive = false", content)
        # 4. Suppression of error banner on already ended calls
        self.assertIn("already ended", content.lower())
        # 5. Guarding recorder chunks if call not active
        self.assertIn("!isLiveCallActive || isAssistantSpeaking", content)
        print("[PASS] Test 7: Frontend index.html ended-call event handlers confirmed present.")


if __name__ == "__main__":
    unittest.main()
