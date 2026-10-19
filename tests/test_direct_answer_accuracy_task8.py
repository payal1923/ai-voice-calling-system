import os
import sys
import asyncio
import unittest

sys.path.insert(0, os.path.abspath("."))

from app.services.knowledge_service import KnowledgeService
from app.services.response_service import ResponseService


class TestDirectAnswerAccuracyTask8(unittest.TestCase):
    """
    Regression test suite for Task 8:
    1. Restaurant-name question returns concise venue names.
    2. Pluck question focuses on Pluck only.
    3. Honk question focuses on Honk only.
    4. Gym response does not include check-in/check-out details.
    5. Unverified 24-hour service is never claimed.
    6. Late check-out availability is not falsely confirmed.
    7. Unclear timing transcripts trigger clarification when needed.
    8. Existing pool, dining, gym, and timing tests still pass.
    """

    def setUp(self):
        self.ks = KnowledgeService()
        self.rs = ResponseService()
        self.company_id = "pullman_aerocity"
        self.sys_instruction = self.ks.get_system_instruction(self.company_id)

    def test_01_restaurant_name_question_returns_concise_venue_names(self):
        """1. Restaurant-name question returns concise venue names without long cuisine descriptions."""
        queries = [
            "What is your restaurant name?",
            "What are your restaurant names?",
            "What restaurants do you have?",
        ]
        for q in queries:
            kb = self.ks.get_relevant_knowledge(self.company_id, q)
            resp = asyncio.run(self.rs.generate_response(
                transcript=q,
                system_instruction=self.sys_instruction,
                knowledge_context=kb,
            ))
            print(f"\n[Test 1 Query]: '{q}' -> Response: '{resp}'")
            resp_lower = resp.lower()
            self.assertTrue(all(v in resp_lower for v in ["pluck", "honk", "chidya ghar"]), f"Missing venue names for query '{q}'")
            # Long cuisine descriptions should not be included
            self.assertNotIn("seasonal modern european", resp_lower, "Included long cuisine descriptions for simple name query!")
            self.assertNotIn("street food and dim sum", resp_lower, "Included long cuisine descriptions for simple name query!")
        print("[PASS] Test 1: Restaurant-name question returns concise venue names.")

    def test_02_pluck_question_focuses_on_pluck(self):
        """2. Pluck question focuses on Pluck only, without listing other venues."""
        q = "Can you tell me about Pluck restaurant?"
        kb = self.ks.get_relevant_knowledge(self.company_id, q)
        resp = asyncio.run(self.rs.generate_response(
            transcript=q,
            system_instruction=self.sys_instruction,
            knowledge_context=kb,
        ))
        print(f"\n[Test 2 Pluck]: '{resp}'")
        resp_lower = resp.lower()
        self.assertIn("pluck", resp_lower, "Pluck not mentioned in Pluck response!")
        self.assertNotIn("honk", resp_lower, "Leaked Honk into Pluck response!")
        self.assertNotIn("chidya ghar", resp_lower, "Leaked Chidya Ghar into Pluck response!")
        print("[PASS] Test 2: Pluck question focuses strictly on Pluck.")

    def test_03_honk_question_focuses_on_honk(self):
        """3. Honk question focuses on Honk only, without listing other venues."""
        q = "What type of food does Honk serve?"
        kb = self.ks.get_relevant_knowledge(self.company_id, q)
        resp = asyncio.run(self.rs.generate_response(
            transcript=q,
            system_instruction=self.sys_instruction,
            knowledge_context=kb,
        ))
        print(f"\n[Test 3 Honk]: '{resp}'")
        resp_lower = resp.lower()
        self.assertIn("honk", resp_lower, "Honk not mentioned in Honk response!")
        self.assertTrue(any(w in resp_lower for w in ["asian", "dim sum", "sushi", "street food"]), "Missing Honk cuisine details!")
        self.assertNotIn("chidya ghar", resp_lower, "Leaked Chidya Ghar into Honk response!")
        print("[PASS] Test 3: Honk question focuses strictly on Honk.")

    def test_04_gym_response_does_not_include_checkin_checkout(self):
        """4. Gym response does not include check-in/check-out details even with check-in history."""
        q = "Do you know how a gym?"
        kb = self.ks.get_relevant_knowledge(self.company_id, q)
        history = "Customer: What time is check-in?\nAssistant: Check-in at Pullman New Delhi Aerocity is at 3:00 PM (15:00) and check-out is at 12:00 PM (12:00)."
        resp = asyncio.run(self.rs.generate_response(
            transcript=q,
            conversation_context=history,
            system_instruction=self.sys_instruction,
            knowledge_context=kb,
        ))
        print(f"\n[Test 4 Gym]: '{resp}'")
        resp_lower = resp.lower()
        self.assertTrue(any(w in resp_lower for w in ["fit lounge", "fitness", "gym"]), "Missing gym confirmation!")
        self.assertNotIn("3:00", resp, "Leaked 3:00 PM check-in time into gym response!")
        self.assertNotIn("12:00", resp, "Leaked 12:00 PM check-out time into gym response!")
        self.assertNotIn("check-in", resp_lower, "Leaked check-in mention into gym response!")
        self.assertNotIn("check-out", resp_lower, "Leaked check-out mention into gym response!")
        print("[PASS] Test 4: Gym response cleanly isolated from check-in/out times.")

    def test_05_unverified_24_hour_service_is_never_claimed(self):
        """5. Unverified 24-hour service is never claimed."""
        queries = [
            "It's traffic block open for 24 hours.",
            "Is the swimming pool open 24 hours?",
        ]
        for q in queries:
            kb = self.ks.get_relevant_knowledge(self.company_id, q)
            resp = asyncio.run(self.rs.generate_response(
                transcript=q,
                system_instruction=self.sys_instruction,
                knowledge_context=kb,
            ))
            print(f"\n[Test 5 Query]: '{q}' -> Response: '{resp}'")
            resp_lower = resp.lower()
            if "traffic block" in q:
                self.assertNotIn("24-hour", resp_lower, "Falsely claimed 24-hour service for traffic block!")
                self.assertNotIn("24 hours", resp_lower, "Falsely claimed 24 hours for traffic block!")
                self.assertTrue(any(p in resp_lower for p in ["didn't catch", "repeat", "clarify"]), "Did not request clarification for garbled traffic block query!")
            if "swimming pool" in q:
                self.assertNotIn("24-hour", resp_lower, "Falsely claimed 24-hour service for swimming pool!")
                self.assertNotIn("24 hours", resp_lower, "Falsely claimed 24 hours for swimming pool!")
        print("[PASS] Test 5: Unverified 24-hour service claims completely suppressed.")

    def test_06_late_checkout_availability_not_falsely_confirmed(self):
        """6. Late check-out availability is not falsely confirmed; standard time given & unconfirmed status stated."""
        q = "Can I check out later?"
        kb = self.ks.get_relevant_knowledge(self.company_id, q)
        resp = asyncio.run(self.rs.generate_response(
            transcript=q,
            system_instruction=self.sys_instruction,
            knowledge_context=kb,
        ))
        print(f"\n[Test 6 Late Check-out]: '{resp}'")
        resp_lower = resp.lower()
        self.assertIn("12:00", resp, "Missing standard check-out time (12:00 PM) in late check-out inquiry!")
        self.assertTrue(
            any(p in resp_lower for p in ["subject to", "availability", "must be confirmed", "front desk"]),
            "Did not clarify that late check-out is unconfirmed and subject to availability!"
        )
        self.assertNotIn("offers an early check-out option", resp_lower, "Contradictory hallucination detected!")
        print("[PASS] Test 6: Late check-out safely handled without false confirmation.")

    def test_07_unclear_timing_triggers_clarification(self):
        """7. Unclear timing transcripts trigger clarification when needed rather than facility refusal."""
        q = "What time is taking?"
        kb = self.ks.get_relevant_knowledge(self.company_id, q)
        resp = asyncio.run(self.rs.generate_response(
            transcript=q,
            system_instruction=self.sys_instruction,
            knowledge_context=kb,
        ))
        print(f"\n[Test 7 Unclear Timing]: '{resp}'")
        resp_lower = resp.lower()
        self.assertTrue(
            any(p in resp_lower for p in ["clarify what timing", "didn't catch", "repeat", "clarify"]),
            "Did not ask for clarification on unclear timing question!"
        )
        self.assertNotIn("do not offer that facility", resp_lower, "Gave inappropriate facility refusal for unclear timing query!")
        self.assertNotIn("taking is currently available", resp_lower, "Hallucinated 'taking' availability!")
        print("[PASS] Test 7: Unclear timing correctly triggers clarification request.")

    def test_08_existing_pool_dining_gym_timing_pass(self):
        """8. Existing pool, dining, gym, and timing tests still pass reliably."""
        # Pool query
        kb_pool = self.ks.get_relevant_knowledge(self.company_id, "Do you have a swimming pool?")
        resp_pool = asyncio.run(self.rs.generate_response(
            transcript="Do you have a swimming pool?",
            system_instruction=self.sys_instruction,
            knowledge_context=kb_pool,
        ))
        self.assertIn("pool", resp_pool.lower())

        # General Dining query
        kb_dining = self.ks.get_relevant_knowledge(self.company_id, "What dining options do you have?")
        resp_dining = asyncio.run(self.rs.generate_response(
            transcript="What dining options do you have?",
            system_instruction=self.sys_instruction,
            knowledge_context=kb_dining,
        ))
        self.assertTrue(any(v in resp_dining.lower() for v in ["pluck", "honk", "dining", "venues"]))

        # Check-in query
        kb_checkin = self.ks.get_relevant_knowledge(self.company_id, "What time is check-in?")
        resp_checkin = asyncio.run(self.rs.generate_response(
            transcript="What time is check-in?",
            system_instruction=self.sys_instruction,
            knowledge_context=kb_checkin,
        ))
        self.assertTrue("3:00" in resp_checkin or "15:00" in resp_checkin)

        # Check-out query
        kb_checkout = self.ks.get_relevant_knowledge(self.company_id, "What time is check-out?")
        resp_checkout = asyncio.run(self.rs.generate_response(
            transcript="What time is check-out?",
            system_instruction=self.sys_instruction,
            knowledge_context=kb_checkout,
        ))
        self.assertIn("12:00", resp_checkout)
        print("[PASS] Test 8: Core amenities, dining overview, and check-in/out times verified.")


if __name__ == "__main__":
    unittest.main()
