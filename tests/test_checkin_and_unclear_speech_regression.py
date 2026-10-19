import os
import sys
import asyncio
import unittest

sys.path.insert(0, os.path.abspath("."))

from app.services.knowledge_service import KnowledgeService
from app.services.response_service import ResponseService


class TestCheckinAndUnclearSpeechRegression(unittest.TestCase):
    """
    Regression test suite for Task 6:
    Verifies reliable check-in/check-out timing answers, protection against hallucinations
    from garbled audio (e.g. 'Booyo helps off.'), and preservation of legitimate amenity answers.
    """

    def setUp(self):
        self.ks = KnowledgeService()
        self.rs = ResponseService()
        self.company_id = "pullman_aerocity"
        self.sys_instruction = self.ks.get_system_instruction(self.company_id)

        self.kb_checkin = self.ks.get_relevant_knowledge(self.company_id, "What time is check-in?")
        self.kb_dining = self.ks.get_relevant_knowledge(self.company_id, "What dining options do you have?")
        self.kb_pool = self.ks.get_relevant_knowledge(self.company_id, "Do you have swimming pool?")
        self.kb_gym = self.ks.get_relevant_knowledge(self.company_id, "Do you have a gym?")

    def test_01_clear_checkin_question(self):
        """1. Clear check-in question (and phonetic STT variants) returns 3:00 PM (15:00)."""
        variations = [
            "what time is check-in",
            "and work time in check-in.",
            "when is check-in?",
            "check in time",
        ]
        for q in variations:
            resp = asyncio.run(self.rs.generate_response(
                transcript=q,
                conversation_context="",
                system_instruction=self.sys_instruction,
                knowledge_context=self.kb_checkin,
            ))
            print(f"\n[Test 1 Query]: '{q}' -> Response: '{resp}'")
            self.assertTrue(
                "3:00 pm" in resp.lower() or "15:00" in resp or "3:00" in resp,
                f"Check-in time (3:00 PM / 15:00) missing for query '{q}'!"
            )
            self.assertNotIn("24-hour service during check-in", resp.lower(), "Unsupported 24-hour service claim generated!")
            self.assertNotIn("12:00", resp, "Confused check-in with check-out time!")
        print("[PASS] Test 1: Clear check-in timing and STT variations verified.")

    def test_02_checkout_followup(self):
        """2. Check-out follow-up returns 12:00 PM (12:00) without confusing with check-in."""
        history = "Customer: what time is check-in\nAssistant: Check-in at Pullman New Delhi Aerocity is at 3:00 PM (15:00)."
        variations = [
            "And what about check-out?",
            "what time is check-out",
            "when is check-out?",
        ]
        for q in variations:
            resp = asyncio.run(self.rs.generate_response(
                transcript=q,
                conversation_context=history,
                system_instruction=self.sys_instruction,
                knowledge_context=self.kb_checkin,
            ))
            print(f"\n[Test 2 Query]: '{q}' -> Response: '{resp}'")
            self.assertTrue(
                "12:00 pm" in resp.lower() or "12:00" in resp or "noon" in resp.lower(),
                f"Check-out time (12:00 PM) missing for query '{q}'!"
            )
            self.assertNotIn("3:00", resp, "Confused check-out with check-in time!")
        print("[PASS] Test 2: Check-out follow-up verified.")

    def test_03_garbled_transcript_clarification(self):
        """3. Garbled transcript 'Booyo helps off.' returns a clarification request, not a fabricated facility."""
        garbled_queries = [
            "Booyo helps off.",
            "asdf ghjkl",
            "help off",
        ]
        for q in garbled_queries:
            resp = asyncio.run(self.rs.generate_response(
                transcript=q,
                conversation_context="Customer: Do you have a gym?\nAssistant: Yes, the Fit Lounge is a 24/7 fitness center.",
                system_instruction=self.sys_instruction,
                knowledge_context=None,
            ))
            print(f"\n[Test 3 Garbled]: '{q}' -> Response: '{resp}'")
            res_lower = resp.lower()
            self.assertTrue(
                any(p in res_lower for p in ["didn't catch", "didn't quite catch", "repeat", "clarify"]),
                f"Did not return clarification request for garbled input '{q}'!"
            )
            self.assertNotIn("booga", res_lower, "Hallucinated 'booga booga' service!")
            self.assertNotIn("we do not offer that facility", res_lower, "Guessed amenity from garbled speech!")
        print("[PASS] Test 3: Garbled audio correctly returns clarification request.")

    def test_04_clear_question_unsupported_facility(self):
        """4. Clear question about an unsupported facility returns safe polite refusal, not an unintelligible clarification."""
        clear_unsupported = [
            "Do you have an ice skating rink?",
            "Can I book a helicopter?",
            "Is there a bowling alley?",
        ]
        for q in clear_unsupported:
            resp = asyncio.run(self.rs.generate_response(
                transcript=q,
                conversation_context="Customer: Do you have swimming pool?\nAssistant: Yes, the hotel offers a heated outdoor swimming pool.",
                system_instruction=self.sys_instruction,
                knowledge_context=None,
            ))
            print(f"\n[Test 4 Unsupported]: '{q}' -> Response: '{resp}'")
            res_lower = resp.lower()
            self.assertTrue(
                any(p in res_lower for p in ["do not offer", "don't have", "do not have", "not confirmed", "front desk"]),
                f"Did not return safe unlisted facility statement for '{q}'!"
            )
            self.assertNotIn("didn't quite catch that", res_lower, "Mistook clear unsupported question for garbled audio!")
        print("[PASS] Test 4: Clear question about unsupported facility safely handled.")

    def test_05_pool_dining_checkin_topic_changes(self):
        """5. Multi-turn topic sequence (Pool -> Dining -> Check-in) never repeats prior answers."""
        # Turn 1: Pool
        resp_pool = asyncio.run(self.rs.generate_response(
            transcript="Do you have swimming pool?",
            conversation_context="",
            system_instruction=self.sys_instruction,
            knowledge_context=self.kb_pool,
        ))
        print(f"\n[Test 5 Turn 1 (Pool)]: '{resp_pool}'")
        self.assertIn("pool", resp_pool.lower())

        # Turn 2: Dining (with Pool in history)
        history_t2 = f"Customer: Do you have swimming pool?\nAssistant: {resp_pool}"
        resp_dining = asyncio.run(self.rs.generate_response(
            transcript="What dining options do you have?",
            conversation_context=history_t2,
            system_instruction=self.sys_instruction,
            knowledge_context=self.kb_dining,
        ))
        print(f"[Test 5 Turn 2 (Dining)]: '{resp_dining}'")
        self.assertNotIn("swimming pool", resp_dining.lower(), "Turn 2 repeated pool answer!")
        self.assertTrue(any(w in resp_dining.lower() for w in ["pluck", "honk", "dining", "venues"]))

        # Turn 3: Check-in (with Dining in history)
        history_t3 = f"{history_t2}\nCustomer: What dining options do you have?\nAssistant: {resp_dining}"
        resp_checkin = asyncio.run(self.rs.generate_response(
            transcript="and work time in check-in.",
            conversation_context=history_t3,
            system_instruction=self.sys_instruction,
            knowledge_context=self.kb_checkin,
        ))
        print(f"[Test 5 Turn 3 (Check-in)]: '{resp_checkin}'")
        self.assertNotIn("dining", resp_checkin.lower(), "Turn 3 repeated dining answer!")
        self.assertNotIn("swimming pool", resp_checkin.lower(), "Turn 3 repeated pool answer!")
        self.assertTrue("3:00" in resp_checkin or "15:00" in resp_checkin, "Turn 3 missed 3:00 PM check-in time!")
        print("[PASS] Test 5: Multi-turn topic changes (Pool -> Dining -> Check-in) verified.")

    def test_06_gym_question_fit_lounge_facts(self):
        """6. Gym question grounds in Fit Lounge facts, never repeating prior pool answer."""
        history = "Customer: Do you have swimming pool?\nAssistant: Yes, the hotel offers a heated outdoor swimming pool for relaxation."
        resp_gym = asyncio.run(self.rs.generate_response(
            transcript="Do you have a gym?",
            conversation_context=history,
            system_instruction=self.sys_instruction,
            knowledge_context=self.kb_gym,
        ))
        print(f"\n[Test 6 Gym]: '{resp_gym}'")
        res_lower = resp_gym.lower()
        self.assertNotIn("swimming pool", res_lower, "Gym response repeated pool answer!")
        self.assertTrue(
            "fit lounge" in res_lower or "fitness" in res_lower or "cardio" in res_lower or "gym" in res_lower,
            "Gym response did not mention fitness center facts!"
        )
        print("[PASS] Test 6: Gym query grounded in Fit Lounge facts without repeating pool.")

    def test_07_relevant_existing_responses_intact(self):
        """7. Verifies that legitimate relevant responses are preserved without modification."""
        # Case A: Valid pool answer
        pool_ans = "Yes, the hotel offers a heated outdoor swimming pool for relaxation."
        clean_pool = self.rs._sanitize_response(
            response_text=pool_ans,
            transcript="Do you have swimming pool?",
            conversation_context="Customer: Hello\nAssistant: Welcome to Pullman.",
            knowledge_context=self.kb_pool,
        )
        self.assertEqual(clean_pool, pool_ans)

        # Case B: Valid dining answer
        dining_ans = "Pullman New Delhi Aerocity features four distinct dining venues: Pluck, Honk, Cafe Pluck, and The Chidya Ghar Bar."
        clean_dining = self.rs._sanitize_response(
            response_text=dining_ans,
            transcript="What dining options do you have?",
            conversation_context="Customer: Do you have swimming pool?\nAssistant: Yes, the hotel offers a pool.",
            knowledge_context=self.kb_dining,
        )
        self.assertEqual(clean_dining, dining_ans)
        print("[PASS] Test 7: Legitimate relevant responses preserved intact.")


if __name__ == "__main__":
    unittest.main()
