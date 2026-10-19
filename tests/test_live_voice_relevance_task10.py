import os
import sys
import asyncio
import unittest

sys.path.insert(0, os.path.abspath("."))

from app.services.knowledge_service import KnowledgeService
from app.services.response_service import ResponseService


class TestLiveVoiceRelevanceTask10(unittest.TestCase):
    """
    Regression test suite for Task 10:
    1. Specific Pluck question returns Pluck facts (no false denial).
    2. Specific Honk question returns Honk facts (no false denial).
    3. Pool opening-hours question does not invent hours.
    4. Unclear pool-hours transcript triggers clarification or a clearly grounded response.
    5. Pool question following a late check-out question does not mention late check-out or spa.
    6. Restaurant question following a pool question does not repeat the pool answer.
    7. General restaurant query returns venue names.
    8. Garbled speech triggers clarification without phonetic guesswork.
    """

    def setUp(self):
        self.ks = KnowledgeService()
        self.rs = ResponseService()
        self.company_id = "pullman_aerocity"
        self.sys_instruction = self.ks.get_system_instruction(self.company_id)

    def test_01_specific_pluck_question_returns_pluck_facts(self):
        """1. A specific Pluck question retrieves Pluck facts and never returns a false denial."""
        queries = [
            "I want information about Pluck restaurant.",
            "Can you tell me about Pluck restaurant?",
            "Tell me about Pluck.",
        ]
        for q in queries:
            kb = self.ks.get_relevant_knowledge(self.company_id, q)
            self.assertIn("pluck", kb.lower(), f"Knowledge retrieval missed Pluck for query '{q}'")
            resp = asyncio.run(self.rs.generate_response(
                transcript=q,
                system_instruction=self.sys_instruction,
                knowledge_context=kb,
            ))
            print(f"\n[Test 1 Pluck Query]: '{q}' -> Response: '{resp}'")
            resp_lower = resp.lower()
            self.assertIn("pluck", resp_lower, f"Pluck not mentioned in response for '{q}'")
            self.assertTrue(
                any(w in resp_lower for w in ["european", "indian", "farm-to-table", "organic"]),
                f"Missing Pluck cuisine/feature details for '{q}'"
            )
            # Must NOT claim lack of information
            self.assertNotIn("don't have", resp_lower, "Falsely claimed lack of information for Pluck!")
            self.assertNotIn("do not have", resp_lower, "Falsely claimed lack of information for Pluck!")
            self.assertNotIn("not available", resp_lower, "Falsely claimed details not available for Pluck!")
            # Isolated from other dining venues
            self.assertNotIn("honk", resp_lower, "Leaked Honk into Pluck response!")
            self.assertNotIn("chidya ghar", resp_lower, "Leaked Chidya Ghar into Pluck response!")
        print("[PASS] Test 1: Specific Pluck question returns verified Pluck facts without false denial.")

    def test_02_specific_honk_question_returns_honk_facts(self):
        """2. A specific Honk question retrieves Honk facts and never returns a false denial."""
        queries = [
            "Can you tell me about Honk?",
            "What kind of food does Honk restaurant serve?",
        ]
        for q in queries:
            kb = self.ks.get_relevant_knowledge(self.company_id, q)
            self.assertIn("honk", kb.lower(), f"Knowledge retrieval missed Honk for query '{q}'")
            resp = asyncio.run(self.rs.generate_response(
                transcript=q,
                system_instruction=self.sys_instruction,
                knowledge_context=kb,
            ))
            print(f"\n[Test 2 Honk Query]: '{q}' -> Response: '{resp}'")
            resp_lower = resp.lower()
            self.assertIn("honk", resp_lower, f"Honk not mentioned in response for '{q}'")
            self.assertTrue(
                any(w in resp_lower for w in ["asian", "dim sum", "sushi", "street food"]),
                f"Missing Honk cuisine details for '{q}'"
            )
            self.assertNotIn("don't have", resp_lower, "Falsely claimed lack of information for Honk!")
            self.assertNotIn("do not have", resp_lower, "Falsely claimed lack of information for Honk!")
            self.assertNotIn("pluck", resp_lower, "Leaked Pluck into Honk response!")
        print("[PASS] Test 2: Specific Honk question returns verified Honk facts without false denial.")

    def test_03_pool_opening_hours_does_not_invent_hours(self):
        """3. Pool opening-hours question does not invent hours and states hours are not available in current info."""
        queries = [
            "What are the swimming pool hours?",
            "When does the swimming pool open?",
            "What time does the pool close?",
            "What are the pool timings?",
        ]
        for q in queries:
            kb = self.ks.get_relevant_knowledge(self.company_id, q)
            resp = asyncio.run(self.rs.generate_response(
                transcript=q,
                system_instruction=self.sys_instruction,
                knowledge_context=kb,
            ))
            print(f"\n[Test 3 Pool Hours Query]: '{q}' -> Response: '{resp}'")
            resp_lower = resp.lower()
            self.assertIn("pool", resp_lower, f"Pool not mentioned in pool hours response for '{q}'")
            # Must explicitly state operating hours are not available in current info / front desk
            self.assertTrue(
                any(p in resp_lower for p in ["not available", "check with the hotel front desk", "front desk"]),
                f"Did not ground lack of pool hours for '{q}'"
            )
            # Must never invent operational hours
            invented_hours = ["9:00", "9 am", "10:00", "10 am", "6:00", "6 am", "8:00", "8 pm", "24 hours", "24-hour", "24/7"]
            for h in invented_hours:
                self.assertNotIn(h, resp_lower, f"Invented pool operating hours '{h}' for '{q}'!")
            # Isolated from spa and check-out
            self.assertNotIn("spa", resp_lower, "Leaked spa into pool hours response!")
            self.assertNotIn("check-out", resp_lower, "Leaked check-out into pool hours response!")
            self.assertNotIn("check-in", resp_lower, "Leaked check-in into pool hours response!")
        print("[PASS] Test 3: Pool opening-hours question is grounded and does not invent hours.")

    def test_04_unclear_pool_hours_transcript_triggers_clarification_or_grounded_response(self):
        """4. Unclear pool-hours transcript triggers clarification or a clearly grounded response."""
        unclear_queries = [
            "swimming pool hours um...",
            "pool timing open what?",
            "pool hours?",
        ]
        for q in unclear_queries:
            kb = self.ks.get_relevant_knowledge(self.company_id, q)
            resp = asyncio.run(self.rs.generate_response(
                transcript=q,
                system_instruction=self.sys_instruction,
                knowledge_context=kb,
            ))
            print(f"\n[Test 4 Unclear Pool Query]: '{q}' -> Response: '{resp}'")
            resp_lower = resp.lower()
            is_clarification = any(p in resp_lower for p in ["clarify", "didn't catch", "repeat"])
            is_grounded_pool = "pool" in resp_lower and any(p in resp_lower for p in ["not available", "front desk", "heated outdoor"])
            self.assertTrue(
                is_clarification or is_grounded_pool,
                f"Response for unclear pool query '{q}' is neither clarification nor grounded pool answer: '{resp}'"
            )
            self.assertNotIn("spa", resp_lower, "Leaked spa details into unclear pool question!")
            self.assertNotIn("late check-out", resp_lower, "Leaked late check-out into unclear pool question!")
        print("[PASS] Test 4: Unclear pool-hours transcript triggers clarification or grounded response.")

    def test_05_pool_question_following_late_checkout_does_not_mention_late_checkout(self):
        """5. A pool question following a late check-out turn does not mention late check-out or spa."""
        history = (
            "Customer: Can I check out later?\n"
            "Assistant: Standard check-out is at 12:00 PM (12:00). Late check-out is subject to room availability "
            "upon request and must be confirmed directly with the hotel front desk."
        )
        pool_queries = [
            "Do you have a swimming pool?",
            "What are the swimming pool hours?",
        ]
        for q in pool_queries:
            kb = self.ks.get_relevant_knowledge(self.company_id, q)
            resp = asyncio.run(self.rs.generate_response(
                transcript=q,
                conversation_context=history,
                system_instruction=self.sys_instruction,
                knowledge_context=kb,
            ))
            print(f"\n[Test 5 Pool following Late Check-out]: Query: '{q}' -> Response: '{resp}'")
            resp_lower = resp.lower()
            self.assertIn("pool", resp_lower, "Pool not mentioned in pool response!")
            self.assertNotIn("late check-out", resp_lower, "Leaked late check-out into pool response!")
            self.assertNotIn("checkout", resp_lower, "Leaked checkout into pool response!")
            self.assertNotIn("12:00", resp, "Leaked 12:00 PM check-out time into pool response!")
            self.assertNotIn("availability", resp_lower, "Leaked availability disclaimer into pool response!")
            self.assertNotIn("spa", resp_lower, "Leaked spa into pool response!")
        print("[PASS] Test 5: Pool question cleanly isolated from preceding late check-out turn.")

    def test_06_restaurant_question_following_pool_does_not_repeat_pool(self):
        """6. A restaurant question following a pool turn does not repeat the pool answer."""
        history = (
            "Customer: Do you have a swimming pool?\n"
            "Assistant: Yes, the hotel offers a heated outdoor swimming pool for relaxation."
        )
        dining_queries = [
            "I want information about Pluck restaurant.",
            "What dining options do you have?",
        ]
        for q in dining_queries:
            kb = self.ks.get_relevant_knowledge(self.company_id, q)
            resp = asyncio.run(self.rs.generate_response(
                transcript=q,
                conversation_context=history,
                system_instruction=self.sys_instruction,
                knowledge_context=kb,
            ))
            print(f"\n[Test 6 Dining following Pool]: Query: '{q}' -> Response: '{resp}'")
            resp_lower = resp.lower()
            self.assertNotIn("swimming pool", resp_lower, "Repeated previous swimming pool answer on dining query!")
            self.assertNotIn("heated outdoor", resp_lower, "Repeated pool attributes on dining query!")
            self.assertTrue(
                any(v in resp_lower for v in ["pluck", "honk", "dining", "cuisine"]),
                "Missing dining facts in response!"
            )
        print("[PASS] Test 6: Dining query following pool query does not repeat pool answer.")

    def test_07_general_restaurant_query_returns_venue_names(self):
        """7. General restaurant query returns concise venue names."""
        q = "What is your restaurant name?"
        kb = self.ks.get_relevant_knowledge(self.company_id, q)
        resp = asyncio.run(self.rs.generate_response(
            transcript=q,
            system_instruction=self.sys_instruction,
            knowledge_context=kb,
        ))
        print(f"\n[Test 7 Restaurant Names]: '{resp}'")
        resp_lower = resp.lower()
        self.assertTrue(
            all(v in resp_lower for v in ["pluck", "honk", "chidya ghar"]),
            "Missing one or more venue names in general restaurant name query!"
        )
        self.assertNotIn("seasonal modern european", resp_lower, "Rambled with long cuisine descriptions!")
        print("[PASS] Test 7: General restaurant query returns concise venue names.")

    def test_08_unclear_garbled_speech_triggers_clarification(self):
        """8. Incomplete or garbled speech triggers clarification rather than guessing."""
        garbled_transcripts = [
            "Booyo helps off.",
            "asdf ghjkl",
            "mumble mumble something",
        ]
        for g in garbled_transcripts:
            kb = self.ks.get_relevant_knowledge(self.company_id, g)
            resp = asyncio.run(self.rs.generate_response(
                transcript=g,
                system_instruction=self.sys_instruction,
                knowledge_context=kb,
            ))
            print(f"\n[Test 8 Garbled]: '{g}' -> Response: '{resp}'")
            resp_lower = resp.lower()
            self.assertTrue(
                any(p in resp_lower for p in ["didn't quite catch", "didn't catch", "repeat", "clarify"]),
                f"Did not request clarification for garbled transcript '{g}'!"
            )
            # Must not guess unlisted facility or invent hours
            self.assertNotIn("yes", resp_lower.split()[:2], f"Falsely affirmed garbled query '{g}'!")
        print("[PASS] Test 8: Garbled speech triggers clarification without guessing.")


if __name__ == "__main__":
    unittest.main()
