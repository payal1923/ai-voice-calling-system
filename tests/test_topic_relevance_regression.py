import os
import sys
import asyncio
import unittest

sys.path.insert(0, os.path.abspath("."))

from app.services.knowledge_service import KnowledgeService
from app.services.response_service import ResponseService
from app.services.conversation_context_service import ConversationContextService
from app.database.models import Call, Customer, Message


class TestTopicRelevanceRegression(unittest.TestCase):
    """
    Regression test suite for Task 3:
    Verifies that the AI never repeats answers from previous topics when a caller
    switches to a new question, and properly provides factually grounded answers.
    """

    def setUp(self):
        self.ks = KnowledgeService()
        self.rs = ResponseService()
        self.ctx = ConversationContextService()
        self.company_id = "pullman_aerocity"
        self.sys_instruction = self.ks.get_system_instruction(self.company_id)

        # Approved Pullman knowledge snippets
        self.kb_pool = self.ks.get_relevant_knowledge(self.company_id, "Do you have a swimming pool?")
        self.kb_dining = self.ks.get_relevant_knowledge(self.company_id, "What dining options do you have?")
        self.kb_checkin = self.ks.get_relevant_knowledge(self.company_id, "What time is check-in?")
        self.kb_absent = self.ks.get_relevant_knowledge(self.company_id, "Do you have an ice skating rink?")

        # Common simulated conversation histories
        self.history_pool = (
            "Customer: Do you have a swimming pool?\n"
            "Assistant: Yes, the hotel offers a heated outdoor swimming pool for relaxation."
        )
        self.history_dining = (
            "Customer: What dining options do you have?\n"
            "Assistant: Pullman New Delhi Aerocity features four distinct dining venues: "
            "Pluck, Honk, Cafe Pluck, and The Chidya Ghar Bar."
        )
        self.history_checkin = (
            "Customer: What time is check-in?\n"
            "Assistant: The check-in time at Pullman New Delhi Aerocity is 3:00 PM (15:00)."
        )

    def test_01_swimming_pool_followed_by_dining_repetition_rejected(self):
        """
        Customer: 'Do you have a swimming pool?' -> answered pool.
        Customer: 'What dining options do you have?'
        LLM attempts to repeat the pool answer.
        The relevance check must reject the pool answer and return approved dining facts.
        """
        stale_pool_output = "Yes, the hotel offers a heated outdoor swimming pool for relaxation."
        result = self.rs._sanitize_response(
            response_text=stale_pool_output,
            transcript="What dining options do you have?",
            conversation_context=self.history_pool,
            knowledge_context=self.kb_dining,
        )

        res_lower = result.lower()
        print(f"\n[Test 1 Result]: '{result}'")
        self.assertNotIn("swimming pool", res_lower, "Repeated previous swimming pool answer!")
        self.assertNotIn("pool", res_lower, "Contained stale pool reference!")
        self.assertTrue(
            "dining venues" in res_lower or "pluck" in res_lower or "honk" in res_lower,
            "Did not ground response in approved dining facts!"
        )
        print("[PASS] Test 1: Pool -> Dining stale repetition successfully rejected and grounded in dining facts.")

    def test_02_dining_followed_by_checkin_repetition_rejected(self):
        """
        Customer: 'What dining options do you have?' -> answered dining venues.
        Customer: 'What time is check-in?'
        LLM attempts to repeat the dining venues answer.
        The relevance check must reject the dining answer and return the check-in time (3:00 PM / 15:00).
        """
        stale_dining_output = (
            "Pullman New Delhi Aerocity features four distinct dining venues: "
            "Pluck, Honk, Cafe Pluck, and The Chidya Ghar Bar."
        )
        result = self.rs._sanitize_response(
            response_text=stale_dining_output,
            transcript="What time is check-in?",
            conversation_context=self.history_dining,
            knowledge_context=self.kb_checkin,
        )

        res_lower = result.lower()
        print(f"\n[Test 2 Result]: '{result}'")
        self.assertNotIn("dining", res_lower, "Repeated previous dining venues answer!")
        self.assertNotIn("pluck", res_lower, "Contained stale restaurant name!")
        self.assertTrue(
            "3:00 pm" in res_lower or "15:00" in res_lower or "3:00" in res_lower,
            "Did not provide check-in time!"
        )
        print("[PASS] Test 2: Dining -> Check-in stale repetition successfully rejected and check-in time returned.")

    def test_03_checkin_followed_by_swimming_pool_repetition_rejected(self):
        """
        Customer: 'What time is check-in?' -> answered 3:00 PM.
        Customer: 'Do you have a swimming pool?'
        LLM attempts to repeat the check-in time answer.
        The relevance check must reject the check-in answer and return pool confirmation.
        """
        stale_checkin_output = "The check-in time at Pullman New Delhi Aerocity is 3:00 PM (15:00)."
        result = self.rs._sanitize_response(
            response_text=stale_checkin_output,
            transcript="Do you have a swimming pool?",
            conversation_context=self.history_checkin,
            knowledge_context=self.kb_pool,
        )

        res_lower = result.lower()
        print(f"\n[Test 3 Result]: '{result}'")
        self.assertNotIn("check-in", res_lower, "Repeated previous check-in answer!")
        self.assertNotIn("15:00", res_lower, "Contained stale check-in time!")
        self.assertTrue(
            "swimming pool" in res_lower or "pool" in res_lower,
            "Did not confirm swimming pool!"
        )
        print("[PASS] Test 3: Check-in -> Pool stale repetition successfully rejected and pool confirmed.")

    def test_04_relevant_answers_remain_unchanged(self):
        """
        Verifies that legitimate, relevant answers generated by the model are preserved
        and NOT rejected by overly broad keyword matching.
        """
        # Case 4a: Legitimate dining answer after pool inquiry
        valid_dining_answer = (
            "Pullman New Delhi Aerocity features four distinct dining venues: "
            "Pluck, Honk, Cafe Pluck, and The Chidya Ghar Bar."
        )
        res_dining = self.rs._sanitize_response(
            response_text=valid_dining_answer,
            transcript="What dining options do you have?",
            conversation_context=self.history_pool,
            knowledge_context=self.kb_dining,
        )
        self.assertEqual(res_dining, valid_dining_answer, "Valid dining answer was altered!")

        # Case 4b: Legitimate pool answer after check-in inquiry
        valid_pool_answer = "Yes, the hotel offers a heated outdoor swimming pool for relaxation."
        res_pool = self.rs._sanitize_response(
            response_text=valid_pool_answer,
            transcript="Do you have a swimming pool?",
            conversation_context=self.history_checkin,
            knowledge_context=self.kb_pool,
        )
        self.assertEqual(res_pool, valid_pool_answer, "Valid pool answer was altered!")

        # Case 4c: Legitimate check-in answer after dining inquiry
        valid_checkin_answer = "The check-in time at Pullman New Delhi Aerocity is 3:00 PM (15:00)."
        res_checkin = self.rs._sanitize_response(
            response_text=valid_checkin_answer,
            transcript="What time is check-in?",
            conversation_context=self.history_dining,
            knowledge_context=self.kb_checkin,
        )
        self.assertEqual(res_checkin, valid_checkin_answer, "Valid check-in answer was altered!")
        print("[PASS] Test 4: All legitimate relevant answers preserved without unwanted modification.")

    def test_05_missing_facts_and_unrelated_questions(self):
        """
        Customer asks about an unlisted facility (e.g. ice skating rink) after pool query.
        The LLM repeats the pool answer or affirms the unlisted facility.
        The relevance check must reject repetition, never invent amenities, and safely state
        the facility is not offered.
        """
        # Scenario: LLM repeats prior pool answer on ice skating inquiry
        stale_pool_output = "Yes, the hotel offers a heated outdoor swimming pool for relaxation."
        result = self.rs._sanitize_response(
            response_text=stale_pool_output,
            transcript="Do you have an ice skating rink?",
            conversation_context=self.history_pool,
            knowledge_context=self.kb_absent,
        )

        res_lower = result.lower()
        print(f"\n[Test 5 Result]: '{result}'")
        self.assertNotIn("swimming pool", res_lower, "Repeated prior pool answer on absent amenity inquiry!")
        self.assertNotIn("yes, we do have an ice skating", res_lower, "Hallucinated absent amenity!")
        self.assertTrue(
            "do not offer" in res_lower or "front desk" in res_lower,
            "Did not provide safe absent-amenity fallback!"
        )
        print("[PASS] Test 5: Missing facts / unrelated questions safely handled without hallucination.")

    def test_06_live_end_to_end_generate_response_flow(self):
        """
        Executes ResponseService.generate_response across simulated multi-turn customer calls
        to verify that the full generation pipeline produces relevant, grounded responses.
        """
        # Turn 1: Customer asks for swimming pool
        resp_turn1 = asyncio.run(self.rs.generate_response(
            transcript="Do you have a swimming pool?",
            conversation_context="",
            system_instruction=self.sys_instruction,
            knowledge_context=self.kb_pool,
        ))
        print(f"\n[Live E2E Turn 1 (Pool)]: '{resp_turn1}'")
        self.assertIn("pool", resp_turn1.lower(), "Turn 1 did not confirm pool!")

        # Turn 2: Customer asks for dining options with Turn 1 in context
        history = f"Customer: Do you have a swimming pool?\nAssistant: {resp_turn1}"
        resp_turn2 = asyncio.run(self.rs.generate_response(
            transcript="What dining options do you have?",
            conversation_context=history,
            system_instruction=self.sys_instruction,
            knowledge_context=self.kb_dining,
        ))
        print(f"[Live E2E Turn 2 (Dining)]: '{resp_turn2}'")
        self.assertNotIn("swimming pool", resp_turn2.lower(), "Turn 2 repeated Turn 1 pool response!")
        self.assertTrue(
            "dining" in resp_turn2.lower() or "pluck" in resp_turn2.lower() or "venues" in resp_turn2.lower(),
            "Turn 2 did not provide dining options!"
        )

        # Turn 3: Customer asks for check-in time with Turns 1 & 2 in context
        history_turn3 = f"{history}\nCustomer: What dining options do you have?\nAssistant: {resp_turn2}"
        resp_turn3 = asyncio.run(self.rs.generate_response(
            transcript="What time is check-in?",
            conversation_context=history_turn3,
            system_instruction=self.sys_instruction,
            knowledge_context=self.kb_checkin,
        ))
        print(f"[Live E2E Turn 3 (Check-in)]: '{resp_turn3}'")
        self.assertNotIn("dining", resp_turn3.lower(), "Turn 3 repeated dining context!")
        self.assertNotIn("swimming pool", resp_turn3.lower(), "Turn 3 repeated pool context!")
        self.assertTrue(
            "3:00" in resp_turn3 or "15:00" in resp_turn3,
            "Turn 3 did not provide check-in time!"
        )
        print("[PASS] Test 6: Live multi-turn sequence (Pool -> Dining -> Check-in) verified.")


if __name__ == "__main__":
    unittest.main()
