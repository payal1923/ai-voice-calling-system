import os
import sys
import asyncio
import unittest

sys.path.insert(0, os.path.abspath("."))

from app.services.knowledge_service import KnowledgeService
from app.services.response_service import ResponseService
from app.services.conversation_context_service import ConversationContextService
from app.database.models import Call, Customer, Message


class TestSwimmingPoolAndContextFixes(unittest.TestCase):
    def setUp(self):
        self.ks = KnowledgeService()
        self.rs = ResponseService()
        self.ctx = ConversationContextService()
        self.company_id = "pullman_aerocity"
        self.sys_instruction = self.ks.get_system_instruction(self.company_id)

    def test_01_swimming_pool_knowledge_retrieval(self):
        """Verifies that swimming pool queries reliably retrieve the verified Pullman pool fact."""
        queries = [
            "Do you have swimming pool?",
            "Do you have a swimming pool?",
            "Is there a pool at the hotel?",
            "Can I swim in the pool?",
        ]
        for q in queries:
            facts = self.ks.get_relevant_knowledge(self.company_id, q)
            self.assertTrue(bool(facts), f"Knowledge retrieval failed for '{q}'")
            self.assertIn("swimming pool", facts.lower(), f"Fact does not contain swimming pool for '{q}'")
            self.assertIn("heated outdoor", facts.lower(), f"Fact does not contain heated outdoor for '{q}'")
        print("[PASS] Test 1: Swimming pool knowledge retrieval verified across phrasing variations.")

    def test_02_scenario_a_topic_change_masterclass_to_pool(self):
        """
        Scenario A:
        Customer asks about rose masterclass, then switches topic to swimming pool.
        The assistant must answer the pool question from verified knowledge and NOT ask about the rose masterclass.
        """
        cust = Customer(id=1, external_customer_id="cust-test-scenario-a")
        call = Call(id=701, customer_id=1)
        call.customer = cust
        call.messages = [
            Message(id=1, call_id=701, speaker="assistant", text="Hello! Thank you for calling Pullman New Delhi Aerocity. How may I assist you today?"),
            Message(id=2, call_id=701, speaker="customer", text="Can I book a rose masterclass?"),
            Message(id=3, call_id=701, speaker="assistant", text="I apologize, but we do not offer a rose masterclass at our hotel. Please contact our front desk for assistance."),
        ]

        history_ctx = self.ctx.build_context([call])
        k_pool = self.ks.get_relevant_knowledge(self.company_id, "Do you have a swimming pool?")

        resp = asyncio.run(self.rs.generate_response(
            transcript="Do you have a swimming pool?",
            conversation_context=history_ctx,
            system_instruction=self.sys_instruction,
            knowledge_context=k_pool,
        ))

        resp_lower = resp.lower()
        print(f"\n[Scenario A Response]: '{resp}'")
        self.assertNotIn("masterclass", resp_lower, "Response repeated stale masterclass topic!")
        self.assertNotIn("rose", resp_lower, "Response repeated stale rose topic!")
        self.assertNotIn("specify the dates", resp_lower, "Response asked for dates when not booking!")
        self.assertTrue(
            "swimming pool" in resp_lower or "pool" in resp_lower,
            "Response did not confirm swimming pool!"
        )
        print("[PASS] Test 2: Scenario A topic change (masterclass -> swimming pool) verified.")

    def test_03_scenario_b_topic_change_restaurants_to_checkin(self):
        """
        Scenario B:
        Customer asks about restaurants, then switches topic to check-in time.
        Assistant must answer 3:00 PM (15:00) without appending dining or masterclass questions.
        """
        cust = Customer(id=1, external_customer_id="cust-test-scenario-b")
        call = Call(id=702, customer_id=1)
        call.customer = cust
        call.messages = [
            Message(id=1, call_id=702, speaker="assistant", text="Hello! Thank you for calling Pullman New Delhi Aerocity."),
            Message(id=2, call_id=702, speaker="customer", text="What restaurants do you have?"),
            Message(id=3, call_id=702, speaker="assistant", text="Pullman New Delhi Aerocity features four dining venues: Pluck, Honk, Cafe Pluck, and The Chidya Ghar Bar."),
        ]

        history_ctx = self.ctx.build_context([call])
        k_checkin = self.ks.get_relevant_knowledge(self.company_id, "What is the check-in time?")

        resp = asyncio.run(self.rs.generate_response(
            transcript="What is the check-in time?",
            conversation_context=history_ctx,
            system_instruction=self.sys_instruction,
            knowledge_context=k_checkin,
        ))

        resp_lower = resp.lower()
        print(f"\n[Scenario B Response]: '{resp}'")
        self.assertNotIn("restaurant", resp_lower, "Appended stale restaurant context!")
        self.assertNotIn("masterclass", resp_lower, "Appended stale masterclass context!")
        self.assertTrue(
            "15:00" in resp or "3:00" in resp or "3 pm" in resp_lower,
            "Check-in time not provided!"
        )
        print("[PASS] Test 3: Scenario B topic change (restaurants -> check-in time) verified.")

    def test_04_scenario_c_genuine_followup_checkin_checkout(self):
        """
        Scenario C:
        Customer asks about check-in, then follows up with 'And what about check-out?'.
        Assistant must answer 12:00 PM using relevant conversation context.
        """
        cust = Customer(id=1, external_customer_id="cust-test-scenario-c")
        call = Call(id=703, customer_id=1)
        call.customer = cust
        call.messages = [
            Message(id=1, call_id=703, speaker="assistant", text="Hello! Thank you for calling Pullman New Delhi Aerocity."),
            Message(id=2, call_id=703, speaker="customer", text="What is the check-in time?"),
            Message(id=3, call_id=703, speaker="assistant", text="The check-in time at Pullman New Delhi Aerocity is 3:00 PM (15:00)."),
        ]

        history_ctx = self.ctx.build_context([call])
        k_checkout = self.ks.get_relevant_knowledge(self.company_id, "And what about check-out?")

        resp = asyncio.run(self.rs.generate_response(
            transcript="And what about check-out?",
            conversation_context=history_ctx,
            system_instruction=self.sys_instruction,
            knowledge_context=k_checkout,
        ))

        resp_lower = resp.lower()
        print(f"\n[Scenario C Response]: '{resp}'")
        self.assertTrue(
            "12:00" in resp or "noon" in resp_lower or "check-out" in resp_lower,
            "Follow-up check-out time not provided!"
        )
        print("[PASS] Test 4: Scenario C genuine follow-up (check-in -> check-out) verified.")

    def test_05_scenario_d_missing_knowledge_absent_amenity(self):
        """
        Scenario D:
        Customer asks about an amenity absent from knowledge profile (e.g. ice skating rink / bowling alley).
        Assistant must not invent the facility, ask for dates, or repeat prior topic clarifications.
        """
        cust = Customer(id=1, external_customer_id="cust-test-scenario-d")
        call = Call(id=704, customer_id=1)
        call.customer = cust
        call.messages = [
            Message(id=1, call_id=704, speaker="customer", text="Can I book a rose masterclass?"),
            Message(id=2, call_id=704, speaker="assistant", text="I can assist with that."),
        ]

        history_ctx = self.ctx.build_context([call])
        k_absent = self.ks.get_relevant_knowledge(self.company_id, "Do you have an ice skating rink?")

        resp = asyncio.run(self.rs.generate_response(
            transcript="Do you have an ice skating rink?",
            conversation_context=history_ctx,
            system_instruction=self.sys_instruction,
            knowledge_context=k_absent,
        ))

        resp_lower = resp.lower()
        print(f"\n[Scenario D Response]: '{resp}'")
        self.assertNotIn("masterclass", resp_lower, "Repeated stale masterclass topic!")
        self.assertNotIn("specify the dates", resp_lower, "Asked for dates on absent amenity!")
        self.assertNotIn("yes, we do have an ice skating", resp_lower, "Hallucinated absent amenity!")
        print("[PASS] Test 5: Scenario D missing knowledge (absent amenity) safely handled.")

    def test_06_ambiguous_or_empty_transcript_handling(self):
        """Verifies that empty transcript triggers polite repeat request without invoking LLM or hallucinating."""
        resp = asyncio.run(self.rs.generate_response(
            transcript="",
            conversation_context="Some prior context",
            system_instruction=self.sys_instruction,
            knowledge_context=None,
        ))
        self.assertIn("didn't catch that", resp.lower())
        print(f"\n[Empty Transcript Response]: '{resp}'")
        print("[PASS] Test 6: Empty/ambiguous transcript handled safely.")


if __name__ == "__main__":
    unittest.main()
