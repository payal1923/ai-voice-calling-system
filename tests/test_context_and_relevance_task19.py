import unittest
import asyncio
from app.services.knowledge_service import KnowledgeService
from app.services.response_service import ResponseService


class TestContextAndRelevanceTask19(unittest.IsolatedAsyncioTestCase):
    """
    Task 19 Regression Suite: Context retention, prevention of unrelated answers,
    and speech recovery safeguards in Pullman Aerocity voice agent.
    """

    def setUp(self):
        self.ks = KnowledgeService()
        self.rs = ResponseService()
        self.company_id = "pullman_aerocity"

    async def test_01_garbled_restaurant_query_never_triggers_checkin_id_policy(self):
        """1. A garbled restaurant query must never trigger an unrelated check-in policy."""
        # Simulated context after Pluck recommendation
        ctx = (
            "Customer: Which restaurant would you recommend for my family?\n"
            "Assistant: For families, we recommend Pluck, the hotel's signature restaurant offering seasonal modern European and Indian cuisine in a welcoming setting."
        )

        garbled_queries = [
            "Can you take my prizes and go with the hidden options?",
            "Can you take my prizes.",
            "take me private and go with hidden options",
        ]

        for q in garbled_queries:
            kb = self.ks.get_relevant_knowledge(self.company_id, q, conversation_context=ctx)
            # Knowledge retrieval must not retrieve check_in_id_policy
            self.assertNotIn("photo id", kb.lower(), f"Check-in ID policy retrieved for garbled query: {q}")
            self.assertNotIn("passport", kb.lower(), f"Check-in ID policy retrieved for garbled query: {q}")

            resp = await self.rs.generate_response(q, conversation_context=ctx, knowledge_context=kb)
            resp_lower = resp.lower()

            # Must NEVER return check-in ID requirements
            self.assertNotIn("government-issued", resp_lower, f"Returned ID policy for garbled query: {q}")
            self.assertNotIn("photo id", resp_lower, f"Returned ID policy for garbled query: {q}")
            self.assertNotIn("passport", resp_lower, f"Returned ID policy for garbled query: {q}")
            self.assertNotIn("aadhaar", resp_lower, f"Returned ID policy for garbled query: {q}")
            self.assertNotIn("pan card", resp_lower, f"Returned ID policy for garbled query: {q}")
            self.assertNotIn("check-in is at", resp_lower, f"Returned check-in time for garbled query: {q}")

            # Must ask a focused clarification narrowing plausible meanings or address prices/options
            has_focused_clarification = any(w in resp_lower for w in ["inquiring", "clarify", "catch", "prices", "options", "reservations", "pluck"])
            self.assertTrue(has_focused_clarification, f"Response did not offer focused clarification: {resp}")

    async def test_02_food_served_after_pluck_recommendation_refers_to_pluck(self):
        """2. 'What kind of food do they serve?' after a Pluck recommendation refers specifically to Pluck."""
        ctx = (
            "Customer: Which restaurant would you recommend for my family?\n"
            "Assistant: For families, we recommend Pluck, the hotel's signature restaurant offering seasonal modern European and Indian cuisine in a welcoming setting."
        )

        queries = [
            "What kind of food they serve.",
            "What kind of food do they serve?",
            "What do they serve?",
            "Tell me about their food.",
        ]

        for q in queries:
            kb = self.ks.get_relevant_knowledge(self.company_id, q, conversation_context=ctx)
            resp = await self.rs.generate_response(q, conversation_context=ctx, knowledge_context=kb)
            resp_lower = resp.lower()

            # Must refer to Pluck
            self.assertIn("pluck", resp_lower, f"Response did not mention Pluck for query: {q}")
            # Must mention European or Indian cuisine
            self.assertTrue(
                "european" in resp_lower or "indian" in resp_lower,
                f"Response did not describe Pluck cuisine: {resp}"
            )
            # Must NOT dump all 4 venues
            self.assertNotIn("honk", resp_lower, f"Leaked Honk into Pluck follow-up: {resp}")
            self.assertNotIn("chidya ghar", resp_lower, f"Leaked Chidya Ghar into Pluck follow-up: {resp}")

    async def test_03_vegetarian_option_questions_answered_only_when_supported(self):
        """3. Vegetarian-option questions are answered only if supported by KB; otherwise states unconfirmed."""
        ctx = (
            "Customer: Which restaurant would you recommend for my family?\n"
            "Assistant: For families, we recommend Pluck, the hotel's signature restaurant offering seasonal modern European and Indian cuisine in a welcoming setting."
        )

        queries = [
            "Do they have vegetarian options?",
            "Can you take me private and go vegetarian?",
            "Is vegetarian food available at Pluck?",
            "Do you offer vegetarian options?",
        ]

        for q in queries:
            kb = self.ks.get_relevant_knowledge(self.company_id, q, conversation_context=ctx)
            resp = await self.rs.generate_response(q, conversation_context=ctx, knowledge_context=kb)
            resp_lower = resp.lower()

            # Must address vegetarian / dietary inquiry
            self.assertTrue(
                "vegetarian" in resp_lower or "dietary" in resp_lower,
                f"Response did not address vegetarian topic: {resp}"
            )
            # Must state that specific options are not confirmed and suggest checking with restaurant or front desk
            self.assertTrue(
                any(p in resp_lower for p in ["not confirmed", "not listed", "not available", "check directly", "speak directly"]),
                f"Did not truthfully qualify missing vegetarian details: {resp}"
            )
            # Must not invent dishes or make up policies
            self.assertNotIn("government-issued", resp_lower)
            self.assertNotIn("check-in at pullman", resp_lower)

    async def test_04_price_and_reservation_questions_do_not_receive_unrelated_answers(self):
        """4. Price and reservation questions do not receive unrelated check-in or room tariff answers."""
        ctx = (
            "Customer: Tell me about Pluck restaurant.\n"
            "Assistant: Pluck is the hotel's signature restaurant offering modern European and contemporary Indian cuisine with a farm-to-table philosophy and an in-house organic herb garden."
        )

        price_queries = [
            "What are the prices of dishes at Pluck?",
            "How much does dinner cost at Pluck?",
            "What are the restaurant prices?",
        ]

        for q in price_queries:
            kb = self.ks.get_relevant_knowledge(self.company_id, q, conversation_context=ctx)
            self.assertNotIn("vary in real time based on stay dates", kb, f"Room rates fact leaked into restaurant price query: {q}")
            resp = await self.rs.generate_response(q, conversation_context=ctx, knowledge_context=kb)
            resp_lower = resp.lower()

            self.assertNotIn("government-issued", resp_lower)
            self.assertNotIn("vary in real time based on stay dates", resp_lower)
            self.assertIn("prices", resp_lower)

        booking_queries = [
            "Can I book a table at Pluck for dinner?",
            "Can I reserve a table for lunch?",
            "I want to make a dining reservation.",
        ]

        for q in booking_queries:
            kb = self.ks.get_relevant_knowledge(self.company_id, q, conversation_context=ctx)
            self.assertNotIn("vary in real time based on stay dates", kb, f"Room rates fact leaked into table booking query: {q}")
            resp = await self.rs.generate_response(q, conversation_context=ctx, knowledge_context=kb)
            resp_lower = resp.lower()

            self.assertNotIn("government-issued", resp_lower)
            self.assertNotIn("check-in at pullman", resp_lower)
            self.assertIn("table reservation", resp_lower)

    async def test_05_clear_topic_change_overrides_prior_context(self):
        """5. Clear topic changes cleanly override prior dining context."""
        ctx = (
            "Customer: Which restaurant would you recommend for my family?\n"
            "Assistant: For families, we recommend Pluck, the hotel's signature restaurant offering seasonal modern European and Indian cuisine in a welcoming setting.\n"
            "Customer: What kind of food do they serve?\n"
            "Assistant: Pluck is the hotel's signature restaurant offering modern European and contemporary Indian cuisine with a farm-to-table philosophy and an in-house organic herb garden."
        )

        # Topic change 1: Check-in
        q_checkin = "What time is check-in?"
        kb_checkin = self.ks.get_relevant_knowledge(self.company_id, q_checkin, conversation_context=ctx)
        resp_checkin = await self.rs.generate_response(q_checkin, conversation_context=ctx, knowledge_context=kb_checkin)
        self.assertIn("3:00", resp_checkin)
        self.assertNotIn("pluck", resp_checkin.lower())

        # Topic change 2: Swimming pool
        q_pool = "What are the swimming pool timings?"
        kb_pool = self.ks.get_relevant_knowledge(self.company_id, q_pool, conversation_context=ctx)
        resp_pool = await self.rs.generate_response(q_pool, conversation_context=ctx, knowledge_context=kb_pool)
        self.assertIn("pool", resp_pool.lower())
        self.assertNotIn("pluck", resp_pool.lower())
        self.assertNotIn("3:00", resp_pool)

        # Topic change 3: Check-out
        q_checkout = "What time is check-out?"
        kb_checkout = self.ks.get_relevant_knowledge(self.company_id, q_checkout, conversation_context=ctx)
        resp_checkout = await self.rs.generate_response(q_checkout, conversation_context=ctx, knowledge_context=kb_checkout)
        self.assertIn("12:00", resp_checkout)
        self.assertNotIn("pluck", resp_checkout.lower())

    async def test_06_unclear_speech_focused_clarification_and_variation(self):
        """6. Unclear speech triggers focused clarification in active context and avoids looping."""
        ctx_dining = (
            "Customer: Tell me about Pluck restaurant.\n"
            "Assistant: Pluck is the hotel's signature restaurant offering modern European and contemporary Indian cuisine with a farm-to-table philosophy and an in-house organic herb garden."
        )

        # In dining context, unclear speech clarifies dining/price/reservation
        resp = await self.rs.generate_response("asdf ghjkl", conversation_context=ctx_dining, knowledge_context="")
        self.assertTrue(
            "pluck" in resp.lower() or "dining" in resp.lower() or "prices" in resp.lower() or "reservations" in resp.lower(),
            f"Expected focused clarification in dining context: {resp}"
        )

        # Repeated unclear queries vary response to avoid robotic repetition
        ctx_repeated = (
            ctx_dining + "\n"
            "Customer: asdf ghjkl\n"
            "Assistant: I'm sorry, I didn't catch that clearly. Were you inquiring about dining options, restaurant prices, or reservations at Pluck?\n"
            "Customer: mumble mumble"
        )
        resp_repeated = await self.rs.generate_response("mumble mumble", conversation_context=ctx_repeated, knowledge_context="")
        self.assertIn("trouble hearing", resp_repeated.lower())

    async def test_07_genuine_checkin_id_policy_query_answered_correctly(self):
        """7. A genuine inquiry for check-in documents receives the official ID policy."""
        id_queries = [
            "What ID documents are required at check-in?",
            "What identification do I need to present at check-in?",
            "Which documents are mandatory at check-in?",
        ]

        for q in id_queries:
            kb = self.ks.get_relevant_knowledge(self.company_id, q)
            self.assertIn("government-issued photo id", kb.lower())
            resp = await self.rs.generate_response(q, knowledge_context=kb)
            resp_lower = resp.lower()
            self.assertTrue(
                "government" in resp_lower or "photo id" in resp_lower or "passport" in resp_lower,
                f"Failed to return ID policy for genuine query '{q}': {resp}"
            )


if __name__ == "__main__":
    unittest.main()
