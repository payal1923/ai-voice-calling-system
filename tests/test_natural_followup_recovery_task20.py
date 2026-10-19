import unittest
import asyncio
from app.services.knowledge_service import KnowledgeService
from app.services.response_service import ResponseService


class TestNaturalFollowupRecoveryTask20(unittest.IsolatedAsyncioTestCase):
    """
    Task 20 Regression Suite:
    Natural follow-up understanding, clarification recovery, conversational statements,
    honest dietary grounding, and loop prevention.
    """

    def setUp(self):
        self.ks = KnowledgeService()
        self.rs = ResponseService()
        self.company_id = "pullman_aerocity"

    async def test_01_pluck_recommendation_followed_by_food_served(self):
        """1. Pluck recommendation followed by 'What kind of food do they serve?'"""
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

            # Must specifically refer to Pluck
            self.assertIn("pluck", resp_lower, f"Failed to focus on Pluck for query '{q}': {resp}")
            # Must explain European and Indian cuisine
            self.assertTrue(
                "european" in resp_lower or "indian" in resp_lower,
                f"Did not describe Pluck cuisine for query '{q}': {resp}"
            )
            # Must not dump all 4 venues
            self.assertNotIn("honk", resp_lower, f"Leaked other venues into Pluck response: {resp}")
            self.assertNotIn("chidya ghar", resp_lower, f"Leaked other venues into Pluck response: {resp}")

    async def test_02_pluck_followed_by_direct_vegetarian_options(self):
        """2. Pluck followed by 'Do they have vegetarian options?' handled honestly."""
        ctx = (
            "Customer: Which restaurant would you recommend for my family?\n"
            "Assistant: For families, we recommend Pluck, the hotel's signature restaurant offering seasonal modern European and Indian cuisine in a welcoming setting.\n"
            "Customer: What kind of food they serve.\n"
            "Assistant: Pluck is the hotel's signature restaurant offering modern European and contemporary Indian cuisine with a farm-to-table philosophy and an in-house organic herb garden."
        )

        queries = [
            "Do they have vegetarian options?",
            "Do they offer vegetarian food?",
            "Is there vegetarian food at Pluck?",
        ]

        for q in queries:
            kb = self.ks.get_relevant_knowledge(self.company_id, q, conversation_context=ctx)
            resp = await self.rs.generate_response(q, conversation_context=ctx, knowledge_context=kb)
            resp_lower = resp.lower()

            # Must address vegetarian topic
            self.assertTrue("vegetarian" in resp_lower or "dietary" in resp_lower, f"Did not address vegetarian: {resp}")
            # Must honestly state specific options are not confirmed
            self.assertTrue(
                any(p in resp_lower for p in ["not confirmed", "not listed", "not available", "check directly", "speak directly"]),
                f"Did not truthfully qualify missing vegetarian details: {resp}"
            )
            # Must offer to check with restaurant or front desk
            self.assertTrue(
                "restaurant" in resp_lower or "front desk" in resp_lower,
                f"Did not offer front desk or restaurant contact: {resp}"
            )
            # Must never invent menu details
            self.assertNotIn("paneer", resp_lower)
            self.assertNotIn("salad bar", resp_lower)

    async def test_03_ambiguous_visible_options_in_dining_context(self):
        """3. An ambiguous transcript like 'Do they have visible options?' triggers focused clarification."""
        ctx = (
            "Customer: Which restaurant would you recommend for my family?\n"
            "Assistant: For families, we recommend Pluck, the hotel's signature restaurant offering seasonal modern European and Indian cuisine in a welcoming setting.\n"
            "Customer: What kind of food they serve.\n"
            "Assistant: Pluck is the hotel's signature restaurant offering modern European and contemporary Indian cuisine with a farm-to-table philosophy and an in-house organic herb garden."
        )

        ambiguous_queries = [
            "Do they have visible options?",
            "Do they have visitor options?",
            "What about visible options?",
            "Do they have vegetable options?",
        ]

        for q in ambiguous_queries:
            kb = self.ks.get_relevant_knowledge(self.company_id, q, conversation_context=ctx)
            resp = await self.rs.generate_response(q, conversation_context=ctx, knowledge_context=kb)
            resp_lower = resp.lower()

            # Must ask focused clarification about vegetarian options
            self.assertIn("are you asking", resp_lower, f"Did not ask focused clarification for '{q}': {resp}")
            self.assertIn("vegetarian options", resp_lower, f"Did not identify vegetarian intent for '{q}': {resp}")
            self.assertIn("pluck", resp_lower, f"Did not maintain Pluck venue in clarification: {resp}")

    async def test_04_conversational_statement_they_have_great_options(self):
        """4. Customer responds 'They have great options' after clarification without identical loop."""
        ctx = (
            "Customer: Which restaurant would you recommend for my family?\n"
            "Assistant: For families, we recommend Pluck, the hotel's signature restaurant offering seasonal modern European and Indian cuisine in a welcoming setting.\n"
            "Customer: What kind of food they serve.\n"
            "Assistant: Pluck is the hotel's signature restaurant offering modern European and contemporary Indian cuisine with a farm-to-table philosophy and an in-house organic herb garden.\n"
            "Customer: Do they have visible options?\n"
            "Assistant: Were you inquiring about dining options, restaurant prices, or reservations at Pluck?"
        )

        statement_queries = [
            "They have great options.",
            "They have good options.",
            "That sounds great.",
            "Sounds good.",
            "Pluck sounds great.",
        ]

        for q in statement_queries:
            kb = self.ks.get_relevant_knowledge(self.company_id, q, conversation_context=ctx)
            resp = await self.rs.generate_response(q, conversation_context=ctx, knowledge_context=kb)
            resp_lower = resp.lower()

            # Must NOT repeat the identical clarification
            self.assertNotIn("were you inquiring about", resp_lower, f"Repeated clarification on statement '{q}': {resp}")
            self.assertNotIn("i didn't catch that clearly", resp_lower, f"Treated conversational statement as garbled: {resp}")
            # Must respond naturally with Pluck offerings & helpful next steps
            self.assertTrue(
                "pluck" in resp_lower or "european" in resp_lower or "indian" in resp_lower or "dishes" in resp_lower or "options" in resp_lower,
                f"Did not acknowledge dining context naturally for '{q}': {resp}"
            )
            self.assertTrue(
                "hours" in resp_lower or "reservations" in resp_lower or "assistance" in resp_lower or "table" in resp_lower,
                f"Did not offer natural next steps for '{q}': {resp}"
            )

    async def test_05_clarification_confirmation_answers_honestly(self):
        """5. Customer confirms vegetarian clarification affirmatively."""
        ctx = (
            "Customer: Which restaurant would you recommend for my family?\n"
            "Assistant: For families, we recommend Pluck, the hotel's signature restaurant offering seasonal modern European and Indian cuisine in a welcoming setting.\n"
            "Customer: What kind of food they serve.\n"
            "Assistant: Pluck is the hotel's signature restaurant offering modern European and contemporary Indian cuisine with a farm-to-table philosophy and an in-house organic herb garden.\n"
            "Customer: Do they have visible options?\n"
            "Assistant: Are you asking whether Pluck has vegetarian options?"
        )

        confirmations = [
            "Yes",
            "Yeah",
            "Yes, vegetarian",
            "That's right",
            "Correct",
            "Yes please",
        ]

        for q in confirmations:
            resp = await self.rs.generate_response(q, conversation_context=ctx)
            resp_lower = resp.lower()

            # Truthfully states options are not confirmed
            self.assertTrue(
                any(p in resp_lower for p in ["not confirmed", "not listed", "not available"]),
                f"Did not state vegetarian unconfirmed for confirmation '{q}': {resp}"
            )
            # Offers front desk or restaurant contact
            self.assertTrue(
                "restaurant" in resp_lower or "front desk" in resp_lower,
                f"Did not offer contact for confirmation '{q}': {resp}"
            )
            # Never invents menu details
            self.assertNotIn("paneer", resp_lower)
            self.assertNotIn("dal makhani", resp_lower)

    async def test_06_repeated_unclear_turns_prevent_identical_loop(self):
        """6. Repeated unclear turns simplify to two choices without an identical clarification loop."""
        ctx_base = (
            "Customer: Which restaurant would you recommend for my family?\n"
            "Assistant: For families, we recommend Pluck, the hotel's signature restaurant offering seasonal modern European and Indian cuisine in a welcoming setting."
        )

        # Turn 1: First unclear turn -> focused 3-option clarification
        resp_turn1 = await self.rs.generate_response("asdf ghjkl", conversation_context=ctx_base)
        self.assertIn("were you inquiring about", resp_turn1.lower())
        self.assertIn("pluck", resp_turn1.lower())

        # Turn 2: Second unclear turn -> simplifies to TWO relevant choices
        ctx_turn2 = f"{ctx_base}\nCustomer: asdf ghjkl\nAssistant: {resp_turn1}"
        resp_turn2 = await self.rs.generate_response("mumble mumble", conversation_context=ctx_turn2)
        resp_turn2_lower = resp_turn2.lower()

        # Must NOT be identical to turn 1 clarification
        self.assertNotEqual(resp_turn1, resp_turn2, "Repeated identical clarification on second unclear turn!")
        self.assertNotIn("were you inquiring about", resp_turn2_lower, "Repeated the 3-part clarification prompt!")
        # Must offer two simplified relevant choices
        self.assertTrue(
            ("menu at pluck" in resp_turn2_lower or "menu" in resp_turn2_lower) and
            ("connect you with the restaurant" in resp_turn2_lower or "front desk" in resp_turn2_lower),
            f"Turn 2 did not simplify to two relevant choices: {resp_turn2}"
        )

        # Turn 3: Third unclear turn -> offers front desk escalation
        ctx_turn3 = f"{ctx_turn2}\nCustomer: mumble mumble\nAssistant: {resp_turn2}"
        resp_turn3 = await self.rs.generate_response("blah blah blah", conversation_context=ctx_turn3)
        resp_turn3_lower = resp_turn3.lower()

        self.assertNotEqual(resp_turn2, resp_turn3, "Repeated identical clarification on third unclear turn!")
        self.assertIn("front desk", resp_turn3_lower)

    async def test_07_clear_topic_change_to_checkin(self):
        """7. A clear topic change to check-in cleanly overrides prior dining context."""
        ctx = (
            "Customer: Which restaurant would you recommend for my family?\n"
            "Assistant: For families, we recommend Pluck, the hotel's signature restaurant offering seasonal modern European and Indian cuisine in a welcoming setting.\n"
            "Customer: What kind of food they serve.\n"
            "Assistant: Pluck is the hotel's signature restaurant offering modern European and contemporary Indian cuisine with a farm-to-table philosophy and an in-house organic herb garden.\n"
            "Customer: Do they have visible options?\n"
            "Assistant: Are you asking whether Pluck has vegetarian options?"
        )

        checkin_queries = [
            "What time is check-in?",
            "What time is check in?",
            "When is check in?",
            "Check-in time",
        ]

        for q in checkin_queries:
            kb = self.ks.get_relevant_knowledge(self.company_id, q, conversation_context=ctx)
            resp = await self.rs.generate_response(q, conversation_context=ctx, knowledge_context=kb)
            resp_lower = resp.lower()

            # Must return check-in timing (3:00 PM / 15:00)
            self.assertIn("3:00", resp, f"Failed to return 3:00 PM check-in time for '{q}': {resp}")
            # Must not mention Pluck or dining
            self.assertNotIn("pluck", resp_lower, f"Contaminated check-in with Pluck context for '{q}': {resp}")
            self.assertNotIn("vegetarian", resp_lower, f"Contaminated check-in with vegetarian context for '{q}': {resp}")

    async def test_08_unsupported_dietary_information_never_invented(self):
        """8. Unsupported dietary information is never invented across any venues."""
        dietary_queries = [
            ("What vegan dishes does Pluck serve?", "pluck"),
            ("Do they have a gluten-free menu at Honk?", "honk"),
            ("Is the food halal or kosher at your restaurants?", None),
            ("Do you have Jain food available?", None),
        ]

        for q, expected_venue in dietary_queries:
            kb = self.ks.get_relevant_knowledge(self.company_id, q)
            resp = await self.rs.generate_response(q, knowledge_context=kb)
            resp_lower = resp.lower()

            # Truthfully qualifies that dietary options are not confirmed or listed
            self.assertTrue(
                any(p in resp_lower for p in ["not confirmed", "not listed", "not available"]),
                f"Did not truthfully qualify missing dietary information for '{q}': {resp}"
            )
            # Recommends checking directly with restaurant or front desk
            self.assertTrue(
                "restaurant" in resp_lower or "front desk" in resp_lower,
                f"Did not suggest checking with restaurant or front desk for '{q}': {resp}"
            )
            # Never invents dishes
            self.assertNotIn("tofu", resp_lower)
            self.assertNotIn("steamed broccoli", resp_lower)


if __name__ == "__main__":
    unittest.main()
