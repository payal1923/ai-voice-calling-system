import unittest
import asyncio
from app.services.response_service import ResponseService
from app.services.knowledge_service import KnowledgeService


class TestMasterTaskIntentRoutingAndState(unittest.TestCase):
    """
    Master Regression Test Suite for:
    - Accommodation routing (Issue A)
    - Dietary routing (Issue C)
    """

    def setUp(self):
        self.response_service = ResponseService()
        self.knowledge_service = KnowledgeService()
        self.company_id = "pullman_aerocity"
        self.system_instruction = self.knowledge_service.get_system_instruction(self.company_id)

    def _get_resp(self, transcript: str, context: str = "") -> str:
        knowledge_context = self.knowledge_service.get_relevant_knowledge(
            self.company_id, transcript, conversation_context=context
        )
        return asyncio.run(self.response_service.generate_response(
            transcript=transcript,
            conversation_context=context,
            system_instruction=self.system_instruction,
            knowledge_context=knowledge_context,
        ))

    # =========================================================================
    # Part 1: Accommodation Routing Scenarios (Issue A)
    # =========================================================================
    def test_scenario_1_weekend_with_parents_three_adults(self):
        """Scenario 1: 'I am planning a weekend with my parents. What two options will you suggest for three adults?'
        Must route to accommodation guidance, NOT Pluck or any restaurant recommendation."""
        q = "I am planning a weekend with my parents. What two options will you suggest for three adults?"
        resp = self._get_resp(q)
        # Must NOT recommend Pluck or dining
        self.assertNotIn("Pluck", resp)
        self.assertNotIn("restaurant", resp.lower())
        # Must provide grounded room guidance
        self.assertIn("rooms and suites", resp.lower())
        self.assertTrue(
            "reservations" in resp.lower() or "accor" in resp.lower() or "rates" in resp.lower(),
            f"Expected reservations guidance, got: {resp}"
        )

    def test_scenario_2_travelling_with_parents_suitable_room(self):
        """Scenario 2: 'I'm travelling with my parents. Which room would be suitable for us?'"""
        q = "I'm travelling with my parents. Which room would be suitable for us?"
        resp = self._get_resp(q)
        self.assertNotIn("Pluck", resp)
        self.assertNotIn("restaurant", resp.lower())
        self.assertIn("rooms and suites", resp.lower())
        self.assertTrue("reservations" in resp.lower() or "accor" in resp.lower())

    def test_scenario_3_room_options_for_three_adults(self):
        """Scenario 3: 'What room options are available for three adults?'"""
        q = "What room options are available for three adults?"
        resp = self._get_resp(q)
        self.assertNotIn("Pluck", resp)
        self.assertIn("rooms and suites", resp.lower())
        self.assertTrue("reservations" in resp.lower() or "accor" in resp.lower())

    def test_scenario_4_family_restaurant_recommendation_preserved(self):
        """Scenario 4: A genuine family restaurant recommendation still returns an appropriate dining response."""
        q = "Which restaurant do you recommend for my family?"
        resp = self._get_resp(q)
        self.assertTrue(
            resp.startswith("For families, we recommend Pluck"),
            f"Expected Pluck family recommendation, got: {resp}"
        )

    def test_scenario_4b_family_dining_recommendation_with_parents(self):
        """Scenario 4b: Genuine family dining question mentioning parents returns Pluck."""
        q = "What restaurant would you suggest for dinner with my parents?"
        resp = self._get_resp(q)
        self.assertIn("Pluck", resp)

    def test_scenario_5_accommodation_answers_do_not_contain_restaurant_recommendations(self):
        """Scenario 5: Accommodation answers do not incorrectly contain restaurant recommendations unless dining is explicitly requested."""
        queries = [
            "What room options do you have?",
            "What rooms are available at the hotel?",
            "Can you suggest a room for 3 adults?",
            "We are planning a weekend stay, which room should we choose?",
        ]
        for q in queries:
            resp = self._get_resp(q)
            self.assertNotIn("Pluck", resp, f"Unexpected Pluck recommendation for query: '{q}' -> {resp}")
            self.assertNotIn("Honk", resp, f"Unexpected Honk recommendation for query: '{q}' -> {resp}")
            self.assertNotIn("Chidya Ghar", resp, f"Unexpected Chidya Ghar for query: '{q}' -> {resp}")

    def test_scenario_6_unsupported_room_types_and_capacities_not_invented(self):
        """Scenario 6: Unsupported room types, capacities, and availability are not invented."""
        q = "What two options will you suggest for three adults?"
        resp = self._get_resp(q)
        # Verify no fabricated room categories
        self.assertNotIn("deluxe king", resp.lower())
        self.assertNotIn("presidential suite", resp.lower())
        self.assertNotIn("executive family", resp.lower())
        # Verify no fabricated capacity guarantees
        self.assertNotIn("maximum 2 adults", resp.lower())
        self.assertNotIn("3 adults per room", resp.lower())
        # Truthfully directs to reservations / Accor live details
        self.assertTrue("reservations" in resp.lower() or "accor" in resp.lower())

    # =========================================================================
    # Part 2: Dietary Routing Scenarios (Issue C)
    # =========================================================================
    def test_dietary_scenario_1_vegetarian_mother_dining_options(self):
        """Dietary Scenario 1: 'My mother is vegetarian with dining options at your hotel might suit her.'
        Must return targeted recommendation highlighting Pluck and Honk cuisines, not generic 4-venue overview."""
        q = "My mother is vegetarian with dining options at your hotel might suit her."
        resp = self._get_resp(q)
        # Must be targeted, mentioning Pluck and Honk
        self.assertIn("Pluck", resp)
        self.assertIn("Honk", resp)
        # Must NOT just be the generic 4-venue list verbatim
        self.assertNotIn("The hotel offers a variety of dining options, including Pluck (seasonal modern European", resp)
        # Must honestly state specific vegetarian dishes are unconfirmed
        self.assertTrue(
            "not confirmed" in resp.lower() or "check" in resp.lower(),
            f"Expected notice that specific dishes are unconfirmed, got: {resp}"
        )

    def test_dietary_scenario_1b_vegetarian_clean_recommendation_question(self):
        """Dietary Scenario 1b: 'My mother is vegetarian. Which dining option might suit her?'"""
        q = "My mother is vegetarian. Which dining option might suit her?"
        resp = self._get_resp(q)
        self.assertIn("Pluck", resp)
        self.assertIn("Honk", resp)
        self.assertTrue("not confirmed" in resp.lower() or "check" in resp.lower())

    def test_dietary_scenario_2_generic_dining_options_returns_full_overview(self):
        """Dietary Scenario 2: 'What dining options are available at the hotel?' receives normal 4-venue dining overview."""
        q = "What dining options are available at the hotel?"
        resp = self._get_resp(q)
        self.assertIn("Pluck", resp)
        self.assertIn("Honk", resp)
        self.assertIn("Café Pluck", resp)
        self.assertIn("The Chidya Ghar Bar", resp)

    def test_dietary_scenario_3_confirmed_vegetarian_dishes_truthful_declination(self):
        """Dietary Scenario 3: 'Does that restaurant have confirmed vegetarian dishes?'"""
        context = (
            "Customer: Which restaurant do you recommend for a vegetarian?\n"
            "Assistant: For vegetarian dining, we recommend Pluck or Honk."
        )
        q = "Does that restaurant have confirmed vegetarian dishes?"
        resp = self._get_resp(q, context=context)
        # Must NOT fabricate dishes
        self.assertTrue(
            "not confirmed" in resp.lower() or "check" in resp.lower(),
            f"Expected truthful declination, got: {resp}"
        )
        self.assertNotIn("paneer", resp.lower())
        self.assertNotIn("dal makhani", resp.lower())

    def test_dietary_scenario_4_supported_vegan_question_reaches_dietary_handler(self):
        """Dietary Scenario 4: A supported vegan or other dietary question reaches the appropriate handler."""
        q = "Do you have any vegan options at your restaurants?"
        resp = self._get_resp(q)
        self.assertIn("vegan", resp.lower())
        self.assertTrue("not confirmed" in resp.lower() or "check" in resp.lower())

    def test_dietary_scenario_5_unsupported_dishes_and_allergy_policies_never_invented(self):
        """Dietary Scenario 5: Unsupported dishes, menu guarantees, and allergy policies are never invented."""
        q = "Can you guarantee a 100% nut-free and gluten-free kitchen with no cross-contamination?"
        resp = self._get_resp(q)
        self.assertNotIn("guarantee", resp.lower().replace("cannot guarantee", "").replace("not guaranteed", ""))
        self.assertTrue("not confirmed" in resp.lower() or "front desk" in resp.lower() or "check" in resp.lower())

    def test_dietary_scenario_6_existing_dining_overview_and_recommendation_tests_pass(self):
        """Dietary Scenario 6: Existing dining overview and restaurant recommendation tests continue to pass."""
        q_overview = "What restaurants do you have?"
        resp_overview = self._get_resp(q_overview)
        self.assertIn("Pluck", resp_overview)
        self.assertIn("Honk", resp_overview)

        q_honk = "Tell me about Honk."
        resp_honk = self._get_resp(q_honk)
        self.assertIn("pan-Asian", resp_honk)


if __name__ == "__main__":
    unittest.main()
