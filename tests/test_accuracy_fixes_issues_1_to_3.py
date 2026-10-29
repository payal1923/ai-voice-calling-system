import unittest
import asyncio
from app.services.response_service import ResponseService


class TestAccuracyFixesIssues1To3(unittest.TestCase):
    """
    Focused unit and regression tests for Issues 1-3:
    1. Follow-up questions about children's food suitability trigger the generic family restaurant recommendation.
    2. Restaurant-list questions can produce invented venues such as 'The Table'.
    3. _clean_text() corrupts 'High-speed' into 'Gh-speed'.
    """

    def setUp(self):
        self.response_service = ResponseService()

    # =========================================================================
    # Fix C: Text Cleanup Word Boundary
    # =========================================================================
    def test_fix_c_clean_text_preserves_high_speed(self):
        """_clean_text() must preserve 'High-speed' and not mangle it into 'Gh-speed'."""
        text = "High-speed Wi-Fi internet access is complimentary for all registered in-house guests throughout the guest rooms and public areas."
        cleaned = ResponseService._clean_text(text)
        self.assertTrue(
            cleaned.startswith("High-speed"),
            f"Expected text to start with 'High-speed', got: '{cleaned}'"
        )
        self.assertNotIn("Gh-speed", cleaned)

    def test_fix_c_clean_text_strips_legitimate_fillers(self):
        """_clean_text() must still strip legitimate standalone conversational fillers."""
        self.assertEqual(ResponseService._clean_text("Hi! Check-in is at 3:00 PM."), "Check-in is at 3:00 PM.")
        self.assertEqual(ResponseService._clean_text("Hi, our check-in time is 3:00 PM."), "Our check-in time is 3:00 PM.")
        self.assertEqual(ResponseService._clean_text("Hello, how may I help you?"), "How may I help you?")
        self.assertEqual(ResponseService._clean_text("Sure, our pool is open daily."), "Our pool is open daily.")

    # =========================================================================
    # Fix B: Restaurant Name Fidelity
    # =========================================================================
    def test_fix_b_dining_overview_singular_and_plural_variations(self):
        """Singular and plural variations return only the four approved dining venues."""
        approved_venues = ["Pluck", "Honk", "Café Pluck", "The Chidya Ghar Bar"]
        queries = [
            "What restaurant are available at the hotel and what type of food does each restaurant serve?",
            "What restaurants are available at the hotel and what type of food does each restaurant serve?",
            "What restaurant is available at the hotel?",
            "Which restaurant are available at the hotel?",
            "Which restaurants are available at the hotel?",
            "What dining options are available?",
        ]
        for q in queries:
            resp = asyncio.run(self.response_service.generate_response(q))
            for venue in approved_venues:
                self.assertIn(
                    venue.replace("Café", "Caf"),
                    resp.replace("Café", "Caf"),
                    f"Approved venue '{venue}' missing in response to '{q}': {resp}"
                )
            self.assertNotIn("The Table", resp, f"Hallucinated venue 'The Table' found in response to '{q}': {resp}")

    def test_fix_b_dining_overview_unaffected_by_active_venue_history(self):
        """A general restaurant overview request must not be shadowed by a prior active venue."""
        prior_context = (
            "Customer: Tell me about your bar.\n"
            "Assistant: The Chidya Ghar Bar is a stylish contemporary bar serving handcrafted artisanal cocktails."
        )
        q = "What restaurant are available at the hotel and what type of food does each restaurant serve?"
        resp = asyncio.run(self.response_service.generate_response(q, conversation_context=prior_context))
        self.assertIn("Pluck", resp)
        self.assertIn("Honk", resp)
        self.assertNotIn("The Table", resp)

    def test_fix_b_sanitize_response_suppresses_the_table_hallucination(self):
        """_sanitize_response() replaces hallucinated 'The Table' venue names with approved overview."""
        hallucinated_text = (
            "The hotel offers dining options including Pluck, Honk, and The Table (modern fusion cuisine)."
        )
        sanitized = ResponseService._sanitize_response(
            hallucinated_text,
            transcript="What restaurant are available at the hotel?",
            conversation_context=""
        )
        self.assertNotIn("The Table", sanitized)
        self.assertIn("Pluck", sanitized)
        self.assertIn("Honk", sanitized)

    # =========================================================================
    # Fix A: Follow-up Intent Precedence
    # =========================================================================
    def test_fix_a_children_food_suitability_not_intercepted_by_family_rec(self):
        """Follow-up questions about children's food suitability must not return the generic family recommendation."""
        prior_context = (
            "Customer: Which restaurant do you recommend for my family?\n"
            "Assistant: For families, we recommend Pluck, the hotel's signature restaurant offering seasonal modern European and Indian cuisine in a welcoming setting."
        )
        q = "You recommended Pluck for my family. Does it offer food suitable for children?"
        resp = asyncio.run(self.response_service.generate_response(q, conversation_context=prior_context))

        # Must NOT be intercepted by generic family recommendation
        self.assertFalse(
            resp.startswith("For families, we recommend Pluck"),
            f"Response incorrectly repeated generic family recommendation: {resp}"
        )
        # Must answer the actual suitability question truthfully using verified facts
        self.assertIn("Pluck", resp)
        self.assertTrue(
            "not confirmed" in resp.lower() or "check" in resp.lower(),
            f"Expected truthful response indicating unconfirmed details, got: {resp}"
        )

    def test_fix_a_children_food_suitability_pronoun_followup(self):
        """Pronoun follow-up ('Does it offer food suitable for children?') resolves active venue and answers truthfully."""
        prior_context = (
            "Customer: Which restaurant do you recommend for my family?\n"
            "Assistant: For families, we recommend Pluck, the hotel's signature restaurant offering seasonal modern European and Indian cuisine in a welcoming setting."
        )
        q = "Does it offer food suitable for children?"
        resp = asyncio.run(self.response_service.generate_response(q, conversation_context=prior_context))

        self.assertFalse(
            resp.startswith("For families, we recommend Pluck"),
            f"Response incorrectly repeated generic family recommendation: {resp}"
        )
        self.assertIn("Pluck", resp)
        self.assertTrue(
            "not confirmed" in resp.lower() or "check" in resp.lower(),
            f"Expected truthful response indicating unconfirmed details, got: {resp}"
        )

    def test_fix_a_explicit_family_recommendation_preserved(self):
        """Explicit requests for family recommendations continue to return Pluck."""
        prior_context = (
            "Customer: What dining options do you have?\n"
            "Assistant: The hotel offers a variety of dining options, including Pluck, Honk, Café Pluck, and The Chidya Ghar Bar."
        )
        q = "Which restaurant do you recommend for my family?"
        resp = asyncio.run(self.response_service.generate_response(q, conversation_context=prior_context))
        self.assertTrue(
            resp.startswith("For families, we recommend Pluck"),
            f"Expected family recommendation for Pluck, got: {resp}"
        )


if __name__ == "__main__":
    unittest.main()
