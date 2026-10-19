import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
import unittest
from app.services.knowledge_service import KnowledgeService
from app.services.response_service import ResponseService


class TestUnseenRestaurantGeneralizationTask15(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.knowledge_service = KnowledgeService()
        cls.response_service = ResponseService()
        cls.company_id = "pullman_aerocity"
        cls.system_instruction = cls.knowledge_service.get_system_instruction(cls.company_id)

    def _ask(self, query: str, context: str = "") -> str:
        async def _run():
            kb = self.knowledge_service.get_relevant_knowledge(self.company_id, query)
            return await self.response_service.generate_response(
                transcript=query,
                conversation_context=context,
                system_instruction=self.system_instruction,
                knowledge_context=kb,
            )
        return asyncio.run(_run())

    # Category 1: General restaurant discovery
    def test_01_general_restaurant_discovery(self):
        resp = self._ask("What restaurants do you have?")
        self.assertTrue(any(v in resp for v in ["Pluck", "Honk", "Café Pluck", "The Chidya Ghar Bar"]))
        self.assertNotIn("+91", resp)

        resp2 = self._ask("What dining options are available?")
        self.assertIn("Pluck", resp2)
        self.assertIn("Honk", resp2)
        self.assertIn("Café Pluck", resp2)
        self.assertIn("The Chidya Ghar Bar", resp2)

    # Category 2: Cuisine-based recommendations
    def test_02_cuisine_based_recommendations(self):
        resp_indian = self._ask("Where can I get Indian food?")
        self.assertIn("Pluck", resp_indian)
        self.assertNotIn("Honk", resp_indian)

        resp_asian = self._ask("Which place serves Asian cuisine?")
        self.assertIn("Honk", resp_asian)
        self.assertNotIn("Pluck is the hotel's signature restaurant offering modern European", resp_asian)

        resp_euro = self._ask("Can I get European food here?")
        self.assertIn("Pluck", resp_euro)

    # Category 3: Restaurant-specific questions about Pluck, Honk, Café Pluck, and The Chidya Ghar Bar
    def test_03_restaurant_specific_inquiries(self):
        resp_pluck = self._ask("Tell me about Pluck's philosophy.")
        self.assertIn("Pluck", resp_pluck)
        self.assertTrue("farm-to-table" in resp_pluck or "organic herb garden" in resp_pluck)

        resp_honk = self._ask("What kind of place is Honk?")
        self.assertIn("Honk", resp_honk)
        self.assertTrue("pan-Asian" in resp_honk or "street food" in resp_honk or "dim sum" in resp_honk)

        resp_bar = self._ask("What does The Chidya Ghar Bar offer?")
        self.assertIn("The Chidya Ghar Bar", resp_bar)
        self.assertTrue("cocktails" in resp_bar or "spirits" in resp_bar or "wines" in resp_bar)

    # Category 4: Coffee and late-night dining questions
    def test_04_coffee_and_late_night_dining(self):
        resp_coffee = self._ask("Where can I grab a cup of coffee?")
        self.assertIn("Café Pluck", resp_coffee)
        self.assertIn("coffee", resp_coffee.lower())

        resp_late = self._ask("Can I eat late at night?")
        self.assertIn("Café Pluck", resp_late)
        self.assertTrue("24-hour" in resp_late or "24 hours" in resp_late)

        resp_2am = self._ask("Can I get food at 2 AM?")
        self.assertIn("Café Pluck", resp_2am)

        resp_bakery = self._ask("Do you have an on-site bakery?")
        self.assertIn("Café Pluck", resp_bakery)
        self.assertIn("bakery", resp_bakery.lower())

    # Category 5: Missing information (room service, table reservations, dish prices, dietary options, direct phone, hours)
    def test_05_missing_information_truthful_handling(self):
        resp_room_service = self._ask("Can I get food delivered to my room?")
        resp_lower = resp_room_service.lower()
        self.assertNotIn("yes, the hotel offers food delivery", resp_lower)
        self.assertTrue("not listed" in resp_lower or "front desk" in resp_lower)

        resp_table = self._ask("Can I book a table at Pluck for dinner?")
        resp_table_lower = resp_table.lower()
        self.assertNotIn("preferred date and time", resp_table_lower)
        self.assertTrue("not listed" in resp_table_lower or "front desk" in resp_table_lower)

        resp_prices = self._ask("What are the prices of dishes at Pluck?")
        resp_prices_lower = resp_prices.lower()
        self.assertNotIn("live room rates", resp_prices_lower)
        self.assertTrue("not available" in resp_prices_lower or "front desk" in resp_prices_lower)

        resp_dietary = self._ask("Do you have vegan or gluten-free options?")
        resp_dietary_lower = resp_dietary.lower()
        self.assertNotIn("i didn't quite catch that", resp_dietary_lower)
        self.assertTrue("not listed" in resp_dietary_lower or "front desk" in resp_dietary_lower)

        resp_phone = self._ask("What is the contact number for Honk restaurant?")
        self.assertNotIn("+91", resp_phone)
        self.assertNotIn("0123", resp_phone)
        self.assertTrue("front desk" in resp_phone.lower())

        resp_hours = self._ask("What are the hours for Honk restaurant?")
        self.assertNotIn("10:00 AM", resp_hours)
        self.assertTrue("not listed" in resp_hours.lower() or "front desk" in resp_hours.lower())

    # Category 6: Multi-part questions
    def test_06_multipart_questions(self):
        resp_multipart = self._ask("Do you have a bar, and is there somewhere for late-night coffee?")
        self.assertIn("The Chidya Ghar Bar", resp_multipart)
        self.assertIn("Café Pluck", resp_multipart)

        resp_multi2 = self._ask("Which restaurant serves Indian food, and what does Honk offer?")
        self.assertIn("Pluck", resp_multi2)
        self.assertIn("Honk", resp_multi2)

    # Category 7: Follow-up questions
    def test_07_followup_questions(self):
        ctx = "Customer: What restaurants do you have?\nAssistant: Our dining venues are Pluck, Honk, Café Pluck, and The Chidya Ghar Bar."
        resp_followup = self._ask("Tell me more about the Asian one.", context=ctx)
        self.assertIn("Honk", resp_followup)
        self.assertNotIn("Our dining venues are", resp_followup)

    # Category 8: Paraphrased questions & STT variations
    def test_08_paraphrased_questions_and_stt_variations(self):
        resp_drinks = self._ask("any spots for a drink tonight?")
        self.assertIn("The Chidya Ghar Bar", resp_drinks)

        resp_dimsum = self._ask("Where can I get dim sum and sushi?")
        self.assertIn("Honk", resp_dimsum)

        resp_thai = self._ask("Can I get Thai or Vietnamese food?")
        self.assertIn("Honk", resp_thai)

    # Category 9: Questions unrelated to restaurants (ensuring no dining bleed)
    def test_09_unrelated_questions_no_dining_bleed(self):
        resp_wifi = self._ask("What is the Wi-Fi access policy?")
        self.assertNotIn("Pluck", resp_wifi)
        self.assertNotIn("Honk", resp_wifi)
        self.assertIn("Wi-Fi", resp_wifi)

        resp_pets = self._ask("Are pets allowed in the hotel?")
        self.assertNotIn("Pluck", resp_pets)
        self.assertNotIn("Honk", resp_pets)
        self.assertTrue("pet" in resp_pets.lower() or "service animal" in resp_pets.lower())

    # Category 10: Unclear questions (verifying clarification is requested)
    def test_10_unclear_questions_request_clarification(self):
        resp_timing = self._ask("What time is the chicken?")
        self.assertTrue("clarify what timing" in resp_timing.lower())

        resp_garbled = self._ask("Booyo helps off.")
        self.assertTrue("repeat your question" in resp_garbled.lower())

        resp_traffic = self._ask("It's traffic block open for 24 hours.")
        self.assertNotIn("24-hour", resp_traffic.lower())
        self.assertTrue("repeat your question" in resp_traffic.lower())


if __name__ == "__main__":
    unittest.main()
