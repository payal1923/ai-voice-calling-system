import unittest
import asyncio
from app.services.response_service import ResponseService
from app.services.knowledge_service import KnowledgeService


class TestBreakfastAndGymHandlers(unittest.TestCase):
    """
    Focused unit tests for:
    1. Grounded breakfast timings & buffet handling (weekday & weekend).
    2. Grounded 24/7 Fit Lounge gym handling.
    3. Compound multi-question routing in both clause orders.
    4. Room-service vs. buffet breakfast distinction.
    5. Table booking & dining regression safeguards.
    6. Normal vs. streaming response consistency.
    """

    def setUp(self):
        self.ks = KnowledgeService()
        self.rs = ResponseService(llm_generator=None)
        self.company_id = "pullman_aerocity"
        self.system_instruction = self.ks.get_system_instruction(self.company_id)

    def _get_resp(self, query: str, context: str = "") -> str:
        knowledge = self.ks.get_relevant_knowledge(
            self.company_id, query, conversation_context=context
        )
        return asyncio.run(
            self.rs.generate_response(
                transcript=query,
                conversation_context=context,
                system_instruction=self.system_instruction,
                knowledge_context=knowledge,
            )
        )

    async def _get_stream_resp(self, query: str, context: str = "") -> str:
        knowledge = self.ks.get_relevant_knowledge(
            self.company_id, query, conversation_context=context
        )
        sentences = []
        async for s in self.rs.generate_response_stream_sentences(
            transcript=query,
            conversation_context=context,
            system_instruction=self.system_instruction,
            knowledge_context=knowledge,
        ):
            sentences.append(s)
        return " ".join(sentences).strip()

    # =========================================================================
    # 1. Direct Breakfast Questions (Accurate Weekday / Weekend Timings)
    # =========================================================================
    def test_01_direct_breakfast_timing_questions(self):
        """Direct breakfast timing queries return grounded Pluck timings for weekdays and weekends."""
        queries = [
            "What time is breakfast?",
            "What are your breakfast hours?",
            "What time is breakfast served?",
            "When can I get breakfast?",
            "What are the breakfast timings?",
        ]
        for q in queries:
            resp = self._get_resp(q)
            self.assertIn("Pluck", resp, f"Failed for query '{q}': missing Pluck venue")
            self.assertIn("6:30 AM", resp, f"Failed for query '{q}': missing 6:30 AM opening")
            self.assertIn("10:30 AM", resp, f"Failed for query '{q}': missing 10:30 AM weekday closing")
            self.assertIn("11:00 AM", resp, f"Failed for query '{q}': missing 11:00 AM weekend closing")
            self.assertIn("weekdays", resp.lower(), f"Failed for query '{q}': missing weekdays specification")
            self.assertIn("weekends", resp.lower(), f"Failed for query '{q}': missing weekends specification")
            self.assertNotIn("check with the front desk for current breakfast timings", resp.lower())

    def test_02_direct_breakfast_buffet_and_general_questions(self):
        """General breakfast and buffet queries return grounded breakfast information."""
        queries = [
            "Do you have a breakfast buffet?",
            "Is breakfast available at the hotel?",
            "Do you serve breakfast?",
            "Tell me about breakfast options.",
        ]
        for q in queries:
            resp = self._get_resp(q)
            self.assertIn("Pluck", resp)
            self.assertIn("buffet", resp.lower())
            self.assertIn("6:30 AM", resp)

    def test_03_specific_weekend_and_weekday_queries(self):
        """Queries specifically asking for weekend or weekday timings are accurately answered."""
        resp_weekend = self._get_resp("What time is breakfast on weekends?")
        self.assertIn("11:00 AM", resp_weekend)
        self.assertIn("weekends", resp_weekend.lower())

        resp_weekday = self._get_resp("What time is breakfast on weekdays?")
        self.assertIn("10:30 AM", resp_weekday)
        self.assertIn("weekdays", resp_weekday.lower())

    # =========================================================================
    # 2. Direct Gym / Fitness Center Questions
    # =========================================================================
    def test_04_direct_gym_inquiries(self):
        """Direct gym and fitness questions return 24/7 Fit Lounge facts."""
        queries = [
            "Do you have a gym?",
            "What are the gym hours?",
            "Is there a fitness center at the hotel?",
            "Where can I work out?",
            "Does the hotel have workout facilities?",
            "Are fitness facilities open 24 hours?",
        ]
        for q in queries:
            resp = self._get_resp(q)
            self.assertIn("Fit Lounge", resp, f"Failed for '{q}': missing Fit Lounge")
            self.assertIn("24/7", resp, f"Failed for '{q}': missing 24/7 access confirmation")
            self.assertTrue(
                "cardiovascular" in resp.lower() or "strength" in resp.lower() or "fitness center" in resp.lower(),
                f"Failed for '{q}': missing equipment details"
            )

    def test_05_gym_followup_pronoun_resolution(self):
        """Follow-up 'Is it open 24 hours?' resolves to Fit Lounge without dining cross-contamination."""
        context = (
            "Customer: Does the hotel have a fitness center?\n"
            "Assistant: The Fit Lounge is a 24/7 fitness center fully equipped with modern cardiovascular and strength-training machines."
        )
        q = "Is it open 24 hours?"
        resp = self._get_resp(q, context=context)
        self.assertIn("Fit Lounge", resp)
        self.assertIn("24/7", resp)
        self.assertNotIn("bar", resp.lower())
        self.assertNotIn("restaurant", resp.lower())

    # =========================================================================
    # 3. Compound Questions in Both Orders
    # =========================================================================
    def test_06_compound_breakfast_and_checkout_both_orders(self):
        """Breakfast + Check-out compound query answers both in forward and reverse orders."""
        q_fwd = "What time is breakfast and what time is check-out?"
        resp_fwd = self._get_resp(q_fwd)
        self.assertIn("Pluck", resp_fwd)
        self.assertIn("6:30 AM", resp_fwd)
        self.assertIn("12:00 PM", resp_fwd)
        # Verify order: breakfast should appear before check-out
        pos_breakfast = resp_fwd.find("Pluck")
        pos_checkout = resp_fwd.find("12:00 PM")
        self.assertLess(pos_breakfast, pos_checkout, "Expected breakfast before check-out")

        q_rev = "What time is check-out and what time is breakfast?"
        resp_rev = self._get_resp(q_rev)
        self.assertIn("Pluck", resp_rev)
        self.assertIn("12:00 PM", resp_rev)
        # Verify order: check-out should appear before breakfast
        pos_checkout_rev = resp_rev.find("12:00 PM")
        pos_breakfast_rev = resp_rev.find("Pluck")
        self.assertLess(pos_checkout_rev, pos_breakfast_rev, "Expected check-out before breakfast")

    def test_07_compound_gym_and_breakfast_both_orders(self):
        """Gym + Breakfast compound query answers both in forward and reverse orders."""
        q_fwd = "Do you have a gym and what time is breakfast?"
        resp_fwd = self._get_resp(q_fwd)
        self.assertIn("Fit Lounge", resp_fwd)
        self.assertIn("24/7", resp_fwd)
        self.assertIn("Pluck", resp_fwd)
        self.assertIn("6:30 AM", resp_fwd)
        self.assertLess(resp_fwd.find("Fit Lounge"), resp_fwd.find("Pluck"))

        q_rev = "What time is breakfast and do you have a gym?"
        resp_rev = self._get_resp(q_rev)
        self.assertIn("Fit Lounge", resp_rev)
        self.assertIn("Pluck", resp_rev)
        self.assertLess(resp_rev.find("Pluck"), resp_rev.find("Fit Lounge"))

    def test_08_compound_gym_and_swimming_pool_both_orders(self):
        """Gym + Swimming Pool compound query answers both in forward and reverse orders."""
        q_fwd = "Do you have a gym and a swimming pool?"
        resp_fwd = self._get_resp(q_fwd)
        self.assertIn("Fit Lounge", resp_fwd)
        self.assertIn("swimming pool", resp_fwd.lower())
        self.assertLess(resp_fwd.find("Fit Lounge"), resp_fwd.lower().find("swimming pool"))

        q_rev = "Do you have a swimming pool and a fitness center?"
        resp_rev = self._get_resp(q_rev)
        self.assertIn("Fit Lounge", resp_rev)
        self.assertIn("swimming pool", resp_rev.lower())
        self.assertLess(resp_rev.lower().find("swimming pool"), resp_rev.find("Fit Lounge"))

    def test_09_compound_breakfast_and_unlisted_facility(self):
        """Breakfast combined with unlisted facility answers breakfast and declines unlisted facility."""
        q_fwd = "What time is breakfast and do you have a casino?"
        resp_fwd = self._get_resp(q_fwd)
        self.assertIn("Pluck", resp_fwd)
        self.assertTrue("do not offer that facility" in resp_fwd.lower() or "not confirmed" in resp_fwd.lower())
        self.assertLess(resp_fwd.find("Pluck"), resp_fwd.lower().find("facility"))

        q_rev = "Do you have a casino and what time is breakfast?"
        resp_rev = self._get_resp(q_rev)
        self.assertIn("Pluck", resp_rev)
        self.assertTrue("do not offer that facility" in resp_rev.lower() or "not confirmed" in resp_rev.lower())
        self.assertLess(resp_rev.lower().find("facility"), resp_rev.find("Pluck"))

    def test_10_compound_gym_and_laundry_both_orders(self):
        """Gym + Laundry pressing compound query answers both cleanly."""
        q_fwd = "Do you have a gym and can I get my clothes pressed?"
        resp_fwd = self._get_resp(q_fwd)
        self.assertIn("Fit Lounge", resp_fwd)
        self.assertIn("pressing", resp_fwd.lower())
        self.assertLess(resp_fwd.find("Fit Lounge"), resp_fwd.lower().find("pressing"))

        q_rev = "Can I get my clothes pressed and do you have a gym?"
        resp_rev = self._get_resp(q_rev)
        self.assertIn("Fit Lounge", resp_rev)
        self.assertIn("pressing", resp_rev.lower())
        self.assertLess(resp_rev.lower().find("pressing"), resp_rev.find("Fit Lounge"))

    # =========================================================================
    # 4. Room Service vs. Breakfast Distinction
    # =========================================================================
    def test_11_room_service_breakfast_requests_route_to_room_service(self):
        """In-room dining / room service requests for breakfast route to room service declination, NOT buffet timings."""
        queries = [
            "Can I order breakfast to my room?",
            "Do you have room service for breakfast?",
            "Can I get breakfast delivered to my room?",
            "Can I order food to my room for breakfast?",
            "How do I order in-room dining for breakfast?",
        ]
        for q in queries:
            resp = self._get_resp(q)
            self.assertTrue(
                "not listed" in resp.lower() or "contact the front desk" in resp.lower(),
                f"Failed for '{q}': expected room service guidance, got: '{resp}'"
            )
            # Must NOT state Pluck restaurant buffet timings as available room service
            self.assertNotIn("Breakfast is served daily at Pluck restaurant", resp)

    # =========================================================================
    # 5. Table Booking & Dining Regression Safeguards
    # =========================================================================
    def test_12_table_reservation_for_breakfast_routes_to_table_guidance(self):
        """Table reservations for breakfast route to dining reservation guidance, NOT buffet hours."""
        q = "Can I reserve a table for breakfast?"
        resp = self._get_resp(q)
        self.assertTrue(
            "reservations" in resp.lower() or "front desk" in resp.lower() or "direct" in resp.lower(),
            f"Expected reservation guidance for table booking, got: '{resp}'"
        )
        self.assertNotIn("Breakfast is served daily at Pluck restaurant", resp)

    def test_13_dining_overview_preserved(self):
        """General dining overview query continues to return 4 venues without breakfast hijacking."""
        q = "What dining options are available at the hotel?"
        resp = self._get_resp(q)
        self.assertIn("Pluck", resp)
        self.assertIn("Honk", resp)
        self.assertIn("Café Pluck", resp)
        self.assertIn("The Chidya Ghar Bar", resp)

    def test_14_gym_and_laundry_isolation(self):
        """Gym inquiries route to Fit Lounge without hitting laundry, and laundry queries route to laundry."""
        q_gym = "Does the gym have iron dumbbells or weights?"
        resp_gym = self._get_resp(q_gym)
        self.assertIn("Fit Lounge", resp_gym)
        self.assertNotIn("laundry", resp_gym.lower())

        q_laundry = "Can you wash and iron my shirts?"
        resp_laundry = self._get_resp(q_laundry)
        self.assertIn("laundry, dry cleaning, and pressing", resp_laundry)
        self.assertNotIn("Fit Lounge", resp_laundry)

    # =========================================================================
    # 6. Streaming Path Consistency
    # =========================================================================
    def test_15_streaming_consistency_for_breakfast_and_gym(self):
        """generate_response and generate_response_stream_sentences produce matching content."""
        queries = [
            "What time is breakfast?",
            "Do you have a gym?",
            "Can I order breakfast to my room?",
            "What time is breakfast and what time is check-out?",
            "Do you have a gym and a swimming pool?",
            "Can I order breakfast to my room or do I have to eat at Pluck?",
            "Can I use Wi-Fi in my room?",
            "Can I get my laundry done and have it delivered to my room?",
            "Can I order food to my room?",
        ]
        for q in queries:
            normal_resp = self._get_resp(q)
            stream_resp = asyncio.run(self._get_stream_resp(q))
            self.assertEqual(
                normal_resp.strip(),
                stream_resp.strip(),
                f"Streaming mismatch for '{q}':\nNormal: '{normal_resp}'\nStream: '{stream_resp}'"
            )

    # =========================================================================
    # 7. Room Service vs. Restaurant Breakfast Compound Inquiries
    # =========================================================================
    def test_16_room_service_or_pluck_breakfast_compound(self):
        """'Can I order breakfast to my room or do I have to eat at Pluck?' includes both room-service info and Pluck breakfast timings."""
        q = "Can I order breakfast to my room or do I have to eat at Pluck?"
        resp = self._get_resp(q)
        self.assertTrue(
            "room service" in resp.lower() or "in-room dining" in resp.lower() or "front desk" in resp.lower(),
            f"Failed for '{q}': missing room-service info in '{resp}'"
        )
        self.assertIn("Pluck", resp)
        self.assertIn("6:30 AM", resp)
        self.assertIn("10:30 AM", resp)

    def test_17_room_service_or_eat_at_club_no_nightclub_refusal(self):
        """'Can I order breakfast to my room or do I have to eat at club?' does not trigger nightclub refusal and includes breakfast timings."""
        q = "Can I order breakfast to my room or do I have to eat at club?"
        resp = self._get_resp(q)
        self.assertNotIn("nightclub", resp.lower())
        self.assertTrue(
            "room service" in resp.lower() or "in-room dining" in resp.lower(),
            f"Failed for '{q}': missing room-service guidance in '{resp}'"
        )
        self.assertIn("Pluck", resp)
        self.assertIn("6:30 AM", resp)

    def test_18_standalone_food_to_room_does_not_return_breakfast_timings(self):
        """'Can I order food to my room?' must not incorrectly answer only with breakfast timings."""
        q = "Can I order food to my room?"
        resp = self._get_resp(q)
        self.assertNotIn("Breakfast is served daily at Pluck restaurant", resp)
        self.assertTrue(
            "room service" in resp.lower() or "in-room dining" in resp.lower() or "front desk" in resp.lower(),
            f"Failed for '{q}': expected room service guidance, got: '{resp}'"
        )

    def test_19_wifi_in_my_room_does_not_append_room_service(self):
        """'Can I use Wi-Fi in my room?' returns only Wi-Fi information without room-service guidance."""
        q = "Can I use Wi-Fi in my room?"
        resp = self._get_resp(q)
        self.assertTrue(
            "wi-fi" in resp.lower() or "wifi" in resp.lower(),
            f"Failed for '{q}': missing Wi-Fi info in '{resp}'"
        )
        self.assertNotIn("room service", resp.lower())
        self.assertNotIn("in-room dining", resp.lower())

    def test_20_laundry_delivered_to_my_room_does_not_append_room_service(self):
        """'Can I get my laundry done and have it delivered to my room?' returns laundry information without in-room dining guidance."""
        q = "Can I get my laundry done and have it delivered to my room?"
        resp = self._get_resp(q)
        self.assertTrue(
            "laundry" in resp.lower() or "dry cleaning" in resp.lower() or "housekeeping" in resp.lower(),
            f"Failed for '{q}': missing laundry info in '{resp}'"
        )
        self.assertNotIn("room service", resp.lower())
        self.assertNotIn("in-room dining", resp.lower())
        self.assertNotIn("dining options", resp.lower())


if __name__ == "__main__":
    unittest.main()
