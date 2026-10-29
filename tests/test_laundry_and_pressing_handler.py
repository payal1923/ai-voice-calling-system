import unittest
import asyncio
from app.services.response_service import ResponseService
from app.services.knowledge_service import KnowledgeService

VERIFIED_LAUNDRY_FACT = (
    "Same-day professional laundry, dry cleaning, and pressing services "
    "are available for all in-house guests upon request through housekeeping."
)


class TestLaundryAndPressingHandler(unittest.TestCase):
    """
    Focused test suite validating the deterministic laundry, ironing, and garment-pressing
    handler in ResponseService, ensuring positive intent variations succeed and negative
    cases are not intercepted.
    """

    def setUp(self):
        self.response_service = ResponseService()
        self.knowledge_service = KnowledgeService()
        self.knowledge_context = self.knowledge_service.get_relevant_knowledge(
            "pullman_aerocity", "laundry and ironing services"
        )

    def test_01_exact_failed_ironing_phrasing(self):
        """The exact failed phrasing from Task 52 must deterministically yield verified fact."""
        query = "Can you help me with ironing my clothes?"
        kb = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", query)
        resp = self.response_service._get_laundry_response(query, kb)
        self.assertEqual(resp, VERIFIED_LAUNDRY_FACT)

        # Full synchronous response path
        sync_resp = asyncio.run(
            self.response_service.generate_response(query, knowledge_context=kb)
        )
        self.assertEqual(sync_resp, VERIFIED_LAUNDRY_FACT)
        self.assertNotIn("type of iron", sync_resp.lower())
        self.assertNotIn("items you would like ironed", sync_resp.lower())

    def test_02_positive_intent_variations(self):
        """All variations of laundry, ironing, dry cleaning, and pressing must match."""
        positive_queries = [
            "Do you have laundry service?",
            "Where can I get clothes ironed?",
            "Can I get my shirt pressed?",
            "Do you offer dry cleaning?",
            "Can I wash my clothes here?",
            "Do you provide an iron and ironing board?",
            "Can someone iron my suit?",
            "What laundry facilities are available for guests?",
            "Can you help me get these garments pressed?",
            "How can I do my laundry?",
            "Can I request pressing service for my dress?",
        ]
        for q in positive_queries:
            kb = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", q)
            resp = self.response_service._get_laundry_response(q, kb)
            self.assertEqual(
                resp,
                VERIFIED_LAUNDRY_FACT,
                f"Positive query '{q}' did not return verified laundry fact.",
            )

    def test_03_negative_unrelated_questions_with_clothes_or_keywords(self):
        """Unrelated questions mentioning 'clothes', 'press', 'iron', or 'wash' must NOT be intercepted."""
        negative_queries = [
            "What clothes should I pack for Delhi weather?",
            "Is there a dress code for dinner at Pluck?",
            "Where can I buy clothes nearby?",
            "I forgot my clothes in room 302 after checkout.",
            "Can I wash my hands in the lobby?",
            "Press 1 for reservations or front desk.",
            "Does the gym have iron dumbbells or weights?",
            "I have a pressing question about check-in.",
            "Is there iron in the tap water?",
            "What time is check-out?",
            "Do you have a heated swimming pool?",
        ]
        for q in negative_queries:
            resp = self.response_service._get_laundry_response(q, self.knowledge_context)
            self.assertIsNone(
                resp,
                f"Negative query '{q}' was incorrectly intercepted by laundry handler.",
            )

    def test_04_streaming_pipeline_execution(self):
        """Streaming generator must yield the verified fact immediately without LLM delay or fluff."""
        query = "Can you help me with ironing my clothes?"

        async def run_stream():
            sentences = []
            async for s in self.response_service.generate_response_stream_sentences(
                query, company_id="pullman_aerocity"
            ):
                sentences.append(s)
            return sentences

        sentences = asyncio.run(run_stream())
        self.assertEqual(len(sentences), 1)
        self.assertEqual(sentences[0], VERIFIED_LAUNDRY_FACT)
        self.assertNotIn("type of iron", sentences[0].lower())

    def test_05_streaming_with_prior_conversation_history(self):
        """Turn 3 in a multi-turn conversation (identical to Call 1127) must yield verified fact."""
        history = (
            "customer: Thank you, thank you, what time it's check-out time.\n"
            "assistant: Check-out at Pullman New Delhi Aerocity is at 12:00 PM (12:00).\n"
            "customer: Where is the nearest elevator?\n"
            "assistant: The nearest elevator is located on the 1st floor. Please check it out."
        )
        query = "Can you help me with ironing my clothes?"

        async def run_stream():
            sentences = []
            async for s in self.response_service.generate_response_stream_sentences(
                query, conversation_context=history, company_id="pullman_aerocity"
            ):
                sentences.append(s)
            return sentences

        sentences = asyncio.run(run_stream())
        self.assertEqual(len(sentences), 1)
        self.assertEqual(sentences[0], VERIFIED_LAUNDRY_FACT)
        self.assertNotIn("elevator", sentences[0].lower())
        self.assertNotIn("1st floor", sentences[0].lower())
        self.assertNotIn("type of iron", sentences[0].lower())

    def test_06_knowledge_context_isolation(self):
        """If knowledge context has no laundry/pressing facts, the handler must return None."""
        query = "Can you help me with ironing my clothes?"
        empty_kb = ""
        resp_empty = self.response_service._get_laundry_response(query, empty_kb)
        self.assertIsNone(resp_empty)

        dining_only_kb = "- Pluck serves modern European cuisine."
        resp_dining = self.response_service._get_laundry_response(query, dining_only_kb)
        self.assertIsNone(resp_dining)

    def test_07_shirt_pressing_routes_to_verified_service(self):
        """Case 1: 'Can I get my shirt pressed?' routes to verified pressing service in knowledge."""
        query = "Can I get my shirt pressed?"
        kb = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", query)
        self.assertIn("pressing services", kb.lower())

        # Synchronous response path
        sync_resp = asyncio.run(
            self.response_service.generate_response(query, knowledge_context=kb)
        )
        self.assertEqual(sync_resp, VERIFIED_LAUNDRY_FACT)

        # Streaming sentence generator path
        async def run_stream():
            sentences = []
            async for s in self.response_service.generate_response_stream_sentences(
                query, company_id="pullman_aerocity"
            ):
                sentences.append(s)
            return sentences

        stream_sentences = asyncio.run(run_stream())
        self.assertEqual(len(stream_sentences), 1)
        self.assertEqual(stream_sentences[0], VERIFIED_LAUNDRY_FACT)

        # First-sentence fast stream path
        first_sentence, _ = asyncio.run(
            self.response_service.generate_response_stream_sentence(query, knowledge_context=kb)
        )
        self.assertEqual(first_sentence, VERIFIED_LAUNDRY_FACT)

    def test_08_ambiguous_shirt_request_asks_clarification_not_facility_denial(self):
        """Case 2: 'Can I get my shirt please?' does not trigger unsupported-facility denial."""
        query = "Can I get my shirt please?"
        kb = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", query)

        # Synchronous path
        sync_resp = asyncio.run(
            self.response_service.generate_response(query, knowledge_context=kb)
        )
        self.assertNotIn("not confirmed in our available hotel information", sync_resp.lower())
        self.assertNotIn("do not offer that facility", sync_resp.lower())
        self.assertIn("clarify", sync_resp.lower())
        self.assertIn("shirt", sync_resp.lower())

        # Streaming path
        async def run_stream():
            sentences = []
            async for s in self.response_service.generate_response_stream_sentences(
                query, company_id="pullman_aerocity"
            ):
                sentences.append(s)
            return sentences

        stream_sentences = asyncio.run(run_stream())
        self.assertTrue(len(stream_sentences) >= 1)
        combined_stream = " ".join(stream_sentences).lower()
        self.assertNotIn("not confirmed in our available hotel information", combined_stream)
        self.assertNotIn("do not offer that facility", combined_stream)
        self.assertIn("clarify", combined_stream)

    def test_09_clearly_unsupported_facilities_receive_safe_denial(self):
        """Case 3: Clearly unsupported facility queries safely receive unsupported-facility response."""
        unsupported_queries = [
            "Do you have a casino?",
            "Can I get into the casino?",
            "Is there a casino at the hotel?",
            "Do you have a bowling alley?",
            "Can I book a helicopter?",
            "Do you have an ice skating rink?",
            "Do you have a business center?",
        ]
        for q in unsupported_queries:
            kb = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", q)
            resp = asyncio.run(
                self.response_service.generate_response(q, knowledge_context=kb)
            )
            resp_lower = resp.lower()
            is_safe_denial = (
                "do not offer that facility" in resp_lower
                or "not confirmed in our available hotel information" in resp_lower
            )
            self.assertTrue(
                is_safe_denial,
                f"Query '{q}' did not receive a safe unsupported facility denial. Got: {resp}",
            )

    def test_10_generic_ambiguous_can_i_get_asks_clarification(self):
        """Case 4: Generic ambiguous 'Can I get...' requests ask for clarification when intent is uncertain."""
        ambiguous_queries = [
            "Can I get that please?",
            "Can I get assistance?",
            "Can I get please?",
            "Can I get...",
            "Could I get that please?",
        ]
        for q in ambiguous_queries:
            kb = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", q)
            resp = asyncio.run(
                self.response_service.generate_response(q, knowledge_context=kb)
            )
            resp_lower = resp.lower()
            self.assertNotIn(
                "not confirmed in our available hotel information",
                resp_lower,
                f"Query '{q}' falsely claimed facility was not confirmed.",
            )
            self.assertNotIn(
                "do not offer that facility",
                resp_lower,
                f"Query '{q}' falsely claimed facility was not offered.",
            )
            self.assertIn(
                "clarify",
                resp_lower,
                f"Query '{q}' did not ask for clarification. Got: {resp}",
            )

    def test_11_unrelated_services_and_policies_preserved(self):
        """Case 5: Existing laundry, ironing, elevator, check-out, and room queries continue to work."""
        # 1. Ironing / iron and board query
        iron_q = "Can I get an iron and ironing board?"
        iron_kb = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", iron_q)
        iron_resp = asyncio.run(
            self.response_service.generate_response(iron_q, knowledge_context=iron_kb)
        )
        self.assertEqual(iron_resp, VERIFIED_LAUNDRY_FACT)

        # 2. Check-out timing query
        checkout_q = "What time is check-out?"
        checkout_kb = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", checkout_q)
        checkout_resp = asyncio.run(
            self.response_service.generate_response(checkout_q, knowledge_context=checkout_kb)
        )
        self.assertIn("12:00", checkout_resp)

        # 3. Check-in timing query
        checkin_q = "What time is check-in?"
        checkin_kb = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", checkin_q)
        checkin_resp = asyncio.run(
            self.response_service.generate_response(checkin_q, knowledge_context=checkin_kb)
        )
        self.assertIn("3:00", checkin_resp)

    def test_12_compound_laundry_and_checkout_both_orders(self):
        """Multi-intent: Laundry + check-out in both question orders across sync and streaming paths."""
        # 1. Order A: Laundry first, then check-out
        q_a = "Can you arrange laundry service and tell me the hotel check-out time?"
        kb_a = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", q_a)
        resp_a = asyncio.run(
            self.response_service.generate_response(q_a, knowledge_context=kb_a)
        )
        self.assertIn("laundry, dry cleaning, and pressing", resp_a)
        self.assertIn("12:00", resp_a)
        laundry_pos_a = resp_a.find("laundry")
        checkout_pos_a = resp_a.find("12:00")
        self.assertTrue(laundry_pos_a < checkout_pos_a, f"Expected laundry before check-out in: {resp_a}")

        # Streaming sentence generator for Order A
        async def run_stream_a():
            sentences = []
            async for s in self.response_service.generate_response_stream_sentences(
                q_a, company_id="pullman_aerocity"
            ):
                sentences.append(s)
            return sentences

        stream_sentences_a = asyncio.run(run_stream_a())
        self.assertTrue(len(stream_sentences_a) >= 2)
        self.assertIn("laundry", stream_sentences_a[0].lower())
        self.assertIn("12:00", stream_sentences_a[1])

        # Fast first-sentence path for Order A
        first_a, _ = asyncio.run(
            self.response_service.generate_response_stream_sentence(q_a, knowledge_context=kb_a)
        )
        self.assertIn("laundry", first_a.lower())

        # 2. Order B: Check-out first, then laundry
        q_b = "What time is check-out and can you help with laundry service?"
        kb_b = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", q_b)
        resp_b = asyncio.run(
            self.response_service.generate_response(q_b, knowledge_context=kb_b)
        )
        self.assertIn("12:00", resp_b)
        self.assertIn("laundry, dry cleaning, and pressing", resp_b)
        checkout_pos_b = resp_b.find("12:00")
        laundry_pos_b = resp_b.find("laundry")
        self.assertTrue(checkout_pos_b < laundry_pos_b, f"Expected check-out before laundry in: {resp_b}")

        # Streaming sentence generator for Order B
        async def run_stream_b():
            sentences = []
            async for s in self.response_service.generate_response_stream_sentences(
                q_b, company_id="pullman_aerocity"
            ):
                sentences.append(s)
            return sentences

        stream_sentences_b = asyncio.run(run_stream_b())
        self.assertTrue(len(stream_sentences_b) >= 2)
        self.assertIn("12:00", stream_sentences_b[0])
        self.assertIn("laundry", stream_sentences_b[1].lower())

    def test_13_compound_laundry_and_swimming_pool(self):
        """Multi-intent: Laundry + swimming pool questions answered with fact-grounded responses in both orders."""
        # Order A: Laundry first, pool second
        q_a = "Can you help me with laundry and do you have a swimming pool?"
        kb_a = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", q_a)
        resp_a = asyncio.run(
            self.response_service.generate_response(q_a, knowledge_context=kb_a)
        )
        self.assertIn("laundry, dry cleaning, and pressing", resp_a)
        self.assertIn("heated outdoor swimming pool", resp_a)
        self.assertTrue(resp_a.find("laundry") < resp_a.find("pool"))

        # Order B: Pool first, laundry second
        q_b = "Is there a swimming pool and can you iron my clothes?"
        kb_b = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", q_b)
        resp_b = asyncio.run(
            self.response_service.generate_response(q_b, knowledge_context=kb_b)
        )
        self.assertIn("heated outdoor swimming pool", resp_b)
        self.assertIn("laundry, dry cleaning, and pressing", resp_b)
        self.assertTrue(resp_b.find("pool") < resp_b.find("laundry"))

    def test_14_compound_wifi_and_checkin(self):
        """Multi-intent: Wi-Fi + check-in timing in both orders."""
        # Order A: Wi-Fi first, check-in second
        q_a = "How do I connect to Wi-Fi and what time is check-in?"
        kb_a = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", q_a)
        resp_a = asyncio.run(
            self.response_service.generate_response(q_a, knowledge_context=kb_a)
        )
        self.assertIn("Wi-Fi", resp_a)
        self.assertIn("3:00 PM", resp_a)
        self.assertTrue(resp_a.lower().find("wi-fi") < resp_a.find("3:00"))

        # Order B: Check-in first, Wi-Fi second
        q_b = "What time is check-in and is Wi-Fi complimentary?"
        kb_b = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", q_b)
        resp_b = asyncio.run(
            self.response_service.generate_response(q_b, knowledge_context=kb_b)
        )
        self.assertIn("3:00 PM", resp_b)
        self.assertIn("Wi-Fi", resp_b)
        self.assertTrue(resp_b.find("3:00") < resp_b.lower().find("wi-fi"))

    def test_15_single_intent_queries_preserved(self):
        """Single-intent laundry, check-out, and pool requests return concise, non-duplicated single answers."""
        # 1. Single laundry query
        q_laundry = "Can you arrange laundry service?"
        kb_laundry = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", q_laundry)
        compound_check = self.response_service._get_compound_knowledge_response(q_laundry, kb_laundry)
        self.assertIsNone(compound_check, "Single-intent laundry query should not trigger compound aggregator.")
        resp_laundry = asyncio.run(
            self.response_service.generate_response(q_laundry, knowledge_context=kb_laundry)
        )
        self.assertEqual(resp_laundry, VERIFIED_LAUNDRY_FACT)

        # 2. Single check-out query
        q_checkout = "What time is check-out?"
        kb_checkout = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", q_checkout)
        compound_check_co = self.response_service._get_compound_knowledge_response(q_checkout, kb_checkout)
        self.assertIsNone(compound_check_co, "Single-intent check-out query should not trigger compound aggregator.")
        resp_checkout = asyncio.run(
            self.response_service.generate_response(q_checkout, knowledge_context=kb_checkout)
        )
        self.assertEqual(resp_checkout, "Check-out at Pullman New Delhi Aerocity is at 12:00 PM (12:00).")

    def test_16_narrow_compound_and_unclear_queries_preserved(self):
        """Existing narrow compound concierge behavior (Call 1108) and unclear requests remain preserved."""
        # 1. Call 1108 compound concierge request
        q_concierge = "I need an airport pickup tomorrow morning, a wake-up call at 6 AM, and packed breakfast."
        kb_concierge = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", q_concierge)
        resp_concierge = asyncio.run(
            self.response_service.generate_response(q_concierge, knowledge_context=kb_concierge)
        )
        self.assertIn("I have noted your requests", resp_concierge)
        self.assertIn("concierge and front desk team", resp_concierge)

        # 2. Unclear / garbled transcript
        q_garbled = "Where is your happy look at you?"
        kb_garbled = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", q_garbled)
        resp_garbled = asyncio.run(
            self.response_service.generate_response(q_garbled, knowledge_context=kb_garbled)
        )
        self.assertIn("didn't quite catch that", resp_garbled.lower())

    def test_17_oxford_comma_three_intents_both_orders(self):
        """Three-intent Oxford-comma inquiries must not drop any intent and must preserve spoken order."""
        # 1. Forward order: laundry, check-out, wifi
        q_fwd = "Can you arrange laundry, tell me check-out time, and is Wi-Fi free?"
        kb_fwd = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", q_fwd)
        resp_fwd = asyncio.run(
            self.response_service.generate_response(q_fwd, knowledge_context=kb_fwd)
        )
        self.assertIn("laundry, dry cleaning, and pressing", resp_fwd)
        self.assertIn("12:00", resp_fwd)
        self.assertIn("Wi-Fi", resp_fwd)
        pos_laundry_fwd = resp_fwd.find("laundry")
        pos_checkout_fwd = resp_fwd.find("12:00")
        pos_wifi_fwd = resp_fwd.find("Wi-Fi")
        self.assertTrue(
            pos_laundry_fwd < pos_checkout_fwd < pos_wifi_fwd,
            f"Expected laundry < checkout < wifi in: {resp_fwd}",
        )

        # Streaming check for forward order
        async def run_stream_fwd():
            sentences = []
            async for s in self.response_service.generate_response_stream_sentences(
                q_fwd, company_id="pullman_aerocity"
            ):
                sentences.append(s)
            return sentences

        stream_fwd = asyncio.run(run_stream_fwd())
        self.assertEqual(len(stream_fwd), 3)
        self.assertIn("laundry", stream_fwd[0].lower())
        self.assertIn("12:00", stream_fwd[1])
        self.assertIn("wi-fi", stream_fwd[2].lower())

        # 2. Reverse order: check-out, wifi, laundry
        q_rev = "Tell me check-out time, is Wi-Fi free, and can you arrange laundry?"
        kb_rev = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", q_rev)
        resp_rev = asyncio.run(
            self.response_service.generate_response(q_rev, knowledge_context=kb_rev)
        )
        self.assertIn("12:00", resp_rev)
        self.assertIn("Wi-Fi", resp_rev)
        self.assertIn("laundry, dry cleaning, and pressing", resp_rev)
        pos_checkout_rev = resp_rev.find("12:00")
        pos_wifi_rev = resp_rev.find("Wi-Fi")
        pos_laundry_rev = resp_rev.find("laundry")
        self.assertTrue(
            pos_checkout_rev < pos_wifi_rev < pos_laundry_rev,
            f"Expected checkout < wifi < laundry in: {resp_rev}",
        )

    def test_18_mixed_supported_and_unsupported_facilities_both_orders(self):
        """Mixed supported and unsupported facility inquiries return both the fact and safe denial in order."""
        # 1. Forward order: laundry + casino (unsupported leisure facility)
        q_fwd = "Can you arrange laundry and do you have a casino?"
        kb_fwd = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", q_fwd)
        resp_fwd = asyncio.run(
            self.response_service.generate_response(q_fwd, knowledge_context=kb_fwd)
        )
        self.assertIn("laundry, dry cleaning, and pressing", resp_fwd)
        self.assertIn("do not offer that facility", resp_fwd)
        pos_laundry = resp_fwd.find("laundry")
        pos_casino_refusal = resp_fwd.find("do not offer that facility")
        self.assertTrue(pos_laundry < pos_casino_refusal, f"Expected laundry before casino refusal in: {resp_fwd}")

        # Streaming check for forward order
        async def run_stream_fwd():
            sentences = []
            async for s in self.response_service.generate_response_stream_sentences(
                q_fwd, company_id="pullman_aerocity"
            ):
                sentences.append(s)
            return sentences

        stream_fwd = asyncio.run(run_stream_fwd())
        self.assertTrue(len(stream_fwd) >= 2)
        self.assertIn("laundry", stream_fwd[0].lower())
        self.assertIn("do not offer that facility", " ".join(stream_fwd[1:]))

        # 2. Reverse order: casino + laundry
        q_rev = "Do you have a casino and can you arrange laundry?"
        kb_rev = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", q_rev)
        resp_rev = asyncio.run(
            self.response_service.generate_response(q_rev, knowledge_context=kb_rev)
        )
        self.assertIn("do not offer that facility", resp_rev)
        self.assertIn("laundry, dry cleaning, and pressing", resp_rev)
        pos_casino_rev = resp_rev.find("do not offer that facility")
        pos_laundry_rev = resp_rev.find("laundry")
        self.assertTrue(pos_casino_rev < pos_laundry_rev, f"Expected casino refusal before laundry in: {resp_rev}")

        # 3. Forward order: laundry + business center (unlisted amenity)
        q_amenity_fwd = "Can you arrange laundry and do you have a business center?"
        kb_amenity_fwd = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", q_amenity_fwd)
        resp_amenity_fwd = asyncio.run(
            self.response_service.generate_response(q_amenity_fwd, knowledge_context=kb_amenity_fwd)
        )
        self.assertIn("laundry, dry cleaning, and pressing", resp_amenity_fwd)
        self.assertIn("not confirmed in our available hotel information", resp_amenity_fwd)
        self.assertTrue(resp_amenity_fwd.find("laundry") < resp_amenity_fwd.find("not confirmed"))

        # 4. Reverse order: business center + check-out timing
        q_amenity_rev = "Do you have a business center and what time is check-out?"
        kb_amenity_rev = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", q_amenity_rev)
        resp_amenity_rev = asyncio.run(
            self.response_service.generate_response(q_amenity_rev, knowledge_context=kb_amenity_rev)
        )
        self.assertIn("not confirmed in our available hotel information", resp_amenity_rev)
        self.assertIn("12:00", resp_amenity_rev)
        self.assertTrue(resp_amenity_rev.find("not confirmed") < resp_amenity_rev.find("12:00"))

    def test_19_oxford_comma_with_eat_wifi_free_variation(self):
        """Exact failed phrasing from live testing: 'Can you arrange laundry, tell me check-out time, and eat Wi-Fi free?'"""
        q = "Can you arrange laundry, tell me check-out time, and eat Wi-Fi free?"
        kb = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", q)

        # 1. Full synchronous response path
        sync_resp = asyncio.run(
            self.response_service.generate_response(q, knowledge_context=kb)
        )
        self.assertIn("laundry, dry cleaning, and pressing", sync_resp)
        self.assertIn("12:00", sync_resp)
        self.assertIn("Wi-Fi", sync_resp)
        pos_laundry = sync_resp.find("laundry")
        pos_checkout = sync_resp.find("12:00")
        pos_wifi = sync_resp.find("Wi-Fi")
        self.assertTrue(
            pos_laundry < pos_checkout < pos_wifi,
            f"Expected laundry < checkout < wifi in: {sync_resp}",
        )

        # 2. Sentence-streaming pipeline
        async def run_stream():
            sentences = []
            async for s in self.response_service.generate_response_stream_sentences(
                q, company_id="pullman_aerocity"
            ):
                sentences.append(s)
            return sentences

        sentences = asyncio.run(run_stream())
        self.assertEqual(len(sentences), 3)
        self.assertIn("laundry", sentences[0].lower())
        self.assertIn("12:00", sentences[1])
        self.assertIn("wi-fi", sentences[2].lower())

        # 3. Fast first-sentence streaming path
        first_sentence, _ = asyncio.run(
            self.response_service.generate_response_stream_sentence(q, knowledge_context=kb)
        )
        self.assertIn("laundry", first_sentence.lower())
        self.assertIn("12:00", first_sentence)
        self.assertIn("wi-fi", first_sentence.lower())

    def test_20_multi_unsupported_facilities_both_orders_and_stt_variations(self):
        """Dual unsupported facility requests explicitly identify both facilities in order, including STT variations."""
        # 1. Forward order: casino + bowling alley
        q_fwd = "Do you have a casino and a bowling alley?"
        kb_fwd = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", q_fwd)
        resp_fwd = asyncio.run(
            self.response_service.generate_response(q_fwd, knowledge_context=kb_fwd)
        )
        self.assertIn("do not offer a casino or a bowling alley", resp_fwd)
        pos_casino = resp_fwd.find("a casino")
        pos_bowling = resp_fwd.find("a bowling alley")
        self.assertTrue(pos_casino < pos_bowling, f"Expected casino before bowling alley in: {resp_fwd}")
        self.assertIn("front desk", resp_fwd)

        # 2. Reverse order: bowling alley + casino
        q_rev = "Do you have a bowling alley and a casino?"
        kb_rev = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", q_rev)
        resp_rev = asyncio.run(
            self.response_service.generate_response(q_rev, knowledge_context=kb_rev)
        )
        self.assertIn("do not offer a bowling alley or a casino", resp_rev)
        pos_bowling_rev = resp_rev.find("a bowling alley")
        pos_casino_rev = resp_rev.find("a casino")
        self.assertTrue(pos_bowling_rev < pos_casino_rev, f"Expected bowling alley before casino in: {resp_rev}")
        self.assertIn("front desk", resp_rev)

        # 3. Speech-recognition phonetic variation: "blowing alley" -> "bowling alley" (forward)
        q_stt_fwd = "Do you have a casino and a blowing alley?"
        kb_stt_fwd = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", q_stt_fwd)
        resp_stt_fwd = asyncio.run(
            self.response_service.generate_response(q_stt_fwd, knowledge_context=kb_stt_fwd)
        )
        self.assertIn("do not offer a casino or a bowling alley", resp_stt_fwd)
        self.assertTrue(resp_stt_fwd.find("a casino") < resp_stt_fwd.find("a bowling alley"))

        # 4. Speech-recognition phonetic variation: "blowing alley" -> "bowling alley" (reverse)
        q_stt_rev = "Do you have a blowing alley and a casino?"
        kb_stt_rev = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", q_stt_rev)
        resp_stt_rev = asyncio.run(
            self.response_service.generate_response(q_stt_rev, knowledge_context=kb_stt_rev)
        )
        self.assertIn("do not offer a bowling alley or a casino", resp_stt_rev)
        self.assertTrue(resp_stt_rev.find("a bowling alley") < resp_stt_rev.find("a casino"))

        # 5. Streaming response parity
        async def run_stream():
            sentences = []
            async for s in self.response_service.generate_response_stream_sentences(
                q_stt_fwd, company_id="pullman_aerocity"
            ):
                sentences.append(s)
            return sentences

        stream_sentences = asyncio.run(run_stream())
        self.assertEqual(len(stream_sentences), 2)
        self.assertEqual(
            stream_sentences[0],
            "I apologize, but we do not offer a casino or a bowling alley at our hotel."
        )
        self.assertEqual(
            stream_sentences[1],
            "Please contact our front desk for information on our available amenities."
        )

        # 6. Mixed supported + dual unsupported compound request
        q_mixed = "Can you arrange laundry and do you have a casino and a bowling alley?"
        kb_mixed = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", q_mixed)
        resp_mixed = asyncio.run(
            self.response_service.generate_response(q_mixed, knowledge_context=kb_mixed)
        )
        self.assertIn("laundry, dry cleaning, and pressing", resp_mixed)
        self.assertIn("do not offer a casino or a bowling alley", resp_mixed)
        self.assertTrue(resp_mixed.find("laundry") < resp_mixed.find("casino or a bowling alley"))

        # 7. Single-facility queries continue to return standard verified template
        q_single_casino = "Do you have a casino?"
        kb_sc = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", q_single_casino)
        resp_sc = asyncio.run(
            self.response_service.generate_response(q_single_casino, knowledge_context=kb_sc)
        )
        self.assertEqual(
            resp_sc,
            "I apologize, but we do not offer that facility at our hotel. Please contact our front desk for information on our available amenities."
        )

        q_single_bowling_stt = "Do you have a blowing alley?"
        kb_sb = self.knowledge_service.get_relevant_knowledge("pullman_aerocity", q_single_bowling_stt)
        resp_sb = asyncio.run(
            self.response_service.generate_response(q_single_bowling_stt, knowledge_context=kb_sb)
        )
        self.assertEqual(
            resp_sb,
            "I apologize, but we do not offer that facility at our hotel. Please contact our front desk for information on our available amenities."
        )


if __name__ == "__main__":
    unittest.main()



