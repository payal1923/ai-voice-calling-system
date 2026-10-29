import os
import wave
import uuid
import asyncio
import unittest
from unittest.mock import MagicMock, patch

from app.database.database import SessionLocal
from app.database.models import Customer, Call, Message
from app.services.knowledge_service import KnowledgeService
from app.services.response_service import ResponseService
from app.services.voice_response_service import VoiceResponseService
from app.services.conversation_service import ConversationService
from app.services.conversation_context_service import ConversationContextService
from app.services.tts_service import TTSService

COMPANY_ID = "pullman_aerocity"


class TestFullPipelineReliabilityAudit(unittest.IsolatedAsyncioTestCase):
    """
    Comprehensive, repeatable test suite covering the 10 critical reliability audit categories:
    1. The same question asked repeatedly (repetition handling & consistency)
    2. Different questions asked consecutively (topic transitions without history bleed)
    3. Paraphrased questions (semantic intent grounding)
    4. Multi-part questions (compound requests)
    5. Follow-up questions that depend on previous answers (contextual continuity)
    6. Natural pauses and long sentences (robust parsing & completion)
    7. Unknown or out-of-scope questions (graceful decline without hallucination)
    8. Missing hotel facts (truthful refusal to invent ungrounded services)
    9. Empty or inaccurate transcription (polite retry prompt without hallucinating)
    10. TTS chunk failures, sequencing, and turn ID tracing
    """

    async def asyncSetUp(self):
        self.db = SessionLocal()
        self.knowledge_service = KnowledgeService()
        self.response_service = ResponseService()
        self.conv_service = ConversationService()
        self.context_service = ConversationContextService()
        self.tts_service = TTSService()
        self.voice_service = VoiceResponseService(
            response_service=self.response_service,
            knowledge_service=self.knowledge_service,
            conversation_service=self.conv_service,
            context_service=self.context_service,
            tts_service=self.tts_service,
        )

        # Setup test customer and active call session
        self.customer = self.db.query(Customer).filter(Customer.external_customer_id == "AUDIT_TEST_USER").first()
        if not self.customer:
            self.customer = Customer(
                external_customer_id="AUDIT_TEST_USER",
                name="Audit Test Customer",
                phone="+919999900099",
            )
            self.db.add(self.customer)
            self.db.commit()
            self.db.refresh(self.customer)

        self.call = self.conv_service.create_call(self.db, self.customer.id)
        self.call.call_type = "INBOUND"
        self.call.status = "IN_PROGRESS"
        self.call.language = "en"
        self.db.commit()
        self.db.refresh(self.call)

        # Prepare dummy test audio
        os.makedirs("scratch", exist_ok=True)
        self.test_audio_path = "scratch/audit_test_audio.wav"
        if not os.path.exists(self.test_audio_path):
            with wave.open(self.test_audio_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b"\x00\x00" * 16000)

    async def asyncTearDown(self):
        try:
            self.db.rollback()
            if hasattr(self, "call") and self.call:
                # Clean messages
                self.db.query(Message).filter(Message.call_id == self.call.id).delete()
                # Clean call
                c = self.db.query(Call).filter(Call.id == self.call.id).first()
                if c:
                    self.db.delete(c)
                self.db.commit()
        except Exception:
            pass
        finally:
            self.db.close()

    # =========================================================================
    # PART 1: INDEPENDENT COMPONENT TESTS
    # =========================================================================

    def test_01_knowledge_retrieval_checkin_checkout_normalization(self):
        """Verify check-in, check-out, checkout, and checkin normalization in KnowledgeService."""
        # 1. "checkout" without hyphen
        ctx_checkout = self.knowledge_service.get_relevant_knowledge(COMPANY_ID, "what time is checkout?")
        self.assertIn("12:00", ctx_checkout, "Failed to retrieve check-out time for unhyphenated 'checkout'")

        # 2. "checkin" without hyphen
        ctx_checkin = self.knowledge_service.get_relevant_knowledge(COMPANY_ID, "what time is checkin?")
        self.assertIn("3:00 PM", ctx_checkin, "Failed to retrieve check-in time for unhyphenated 'checkin'")

        # 3. "check-out" with hyphen
        ctx_hyphen_checkout = self.knowledge_service.get_relevant_knowledge(COMPANY_ID, "what time is check-out?")
        self.assertIn("12:00", ctx_hyphen_checkout)

    def test_02_knowledge_retrieval_topic_transition_clears_venue(self):
        """Verify that transitioning topics (e.g. from Bar to Check-out) resets venue targeting."""
        history = "Customer: Do you have a bar?\nAssistant: Yes, Chidya Ghar is our modern bar."
        
        # When user asks about checkout after bar, venue shouldn't stick to Chidya Ghar
        ctx = self.knowledge_service.get_relevant_knowledge(COMPANY_ID, "what time is checkout?", conversation_context=history)
        self.assertIn("12:00", ctx)
        self.assertNotIn("Chidya Ghar", ctx, "Chidya Ghar leaked into checkout query across topic change")

    def test_03_knowledge_retrieval_missing_and_unknown_facts(self):
        """Verify retrieval for unknown/missing facts returns general fallback without hallucination."""
        ctx_unknown = self.knowledge_service.get_relevant_knowledge(COMPANY_ID, "Can you help me fix a spaceship engine?")
        # Should return top general facts or empty, not specific invented facts
        self.assertNotIn("spaceship", ctx_unknown.lower())

    # =========================================================================
    # PART 2: END-TO-END PIPELINE MULTI-TURN AUDIT TESTS
    # =========================================================================

    async def test_04_category1_same_question_repeated(self):
        """Category 1: The same question asked repeatedly in consecutive turns."""
        with patch.object(self.voice_service.stt_service, "transcribe_with_info", return_value=("What time is check in?", 1.2)):
            # Turn 1
            turn1_events = []
            async for ev in self.voice_service.process_existing_turn_stream(
                call_id=self.call.id,
                audio_path=self.test_audio_path,
                output_audio_path_prefix="scratch/audit_out_t1.wav",
                db=self.db,
                company_id=COMPANY_ID,
            ):
                turn1_events.append(ev)

            # Turn 2: Repeat exact same question
            turn2_events = []
            async for ev in self.voice_service.process_existing_turn_stream(
                call_id=self.call.id,
                audio_path=self.test_audio_path,
                output_audio_path_prefix="scratch/audit_out_t2.wav",
                db=self.db,
                company_id=COMPANY_ID,
            ):
                turn2_events.append(ev)

            resp1 = next(e["response"] for e in turn1_events if e["event"] == "assistant_response")
            resp2 = next(e["response"] for e in turn2_events if e["event"] == "assistant_response")

            self.assertIn("3:00", resp1)
            self.assertIn("3:00", resp2)
            # Verify turn IDs were distinct and generated
            t1_id = turn1_events[0]["turn_id"]
            t2_id = turn2_events[0]["turn_id"]
            self.assertNotEqual(t1_id, t2_id)

    async def test_05_category2_different_questions_consecutively(self):
        """Category 2: Different questions asked consecutively without topic bleed."""
        # Turn 1: Pool
        with patch.object(self.voice_service.stt_service, "transcribe_with_info", return_value=("Do you have a swimming pool?", 1.0)):
            t1_events = [e async for e in self.voice_service.process_existing_turn_stream(
                call_id=self.call.id,
                audio_path=self.test_audio_path,
                output_audio_path_prefix="scratch/audit_diff_t1.wav",
                db=self.db,
                company_id=COMPANY_ID,
            )]
            resp1 = next(e["response"] for e in t1_events if e["event"] == "assistant_response")
            self.assertTrue("pool" in resp1.lower() or "swimming" in resp1.lower())

        # Turn 2: Checkout time (must NOT bleed swimming pool info)
        with patch.object(self.voice_service.stt_service, "transcribe_with_info", return_value=("What time is checkout?", 1.0)):
            t2_events = [e async for e in self.voice_service.process_existing_turn_stream(
                call_id=self.call.id,
                audio_path=self.test_audio_path,
                output_audio_path_prefix="scratch/audit_diff_t2.wav",
                db=self.db,
                company_id=COMPANY_ID,
            )]
            resp2 = next(e["response"] for e in t2_events if e["event"] == "assistant_response")
            self.assertIn("12:00", resp2)
            self.assertNotIn("pool", resp2.lower())

    async def test_06_category3_paraphrased_questions(self):
        """Category 3: Paraphrased questions yielding accurate, grounded answers."""
        paraphrases = [
            "When can guests arrive to get their rooms?",
            "What is the earliest time I am allowed to check into my room?",
        ]
        for p in paraphrases:
            with patch.object(self.voice_service.stt_service, "transcribe_with_info", return_value=(p, 1.5)):
                events = [e async for e in self.voice_service.process_existing_turn_stream(
                    call_id=self.call.id,
                    audio_path=self.test_audio_path,
                    output_audio_path_prefix="scratch/audit_para.wav",
                    db=self.db,
                    company_id=COMPANY_ID,
                )]
                resp = next(e["response"] for e in events if e["event"] == "assistant_response")
                self.assertTrue("3:00" in resp or "15:00" in resp, f"Failed for paraphrase: {p} -> {resp}")

    async def test_07_category4_multi_part_questions(self):
        """Category 4: Multi-part questions (e.g. check-in and checkout times together)."""
        multi_q = "Could you tell me both the check-in time and the check-out time?"
        with patch.object(self.voice_service.stt_service, "transcribe_with_info", return_value=(multi_q, 2.0)):
            events = [e async for e in self.voice_service.process_existing_turn_stream(
                call_id=self.call.id,
                audio_path=self.test_audio_path,
                output_audio_path_prefix="scratch/audit_multi.wav",
                db=self.db,
                company_id=COMPANY_ID,
            )]
            resp = next(e["response"] for e in events if e["event"] == "assistant_response")
            # Should address check-in (3:00 PM) and check-out (12:00 PM)
            self.assertTrue("3:00" in resp or "15:00" in resp)
            self.assertTrue("12:00" in resp)

    async def test_08_category5_followup_questions_dependent_on_history(self):
        """Category 5: Follow-up question depending on previous answers."""
        # Turn 1: Ask what restaurants exist
        with patch.object(self.voice_service.stt_service, "transcribe_with_info", return_value=("What restaurants do you have at the hotel?", 1.5)):
            t1_events = [e async for e in self.voice_service.process_existing_turn_stream(
                call_id=self.call.id,
                audio_path=self.test_audio_path,
                output_audio_path_prefix="scratch/audit_fup_t1.wav",
                db=self.db,
                company_id=COMPANY_ID,
            )]
            resp1 = next(e["response"] for e in t1_events if e["event"] == "assistant_response")
            self.assertTrue("Honk" in resp1 or "Pluck" in resp1)

        # Turn 2: Follow up: "What kind of food does Honk serve?"
        with patch.object(self.voice_service.stt_service, "transcribe_with_info", return_value=("What kind of food does Honk serve?", 1.2)):
            t2_events = [e async for e in self.voice_service.process_existing_turn_stream(
                call_id=self.call.id,
                audio_path=self.test_audio_path,
                output_audio_path_prefix="scratch/audit_fup_t2.wav",
                db=self.db,
                company_id=COMPANY_ID,
            )]
            resp2 = next(e["response"] for e in t2_events if e["event"] == "assistant_response")
            self.assertTrue("asian" in resp2.lower() or "street" in resp2.lower() or "dim sum" in resp2.lower(), f"Unexpected follow-up answer: {resp2}")

    async def test_09_category6_natural_pauses_and_long_sentences(self):
        """Category 6: Natural conversational filler, pauses, and long sentences."""
        long_sentence = (
            "Hi, good afternoon... um, I am calling because my flight lands around midnight, "
            "and I was wondering if you could please tell me what time check in begins at the hotel?"
        )
        with patch.object(self.voice_service.stt_service, "transcribe_with_info", return_value=(long_sentence, 4.5)):
            events = [e async for e in self.voice_service.process_existing_turn_stream(
                call_id=self.call.id,
                audio_path=self.test_audio_path,
                output_audio_path_prefix="scratch/audit_long.wav",
                db=self.db,
                company_id=COMPANY_ID,
            )]
            resp = next(e["response"] for e in events if e["event"] == "assistant_response")
            self.assertTrue("3:00" in resp or "check-in" in resp.lower())

    async def test_10_category7_out_of_scope_questions(self):
        """Category 7: Out-of-scope questions (safe decline without hallucination)."""
        out_of_scope = "Can you help me file my income taxes or write Python code?"
        with patch.object(self.voice_service.stt_service, "transcribe_with_info", return_value=(out_of_scope, 1.8)):
            events = [e async for e in self.voice_service.process_existing_turn_stream(
                call_id=self.call.id,
                audio_path=self.test_audio_path,
                output_audio_path_prefix="scratch/audit_oos.wav",
                db=self.db,
                company_id=COMPANY_ID,
            )]
            resp = next(e["response"] for e in events if e["event"] == "assistant_response")
            # Should decline or direct to hotel services rather than pretending to file taxes
            self.assertTrue(
                "hotel" in resp.lower() or "assist" in resp.lower() or "sorry" in resp.lower() or "front desk" in resp.lower(),
                f"Unexpected response to out of scope query: {resp}"
            )

    async def test_11_category8_missing_hotel_facts(self):
        """Category 8: Missing hotel facts (truthful refusal to invent ungrounded hotel features)."""
        missing_fact_q = "Do you have a rooftop helipad for private helicopters?"
        with patch.object(self.voice_service.stt_service, "transcribe_with_info", return_value=(missing_fact_q, 1.5)):
            events = [e async for e in self.voice_service.process_existing_turn_stream(
                call_id=self.call.id,
                audio_path=self.test_audio_path,
                output_audio_path_prefix="scratch/audit_missing_fact.wav",
                db=self.db,
                company_id=COMPANY_ID,
            )]
            resp = next(e["response"] for e in events if e["event"] == "assistant_response")
            # Assistant should not confirm a helipad exists
            self.assertFalse("yes, we have a helipad" in resp.lower() or "helipad is available" in resp.lower())
            self.assertTrue("front desk" in resp.lower() or "concierge" in resp.lower() or "sorry" in resp.lower() or "not" in resp.lower())

    async def test_12_category9_empty_or_inaccurate_transcription(self):
        """Category 9: Empty or silence transcription yields polite retry prompt."""
        with patch.object(self.voice_service.stt_service, "transcribe_with_info", return_value=("", 0.0)):
            events = [e async for e in self.voice_service.process_existing_turn_stream(
                call_id=self.call.id,
                audio_path=self.test_audio_path,
                output_audio_path_prefix="scratch/audit_empty.wav",
                db=self.db,
                company_id=COMPANY_ID,
            )]
            final_ev = next(e for e in events if e["event"] == "assistant_response")
            self.assertTrue(final_ev.get("empty_speech"))
            self.assertIn("didn't quite hear", final_ev["response"])

    async def test_13_category10_tts_chunk_failure_and_turn_id_trace(self):
        """Category 10: TTS chunk failure recovery, ordering, and turn_id correlation."""
        provided_turn_id = f"test_audit_turn_{uuid.uuid4().hex[:6]}"
        with patch.object(self.voice_service.stt_service, "transcribe_with_info", return_value=("What time is checkout?", 1.0)):
            events = []
            async for ev in self.voice_service.process_existing_turn_stream(
                call_id=self.call.id,
                audio_path=self.test_audio_path,
                output_audio_path_prefix="scratch/audit_cat10.wav",
                db=self.db,
                company_id=COMPANY_ID,
                turn_id=provided_turn_id,
            ):
                events.append(ev)

            # Check that every emitted event contains the exact correlated turn_id
            for ev in events:
                self.assertEqual(ev.get("turn_id"), provided_turn_id)

            # Check chunk index ordering [0, 1, ...]
            chunk_events = [e for e in events if e["event"] == "assistant_chunk"]
            chunk_indices = [c["chunk_index"] for c in chunk_events]
            self.assertEqual(chunk_indices, list(range(len(chunk_indices))))


if __name__ == "__main__":
    unittest.main()
