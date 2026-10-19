import unittest
import os
import asyncio
import wave
from unittest.mock import MagicMock, patch

from app.database.database import SessionLocal
from app.database.models import Customer, Call, Message
from app.services.response_service import ResponseService
from app.services.voice_response_service import VoiceResponseService
from app.services.conversation_service import ConversationService
from app.services.tts_service import TTSService

COMPANY_ID = "pullman_aerocity"

class TestSentenceStreamingTask22(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db = SessionLocal()
        self.response_service = ResponseService()
        self.voice_service = VoiceResponseService()
        self.conv_service = ConversationService()
        
        # Ensure a test customer and call session exist in SQLite / MySQL
        self.customer = self.db.query(Customer).filter(Customer.external_customer_id == "TASK22_TEST_USER").first()
        if not self.customer:
            self.customer = Customer(
                external_customer_id="TASK22_TEST_USER",
                name="Task22 Test Customer",
                phone="+919999900022"
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

        # Create a small valid WAV file for testing if needed
        self.test_audio_path = "scratch/task22_test_input.wav"
        os.makedirs("scratch", exist_ok=True)
        if not os.path.exists(self.test_audio_path):
            with wave.open(self.test_audio_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b"\x00\x00" * 16000) # 1 sec silence

    async def asyncTearDown(self):
        try:
            self.db.rollback()
            if hasattr(self, "call") and self.call:
                c = self.db.query(Call).filter(Call.id == self.call.id).first()
                if c:
                    self.db.delete(c)
                    self.db.commit()
        except Exception:
            pass
        finally:
            self.db.close()

    # 1. One-sentence responses
    async def test_01_one_sentence_response_streaming(self):
        query = "What time is check in?"
        sentences = []
        async for s in self.response_service.generate_response_stream_sentences(
            transcript=query,
            company_id=COMPANY_ID
        ):
            sentences.append(s)

        self.assertGreaterEqual(len(sentences), 1, "Expected at least one complete sentence.")
        self.assertIn("3:00 PM", sentences[0], "Expected check-in time 3:00 PM in first sentence.")

        # Test process_existing_turn_stream with mocked STT
        with patch.object(self.voice_service.stt_service, "transcribe_with_info", return_value=(query, 1.5)):
            events = []
            async for ev in self.voice_service.process_existing_turn_stream(
                call_id=self.call.id,
                audio_path=self.test_audio_path,
                output_audio_path_prefix=f"recordings/test_turn_{self.call.id}.wav",
                db=self.db,
                company_id=COMPANY_ID,
            ):
                events.append(ev)

            chunk_events = [e for e in events if e.get("event") == "assistant_chunk"]
            resp_events = [e for e in events if e.get("event") == "assistant_response"]

            self.assertEqual(len(resp_events), 1, "Must yield exactly one assistant_response event.")
            self.assertGreaterEqual(len(chunk_events), 1, "Must yield at least one assistant_chunk event.")
            self.assertEqual(chunk_events[0]["chunk_index"], 0, "First chunk index must be 0.")
            chunk_file = os.path.join("recordings", os.path.basename(chunk_events[0]["audio_url"]))
            self.assertTrue(os.path.exists(chunk_file), f"Audio file {chunk_file} must exist.")

    # 2. Multi-sentence responses
    async def test_02_multi_sentence_response_streaming(self):
        query = "What dining options do you have at the hotel?"
        sentences = []
        async for s in self.response_service.generate_response_stream_sentences(
            transcript=query,
            company_id=COMPANY_ID
        ):
            sentences.append(s)

        self.assertGreaterEqual(len(sentences), 1)

        # Test streaming turn yields sequential chunks
        with patch.object(self.voice_service.stt_service, "transcribe_with_info", return_value=(query, 2.0)):
            events = []
            async for ev in self.voice_service.process_existing_turn_stream(
                call_id=self.call.id,
                audio_path=self.test_audio_path,
                output_audio_path_prefix=f"recordings/test_multi_{self.call.id}.wav",
                db=self.db,
                company_id=COMPANY_ID,
            ):
                events.append(ev)

            chunk_events = [e for e in events if e.get("event") == "assistant_chunk"]
            for idx, c in enumerate(chunk_events):
                self.assertEqual(c["chunk_index"], idx, f"Chunk index {c['chunk_index']} must match order {idx}.")
                self.assertTrue(c["text"].strip())
                chunk_file = os.path.join("recordings", os.path.basename(c["audio_url"]))
                self.assertTrue(os.path.exists(chunk_file), f"Chunk audio file {chunk_file} must exist.")

    # 3. Sentence ordering and abbreviation handling
    def test_03_sentence_ordering_and_abbreviations(self):
        text = "The pool is open from 6:00 a.m. to 8:00 p.m. daily. Towels are provided at the counter. Enjoy your swim!"
        sentences = ResponseService.split_into_sentences(text)
        self.assertEqual(len(sentences), 3, f"Expected 3 sentences, got {len(sentences)}: {sentences}")
        self.assertEqual(sentences[0], "The pool is open from 6:00 a.m. to 8:00 p.m. daily.")
        self.assertEqual(sentences[1], "Towels are provided at the counter.")
        self.assertEqual(sentences[2], "Enjoy your swim!")

    # 4. Empty or malformed chunks
    def test_04_empty_or_malformed_chunks(self):
        text_with_blanks = "   ...   \n\n   Hello there!  \t  ...   "
        sentences = ResponseService.split_into_sentences(text_with_blanks)
        for s in sentences:
            self.assertTrue(s.strip(), "Sentence must not be whitespace or empty.")
            self.assertFalse(s.strip() in (".", "!", "?", "..."), "Sentence must not be standalone punctuation.")

    # 5. TTS failure after the first sentence
    async def test_05_tts_failure_after_first_sentence(self):
        query = "What restaurants are available at the hotel?"
        orig_synth = self.voice_service.tts_service.synthesize

        synth_call_count = 0
        def faulty_synthesize(text, out_path, language="en"):
            nonlocal synth_call_count
            synth_call_count += 1
            if synth_call_count > 1:
                raise RuntimeError("Simulated TTS synthesis engine failure!")
            return orig_synth(text, out_path, language=language)

        with patch.object(self.voice_service.stt_service, "transcribe_with_info", return_value=(query, 1.2)), \
             patch.object(self.voice_service.tts_service, "synthesize", side_effect=faulty_synthesize):
            events = []
            async for ev in self.voice_service.process_existing_turn_stream(
                call_id=self.call.id,
                audio_path=self.test_audio_path,
                output_audio_path_prefix=f"recordings/test_fault_{self.call.id}.wav",
                db=self.db,
                company_id=COMPANY_ID,
            ):
                events.append(ev)

            # Succeeded chunk 0 must be present
            chunk_events = [e for e in events if e.get("event") == "assistant_chunk"]
            self.assertGreaterEqual(len(chunk_events), 1, "First chunk must succeed.")

            # Final assistant_response must still be delivered despite subsequent failure
            resp_events = [e for e in events if e.get("event") == "assistant_response"]
            self.assertEqual(len(resp_events), 1, "assistant_response must still be yielded.")
            self.assertTrue(resp_events[0]["response"].strip())

    # 6. WebSocket disconnection during streaming
    async def test_06_websocket_disconnection_during_streaming(self):
        # Verify generator handles external cancellation cleanly (e.g. client disconnect)
        query = "Tell me about all the amenities."
        with patch.object(self.voice_service.stt_service, "transcribe_with_info", return_value=(query, 1.0)):
            stream_gen = self.voice_service.process_existing_turn_stream(
                call_id=self.call.id,
                audio_path=self.test_audio_path,
                output_audio_path_prefix=f"recordings/test_disconnect_{self.call.id}.wav",
                db=self.db,
                company_id=COMPANY_ID,
            )

            # Consume first event, then close generator to simulate client disconnect
            first_event = await stream_gen.__anext__()
            self.assertIn(first_event.get("event"), ("assistant_chunk", "assistant_response"))
            await stream_gen.aclose()

    # 7. Call ending during playback
    async def test_07_call_ending_during_playback(self):
        # End call
        self.conv_service.end_call(self.db, self.call.id)
        self.db.commit()

        # Attempting turn on ended call must raise ValueError
        with self.assertRaises(ValueError) as ctx:
            async for _ in self.voice_service.process_existing_turn_stream(
                call_id=self.call.id,
                audio_path=self.test_audio_path,
                output_audio_path_prefix=f"recordings/test_ended_{self.call.id}.wav",
                db=self.db,
                company_id=COMPANY_ID,
            ):
                pass
        self.assertIn("already ended", str(ctx.exception).lower())

    # 8. No cross-session audio leakage
    async def test_08_no_cross_session_audio_leakage(self):
        # Create second call
        call_b = self.conv_service.create_call(self.db, self.customer.id)
        call_b.status = "IN_PROGRESS"
        self.db.commit()

        try:
            with patch.object(self.voice_service.stt_service, "transcribe_with_info", return_value=("What time is checkout?", 1.0)):
                events_a = []
                async for ev in self.voice_service.process_existing_turn_stream(
                    call_id=self.call.id,
                    audio_path=self.test_audio_path,
                    output_audio_path_prefix=f"recordings/test_iso_a_{self.call.id}.wav",
                    db=self.db,
                    company_id=COMPANY_ID,
                ):
                    events_a.append(ev)

                events_b = []
                async for ev in self.voice_service.process_existing_turn_stream(
                    call_id=call_b.id,
                    audio_path=self.test_audio_path,
                    output_audio_path_prefix=f"recordings/test_iso_b_{call_b.id}.wav",
                    db=self.db,
                    company_id=COMPANY_ID,
                ):
                    events_b.append(ev)

                chunks_a = [e for e in events_a if e.get("event") == "assistant_chunk"]
                chunks_b = [e for e in events_b if e.get("event") == "assistant_chunk"]

                for c in chunks_a:
                    self.assertEqual(c["call_id"], self.call.id)
                    self.assertIn(f"test_iso_a_{self.call.id}", c["audio_url"])

                for c in chunks_b:
                    self.assertEqual(c["call_id"], call_b.id)
                    self.assertIn(f"test_iso_b_{call_b.id}", c["audio_url"])
        finally:
            c_b = self.db.query(Call).filter(Call.id == call_b.id).first()
            if c_b:
                self.db.delete(c_b)
                self.db.commit()

    # 9. Grounded responses and preserved clarification behavior
    async def test_09_grounded_responses_and_clarification(self):
        # Unclear query
        unclear_query = "What time of the chicken?"
        sentences = []
        async for s in self.response_service.generate_response_stream_sentences(
            transcript=unclear_query,
            company_id=COMPANY_ID
        ):
            sentences.append(s)
        full_text = " ".join(sentences)
        self.assertIn("clarify", full_text.lower(), "Unclear query must trigger clarification response.")

        # Unlisted facility query
        unlisted_query = "Do you have an ice skating rink?"
        sentences = []
        async for s in self.response_service.generate_response_stream_sentences(
            transcript=unlisted_query,
            company_id=COMPANY_ID
        ):
            sentences.append(s)
        full_text = " ".join(sentences)
        self.assertTrue(any(w in full_text.lower() for w in ("do not offer", "do not have", "not available")), "Must ground unlisted facility.")

    # 10. Performance instrumentation & latency comparison
    async def test_10_performance_latency_instrumentation(self):
        query = "What time is check in?"
        with patch.object(self.voice_service.stt_service, "transcribe_with_info", return_value=(query, 1.0)):
            events = []
            async for ev in self.voice_service.process_existing_turn_stream(
                call_id=self.call.id,
                audio_path=self.test_audio_path,
                output_audio_path_prefix=f"recordings/test_perf_{self.call.id}.wav",
                db=self.db,
                company_id=COMPANY_ID,
            ):
                events.append(ev)

            resp = next(e for e in events if e.get("event") == "assistant_response")
            timings = resp.get("timings", {})
            self.assertIn("stt_sec", timings)
            self.assertIn("time_to_first_text_sec", timings)
            self.assertIn("time_to_first_audio_sec", timings)
            self.assertIn("total_turn_sec", timings)

            ttfa = timings["time_to_first_audio_sec"]
            total = timings["total_turn_sec"]
            self.assertLessEqual(ttfa, total + 0.05, "Time to first audio must be <= total turn duration.")


if __name__ == "__main__":
    unittest.main()
