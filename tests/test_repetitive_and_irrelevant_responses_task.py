import os
import sys
import json
import base64
import asyncio
import unittest
from datetime import timedelta
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("."))

from main import app
from app.services.response_service import ResponseService
from app.services.knowledge_service import KnowledgeService
from app.services.tts_service import TTSService
from app.services.conversation_context_service import ConversationContextService
from app.database.database import SessionLocal
from app.database.models import Call, Message, SystemUser
from app.utils.auth import create_access_token, hash_password
from app.api.call_routes import ENGLISH_GREETING_MESSAGE


class TestRepetitiveAndIrrelevantResponses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rs = ResponseService()
        cls.ks = KnowledgeService()
        cls.tts = TTSService()
        cls.context_service = ConversationContextService()
        cls.company_id = "pullman_aerocity"
        cls.sys_inst = cls.ks.get_system_instruction(cls.company_id)
        cls.client = TestClient(app)

        # Ensure admin user exists for authenticated WebSocket tests
        with SessionLocal() as db:
            user = db.query(SystemUser).filter(SystemUser.username == "admin").first()
            if not user:
                user = SystemUser(
                    username="admin",
                    password_hash=hash_password("admin123"),
                    role="admin",
                )
                db.add(user)
                db.commit()

    def test_call_907_repetition_fix(self):
        """
        Reproduce Call #907 scenario:
        Turn 1: "I want to book a room for three persons."
        Turn 2: "What type of service do you offer?"
        Verify that Turn 2 does NOT repeat room rates or booking pricing.
        """
        turn1_q = "I want to book a room for three persons."
        k1 = self.ks.get_relevant_knowledge(self.company_id, turn1_q)
        ans1 = asyncio.run(self.rs.generate_response(
            transcript=turn1_q,
            conversation_context="",
            system_instruction=self.sys_inst,
            knowledge_context=k1,
        ))
        self.assertTrue(
            any(w in ans1.lower() for w in ["rates", "vary", "availability", "spacious rooms", "suites", "multiple guests", "families"]),
            f"Turn 1 should provide room booking guidance, got: {ans1}"
        )

        # Context after Turn 1
        history = f"Customer: {turn1_q}\nAssistant: {ans1}\n"

        # Turn 2: "What type of service do you offer?"
        turn2_q = "What type of service do you offer?"
        k2 = self.ks.get_relevant_knowledge(self.company_id, turn2_q, conversation_context=history)
        ans2 = asyncio.run(self.rs.generate_response(
            transcript=turn2_q,
            conversation_context=history,
            system_instruction=self.sys_inst,
            knowledge_context=k2,
        ))

        # Crucial assertions: Turn 2 MUST NOT repeat room rates/availability
        self.assertNotIn("room rates and availability vary in real time", ans2.lower())
        self.assertNotIn("vary in real time based on stay dates and room type", ans2.lower())
        self.assertTrue(
            any(w in ans2.lower() for w in ["services", "accommodations", "dining", "spa", "pool", "fitness", "luggage", "front desk", "amenities"]),
            f"Turn 2 should address service inquiry appropriately without repeating room rates, got: {ans2}"
        )

    def test_requirement_9_consecutive_questions_with_followup(self):
        """
        Executes the exact list of required questions in consecutive order,
        verifying each answer is relevant, factually grounded, and not repetitive:
        1. "What time is check-in?"
        2. "Do you have a swimming pool?"
        3. "What dining options do you offer?"
        4. "Can I store my luggage?"
        5. "What type of services do you offer?"
        6. Follow-up: "What kind of food does Honk serve?" (dependent on dining)
        7. Follow-up: "Is it heated?" (dependent on pool)
        """
        conversation_turns = [
            ("What time is check-in?", ["3:00", "15:00"]),
            ("Do you have a swimming pool?", ["pool", "swimming", "heated"]),
            ("Is it heated?", ["heated", "outdoor", "pool"]),
            ("What dining options do you offer?", ["pluck", "honk", "café pluck", "chidya ghar"]),
            ("What kind of food does Honk serve?", ["asian", "pan-asian", "dim sum", "sushi", "street food"]),
            ("Can I store my luggage?", ["luggage", "storage", "concierge", "bags"]),
            ("What type of services do you offer?", ["accommodations", "dining", "fitness", "spa", "pool", "luggage", "services"]),
        ]

        history = ""
        prior_answers = []

        for q, expected_keywords in conversation_turns:
            k = self.ks.get_relevant_knowledge(self.company_id, q, conversation_context=history)
            ans = asyncio.run(self.rs.generate_response(
                transcript=q,
                conversation_context=history,
                system_instruction=self.sys_inst,
                knowledge_context=k,
            ))

            self.assertIsNotNone(ans)
            self.assertGreater(len(ans), 10, f"Answer too short for '{q}': {ans}")

            # Verify keywords
            matches = [kw for kw in expected_keywords if kw in ans.lower()]
            self.assertTrue(
                len(matches) > 0,
                f"Question: '{q}'\nAnswer: '{ans}'\nExpected at least one of: {expected_keywords}"
            )

            # Verify no direct copy of any prior answer
            for idx, prior_ans in enumerate(prior_answers):
                self.assertNotEqual(
                    ans.strip().lower(), prior_ans.strip().lower(),
                    f"Turn for '{q}' duplicated prior answer from turn {idx}: '{prior_ans}'"
                )

            prior_answers.append(ans)
            history += f"Customer: {q}\nAssistant: {ans}\n"

    def test_cumulative_conversation_sequence_streaming(self):
        """
        Runs questions sequentially in a multi-turn conversation in streaming mode,
        testing that chunks do not have duplicate 'Yes, ' and Piper TTS can synthesize every sentence.
        """
        questions = [
            "What time is check-in?",
            "Do you have a swimming pool?",
            "What dining options do you offer?",
            "Can I store my luggage?",
            "What type of services do you offer?",
            "What kind of food does Honk serve?",
        ]

        history = ""
        out_audio_dir = "recordings"
        os.makedirs(out_audio_dir, exist_ok=True)

        for idx, q in enumerate(questions, 1):
            k = self.ks.get_relevant_knowledge(self.company_id, q, conversation_context=history)
            chunks = []

            async def collect():
                async for chunk in self.rs.generate_response_stream_sentences(
                    transcript=q,
                    conversation_context=history,
                    system_instruction=self.sys_inst,
                    knowledge_context=k,
                ):
                    chunks.append(chunk)

            asyncio.run(collect())
            self.assertTrue(len(chunks) >= 1, f"Should yield at least 1 chunk for: '{q}'")

            # Check that "Yes, " is not duplicated across chunks
            yes_chunks = [c for c in chunks if c.lower().startswith("yes")]
            self.assertLessEqual(
                len(yes_chunks), 1,
                f"Question '{q}' has multiple chunks starting with 'Yes': {chunks}"
            )

            # Check that sentences within this turn are unique
            seen_sentences = set()
            for c in chunks:
                norm_c = c.strip().lower()
                self.assertNotIn(
                    norm_c, seen_sentences,
                    f"Question '{q}' yielded duplicate chunk '{c}' in streaming chunks: {chunks}"
                )
                seen_sentences.add(norm_c)

            full_ans = " ".join(chunks)

            # Test TTS synthesis for chunk 0
            chunk0_path = os.path.join(out_audio_dir, f"test_task_chunk_{idx}_0.wav")
            synth_path = self.tts.synthesize(chunks[0], chunk0_path, language="en")
            self.assertTrue(os.path.exists(synth_path), f"TTS output not created for chunk 0: {chunks[0]}")
            self.assertGreater(os.path.getsize(synth_path), 1000, "Audio file too small")

            # Update history for next turn
            history += f"Customer: {q}\nAssistant: {full_ans}\n"

    def test_context_service_exclude_in_flight_question(self):
        """
        Verifies that ConversationContextService.build_context excludes the in-flight
        customer question when passed via exclude_transcript.
        """
        call = Call(id=999)
        msg1 = Message(id=1, call_id=999, speaker="assistant", text="Hello! How may I assist you?")
        msg2 = Message(id=2, call_id=999, speaker="customer", text="I want to book a room for three persons.")
        msg3 = Message(id=3, call_id=999, speaker="assistant", text="We offer spacious rooms and suites.")
        msg4 = Message(id=4, call_id=999, speaker="customer", text="What type of service do you offer?")
        call.messages = [msg1, msg2, msg3, msg4]

        # Context with exclusion
        ctx = self.context_service.build_context([call], exclude_transcript="What type of service do you offer?")
        self.assertNotIn("What type of service do you offer?", ctx)
        self.assertIn("I want to book a room for three persons.", ctx)
        self.assertIn("We offer spacious rooms and suites.", ctx)

    def test_live_websocket_multi_turn_voice_call(self):
        """
        Live WebSocket multi-turn test over /api/ws/live-call verifying:
        1. Session start and initial greeting event.
        2. Audio synthesized and sent for Turn 1: 'What time is check-in?'
        3. Server responds with relevant check-in answer and valid Piper audio URL.
        4. Audio synthesized and sent for Turn 2: 'Do you have a swimming pool?'
        5. Server responds with relevant pool answer, NOT repeating check-in.
        6. After each turn, session remains active ('IN_PROGRESS') so the assistant returns to listening.
        """
        valid_token = create_access_token(data={"sub": "admin"})
        test_dir = "recordings"
        os.makedirs(test_dir, exist_ok=True)

        # Synthesize real voice WAV files for the questions
        wav1_path = os.path.join(test_dir, "ws_q1_checkin.wav")
        self.tts.synthesize("What time is check-in?", wav1_path)
        with open(wav1_path, "rb") as f:
            b64_audio1 = base64.b64encode(f.read()).decode("utf-8")

        wav2_path = os.path.join(test_dir, "ws_q2_pool.wav")
        self.tts.synthesize("Do you have a swimming pool?", wav2_path)
        with open(wav2_path, "rb") as f:
            b64_audio2 = base64.b64encode(f.read()).decode("utf-8")

        with self.client.websocket_connect(f"/api/ws/live-call?token={valid_token}") as ws:
            # 1. Handshake
            s_msg = ws.receive_json()
            self.assertEqual(s_msg["event"], "session_started")
            call_id = s_msg["call_id"]

            # 2. Greeting
            g_msg = ws.receive_json()
            self.assertEqual(g_msg["event"], "assistant_response")
            self.assertTrue(
                "pullman" in g_msg["response"].lower() or "how" in g_msg["response"].lower() or g_msg["response"] == ENGLISH_GREETING_MESSAGE,
                f"Unexpected greeting: {g_msg['response']}"
            )

            # 3. Turn 1: Check-in inquiry
            ws.send_json({"event": "user_audio", "audio": b64_audio1})

            # Consume events until final assistant_response
            turn1_chunks = []
            turn1_final = None
            while True:
                evt = ws.receive_json()
                if evt["event"] == "processing":
                    continue
                elif evt["event"] == "assistant_chunk":
                    turn1_chunks.append(evt)
                elif evt["event"] == "assistant_response":
                    turn1_final = evt
                    break

            self.assertIsNotNone(turn1_final)
            self.assertTrue(
                any(t in turn1_final["transcript"].lower() for t in ["check", "time", "taking", "check-in"]),
                f"STT Transcript should reflect check-in, got: {turn1_final['transcript']}"
            )
            self.assertTrue(
                any(t in turn1_final["response"].lower() for t in ["3:00", "15:00", "check-in"]),
                f"AI Response should answer check-in time, got: {turn1_final['response']}"
            )
            self.assertEqual(turn1_final.get("call_status"), "IN_PROGRESS", "Assistant must return to listening (IN_PROGRESS)")

            # 4. Turn 2: Pool inquiry
            ws.send_json({"event": "user_audio", "audio": b64_audio2})

            turn2_chunks = []
            turn2_final = None
            while True:
                evt = ws.receive_json()
                if evt["event"] == "processing":
                    continue
                elif evt["event"] == "assistant_chunk":
                    turn2_chunks.append(evt)
                elif evt["event"] == "assistant_response":
                    turn2_final = evt
                    break

            self.assertIsNotNone(turn2_final)
            self.assertTrue(
                any(t in turn2_final["transcript"].lower() for t in ["pool", "swimming"]),
                f"STT Transcript should reflect pool inquiry, got: {turn2_final['transcript']}"
            )
            self.assertTrue(
                any(t in turn2_final["response"].lower() for t in ["pool", "swimming", "heated"]),
                f"AI Response should answer pool inquiry, got: {turn2_final['response']}"
            )
            # Crucial check: Turn 2 must NOT repeat Turn 1's check-in response
            self.assertNotIn("3:00 pm", turn2_final["response"].lower())
            self.assertEqual(turn2_final.get("call_status"), "IN_PROGRESS", "Assistant must return to listening after speaking")

            # 5. Cleanly end call
            ws.send_json({"event": "end_call"})


if __name__ == "__main__":
    unittest.main()
