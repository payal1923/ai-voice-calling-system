import os
import sys
import unittest
import asyncio
from unittest.mock import patch

sys.path.insert(0, os.path.abspath("."))

from app.services.knowledge_service import KnowledgeService
from app.services.response_service import ResponseService
from app.services.stt_service import STTService, normalize_acoustic_phonetic_errors
from app.utils.audio_utils import trim_audio_silence


class TestPhase2QSpeechRecognitionRegression(unittest.TestCase):
    """
    Phase 2Q Master Regression Suite:
    Validates all 8 required test cases covering vocabulary guidance,
    phonetic error handling, silence trimming safety, and Ollama offline fallback.
    """

    @classmethod
    def setUpClass(cls):
        cls.ks = KnowledgeService()
        cls.rs = ResponseService()
        cls.stt = STTService()
        cls.company_id = "pullman_aerocity"
        cls.prompt = cls.ks.get_stt_prompt(cls.company_id)
        cls.sys_inst = cls.ks.get_system_instruction(cls.company_id)

    # 1. Fresh towels request
    def test_fresh_towels_request(self):
        """1. Fresh towels request returns housekeeping guidance without contradictions."""
        query = "I need fresh towels in my room."
        facts = self.ks.get_relevant_knowledge(self.company_id, query)
        resp = asyncio.run(self.rs.generate_response(
            transcript=query,
            knowledge_context=facts,
            system_instruction=self.sys_inst,
        ))
        resp_lower = resp.lower()
        self.assertIn("towels", resp_lower)
        self.assertTrue("housekeeping" in resp_lower or "front desk" in resp_lower)
        self.assertNotIn("service animals", resp_lower)
        self.assertNotIn("pet", resp_lower)

        # Acoustic slip normalization
        slip = "I need a fresh dog in my room."
        norm = normalize_acoustic_phonetic_errors(slip)
        self.assertEqual(norm.rstrip("."), "I need fresh towels in my room")

    # 2. Extra pillow request
    def test_extra_pillow_request(self):
        """2. Extra pillow request returns housekeeping guidance."""
        query = "I need an extra pillow."
        facts = self.ks.get_relevant_knowledge(self.company_id, query)
        resp = asyncio.run(self.rs.generate_response(
            transcript=query,
            knowledge_context=facts,
            system_instruction=self.sys_inst,
        ))
        resp_lower = resp.lower()
        self.assertIn("pillow", resp_lower)
        self.assertTrue("housekeeping" in resp_lower or "front desk" in resp_lower)

    # 3. Genuine dog / pet request
    def test_genuine_pet_request(self):
        """3. Genuine pet-related requests route to pet policy and NEVER to towels."""
        pet_queries = [
            "Are dogs allowed at the hotel?",
            "Can I bring my pet dog?",
            "Do you allow pets?",
            "What is your pet policy?",
        ]
        for q in pet_queries:
            # Must NOT be modified by phonetic normalizer
            self.assertEqual(normalize_acoustic_phonetic_errors(q), q)

            facts = self.ks.get_relevant_knowledge(self.company_id, q)
            resp = asyncio.run(self.rs.generate_response(
                transcript=q,
                knowledge_context=facts,
                system_instruction=self.sys_inst,
            ))
            resp_lower = resp.lower()
            self.assertTrue("service animals" in resp_lower or "pets" in resp_lower, f"Failed for '{q}': {resp}")
            self.assertNotIn("towels", resp_lower)
            self.assertNotIn("housekeeping", resp_lower)

    # 4. Bottled water request
    def test_bottled_water_request(self):
        """4. Bottled water requests return water assistance and NEVER convert to towels."""
        water_queries = [
            "Can you send fresh bottles of water to my room?",
            "I need water bottles in my room.",
            "Can I get bottled water?",
        ]
        for q in water_queries:
            # Must NOT convert to towels
            norm = normalize_acoustic_phonetic_errors(q)
            self.assertNotIn("towels", norm.lower())

            facts = self.ks.get_relevant_knowledge(self.company_id, q)
            resp = asyncio.run(self.rs.generate_response(
                transcript=q,
                knowledge_context=facts,
                system_instruction=self.sys_inst,
            ))
            resp_lower = resp.lower()
            self.assertIn("water", resp_lower)
            self.assertNotIn("towels", resp_lower)

    # 5. Wi-Fi response
    def test_wifi_response_preservation(self):
        """5. Wi-Fi responses remain complimentary without contradiction or apology."""
        wifi_queries = [
            "Do you have Wi-Fi?",
            "Do you have wifi?",
            "Is there Wi-Fi?",
            "Can I get Wi-Fi?",
        ]
        for q in wifi_queries:
            facts = self.ks.get_relevant_knowledge(self.company_id, q)
            resp = asyncio.run(self.rs.generate_response(
                transcript=q,
                knowledge_context=facts,
                system_instruction=self.sys_inst,
            ))
            resp_lower = resp.lower()
            self.assertIn("complimentary", resp_lower)
            self.assertTrue("wi-fi" in resp_lower or "wifi" in resp_lower)
            self.assertNotIn("not confirmed", resp_lower)
            self.assertNotIn("apologize", resp_lower)

    # 6. Check-in time request
    def test_checkin_time_request(self):
        """6. Check-in time inquiries return 3:00 PM (15:00)."""
        checkin_queries = [
            "What is check-in time?",
            "What time is check-in?",
            "What is check-in timing?",
        ]
        for q in checkin_queries:
            facts = self.ks.get_relevant_knowledge(self.company_id, q)
            resp = asyncio.run(self.rs.generate_response(
                transcript=q,
                knowledge_context=facts,
                system_instruction=self.sys_inst,
            ))
            resp_lower = resp.lower()
            self.assertIn("3:00 pm", resp_lower)
            self.assertIn("check-in", resp_lower)

        # Verify 'avoid eat-check-in' is NOT rewritten into 'what is check-in time'
        garble = "Avoid eat-check-in time."
        norm_garble = normalize_acoustic_phonetic_errors(garble)
        self.assertNotIn("what is check-in time", norm_garble.lower())

    # 7. Quiet speech and silence trimming
    def test_quiet_speech_and_silence_trimming(self):
        """7. Silence trimming safely handles quiet speech and audio with trailing silence."""
        import tempfile
        import wave
        import numpy as np

        sr = 16000
        # A. Quiet speech (RMS ~280) must NOT be truncated or corrupted
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_quiet:
            quiet_path = f_quiet.name
        try:
            # 500ms lead + 1.0s low amplitude tone (amp=400, RMS~282) + 500ms tail
            t = np.linspace(0, 1.0, sr, endpoint=False)
            tone = (np.sin(2 * np.pi * 300 * t) * 400).astype(np.int16)
            lead = np.zeros(int(0.5 * sr), dtype=np.int16)
            tail = np.zeros(int(0.5 * sr), dtype=np.int16)
            samples = np.concatenate([lead, tone, tail])

            with wave.open(quiet_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(samples.tobytes())

            res = trim_audio_silence(quiet_path)
            self.assertTrue(os.path.exists(res))
            with wave.open(res, "rb") as wf:
                dur = wf.getnframes() / wf.getframerate()
            # Duration must retain tone (> 0.8s)
            self.assertGreater(dur, 0.8)
        finally:
            if os.path.exists(quiet_path):
                os.remove(quiet_path)

        # B. Trailing silence trimming on 4.5s audio (1.5s lead + 1.0s tone + 2.0s tail)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_silence:
            silence_path = f_silence.name
        try:
            t = np.linspace(0, 1.0, sr, endpoint=False)
            tone = (np.sin(2 * np.pi * 440 * t) * 5000).astype(np.int16)
            lead = np.zeros(int(1.5 * sr), dtype=np.int16)
            tail = np.zeros(int(2.0 * sr), dtype=np.int16)
            samples = np.concatenate([lead, tone, tail])

            with wave.open(silence_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(samples.tobytes())

            trimmed = trim_audio_silence(silence_path)
            self.assertTrue(os.path.exists(trimmed))
            with wave.open(trimmed, "rb") as wf:
                trim_dur = wf.getnframes() / wf.getframerate()
            # Trimmed duration should be around 1.75s (350ms lead + 1.0s speech + 400ms pad)
            self.assertLess(trim_dur, 3.0)
            self.assertGreater(trim_dur, 1.3)
        finally:
            if os.path.exists(silence_path):
                os.remove(silence_path)
            if trimmed != silence_path and os.path.exists(trimmed):
                os.remove(trimmed)

    # 8. Ollama unavailable fallback
    def test_ollama_unavailable_fallback(self):
        """8. Deterministic paths work when Ollama is down, and unknown queries yield graceful apology."""
        # A. Deterministic Wi-Fi, check-in, towels, pets work even if Ollama connection fails
        with patch("app.services.llm_service.generate_response_stream", side_effect=ConnectionRefusedError("Ollama offline")):
            # Wi-Fi works
            resp_wifi = asyncio.run(self.rs.generate_response("Do you have Wi-Fi?", knowledge_context="Wi-Fi is complimentary.", system_instruction=self.sys_inst))
            self.assertIn("complimentary", resp_wifi.lower())

            # Check-in works
            resp_ci = asyncio.run(self.rs.generate_response("What is check-in time?", knowledge_context="Check-in is at 3:00 PM.", system_instruction=self.sys_inst))
            self.assertIn("3:00 pm", resp_ci.lower())

            # Towels works
            resp_towel = asyncio.run(self.rs.generate_response("I need fresh towels in my room.", knowledge_context="Housekeeping available.", system_instruction=self.sys_inst))
            self.assertIn("towels", resp_towel.lower())

            # Pet works
            resp_pet = asyncio.run(self.rs.generate_response("Can I bring my dog?", knowledge_context="Service animals permitted.", system_instruction=self.sys_inst))
            self.assertIn("service animals", resp_pet.lower())

            # B. Unknown request yields graceful fallback
            async def _stream_unknown():
                sentences = []
                async for s in self.rs.generate_response_stream_sentences(
                    transcript="Do you know what it is like?",
                    conversation_context="",
                    system_instruction=self.sys_inst,
                    knowledge_context="",
                ):
                    sentences.append(s)
                return " ".join(sentences)

            fallback_resp = asyncio.run(_stream_unknown())
            self.assertIn("unable to access that detail", fallback_resp.lower())
            self.assertIn("front desk", fallback_resp.lower())

    # 9. Replay actual saved recordings
    def test_replay_actual_saved_recordings(self):
        """9. Replay real microphone recordings from Call #1181 and verify recognition."""
        rec_dir = "recordings"

        # Replay Attempt 1: live_in_1181_14b86579_16k.wav (was 'fresh dog')
        p1 = os.path.join(rec_dir, "live_in_1181_14b86579_16k.wav")
        if os.path.exists(p1):
            transcript, dur = self.stt.transcribe_with_info(p1, initial_prompt=self.prompt)
            print(f"\n[REPLAY 1181 ATTEMPT 1] Transcript: '{transcript}'")
            # Must NOT contain 'fresh dog'
            self.assertNotIn("fresh dog", transcript.lower())
            self.assertIn("towels", transcript.lower())

        # Replay Attempt 4: live_in_1181_4b2ccec6_16k.wav (fresh towels)
        p4 = os.path.join(rec_dir, "live_in_1181_4b2ccec6_16k.wav")
        if os.path.exists(p4):
            transcript, dur = self.stt.transcribe_with_info(p4, initial_prompt=self.prompt)
            print(f"[REPLAY 1181 ATTEMPT 4] Transcript: '{transcript}'")
            self.assertIn("fresh towels", transcript.lower())

        # Replay Wi-Fi: live_in_1181_3781c743_16k.wav
        pw = os.path.join(rec_dir, "live_in_1181_3781c743_16k.wav")
        if os.path.exists(pw):
            transcript, dur = self.stt.transcribe_with_info(pw, initial_prompt=self.prompt)
            print(f"[REPLAY 1181 WI-FI] Transcript: '{transcript}'")
            self.assertTrue("wi-fi" in transcript.lower() or "wifi" in transcript.lower())

        # Replay Check-in: live_in_1181_3a818c1c_16k.wav
        pci = os.path.join(rec_dir, "live_in_1181_3a818c1c_16k.wav")
        if os.path.exists(pci):
            transcript, dur = self.stt.transcribe_with_info(pci, initial_prompt=self.prompt)
            print(f"[REPLAY 1181 CHECK-IN] Transcript: '{transcript}'")
            self.assertIn("check-in", transcript.lower())


if __name__ == "__main__":
    unittest.main()
