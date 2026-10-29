import os
import unittest
from app.services.stt_service import STTService, normalize_acoustic_phonetic_errors
from app.services.knowledge_service import KnowledgeService
from app.services.tts_service import TTSService


class TestSTTAccuracyTask28(unittest.TestCase):
    """
    Master Task 28 Regression & Verification Tests:
    Validates live microphone audio recordings and synthetic audio pipelines
    for the target phrases observed during live testing:
    1. 'I will be arriving at the hotel at half past eleven in the evening.'
    2. 'Could you arrange airport transportation for four passengers with three large suitcases?'
    """

    @classmethod
    def setUpClass(cls):
        cls.stt = STTService()
        cls.ks = KnowledgeService()
        cls.tts = TTSService()
        cls.pullman_prompt = cls.ks.get_stt_prompt("pullman_aerocity")

    def test_real_mic_arrival_recording_recognition(self):
        """Verifies actual Call #1084 live recording decodes half past eleven instead of 'room 2'."""
        path = "recordings/live_in_1084_64ada58b_16k.wav"
        if not os.path.exists(path):
            self.skipTest(f"Recording {path} not found.")

        transcript, dur = self.stt.transcribe_with_info(path, initial_prompt=self.pullman_prompt)
        t_lower = transcript.lower()

        # Must not contain the previous distortion "room 2" or "room 2 or past"
        self.assertNotIn("room 2", t_lower)
        self.assertNotIn("room to", t_lower)

        # Must accurately capture arrival and time intent
        self.assertIn("arriving", t_lower)
        self.assertIn("half past eleven", t_lower)
        self.assertIn("evening", t_lower)

    def test_real_mic_transport_recording_recognition(self):
        """Verifies actual Call #1084 live recording decodes large suitcases instead of 'lounge suitcases'."""
        path = "recordings/live_in_1084_a96c3804_16k.wav"
        if not os.path.exists(path):
            self.skipTest(f"Recording {path} not found.")

        transcript, dur = self.stt.transcribe_with_info(path, initial_prompt=self.pullman_prompt)
        t_lower = transcript.lower()

        # Must not contain the acoustic artifact "lounge suitcases" or "long suitcase"
        self.assertNotIn("lounge suitcase", t_lower)

        # Must accurately recognize airport transportation, passenger count, and large suitcases
        self.assertIn("airport transportation", t_lower)
        self.assertTrue("4 passengers" in t_lower or "four passengers" in t_lower)
        self.assertIn("large suitcases", t_lower)

    def test_real_mic_reservation_name_recording_recognition(self):
        """Verifies actual Call #1084 live recording preserves reservation name mismatch details."""
        path = "recordings/live_in_1084_bbd6e7d9_16k.wav"
        if not os.path.exists(path):
            self.skipTest(f"Recording {path} not found.")

        transcript, dur = self.stt.transcribe_with_info(path, initial_prompt=self.pullman_prompt)
        t_lower = transcript.lower()

        self.assertIn("different name", t_lower)
        self.assertIn("person", t_lower)
        self.assertTrue("check-in" in t_lower or "checking in" in t_lower)

    def test_synthetic_target_phrase_arrival(self):
        """Verifies synthetic speech for arrival time phrase decodes cleanly."""
        text = "I will be arriving at the hotel at half past eleven in the evening."
        path = "recordings/test_task28_synth_arrival.wav"
        self.tts.synthesize(text, path)

        transcript, dur = self.stt.transcribe_with_info(path, initial_prompt=self.pullman_prompt)
        t_lower = transcript.lower()

        self.assertIn("arriving", t_lower)
        self.assertTrue("half past eleven" in t_lower or "half-past 11" in t_lower or "half past 11" in t_lower)
        self.assertIn("evening", t_lower)

    def test_synthetic_target_phrase_transportation(self):
        """Verifies synthetic speech for airport transportation phrase preserves quantities."""
        text = "Could you arrange airport transportation for four passengers with three large suitcases?"
        path = "recordings/test_task28_synth_transport.wav"
        self.tts.synthesize(text, path)

        transcript, dur = self.stt.transcribe_with_info(path, initial_prompt=self.pullman_prompt)
        t_lower = transcript.lower()

        self.assertIn("airport transportation", t_lower)
        self.assertTrue("four passengers" in t_lower or "4 passengers" in t_lower)
        self.assertTrue("three large suitcases" in t_lower or "3 large suitcases" in t_lower)

    def test_wifi_acoustic_normalizer_positive_and_negative(self):
        """Validates positive acoustic corrections and strict negative guards for Wi-Fi room phrases."""
        # Positive cases
        pos_cases = [
            ("Wi-Fi in the middle", "Wi-Fi in my room"),
            ("Can I use Wi-Fi in the middle?", "Can I use Wi-Fi in my room?"),
            ("Wi-Fi in mood room", "Wi-Fi in my room"),
            ("Is there Wi-Fi in mood room?", "Is there Wi-Fi in my room?"),
            ("Again, I'm not Wi-Fi in my room.", "Again, I need Wi-Fi in my room."),
            ("I'm not Wi-Fi in my room", "I need Wi-Fi in my room"),
            ("Again, high note Wi-Fi in my room.", "Again, I need Wi-Fi in my room."),
        ]
        for inp, expected in pos_cases:
            res = normalize_acoustic_phonetic_errors(inp)
            self.assertEqual(res, expected, f"Failed positive normalizer for '{inp}'")

        # Negative safeguards: must NOT rewrite legitimate customer phrases
        neg_cases = [
            "Wi-Fi in the middle of the table is disconnected.",
            "I'm not sure if the Wi-Fi is working.",
            "I'm not happy with the internet speed.",
            "I am not Wi-Fi certified.",
            "Meet me in the middle of the lobby.",
            "The mood in the room was calm and relaxing.",
            "Is breakfast served in the middle room?",
        ]
        for inp in neg_cases:
            res = normalize_acoustic_phonetic_errors(inp)
            self.assertEqual(res, inp, f"False positive normalizer mutation for '{inp}'")

    def test_call_1164_transcription_resolution(self):
        """Verifies actual Call #1164 live recording decodes 'I need Wi-Fi in my room' cleanly without 'Go,' or 'middle'."""
        path = "recordings/live_in_1164_e9d96224_16k.wav"
        if not os.path.exists(path):
            self.skipTest(f"Recording {path} not found.")

        transcript, dur = self.stt.transcribe_with_info(path, initial_prompt=self.pullman_prompt)
        t_lower = transcript.lower()

        # Must not contain opening phantom 'Go,'
        self.assertFalse(transcript.startswith("Go,"), f"Expected no spurious 'Go,', got: '{transcript}'")
        self.assertFalse(transcript.startswith("Go "), f"Expected no spurious 'Go ', got: '{transcript}'")
        # Must not contain 'the middle'
        self.assertNotIn("the middle", t_lower)
        # Must capture Wi-Fi and room intent
        self.assertIn("wi-fi", t_lower)
        self.assertTrue("in my room" in t_lower or "in room" in t_lower)

    def test_opening_words_preserved_on_legitimate_queries(self):
        """Verifies that legitimate question-starter words are never pruned by opening hallucination filter."""
        from app.services.response_service import ResponseService
        test_queries = [
            "Can I use Wi-Fi in my room?",
            "What time is check-in?",
            "Do you have a gym?",
            "I need Wi-Fi in my room.",
        ]
        for q in test_queries:
            wav_path = f"scratch/test_legit_{abs(hash(q)) % 10000}.wav"
            self.tts.synthesize(q, wav_path)
            try:
                transcribed, _ = self.stt.transcribe_with_info(wav_path, initial_prompt=self.pullman_prompt)
                first_word_orig = q.split()[0].rstrip(",.!?").lower()
                first_word_trans = transcribed.split()[0].rstrip(",.!?").lower()
                self.assertEqual(
                    first_word_trans,
                    first_word_orig,
                    f"Legitimate opening word '{first_word_orig}' was incorrectly pruned! Got '{transcribed}' for query '{q}'"
                )
            finally:
                if os.path.exists(wav_path):
                    try:
                        os.remove(wav_path)
                    except OSError:
                        pass

    def test_wifi_in_my_room_does_not_trigger_room_service(self):
        """Verifies that 'I need Wi-Fi in my room.' routes to Wi-Fi only and does NOT append room-service response."""
        import asyncio
        from app.services.response_service import ResponseService
        rs = ResponseService(llm_generator=None)
        kc = self.ks.get_relevant_knowledge("pullman_aerocity", "I need Wi-Fi in my room.")
        resp = asyncio.run(rs.generate_response(transcript="I need Wi-Fi in my room.", knowledge_context=kc))
        resp_lower = resp.lower()

        self.assertTrue("wi-fi" in resp_lower or "wifi" in resp_lower, f"Missing Wi-Fi fact in: '{resp}'")
        self.assertNotIn("room service", resp_lower)
        self.assertNotIn("in-room dining", resp_lower)
        self.assertNotIn("dining options", resp_lower)


if __name__ == "__main__":
    unittest.main()

