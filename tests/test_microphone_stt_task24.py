import os
import sys
import wave
import unittest
import numpy as np

sys.path.insert(0, ".")

from app.utils.audio_utils import convert_to_16k_mono_wav
from app.services.knowledge_service import KnowledgeService
from app.services.stt_service import STTService, DEFAULT_CONVERSATION_PROMPT, TELECOM_DOMAIN_PROMPT


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Computes Word Error Rate (WER) between reference and hypothesis strings."""
    import re
    ref_words = re.findall(r"\b\w+\b", reference.lower())
    hyp_words = re.findall(r"\b\w+\b", hypothesis.lower())

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=int)
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return float(d[len(ref_words)][len(hyp_words)]) / float(len(ref_words))


class TestKnowledgeSTTPrompt(unittest.TestCase):
    """Tests company-grounded STT prompt isolation and neutral fallbacks."""

    def test_pullman_stt_prompt(self):
        ks = KnowledgeService()
        prompt = ks.get_stt_prompt("pullman_aerocity")
        self.assertIsNotNone(prompt)
        self.assertIn("pullman", prompt.lower())
        self.assertIn("pluck", prompt.lower())
        self.assertIn("honk", prompt.lower())
        self.assertNotIn("broadband", prompt.lower())
        self.assertNotIn("router", prompt.lower())

    def test_generic_stt_prompt_fallback(self):
        ks = KnowledgeService()
        prompt = ks.get_stt_prompt(None)
        self.assertEqual(prompt, DEFAULT_CONVERSATION_PROMPT)
        self.assertNotIn("router", prompt.lower())
        self.assertNotIn("broadband", prompt.lower())


class TestAudioConverter(unittest.TestCase):
    """Tests incoming audio format conversion to standard 16kHz mono PCM WAV."""

    def test_audio_resampling_properties(self):
        test_webm = "recordings/live_in_746_2e62f54d.webm"
        if not os.path.exists(test_webm):
            self.skipTest("Test recording fixture not available.")

        out_wav = convert_to_16k_mono_wav(test_webm)
        self.assertTrue(os.path.exists(out_wav))
        with wave.open(out_wav, "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getframerate(), 16000)
            self.assertEqual(wf.getsampwidth(), 2)
            duration = wf.getnframes() / wf.getframerate()
            self.assertGreaterEqual(duration, 2.5)
            self.assertLessEqual(duration, 4.0)


class TestPrerecordedCleanSpeechSTT(unittest.TestCase):
    """Tests STT accuracy and Word Error Rate on clean prerecorded speech."""

    def test_clean_speech_zero_wer(self):
        clean_path = "scratch/clean_test.wav"
        if not os.path.exists(clean_path):
            self.skipTest("clean_test.wav fixture not found.")

        stt = STTService()
        tx, dur = stt.transcribe_with_info(clean_path)
        wer = calculate_wer("Hello this is a test call", tx)
        self.assertEqual(wer, 0.0, f"Expected 0.0 WER, got {wer:.2f} (Recognized: '{tx}')")

    def test_order_speech_low_wer(self):
        order_path = "scratch/test_order.wav"
        if not os.path.exists(order_path):
            self.skipTest("test_order.wav fixture not found.")

        stt = STTService()
        tx, dur = stt.transcribe_with_info(order_path)
        wer = calculate_wer("Hello I want to place an order", tx)
        self.assertLessEqual(wer, 0.25, f"Expected low WER, got {wer:.2f} (Recognized: '{tx}')")


class TestObservedBrowserTurnsRecovery(unittest.TestCase):
    """
    Verifies that the observed garbled browser recordings from Task 24
    transcribe accurately and are NOT corrupted by the telecom router prompt.
    """

    def test_turn_746_2_no_telecom_hallucination(self):
        fpath = "recordings/live_in_746_2e62f54d.webm"
        if not os.path.exists(fpath):
            self.skipTest("Fixture not found.")

        ks = KnowledgeService()
        stt = STTService()
        prompt = ks.get_stt_prompt("pullman_aerocity")
        clean_wav = convert_to_16k_mono_wav(fpath)

        tx, dur = stt.transcribe_with_info(clean_wav, initial_prompt=prompt)
        tx_lower = tx.lower().strip()
        # Verify it transcribes the actual audio and does NOT hallucinate telecom 'let's go'
        self.assertNotIn("let's go", tx_lower)
        self.assertNotIn("router", tx_lower)
        self.assertNotIn("broadband", tx_lower)
        self.assertTrue(len(tx_lower) > 0)

    def test_turn_747_1_restaurant_family(self):
        fpath = "recordings/live_in_747_7d52ea7c.webm"
        if not os.path.exists(fpath):
            self.skipTest("Fixture not found.")

        ks = KnowledgeService()
        stt = STTService()
        prompt = ks.get_stt_prompt("pullman_aerocity")
        clean_wav = convert_to_16k_mono_wav(fpath)

        tx, dur = stt.transcribe_with_info(clean_wav, initial_prompt=prompt)
        tx_lower = tx.lower()
        self.assertIn("restaurant", tx_lower)
        self.assertIn("recommend", tx_lower)
        self.assertIn("family", tx_lower)
        # Must not hallucinate 'time ago' or telecom terms
        self.assertNotIn("time ago", tx_lower)
        self.assertNotIn("broadband", tx_lower)

    def test_turn_747_2_services(self):
        fpath = "recordings/live_in_747_0369f02a.webm"
        if not os.path.exists(fpath):
            self.skipTest("Fixture not found.")

        ks = KnowledgeService()
        stt = STTService()
        prompt = ks.get_stt_prompt("pullman_aerocity")
        clean_wav = convert_to_16k_mono_wav(fpath)

        tx, dur = stt.transcribe_with_info(clean_wav, initial_prompt=prompt)
        self.assertIn("services", tx.lower())


if __name__ == "__main__":
    unittest.main()
