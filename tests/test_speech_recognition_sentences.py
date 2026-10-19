import os
import sys
import json
import base64
from fastapi.testclient import TestClient

# Ensure root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from app.services.tts_service import TTSService
from app.utils.auth import create_access_token
from tests.test_live_voice_calling import ensure_test_user

client = TestClient(app)

TEST_SENTENCES = [
    "Hello, can you help me check my billing statement?",
    "My internet is running very slow today and disconnecting.",
    "I would like to speak with a customer support representative.",
    "Can you tell me what broadband plans you offer?",
    "Thank you so much, that solved my issue.",
    "My internet connection is not working."
]


def test_spoken_sentences_recognition_matches():
    """
    Test using several spoken sentences and verify that the recognized transcript
    matches the words spoken over the live voice WebSocket.
    """
    print("\n========================================================")
    print("TESTING SPEECH RECOGNITION ACROSS MULTIPLE SPOKEN SENTENCES")
    print("========================================================")
    ensure_test_user()
    token = create_access_token(data={"sub": "admin"})
    tts = TTSService()

    for idx, sentence in enumerate(TEST_SENTENCES, start=1):
        print(f"\n--- [Sentence #{idx}] Testing: \"{sentence}\" ---")
        temp_wav = os.path.join("recordings", f"test_verify_sentence_{idx}.wav")
        tts.synthesize(sentence, temp_wav)
        assert os.path.exists(temp_wav), f"Failed to synthesize audio for sentence #{idx}"

        try:
            with open(temp_wav, "rb") as f:
                audio_bytes = f.read()

            with client.websocket_connect(f"/api/ws/live-call?token={token}") as ws:
                # 1. Handshake
                h_msg = ws.receive_json()
                assert h_msg["event"] == "session_started"
                call_id = h_msg["call_id"]

                # 2. Initial Greeting
                g_msg = ws.receive_json()
                assert g_msg["event"] == "assistant_response"

                # 3. Send spoken audio turn
                b64_data = base64.b64encode(audio_bytes).decode("utf-8")
                ws.send_json({
                    "event": "user_audio",
                    "audio": b64_data
                })

                # 4. Receive processing state
                p_msg = ws.receive_json()
                assert p_msg["event"] == "processing", f"Expected processing event, got {p_msg}"
                print(f"  [PASS] Server acknowledged speech turn into processing state")

                # 5. Receive recognized transcript & response (supporting Task 22 chunk streaming)
                r_msg = ws.receive_json()
                while r_msg.get("event") == "assistant_chunk":
                    r_msg = ws.receive_json()

                assert r_msg["event"] == "assistant_response", f"Expected assistant_response, got {r_msg}"

                recognized_transcript = r_msg.get("transcript", "").strip()
                audio_duration = r_msg.get("audio_duration", 0.0)
                ai_response = r_msg.get("response", "").strip()

                print(f"  [Original Spoken]: \"{sentence}\"")
                print(f"  [Recognized]:      \"{recognized_transcript}\"")
                print(f"  [Audio Duration]:  {audio_duration:.2f}s")
                print(f"  [AI Response]:     \"{ai_response[:60]}...\"")

                # Verify accuracy: normalize punctuation and compare
                clean_spoken = sentence.lower().replace(",", "").replace(".", "").replace("?", "").strip()
                clean_rec = recognized_transcript.lower().replace(",", "").replace(".", "").replace("?", "").strip()

                # Calculate word overlap
                spoken_words = set(clean_spoken.split())
                rec_words = set(clean_rec.split())
                overlap = spoken_words.intersection(rec_words)
                accuracy = len(overlap) / float(len(spoken_words))

                print(f"  [Word Match Accuracy]: {accuracy * 100:.1f}% ({len(overlap)}/{len(spoken_words)} words)")
                assert accuracy >= 0.85, (
                    f"Recognition accuracy too low! Expected '{sentence}', got '{recognized_transcript}'"
                )
                print(f"  [PASS] Recognized transcript matched words spoken!")

                ws.send_json({"event": "end_call"})
        finally:
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except OSError:
                    pass

    print("\n========================================================")
    print("ALL SPOKEN SENTENCES RECOGNIZED ACCURATELY AND VERIFIED!")
    print("========================================================")


if __name__ == "__main__":
    test_spoken_sentences_recognition_matches()
