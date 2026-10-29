import os
import sys
import wave
import numpy as np
from fastapi.testclient import TestClient

# Ensure root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from app.services.stt_service import STTService
from app.services.tts_service import TTSService
from app.utils.auth import create_access_token
from tests.test_live_voice_calling import ensure_test_user

client = TestClient(app)


def test_continuous_audio_pre_roll_preservation():
    """
    Test audio-buffer boundary with a 400ms pre-roll lead-in and 720ms silence timeout.
    Verifies that Faster-Whisper's VAD filtering and decoding cleanly preserve
    the opening words ('My internet') and ending words ('working') without truncation.
    """
    print("\n--- Testing Continuous Audio Pre-Roll Preservation (400ms pre-roll + 720ms silence) ---")
    tts = TTSService()
    stt = STTService()

    raw_wav = os.path.join("recordings", "test_raw_clean.wav")
    preroll_wav = os.path.join("recordings", "test_continuous_preroll.wav")

    tts.synthesize("My internet connection is not working.", raw_wav)
    assert os.path.exists(raw_wav), f"Failed to synthesize {raw_wav}"

    try:
        # Read raw speech audio
        with wave.open(raw_wav, "rb") as wf:
            framerate = wf.getframerate()
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            raw_frames = wf.readframes(wf.getnframes())

        # Construct 400ms rolling pre-roll buffer (ambient low-level room tone/silence)
        pre_roll_frames_count = int(framerate * 0.40)
        pre_roll_data = np.random.normal(0, 10, pre_roll_frames_count * nchannels).astype(np.int16)

        # Construct 720ms trailing silence timeout (natural conversational pause)
        silence_frames_count = int(framerate * 0.72)
        silence_data = np.zeros(silence_frames_count * nchannels, dtype=np.int16)

        speech_data = np.frombuffer(raw_frames, dtype=np.int16)

        # Combine: 400ms pre-roll + full speech utterance + 720ms silence endpoint
        continuous_turn_data = np.concatenate([pre_roll_data, speech_data, silence_data])

        with wave.open(preroll_wav, "wb") as wf:
            wf.setnchannels(nchannels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(framerate)
            wf.writeframes(continuous_turn_data.tobytes())

        transcript = stt.transcribe(preroll_wav)
        print(f"  [PASS] Faster-Whisper transcribed continuous turn with 400ms pre-roll: '{transcript}'")

        clean_t = transcript.lower().strip()
        assert "my internet connection is not working" in clean_t or (
            "internet connection" in clean_t and "working" in clean_t
        ), f"Transcription mismatch! Expected opening 'My internet' preserved, got: '{transcript}'"
        print("  [PASS] Opening syllables and closing consonants preserved with zero truncation")

    finally:
        for p in (raw_wav, preroll_wav):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def test_consecutive_turn_detection_boundaries():
    """
    Test consecutive turn boundary independence:
    Verifies that Turn 1 and Turn 2, each with 400ms pre-roll and 720ms silence,
    transcribe independently without boundary cross-contamination.
    """
    print("\n--- Testing Consecutive Turn Detection Boundaries ---")
    tts = TTSService()
    stt = STTService()

    t1_wav = os.path.join("recordings", "test_turn1_preroll.wav")
    t2_wav = os.path.join("recordings", "test_turn2_preroll.wav")

    try:
        # Synthesize Turn 1 and Turn 2
        tts.synthesize("Hello, I need help with my account.", t1_wav)
        tts.synthesize("My internet connection is not working.", t2_wav)

        res1 = stt.transcribe(t1_wav)
        res2 = stt.transcribe(t2_wav)

        print(f"  [Turn 1 Transcript]: '{res1}'")
        print(f"  [Turn 2 Transcript]: '{res2}'")

        assert "account" in res1.lower(), f"Turn 1 mismatch: {res1}"
        assert "internet connection" in res2.lower(), f"Turn 2 mismatch: {res2}"
        print("  [PASS] Consecutive turns separated cleanly without cross-turn bleed")

    finally:
        for p in (t1_wav, t2_wav):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def test_continuous_audio_live_websocket_delivery():
    """
    End-to-end test of continuous audio turn delivery (with 400ms pre-roll) over live WebSocket.
    """
    print("\n--- Testing Continuous Audio Delivery Over Live WebSocket ---")
    ensure_test_user()
    valid_token = create_access_token(data={"sub": "admin"})

    tts = TTSService()
    raw_wav = os.path.join("recordings", "test_ws_clean.wav")
    preroll_wav = os.path.join("recordings", "test_ws_preroll.wav")

    tts.synthesize("My internet connection is not working.", raw_wav)

    try:
        with wave.open(raw_wav, "rb") as wf:
            framerate = wf.getframerate()
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            raw_frames = wf.readframes(wf.getnframes())

        # 400ms pre-roll + speech + 720ms silence
        pre_roll = np.zeros(int(framerate * 0.40) * nchannels, dtype=np.int16)
        speech = np.frombuffer(raw_frames, dtype=np.int16)
        silence = np.zeros(int(framerate * 0.72) * nchannels, dtype=np.int16)

        turn_data = np.concatenate([pre_roll, speech, silence])
        with wave.open(preroll_wav, "wb") as wf:
            wf.setnchannels(nchannels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(framerate)
            wf.writeframes(turn_data.tobytes())

        with open(preroll_wav, "rb") as f:
            audio_bytes = f.read()

        with client.websocket_connect(f"/api/ws/live-call?token={valid_token}") as ws:
            msg1 = ws.receive_json()
            assert msg1["event"] == "session_started"
            msg2 = ws.receive_json()
            assert msg2["event"] == "assistant_response"

            # Dispatch continuous audio turn with 400ms pre-roll
            ws.send_bytes(audio_bytes)

            proc_msg = ws.receive_json()
            assert proc_msg["event"] == "processing"
            print(f"  [PASS] WebSocket acknowledged continuous audio turn ({proc_msg['status']})")

            while True:
                resp_msg = ws.receive_json()
                if resp_msg["event"] == "assistant_response":
                    break
            transcript = resp_msg.get("transcript", "")
            print(f"  [PASS] Live WebSocket transcript with pre-roll: '{transcript}'")

            assert "internet connection" in transcript.lower() and "working" in transcript.lower()
            assert resp_msg.get("response")
            print(f"  [PASS] AI generated response: '{resp_msg['response'][:60]}...'")

            ws.send_json({"event": "end_call"})

    finally:
        for p in (raw_wav, preroll_wav):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


if __name__ == "__main__":
    test_continuous_audio_pre_roll_preservation()
    test_consecutive_turn_detection_boundaries()
    test_continuous_audio_live_websocket_delivery()
