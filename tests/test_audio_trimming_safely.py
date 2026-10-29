import os
import sys
import wave
import tempfile
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from app.utils.audio_utils import trim_audio_silence
from app.services.stt_service import STTService
from app.services.knowledge_service import KnowledgeService

def test_short_wav_safety():
    """Test 1: Very short WAV input does not crash and returns original path."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_name = f.name
    try:
        with wave.open(tmp_name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(np.zeros(1600, dtype=np.int16).tobytes())  # 100ms
        res = trim_audio_silence(tmp_name)
        assert res == tmp_name, f"Expected {tmp_name}, got {res}"
        print("[PASS] Test 1: Very short WAV input handled safely without crash")
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

def test_silent_wav_safety():
    """Test 2: Silent or near-silent WAV is handled safely without crashing or corrupting."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_name = f.name
    try:
        with wave.open(tmp_name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            # 1.5 seconds of near-silent background noise (RMS ~20)
            noise = np.random.randint(-30, 30, 24000, dtype=np.int16)
            wf.writeframes(noise.tobytes())
        res = trim_audio_silence(tmp_name)
        assert res == tmp_name, f"Expected {tmp_name}, got {res}"
        print("[PASS] Test 2: Silent or near-silent WAV handled safely")
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

def test_leading_and_trailing_silence_removal():
    """Test 3: Leading silence is removed without clipping initial word, trailing silence is removed while retaining final sound."""
    sr = 16000
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
        in_path = f_in.name
        out_path = f_out.name

    try:
        # 1.5s leading silence + 1.0s tone (amp=5000) + 2.0s trailing silence (Total: 4.5s)
        t = np.linspace(0, 1.0, sr, endpoint=False)
        tone = (np.sin(2 * np.pi * 440 * t) * 5000).astype(np.int16)
        lead = np.zeros(int(1.5 * sr), dtype=np.int16)
        trail = np.zeros(int(2.0 * sr), dtype=np.int16)
        full = np.concatenate([lead, tone, trail])

        with wave.open(in_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(full.tobytes())

        res_path = trim_audio_silence(in_path, out_path)
        assert res_path == out_path
        with wave.open(out_path, "rb") as wf:
            dur = wf.getnframes() / sr
            frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16)

        # Expected trimmed duration: 1.0s tone + ~350ms lead + ~400ms trail = ~1.75s
        assert 1.6 <= dur <= 1.9, f"Unexpected trimmed duration: {dur:.2f}s (original 4.50s)"
        # Verify speech signal exists at onset and ending within padded boundaries
        # Tone begins around index 0.35 * sr (padded lead)
        onset_region = samples[int(0.20 * sr) : int(0.50 * sr)]
        assert np.max(np.abs(onset_region)) > 4000, "Initial tone signal was clipped at onset"
        # Tone ends around index (0.35 + 1.0) * sr = 1.35 * sr
        end_region = samples[int(1.20 * sr) : int(1.45 * sr)]
        assert np.max(np.abs(end_region)) > 4000, "Final tone signal was clipped at ending"

        print(f"[PASS] Test 3: Leading & trailing silence removed correctly (4.50s -> {dur:.2f}s) without clipping onset/ending")
    finally:
        for p in (in_path, out_path):
            if os.path.exists(p):
                os.remove(p)

def test_long_internal_pause_preserved():
    """Test 4: A long internal pause (>500ms, e.g. 2.0s) does NOT truncate the second speech segment."""
    sr = 16000
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
        in_path = f_in.name
        out_path = f_out.name

    try:
        # 0.5s lead + 0.5s tone1 + 2.0s internal pause + 0.5s tone2 + 1.5s trail (Total: 5.0s)
        tone1 = (np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)) * 5000).astype(np.int16)
        tone2 = (np.sin(2 * np.pi * 880 * np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)) * 5000).astype(np.int16)
        lead = np.zeros(int(0.5 * sr), dtype=np.int16)
        pause = np.zeros(int(2.0 * sr), dtype=np.int16)
        trail = np.zeros(int(1.5 * sr), dtype=np.int16)
        full = np.concatenate([lead, tone1, pause, tone2, trail])

        with wave.open(in_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(full.tobytes())

        res_path = trim_audio_silence(in_path, out_path)
        assert res_path == out_path
        with wave.open(out_path, "rb") as wf:
            dur = wf.getnframes() / sr
            frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16)

        # Expected trimmed duration: ~0.35s + 0.5s + 2.0s + 0.5s + ~0.40s = ~3.75s
        assert dur >= 3.6, f"Internal pause was truncated! dur={dur:.2f}s"
        # Confirm tone2 is intact near the end of the trimmed file
        tail_samples = samples[-int(0.8 * sr):]
        assert np.max(np.abs(tail_samples)) > 4000, "Second speech segment was truncated by internal pause"
        print(f"[PASS] Test 4: Long internal pause (2.0s) preserved without truncation (5.00s -> {dur:.2f}s)")
    finally:
        for p in (in_path, out_path):
            if os.path.exists(p):
                os.remove(p)

def test_five_room_recording_e2e():
    """Test 5 & 6: Five-room recording retains the final phrase after trimming and matches original transcript."""
    five_path = "recordings/live_in_930_99e767d2_16k.wav"
    if not os.path.exists(five_path):
        print(f"[SKIP] Five-room recording fixture not found: {five_path}")
        return

    # 1. Direct trimming check
    trimmed_path = trim_audio_silence(five_path, "scratch/five_room_test_trimmed.wav")
    try:
        with wave.open(five_path, "rb") as wf:
            orig_dur = wf.getnframes() / wf.getframerate()
        with wave.open(trimmed_path, "rb") as wf:
            trim_dur = wf.getnframes() / wf.getframerate()

        print(f"Five-room duration: Original = {orig_dur:.2f}s | Trimmed = {trim_dur:.2f}s")
        # In the buggy code, duration was 3.22s. It must be > 10s to preserve "for my family"
        assert trim_dur >= 10.0, f"Expected trimmed duration >= 10.0s, got {trim_dur:.2f}s (was truncated!)"

        # 2. Transcription via STTService (which internally calls trim_audio_silence)
        stt = STTService()
        prompt = KnowledgeService().get_stt_prompt("pullman_aerocity")
        
        # Test A: Call transcribe_with_info on the original recording (STTService internally trims it)
        transcript, dur_out = stt.transcribe_with_info(five_path, initial_prompt=prompt)
        print(f"STTService transcript on five-room recording: \"{transcript}\" (Duration: {dur_out:.2f}s)")

        expected = "I want five rooms for my family."
        assert expected.lower().rstrip(".") in transcript.lower(), (
            f"Transcript mismatch! Expected phrase '{expected}' in '{transcript}'"
        )
        print("[PASS] Test 5: Five-room recording retains final phrase after trimming")
        print(f"[PASS] Test 6: Trimmed transcript matches expected ground truth: \"{transcript}\"")
    finally:
        if os.path.exists("scratch/five_room_test_trimmed.wav"):
            os.remove("scratch/five_room_test_trimmed.wav")

def main():
    print("======================================================================")
    print("RUNNING FOCUSED AUDIO TRIMMING SAFETY TESTS")
    print("======================================================================")
    test_short_wav_safety()
    test_silent_wav_safety()
    test_leading_and_trailing_silence_removal()
    test_long_internal_pause_preserved()
    test_five_room_recording_e2e()
    print("======================================================================")
    print("ALL FOCUSED AUDIO TRIMMING TESTS PASSED SUCCESSFULLY!")
    print("======================================================================")

if __name__ == "__main__":
    main()
