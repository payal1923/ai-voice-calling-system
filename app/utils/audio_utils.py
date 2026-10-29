import os
import wave
import av

# PyAV compatibility patch
_orig_av_open = av.open
def _patched_av_open(*args, **kwargs):
    kwargs.pop("metadata_errors", None)
    return _orig_av_open(*args, **kwargs)
av.open = _patched_av_open


def convert_to_16k_mono_wav(input_path: str, output_path: str = None) -> str:
    """
    Converts incoming audio (WebM/Opus, Ogg, or WAV) into a standard 16kHz mono
    16-bit PCM WAV file optimized for Faster-Whisper transcription.

    Preserves original speech dynamics without artificial distortion or clipping,
    avoiding stereo-to-mono downmix attenuation and Opus timestamp boundary truncation.
    Returns path to the 16kHz mono WAV file, or original path on failure.
    """
    if not input_path or not os.path.exists(input_path):
        return input_path

    # If input is already 16kHz mono WAV, return without redundant re-encoding
    if not output_path:
        if input_path.endswith("_16k.wav") and os.path.exists(input_path) and os.path.getsize(input_path) > 44:
            return input_path
        if input_path.endswith(".wav"):
            try:
                with wave.open(input_path, "rb") as wf:
                    if wf.getnchannels() == 1 and wf.getframerate() == 16000 and wf.getsampwidth() == 2:
                        return input_path
            except Exception:
                pass
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}_16k.wav"

    try:
        container = av.open(input_path)
        if not container.streams.audio:
            container.close()
            return input_path

        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)

            for frame in container.decode(audio=0):
                for resampled in resampler.resample(frame):
                    wf.writeframes(resampled.to_ndarray().tobytes())

        container.close()

        # Verify output file has content
        if os.path.exists(output_path) and os.path.getsize(output_path) > 44:
            return output_path
        return input_path
    except Exception as e:
        print(f"[AUDIO_UTILS WARNING] Failed converting {input_path} to 16k mono WAV: {e}")
        return input_path


def trim_audio_silence(input_wav_path: str, output_wav_path: str = None) -> str:
    """
    Trims dead leading and trailing silence/background noise from a 16kHz mono WAV file,
    preserving natural speech dynamics with generous padding buffers.

    Prevents Faster-Whisper hallucinations on trailing microphone hiss/fan noise
    WITHOUT truncating valid speech during natural mid-sentence pauses.
    Returns path to the trimmed WAV file (or original path if trimming is not applicable).
    """
    if not input_wav_path or not os.path.exists(input_wav_path):
        return input_wav_path
    if not input_wav_path.lower().endswith(".wav"):
        return input_wav_path

    try:
        import numpy as np

        with wave.open(input_wav_path, "rb") as wf:
            framerate = wf.getframerate()
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            nframes = wf.getnframes()
            if framerate != 16000 or nchannels != 1 or sampwidth != 2:
                return input_wav_path
            raw = wf.readframes(nframes)

        samples = np.frombuffer(raw, dtype=np.int16)
        if len(samples) == 0:
            return input_wav_path

        win_size = int(framerate * 0.02)  # 20ms frames = 320 samples
        n_wins = len(samples) // win_size
        if n_wins < 15:  # less than 300ms total, don't trim
            return input_wav_path

        energies = [np.sqrt(np.mean(samples[i * win_size : (i + 1) * win_size].astype(np.float32) ** 2)) for i in range(n_wins)]
        sorted_e = sorted(energies)
        noise_floor = float(np.median(sorted_e[: max(5, int(n_wins * 0.3))]))
        speech_thresh = max(450.0, noise_floor * 2.5)

        # 1. Find leading speech onset (first frame >= speech_thresh)
        start_win = None
        for i in range(n_wins):
            if energies[i] >= speech_thresh:
                start_win = i
                break

        if start_win is None:
            # No clear speech detected (silent or near-silent audio), keep original audio
            return input_wav_path

        # 2. Find trailing speech boundary (scan backward from file end for last frame >= speech_thresh)
        end_win = None
        for i in range(n_wins - 1, -1, -1):
            if energies[i] >= speech_thresh:
                end_win = i + 1
                break

        if end_win is None or end_win <= start_win:
            return input_wav_path

        # Apply padding (350ms lead-in, 400ms trailing pad)
        pad_in = int(0.35 / 0.02)
        pad_out = int(0.40 / 0.02)
        s_idx = max(0, (start_win - pad_in) * win_size)
        e_idx = min(len(samples), (end_win + pad_out) * win_size)

        trimmed_samples = samples[s_idx:e_idx]
        if len(trimmed_samples) < framerate * 0.4:
            # Don't over-trim below 400ms
            return input_wav_path

        # If trimmed audio is essentially the same as original, don't rewrite
        if len(trimmed_samples) >= len(samples) - (framerate * 0.4):
            return input_wav_path

        if not output_wav_path:
            base, _ = os.path.splitext(input_wav_path)
            output_wav_path = f"{base}_trimmed.wav"

        with wave.open(output_wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(framerate)
            wf.writeframes(trimmed_samples.tobytes())

        return output_wav_path
    except Exception as e:
        print(f"[AUDIO_UTILS WARNING] Error during silence trimming: {e}")
        return input_wav_path
