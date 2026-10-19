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

    if not output_path:
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
