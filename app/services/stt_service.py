import os
import threading
import av

# PyAV 19.0.0 compatibility patch for faster-whisper
_orig_av_open = av.open
def _patched_av_open(*args, **kwargs):
    kwargs.pop("metadata_errors", None)
    return _orig_av_open(*args, **kwargs)
av.open = _patched_av_open

# Cache loaded Whisper models so repeated STTService() instantiation (across
# VoiceResponseService, CallService, routes and scripts) never reloads weights.
_MODEL_CACHE = {}
_CACHE_LOCK = threading.Lock()


DEFAULT_CONVERSATION_PROMPT = (
    "Customer service conversation: inquiries, assistance, questions, and support."
)

TELECOM_DOMAIN_PROMPT = (
    "Customer service support conversation regarding billing, payments, internet connection, "
    "WiFi, broadband, network, router, data plans, and account assistance."
)



class STTService:
    """
    Speech-to-Text service backed by Faster-Whisper.

    The local Whisper model is loaded lazily on first use and shared across all
    instances to keep application boot fast and memory usage low.
    """

    def __init__(
        self,
        model_path: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = None,
        beam_size: int = 5,
    ):
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self._model = self._get_model()

    def _get_model(self):
        """Loads (once) and returns the shared Faster-Whisper model instance."""
        cache_key = (
            os.path.abspath(self.model_path),
            self.device,
            self.compute_type,
        )
        with _CACHE_LOCK:
            if cache_key not in _MODEL_CACHE:
                from faster_whisper import WhisperModel

                _MODEL_CACHE[cache_key] = WhisperModel(
                    self.model_path,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            return _MODEL_CACHE[cache_key]

    def transcribe_with_info(
        self,
        audio_path: str,
        language: str = None,
        initial_prompt: str = None,
    ) -> tuple[str, float]:
        """
        Transcribes an audio file to text using Faster-Whisper with Silero VAD filtering
        and domain-anchored vocabulary prompting.

        Returns tuple of (transcript_text, audio_duration_seconds).
        Raises FileNotFoundError if the supplied audio path does not exist.
        """
        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found for transcription: {audio_path}")

        if self._model is None:
            self._model = self._get_model()

        target_lang = language or self.language
        domain_prompt = initial_prompt
        if domain_prompt is None and (not target_lang or str(target_lang).lower() in ("en", "english")):
            domain_prompt = DEFAULT_CONVERSATION_PROMPT

        segments_gen, info = self._model.transcribe(
            audio_path,
            beam_size=self.beam_size,
            language=target_lang,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=1200,
                speech_pad_ms=400,
            ),
            initial_prompt=domain_prompt,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
        )

        segments = list(segments_gen)
        valid_segments = []
        for segment in segments:
            seg_text = segment.text.strip()
            if not seg_text:
                continue
            seg_lower = seg_text.lower()
            if any(h in seg_lower for h in ("next video", "watching", "subtitles by", "subscribe", "like and share", "see you in the next")):
                continue
            # Filter pure non-speech or severe noise without discarding accented speech
            if getattr(segment, "no_speech_prob", 0.0) > 0.85:
                continue
            if getattr(segment, "avg_logprob", 0.0) < -2.0 and getattr(segment, "no_speech_prob", 0.0) > 0.60:
                continue
            valid_segments.append(segment)

        segments = valid_segments
        text = " ".join(segment.text.strip() for segment in segments if segment.text).strip()

        # If voice detection finds no speech, safely retain empty transcript without forcing hallucination
        if not text:
            segments = []

        file_size = os.path.getsize(audio_path)
        audio_duration = float(info.duration) if info and hasattr(info, "duration") else 0.0
        
        # Calculate confidence statistics across segments
        if segments:
            logprobs = [float(getattr(s, "avg_logprob", 0.0)) for s in segments]
            no_speech_probs = [float(getattr(s, "no_speech_prob", 0.0)) for s in segments]
            comp_ratios = [float(getattr(s, "compression_ratio", 1.0)) for s in segments]
            mean_logprob = sum(logprobs) / len(logprobs)
            max_no_speech = max(no_speech_probs)
            mean_comp = sum(comp_ratios) / len(comp_ratios)
        else:
            mean_logprob = 0.0
            max_no_speech = 1.0 if not text else 0.0
            mean_comp = 1.0

        print(f"[FASTER-WHISPER DIAGNOSTIC] Audio File: {os.path.basename(audio_path)} ({file_size} bytes, duration: {audio_duration:.2f}s, lang: {info.language} {info.language_probability:.2f})")
        print(f"[FASTER-WHISPER CONFIDENCE] Metrics: mean_avg_logprob={mean_logprob:.3f}, max_no_speech_prob={max_no_speech:.3f}, mean_compression_ratio={mean_comp:.2f}")
        for idx, seg in enumerate(segments):
            print(f"  [Seg {idx+1}] [{seg.start:.2f}s -> {seg.end:.2f}s] \"{seg.text.strip()}\" (avg_logprob: {getattr(seg, 'avg_logprob', 0.0):.3f}, no_speech_prob: {getattr(seg, 'no_speech_prob', 0.0):.3f}, compression_ratio: {getattr(seg, 'compression_ratio', 1.0):.2f})")
        print(f"[FASTER-WHISPER RECOGNITION] Final Recognized Transcript: \"{text}\" (Duration: {audio_duration:.2f}s)")

        return text, audio_duration

    def transcribe_with_confidence(
        self,
        audio_path: str,
        language: str = None,
        initial_prompt: str = None,
    ) -> dict:
        """
        Transcribes audio and returns full confidence metrics alongside text and duration.
        """
        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if self._model is None:
            self._model = self._get_model()

        target_lang = language or self.language
        domain_prompt = initial_prompt
        if domain_prompt is None and (not target_lang or str(target_lang).lower() in ("en", "english")):
            domain_prompt = DEFAULT_CONVERSATION_PROMPT

        segments_gen, info = self._model.transcribe(
            audio_path,
            beam_size=self.beam_size,
            language=target_lang,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=1200,
                speech_pad_ms=400,
            ),
            initial_prompt=domain_prompt,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
        )

        segments = list(segments_gen)
        valid_segments = []
        for segment in segments:
            seg_text = segment.text.strip()
            if not seg_text:
                continue
            seg_lower = seg_text.lower()
            if any(h in seg_lower for h in ("next video", "watching", "subtitles by", "subscribe", "like and share", "see you in the next")):
                continue
            # Filter pure non-speech or severe noise without discarding accented speech
            if getattr(segment, "no_speech_prob", 0.0) > 0.85:
                continue
            if getattr(segment, "avg_logprob", 0.0) < -2.0 and getattr(segment, "no_speech_prob", 0.0) > 0.60:
                continue
            valid_segments.append(segment)

        text = " ".join(s.text.strip() for s in valid_segments if s.text).strip()
        audio_dur = float(info.duration) if info and hasattr(info, "duration") else 0.0

        if valid_segments:
            logprobs = [float(getattr(s, "avg_logprob", 0.0)) for s in valid_segments]
            no_speech = [float(getattr(s, "no_speech_prob", 0.0)) for s in valid_segments]
            comp = [float(getattr(s, "compression_ratio", 1.0)) for s in valid_segments]
            mean_logprob = sum(logprobs) / len(logprobs)
            max_no_speech = max(no_speech)
            mean_comp = sum(comp) / len(comp)
        else:
            mean_logprob = -2.0 if not text else 0.0
            max_no_speech = 1.0 if not text else 0.0
            mean_comp = 1.0

        return {
            "transcript": text,
            "duration": audio_dur,
            "mean_avg_logprob": round(mean_logprob, 3),
            "max_no_speech_prob": round(max_no_speech, 3),
            "mean_compression_ratio": round(mean_comp, 2),
            "segment_count": len(valid_segments),
        }

    def transcribe(
        self,
        audio_path: str,
        language: str = None,
        initial_prompt: str = None,
    ) -> str:
        """
        Transcribes an audio file to text using Faster-Whisper.
        Returns the concatenated transcript string.
        """
        text, _ = self.transcribe_with_info(
            audio_path=audio_path,
            language=language,
            initial_prompt=initial_prompt,
        )
        return text


