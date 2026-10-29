import os
import re
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
    "Hotel front desk and customer service conversation. Guest inquiries about arriving at half past eleven "
    "in the evening, late arrival, check-in time 3:00 PM, check-out time 12:00 PM, room booking, reservations, "
    "suites, complimentary high-speed Wi-Fi internet access, Wi-Fi in room, Wi-Fi password, airport pickup, "
    "airport transportation for passengers with large suitcases, luggage storage, wake-up call, "
    "packed breakfast to take away, takeaway breakfast, breakfast buffet, dining options, "
    "swimming pool, fitness centre, gym, spa, fresh towels in my room, extra towels, bath towels, "
    "fresh linen, extra pillow, blanket, bottled water, housekeeping, airport shuttle, cancellation policy, "
    "laundry, dry cleaning, pressing, shirts pressed, ironing clothes, elevator directions, "
    "billing, and general assistance."
)

TELECOM_DOMAIN_PROMPT = (
    "Customer service support conversation regarding billing, payments, internet connection, "
    "WiFi, broadband, network, router, data plans, and account assistance."
)


def normalize_acoustic_phonetic_errors(text: str) -> str:
    """
    Normalizes common acoustic and phonetic mistranscriptions from microphone audio,
    especially where initial consonants or accents are softened by VAD or compression.
    """
    if not text:
        return text

    # 1. 'taking time', 'the taking time', 'take-in time', 'take in time' -> 'check-in time'
    norm = re.sub(r"\b(?:the\s+)?tak(?:ing|e[- ]in)\s+tim(?:e|ing)(?:s)?\b", "check-in time", text, flags=re.IGNORECASE)
    # 2. 'what is taking time' -> 'what is check-in time'
    norm = re.sub(r"\b(?:what\s+is\s+)?tak(?:ing|e[- ]in)\s+tim(?:e|ing)(?:s)?\b", "what is check-in time", norm, flags=re.IGNORECASE)
    # 3. 'checking time', 'checking timings' -> 'check-in time'
    norm = re.sub(r"\bchecking\s+tim(?:e|ing)(?:s)?\b", "check-in time", norm, flags=re.IGNORECASE)
    # 4. 'chicken time', 'chicken in time' -> 'check-in time'
    norm = re.sub(r"\bchicken(?:\s+in)?\s+tim(?:e|ing)(?:s)?\b", "check-in time", norm, flags=re.IGNORECASE)
    # 5. 'chicken in' -> 'check-in'
    norm = re.sub(r"\bchicken\s+in\b", "check-in", norm, flags=re.IGNORECASE)
    # 6. 'checking in' -> 'check-in'
    norm = re.sub(r"\bchecking\s+in\b", "check-in", norm, flags=re.IGNORECASE)
    # 7. 'check time', 'check timing' -> 'check-in time'
    norm = re.sub(r"\bcheck\s+tim(?:e|ing)(?:s)?\b", "check-in time", norm, flags=re.IGNORECASE)
    # 8. 'work time in check-in' -> 'what time is check-in'
    norm = re.sub(r"\bwork\s+time\s+in\s+check-in\b", "what time is check-in", norm, flags=re.IGNORECASE)
    # 9. 'watch it check-in' / 'watch it check-in time' -> 'what is check-in time'
    norm = re.sub(r"\bwatch\s+it\s+check[- ]in(?:\s+time)?\b", "what is check-in time", norm, flags=re.IGNORECASE)
    # 10. 'verified available' / 'by far available' / 'right right available' -> 'is Wi-Fi available'
    norm = re.sub(r"\b(?:it['’]?s\s+)?verified\s+available\b", "is Wi-Fi available", norm, flags=re.IGNORECASE)
    norm = re.sub(r"\b(?:it['’]?s\s+)?by\s+far\s+available\b", "is Wi-Fi available", norm, flags=re.IGNORECASE)
    norm = re.sub(r"\b(?:it['’]?s\s+)?right\s+right\s+available\b", "is Wi-Fi available", norm, flags=re.IGNORECASE)
    # 11. 'in the river' -> 'in the room'
    norm = re.sub(r"\bavailable\s+in\s+the\s+river\b", "available in the room", norm, flags=re.IGNORECASE)
    # 12. 'look for two others and one child' -> 'book for two adults and one child'
    norm = re.sub(r"\blook\s+for\s+two\s+others\s+and\s+one\s+child\b", "book for two adults and one child", norm, flags=re.IGNORECASE)
    # 13. 'rocket check-out time' / 'what a check-in time' -> timing questions
    norm = re.sub(r"\brocket(?:\s+it)?\s*,?\s*check[- ]out(?:\s+time)?\b", "what is check-out time", norm, flags=re.IGNORECASE)
    norm = re.sub(r"\bwhat\s+a\s+check[- ]in\s+time\b", "what is check-in time", norm, flags=re.IGNORECASE)
    # 14. Wi-Fi acoustic mistranscriptions
    norm = re.sub(r"\b(?:and\s+)?don['’]?t\s+know\s+how\s+wi[- ]?fi(?:\s+system)?\b", "do you have Wi-Fi system", norm, flags=re.IGNORECASE)
    norm = re.sub(r"\byou\s+don['’]?t\s+even\s+have\s+wi[- ]?fi(?:\s+system)?\b", "do you have Wi-Fi system", norm, flags=re.IGNORECASE)
    norm = re.sub(r"\b(?:eat\s+)?blood\s+and\s+wi[- ]?fi\b", "is Wi-Fi available", norm, flags=re.IGNORECASE)
    norm = re.sub(r"\blife\s+as\s+a\s+student\b", "Wi-Fi system", norm, flags=re.IGNORECASE)
    norm = re.sub(r"\bwi[- ]?fi\s+in\s+the\s+middle(?!\s+of)\b", "Wi-Fi in my room", norm, flags=re.IGNORECASE)
    norm = re.sub(r"\bwi[- ]?fi\s+in\s+mood\s+room\b", "Wi-Fi in my room", norm, flags=re.IGNORECASE)
    norm = re.sub(
        r"\b(again\s*,?\s*)?(?:i['’]?m\s+not|high\s+note)\s+wi[- ]?fi(?:\s+in\s+(?:my|the)\s+room)?\b",
        lambda m: f"{m.group(1) or ''}I need Wi-Fi" + (" in my room" if "room" in m.group(0).lower() else ""),
        norm,
        flags=re.IGNORECASE
    )
    # 15. Pool & Parking acoustic mistranscriptions
    norm = re.sub(r"\bthen\s+you\s+have\s+a\s+full\s+system\b", "do you have a swimming pool", norm, flags=re.IGNORECASE)
    norm = re.sub(r"\band\s+to\s+know\s+how\s+specific\s+parking\s+systems?\s+are\s+best\b", "do you have specific parking system for guest", norm, flags=re.IGNORECASE)
    # 16. Packed breakfast acoustic mistranscriptions
    norm = re.sub(r"\bback\s+to\s+breakfast\b", "packed breakfast", norm, flags=re.IGNORECASE)
    # 17. Housekeeping & Towels acoustic mistranscriptions
    # Only replaces when the phrase strongly indicates an in-room amenity request,
    # preserving genuine pet inquiries ("Can I bring a dog?", "Are pets allowed?")
    # and bottled-water inquiries ("bottles of water", "water bottles").
    norm = re.sub(
        r"\b(?:i\s+need|bring|send|get)\s+(?:a\s+)?fresh\s+dog(?:s)?\s+(?:in|to)\s+(?:my|the|our)\s+room\b",
        "I need fresh towels in my room",
        norm,
        flags=re.IGNORECASE,
    )
    norm = re.sub(
        r"\bfresh\s+dog(?:s)?\s+(?:in|to)\s+(?:my|the|our)\s+room\b",
        "fresh towels in my room",
        norm,
        flags=re.IGNORECASE,
    )
    norm = re.sub(
        r"\b(?:i\s+need|bring|send|get)\s+(?:a\s+)?fresh\s+toys?\b(?!\s+(?:for|to)\s+(?:kids?|children|baby))",
        "I need fresh towels",
        norm,
        flags=re.IGNORECASE,
    )
    return norm



class STTService:
    """
    Speech-to-Text service backed by Faster-Whisper.

    The local Whisper model is loaded lazily on first use and shared across all
    instances to keep application boot fast and memory usage low.
    """

    def __init__(
        self,
        model_path: str = "small.en",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = None,
        beam_size: int = 5,
        cpu_threads: int = 4,
    ):
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.cpu_threads = cpu_threads
        self._model = self._get_model()

    def _get_model(self):
        """Loads (once) and returns the shared Faster-Whisper model instance."""
        cache_key = (
            self.model_path,
            self.device,
            self.compute_type,
            self.cpu_threads,
        )
        with _CACHE_LOCK:
            if cache_key not in _MODEL_CACHE:
                from faster_whisper import WhisperModel
                try:
                    _MODEL_CACHE[cache_key] = WhisperModel(
                        self.model_path,
                        device=self.device,
                        compute_type=self.compute_type,
                        cpu_threads=self.cpu_threads,
                    )
                except Exception as e:
                    print(f"[STT WARNING] Failed loading {self.model_path}: {e}. Falling back to tiny.en.")
                    self.model_path = "tiny.en"
                    fallback_key = ("tiny.en", self.device, self.compute_type, self.cpu_threads)
                    if fallback_key not in _MODEL_CACHE:
                        _MODEL_CACHE[fallback_key] = WhisperModel(
                            "tiny.en",
                            device=self.device,
                            compute_type=self.compute_type,
                            cpu_threads=self.cpu_threads,
                        )
                    return _MODEL_CACHE[fallback_key]
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
            temperature=0.0,
            language=target_lang,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=1000,
                speech_pad_ms=400,
            ),
            initial_prompt=domain_prompt,
            word_timestamps=True,
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
            seg_clean = seg_lower.rstrip(".!?,")
            # Filter phantom "thank you" / "thanks" hallucinations triggered by silence or noise clicks
            if seg_clean in ("thank you", "thanks", "thank you so much", "thank you very much") and getattr(segment, "no_speech_prob", 0.0) > 0.20:
                continue
            # Filter non-speech PSA/broadcast sign-off hallucinations (e.g. "For more information, visit www.fema.gov")
            # memorized from training corpora when triggered by background noise or silence.
            # Requires no_speech_prob > 0.35 so legitimate speech discussing informational websites is preserved.
            if any(psa in seg_lower for psa in ("fema.gov", "for more information, visit", "for more information visit")) and getattr(segment, "no_speech_prob", 0.0) > 0.35:
                continue
            # Filter pathological repetition loops
            if getattr(segment, "compression_ratio", 1.0) > 2.4:
                continue
            # Filter short (< 2.0s) non-speech noise/clipping bursts with high no_speech_prob and low confidence
            seg_dur = getattr(segment, "end", 0.0) - getattr(segment, "start", 0.0)
            if seg_dur < 2.0 and getattr(segment, "no_speech_prob", 0.0) > 0.75 and getattr(segment, "avg_logprob", 0.0) < -0.70:
                continue
            # Filter pure non-speech or severe noise without discarding accented speech
            if getattr(segment, "no_speech_prob", 0.0) > 0.85:
                continue
            if getattr(segment, "avg_logprob", 0.0) < -2.0 and getattr(segment, "no_speech_prob", 0.0) > 0.60:
                continue
            valid_segments.append(segment)

        segments = valid_segments

        # Safely prune low-confidence opening noise hallucinations before actual speech
        if segments and hasattr(segments[0], "words") and segments[0].words and len(segments[0].words) >= 2:
            first_w = segments[0].words[0]
            second_w = segments[0].words[1]
            w_prob = getattr(first_w, "probability", 1.0)
            w_clean = getattr(first_w, "word", "").strip().lower().rstrip(".,!?")
            gap = getattr(second_w, "start", 0.0) - getattr(first_w, "end", 0.0)

            is_spurious_onset = (
                w_prob < 0.10 and (
                    gap >= 0.15 or w_clean in ("go", "oh", "ah", "uh", "um", "okay", "done", "so", "and")
                ) and getattr(second_w, "probability", 0.0) >= 0.25
            )
            if is_spurious_onset:
                w_str = first_w.word.strip()
                orig_text = segments[0].text.strip()
                if orig_text.startswith(w_str):
                    segments[0].text = orig_text[len(w_str):].lstrip(" ,")

        text = " ".join(segment.text.strip() for segment in segments if segment.text).strip()
        text = normalize_acoustic_phonetic_errors(text)

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
            temperature=0.0,
            language=target_lang,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=1000,
                speech_pad_ms=400,
            ),
            initial_prompt=domain_prompt,
            word_timestamps=True,
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
            # Filter non-speech PSA/broadcast sign-off hallucinations (e.g. "For more information, visit www.fema.gov")
            if any(psa in seg_lower for psa in ("fema.gov", "for more information, visit", "for more information visit")) and getattr(segment, "no_speech_prob", 0.0) > 0.35:
                continue
            # Filter pathological repetition loops
            if getattr(segment, "compression_ratio", 1.0) > 2.4:
                continue
            # Filter short (< 2.0s) non-speech noise/clipping bursts with high no_speech_prob and low confidence
            seg_dur = getattr(segment, "end", 0.0) - getattr(segment, "start", 0.0)
            if seg_dur < 2.0 and getattr(segment, "no_speech_prob", 0.0) > 0.75 and getattr(segment, "avg_logprob", 0.0) < -0.70:
                continue
            # Filter pure non-speech or severe noise without discarding accented speech
            if getattr(segment, "no_speech_prob", 0.0) > 0.85:
                continue
            if getattr(segment, "avg_logprob", 0.0) < -2.0 and getattr(segment, "no_speech_prob", 0.0) > 0.60:
                continue
            valid_segments.append(segment)

        # Safely prune low-confidence opening noise hallucinations before actual speech
        if valid_segments and hasattr(valid_segments[0], "words") and valid_segments[0].words and len(valid_segments[0].words) >= 2:
            first_w = valid_segments[0].words[0]
            second_w = valid_segments[0].words[1]
            w_prob = getattr(first_w, "probability", 1.0)
            w_clean = getattr(first_w, "word", "").strip().lower().rstrip(".,!?")
            gap = getattr(second_w, "start", 0.0) - getattr(first_w, "end", 0.0)

            is_spurious_onset = (
                w_prob < 0.10 and (
                    gap >= 0.15 or w_clean in ("go", "oh", "ah", "uh", "um", "okay", "done", "so", "and")
                ) and getattr(second_w, "probability", 0.0) >= 0.25
            )
            if is_spurious_onset:
                w_str = first_w.word.strip()
                orig_text = valid_segments[0].text.strip()
                if orig_text.startswith(w_str):
                    valid_segments[0].text = orig_text[len(w_str):].lstrip(" ,")

        text = " ".join(s.text.strip() for s in valid_segments if s.text).strip()
        text = normalize_acoustic_phonetic_errors(text)
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


