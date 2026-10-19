# TASK 24: Test Results and Code Modifications Report

## 1. Summary of File Changes

### Created Files
1. `app/utils/audio_utils.py` — Audio conversion utility using PyAV to convert incoming WebM/Opus and Ogg audio into standardized 16kHz mono 16-bit PCM WAV.
2. `tests/test_microphone_stt_task24.py` — Dedicated test suite verifying domain prompt isolation, audio resampling properties, clean speech STT WER, and recovery of previously garbled browser turns.
3. `TASK24_DIAGNOSTIC_REPORT.md` — Detailed root-cause analysis, before-and-after transcript comparisons, and WER benchmarks.
4. `TASK24_TEST_RESULTS.md` — Test commands, captured logs, and complete code diffs.

### Modified Files
1. `app/knowledge/companies/pullman_aerocity.json` — Added `"stt_prompt"` hotel domain vocabulary.
2. `app/services/knowledge_service.py` — Added `get_stt_prompt(company_id)` for domain-anchored vocabulary lookup.
3. `app/services/stt_service.py` — Replaced telecom fallback with neutral customer service prompt, increased Silero VAD pause tolerance to 1200ms, and relaxed segment rejection thresholds.
4. `app/api/call_routes.py` — Converted incoming WebSocket audio to 16kHz mono WAV via `convert_to_16k_mono_wav` and lowered turn size threshold to 1000 bytes.
5. `app/services/voice_response_service.py` — Passed company-grounded STT prompt and converted audio in `process_existing_turn_stream` and `process_existing_turn`.
6. `frontend/index.html` — Increased `silenceFramesLimit` to 14 (~1.12s) and `maxTurnDurationMs` to 15000 (15s), lowered blob threshold to 1200 bytes, and set recorder stop tail to 400ms.

### Deleted Files
- None.

---

## 2. Concise Code Diffs for Modified Files

### A. `app/knowledge/companies/pullman_aerocity.json`
```diff
--- app/knowledge/companies/pullman_aerocity.json.bak_task24
+++ app/knowledge/companies/pullman_aerocity.json
@@ -8,6 +8,7 @@
   },
   "system_instruction": "You are the AI Front Desk Concierge for Pullman New Delhi Aerocity. Assist guests with hotel information, check-in and check-out, dining, wellness facilities, and hotel policies. Keep responses concise (1-2 sentences), polite, and suitable for a voice call. Answer ONLY based on the approved hotel facts provided. If a requested detail, live room rate, or current availability is not in the facts, politely state that you do not have that live information and offer to connect the caller with the hotel front desk or reservations team. Do NOT invent prices, availability, or amenities.",
   "greeting_message": "Hello! Thank you for calling Pullman New Delhi Aerocity. How may I assist you with your stay or visit today?",
+  "stt_prompt": "Pullman New Delhi Aerocity hotel front desk conversation: check-in, check-out, room reservations, dining restaurants Pluck, Honk, Cafe Pluck, Chidya Ghar Bar, swimming pool timings, airport transfers, and guest amenities.",
   "facts": [
     {
       "id": "check_in_out",
```

### B. `app/services/knowledge_service.py`
```diff
--- app/services/knowledge_service.py.bak_task24
+++ app/services/knowledge_service.py
@@ -59,6 +59,26 @@
         if profile:
             return profile.get("company_name")
         return None
+
+    def get_stt_prompt(self, company_id: Optional[str] = None) -> str:
+        """
+        Returns a domain-anchored Faster-Whisper vocabulary prompt tailored to the company.
+        Avoids biasing hospitality or general queries with irrelevant telecom jargon.
+        """
+        if company_id:
+            profile = self.load_company_profile(company_id)
+            if profile:
+                custom_prompt = profile.get("stt_prompt")
+                if custom_prompt and str(custom_prompt).strip():
+                    return str(custom_prompt).strip()
+
+                c_name = profile.get("company_name") or "Hotel"
+                return (
+                    f"{c_name} front desk conversation: check-in, check-out, "
+                    f"room reservations, dining options, venues, amenities, and guest assistance."
+                )
+
+        return "Customer service conversation: inquiries, assistance, questions, and support."
 
     def get_relevant_knowledge(
         self,
```

### C. `app/services/stt_service.py`
```diff
--- app/services/stt_service.py.bak_task24
+++ app/services/stt_service.py
@@ -15,10 +15,15 @@
 _CACHE_LOCK = threading.Lock()
 
+DEFAULT_CONVERSATION_PROMPT = (
+    "Customer service conversation: inquiries, assistance, questions, and support."
+)
+
 TELECOM_DOMAIN_PROMPT = (
     "Customer service support conversation regarding billing, payments, internet connection, "
     "WiFi, broadband, network, router, data plans, and account assistance."
 )
@@ -84,7 +89,7 @@
         target_lang = language or self.language
         domain_prompt = initial_prompt
         if domain_prompt is None and (not target_lang or str(target_lang).lower() in ("en", "english")):
-            domain_prompt = TELECOM_DOMAIN_PROMPT
+            domain_prompt = DEFAULT_CONVERSATION_PROMPT
 
         segments_gen, info = self._model.transcribe(
             audio_path,
@@ -93,8 +98,8 @@
             condition_on_previous_text=False,
             vad_filter=True,
             vad_parameters=dict(
-                min_silence_duration_ms=500,
-                speech_pad_ms=350,
+                min_silence_duration_ms=1200,
+                speech_pad_ms=400,
             ),
             initial_prompt=domain_prompt,
             no_repeat_ngram_size=3,
@@ -110,7 +115,10 @@
             seg_lower = seg_text.lower()
             if any(h in seg_lower for h in ("next video", "watching", "subtitles by", "subscribe", "like and share", "see you in the next")):
                 continue
-            if getattr(segment, "avg_logprob", 0.0) < -1.15 and getattr(segment, "no_speech_prob", 0.0) > 0.45:
+            # Filter pure non-speech or severe noise without discarding accented speech
+            if getattr(segment, "no_speech_prob", 0.0) > 0.85:
+                continue
+            if getattr(segment, "avg_logprob", 0.0) < -2.0 and getattr(segment, "no_speech_prob", 0.0) > 0.60:
                 continue
             valid_segments.append(segment)
```

### D. `app/utils/audio_utils.py` (New Module)
```python
import os
import wave
import av

_orig_av_open = av.open
def _patched_av_open(*args, **kwargs):
    kwargs.pop("metadata_errors", None)
    return _orig_av_open(*args, **kwargs)
av.open = _patched_av_open


def convert_to_16k_mono_wav(input_path: str, output_path: str = None) -> str:
    """
    Converts incoming audio (WebM/Opus, Ogg, or WAV) into a standard 16kHz mono
    16-bit PCM WAV file optimized for Faster-Whisper transcription.
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
        if os.path.exists(output_path) and os.path.getsize(output_path) > 44:
            return output_path
        return input_path
    except Exception as e:
        print(f"[AUDIO_UTILS WARNING] Failed converting {input_path} to 16k mono WAV: {e}")
        return input_path
```

### E. `app/api/call_routes.py`
```diff
--- app/api/call_routes.py.bak_task24
+++ app/api/call_routes.py
@@ -16,6 +16,7 @@
 from app.database.database import get_db
 from app.database.models import Customer, Call, Message, Requirement, SystemUser
 from app.utils.auth import get_current_user, decode_access_token
+from app.utils.audio_utils import convert_to_16k_mono_wav
 
 router = APIRouter()
@@ -994,7 +995,7 @@
                     })
                     continue
 
-                if len(audio_data) < 1500:
+                if len(audio_data) < 1000:
                     print(f"[LIVE CALL] Ignoring undersized audio chunk ({len(audio_data)} bytes) for Call #{call.id}")
                     continue
@@ -1018,6 +1019,9 @@
 
                 with open(temp_input_path, "wb") as f_in:
                     f_in.write(audio_data)
+
+                # Convert incoming audio (WebM/Opus, Ogg, or WAV) to clean 16kHz mono WAV for high-accuracy Faster-Whisper STT
+                stt_input_path = await asyncio.to_thread(convert_to_16k_mono_wav, temp_input_path)
 
                 await websocket.send_json({
                     "event": "processing",
@@ -1085,7 +1089,7 @@
                         # Retain input audio for session diagnostics and audit inspection
                         pass
 
-                active_turn_task = asyncio.create_task(_run_turn_worker(temp_input_path, temp_output_path))
+                active_turn_task = asyncio.create_task(_run_turn_worker(stt_input_path, temp_output_path))
```

### F. `app/services/voice_response_service.py`
```diff
--- app/services/voice_response_service.py.bak_task24
+++ app/services/voice_response_service.py
@@ -13,6 +13,7 @@
 from app.services.escalation_service import EscalationService
 from app.services.knowledge_service import KnowledgeService
 from app.database.models import Requirement, Message
+from app.utils.audio_utils import convert_to_16k_mono_wav
 
@@ -145,21 +146,16 @@
 
         # 5. Transcribe with Faster-Whisper
         stt_start = time.perf_counter()
-        stt_prompt = None
-        if company_id:
-            c_name = self.knowledge_service.get_company_name(company_id) or "Hotel"
-            stt_prompt = (
-                f"{c_name} front desk: check-in time, check-out time, "
-                f"room reservations, dining, guest amenities."
-            )
+        stt_prompt = self.knowledge_service.get_stt_prompt(company_id) if hasattr(self.knowledge_service, "get_stt_prompt") else None
+        stt_audio_path = convert_to_16k_mono_wav(audio_path)
         if hasattr(self.stt_service, "transcribe_with_info"):
             transcript, audio_duration = self.stt_service.transcribe_with_info(
-                audio_path,
+                stt_audio_path,
                 initial_prompt=stt_prompt,
             )
         else:
             transcript = self.stt_service.transcribe(
-                audio_path,
+                stt_audio_path,
                 initial_prompt=stt_prompt,
             )
@@ -401,9 +397,15 @@
             raise ValueError(f"Call with ID {call_id} has already ended or has been routed to human assistance.")
         customer = call.customer
 
-        # 1. Transcribe audio with Faster-Whisper
+        # 1. Transcribe audio with Faster-Whisper using company-grounded STT prompt
         stt_start_at = time.perf_counter()
-        transcript, audio_duration = await asyncio.to_thread(self.stt_service.transcribe_with_info, audio_path)
+        stt_prompt = self.knowledge_service.get_stt_prompt(company_id) if hasattr(self.knowledge_service, "get_stt_prompt") else None
+        stt_audio_path = await asyncio.to_thread(convert_to_16k_mono_wav, audio_path)
+        transcript, audio_duration = await asyncio.to_thread(
+            self.stt_service.transcribe_with_info,
+            stt_audio_path,
+            initial_prompt=stt_prompt,
+        )
```

### G. `frontend/index.html`
```diff
--- frontend/index.html.bak_task24
+++ frontend/index.html
@@ -2217,8 +2217,8 @@
             speechThreshold: 22,          // Real voice vocal energy threshold (clear voice is 28-80)
             silenceThreshold: 18,         // Hysteresis silence cutoff threshold (calibrated above ambient fan hum)
             speechFramesTrigger: 1,       // 1 * 80ms = 80ms fast trigger to prevent losing initial consonants
-            silenceFramesLimit: 9,        // 9 * 80ms = ~720ms conversational pause threshold
-            maxTurnDurationMs: 7000,      // Safety limit: avoid trailing silence runaway
+            silenceFramesLimit: 14,       // 14 * 80ms = ~1.12s natural conversational pause threshold (prevents mid-sentence cuts)
+            maxTurnDurationMs: 15000,     // Generous 15s limit: prevents cutting off detailed caller questions
             preRollSeconds: 0.40,         // Pre-roll duration
         };
@@ -3019,8 +3019,8 @@
                         const audioBlob = new Blob(liveLiveAudioChunks, { type: liveRecorder.mimeType || 'audio/webm' });
                         const turnDurationSec = ((Date.now() - turnStartTime) / 1000).toFixed(2);
                         
-                        // Only dispatch turns with meaningful audio content (> 2000 bytes, approx > 0.5s of speech)
-                        if (audioBlob.size >= 2000) {
+                        // Only dispatch turns with meaningful audio content (> 1200 bytes, allows short affirmations and responses)
+                        if (audioBlob.size >= 1200) {
                             isAssistantProcessing = true;
                             logSip(`Sending speech turn (${(audioBlob.size / 1024).toFixed(1)} KB, ~${turnDurationSec}s) to AI pipeline...`);
                             setVoiceState('processing', `Processing speech (~${turnDurationSec}s)...`);
@@ -3058,6 +3058,8 @@
             setVoiceState('processing');
 
             if (flushTail) {
+                // Ensure the rolling pre-roll delay buffer (400ms) has fully flushed into the stream
+                const tailDrainMs = Math.max(180, Math.round((VAD_CONFIG.preRollSeconds || 0.40) * 1000));
                 setTimeout(() => {
                     if (liveRecorder && liveRecorder.state !== 'inactive') {
                         try {
@@ -3066,7 +3068,7 @@
                             console.warn(e);
                         }
                     }
-                }, 80);
+                }, tailDrainMs);
             } else {
```

---

## 3. Test Suites Executed and Exact Outputs

### Suite 1: `tests/test_microphone_stt_task24.py`
**Command**: `.\.venv\Scripts\python.exe -m unittest tests/test_microphone_stt_task24.py -v`
**Output**:
```text
test_audio_resampling_properties (tests.test_microphone_stt_task24.TestAudioConverter.test_audio_resampling_properties) ... ok
test_generic_stt_prompt_fallback (tests.test_microphone_stt_task24.TestKnowledgeSTTPrompt.test_generic_stt_prompt_fallback) ... ok
test_pullman_stt_prompt (tests.test_microphone_stt_task24.TestKnowledgeSTTPrompt.test_pullman_stt_prompt) ... ok
test_turn_746_2_no_telecom_hallucination (tests.test_microphone_stt_task24.TestObservedBrowserTurnsRecovery.test_turn_746_2_no_telecom_hallucination) ... ok
test_turn_747_1_restaurant_family (tests.test_microphone_stt_task24.TestObservedBrowserTurnsRecovery.test_turn_747_1_restaurant_family) ... ok
test_turn_747_2_services (tests.test_microphone_stt_task24.TestObservedBrowserTurnsRecovery.test_turn_747_2_services) ... ok
test_clean_speech_zero_wer (tests.test_microphone_stt_task24.TestPrerecordedCleanSpeechSTT.test_clean_speech_zero_wer) ... ok
test_order_speech_low_wer (tests.test_microphone_stt_task24.TestPrerecordedCleanSpeechSTT.test_order_speech_low_wer) ... ok

----------------------------------------------------------------------
Ran 8 tests in 23.929s

OK
```

### Suite 2: `tests/test_sentence_streaming_task22.py`
**Command**: `.\.venv\Scripts\python.exe -m unittest tests/test_sentence_streaming_task22.py -v`
**Output**:
```text
test_01_one_sentence_response_streaming ... ok
test_02_multi_sentence_response_streaming ... ok
test_03_sentence_ordering_and_abbreviations ... ok
test_04_empty_or_malformed_chunks ... ok
test_05_tts_failure_after_first_sentence ... ok
test_06_websocket_disconnection_during_streaming ... ok
test_07_call_ending_during_playback ... ok
test_08_no_cross_session_audio_leakage ... ok
test_09_grounded_responses_and_clarification ... ok
test_10_performance_latency_instrumentation ... ok

----------------------------------------------------------------------
Ran 10 tests in 65.245s

OK
```

### Suite 3: `tests/test_natural_followup_recovery_task20.py`
**Command**: `.\.venv\Scripts\python.exe -m unittest tests/test_natural_followup_recovery_task20.py -v`
**Output**:
```text
test_01_pluck_recommendation_followed_by_food_served ... ok
test_02_pluck_followed_by_direct_vegetarian_options ... ok
test_03_ambiguous_visible_options_in_dining_context ... ok
test_04_conversational_statement_they_have_great_options ... ok
test_05_clarification_confirmation_answers_honestly ... ok
test_06_repeated_unclear_turns_prevent_identical_loop ... ok
test_07_clear_topic_change_to_checkin ... ok
test_08_unsupported_dietary_information_never_invented ... ok

----------------------------------------------------------------------
Ran 8 tests in 0.332s

OK
```

### Suite 4: `tests/test_context_and_relevance_task19.py`
**Command**: `.\.venv\Scripts\python.exe -m unittest tests/test_context_and_relevance_task19.py -v`
**Output**:
```text
test_01_garbled_restaurant_query_never_triggers_checkin_id_policy ... ok
test_02_food_served_after_pluck_recommendation_refers_to_pluck ... ok
test_03_vegetarian_option_questions_answered_only_when_supported ... ok
test_04_price_and_reservation_questions_do_not_receive_unrelated_answers ... ok
test_05_clear_topic_change_overrides_prior_context ... ok
test_06_unclear_speech_focused_clarification_and_variation ... ok
test_07_genuine_checkin_id_policy_query_answered_correctly ... ok

----------------------------------------------------------------------
Ran 7 tests in 11.135s

OK
```

### Suite 5: `tests/test_speech_recognition_sentences.py`
**Command**: `.\.venv\Scripts\python.exe tests/test_speech_recognition_sentences.py`
**Output**:
```text
========================================================
TESTING SPEECH RECOGNITION ACROSS MULTIPLE SPOKEN SENTENCES
========================================================

--- [Sentence #1] Testing: "Hello, can you help me check my billing statement?" ---
  [Original Spoken]: "Hello, can you help me check my billing statement?"
  [Recognized]:      "Hello can you help me check my billing statement"
  [Audio Duration]:  2.69s
  [Word Match Accuracy]: 100.0% (8/8 words)
  [PASS] Recognized transcript matched words spoken!

--- [Sentence #2] Testing: "My internet is running very slow today and disconnecting." ---
  [Original Spoken]: "My internet is running very slow today and disconnecting."
  [Recognized]:      "My internet is running very slow today and disconnecting."
  [Audio Duration]:  2.88s
  [Word Match Accuracy]: 100.0% (8/8 words)
  [PASS] Recognized transcript matched words spoken!

--- [Sentence #3] Testing: "I would like to speak with a customer support representative." ---
  [Original Spoken]: "I would like to speak with a customer support representative."
  [Recognized]:      "I would like to speak with a customer support representative."
  [Audio Duration]:  3.01s
  [Word Match Accuracy]: 100.0% (10/10 words)
  [PASS] Recognized transcript matched words spoken!

--- [Sentence #4] Testing: "Can you tell me what broadband plans you offer?" ---
  [Original Spoken]: "Can you tell me what broadband plans you offer?"
  [Recognized]:      "Can you tell me what broadband plans you offer?"
  [Audio Duration]:  2.63s
  [Word Match Accuracy]: 100.0% (8/8 words)
  [PASS] Recognized transcript matched words spoken!

--- [Sentence #5] Testing: "Thank you so much, that solved my issue." ---
  [Original Spoken]: "Thank you so much, that solved my issue."
  [Recognized]:      "Thank you so much that solved my issue."
  [Audio Duration]:  2.40s
  [Word Match Accuracy]: 100.0% (8/8 words)
  [PASS] Recognized transcript matched words spoken!

--- [Sentence #6] Testing: "My internet connection is not working." ---
  [Original Spoken]: "My internet connection is not working."
  [Recognized]:      "My internet connection is not working."
  [Audio Duration]:  2.14s
  [Word Match Accuracy]: 100.0% (6/6 words)
  [PASS] Recognized transcript matched words spoken!

========================================================
ALL SPOKEN SENTENCES RECOGNIZED ACCURATELY AND VERIFIED!
========================================================
```

---

## 4. Aggregate Test Status

| Metric | Count | Percentage |
| :--- | :---: | :---: |
| **Total Tests Ran** | 39 | 100% |
| **Total Passed** | 39 | 100% |
| **Total Failed** | 0 | 0% |
| **Total Skipped** | 0 | 0% |
| **Overall Suite Status** | **ALL PASSED** | **100%** |
