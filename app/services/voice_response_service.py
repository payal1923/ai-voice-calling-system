import os
import time
import uuid
import json
import asyncio
from typing import Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from app.services.stt_service import STTService
from app.services.response_service import ResponseService
from app.services.tts_service import TTSService
from app.services.conversation_service import ConversationService
from app.services.conversation_context_service import ConversationContextService
from app.services.requirement_service import RequirementService
from app.services.escalation_service import EscalationService
from app.services.knowledge_service import KnowledgeService
from app.database.models import Requirement, Message, Call
from app.utils.audio_utils import convert_to_16k_mono_wav, trim_audio_silence


class VoiceResponseService:
    def __init__(
        self,
        stt_service: STTService = None,
        response_service: ResponseService = None,
        tts_service: TTSService = None,
        conversation_service: ConversationService = None,
        context_service: ConversationContextService = None,
        requirement_service: RequirementService = None,
        knowledge_service: KnowledgeService = None,
    ):
        """
        Initializes VoiceResponseService by reusing STTService, ResponseService, TTSService,
        ConversationService, ConversationContextService, RequirementService, and KnowledgeService.
        """
        self.stt_service = stt_service if stt_service is not None else STTService()
        self.response_service = (
            response_service if response_service is not None else ResponseService()
        )
        self.tts_service = tts_service if tts_service is not None else TTSService()
        self.conversation_service = (
            conversation_service if conversation_service is not None else ConversationService()
        )
        self.context_service = (
            context_service if context_service is not None else ConversationContextService()
        )
        self.requirement_service = (
            requirement_service if requirement_service is not None else RequirementService()
        )
        self.knowledge_service = (
            knowledge_service if knowledge_service is not None else KnowledgeService()
        )

    def _print_performance_summary(
        self,
        stt_time: float,
        context_db_time: float,
        llm_time: float,
        tts_time: float,
        db_persistence_time: float,
        total_time: float,
        used_streaming: bool = False,
    ):
        print("=== PERFORMANCE ===")
        print(f"[PERF] STT: {stt_time:.3f} sec")
        print(f"[PERF] DB: {(context_db_time + db_persistence_time):.3f} sec")
        print(f"[PERF] LLM_RESPONSE: {llm_time:.3f} sec ({'STREAMING' if used_streaming else 'NON-STREAMING'})")
        print(f"[PERF] TTS: {tts_time:.3f} sec")
        print(f"[PERF] TOTAL_CRITICAL_PATH: {total_time:.3f} sec")
        print("===================")

    def _schedule_background_requirements(self, call_id: int, full_text: str, is_resolved: bool):
        """Dispatches requirements extraction in a non-blocking background task."""
        async def _bg_worker():
            t_bg_start = time.perf_counter()
            from app.database.database import SessionLocal
            bg_db = SessionLocal()
            try:
                await self._update_requirements(bg_db, call_id, full_text, is_resolved)
                t_bg_dur = time.perf_counter() - t_bg_start
                print(f"[PERF] REQUIREMENT_EXTRACTION_BACKGROUND: {t_bg_dur:.3f} sec (Call #{call_id})")
            except Exception as e:
                print(f"Background requirements extraction error for call {call_id}: {e}")
            finally:
                bg_db.close()

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_bg_worker())
        except RuntimeError:
            pass

    async def _update_requirements(self, db: Session, call_id: int, full_text: str, is_resolved: bool):
        """Extracts and persists customer requirements into requirements table with smart field preservation."""
        try:
            # Check if Call exists before processing to avoid foreign key errors in transient/deleted sessions
            call_exists = db.query(Call).filter(Call.id == call_id).first()
            if not call_exists:
                db.rollback()
                return

            req_data = await self.requirement_service.extract_requirements(full_text, is_resolved=is_resolved)
            now = datetime.now(timezone.utc)
            req_record = db.query(Requirement).filter(Requirement.call_id == call_id).first()
            if not req_record:
                req_record = Requirement(
                    call_id=call_id,
                    status="COLLECTED" if is_resolved else "IN_PROGRESS",
                    structured_requirements=req_data,
                    created_at=now,
                    updated_at=now,
                )
                db.add(req_record)
            else:
                # Merge existing requirements to preserve non-null values
                existing = req_record.structured_requirements or {}
                merged = dict(req_data)
                for key in ["requirements", "products_or_services", "quantity", "budget", "deadline"]:
                    if not merged.get(key) and existing.get(key):
                        merged[key] = existing[key]
                if not merged.get("important_details") and existing.get("important_details"):
                    merged["important_details"] = existing["important_details"]

                req_record.structured_requirements = merged
                req_record.status = "COLLECTED" if is_resolved else "IN_PROGRESS"
                req_record.updated_at = now
            db.commit()
        except Exception as e:
            try:
                db.rollback()
            except Exception:
                pass
            print(f"Warning: Failed to update requirements for call {call_id}: {e}")

    async def process_existing_turn(
        self,
        call_id: int,
        audio_path: str,
        output_audio_path: str,
        db: Session,
        company_id: str = None,
        turn_id: Optional[str] = None,
    ) -> dict:
        """
        Processes an additional voice turn for an already active call.
        Optimized pipeline: Transcribe -> LLM Response -> Immediate TTS Synthesis -> Return audio.
        Requirements are extracted asynchronously in the background.
        Includes correlated turn_id.
        """
        total_start = time.perf_counter()
        effective_company_id = (company_id or "").strip().lower() or "pullman_aerocity"
        effective_turn_id = turn_id or f"turn_{call_id}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"

        # 1-4. DB Retrieval & Context Building
        db_start = time.perf_counter()
        call = self.conversation_service.get_call(db, call_id)
        if call.ended_at is not None:
            raise ValueError(f"Call with id {call_id} has already ended.")
        customer = call.customer
        history_context = self.context_service.build_context([call])
        context_db_time = time.perf_counter() - db_start

        # 5. Transcribe with Faster-Whisper
        stt_start = time.perf_counter()
        stt_prompt = self.knowledge_service.get_stt_prompt(effective_company_id) if hasattr(self.knowledge_service, "get_stt_prompt") else None
        stt_audio_path = convert_to_16k_mono_wav(audio_path)
        trimmed_stt_path = trim_audio_silence(stt_audio_path)
        effective_stt_path = trimmed_stt_path if trimmed_stt_path and os.path.exists(trimmed_stt_path) else stt_audio_path
        if hasattr(self.stt_service, "transcribe_with_info"):
            transcript, audio_duration = self.stt_service.transcribe_with_info(
                effective_stt_path,
                initial_prompt=stt_prompt,
                company_id=effective_company_id,
            )
        else:
            transcript = self.stt_service.transcribe(
                stt_audio_path,
                initial_prompt=stt_prompt,
                company_id=effective_company_id,
            )
            audio_duration = 0.0
        stt_time = time.perf_counter() - stt_start
        stt_end_at = time.perf_counter()

        clean_transcript = (transcript or "").strip()
        if clean_transcript:
            print(f"[STT] Call #{call.id} Transcribed (Duration: {audio_duration:.2f}s): \"{clean_transcript}\" ({stt_time:.3f}s)")
        else:
            print(f"[STT] Call #{call.id} [EMPTY TRANSCRIPTION] No speech detected in audio turn ({audio_duration:.2f}s, {stt_time:.3f}s)")
            # Conversational safety: Do not invent answers or invoke LLM on empty/undetected speech
            response_text = "I'm sorry, I didn't catch that clearly. Could you please repeat your request?"
            self.conversation_service.add_message(
                db, call_id=call.id, speaker="customer", text=""
            )
            self.conversation_service.add_message(
                db, call_id=call.id, speaker="assistant", text=response_text
            )
            tts_start = time.perf_counter()
            generated_audio_path = self.tts_service.synthesize(response_text, output_audio_path)
            tts_time = time.perf_counter() - tts_start
            total_time = time.perf_counter() - total_start
            return {
                "turn_id": effective_turn_id,
                "transcript": "",
                "audio_duration": audio_duration,
                "response": response_text,
                "audio_path": generated_audio_path,
                "customer_id": customer.external_customer_id,
                "call_id": call.id,
                "call_status": call.status,
                "is_escalated": False,
                "timings": {
                    "stt_sec": round(stt_time, 2),
                    "llm_sec": 0.0,
                    "tts_sec": round(tts_time, 2),
                    "context_db_sec": round(context_db_time, 2),
                    "total_sec": round(total_time, 2),
                },
            }

        # 6. Save customer message
        p1_start = time.perf_counter()
        self.conversation_service.add_message(
            db, call_id=call.id, speaker="customer", text=clean_transcript
        )
        p1_time = time.perf_counter() - p1_start

        # Check Human Escalation
        should_esc, esc_reason = EscalationService.should_escalate(transcript)
        if should_esc:
            EscalationService.create_escalation(
                db=db,
                call_id=call.id,
                reason=esc_reason,
                language_selected=call.language or "en",
                mark_call_ended=False,
            )
            response_text = (
                "A live representative transfer is not available on this line. "
                "I have registered an assistance request with our front desk team. "
                "You may continue speaking with me, or contact the front desk directly for immediate assistance."
            )
            self.conversation_service.add_message(
                db, call_id=call.id, speaker="assistant", text=response_text
            )
            db.commit()

            tts_start = time.perf_counter()
            generated_audio_path = self.tts_service.synthesize(response_text, output_audio_path)
            tts_time = time.perf_counter() - tts_start
            total_time = time.perf_counter() - total_start

            return {
                "turn_id": effective_turn_id,
                "transcript": transcript,
                "audio_duration": audio_duration,
                "response": response_text,
                "audio_path": generated_audio_path,
                "customer_id": customer.external_customer_id,
                "call_id": call.id,
                "call_status": "IN_PROGRESS",
                "is_escalated": True,
                "timings": {
                    "stt_sec": round(stt_time, 2),
                    "llm_sec": 0.0,
                    "tts_sec": round(tts_time, 2),
                    "context_db_sec": round(context_db_time, 2),
                    "total_sec": round(total_time, 2),
                },
            }

        # 7. Generate assistant response
        #
        # Two interchangeable paths, identical prompt / model / response semantics:
        #   - non-streaming (default): waits for the COMPLETE LLM response, then TTS.
        #   - streaming (opt-in): returns at the first COMPLETE SENTENCE, then TTS.
        # Enable streaming with LLM_STREAMING_ENABLED=1. Any streaming failure
        # falls back to the non-streaming path.
        streaming_enabled = str(os.environ.get("LLM_STREAMING_ENABLED", "0")).strip() in ("1", "true", "True", "yes")
        llm_start = time.perf_counter()
        llm_details = {}
        used_streaming = False

        # Resolve company persona and knowledge context (defaults to pullman_aerocity)
        system_instruction = self.knowledge_service.get_system_instruction(effective_company_id)
        knowledge_context = self.knowledge_service.get_relevant_knowledge(
            effective_company_id, transcript, conversation_context=history_context
        )

        if streaming_enabled:
            try:
                response_text, llm_details = await self.response_service.generate_response_stream_sentence(
                    transcript=transcript,
                    conversation_context=history_context,
                    system_instruction=system_instruction,
                    knowledge_context=knowledge_context,
                    company_id=effective_company_id,
                )
                used_streaming = True
            except Exception as e:
                print(f"Warning: streaming LLM path failed, falling back to non-streaming: {e}")
                used_streaming = False

        if not used_streaming:
            llm_res = await self.response_service.generate_response(
                transcript=transcript,
                conversation_context=history_context,
                return_details=True,
                system_instruction=system_instruction,
                knowledge_context=knowledge_context,
                company_id=effective_company_id,
            )
            if isinstance(llm_res, tuple):
                response_text, llm_details = llm_res
            else:
                response_text = llm_res
        llm_time = time.perf_counter() - llm_start

        # 8. Save assistant response
        p2_start = time.perf_counter()
        self.conversation_service.add_message(
            db, call_id=call.id, speaker="assistant", text=response_text
        )
        p2_time = time.perf_counter() - p2_start

        # 8b. Fast resolution check
        is_resolved = await self.response_service.check_resolution(transcript, history_context)
        call_status = "RESOLVED" if is_resolved else "IN_PROGRESS"

        if is_resolved:
            self.conversation_service.end_call(db, call.id, status="RESOLVED")
        else:
            self.conversation_service.update_call_status(db, call.id, status="IN_PROGRESS")

        db_persistence_time = p1_time + p2_time

        # 9. Synthesize TTS immediately so the customer gets audio as fast as possible
        tts_start = time.perf_counter()
        generated_audio_path = self.tts_service.synthesize(response_text, output_audio_path)
        tts_time = time.perf_counter() - tts_start
        first_audio_at = time.perf_counter()

        total_time = time.perf_counter() - total_start

        # 10. Extract and persist structured customer requirements asynchronously in background
        customer_msgs = (
            db.query(Message)
            .filter(Message.call_id == call.id, Message.speaker == "customer")
            .order_by(Message.id)
            .all()
        )
        full_customer_transcript = "\n".join(m.text for m in customer_msgs if m.text)
        if not full_customer_transcript.strip():
            full_customer_transcript = transcript

        self._schedule_background_requirements(call.id, full_customer_transcript, is_resolved)

        self._print_performance_summary(
            stt_time=stt_time,
            context_db_time=context_db_time,
            llm_time=llm_time,
            tts_time=tts_time,
            db_persistence_time=db_persistence_time,
            total_time=total_time,
            used_streaming=used_streaming,
        )

        return {
            "turn_id": effective_turn_id,
            "transcript": transcript,
            "audio_duration": audio_duration,
            "response": response_text,
            "audio_path": generated_audio_path,
            "customer_id": customer.external_customer_id,
            "call_id": call.id,
            "call_status": call_status,
            "is_escalated": False,
            "timings": {
                "stt_sec": round(stt_time, 3),
                "llm_sec": round(llm_time, 3),
                "tts_sec": round(tts_time, 3),
                "context_db_sec": round(context_db_time + db_persistence_time, 3),
                "total_sec": round(total_time, 3),
                "lock_wait_sec": round(llm_details.get("lock_wait_sec", 0.0), 3),
                "ollama_req_sec": round(llm_details.get("ollama_req_sec", 0.0), 3),
                # Streaming-specific instrumentation
                "used_streaming": used_streaming,
                "llm_ttft_sec": round(llm_details.get("llm_ttft_sec", 0.0), 3),
                "llm_sentence_sec": round(llm_details.get("llm_sentence_sec", 0.0), 3),
                "stt_end_to_first_audio_sec": round(first_audio_at - stt_end_at, 3),
            },
        }

    async def process_existing_turn_stream(
        self,
        call_id: int,
        audio_path: str,
        output_audio_path_prefix: str,
        db: Session,
        company_id: Optional[str] = None,
        turn_id: Optional[str] = None,
    ):
        """
        Sentence-level streaming turn processor:
        1. Transcribes audio turn with Faster-Whisper.
        2. Retrieves relevant grounded knowledge and conversation context.
        3. Streams sentences one-by-one from ResponseService as soon as each sentence boundary is detected.
        4. Synthesizes Piper TTS audio for that sentence chunk immediately and yields an 'assistant_chunk'
           event over WebSocket so the frontend starts audio playback with minimum latency.
        5. Finally yields the complete 'assistant_response' event for conversation history and transcript display.
        Includes correlated turn_id and structured audit trace logging.
        """
        import time
        import os
        import asyncio

        turn_start_at = time.perf_counter()
        effective_company_id = (company_id or "").strip().lower() or "pullman_aerocity"

        call = self.conversation_service.get_call(db, call_id)
        if not call:
            raise ValueError(f"Call with ID {call_id} not found.")
        if call.ended_at is not None or call.status in ("ESCALATED", "RESOLVED"):
            raise ValueError(f"Call with ID {call_id} has already ended or has been routed to human assistance.")
        customer = call.customer

        effective_turn_id = turn_id or f"turn_{call.id}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
        print(f"[TURN {effective_turn_id}] Starting streaming turn for Call #{call.id}", flush=True)

        # 1. Transcribe audio with Faster-Whisper using company-grounded STT prompt
        stt_start_at = time.perf_counter()
        stt_prompt = self.knowledge_service.get_stt_prompt(effective_company_id) if hasattr(self.knowledge_service, "get_stt_prompt") else None
        stt_audio_path = await asyncio.to_thread(convert_to_16k_mono_wav, audio_path)
        trimmed_stt_path = await asyncio.to_thread(trim_audio_silence, stt_audio_path)
        effective_stt_path = trimmed_stt_path if trimmed_stt_path and os.path.exists(trimmed_stt_path) else stt_audio_path
        transcript, audio_duration = await asyncio.to_thread(
            self.stt_service.transcribe_with_info,
            effective_stt_path,
            None,
            stt_prompt,
            effective_company_id,
        )
        stt_end_at = time.perf_counter()
        stt_time = stt_end_at - stt_start_at

        clean_transcript = (transcript or "").strip()
        print(f"[STAGE 12] [Turn {effective_turn_id}] Faster-Whisper transcribed in {stt_time:.2f}s -> \"{clean_transcript}\" (Duration: {audio_duration:.2f}s)", flush=True)

        out_dir = os.path.dirname(output_audio_path_prefix) or "."
        base_name = os.path.basename(output_audio_path_prefix)
        if base_name.endswith(".wav"):
            base_name = base_name[:-4]
        os.makedirs(out_dir, exist_ok=True)

        # Handle empty speech / silence gracefully without hallucinating
        if not clean_transcript or clean_transcript == "[Silence / No speech detected]":
            empty_msg = "I didn't quite hear that. Please speak directly into your microphone or ask your question again."
            print(f"[LIVE CALL] [Turn {effective_turn_id}] Empty speech / silence on Call #{call.id}. Returning polite retry prompt.")
            self.conversation_service.add_message(
                db, call_id=call.id, speaker="assistant", text=empty_msg
            )
            db.commit()

            chunk_filename = f"{base_name}_chunk0.wav"
            chunk_path = os.path.join(out_dir, chunk_filename)
            await asyncio.to_thread(self.tts_service.synthesize, empty_msg, chunk_path, language="en")
            now = time.perf_counter()
            print(f"[STAGE 14] [Turn {effective_turn_id}] Piper TTS synthesized fallback chunk -> {chunk_filename}")
            yield {
                "event": "assistant_chunk",
                "turn_id": effective_turn_id,
                "call_id": call.id,
                "chunk_index": 0,
                "text": empty_msg,
                "audio_url": f"/audio/{chunk_filename}",
                "is_final": True,
                "timings": {
                    "stt_sec": round(stt_time, 3),
                    "time_to_first_text_sec": round(now - turn_start_at, 3),
                    "time_to_first_audio_sec": round(now - turn_start_at, 3),
                },
            }
            yield {
                "event": "assistant_response",
                "turn_id": effective_turn_id,
                "call_id": call.id,
                "speaker": "assistant",
                "transcript": "",
                "audio_duration": audio_duration,
                "response": empty_msg,
                "audio_url": f"/audio/{chunk_filename}",
                "chunks_count": 1,
                "is_escalated": False,
                "session_ended": False,
                "call_status": call.status,
                "empty_speech": True,
                "timings": {
                    "stt_sec": round(stt_time, 3),
                    "time_to_first_text_sec": round(now - turn_start_at, 3),
                    "time_to_first_audio_sec": round(now - turn_start_at, 3),
                    "total_turn_sec": round(now - turn_start_at, 3),
                },
            }

            turn_trace_record = {
                "trace_type": "turn_audit",
                "turn_id": effective_turn_id,
                "call_id": call.id,
                "company_id": effective_company_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "audio": {"input_path": audio_path, "duration_sec": audio_duration},
                "stt": {"clean_transcript": "", "stt_duration_sec": round(stt_time, 3)},
                "retrieval": {"knowledge_context_chars": 0, "history_context_chars": 0},
                "llm": {"full_response": empty_msg, "ttft_sec": 0.0, "ttfs_sec": 0.0, "total_llm_sec": 0.0},
                "tts": {"chunks_synthesized": 1, "first_chunk_tts_sec": round(now - turn_start_at - stt_time, 3)},
                "timings": {
                    "stt_sec": round(stt_time, 3),
                    "time_to_first_text_sec": round(now - turn_start_at, 3),
                    "time_to_first_audio_sec": round(now - turn_start_at, 3),
                    "total_turn_sec": round(now - turn_start_at, 3),
                },
                "status": {"is_resolved": False, "call_status": call.status, "is_escalated": False, "empty_speech": True},
                "error": None,
            }
            print(f"[TURN TRACE] {json.dumps(turn_trace_record)}", flush=True)
            return

        # 2. Context & Knowledge retrieval (captured from prior conversation BEFORE saving current customer turn)
        history_context = self.context_service.build_context([call], exclude_transcript=clean_transcript)
        system_instruction = self.knowledge_service.get_system_instruction(effective_company_id)
        knowledge_context = self.knowledge_service.get_relevant_knowledge(
            effective_company_id, clean_transcript, conversation_context=history_context
        )

        # 3. Save customer message to DB
        self.conversation_service.add_message(
            db, call_id=call.id, speaker="customer", text=clean_transcript
        )
        db.commit()

        # 4. Check Escalation
        should_esc, esc_reason = EscalationService.should_escalate(clean_transcript)
        if should_esc:
            EscalationService.create_escalation(
                db=db,
                call_id=call.id,
                reason=esc_reason,
                language_selected=call.language or "en",
                mark_call_ended=False,
            )
            esc_response = (
                "A live representative transfer is not available on this line. "
                "I have registered an assistance request with our front desk team. "
                "You may continue speaking with me, or contact the front desk directly for immediate assistance."
            )
            self.conversation_service.add_message(
                db, call_id=call.id, speaker="assistant", text=esc_response
            )
            db.commit()

            esc_chunk_filename = f"{base_name}_chunk0.wav"
            esc_chunk_path = os.path.join(out_dir, esc_chunk_filename)
            await asyncio.to_thread(self.tts_service.synthesize, esc_response, esc_chunk_path)

            now = time.perf_counter()
            yield {
                "event": "assistant_chunk",
                "turn_id": effective_turn_id,
                "call_id": call.id,
                "chunk_index": 0,
                "text": esc_response,
                "audio_url": f"/audio/{esc_chunk_filename}",
                "is_final": True,
                "timings": {
                    "stt_sec": round(stt_time, 3),
                    "time_to_first_text_sec": round(now - turn_start_at, 3),
                    "time_to_first_audio_sec": round(now - turn_start_at, 3),
                },
            }
            yield {
                "event": "assistant_response",
                "turn_id": effective_turn_id,
                "call_id": call.id,
                "speaker": "assistant",
                "transcript": clean_transcript,
                "audio_duration": audio_duration,
                "response": esc_response,
                "audio_url": f"/audio/{esc_chunk_filename}",
                "chunks_count": 1,
                "is_escalated": True,
                "session_ended": False,
                "call_status": "IN_PROGRESS",
                "timings": {
                    "stt_sec": round(stt_time, 3),
                    "time_to_first_text_sec": round(now - turn_start_at, 3),
                    "time_to_first_audio_sec": round(now - turn_start_at, 3),
                    "total_turn_sec": round(now - turn_start_at, 3),
                },
            }

            turn_trace_record = {
                "trace_type": "turn_audit",
                "turn_id": effective_turn_id,
                "call_id": call.id,
                "company_id": effective_company_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "audio": {"input_path": audio_path, "duration_sec": audio_duration},
                "stt": {"clean_transcript": clean_transcript, "stt_duration_sec": round(stt_time, 3)},
                "retrieval": {"knowledge_context_chars": 0, "history_context_chars": 0},
                "llm": {"full_response": esc_response, "ttft_sec": 0.0, "ttfs_sec": 0.0, "total_llm_sec": 0.0},
                "tts": {"chunks_synthesized": 1, "first_chunk_tts_sec": round(now - turn_start_at - stt_time, 3)},
                "timings": {
                    "stt_sec": round(stt_time, 3),
                    "time_to_first_text_sec": round(now - turn_start_at, 3),
                    "time_to_first_audio_sec": round(now - turn_start_at, 3),
                    "total_turn_sec": round(now - turn_start_at, 3),
                },
                "status": {"is_resolved": False, "call_status": "IN_PROGRESS", "is_escalated": True, "escalation_reason": esc_reason},
                "error": None,
            }
            print(f"[TURN TRACE] {json.dumps(turn_trace_record)}", flush=True)
            return

        # 5. Sentence streaming generation and audio chunk synthesis
        print(f"[STAGE 13] [Turn {effective_turn_id}] Ollama LLM generating response for Call #{call.id}...", flush=True)
        llm_timings = {}
        sentence_gen = self.response_service.generate_response_stream_sentences(
            transcript=clean_transcript,
            conversation_context=history_context,
            system_instruction=system_instruction,
            knowledge_context=knowledge_context,
            company_id=effective_company_id,
            timings=llm_timings,
        )

        t_first_text = None
        t_first_audio = None
        chunk_index = 0
        first_chunk_tts_dur = 0.0
        collected_sentences = []

        async for sentence in sentence_gen:
            s_clean = (sentence or "").strip()
            if not s_clean:
                continue

            now = time.perf_counter()
            if t_first_text is None:
                t_first_text = now

            chunk_filename = f"{base_name}_chunk{chunk_index}.wav"
            chunk_path = os.path.join(out_dir, chunk_filename)

            tts_t0 = time.perf_counter()
            try:
                await asyncio.to_thread(self.tts_service.synthesize, s_clean, chunk_path, language="en")
            except Exception as tts_err:
                print(f"[STREAMING TTS ERROR] [Turn {effective_turn_id}] Failed to synthesize chunk {chunk_index}: {tts_err}")
                continue
            tts_dur = time.perf_counter() - tts_t0
            if chunk_index == 0:
                first_chunk_tts_dur = tts_dur
            print(f"[STAGE 14] [Turn {effective_turn_id}] Piper TTS synthesized chunk {chunk_index} in {tts_dur:.2f}s -> \"{s_clean}\"", flush=True)

            now_audio = time.perf_counter()
            if t_first_audio is None:
                t_first_audio = now_audio

            collected_sentences.append(s_clean)

            yield {
                "event": "assistant_chunk",
                "turn_id": effective_turn_id,
                "call_id": call.id,
                "chunk_index": chunk_index,
                "text": s_clean,
                "audio_url": f"/audio/{chunk_filename}",
                "is_final": False,
                "timings": {
                    "stt_sec": round(stt_time, 3),
                    "llm_ttft_sec": round(llm_timings.get("llm_ttft", 0.0), 3),
                    "llm_ttfs_sec": round(llm_timings.get("llm_ttfs", (t_first_text - turn_start_at) if t_first_text else 0.0), 3),
                    "chunk_tts_sec": round(tts_dur, 3),
                    "time_to_first_text_sec": round(t_first_text - turn_start_at, 3),
                    "time_to_first_audio_sec": round(t_first_audio - turn_start_at, 3),
                },
            }
            chunk_index += 1

        # 6. Finalize Turn
        full_response_text = " ".join(collected_sentences).strip()
        if not full_response_text:
            full_response_text = "I apologize, but I could not process your request. Please ask again."

        # Save assistant message
        self.conversation_service.add_message(
            db, call_id=call.id, speaker="assistant", text=full_response_text
        )

        is_resolved = await self.response_service.check_resolution(clean_transcript, history_context)
        call_status = "RESOLVED" if is_resolved else "IN_PROGRESS"

        if is_resolved:
            self.conversation_service.end_call(db, call.id, status="RESOLVED")
        else:
            self.conversation_service.update_call_status(db, call.id, status="IN_PROGRESS")
        db.commit()

        # Extract and persist structured customer requirements asynchronously in background
        customer_msgs = (
            db.query(Message)
            .filter(Message.call_id == call.id, Message.speaker == "customer")
            .order_by(Message.id)
            .all()
        )
        full_customer_transcript = "\n".join(m.text for m in customer_msgs if m.text)
        if not full_customer_transcript.strip():
            full_customer_transcript = clean_transcript

        self._schedule_background_requirements(call.id, full_customer_transcript, is_resolved)

        t_final = time.perf_counter()
        total_time = t_final - turn_start_at
        first_chunk_file = f"{base_name}_chunk0.wav" if chunk_index > 0 else ""

        print(
            f"[LIVE STREAMING SUMMARY] Turn {effective_turn_id} | Call #{call.id} | Chunks: {chunk_index} | "
            f"STT: {stt_time:.3f}s | LLM TTFT: {llm_timings.get('llm_ttft', 0.0):.3f}s | "
            f"TTFS: {(t_first_text - turn_start_at if t_first_text else 0.0):.3f}s | "
            f"TTS Chunk0: {first_chunk_tts_dur:.3f}s | "
            f"TTFA: {(t_first_audio - turn_start_at if t_first_audio else 0.0):.3f}s | Total: {total_time:.3f}s",
            flush=True,
        )

        yield {
            "event": "assistant_response",
            "turn_id": effective_turn_id,
            "call_id": call.id,
            "speaker": "assistant",
            "transcript": clean_transcript,
            "audio_duration": audio_duration,
            "response": full_response_text,
            "audio_url": f"/audio/{first_chunk_file}" if first_chunk_file else "",
            "chunks_count": chunk_index,
            "is_escalated": False,
            "session_ended": is_resolved,
            "call_status": call_status,
            "timings": {
                "stt_sec": round(stt_time, 3),
                "llm_ttft_sec": round(llm_timings.get("llm_ttft", 0.0), 3),
                "llm_ttfs_sec": round(llm_timings.get("llm_ttfs", (t_first_text - turn_start_at) if t_first_text else total_time), 3),
                "llm_total_sec": round(llm_timings.get("llm_total", (t_first_text - turn_start_at) if t_first_text else total_time), 3),
                "first_chunk_tts_sec": round(first_chunk_tts_dur, 3),
                "time_to_first_text_sec": round((t_first_text - turn_start_at) if t_first_text else total_time, 3),
                "time_to_first_audio_sec": round((t_first_audio - turn_start_at) if t_first_audio else total_time, 3),
                "total_turn_sec": round(total_time, 3),
            },
        }

        turn_trace_record = {
            "trace_type": "turn_audit",
            "turn_id": effective_turn_id,
            "call_id": call.id,
            "company_id": effective_company_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "audio": {"input_path": audio_path, "duration_sec": audio_duration},
            "stt": {"clean_transcript": clean_transcript, "stt_duration_sec": round(stt_time, 3)},
            "retrieval": {
                "system_instruction_chars": len(system_instruction or ""),
                "knowledge_context_chars": len(knowledge_context or ""),
                "history_context_chars": len(history_context or ""),
            },
            "llm": {
                "full_response": full_response_text,
                "ttft_sec": round(llm_timings.get("llm_ttft", 0.0), 3),
                "ttfs_sec": round(llm_timings.get("llm_ttfs", (t_first_text - turn_start_at) if t_first_text else total_time), 3),
                "total_llm_sec": round(llm_timings.get("llm_total", (t_first_text - turn_start_at) if t_first_text else total_time), 3),
            },
            "tts": {
                "chunks_synthesized": chunk_index,
                "first_chunk_tts_sec": round(first_chunk_tts_dur, 3),
            },
            "timings": {
                "stt_sec": round(stt_time, 3),
                "llm_ttft_sec": round(llm_timings.get("llm_ttft", 0.0), 3),
                "llm_ttfs_sec": round(llm_timings.get("llm_ttfs", (t_first_text - turn_start_at) if t_first_text else total_time), 3),
                "first_chunk_tts_sec": round(first_chunk_tts_dur, 3),
                "time_to_first_text_sec": round((t_first_text - turn_start_at) if t_first_text else total_time, 3),
                "time_to_first_audio_sec": round((t_first_audio - turn_start_at) if t_first_audio else total_time, 3),
                "total_turn_sec": round(total_time, 3),
            },
            "status": {
                "is_resolved": is_resolved,
                "call_status": call_status,
                "is_escalated": False,
            },
            "error": None,
        }
        print(f"[TURN TRACE] {json.dumps(turn_trace_record)}", flush=True)

    async def process_voice_turn(
        self,
        audio_path: str,
        output_audio_path: str,
        external_customer_id: str,
        db: Session,
        language: str = "en",
        company_id: str = None,
        turn_id: Optional[str] = None,
    ) -> dict:
        """
        Processes a single-turn voice interaction with persistent database memory.
        """
        total_start = time.perf_counter()

        # a-d. Customer & Call creation + history context retrieval
        db_start = time.perf_counter()
        effective_company_id = (company_id or "").strip().lower() or "pullman_aerocity"
        customer = self.conversation_service.get_or_create_customer(
            db, external_customer_id=external_customer_id
        )
        call = self.conversation_service.create_call(db, customer.id)
        call.language = language
        db.commit()
        effective_turn_id = turn_id or f"turn_{call.id}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
        history_context = self.context_service.build_context([call])
        context_db_time = time.perf_counter() - db_start

        # e. Transcribe current audio
        stt_start = time.perf_counter()
        stt_prompt = self.knowledge_service.get_stt_prompt(effective_company_id) if hasattr(self.knowledge_service, "get_stt_prompt") else None
        stt_audio_path = convert_to_16k_mono_wav(audio_path)
        trimmed_stt_path = trim_audio_silence(stt_audio_path)
        effective_stt_path = trimmed_stt_path if trimmed_stt_path and os.path.exists(trimmed_stt_path) else stt_audio_path
        if hasattr(self.stt_service, "transcribe_with_info"):
            transcript, _ = self.stt_service.transcribe_with_info(
                effective_stt_path,
                initial_prompt=stt_prompt,
                company_id=effective_company_id,
            )
        else:
            transcript = self.stt_service.transcribe(
                stt_audio_path,
                initial_prompt=stt_prompt,
                company_id=effective_company_id,
            )
        stt_time = time.perf_counter() - stt_start

        clean_transcript = (transcript or "").strip()
        if clean_transcript:
            print(f"[STT] Call #{call.id} Transcribed: \"{clean_transcript}\" ({stt_time:.3f}s)")
        else:
            print(f"[STT] Call #{call.id} [EMPTY TRANSCRIPTION] No speech detected in audio turn ({stt_time:.3f}s)")
            response_text = "I'm sorry, I didn't catch that clearly. Could you please repeat your request?"
            self.conversation_service.add_message(
                db, call_id=call.id, speaker="customer", text=""
            )
            self.conversation_service.add_message(
                db, call_id=call.id, speaker="assistant", text=response_text
            )
            tts_start = time.perf_counter()
            generated_audio_path = self.tts_service.synthesize(response_text, output_audio_path)
            tts_time = time.perf_counter() - tts_start
            total_time = time.perf_counter() - total_start
            return {
                "turn_id": effective_turn_id,
                "transcript": "",
                "response": response_text,
                "audio_path": generated_audio_path,
                "customer_id": external_customer_id,
                "call_id": call.id,
                "call_status": call.status,
                "is_escalated": False,
                "timings": {
                    "stt_sec": round(stt_time, 2),
                    "llm_sec": 0.0,
                    "tts_sec": round(tts_time, 2),
                    "context_db_sec": round(context_db_time, 2),
                    "total_sec": round(total_time, 2),
                },
            }

        # f. Save customer transcript
        p1_start = time.perf_counter()
        self.conversation_service.add_message(
            db, call_id=call.id, speaker="customer", text=clean_transcript
        )
        p1_time = time.perf_counter() - p1_start

        # Check Human Escalation
        should_esc, esc_reason = EscalationService.should_escalate(transcript)
        if should_esc:
            EscalationService.create_escalation(
                db=db,
                call_id=call.id,
                reason=esc_reason,
                language_selected=language,
            )
            response_text = (
                "I understand your request. I am immediately connecting you to our human support team."
            )
            self.conversation_service.add_message(
                db, call_id=call.id, speaker="assistant", text=response_text
            )
            self.conversation_service.end_call(db, call.id, status="ESCALATED")

            tts_start = time.perf_counter()
            generated_audio_path = self.tts_service.synthesize(response_text, output_audio_path)
            tts_time = time.perf_counter() - tts_start

            return {
                "turn_id": effective_turn_id,
                "transcript": transcript,
                "response": response_text,
                "audio_path": generated_audio_path,
                "customer_id": customer.external_customer_id,
                "call_id": call.id,
                "call_status": "ESCALATED",
                "is_escalated": True,
            }

        # g. Generate AI response using ResponseService with history and company knowledge
        llm_start = time.perf_counter()
        system_instruction = self.knowledge_service.get_system_instruction(effective_company_id)
        knowledge_context = self.knowledge_service.get_relevant_knowledge(
            effective_company_id, transcript, conversation_context=history_context
        )

        response_text = await self.response_service.generate_response(
            transcript=transcript,
            conversation_context=history_context,
            system_instruction=system_instruction,
            knowledge_context=knowledge_context,
            company_id=effective_company_id,
        )
        llm_time = time.perf_counter() - llm_start

        # h. Save assistant response
        p2_start = time.perf_counter()
        self.conversation_service.add_message(
            db, call_id=call.id, speaker="assistant", text=response_text
        )
        p2_time = time.perf_counter() - p2_start

        # i. Generate TTS audio
        tts_start = time.perf_counter()
        generated_audio_path = self.tts_service.synthesize(response_text, output_audio_path)
        tts_time = time.perf_counter() - tts_start

        # j. End the Call
        p3_start = time.perf_counter()
        self.conversation_service.end_call(db, call.id)
        p3_time = time.perf_counter() - p3_start

        db_persistence_time = p1_time + p2_time + p3_time
        total_time = time.perf_counter() - total_start

        # Extract and persist structured customer requirements synchronously
        customer_msgs = (
            db.query(Message)
            .filter(Message.call_id == call.id, Message.speaker == "customer")
            .order_by(Message.id)
            .all()
        )
        full_customer_transcript = "\n".join(m.text for m in customer_msgs if m.text)
        if not full_customer_transcript.strip():
            full_customer_transcript = transcript

        # Extract and persist structured customer requirements asynchronously in background
        self._schedule_background_requirements(call.id, full_customer_transcript, is_resolved=True)

        self._print_performance_summary(
            stt_time=stt_time,
            context_db_time=context_db_time,
            llm_time=llm_time,
            tts_time=tts_time,
            db_persistence_time=db_persistence_time,
            total_time=total_time,
        )

        return {
            "turn_id": effective_turn_id,
            "transcript": transcript,
            "response": response_text,
            "audio_path": generated_audio_path,
            "customer_id": customer.external_customer_id,
            "call_id": call.id,
            "call_status": "RESOLVED",
            "is_escalated": False,
            "timings": {
                "stt_sec": round(stt_time, 2),
                "llm_sec": round(llm_time, 2),
                "tts_sec": round(tts_time, 2),
                "context_db_sec": round(context_db_time, 2),
                "total_sec": round(total_time, 2),
            },
        }
