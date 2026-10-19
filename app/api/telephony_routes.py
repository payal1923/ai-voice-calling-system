import os
import uuid
import json
import base64
import asyncio
import collections
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from fastapi import APIRouter, Request, Depends, HTTPException, Form, Query, Response, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import Response as FastAPIResponse
from sqlalchemy.orm import Session, joinedload

from app.database.database import get_db
from app.database.models import Customer, Call, Message, SystemUser
from app.services.conversation_service import ConversationService
from app.services.voice_response_service import VoiceResponseService
from app.services.telephony_adapter import get_telephony_adapter, TelephonyAdapter
from app.api.call_routes import handle_dtmf_routing, IVR_PROMPT_MESSAGE, ENGLISH_GREETING_MESSAGE
from app.utils.auth import get_current_user

router = APIRouter()
conversation_service = ConversationService()
voice_service = VoiceResponseService()


async def parse_webhook_payload(request: Request) -> Dict[str, Any]:
    """Extracts parameters from form data, query params, or JSON body."""
    payload: Dict[str, Any] = {}
    try:
        form = await request.form()
        for k, v in form.items():
            payload[k] = v
    except Exception:
        pass

    if not payload:
        try:
            body = await request.json()
            if isinstance(body, dict):
                payload.update(body)
        except Exception:
            pass

    if not payload:
        for k, v in request.query_params.items():
            payload[k] = v

    return payload


# ---------------------------------------------------------------------------
# Carrier Webhook Endpoints
# ---------------------------------------------------------------------------

@router.api_route("/webhook/incoming", methods=["GET", "POST"])
async def carrier_incoming_call_webhook(
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Carrier webhook endpoint for incoming calls (Twilio, Exotel, Plivo, Generic Carrier).
    Normalizes carrier parameters, auto-registers customer, creates Call session,
    and returns carrier XML/TwiML response.
    """
    payload = await parse_webhook_payload(request)
    adapter = get_telephony_adapter()
    normalized = adapter.handle_incoming_call(payload)

    customer_phone = normalized.get("customer_phone") or "+910000000000"
    ext_call_id = normalized.get("external_call_id") or "carrier_inbound"

    # 1. Get or create customer by phone/external_id
    ext_cust_id = f"TEL_{customer_phone.replace('+', '').replace(' ', '')}"
    customer = conversation_service.get_or_create_customer(
        db,
        external_customer_id=ext_cust_id,
        phone=customer_phone,
        name=f"Caller {customer_phone[-4:]}" if len(customer_phone) >= 4 else "Caller",
    )

    # 2. Create Call record in DB
    call = conversation_service.create_call(db, customer.id)
    call.call_type = "INBOUND"
    call.status = "AWAITING_LANGUAGE"
    call.language = "pending"
    db.commit()

    # 3. Store IVR Prompt in Messages table
    conversation_service.add_message(
        db,
        call_id=call.id,
        speaker="assistant",
        text=IVR_PROMPT_MESSAGE,
    )

    # 4. Generate provider XML response
    base_url = os.environ.get("TELEPHONY_WEBHOOK_BASE_URL", "http://localhost:8000").rstrip("/")
    action_url = f"{base_url}/api/telephony/webhook/dtmf?call_id={call.id}"
    xml_content = adapter.generate_carrier_response(
        action_type="ivr_prompt",
        text=IVR_PROMPT_MESSAGE,
        action_url=action_url,
    )

    return FastAPIResponse(content=xml_content, media_type="text/xml")


@router.api_route("/webhook/dtmf", methods=["GET", "POST"])
async def carrier_dtmf_webhook(
    request: Request,
    call_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    """
    Carrier webhook endpoint for DTMF keypad inputs from carrier.
    Routes language choice (1 -> English, 2 -> Hindi, 3 -> Marathi) and returns TwiML/XML.
    """
    payload = await parse_webhook_payload(request)
    adapter = get_telephony_adapter()
    normalized = adapter.handle_incoming_call(payload)

    digit_val = normalized.get("digits") or payload.get("Digits") or payload.get("digit") or payload.get("key") or "1"
    target_call_id = call_id or payload.get("call_id")

    if not target_call_id:
        # Find latest awaiting language call
        call = db.query(Call).filter(Call.status == "AWAITING_LANGUAGE").order_by(Call.id.desc()).first()
        if call:
            target_call_id = call.id

    if not target_call_id:
        xml_content = adapter.generate_carrier_response(action_type="hangup")
        return FastAPIResponse(content=xml_content, media_type="text/xml")

    call = conversation_service.get_call(db, int(target_call_id))
    routing = handle_dtmf_routing(db, call, str(digit_val))

    base_url = os.environ.get("TELEPHONY_WEBHOOK_BASE_URL", "http://localhost:8000").rstrip("/")
    if routing["is_escalated"]:
        xml_content = adapter.generate_carrier_response(
            action_type="play_and_hangup",
            text=routing["response"],
            audio_url=f"{base_url}{routing['audio_url']}",
        )
    else:
        turn_url = f"{base_url}/api/telephony/webhook/turn?call_id={call.id}"
        xml_content = adapter.generate_carrier_response(
            action_type="gather_speech",
            text=routing["response"],
            action_url=turn_url,
        )

    return FastAPIResponse(content=xml_content, media_type="text/xml")


@router.api_route("/webhook/turn", methods=["GET", "POST"])
async def carrier_speech_turn_webhook(
    request: Request,
    call_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    """
    Carrier webhook endpoint for speech recognition / turn processing.
    Processes customer speech, updates DB transcript, extracts requirements, and returns TwiML/XML.
    """
    payload = await parse_webhook_payload(request)
    adapter = get_telephony_adapter()
    normalized = adapter.handle_incoming_call(payload)

    speech_text = normalized.get("speech_text") or payload.get("SpeechResult") or "Hello, I have an inquiry."
    target_call_id = call_id or payload.get("call_id")

    if not target_call_id:
        call = db.query(Call).filter(Call.status == "IN_PROGRESS").order_by(Call.id.desc()).first()
        if call:
            target_call_id = call.id

    if not target_call_id:
        xml_content = adapter.generate_carrier_response(action_type="hangup")
        return FastAPIResponse(content=xml_content, media_type="text/xml")

    call = conversation_service.get_call(db, int(target_call_id))

    # Add customer message
    conversation_service.add_message(db, call_id=call.id, speaker="customer", text=speech_text)

    # Generate assistant response
    history_context = voice_service.context_service.build_context([call])
    response_text = await voice_service.response_service.generate_response(
        transcript=speech_text, conversation_context=history_context
    )
    conversation_service.add_message(db, call_id=call.id, speaker="assistant", text=response_text)

    # Check resolution
    is_resolved = await voice_service.response_service.check_resolution(speech_text, history_context)
    base_url = os.environ.get("TELEPHONY_WEBHOOK_BASE_URL", "http://localhost:8000").rstrip("/")
    if is_resolved:
        conversation_service.end_call(db, call.id, status="RESOLVED")
        xml_content = adapter.generate_carrier_response(
            action_type="play_and_hangup",
            text=response_text,
        )
    else:
        turn_url = f"{base_url}/api/telephony/webhook/turn?call_id={call.id}"
        xml_content = adapter.generate_carrier_response(
            action_type="gather_speech",
            text=response_text,
            action_url=turn_url,
        )

    # Update requirements synchronously
    customer_msgs = (
        db.query(Message)
        .filter(Message.call_id == call.id, Message.speaker == "customer")
        .order_by(Message.id)
        .all()
    )
    full_text = "\n".join(m.text for m in customer_msgs if m.text)
    await voice_service._update_requirements(db, call.id, full_text, is_resolved)

    return FastAPIResponse(content=xml_content, media_type="text/xml")


@router.api_route("/webhook/status", methods=["GET", "POST"])
async def carrier_status_webhook(
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Carrier callback endpoint for call status updates (completed, failed, busy, no-answer).
    """
    payload = await parse_webhook_payload(request)
    call_sid = payload.get("CallSid") or payload.get("Sid") or payload.get("call_id")
    call_status = (payload.get("CallStatus") or payload.get("status") or "completed").upper()

    if call_sid:
        call = db.query(Call).filter(Call.recording_url.like(f"%{call_sid}%")).first()
        if call and call.ended_at is None:
            new_status = "RESOLVED" if call_status in ("COMPLETED", "ANSWERED") else "FAILED"
            conversation_service.end_call(db, call.id, status=new_status)

    return {"status": "received", "call_status": call_status}


@router.api_route("/webhook/recording", methods=["GET", "POST"])
async def carrier_recording_webhook(
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Carrier callback endpoint for recording availability.
    """
    payload = await parse_webhook_payload(request)
    recording_url = payload.get("RecordingUrl") or payload.get("RecordUrl")
    call_sid = payload.get("CallSid")

    if call_sid and recording_url:
        call = db.query(Call).filter(Call.recording_url.like(f"%{call_sid}%")).first()
        if call:
            call.recording_url = recording_url
            db.commit()

    return {"status": "recording_updated", "recording_url": recording_url}


# ---------------------------------------------------------------------------
# Operator Telephony Management APIs (Authenticated)
# ---------------------------------------------------------------------------

@router.get("/config")
def get_telephony_config(
    current_user: SystemUser = Depends(get_current_user),
):
    """
    Retrieves current active telephony provider configuration status.
    """
    provider_name = os.environ.get("TELEPHONY_PROVIDER", "mock").lower()
    adapter = get_telephony_adapter(provider_name)
    is_live = provider_name != "mock"

    return {
        "telephony_provider": provider_name,
        "is_live_carrier": is_live,
        "adapter_class": adapter.__class__.__name__,
        "webhook_base_url": os.environ.get("TELEPHONY_WEBHOOK_BASE_URL", "http://localhost:8000"),
        "status": "active",
        "supported_providers": ["mock", "twilio", "exotel", "plivo"],
    }


@router.post("/outbound")
def initiate_operator_outbound_call(
    customer_id: int = Form(...),
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Initiates an outbound carrier phone call to a customer using the active telephony provider.
    """
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    adapter = get_telephony_adapter()
    call = conversation_service.create_call(db, customer.id)
    call.call_type = "OUTBOUND"
    call.status = "IN_PROGRESS"
    db.commit()

    try:
        provider_call_id = adapter.start_outbound_call(
            customer_id=customer.id,
            phone_number=customer.phone or "+910000000000",
            call_id=call.id,
        )
    except Exception as exc:
        call.status = "FAILED"
        db.commit()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initiate outbound call with telephony provider: {exc}",
        )

    call.recording_url = f"provider_call_id:{provider_call_id}"
    db.commit()

    return {
        "status": "outbound_call_initiated",
        "call_id": call.id,
        "customer_id": customer.id,
        "customer_name": customer.name,
        "customer_phone": customer.phone,
        "provider_call_id": provider_call_id,
        "telephony_provider": os.environ.get("TELEPHONY_PROVIDER", "mock"),
        "is_live_carrier": os.environ.get("TELEPHONY_PROVIDER", "mock").lower() != "mock",
    }


# ---------------------------------------------------------------------------
# Asterisk AGI Telephony Integration Bridge
# ---------------------------------------------------------------------------

def convert_to_telephony_wav(input_path: str) -> str:
    """Converts synthesized audio to 8000 Hz, 16-bit PCM mono WAV for Asterisk telephony."""
    output_path = input_path + ".telephony.wav"
    try:
        import av
        import wave
        with av.open(input_path) as in_container:
            in_stream = in_container.streams.audio[0]
            with wave.open(output_path, "wb") as out_wav:
                out_wav.setnchannels(1)
                out_wav.setsampwidth(2)
                out_wav.setframerate(8000)

                resampler = av.AudioResampler(format="s16", layout="mono", rate=8000)
                for frame in in_container.decode(in_stream):
                    for resampled in resampler.resample(frame):
                        out_wav.writeframes(resampled.to_ndarray().tobytes())
        return output_path
    except Exception as e:
        print(f"Warning: Telephony audio conversion error: {e}")
        return input_path


def sanitize_header_str(val: Any) -> str:
    """Sanitizes text to ASCII safe string for HTTP response headers, stripping newlines."""
    if not val:
        return ""
    cleaned = str(val).replace("\r", " ").replace("\n", " ")
    return cleaned.encode("ascii", "ignore").decode("ascii").strip()


@router.post("/asterisk-start")
async def asterisk_start_call_endpoint(
    customer_phone: Optional[str] = Form(None),
    language: Optional[str] = Form("en"),
    db: Session = Depends(get_db),
):
    """
    Initializes a new Asterisk call session.
    Creates a new Call in MySQL DB, stores the initial assistant greeting message,
    synthesizes 8000Hz PCM mono WAV audio greeting, and returns the audio bytes with X-Call-ID header.
    """
    phone = customer_phone or "+919999999999"
    ext_cust_id = f"TEL_{phone.replace('+', '').replace(' ', '')}"
    customer = conversation_service.get_or_create_customer(
        db,
        external_customer_id=ext_cust_id,
        phone=phone,
        name=f"Caller {phone[-4:]}" if len(phone) >= 4 else "Caller",
    )
    call = conversation_service.create_call(db, customer.id)
    call.call_type = "INBOUND"
    call.status = "IN_PROGRESS"
    call.language = language or "en"
    db.commit()

    greeting_text = "Hello! Thank you for calling. How can I help you today?"
    conversation_service.add_message(db, call_id=call.id, speaker="assistant", text=greeting_text)

    import uuid
    init_uuid = uuid.uuid4().hex[:8]
    recordings_dir = "recordings"
    os.makedirs(recordings_dir, exist_ok=True)
    raw_greeting_path = os.path.join(recordings_dir, f"asterisk_init_{init_uuid}.wav")

    try:
        voice_service.tts_service.synthesize(greeting_text, raw_greeting_path, language=language or "en")
        telephony_greeting_path = convert_to_telephony_wav(raw_greeting_path)
        with open(telephony_greeting_path, "rb") as f:
            greeting_bytes = f.read()
    except Exception as e:
        print(f"Warning: Initial greeting synthesis error: {e}")
        greeting_bytes = b""
        telephony_greeting_path = None

    for temp_f in [raw_greeting_path, telephony_greeting_path]:
        if temp_f and os.path.exists(temp_f):
            try:
                os.remove(temp_f)
            except OSError:
                pass

    print(f"[CALL_START] New Asterisk Call #{call.id} created for Customer {customer.external_customer_id} ({customer.phone})")
    print(f"[GREETING_PLAYBACK] Initial greeting rendered for Call #{call.id}: '{greeting_text}'")

    return FastAPIResponse(
        content=greeting_bytes,
        media_type="audio/wav",
        headers={
            "X-Call-ID": str(call.id),
            "X-Greeting-Text": sanitize_header_str(greeting_text),
            "X-Call-Status": "IN_PROGRESS",
        },
    )


@router.post("/asterisk-turn")
async def asterisk_speech_turn_endpoint(
    file: UploadFile = File(...),
    call_id: Optional[int] = Form(None),
    customer_phone: Optional[str] = Form(None),
    language: Optional[str] = Form("en"),
    db: Session = Depends(get_db),
):
    """
    Asterisk AGI bridge endpoint for incoming call turns.
    Receives WAV audio file from Asterisk AGI on Ubuntu, processes turn via VoiceResponseService,
    converts synthesized TTS audio to 8000Hz 16-bit mono WAV for telephony, and returns raw audio bytes.
    """
    import uuid
    import shutil
    import time

    ep_start = time.perf_counter()

    turn_uuid = uuid.uuid4().hex[:8]
    recordings_dir = "recordings"
    os.makedirs(recordings_dir, exist_ok=True)

    in_path = os.path.join(recordings_dir, f"asterisk_in_{turn_uuid}.wav")
    out_path = os.path.join(recordings_dir, f"asterisk_out_{turn_uuid}.wav")

    # 1. Save uploaded audio file to disk
    t0 = time.perf_counter()
    try:
        with open(in_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read audio payload: {e}")
    t_upload = time.perf_counter() - t0

    # 2. Get or create Call & Customer record
    t0 = time.perf_counter()
    target_call_id = call_id
    if not target_call_id:
        phone = customer_phone or "+919999999999"
        ext_cust_id = f"TEL_{phone.replace('+', '').replace(' ', '')}"
        customer = conversation_service.get_or_create_customer(
            db,
            external_customer_id=ext_cust_id,
            phone=phone,
            name=f"Caller {phone[-4:]}" if len(phone) >= 4 else "Caller",
        )
        call = conversation_service.create_call(db, customer.id)
        call.call_type = "INBOUND"
        call.status = "IN_PROGRESS"
        call.language = language or "en"
        db.commit()
        target_call_id = call.id
        print(f"[CALL_START] Auto-created Call #{target_call_id} for turn processing")
    t_db_init = time.perf_counter() - t0

    # 3. Process voice turn using existing VoiceResponseService
    t0 = time.perf_counter()
    try:
        result = await voice_service.process_existing_turn(
            call_id=int(target_call_id),
            audio_path=in_path,
            output_audio_path=out_path,
            db=db,
        )
    except Exception as e:
        if os.path.exists(in_path):
            os.remove(in_path)
        print(f"[ERROR] Call #{target_call_id} turn processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Voice turn processing failed: {e}")
    t_turn_process = time.perf_counter() - t0

    print(f"[TRANSCRIBED_TEXT] Call #{target_call_id} | Whisper STT: '{result.get('transcript')}'")
    print(f"[LLM_RESPONSE] Call #{target_call_id} | Ollama LLM: '{result.get('response')}'")

    # 4. Convert synthesized TTS output to 8000Hz PCM mono WAV for Asterisk
    t0 = time.perf_counter()
    generated_audio = result.get("audio_path", out_path)
    telephony_audio = convert_to_telephony_wav(generated_audio)
    t_conv = time.perf_counter() - t0

    # 5. Read response audio bytes
    t0 = time.perf_counter()
    try:
        with open(telephony_audio, "rb") as f:
            wav_bytes = f.read()
    except Exception as e:
        print(f"[ERROR] Call #{target_call_id} failed to read generated audio: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read generated response audio: {e}")
    t_read = time.perf_counter() - t0

    # 6. Cleanup temporary turn files asynchronously/after read
    for temp_f in [in_path, out_path, telephony_audio]:
        if temp_f and os.path.exists(temp_f) and temp_f != generated_audio:
            try:
                os.remove(temp_f)
            except OSError:
                pass

    total_ep_dur = time.perf_counter() - ep_start

    try:
        import psutil
        cpu_pct = psutil.cpu_percent(interval=None)
    except Exception:
        cpu_pct = 0.0

    timings = result.get("timings", {})
    stt_dur = timings.get('stt_sec', 0.0)
    llm_dur = timings.get('llm_sec', 0.0)
    tts_dur = timings.get('tts_sec', 0.0)
    db_dur = timings.get('context_db_sec', 0.0)
    lock_wait = timings.get('lock_wait_sec', 0.0)
    ollama_req = timings.get('ollama_req_sec', 0.0)

    # Calculate exact unexplained latency
    explained_dur = t_upload + t_db_init + stt_dur + llm_dur + tts_dur + t_conv + t_read
    unexplained_dur = max(0.0, total_ep_dur - explained_dur)

    print("=== TURN FORENSICS LOG ===")
    print(f"[FORENSICS] upload_sec: {t_upload:.3f}")
    print(f"[FORENSICS] db_init_sec: {t_db_init:.3f}")
    print(f"[FORENSICS] stt_sec: {stt_dur:.3f}")
    print(f"[FORENSICS] llm_lock_wait_sec: {lock_wait:.3f}")
    print(f"[FORENSICS] ollama_req_sec: {ollama_req:.3f}")
    print(f"[FORENSICS] llm_total_sec: {llm_dur:.3f}")
    print(f"[FORENSICS] tts_sec: {tts_dur:.3f}")
    print(f"[FORENSICS] audio_conv_sec: {t_conv:.3f}")
    print(f"[FORENSICS] db_persistence_sec: {db_dur:.3f}")
    print(f"[FORENSICS] response_read_sec: {t_read:.3f}")
    print(f"[FORENSICS] unexplained_sec: {unexplained_dur:.3f}")
    print(f"[FORENSICS] total_ep_dur_sec: {total_ep_dur:.3f}")
    print(f"[FORENSICS] cpu_utilization_pct: {cpu_pct:.1f}%")
    print("==========================")

    # 7. Return audio response with session headers
    return FastAPIResponse(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Call-ID": str(target_call_id),
            "X-Transcript": sanitize_header_str(result.get("transcript", "")),
            "X-Response-Text": sanitize_header_str(result.get("response", "")),
            "X-Call-Status": sanitize_header_str(result.get("call_status", "IN_PROGRESS")),
            "X-Is-Escalated": str(result.get("is_escalated", False)),
        },
    )


def pcm_to_wav_file(pcm_bytes: bytes, sample_rate: int, output_path: str) -> str:
    """
    Converts raw 16-bit PCM mono bytes to a standard WAV file.
    """
    import wave
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return output_path


def convert_wav_to_exotel_pcm(input_wav_path: str, target_sample_rate: int = 8000) -> bytes:
    """
    Converts a WAV audio file into raw 16-bit PCM mono bytes at target_sample_rate (default 8000 Hz).
    """
    import av
    with av.open(input_wav_path) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=target_sample_rate)
        pcm_out = bytearray()
        for frame in container.decode(stream):
            for resample_frame in resampler.resample(frame):
                pcm_out.extend(resample_frame.to_ndarray().tobytes())
        return bytes(pcm_out)


# ---------------------------------------------------------------------------
# Exotel Voice AI Turn Detection Constants & Helpers
# ---------------------------------------------------------------------------

SILENCE_THRESHOLD = 400
SILENCE_FRAMES_REQUIRED = 40  # ~800 ms of continuous silence (40 frames * 20 ms)
PREROLL_FRAMES = 10  # ~200 ms of pre-speech audio (10 frames * 20 ms)


def calculate_pcm_rms(pcm_bytes: bytes) -> float:
    """Calculates RMS energy of 16-bit signed little-endian PCM audio bytes."""
    import struct
    import math
    if not pcm_bytes or len(pcm_bytes) < 2:
        return 0.0
    num_samples = len(pcm_bytes) // 2
    samples = struct.unpack(f"<{num_samples}h", pcm_bytes[:num_samples * 2])
    if not samples:
        return 0.0
    sum_squares = sum(s * s for s in samples)
    return math.sqrt(sum_squares / num_samples)


async def process_exotel_turn(
    websocket: WebSocket,
    call_id: Any,
    pcm_bytes: bytes,
    sample_rate: int,
    call: Any,
    db: Session,
    stream_sid: Optional[str] = None,
):
    """
    Processes a completed turn: WAV conversion -> STT -> Conversation DB -> LLM -> TTS -> Webhook Media Response.
    Runs blocking operations in worker threads via asyncio.to_thread.
    """
    import os
    import uuid
    import json
    import base64
    import asyncio

    if not pcm_bytes or len(pcm_bytes) < 3200:  # Ignore audio < 200 ms
        print(f"[EXOTEL TURN WARNING] Turn audio too short ({len(pcm_bytes)} bytes), ignoring | Call ID: {call_id}")
        return

    temp_wav_path = os.path.join("recordings", f"temp_exotel_ws_{call_id}_{uuid.uuid4().hex[:8]}.wav")
    try:
        pcm_to_wav_file(pcm_bytes, sample_rate, temp_wav_path)
        print(f"[EXOTEL TURN] STT started | Call ID: {call_id}")

        stt_lang = call.language if (call and call.language and call.language != "pending") else None
        transcript = None
        try:
            transcript = await asyncio.to_thread(
                voice_service.stt_service.transcribe, temp_wav_path, language=stt_lang
            )
            print(f"[EXOTEL TURN] Transcript: {transcript}")
        except Exception as stt_err:
            print(f"[EXOTEL TURN ERROR] STT failed | Call ID: {call_id} | Error: {stt_err}")

        if transcript and transcript.strip():
            try:
                if call:
                    conversation_service.add_message(
                        db, call_id=call.id, speaker="customer", text=transcript
                    )
            except Exception as conv_err:
                print(f"[EXOTEL CONVERSATION ERROR] Call ID: {call_id} | Error: {conv_err}")

            try:
                history_context = voice_service.context_service.build_context([call]) if call else ""
                print(f"[EXOTEL TURN] LLM started | Call ID: {call_id}")
                response_text = await voice_service.response_service.generate_response(
                    transcript=transcript, conversation_context=history_context
                )
                print(f"[EXOTEL TURN] Response: {response_text}")

                try:
                    if call:
                        conversation_service.add_message(
                            db, call_id=call.id, speaker="assistant", text=response_text
                        )
                except Exception as save_asst_err:
                    print(f"[EXOTEL CONVERSATION ERROR] Call ID: {call_id} | Error: {save_asst_err}")

                if response_text and response_text.strip():
                    temp_tts_wav = os.path.join("recordings", f"temp_exotel_tts_{call_id}_{uuid.uuid4().hex[:8]}.wav")
                    try:
                        print(f"[EXOTEL TURN] TTS started | Call ID: {call_id}")
                        tts_lang = call.language if (call and call.language and call.language != "pending") else "en"
                        
                        await asyncio.to_thread(
                            voice_service.tts_service.synthesize, response_text, temp_tts_wav, language=tts_lang
                        )

                        try:
                            tts_pcm_bytes = await asyncio.to_thread(
                                convert_wav_to_exotel_pcm, temp_tts_wav, target_sample_rate=sample_rate
                            )
                            total_pcm_len = len(tts_pcm_bytes)
                            print(f"[EXOTEL TURN] Sending audio | Call ID: {call_id}")

                            chunk_size = 640
                            for i in range(0, total_pcm_len, chunk_size):
                                pcm_chunk = tts_pcm_bytes[i:i + chunk_size]
                                payload_b64 = base64.b64encode(pcm_chunk).decode("utf-8")
                                media_payload = {"payload": payload_b64}
                                media_msg = {
                                    "event": "media",
                                    "media": media_payload,
                                }
                                if stream_sid and stream_sid != "N/A":
                                    media_payload["streamSid"] = stream_sid
                                    media_payload["stream_sid"] = stream_sid
                                    media_msg["streamSid"] = stream_sid
                                    media_msg["stream_sid"] = stream_sid
                                await websocket.send_text(json.dumps(media_msg))

                        except Exception as audio_err:
                            print(f"[EXOTEL AUDIO ERROR] Call ID: {call_id} | Error: {audio_err}")

                    except Exception as tts_err:
                        print(f"[EXOTEL TURN ERROR] TTS failed | Call ID: {call_id} | Error: {tts_err}")
                    finally:
                        if os.path.exists(temp_tts_wav):
                            try:
                                os.remove(temp_tts_wav)
                            except Exception:
                                pass

            except Exception as llm_err:
                print(f"[EXOTEL TURN ERROR] LLM failed | Call ID: {call_id} | Error: {llm_err}")

    except Exception as turn_err:
        print(f"[EXOTEL TURN ERROR] Unexpected turn processing error | Call ID: {call_id} | Error: {turn_err}")
    finally:
        if os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Exotel Flow Builder Dynamic Stream URL Webhook
# ---------------------------------------------------------------------------

@router.api_route("/webhook/stream-url", methods=["GET", "POST"])
async def exotel_stream_url_webhook(
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Exotel Flow Builder Dynamic Stream URL webhook endpoint.
    Creates an inbound Call record and returns JSON {"url": "wss://.../api/telephony/ws/telephony/<call_id>"}.
    """
    payload = await parse_webhook_payload(request)
    adapter = get_telephony_adapter()
    normalized = adapter.handle_incoming_call(payload)
    customer_phone = normalized.get("customer_phone") or "+910000000000"
    ext_call_id = normalized.get("external_call_id") or "carrier_inbound"
    ext_cust_id = f"TEL_{customer_phone.replace('+', '').replace(' ', '')}"

    customer = conversation_service.get_or_create_customer(
        db,
        external_customer_id=ext_cust_id,
        phone=customer_phone,
        name=f"Caller {customer_phone[-4:]}" if len(customer_phone) >= 4 else "Caller",
    )
    call = conversation_service.create_call(db, customer.id)
    call.call_type = "INBOUND"
    call.status = "IN_PROGRESS"
    call.language = "en"
    call.recording_url = f"provider_call_id:{ext_call_id}"
    db.commit()

    base_url = os.environ.get("TELEPHONY_WEBHOOK_BASE_URL", "http://localhost:8000")
    ws_base = base_url.replace("http://", "ws://").replace("https://", "wss://").rstrip("/")
    stream_url = f"{ws_base}/api/telephony/ws/telephony/{call.id}"
    print(f"[EXOTEL STREAM-URL] Returning dynamic stream URL for Call #{call.id}: {stream_url}")
    return {"url": stream_url}


# ---------------------------------------------------------------------------
# Exotel Voice AI WebSocket Streaming Endpoint
# ---------------------------------------------------------------------------

@router.websocket("/ws/telephony/inbound")
@router.websocket("/ws/telephony/{call_id}")
async def exotel_telephony_websocket(
    websocket: WebSocket,
    call_id: Optional[str] = "inbound",
    db: Session = Depends(get_db),
):
    """
    WebSocket endpoint for Exotel Connect Voice AI bidirectional audio streaming.
    Buffers incoming PCM audio, performs real-time speech/silence turn detection,
    transcribes turns with STT, generates LLM response, synthesizes TTS audio,
    and streams raw PCM audio back over the WebSocket.
    Supports both pre-allocated Call IDs (outbound) and dynamic inbound sessions.
    """
    import json
    import base64
    import collections

    call = None
    is_inbound = (call_id is None or str(call_id).lower() in ("inbound", "stream", "0", "new"))

    if not is_inbound:
        try:
            numeric_id = int(str(call_id))
            call = conversation_service.get_call(db, numeric_id)
        except Exception:
            call = None

        if not call:
            print(f"[EXOTEL WS REJECTED] Call ID {call_id} not found in database.")
            await websocket.accept()
            await websocket.close(code=1008, reason="Call ID not found")
            return

    await websocket.accept()
    print(f"[EXOTEL WS] Connected | Call ID: {call_id}")

    pcm_buffer = bytearray()
    preroll_buffer = collections.deque(maxlen=PREROLL_FRAMES)
    speech_detected = False
    silence_frames_count = 0
    sample_rate = 8000
    stream_sid = "N/A"
    call_sid = "N/A"
    resolved_call_id = call.id if call else call_id

    greeting_sent = False
    greeting_task: Optional[asyncio.Task] = None

    async def send_inbound_greeting(target_stream_sid: str, target_sample_rate: int):
        nonlocal greeting_task
        temp_tts_wav = os.path.join("recordings", f"temp_exotel_greeting_{resolved_call_id}_{uuid.uuid4().hex[:8]}.wav")
        try:
            print(f"[EXOTEL GREETING] Synthesizing greeting | Call ID: {resolved_call_id}")
            tts_lang = call.language if (call and call.language and call.language != "pending") else "en"
            try:
                await asyncio.to_thread(
                    voice_service.tts_service.synthesize, ENGLISH_GREETING_MESSAGE, temp_tts_wav, language=tts_lang
                )
                wav_to_convert = temp_tts_wav
            except Exception as synth_err:
                print(f"[EXOTEL GREETING WARNING] TTS synthesis error, attempting fallback to static greeting: {synth_err}")
                if os.path.exists("recordings/en_greeting.wav"):
                    wav_to_convert = "recordings/en_greeting.wav"
                else:
                    raise

            tts_pcm_bytes = await asyncio.to_thread(
                convert_wav_to_exotel_pcm, wav_to_convert, target_sample_rate=target_sample_rate
            )
            total_pcm_len = len(tts_pcm_bytes)
            print(f"[EXOTEL GREETING] Streaming greeting ({total_pcm_len} bytes) | Call ID: {resolved_call_id} | Stream SID: {target_stream_sid}")

            chunk_size = 640
            for i in range(0, total_pcm_len, chunk_size):
                pcm_chunk = tts_pcm_bytes[i:i + chunk_size]
                payload_b64 = base64.b64encode(pcm_chunk).decode("utf-8")
                media_payload = {"payload": payload_b64}
                media_msg = {
                    "event": "media",
                    "media": media_payload,
                }
                if target_stream_sid and target_stream_sid != "N/A":
                    media_payload["streamSid"] = target_stream_sid
                    media_payload["stream_sid"] = target_stream_sid
                    media_msg["streamSid"] = target_stream_sid
                    media_msg["stream_sid"] = target_stream_sid
                try:
                    await websocket.send_text(json.dumps(media_msg))
                except (WebSocketDisconnect, RuntimeError):
                    print(f"[EXOTEL GREETING] WebSocket disconnected while sending audio | Call ID: {resolved_call_id}")
                    return
                await asyncio.sleep(0.02)

            print(f"[EXOTEL GREETING] Greeting streaming completed | Call ID: {resolved_call_id}")
        except asyncio.CancelledError:
            print(f"[EXOTEL GREETING] Greeting streaming cancelled (barge-in or disconnect) | Call ID: {resolved_call_id}")
            raise
        except Exception as greet_err:
            print(f"[EXOTEL GREETING ERROR] Failed sending greeting audio | Call ID: {resolved_call_id} | Error: {greet_err}")
        finally:
            if os.path.exists(temp_tts_wav):
                try:
                    os.remove(temp_tts_wav)
                except Exception:
                    pass

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                print(f"[EXOTEL WS] Disconnected | Call ID: {resolved_call_id}")
                if greeting_task and not greeting_task.done():
                    greeting_task.cancel()
                break

            raw_text = message.get("text")
            if not raw_text and message.get("bytes"):
                try:
                    raw_text = message["bytes"].decode("utf-8")
                except Exception:
                    print(f"[EXOTEL WS ERROR] Raw binary payload non-UTF8 | Call ID: {resolved_call_id}")
                    continue

            if not raw_text:
                continue

            try:
                data = json.loads(raw_text)
            except Exception:
                print(f"[EXOTEL WS ERROR] Invalid JSON | Call ID: {resolved_call_id}")
                continue

            if not isinstance(data, dict):
                print(f"[EXOTEL WS ERROR] Invalid JSON | Call ID: {resolved_call_id}")
                continue

            event_type = str(data.get("event") or data.get("event_type") or data.get("type") or "").lower().strip()

            if event_type == "connected":
                print(f"[EXOTEL WS] Event: connected | Call ID: {resolved_call_id}")

            elif event_type == "start":
                start_obj = data.get("start") if isinstance(data.get("start"), dict) else data
                stream_sid = start_obj.get("streamSid") or start_obj.get("stream_sid") or "N/A"
                call_sid = start_obj.get("callSid") or start_obj.get("call_sid") or start_obj.get("accountSid") or "N/A"
                media_fmt = start_obj.get("mediaFormat") or start_obj.get("media_format") if isinstance(start_obj.get("mediaFormat") or start_obj.get("media_format"), dict) else {}
                sr_val = media_fmt.get("sampleRate") or media_fmt.get("sample_rate") or start_obj.get("sampleRate") or start_obj.get("sample_rate")
                if sr_val:
                    try:
                        sample_rate = int(str(sr_val).replace("Hz", "").strip())
                    except ValueError:
                        sample_rate = 8000

                # If this is an inbound connection without a pre-existing Call record, auto-create it now
                if call is None:
                    custom_params = start_obj.get("customParameters") if isinstance(start_obj.get("customParameters"), dict) else {}
                    customer_phone = custom_params.get("from") or custom_params.get("caller_id") or "+910000000000"
                    ext_cust_id = f"TEL_{str(customer_phone).replace('+', '').replace(' ', '')}"
                    try:
                        customer = conversation_service.get_or_create_customer(
                            db,
                            external_customer_id=ext_cust_id,
                            phone=customer_phone,
                            name=f"Caller {str(customer_phone)[-4:]}" if len(str(customer_phone)) >= 4 else "Inbound Caller",
                        )
                        call = conversation_service.create_call(db, customer.id)
                        call.call_type = "INBOUND"
                        call.status = "IN_PROGRESS"
                        call.language = "en"
                        call.recording_url = f"provider_call_id:{call_sid}"
                        db.commit()
                        resolved_call_id = call.id
                        print(f"[EXOTEL WS] Auto-created inbound Call #{resolved_call_id} for Call SID: {call_sid}")
                    except Exception as db_err:
                        print(f"[EXOTEL WS ERROR] Failed to auto-create inbound call record: {db_err}")

                print(f"[EXOTEL WS] Event: start | Call ID: {resolved_call_id} | Stream SID: {stream_sid} | Call SID: {call_sid} | Sample Rate: {sample_rate} Hz")

                # Inbound Call Greeting Initiation (synthesize and stream exactly once for inbound session)
                is_inbound_session = is_inbound or (call and getattr(call, "call_type", None) == "INBOUND")
                if is_inbound_session and not greeting_sent:
                    greeting_sent = True

                    # Verify no duplicate greeting message in database
                    has_db_greeting = False
                    if call and hasattr(call, "messages") and call.messages:
                        for m in call.messages:
                            if getattr(m, "speaker", None) == "assistant" and ENGLISH_GREETING_MESSAGE in (getattr(m, "text", "") or ""):
                                has_db_greeting = True
                                break

                    if not has_db_greeting and call:
                        try:
                            conversation_service.add_message(
                                db,
                                call_id=call.id,
                                speaker="assistant",
                                text=ENGLISH_GREETING_MESSAGE,
                            )
                            print(f"[EXOTEL WS] Logged initial greeting message for Call #{call.id}")
                        except Exception as msg_err:
                            print(f"[EXOTEL WS ERROR] Failed logging greeting message: {msg_err}")

                    greeting_task = asyncio.create_task(send_inbound_greeting(stream_sid, sample_rate))

            elif event_type == "media":
                media_obj = data.get("media")
                payload_b64 = media_obj.get("payload") if isinstance(media_obj, dict) else None
                if not payload_b64 and isinstance(data.get("payload"), str):
                    payload_b64 = data.get("payload")

                if payload_b64:
                    try:
                        decoded_bytes = base64.b64decode(payload_b64)
                        rms = calculate_pcm_rms(decoded_bytes)

                        if not speech_detected:
                            if rms >= SILENCE_THRESHOLD:
                                speech_detected = True
                                silence_frames_count = 0
                                pcm_buffer = bytearray()
                                for p_chunk in preroll_buffer:
                                    pcm_buffer.extend(p_chunk)
                                pcm_buffer.extend(decoded_bytes)
                                preroll_buffer.clear()
                                print(f"[EXOTEL TURN] Speech started | Call ID: {resolved_call_id}")

                                # Caller interruption / barge-in: cancel greeting playback immediately
                                if greeting_task and not greeting_task.done():
                                    print(f"[EXOTEL GREETING] Caller interruption detected; cancelling greeting playback | Call ID: {resolved_call_id}")
                                    greeting_task.cancel()
                            else:
                                preroll_buffer.append(decoded_bytes)
                        else:
                            pcm_buffer.extend(decoded_bytes)
                            if rms < SILENCE_THRESHOLD:
                                silence_frames_count += 1
                                if silence_frames_count % 10 == 0:
                                    print(f"[EXOTEL TURN] Silence detected | Call ID: {resolved_call_id} | Duration: {silence_frames_count * 20} ms")
                                if silence_frames_count >= SILENCE_FRAMES_REQUIRED:
                                    silence_duration_ms = silence_frames_count * 20
                                    print(f"[EXOTEL TURN] Processing turn | Call ID: {resolved_call_id} | Audio Bytes: {len(pcm_buffer)}")

                                    # Trim trailing silence (leave ~100 ms of trailing silence)
                                    trim_bytes = max(0, (silence_frames_count - 5) * 320)
                                    if trim_bytes > 0 and len(pcm_buffer) > trim_bytes:
                                        turn_pcm = bytes(pcm_buffer[:-trim_bytes])
                                    else:
                                        turn_pcm = bytes(pcm_buffer)

                                    # Reset turn state before starting async turn processing
                                    pcm_buffer = bytearray()
                                    speech_detected = False
                                    silence_frames_count = 0
                                    preroll_buffer.clear()
                                    print(f"[EXOTEL TURN] Turn state reset | Call ID: {resolved_call_id}")

                                    # Process turn (STT -> LLM -> TTS -> Send Audio)
                                    await process_exotel_turn(websocket, resolved_call_id, turn_pcm, sample_rate, call, db, stream_sid)
                                    print(f"[EXOTEL TURN] Turn completed | Call ID: {resolved_call_id}")
                            else:
                                silence_frames_count = 0

                    except Exception as b64_err:
                        print(f"[EXOTEL WS WARNING] Malformed base64 media payload | Call ID: {resolved_call_id} | Error: {b64_err}")
                else:
                    print(f"[EXOTEL WS WARNING] Missing media payload in media event | Call ID: {resolved_call_id}")

            elif event_type == "dtmf":
                digit = data.get("dtmf") or data.get("digit") or data.get("digit_event") or "N/A"
                print(f"[EXOTEL WS] Event: dtmf | Call ID: {resolved_call_id} | Digit: {digit}")

            elif event_type == "mark":
                mark_obj = data.get("mark")
                mark_name = mark_obj.get("name") if isinstance(mark_obj, dict) else data.get("name", "N/A")
                print(f"[EXOTEL WS] Event: mark | Call ID: {resolved_call_id} | Mark Name: {mark_name}")

            elif event_type == "stop":
                if greeting_task and not greeting_task.done():
                    greeting_task.cancel()
                stop_obj = data.get("stop") if isinstance(data.get("stop"), dict) else data
                stop_stream_sid = stop_obj.get("streamSid") or stop_obj.get("stream_sid") or "N/A"
                print(f"[EXOTEL WS] Event: stop | Call ID: {resolved_call_id} | Stream SID: {stop_stream_sid}")
                
                pcm_buffer = bytearray()
                speech_detected = False
                silence_frames_count = 0
                preroll_buffer.clear()
                print(f"[EXOTEL WS] Session cleanup complete for stop event | Call ID: {resolved_call_id}")

            else:
                print(f"[EXOTEL WS] Event: unknown | Call ID: {resolved_call_id} | Type: {event_type or 'unspecified'}")

    except WebSocketDisconnect:
        print(f"[EXOTEL WS] Disconnected | Call ID: {resolved_call_id}")
    except Exception as e:
        print(f"[EXOTEL WS ERROR] Call ID: {resolved_call_id} | Error: {e}")
    finally:
        if greeting_task and not greeting_task.done():
            greeting_task.cancel()




