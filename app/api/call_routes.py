import os
import uuid
import json
import base64
import asyncio
import shutil
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form, Request, Query, WebSocket, WebSocketDisconnect, status
from sqlalchemy.orm import Session, joinedload

from app.services.voice_response_service import VoiceResponseService
from app.services.conversation_service import ConversationService
from app.services.escalation_service import EscalationService
from app.services.requirement_service import RequirementService
from app.database.database import get_db
from app.database.models import Customer, Call, Message, Requirement, SystemUser
from app.utils.auth import get_current_user, decode_access_token
from app.utils.audio_utils import convert_to_16k_mono_wav

router = APIRouter()
voice_service = VoiceResponseService()
conversation_service = ConversationService()
requirement_service = RequirementService()

RECORDINGS_DIR = "recordings"
GENERATED_AUDIO_DIR = "recordings"

IVR_PROMPT_MESSAGE = (
    "Press 1 for English. Press 2 for Hindi. Press 3 for Marathi."
)
ENGLISH_GREETING_MESSAGE = (
    "Hello! I am your AI voice assistant. How can I help you today?"
)
HINDI_ASSISTANCE_MESSAGE = (
    "हमारी टीम का एक सदस्य जल्द ही आपसे संपर्क करेगा। कृपया कुछ समय बाद हमारे कॉल का इंतजार करें। धन्यवाद।"
)
MARATHI_ASSISTANCE_MESSAGE = (
    "आमच्या टीममधील एक सदस्य लवकरच आपल्याशी संपर्क साधेल. कृपया काही वेळाने आमच्या कॉलची वाट पाहा. धन्यवाद."
)


# ---------------------------------------------------------------------------
# Customers Endpoints (Authenticated)
# ---------------------------------------------------------------------------

@router.get("/customers")
def get_customers(
    q: Optional[str] = Query(None, description="Search query for customer name, phone, or ID"),
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve actual customer records from MySQL database with optional search filtering."""
    try:
        query = db.query(Customer).options(joinedload(Customer.calls)).order_by(Customer.id.desc())
        if q and q.strip():
            search_term = f"%{q.strip()}%"
            query = query.filter(
                (Customer.name.ilike(search_term))
                | (Customer.phone.ilike(search_term))
                | (Customer.external_customer_id.ilike(search_term))
            )
        customers = query.all()
        return [
            {
                "id": c.id,
                "external_customer_id": c.external_customer_id,
                "name": c.name or "N/A",
                "phone": c.phone or "N/A",
                "created_at": c.created_at.isoformat() if c.created_at else None,
                "call_count": len(c.calls),
            }
            for c in customers
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/customers")
def create_customer(
    external_customer_id: str = Form(...),
    name: str = Form(...),
    phone: str = Form(...),
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new customer in MySQL database."""
    try:
        ext_id = external_customer_id.strip()
        existing = db.query(Customer).filter(Customer.external_customer_id == ext_id).first()
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"Customer with ID '{ext_id}' already exists",
            )

        customer = conversation_service.get_or_create_customer(
            db,
            external_customer_id=ext_id,
            name=name.strip(),
            phone=phone.strip(),
        )
        return {
            "id": customer.id,
            "external_customer_id": customer.external_customer_id,
            "name": customer.name,
            "phone": customer.phone,
            "created_at": customer.created_at.isoformat() if customer.created_at else None,
            "call_count": len(customer.calls),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/customers/{customer_id}")
def update_customer(
    customer_id: int,
    external_customer_id: str = Form(...),
    name: str = Form(...),
    phone: str = Form(...),
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update existing customer details in MySQL database."""
    try:
        customer = db.query(Customer).filter(Customer.id == customer_id).first()
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")

        ext_id = external_customer_id.strip()
        if ext_id != customer.external_customer_id:
            existing = db.query(Customer).filter(Customer.external_customer_id == ext_id).first()
            if existing:
                raise HTTPException(
                    status_code=400,
                    detail=f"External customer ID '{ext_id}' is already assigned to another customer.",
                )
            customer.external_customer_id = ext_id

        customer.name = name.strip()
        customer.phone = phone.strip()
        db.commit()
        db.refresh(customer)

        return {
            "id": customer.id,
            "external_customer_id": customer.external_customer_id,
            "name": customer.name,
            "phone": customer.phone,
            "created_at": customer.created_at.isoformat() if customer.created_at else None,
            "call_count": len(customer.calls),
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/customers/{customer_id}")
def delete_customer(
    customer_id: int,
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a customer and all associated records from MySQL database."""
    try:
        customer = db.query(Customer).filter(Customer.id == customer_id).first()
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")

        db.delete(customer)
        db.commit()
        return {"status": "deleted", "customer_id": customer_id}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/customers/{customer_id}/calls")
def get_customer_calls(
    customer_id: int,
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve call history for a specific customer."""
    try:
        calls = conversation_service.get_call_history(db, customer_id)
        return [
            {
                "id": c.id,
                "status": c.status,
                "call_type": c.call_type,
                "language": c.language,
                "started_at": c.started_at.isoformat() if c.started_at else None,
                "ended_at": c.ended_at.isoformat() if c.ended_at else None,
                "duration_seconds": round((c.ended_at - c.started_at).total_seconds()) if c.ended_at and c.started_at else None,
                "message_count": len(c.messages),
                "messages": [
                    {
                        "speaker": msg.speaker,
                        "text": msg.text,
                        "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                    }
                    for msg in c.messages
                ],
            }
            for c in calls
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Calls & Call History Endpoints (Authenticated)
# ---------------------------------------------------------------------------

@router.get("/calls")
def get_all_calls(
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve all call history sessions ordered chronologically with customer details."""
    try:
        calls = (
            db.query(Call)
            .options(joinedload(Call.customer), joinedload(Call.messages), joinedload(Call.requirement))
            .order_by(Call.started_at.desc())
            .all()
        )
        result = []
        for c in calls:
            cust = c.customer
            duration = round((c.ended_at - c.started_at).total_seconds()) if c.ended_at and c.started_at else None
            result.append({
                "id": c.id,
                "customer_id": cust.id if cust else None,
                "customer_name": cust.name if cust else "Unknown",
                "customer_phone": cust.phone if cust else "N/A",
                "external_customer_id": cust.external_customer_id if cust else "N/A",
                "status": c.status,
                "call_type": c.call_type,
                "language": c.language or "en",
                "started_at": c.started_at.isoformat() if c.started_at else None,
                "ended_at": c.ended_at.isoformat() if c.ended_at else None,
                "duration_seconds": duration,
                "message_count": len(c.messages),
                "has_requirements": c.requirement is not None,
            })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def handle_dtmf_routing(db: Session, call: Call, digit_or_lang: str) -> dict:
    """Core logic for DTMF / keypad / language routing."""
    choice = str(digit_or_lang or "").strip().lower()

    if choice in ["2", "hi", "hindi"]:
        call.language = "Hindi"
        call.status = "ESCALATED"
        call.ended_at = datetime.now(timezone.utc)

        # 1. Record customer DTMF choice in messages
        conversation_service.add_message(
            db,
            call_id=call.id,
            speaker="customer",
            text="[DTMF Keypad: 2 - Hindi]",
        )
        # 2. Record native Hindi notice in messages
        conversation_service.add_message(
            db,
            call_id=call.id,
            speaker="assistant",
            text=HINDI_ASSISTANCE_MESSAGE,
        )
        # 3. Create Human Assistance request
        EscalationService.create_escalation(
            db=db,
            call_id=call.id,
            reason="Customer selected Hindi",
            language_selected="Hindi",
            status="waiting",
            notes="Customer selected Hindi via language selection prompt. Routed to Human Assistance queue.",
        )
        db.commit()

        # Ensure native audio file exists on disk
        hi_path = os.path.join(RECORDINGS_DIR, "hi_notice.wav")
        if not os.path.exists(hi_path):
            try:
                voice_service.tts_service.synthesize(HINDI_ASSISTANCE_MESSAGE, hi_path, language="hi")
            except Exception as e:
                print(f"Warning generating Hindi audio: {e}")

        return {
            "call_id": call.id,
            "language": "Hindi",
            "status": "ESCALATED",
            "is_escalated": True,
            "escalation_reason": "Customer selected Hindi",
            "audio_url": "/audio/hi_notice.wav",
            "response": HINDI_ASSISTANCE_MESSAGE,
        }

    elif choice in ["3", "mr", "marathi"]:
        call.language = "Marathi"
        call.status = "ESCALATED"
        call.ended_at = datetime.now(timezone.utc)

        # 1. Record customer DTMF choice in messages
        conversation_service.add_message(
            db,
            call_id=call.id,
            speaker="customer",
            text="[DTMF Keypad: 3 - Marathi]",
        )
        # 2. Record native Marathi notice in messages
        conversation_service.add_message(
            db,
            call_id=call.id,
            speaker="assistant",
            text=MARATHI_ASSISTANCE_MESSAGE,
        )
        # 3. Create Human Assistance request
        EscalationService.create_escalation(
            db=db,
            call_id=call.id,
            reason="Customer selected Marathi",
            language_selected="Marathi",
            status="waiting",
            notes="Customer selected Marathi via language selection prompt. Routed to Human Assistance queue.",
        )
        db.commit()

        # Ensure native audio file exists on disk
        mr_path = os.path.join(RECORDINGS_DIR, "mr_notice.wav")
        if not os.path.exists(mr_path):
            try:
                voice_service.tts_service.synthesize(MARATHI_ASSISTANCE_MESSAGE, mr_path, language="mr")
            except Exception as e:
                print(f"Warning generating Marathi audio: {e}")

        return {
            "call_id": call.id,
            "language": "Marathi",
            "status": "ESCALATED",
            "is_escalated": True,
            "escalation_reason": "Customer selected Marathi",
            "audio_url": "/audio/mr_notice.wav",
            "response": MARATHI_ASSISTANCE_MESSAGE,
        }

    else:
        # Option 1: English
        call.language = "en"
        call.status = "IN_PROGRESS"

        # 1. Record customer DTMF choice in messages
        conversation_service.add_message(
            db,
            call_id=call.id,
            speaker="customer",
            text="[DTMF Keypad: 1 - English]",
        )
        # 2. Record assistant greeting in messages
        conversation_service.add_message(
            db,
            call_id=call.id,
            speaker="assistant",
            text=ENGLISH_GREETING_MESSAGE,
        )
        db.commit()

        return {
            "call_id": call.id,
            "language": "en",
            "status": "IN_PROGRESS",
            "is_escalated": False,
            "escalation_reason": None,
            "audio_url": "/audio/en_greeting.wav",
            "response": ENGLISH_GREETING_MESSAGE,
        }


@router.post("/calls")
def create_call(
    external_customer_id: str = Form(...),
    language: str = Form("prompt"),
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Create a new Call session for a customer.
    By default (or language='prompt'/'pending'), the call starts by asking the customer to select a language:
    Press 1 for English.
    Press 2 for Hindi.
    Press 3 for Marathi.
    """
    try:
        customer = conversation_service.get_or_create_customer(db, external_customer_id=external_customer_id)
        call = conversation_service.create_call(db, customer.id)

        lang_input = str(language or "").strip().lower()

        if lang_input in ["prompt", "pending", "none", ""]:
            # System asks customer to choose language
            call.status = "AWAITING_LANGUAGE"
            call.language = "pending"
            db.commit()

            # Record IVR prompt in MySQL messages table
            conversation_service.add_message(
                db,
                call_id=call.id,
                speaker="assistant",
                text=IVR_PROMPT_MESSAGE,
            )

            return {
                "call_id": call.id,
                "customer_id": customer.id,
                "external_customer_id": customer.external_customer_id,
                "name": customer.name or "N/A",
                "phone": customer.phone or "N/A",
                "language": "pending",
                "status": "AWAITING_LANGUAGE",
                "is_escalated": False,
                "escalation_reason": None,
                "audio_url": "/audio/ivr_prompt.wav",
                "response": IVR_PROMPT_MESSAGE,
                "started_at": call.started_at.isoformat() if call.started_at else None,
                "ended_at": None,
            }
        else:
            # Caller already provided language input (1, 2, or 3)
            routing = handle_dtmf_routing(db, call, lang_input)
            return {
                "call_id": call.id,
                "customer_id": customer.id,
                "external_customer_id": customer.external_customer_id,
                "name": customer.name or "N/A",
                "phone": customer.phone or "N/A",
                "language": routing["language"],
                "status": routing["status"],
                "is_escalated": routing["is_escalated"],
                "escalation_reason": routing.get("escalation_reason"),
                "audio_url": routing["audio_url"],
                "response": routing["response"],
                "started_at": call.started_at.isoformat() if call.started_at else None,
                "ended_at": call.ended_at.isoformat() if call.ended_at else None,
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calls/{call_id}/dtmf")
@router.post("/calls/{call_id}/select-language")
async def process_dtmf_or_language_selection(
    call_id: int,
    request: Request,
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Unified DTMF / Language Selection endpoint.
    Option 1 -> English AI conversation continues.
    Option 2 -> Hindi voice notice played and customer queued to Human Assistance.
    Option 3 -> Marathi voice notice played and customer queued to Human Assistance.
    """
    try:
        call = conversation_service.get_call(db, call_id)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))

    # Read digit/key/language from form or json
    digit_val = None
    try:
        form = await request.form()
        digit_val = form.get("digit") or form.get("key") or form.get("language")
    except Exception:
        pass

    if not digit_val:
        try:
            body = await request.json()
            if isinstance(body, dict):
                digit_val = body.get("digit") or body.get("key") or body.get("language")
        except Exception:
            pass

    if not digit_val:
        digit_val = "1"

    return handle_dtmf_routing(db, call, str(digit_val))


@router.get("/calls/{call_id}")
async def get_call_details(
    call_id: int,
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Retrieve comprehensive details of a specific call session:
    Customer Information -> Call Information -> Full Transcript -> Requirements Collected.
    """
    try:
        call = (
            db.query(Call)
            .options(
                joinedload(Call.customer),
                joinedload(Call.messages),
                joinedload(Call.requirement),
                joinedload(Call.escalations),
            )
            .filter(Call.id == call_id)
            .first()
        )
        if not call:
            raise HTTPException(status_code=404, detail="Call session not found")

        customer = call.customer
        sorted_messages = sorted(
            call.messages,
            key=lambda m: (m.timestamp or m.id, m.id or 0),
        )

        # Retrieve requirements
        requirements_data = None
        if call.requirement:
            requirements_data = {
                "id": call.requirement.id,
                "status": call.requirement.status,
                "structured": call.requirement.structured_requirements,
                "confirmed_at": call.requirement.confirmed_at.isoformat() if call.requirement.confirmed_at else None,
            }
        elif len(sorted_messages) > 0:
            # If transcript exists but requirements not yet saved, extract and persist now
            full_text = "\n".join(m.text for m in sorted_messages if m.speaker == "customer")
            if full_text.strip():
                try:
                    await voice_service._update_requirements(db, call.id, full_text, is_resolved=True)
                    db.refresh(call)
                    if call.requirement:
                        requirements_data = {
                            "id": call.requirement.id,
                            "status": call.requirement.status,
                            "structured": call.requirement.structured_requirements,
                            "confirmed_at": None,
                        }
                except Exception as ex:
                    print(f"Error in on-demand requirement extraction: {ex}")

        duration = (
            round((call.ended_at - call.started_at).total_seconds())
            if call.ended_at and call.started_at
            else None
        )

        return {
            "call_id": call.id,
            "customer": {
                "id": customer.id if customer else None,
                "external_customer_id": customer.external_customer_id if customer else "N/A",
                "name": customer.name if customer else "Unknown",
                "phone": customer.phone if customer else "N/A",
                "created_at": customer.created_at.isoformat() if customer and customer.created_at else None,
            },
            "call": {
                "id": call.id,
                "status": call.status,
                "call_type": call.call_type,
                "language": call.language or "en",
                "started_at": call.started_at.isoformat() if call.started_at else None,
                "ended_at": call.ended_at.isoformat() if call.ended_at else None,
                "duration_seconds": duration,
                "is_escalated": len(call.escalations) > 0 or call.status == "ESCALATED",
            },
            "transcript": [
                {
                    "id": msg.id,
                    "speaker": msg.speaker,
                    "text": msg.text,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                }
                for msg in sorted_messages
            ],
            "requirements": requirements_data,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calls/{call_id}/turn")
async def add_turn(
    call_id: int,
    request: Request,
    file: UploadFile = File(...),
    company_id: Optional[str] = Form(None),
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Process a voice turn for an existing call session:
    1. If call is AWAITING_LANGUAGE, transcribes speech to route language.
    2. If call is IN_PROGRESS, transcribes speech, generates LLM response, updates requirements, and returns TTS audio.
    """
    try:
        call = conversation_service.get_call(db, call_id)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))

    if call.ended_at is not None or call.status == "ESCALATED":
        raise HTTPException(
            status_code=400,
            detail=f"Call with ID {call_id} has already ended or has been routed to human assistance.",
        )

    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    ext = os.path.splitext(file.filename)[1] or ".wav"
    temp_filename = f"{uuid.uuid4()}{ext}"
    temp_path = os.path.join(RECORDINGS_DIR, temp_filename)
    output_filename = f"{uuid.uuid4()}_response.wav"
    output_path = os.path.join(GENERATED_AUDIO_DIR, output_filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # If call is awaiting language selection, handle speech-based language selection
        if call.status == "AWAITING_LANGUAGE":
            transcript = voice_service.stt_service.transcribe(temp_path)
            t_lower = transcript.lower()

            if any(k in t_lower for k in ["2", "hindi", "हिन्दी"]):
                routing = handle_dtmf_routing(db, call, "2")
                return {
                    "transcript": transcript,
                    "response": routing["response"],
                    "audio_url": routing["audio_url"],
                    "call_id": call.id,
                    "call_status": routing["status"],
                    "is_escalated": True,
                }
            elif any(k in t_lower for k in ["3", "marathi", "मराठी"]):
                routing = handle_dtmf_routing(db, call, "3")
                return {
                    "transcript": transcript,
                    "response": routing["response"],
                    "audio_url": routing["audio_url"],
                    "call_id": call.id,
                    "call_status": routing["status"],
                    "is_escalated": True,
                }
            elif any(k in t_lower for k in ["1", "english", "one"]):
                routing = handle_dtmf_routing(db, call, "1")
                return {
                    "transcript": transcript,
                    "response": routing["response"],
                    "audio_url": routing["audio_url"],
                    "call_id": call.id,
                    "call_status": routing["status"],
                    "is_escalated": False,
                }
            else:
                # Repeat IVR prompt if unrecognized
                return {
                    "transcript": transcript,
                    "response": IVR_PROMPT_MESSAGE,
                    "audio_url": "/audio/ivr_prompt.wav",
                    "call_id": call.id,
                    "call_status": "AWAITING_LANGUAGE",
                    "is_escalated": False,
                }

        # Normal English AI Conversation Pipeline Turn
        result = await voice_service.process_existing_turn(
            call_id=call_id,
            audio_path=temp_path,
            output_audio_path=output_path,
            db=db,
            company_id=company_id,
        )

        filename = os.path.basename(output_path)
        audio_url = request.url_for("audio_files", path=filename)
        result["audio_url"] = str(audio_url)
        if "audio_path" in result:
            del result["audio_path"]
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


@router.post("/calls/{call_id}/end")
async def end_call_endpoint(
    call_id: int,
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Mark an existing call as ended and guarantee requirements are extracted and committed."""
    try:
        call = conversation_service.get_call(db, call_id)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    if call.ended_at is not None:
        return {"status": "already_ended", "call_id": call_id}

    try:
        # Guarantee requirements extraction on call completion if customer spoke
        customer_msgs = (
            db.query(Message)
            .filter(Message.call_id == call_id, Message.speaker == "customer")
            .order_by(Message.id)
            .all()
        )
        full_text = "\n".join(m.text for m in customer_msgs if m.text)
        if full_text.strip():
            try:
                await voice_service._update_requirements(db, call_id, full_text, is_resolved=True)
            except Exception as e:
                print(f"Error finalizing requirements on end_call: {e}")

        conversation_service.end_call(db, call_id)
        return {"status": "ended", "call_id": call_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# In-App Live Voice Calling WebSocket Endpoint
# ---------------------------------------------------------------------------

@router.websocket("/ws/live-call")
@router.websocket("/ws/live-call/{call_id}")
async def live_voice_call_websocket(
    websocket: WebSocket,
    call_id: Optional[str] = None,
    customer_id: Optional[str] = Query(None),
    company_id: Optional[str] = Query(None),
    token: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """
    Bidirectional WebSocket endpoint for direct in-app real-time live voice calling with the AI Assistant.
    Provides low-latency conversational audio exchange between the user's browser and local AI:
    - Automatically provisions or resumes the Call session in MySQL.
    - Sends opening AI greeting audio upon connection.
    - Receives user microphone audio turns in real-time.
    - Transcribes with Faster-Whisper STT, calls Ollama LLM, synthesizes Piper TTS.
    - Returns assistant voice response with audio URL.
    - Supports live interruption / barge-in cancellation and clean session termination.
    """
    # 1. Authenticate WebSocket connection via query param or Authorization header
    raw_token = token
    if not raw_token:
        auth_hdr = websocket.headers.get("authorization")
        if auth_hdr and auth_hdr.startswith("Bearer "):
            raw_token = auth_hdr[7:].strip()
        elif auth_hdr:
            raw_token = auth_hdr.strip()

    auth_user = None
    auth_role = None
    caller_meta = {}
    if raw_token:
        try:
            payload = decode_access_token(raw_token)
            if payload and "sub" in payload:
                role = payload.get("role", "system_user")
                if role == "caller":
                    auth_user = payload["sub"]
                    auth_role = "caller"
                    caller_meta = {
                        "name": payload.get("caller_name", "In-App Voice Caller"),
                        "client_type": payload.get("client_type", "web"),
                    }
                else:
                    user = db.query(SystemUser).filter(SystemUser.username == payload["sub"]).first()
                    if user:
                        auth_user = user.username
                        auth_role = user.role
        except Exception as auth_err:
            print(f"[LIVE CALL AUTH ERROR] Token decode failed: {auth_err}")
            auth_user = None

    if not auth_user:
        print("[LIVE CALL] Rejecting WebSocket connection: missing or invalid authentication token.")
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Unauthorized: missing or invalid authentication token",
        )
        return

    # 2. Accept WebSocket connection after successful authentication
    await websocket.accept()
    print(f"[LIVE CALL] WebSocket authenticated for user '{auth_user}' (role: '{auth_role}')")

    # Resolve Customer & Call session
    call = None
    customer = None
    try:
        if call_id and str(call_id).lower() not in ("new", "inbound", "undefined", "none", "0"):
            try:
                num_id = int(str(call_id))
                call = conversation_service.get_call(db, num_id)
                if call:
                    customer = call.customer
            except Exception:
                call = None

        if not call:
            if auth_role == "caller":
                # Strict customer data isolation: guest/app callers are bound to their own isolated record
                guest_name = caller_meta.get("name", "In-App Voice Caller")
                client_tag = caller_meta.get("client_type", "app").upper()
                customer = conversation_service.get_or_create_customer(
                    db,
                    external_customer_id=f"CALLER_{auth_user}",
                    name=f"{guest_name} ({client_tag})",
                    phone=f"+INAPP_{auth_user[-8:]}",
                )
            else:
                if customer_id:
                    str_cid = str(customer_id).strip()
                    try:
                        if str_cid.isdigit():
                            customer = db.query(Customer).filter(Customer.id == int(str_cid)).first()
                        if not customer:
                            customer = db.query(Customer).filter(Customer.external_customer_id == str_cid).first()
                    except Exception as c_err:
                        print(f"[LIVE CALL] Warning resolving customer: {c_err}")
                        customer = None
                if not customer:
                    customer = db.query(Customer).order_by(Customer.id.asc()).first()
                if not customer:
                    customer = conversation_service.get_or_create_customer(
                        db,
                        external_customer_id="TEL_WEB_LIVE_USER",
                        name="Live Web User",
                        phone="+919999000000",
                    )

            call = conversation_service.create_call(db, customer.id)
            call.call_type = "INBOUND"
            call.status = "IN_PROGRESS"
            call.language = "en"
            call.recording_url = f"live_web_call:{call.id}"
            db.commit()

        # Notify client of established session
        await websocket.send_json({
            "event": "session_started",
            "call_id": call.id,
            "customer_id": customer.id if customer else None,
            "customer_name": customer.name if customer else "App User",
            "customer_phone": customer.phone if customer else "In-App Voice",
            "status": call.status,
            "user": auth_user or "operator",
        })

        # Play opening greeting if not already sent
        has_greeting = False
        if call.messages:
            for m in call.messages:
                if m.speaker == "assistant" and (m.text or "").strip():
                    has_greeting = True
                    break

        if not has_greeting:
            greeting_msg = ENGLISH_GREETING_MESSAGE
            greeting_filename = "en_greeting.wav"
            if company_id:
                from app.services.knowledge_service import KnowledgeService
                c_greeting = KnowledgeService().get_greeting(company_id)
                if c_greeting:
                    greeting_msg = c_greeting
                    clean_comp = str(company_id).strip().lower()
                    greeting_filename = f"greeting_{clean_comp}.wav"

            try:
                conversation_service.add_message(
                    db, call_id=call.id, speaker="assistant", text=greeting_msg
                )
                db.commit()
            except Exception as msg_err:
                print(f"[LIVE CALL] Warning logging initial greeting: {msg_err}")

            greeting_path = os.path.join(RECORDINGS_DIR, greeting_filename)
            if not os.path.exists(greeting_path) or os.path.getsize(greeting_path) < 1000:
                try:
                    await asyncio.to_thread(
                        voice_service.tts_service.synthesize,
                        greeting_msg,
                        greeting_path,
                        language="en",
                    )
                except Exception as tts_err:
                    print(f"[LIVE CALL] Warning synthesizing greeting: {tts_err}")

            await websocket.send_json({
                "event": "assistant_response",
                "call_id": call.id,
                "speaker": "assistant",
                "transcript": "",
                "response": greeting_msg,
                "audio_url": f"/audio/{greeting_filename}",
                "is_escalated": False,
            })

        active_turn_task: Optional[asyncio.Task] = None

        # WebSocket receive loop
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                if active_turn_task and not active_turn_task.done():
                    active_turn_task.cancel()
                break

            raw_bytes = message.get("bytes")
            raw_text = message.get("text")

            audio_data: Optional[bytes] = None
            is_interrupt = False
            is_end_call = False

            if raw_bytes and len(raw_bytes) > 0:
                audio_data = raw_bytes
            elif raw_text:
                try:
                    data = json.loads(raw_text)
                    event_type = str(data.get("event") or "").lower().strip()

                    if event_type == "interrupt":
                        is_interrupt = True
                    elif event_type in ("end_call", "hangup", "stop"):
                        is_end_call = True
                    elif event_type in ("user_audio", "audio_turn", "audio_chunk"):
                        b64_str = data.get("audio") or data.get("payload") or ""
                        if b64_str:
                            audio_data = base64.b64decode(b64_str)
                    elif event_type == "ping":
                        await websocket.send_json({"event": "pong"})
                        continue
                except Exception as parse_err:
                    print(f"[LIVE CALL] Non-JSON payload received: {parse_err}")
                    continue

            # Handle Interruption / Barge-in
            if is_interrupt:
                if active_turn_task and not active_turn_task.done():
                    active_turn_task.cancel()
                    print(f"[LIVE CALL] User interrupted; cancelled response generation for Call #{call.id}")
                await websocket.send_json({
                    "event": "interrupted",
                    "call_id": call.id,
                })
                continue

            # Handle End Call / Hangup
            if is_end_call:
                if active_turn_task and not active_turn_task.done():
                    active_turn_task.cancel()
                print(f"[LIVE CALL] User hung up Call #{call.id}")
                break

            # Process User Audio Turn
            if audio_data:
                # If call already ended or escalated, discard turn and notify client cleanly
                call_state = conversation_service.get_call(db, call.id)
                if call_state and (call_state.ended_at is not None or call_state.status in ("ESCALATED", "RESOLVED")):
                    print(f"[LIVE CALL] Discarding incoming audio turn for ended/escalated Call #{call.id}")
                    await websocket.send_json({
                        "event": "session_ended",
                        "call_id": call.id,
                        "status": call_state.status,
                        "message": "Call session has already ended.",
                    })
                    continue

                if len(audio_data) < 1000:
                    print(f"[LIVE CALL] Ignoring undersized audio chunk ({len(audio_data)} bytes) for Call #{call.id}")
                    continue

                if active_turn_task and not active_turn_task.done():
                    active_turn_task.cancel()

                temp_suffix = uuid.uuid4().hex[:8]
                if audio_data.startswith(b"RIFF"):
                    in_ext = ".wav"
                elif audio_data.startswith(b"\x1a\x45\xdf\xa3"):
                    in_ext = ".webm"
                elif audio_data.startswith(b"OggS"):
                    in_ext = ".ogg"
                else:
                    in_ext = ".wav"

                print(f"[LIVE CALL] Received speech turn ({len(audio_data)} bytes, format {in_ext}) for Call #{call.id}")

                temp_input_path = os.path.join(RECORDINGS_DIR, f"live_in_{call.id}_{temp_suffix}{in_ext}")
                temp_output_path = os.path.join(GENERATED_AUDIO_DIR, f"live_out_{call.id}_{temp_suffix}.wav")

                with open(temp_input_path, "wb") as f_in:
                    f_in.write(audio_data)

                # Convert incoming audio (WebM/Opus, Ogg, or WAV) to clean 16kHz mono WAV for high-accuracy Faster-Whisper STT
                stt_input_path = await asyncio.to_thread(convert_to_16k_mono_wav, temp_input_path)

                await websocket.send_json({
                    "event": "processing",
                    "status": "thinking",
                    "call_id": call.id,
                })

                async def _run_turn_worker(inp_path, out_path):
                    try:
                        # Safety check: if call ended/escalated during audio receipt, abort processing cleanly
                        call_check = conversation_service.get_call(db, call.id)
                        if call_check and (call_check.ended_at is not None or call_check.status in ("ESCALATED", "RESOLVED")):
                            print(f"[LIVE CALL] Call #{call.id} already ended/escalated before turn execution.")
                            await websocket.send_json({
                                "event": "session_ended",
                                "call_id": call.id,
                                "status": call_check.status,
                                "message": "Call session has already ended.",
                            })
                            return

                        stream_gen = voice_service.process_existing_turn_stream(
                            call_id=call.id,
                            audio_path=inp_path,
                            output_audio_path_prefix=out_path,
                            db=db,
                            company_id=company_id,
                        )
                        async for event in stream_gen:
                            # Verify client is still connected before sending
                            try:
                                if websocket.client_state.name == "DISCONNECTED":
                                    print(f"[LIVE CALL] WebSocket client disconnected mid-turn for Call #{call.id}")
                                    break
                            except Exception:
                                pass
                            
                            event_name = event.get("event")
                            if event_name == "assistant_chunk":
                                print(f"[LIVE CALL CHUNK] Call #{call.id} Chunk {event.get('chunk_index')}: \"{event.get('text')}\" -> {event.get('audio_url')}")
                            elif event_name == "assistant_response":
                                print(f"[LIVE CALL RECOGNITION] Call #{call.id} | Recognized: \"{event.get('transcript')}\" | Full Response: \"{event.get('response')}\"")
                            
                            await websocket.send_json(event)
                    except asyncio.CancelledError:
                        print(f"[LIVE CALL] Turn worker cancelled for Call #{call.id}")
                        raise
                    except Exception as err:
                        err_msg = str(err)
                        print(f"[LIVE CALL ERROR] Turn processing error: {err_msg}")
                        if "already ended" in err_msg.lower():
                            await websocket.send_json({
                                "event": "session_ended",
                                "call_id": call.id,
                                "status": "ENDED",
                                "message": "Call session has already ended.",
                            })
                        else:
                            await websocket.send_json({
                                "event": "error",
                                "call_id": call.id,
                                "message": err_msg,
                            })
                    finally:
                        # Retain input audio for session diagnostics and audit inspection
                        pass

                active_turn_task = asyncio.create_task(_run_turn_worker(stt_input_path, temp_output_path))

    except WebSocketDisconnect:
        print(f"[LIVE CALL] WebSocket disconnected for Call #{call.id if call else 'unbound'}")
    except Exception as e:
        print(f"[LIVE CALL ERROR] Exception: {e}")
    finally:
        if call:
            try:
                db.rollback()
                conversation_service.end_call(db, call.id)
                print(f"[LIVE CALL] Session finalized and marked ended for Call #{call.id}")
            except Exception as fin_err:
                print(f"[LIVE CALL] Error in final call end: {fin_err}")
