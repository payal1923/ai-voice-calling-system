# Exotel Telephony Provider Forensic Audit & Migration Report

## Executive Summary
This document provides an exhaustive, code-level forensic audit of the AI Voice Calling application codebase (`c:\Users\User\Downloads\ai_voice_calling_F\ai_voice_calling_F`). The objective is to evaluate the existing telephony infrastructure, map Twilio dependencies, analyze Exotel readiness, and detail the complete call and audio flows before executing any migration changes.

---

## 1. Exotel Adapter Existence Analysis
* **Status**: **EXISTS (Partially Implemented / Incomplete)**
* **Location**: [app/services/telephony_adapter.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/services/telephony_adapter.py#L230-L314)
* **Class Name**: `ExotelTelephonyAdapter` (inheriting from `TelephonyAdapter`)
* **Factory Registration**: [app/services/telephony_adapter.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/services/telephony_adapter.py#L418-L419)
  ```python
  elif p_name == "exotel":
      return ExotelTelephonyAdapter()
  ```

---

## 2. Inventory of All Twilio-Specific Components

### A. Classes & Functions
* `TwilioTelephonyAdapter` in [app/services/telephony_adapter.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/services/telephony_adapter.py#L117-L228)

### B. Imports & Dependencies
* `httpx`: Used for making raw REST API calls to Twilio endpoints (`https://api.twilio.com/2010-04-01/Accounts/...`). No official `twilio` Python SDK package is imported in the codebase.

### C. Environment Variables & Configuration
* `TWILIO_ACCOUNT_SID` in [.env](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/.env#L3) & [.env.example](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/.env.example#L9)
* `TWILIO_AUTH_TOKEN` in [.env](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/.env#L4) & [.env.example](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/.env.example#L10)
* `TWILIO_PHONE_NUMBER` in [.env](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/.env#L5) & [.env.example](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/.env.example#L11)
* Default fallback credentials in `TwilioTelephonyAdapter.__init__`: `self.from_number = os.environ.get("TWILIO_PHONE_NUMBER", "+15005550006")`

### D. REST API Endpoints Called
1. **Outbound Call Initiation**:
   `POST https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Calls.json`
   * Auth: HTTP Basic Auth `(account_sid, auth_token)`
   * Data payload: `{"To": phone_number, "From": self.from_number, "Url": target_url}`
2. **Call Termination**:
   `POST https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Calls/{external_call_id}.json`
   * Auth: HTTP Basic Auth `(account_sid, auth_token)`
   * Data payload: `{"Status": "completed"}`
3. **Recording Retrieval URL Pattern**:
   `https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Calls/{external_call_id}/Recordings`

### E. XML / TwiML Responses Generated
1. **IVR Language Selection (`ivr_prompt`)**:
   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <Response>
     <Gather numDigits="1" action="{act}" method="POST">
       <Say>{prompt_text}</Say>
     </Gather>
     <Say>{prompt_text}</Say>
   </Response>
   ```
2. **Speech Gathering / Turn Processing (`gather_speech`)**:
   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <Response>
     <Say>{text}</Say>
     <Play>{audio_url}</Play>
     <Gather input="speech" action="{act}" method="POST" timeout="5"/>
   </Response>
   ```
3. **Play Audio & Hangup (`play_and_hangup`)**:
   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <Response>
     <Play>{audio_url}</Play>
     <Say>{text}</Say>
     <Hangup/>
   </Response>
   ```

### F. Payload Parameters Parsed
* `CallSid` (Twilio Call Identifier string, e.g., `CAacb162f1449ee7946...`)
* `From` & `To` (Phone numbers)
* `Digits` (Keypad DTMF press value)
* `SpeechResult` (Twilio built-in speech recognition transcript text)
* `RecordingUrl` (Carrier recording URL on completion)
* `CallStatus` (`completed`, `in-progress`, `failed`, `busy`, `no-answer`)

---

## 3. Provider-Independent Components (MUST REMAIN UNCHANGED)

The following components encapsulate application business logic, AI orchestration, database models, and authentication, and **must not be modified**:

| Module / Component | File Location | Responsibilities |
| :--- | :--- | :--- |
| `VoiceResponseService` | [app/services/voice_response_service.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/services/voice_response_service.py) | Orchestrates STT, LLM generation, TTS synthesis, context building, and async requirements extraction. |
| `ConversationService` | [app/services/conversation_service.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/services/conversation_service.py) | High-level database transactions for Customers, Calls, and Messages. |
| `ConversationContextService` | [app/services/conversation_context_service.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/services/conversation_context_service.py) | Constructs multi-turn conversation memory for LLM prompts. |
| `ResponseService` | [app/services/response_service.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/services/response_service.py) | Invokes Ollama LLM (`gemma3:4b` / `qwen2.5:0.5b`) and performs resolution evaluation. |
| `RequirementService` | [app/services/requirement_service.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/services/requirement_service.py) | Extracts structured JSON requirements from customer speech. |
| `EscalationService` | [app/services/escalation_service.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/services/escalation_service.py) | Detects human transfer triggers and manages assistance requests queue. |
| `STTService` | [app/services/stt_service.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/services/stt_service.py) | Local speech-to-text processing using Faster-Whisper. |
| `TTSService` | [app/services/tts_service.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/services/tts_service.py) | Local audio synthesis using Piper ONNX / Edge-TTS models. |
| `LLM Engine` | [app/services/llm_service.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/services/llm_service.py) | Async HTTP client communicating with local Ollama daemon. |
| `Database ORM Models` | [app/database/models.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/database/models.py) | `SystemUser`, `Customer`, `Call`, `Message`, `Requirement`, `HumanAssistanceRequest`. |
| `Database Connection` | [app/database/database.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/database/database.py) | MySQL SQLAlchemy engine & session factory. |
| `Web Live Call Endpoints` | [app/api/call_routes.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/api/call_routes.py) | Dashboard customer CRUD, manual call creation, and direct audio file turn upload. |
| `Auth & Escalation Routers` | [app/api/auth_routes.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/api/auth_routes.py), [app/api/escalation_routes.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/api/escalation_routes.py) | JWT login, user management, and escalation management. |
| `Frontend Dashboard` | [frontend/index.html](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/frontend/index.html) | Single-page UI web application. |

---

## 4. Complete Outbound Call Flow

```
[ Operator clicks "Call Customer" in Dashboard ]
                       │
                       ▼
    POST /api/telephony/outbound (Form: customer_id)
                       │
                       ▼
1. Fetch Customer record from MySQL database
2. conversation_service.create_call(db, customer_id) -> Call row created (status="IN_PROGRESS", call_type="OUTBOUND")
3. adapter = get_telephony_adapter() -> Instantiates configured adapter (Exotel/Twilio)
4. provider_call_id = adapter.start_outbound_call(customer_id, phone_number)
   - Twilio: POST https://api.twilio.com/2010-04-01/Accounts/{SID}/Calls.json
   - Exotel: POST https://{subdomain}/v1/Accounts/{SID}/Calls/connect.json
5. call.recording_url = f"provider_call_id:{provider_call_id}" (Persisted in MySQL)
6. Return JSON response to frontend UI with call_id and provider_call_id
                       │
                       ▼
[ Telephony Provider Dials Destination Phone Number ]
                       │
                       ▼
[ Customer Answers Phone -> Carrier Triggers Webhook: POST /api/telephony/webhook/incoming ]
```

---

## 5. Complete Inbound Call Flow

```
[ Customer Dials Virtual Phone Number (Exophone / Twilio Number) ]
                       │
                       ▼
[ Carrier HTTP POST to /api/telephony/webhook/incoming ]
                       │
                       ▼
1. parse_webhook_payload(request) -> Extract form/query/JSON data
2. normalized = adapter.handle_incoming_call(payload) -> Extract external_call_id, customer_phone, to_phone
3. ext_cust_id = f"TEL_{customer_phone}"
4. customer = conversation_service.get_or_create_customer(db, external_customer_id, phone)
5. call = conversation_service.create_call(db, customer.id)
   - Set call_type="INBOUND", status="AWAITING_LANGUAGE", language="pending"
6. conversation_service.add_message(db, call_id=call.id, speaker="assistant", text=IVR_PROMPT_MESSAGE)
7. action_url = f"{base_url}/api/telephony/webhook/dtmf?call_id={call.id}"
8. xml_content = adapter.generate_carrier_response(action_type="ivr_prompt", text=IVR_PROMPT_MESSAGE, action_url=action_url)
9. Return FastAPIResponse(content=xml_content, media_type="text/xml")
                       │
                       ▼
[ Carrier plays IVR prompt ("Press 1 for English, 2 for Hindi, 3 for Marathi") & waits for DTMF ]
```

---

## 6. DTMF Flow

```
[ Customer presses keypad button (1, 2, or 3) ]
                       │
                       ▼
[ Carrier HTTP POST to /api/telephony/webhook/dtmf?call_id={call.id} ]
                       │
                       ▼
1. normalized = adapter.handle_incoming_call(payload)
2. digit_val = normalized["digits"] or payload["Digits"]
3. call = conversation_service.get_call(db, call_id)
4. handle_dtmf_routing(db, call, digit_val):
   ┌───────────────────────┬───────────────────────────────┬───────────────────────────────┐
   │ Digit = 1 (English)   │ Digit = 2 (Hindi)             │ Digit = 3 (Marathi)           │
   ├───────────────────────┼───────────────────────────────┼───────────────────────────────┤
   │ call.language = "en"  │ call.language = "Hindi"       │ call.language = "Marathi"     │
   │ call.status="IN_PROG" │ call.status = "ESCALATED"     │ call.status = "ESCALATED"     │
   │ Log customer msg      │ Log customer msg              │ Log customer msg              │
   │ Log English greeting  │ Log Hindi assistance notice   │ Log Marathi assistance notice │
   │                       │ Create Human Assistance Req   │ Create Human Assistance Req   │
   └───────────────────────┴───────────────────────────────┴───────────────────────────────┘
5. Response XML Generation:
   - For English (Digit 1):
     adapter.generate_carrier_response(
         action_type="gather_speech",
         text=ENGLISH_GREETING_MESSAGE,
         action_url=f"{base_url}/api/telephony/webhook/turn?call_id={call.id}"
     )
   - For Hindi/Marathi (Digit 2/3):
     adapter.generate_carrier_response(
         action_type="play_and_hangup",
         text=HINDI_ASSISTANCE_MESSAGE / MARATHI_ASSISTANCE_MESSAGE,
         audio_url="..."
     )
```

---

## 7. Speech / Audio Flow

The architecture contains two separate audio processing paths:

### Path A: Telephony Carrier Passthrough Pathway (`/api/telephony/webhook/turn`)
1. **Speech Recognition**: Handled on carrier network (or carrier STT plugin). Provider posts text transcript in `SpeechResult` payload field to `/api/telephony/webhook/turn`.
2. **Transcript Persistence**: Customer text logged to `messages` table (`speaker="customer"`).
3. **Context & LLM**: `ConversationContextService` fetches conversation history. `ResponseService` calls Ollama LLM (`gemma3:4b`).
4. **Resolution Evaluation**: `ResponseService.check_resolution()` determines if customer inquiry is complete.
5. **Response Persistence**: Assistant response text logged to `messages` table (`speaker="assistant"`).
6. **Async Requirements Extraction**: `voice_service._update_requirements()` runs in background task to update `requirements` table.
7. **Voice XML Return**: `adapter.generate_carrier_response()` generates XML containing `<Say>{response_text}</Say>` or `<Play>{audio_url}</Play>`.
8. **Playback**: Telephony provider engine reads response text back to caller over PSTN.

### Path B: Local / Web / Asterisk AGI Pathway (`/api/telephony/asterisk-turn` & `/api/calls/{call_id}/turn`)
1. **Audio Upload**: Raw WAV audio uploaded via HTTP POST multipart form.
2. **Local STT**: `STTService` transcribes WAV audio using local Faster-Whisper model.
3. **LLM Generation**: Ollama LLM generates response text.
4. **Local TTS**: `TTSService` synthesizes voice audio using local Piper ONNX model.
5. **Audio Resampling**: (Asterisk only) Resampled to 8000Hz 16-bit PCM mono WAV via PyAV/wave.
6. **Audio Stream Return**: Binary WAV bytes returned in HTTP response with custom headers (`X-Call-ID`, `X-Transcript`, `X-Response-Text`).

---

## 8. Call-Status Flow

```
[ Call Ends, Disconnects, or Times Out on Carrier Network ]
                       │
                       ▼
[ Carrier HTTP POST to /api/telephony/webhook/status ]
                       │
                       ▼
1. Extract call_sid and call_status from payload
2. Query MySQL database:
   call = db.query(Call).filter(Call.recording_url.like(f"%{call_sid}%")).first()
3. If call found and call.ended_at is None:
   - If call_status in ("COMPLETED", "ANSWERED"):
     conversation_service.end_call(db, call.id, status="RESOLVED")
   - Else (BUSY, FAILED, NO-ANSWER):
     conversation_service.end_call(db, call.id, status="FAILED")
4. Return {"status": "received", "call_status": call_status}
```

---

## 9. Recording Flow

```
[ Carrier completes call recording processing ]
                       │
                       ▼
[ Carrier HTTP POST to /api/telephony/webhook/recording ]
                       │
                       ▼
1. Extract call_sid and recording_url from payload
2. Query MySQL database:
   call = db.query(Call).filter(Call.recording_url.like(f"%{call_sid}%")).first()
3. If call found:
   call.recording_url = recording_url
   db.commit()
4. Return {"status": "recording_updated", "recording_url": recording_url}
```

---

## 10. Provider Call ID Storage & Mapping Mechanism

* **Database Schema Analysis**: In [app/database/models.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/database/models.py#L36-L53), the `calls` table does **NOT** contain a dedicated `provider_call_id` or `twilio_sid` column.
* **Mapping Strategy in Code**:
  1. **Outbound Call Tracking**: When an outbound call is initiated via `POST /api/telephony/outbound` ([app/api/telephony_routes.py:L310](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/api/telephony_routes.py#L310)), the provider's returned ID (e.g., `CA...` or `EX...`) is formatted and saved directly into the `recording_url` column:
     ```python
     call.recording_url = f"provider_call_id:{provider_call_id}"
     ```
  2. **Callback Lookups**: Status and recording webhooks search for the internal `Call` record using SQL string matching:
     ```python
     call = db.query(Call).filter(Call.recording_url.like(f"%{call_sid}%")).first()
     ```
  3. **Inbound Call Session Linkage**: For multi-turn inbound webhooks, `call.id` (integer primary key) is appended explicitly to callback URLs:
     `action_url = f"{base_url}/api/telephony/webhook/dtmf?call_id={call.id}"`
     `turn_url = f"{base_url}/api/telephony/webhook/turn?call_id={call.id}"`

---

## 11. WebSocket & Audio Streaming Analysis
* **Status**: **NOT IMPLEMENTED**
* **Verification**: Codebase search across `main.py`, `telephony_routes.py`, `call_routes.py`, and `telephony_adapter.py` confirms zero WebSocket connections, no `websockets` dependency, and no implementation of Twilio Media Streams or Exotel WebSocket streams.
* **Architecture**: Telephony interactions rely entirely on standard HTTP REST API triggers and HTTP Passthrough XML Webhooks (`/webhook/incoming`, `/webhook/dtmf`, `/webhook/turn`).

---

## 12. Speech Processing Engine (Local vs. Provider-Side)

* **Carrier Telephony Calls (`/api/telephony/webhook/*`)**:
  * **STT**: Uses **Provider-Side STT** (`SpeechResult` payload parameter POSTed by carrier).
  * **TTS**: Uses **Provider-Side TTS** (`<Say>{text}</Say>` tags rendered in XML response).
* **Direct Web Dashboard & Asterisk AGI Calls (`/api/calls/*` and `/api/telephony/asterisk-*`)**:
  * **STT**: Uses **Local STT** (`STTService` with Faster-Whisper `whisper-base`).
  * **TTS**: Uses **Local TTS** (`TTSService` with Piper ONNX `en_US-lessac-high`).

---

## 13. Evaluation of Existing `ExotelTelephonyAdapter` Implementation

* **Overall Status**: **Incomplete & Non-Functional for Production**
* **Detailed Defects & Deficiencies**:
  1. **`generate_carrier_response()` Missing Actions**:
     - `action_type == "gather_speech"` is completely missing from `ExotelTelephonyAdapter.generate_carrier_response()`. When `carrier_speech_turn_webhook` executes turn 2+, the Exotel adapter falls through to line 313 returning `<Response><Hangup/></Response>`, causing Exotel to **hang up on the caller after turn 1**.
     - `action_type == "ivr_prompt"` renders `<Response><Say>{prompt_text}</Say></Response>` **without** a `<Gather>` tag, preventing keypress capture.
  2. **Outbound API Data Formatting**:
     - `start_outbound_call()` in `ExotelTelephonyAdapter` passes:
       `data={"From": phone_number, "To": self.caller_id, "CallerId": self.caller_id, "Url": target_url}`
       In Exotel's Connect API (`/v1/Accounts/{SID}/Calls/connect.json`), `From` is the first leg (Agent/Destination number), `To` is the customer phone number, and `CallerId` must be the verified Exophone.
  3. **Call Termination Stubbed Out**:
     - `end_call()` in `ExotelTelephonyAdapter` simply returns `True` without executing an HTTP POST request to terminate the active session on Exotel's network.
  4. **Basic Authentication Configuration**:
     - Exotel API requires HTTP Basic Authentication using `EXOTEL_KEY` and `EXOTEL_TOKEN` (or `EXOTEL_SID` and `EXOTEL_TOKEN`). The adapter currently attempts `auth=(self.key, self.token)` which fails if `EXOTEL_KEY` is not set in `.env`.

---

## 14. Identified Bugs & Integration Blockers

### Bug 1: Missing `<Gather>` and `gather_speech` in `ExotelTelephonyAdapter`
* **Impact**: Critical / Call Termination Failure.
* **Root Cause**: In [app/services/telephony_adapter.py:L294-L314](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/services/telephony_adapter.py#L294-L314), `ExotelTelephonyAdapter.generate_carrier_response()` lacks `<Gather>` tag handling for `ivr_prompt` and ignores `gather_speech` entirely. Exotel immediately hangs up on the caller.

### Bug 2: Localhost Callback URL Inaccessibility
* **Impact**: Critical / Webhook Failure.
* **Root Cause**: If `TELEPHONY_WEBHOOK_BASE_URL` in `.env` is set to `http://localhost:8000` or an inactive ngrok URL, Exotel's public cloud servers cannot deliver HTTP webhooks to the local machine, resulting in HTTP 404 or connection timeouts.

### Bug 3: Outbound API Parameter Mismatch in Exotel Call Connect API
* **Impact**: High / Outbound Calling Failure.
* **Root Cause**: `ExotelTelephonyAdapter.start_outbound_call()` sets `"To": self.caller_id` instead of `"To": phone_number`. Exotel rejects the payload with HTTP 400 Bad Request.

### Bug 4: Incomplete `end_call` Implementation in Exotel Adapter
* **Impact**: Medium / Resource Leak.
* **Root Cause**: `ExotelTelephonyAdapter.end_call()` returns `True` without performing an HTTP request to Exotel's call termination API.

---

## Required Summary Sections

### A. Files That MUST Change
1. [app/services/telephony_adapter.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/services/telephony_adapter.py): Complete `ExotelTelephonyAdapter` implementation (API auth, outbound params, full XML response generation with `<Gather>` and `<Say>`, end_call API).
2. [.env](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/.env): Update `TELEPHONY_PROVIDER=exotel`, Exotel credentials (`EXOTEL_SID`, `EXOTEL_KEY`, `EXOTEL_TOKEN`, `EXOTEL_CALLER_ID`), and `TELEPHONY_WEBHOOK_BASE_URL`.

### B. Files That MUST NOT Change
* `app/services/voice_response_service.py`
* `app/services/conversation_service.py`
* `app/services/conversation_context_service.py`
* `app/services/response_service.py`
* `app/services/requirement_service.py`
* `app/services/stt_service.py`
* `app/services/tts_service.py`
* `app/services/escalation_service.py`
* `app/services/llm_service.py`
* `app/database/database.py`
* `app/database/models.py`
* `app/api/auth_routes.py`
* `app/api/call_routes.py`
* `app/api/automation_routes.py`
* `app/api/escalation_routes.py`
* `frontend/index.html`
* `main.py`

### C. Twilio-Specific Code That Must Be Replaced
1. Active instantiation of `TwilioTelephonyAdapter` when `TELEPHONY_PROVIDER=twilio` in [.env](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/.env).
2. Twilio API credentials (`TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER`).
3. Twilio-specific API URLs (`https://api.twilio.com/2010-04-01/Accounts/...`).

### D. Existing Exotel Code That Can Be Reused
1. `ExotelTelephonyAdapter` class structure and registration in `get_telephony_adapter()` factory in [app/services/telephony_adapter.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/services/telephony_adapter.py).
2. Incoming payload normalization logic in `ExotelTelephonyAdapter.handle_incoming_call()`.
3. Existing carrier webhook endpoints in [app/api/telephony_routes.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/api/telephony_routes.py) (`/webhook/incoming`, `/webhook/dtmf`, `/webhook/turn`, `/webhook/status`, `/webhook/recording`).

### E. Exact Next Implementation Step
1. **Update [.env](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/.env)**: Set `TELEPHONY_PROVIDER=exotel` and add valid `EXOTEL_SID`, `EXOTEL_KEY`, `EXOTEL_TOKEN`, `EXOTEL_CALLER_ID`, and a publicly accessible `TELEPHONY_WEBHOOK_BASE_URL` (e.g. active ngrok domain).
2. **Refine `ExotelTelephonyAdapter` in [app/services/telephony_adapter.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/services/telephony_adapter.py)**: Implement complete XML response generation (`ivr_prompt` with `<Gather>`, `gather_speech` with `<Say>`/`<Gather>`, `play_and_hangup`), correct outbound HTTP request payload formatting, and live call termination (`end_call`).
