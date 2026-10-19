# Exotel Telephony Provider Migration Analysis

## Executive Summary
This document provides a comprehensive architectural forensic audit and migration plan to transition the existing AI Voice Calling application from Twilio to **Exotel** for real Indian telephone calling.

The application architecture utilizes a clean **Adapter Pattern** (`TelephonyAdapter`), ensuring that replacing the telephony carrier requires **zero changes** to the core AI pipeline, database models, frontend interface, or conversation management services.

---

## A. Current Architecture

```
                                 ┌─────────────────────────────────────────┐
                                 │              Web Dashboard              │
                                 │            (frontend/index.html)        │
                                 └────────────────────┬────────────────────┘
                                                      │ HTTP / REST API
                                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       FastAPI Backend Engine                                    │
│                                                                                                 │
│  ┌───────────────────────────────┐     ┌─────────────────────────────────────────────────────┐  │
│  │    Telephony Webhook Layer    │     │                 AI Core Pipeline                    │  │
│  │   (app/api/telephony_routes)  │     │           (app/services/voice_response)             │  │
│  └───────────────┬───────────────┘     └──────────────────────────┬──────────────────────────┘  │
│                  │                                                │                             │
│                  ▼                                                ▼                             │
│  ┌───────────────────────────────┐     ┌─────────────────────────────────────────────────────┐  │
│  │   Telephony Adapter Interface │     │   Context - STT - Ollama LLM - Piper TTS - Reqs    │  │
│  │   (Mock / Twilio / Exotel)    │     │   (Conversation, Response, Requirement Services)   │  │
│  └───────────────┬───────────────┘     └──────────────────────────┬──────────────────────────┘  │
└──────────────────┼────────────────────────────────────────────────┼─────────────────────────────┘
                   │                                                │
                   ▼                                                ▼
┌──────────────────────────────────────┐        ┌─────────────────────────────────────────────────┐
│     Telephony Carrier (Exotel)       │        │          MySQL Relational Database              │
│  - Outbound: REST API Connect        │        │  (system_users, customers, calls, messages,     │
│  - Inbound: Passthrough Webhooks     │        │   requirements, human_assistance_requests)      │
└──────────────────────────────────────┘        └─────────────────────────────────────────────────┘
```

---

## B. Current Telephony Provider Implementation

The telephony provider abstraction is defined in [app/services/telephony_adapter.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/services/telephony_adapter.py):

* **Abstract Interface (`TelephonyAdapter`)**: Defines five mandatory carrier methods:
  1. `start_outbound_call(customer_id, phone_number, callback_url)`
  2. `handle_incoming_call(external_payload)`
  3. `end_call(external_call_id)`
  4. `get_recording(external_call_id)`
  5. `generate_carrier_response(action_type, text, audio_url, action_url)`

* **Provider Factory (`get_telephony_adapter()`)**: Reads `TELEPHONY_PROVIDER` from environment (`mock`, `twilio`, `exotel`, `plivo`) and dynamically instantiates the appropriate adapter class.

---

## C. Provider-Specific Logic (Twilio / Carrier Specific)

The following components encapsulate provider-specific payload formats, XML response syntax, and REST API calls:

1. **[app/services/telephony_adapter.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/services/telephony_adapter.py)**:
   - `TwilioTelephonyAdapter`: Formats TwiML `<Gather>`, `<Say>`, `<Play>`, `<Hangup>`, and calls `https://api.twilio.com`.
   - `ExotelTelephonyAdapter`: Exotel-specific adapter class skeleton (to be refined for live Exotel API format).
2. **[app/api/telephony_routes.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/api/telephony_routes.py)**:
   - `parse_webhook_payload()`: Normalizes incoming carrier form data/JSON payloads into standardized dictionary keys (`CallSid`, `From`, `To`, `Digits`, `SpeechResult`).

---

## D. Provider-Independent Files & Functions (MUST NOT BE MODIFIED)

The following core modules handle AI logic, database persistence, and system operations. They are strictly provider-agnostic and **must remain untouched**:

| File Path | Responsible Class / Function | Purpose |
| :--- | :--- | :--- |
| [app/services/voice_response_service.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/services/voice_response_service.py) | `VoiceResponseService.process_existing_turn()` | Orchestrates STT, LLM generation, TTS synthesis, requirement extraction. |
| [app/services/conversation_service.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/services/conversation_service.py) | `ConversationService` | Manages MySQL Customer and Call records. |
| [app/services/conversation_context_service.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/services/conversation_context_service.py) | `ConversationContextService` | Fetches prior call transcripts from DB for LLM prompt context. |
| [app/services/response_service.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/services/response_service.py) | `ResponseService` | Queries Ollama LLM (`qwen2.5:0.5b`) and performs resolution checks. |
| [app/services/requirement_service.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/services/requirement_service.py) | `RequirementService` | Extracts structured JSON requirements from caller speech. |
| [app/services/stt_service.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/services/stt_service.py) | `STTService` | Faster-Whisper speech-to-text engine. |
| [app/services/tts_service.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/services/tts_service.py) | `TTSService` | Piper ONNX / Edge-TTS audio synthesis engine. |
| [app/database/database.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/database/database.py) | `engine`, `SessionLocal`, `get_db` | MySQL SQLAlchemy database connection. |
| [app/database/models.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/database/models.py) | `Customer`, `Call`, `Message`, `Requirement` | MySQL ORM schema definitions. |
| [frontend/index.html](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/frontend/index.html) | Web Dashboard | Operations UI. |

---

## E. Existing AI Conversation Flow

```
[ Inbound Call or Outbound Start ]
               │
               ▼
   [ DB Call Record Created ] (Status: AWAITING_LANGUAGE)
               │
               ▼
    [ IVR Language Selection ] -> Customer selects language (1 = English)
               │
               ▼
   [ Call Status -> IN_PROGRESS ] -> Initial Greeting Played
               │
   ┌───────────┴────────────────────────────────────────┐
   ▼                                                    ▼
[ Speech Received from Caller ]              [ Context Service Builds History ]
   │                                                    │
   └───────────────────────────┬────────────────────────┘
                               ▼
            [ Ollama LLM Generates Response ]
                               │
            [ Response Saved to MySQL `messages` ]
                               │
   ┌───────────────────────────┴────────────────────────┐
   ▼                                                    ▼
[ Resolution Check Executed ]               [ Requirements Extracted in DB ]
   │                                                    │
   ├─► Resolved? -> End Call & Play Goodbye             └─► Saved to `requirements`
   └─► Unresolved? -> Loop for Next Turn
```

---

## F. Existing Audio Flow

The system supports two parallel audio pathways:

1. **PSTN / Telephony Passthrough Pathway (Exotel / Twilio)**:
   - Caller speaks into phone handset -> Telephony Carrier transcribes speech or collects DTMF.
   - Carrier POSTs `SpeechResult` / `Digits` text to webhook `/api/telephony/webhook/turn`.
   - Backend generates response text -> Returns Carrier XML containing `<Say>{response}</Say>` or `<Play>https://<ngrok>/audio/{file}.wav</Play>`.

2. **Native Audio Processing Pathway (Asterisk AGI / REST File Uploads)**:
   - Raw audio WAV bytes uploaded to `/api/telephony/asterisk-turn` or `/api/calls/{call_id}/turn`.
   - `STTService` transcribes WAV via Faster-Whisper -> LLM generates response -> `TTSService` synthesizes Piper WAV -> Converted to 8000Hz mono PCM WAV -> Returned as raw audio bytes in HTTP response.

---

## G. Existing Database & Call-Record Flow

1. **Customer Registration**: Auto-creates or fetches `Customer` row by phone number (`TEL_<phone>`).
2. **Call Session**: Creates `Call` row (`call_type="INBOUND"` or `"OUTBOUND"`, `status="AWAITING_LANGUAGE"`).
3. **Transcript Persistence**: Every turn logs a `Message` row (`speaker="customer"` or `speaker="assistant"`).
4. **Structured Requirements**: Extracted requirements saved to `Requirement` row linked via `call_id`.
5. **Escalations**: Non-English language selections (Hindi/Marathi) logged in `human_assistance_requests`.

---

## H. Required Exotel Integration Architecture

To integrate Exotel seamlessly into the existing application:

1. **API Subdomain**: `api.exotel.com` (Singapore Region).
2. **Authentication**: HTTP Basic Authentication using `EXOTEL_KEY` and `EXOTEL_TOKEN`.
3. **Outbound API**: Exotel Connect API:
   - Endpoint: `https://<EXOTEL_KEY>:<EXOTEL_TOKEN>@api.exotel.com/v1/Accounts/rsl6a1/Calls/connect.json`
   - Form Data parameters:
     - `From`: Customer phone number (e.g. `+919876543210`)
     - `To`: Exophone number (virtual number)
     - `CallerId`: Exophone number
     - `Url`: `https://<ngrok-domain>/api/telephony/webhook/incoming`
4. **Passthrough XML Responses**:
   - Exotel Passthrough applet supports standard Voice XML tags (`<Response><Say>...</Say><Gather action="...">...</Gather></Response>`).

---

## I. Outbound Call Flow with Exotel

```
Operator clicks "Call" on UI -> POST /api/telephony/outbound
            │
            ▼
conversation_service.create_call() (MySQL Call ID created)
            │
            ▼
ExotelTelephonyAdapter.start_outbound_call()
            │
            ▼
HTTP POST to https://api.exotel.com/v1/Accounts/rsl6a1/Calls/connect.json
  - Body: From=<CustomerPhone>, To=<Exophone>, CallerId=<Exophone>, Url=<WebhookURL>
            │
            ▼
Exotel dials customer -> Connects call -> Fetches XML from /api/telephony/webhook/incoming
```

---

## J. Inbound Call Flow with Exotel

```
Customer dials Exophone -> Exotel Passthrough Applet triggers HTTP POST to /api/telephony/webhook/incoming
            │
            ▼
carrier_incoming_call_webhook()
  - Parses payload (CallSid, From, To)
  - Creates Customer & Call in MySQL (status: AWAITING_LANGUAGE)
  - Returns Exotel XML:
    <Response>
      <Gather numDigits="1" action="https://<ngrok>/api/telephony/webhook/dtmf?call_id=X" method="POST">
        <Say>Press 1 for English...</Say>
      </Gather>
    </Response>
            │
            ▼
Customer presses 1 -> Exotel POSTs to /api/telephony/webhook/dtmf?call_id=X
            │
            ▼
carrier_dtmf_webhook() sets language to English, returns greeting & <Gather input="speech" action="...">
            │
            ▼
Customer speaks -> Exotel STT sends text in SpeechResult to /api/telephony/webhook/turn?call_id=X
            │
            ▼
carrier_speech_turn_webhook() runs LLM, logs messages in DB, and returns next XML turn
```

---

## K. WebSocket / Audio Streaming Requirements

* **Exotel PSTN Technical Limitation**: Standard Exotel voice calls over PSTN in India **do NOT support raw bi-directional WebSocket audio streaming** (unlike Twilio Media Streams or Asterisk AGI).
* **Supported Mechanism**: Exotel operates via **HTTP Passthrough Webhooks** + **`<Play>` audio URLs** / **`<Say>` speech tags**.
* **Impact on Application**: No WebSocket handler is required for Exotel. The existing HTTP webhook flow (`/webhook/incoming`, `/webhook/dtmf`, `/webhook/turn`) is 100% compatible with Exotel Passthrough Applets.

---

## L. Exact Files That Need Modification

Only **two Python files** and the **`.env`** configuration file need modification:

1. **[app/services/telephony_adapter.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/services/telephony_adapter.py)**:
   - Update `ExotelTelephonyAdapter` to use `EXOTEL_KEY`, `EXOTEL_TOKEN`, `EXOTEL_SID`, `EXOTEL_CALLER_ID`, and `EXOTEL_SUBDOMAIN`.
   - Update `start_outbound_call()` to send HTTP Basic Auth and format Exotel Connect API form data.
2. **[app/api/telephony_routes.py](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/app/api/telephony_routes.py)**:
   - Ensure parameter normalization handles Exotel fields (`CallSid`, `From`, `To`, `digits`, `Digits`).
3. **[.env](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/.env)**:
   - Configure Exotel credentials and base webhook URL.

---

## M. Exact Files That Should NOT Be Modified

* `app/services/voice_response_service.py`
* `app/services/conversation_service.py`
* `app/services/conversation_context_service.py`
* `app/services/response_service.py`
* `app/services/requirement_service.py`
* `app/services/stt_service.py`
* `app/services/tts_service.py`
* `app/services/llm_service.py`
* `app/database/database.py`
* `app/database/models.py`
* `app/api/call_routes.py`
* `app/api/auth_routes.py`
* `app/api/automation_routes.py`
* `app/api/escalation_routes.py`
* `frontend/index.html`

---

## N. Environment Variables Required for Exotel

Add/update the following in [.env](file:///c:/Users/User/Downloads/ai_voice_calling_F/ai_voice_calling_F/.env):

```env
TELEPHONY_PROVIDER=exotel
EXOTEL_SID=rsl6a1
EXOTEL_KEY=<YOUR_EXOTEL_API_KEY>
EXOTEL_TOKEN=<YOUR_EXOTEL_API_TOKEN>
EXOTEL_SUBDOMAIN=api.exotel.com
EXOTEL_CALLER_ID=<YOUR_EXOPHONE_NUMBER>
TELEPHONY_WEBHOOK_BASE_URL=https://<your-ngrok-subdomain>.ngrok-free.dev
```

---

## O. Risks & Issues Discovered

1. **Exotel Basic Authentication**: Unlike Twilio (`ACCOUNT_SID:AUTH_TOKEN`), Exotel Connect API requires `API_KEY:API_TOKEN` for HTTP Basic Authentication.
2. **Exophone CallerID Validation**: Exotel Connect API requires `CallerId` to match an active Exophone provisioned on account `rsl6a1`.
3. **PSTN Response Latency**: Exotel webhooks drop calls if HTTP response exceeds 5 seconds. Ollama LLM warmup (`generate_response`) on startup is critical.
4. **Trial Account Restrictions**: Exotel trial accounts require destination phone numbers (`From` parameter in outbound calls) to be pre-verified in the Exotel Dashboard.

---

## P. Recommended Implementation Order

1. **Configure Environment**: Update `.env` with Exotel Account SID (`rsl6a1`), API Key, API Token, Exophone number, and `TELEPHONY_PROVIDER=exotel`.
2. **Refine Exotel Telephony Adapter**: Update `ExotelTelephonyAdapter` in `app/services/telephony_adapter.py` for API Basic Auth and ExotML / Voice XML generation.
3. **Verify Webhook Normalization**: Ensure `telephony_routes.py` handles both Exotel form keys (`CallSid`, `From`, `digits`) and Twilio form keys smoothly.
4. **Compile & Regression Test**: Run `py_compile` and `tests/test_complete_regression_13.py` to ensure zero regressions.
5. **Live Verification Call**: Trigger an outbound test call and dial the Exophone number to verify inbound IVR and AI speech turns live.
