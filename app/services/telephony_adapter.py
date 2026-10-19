import os
import uuid
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import httpx
from dotenv import load_dotenv

load_dotenv()


class TelephonyAdapter(ABC):
    """
    Abstract base class for Telephony Providers (Twilio, Exotel, Plivo, Mock).
    Defines the unified contract for outbound calling, webhook parsing, call termination,
    recording retrieval, and XML/TwiML response generation.
    """

    @abstractmethod
    def start_outbound_call(
        self, customer_id: int, phone_number: str, callback_url: Optional[str] = None, call_id: Optional[int] = None
    ) -> str:
        """
        Initiates an outbound carrier call.
        Returns the external provider's unique call/session ID.
        """
        pass

    @abstractmethod
    def handle_incoming_call(self, external_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes an incoming webhook from the telephony provider.
        Returns normalized call details:
        {
            "external_call_id": str,
            "customer_phone": str,
            "to_phone": str,
            "digits": Optional[str],
            "speech_text": Optional[str],
            "recording_url": Optional[str],
            "provider": str
        }
        """
        pass

    @abstractmethod
    def end_call(self, external_call_id: str) -> bool:
        """
        Sends a command to the telephony provider to hang up the call.
        """
        pass

    @abstractmethod
    def get_recording(self, external_call_id: str) -> Optional[str]:
        """
        Retrieves the recording URL from the telephony provider after call completion.
        """
        pass

    @abstractmethod
    def generate_carrier_response(
        self,
        action_type: str,
        text: Optional[str] = None,
        audio_url: Optional[str] = None,
        action_url: Optional[str] = None,
    ) -> str:
        """
        Generates provider-specific XML/TwiML response for voice control flow.
        action_type: 'ivr_prompt', 'play_and_hangup', 'gather_speech', 'hangup'
        """
        pass


class MockTelephonyAdapter(TelephonyAdapter):
    """
    Local mock implementation for testing and development environments
    without making real carrier phone calls.
    """

    def start_outbound_call(
        self, customer_id: int, phone_number: str, callback_url: Optional[str] = None, call_id: Optional[int] = None
    ) -> str:
        unique_suffix = uuid.uuid4().hex[:8]
        return f"mock_call_{customer_id}_{unique_suffix}"

    def handle_incoming_call(self, external_payload: Dict[str, Any]) -> Dict[str, Any]:
        call_sid = external_payload.get("CallSid") or external_payload.get("call_id") or uuid.uuid4().hex[:10]
        return {
            "external_call_id": f"mock_inbound_{call_sid}",
            "customer_phone": external_payload.get("From") or external_payload.get("phone") or "+919999999999",
            "to_phone": external_payload.get("To") or "+911800000000",
            "digits": external_payload.get("Digits") or external_payload.get("digit"),
            "speech_text": external_payload.get("SpeechResult"),
            "recording_url": external_payload.get("RecordingUrl"),
            "provider": "mock",
        }

    def end_call(self, external_call_id: str) -> bool:
        return True

    def get_recording(self, external_call_id: str) -> Optional[str]:
        return f"http://localhost:8000/audio/mock_recording_{external_call_id}.wav"

    def generate_carrier_response(
        self,
        action_type: str,
        text: Optional[str] = None,
        audio_url: Optional[str] = None,
        action_url: Optional[str] = None,
    ) -> str:
        if action_type == "ivr_prompt":
            return f'<Response><Say>{text or "Select language"}</Say></Response>'
        elif action_type == "play_and_hangup":
            return f'<Response><Play>{audio_url}</Play><Hangup/></Response>'
        elif action_type == "gather_speech":
            return f'<Response><Gather action="{action_url}"><Say>{text}</Say></Gather></Response>'
        return '<Response><Hangup/></Response>'


class TwilioTelephonyAdapter(TelephonyAdapter):
    """
    Twilio Telephony Adapter supporting TwiML response generation,
    REST API outbound calls, webhook parsing, call termination, and recording URLs.
    """

    def __init__(self):
        self.account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
        self.auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "")
        self.from_number = os.environ.get("TWILIO_PHONE_NUMBER", "+15005550006")
        self.webhook_base = os.environ.get("TELEPHONY_WEBHOOK_BASE_URL", "http://localhost:8000")

    def start_outbound_call(
        self, customer_id: int, phone_number: str, callback_url: Optional[str] = None, call_id: Optional[int] = None
    ) -> str:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Calls.json"
        target_url = callback_url or f"{self.webhook_base}/api/telephony/webhook/incoming"

        if self.account_sid and self.auth_token and not self.account_sid.startswith("mock"):
            try:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(
                        url,
                        data={
                            "To": phone_number,
                            "From": self.from_number,
                            "Url": target_url,
                        },
                        auth=(self.account_sid, self.auth_token),
                    )
                    if resp.status_code in (200, 201):
                        return resp.json().get("sid", f"CA{uuid.uuid4().hex[:28]}")
            except Exception as e:
                print(f"Warning: Twilio outbound call HTTP request error: {e}")

        # Fallback SID for sandbox/test environments
        return f"CA_twilio_{customer_id}_{uuid.uuid4().hex[:12]}"

    def handle_incoming_call(self, external_payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "external_call_id": external_payload.get("CallSid") or f"CA{uuid.uuid4().hex[:28]}",
            "customer_phone": external_payload.get("From") or "+910000000000",
            "to_phone": external_payload.get("To") or self.from_number,
            "digits": external_payload.get("Digits"),
            "speech_text": external_payload.get("SpeechResult"),
            "recording_url": external_payload.get("RecordingUrl"),
            "provider": "twilio",
        }

    def end_call(self, external_call_id: str) -> bool:
        if self.account_sid and self.auth_token and external_call_id.startswith("CA"):
            url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Calls/{external_call_id}.json"
            try:
                with httpx.Client(timeout=5.0) as client:
                    resp = client.post(
                        url,
                        data={"Status": "completed"},
                        auth=(self.account_sid, self.auth_token),
                    )
                    return resp.status_code == 200
            except Exception as e:
                print(f"Warning: Twilio end call error: {e}")
        return True

    def get_recording(self, external_call_id: str) -> Optional[str]:
        return f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Calls/{external_call_id}/Recordings"

    def generate_carrier_response(
        self,
        action_type: str,
        text: Optional[str] = None,
        audio_url: Optional[str] = None,
        action_url: Optional[str] = None,
    ) -> str:
        base_url = self.webhook_base.rstrip("/")
        if action_type == "ivr_prompt":
            act = action_url or f"{base_url}/api/telephony/webhook/dtmf"
            prompt_text = text or "Press 1 for English. Press 2 for Hindi. Press 3 for Marathi."
            return (
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<Response>\n'
                f'  <Gather numDigits="1" action="{act}" method="POST">\n'
                f'    <Say>{prompt_text}</Say>\n'
                '  </Gather>\n'
                f'  <Say>{prompt_text}</Say>\n'
                '</Response>'
            )
        elif action_type == "play_and_hangup":
            play_tag = f"  <Play>{audio_url}</Play>\n" if audio_url else ""
            say_tag = f"  <Say>{text}</Say>\n" if text else ""
            return (
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<Response>\n'
                f'{play_tag}'
                f'{say_tag}'
                '  <Hangup/>\n'
                '</Response>'
            )
        elif action_type == "gather_speech":
            act = action_url or f"{base_url}/api/telephony/webhook/turn"
            say_tag = f"  <Say>{text}</Say>\n" if text else ""
            play_tag = f"  <Play>{audio_url}</Play>\n" if audio_url else ""
            return (
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<Response>\n'
                f'{say_tag}'
                f'{play_tag}'
                f'  <Gather input="speech" action="{act}" method="POST" timeout="5"/>\n'
                '</Response>'
            )
        return '<?xml version="1.0" encoding="UTF-8"?>\n<Response>\n  <Hangup/>\n</Response>'


class ExotelTelephonyAdapter(TelephonyAdapter):
    """
    Exotel Telephony Adapter supporting Passthrough/Flow APIs, XML response generation,
    outbound call initiation, webhook parsing, and status callbacks.
    """

    def __init__(self):
        self.sid = os.environ.get("EXOTEL_SID", "")
        self.token = os.environ.get("EXOTEL_TOKEN", "")
        self.key = os.environ.get("EXOTEL_KEY", "")
        self.subdomain = os.environ.get("EXOTEL_SUBDOMAIN", "api.exotel.com")
        self.caller_id = os.environ.get("EXOTEL_CALLER_ID", "08000000000")
        self.webhook_base = os.environ.get("TELEPHONY_WEBHOOK_BASE_URL", "http://localhost:8000")

    def start_outbound_call(
        self, customer_id: int, phone_number: str, callback_url: Optional[str] = None, call_id: Optional[int] = None
    ) -> str:
        url = f"https://{self.subdomain}/v1/Accounts/{self.sid}/Calls/connect.json"
        
        # Build Exotel AgentStream / Connect Voice AI StreamUrl using WebSocket base URL
        ws_base = self.webhook_base.replace("http://", "ws://").replace("https://", "wss://").rstrip("/")
        cid = call_id or customer_id
        stream_url = callback_url or f"{ws_base}/api/telephony/ws/telephony/{cid}"

        masked_phone = (phone_number[:3] + "*****" + phone_number[-3:]) if len(phone_number) >= 6 else phone_number
        print(f"[EXOTEL OUTBOUND] Endpoint: https://{self.subdomain}/v1/Accounts/*****/Calls/connect.json")
        print(f"[EXOTEL OUTBOUND] Stream URL: {stream_url}")
        print(f"[EXOTEL OUTBOUND] Initiating call | Destination: {masked_phone} | ExoPhone: {self.caller_id}")

        if self.sid and (self.token or self.key) and not self.sid.startswith("mock"):
            auth_creds = (self.key, self.token) if self.key else (self.sid, self.token)
            try:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(
                        url,
                        data={
                            "From": phone_number,
                            "CallerId": self.caller_id,
                            "StreamUrl": stream_url,
                            "StreamType": "bidirectional",
                        },
                        auth=auth_creds,
                    )

                    print(f"[EXOTEL OUTBOUND] HTTP Status: {resp.status_code}")

                    try:
                        resp_data = resp.json()
                    except Exception:
                        resp_data = {}

                    call_sid = resp_data.get("Call", {}).get("Sid") or resp_data.get("sid")

                    if resp.status_code in (200, 201) and call_sid:
                        print(f"[EXOTEL OUTBOUND] Exotel accepted call | Call SID: {call_sid}")
                        return call_sid
                    else:
                        print(f"[EXOTEL OUTBOUND ERROR] HTTP Status: {resp.status_code} | Body: {resp.text}")
                        raise RuntimeError(f"Exotel API rejected call request with status {resp.status_code}: {resp.text}")
            except Exception as e:
                print(f"[EXOTEL OUTBOUND ERROR] HTTP Request Exception: {e}")
                raise

        fallback_sid = f"EX_exotel_{customer_id}_{uuid.uuid4().hex[:12]}"
        print(f"[EXOTEL OUTBOUND MOCK] Using mock Call SID: {fallback_sid}")
        return fallback_sid

    def handle_incoming_call(self, external_payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "external_call_id": external_payload.get("CallSid") or external_payload.get("Sid") or f"EX{uuid.uuid4().hex[:20]}",
            "customer_phone": external_payload.get("From") or external_payload.get("CallFrom") or "+910000000000",
            "to_phone": external_payload.get("To") or external_payload.get("CallTo") or self.caller_id,
            "digits": external_payload.get("digits") or external_payload.get("Digits"),
            "speech_text": external_payload.get("SpeechResult"),
            "recording_url": external_payload.get("RecordingUrl"),
            "provider": "exotel",
        }

    def end_call(self, external_call_id: str) -> bool:
        return True

    def get_recording(self, external_call_id: str) -> Optional[str]:
        return f"https://{self.subdomain}/v1/Accounts/{self.sid}/Recordings/{external_call_id}"

    def generate_carrier_response(
        self,
        action_type: str,
        text: Optional[str] = None,
        audio_url: Optional[str] = None,
        action_url: Optional[str] = None,
    ) -> str:
        if action_type == "ivr_prompt":
            prompt_text = text or "Press 1 for English. Press 2 for Hindi. Press 3 for Marathi."
            return (
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<Response>\n'
                f'  <Say>{prompt_text}</Say>\n'
                '</Response>'
            )
        elif action_type == "play_and_hangup":
            play_tag = f"  <Play>{audio_url}</Play>\n" if audio_url else ""
            say_tag = f"  <Say>{text}</Say>\n" if text else ""
            return (
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<Response>\n'
                f'{play_tag}'
                f'{say_tag}'
                '  <Hangup/>\n'
                '</Response>'
            )
        return '<?xml version="1.0" encoding="UTF-8"?>\n<Response>\n  <Hangup/>\n</Response>'


class PlivoTelephonyAdapter(TelephonyAdapter):
    """
    Plivo Telephony Adapter supporting XML response generation,
    outbound API calls, webhook parsing, and recording URLs.
    """

    def __init__(self):
        self.auth_id = os.environ.get("PLIVO_AUTH_ID", "")
        self.auth_token = os.environ.get("PLIVO_AUTH_TOKEN", "")
        self.from_number = os.environ.get("PLIVO_PHONE_NUMBER", "+14155550100")
        self.webhook_base = os.environ.get("TELEPHONY_WEBHOOK_BASE_URL", "http://localhost:8000")

    def start_outbound_call(
        self, customer_id: int, phone_number: str, callback_url: Optional[str] = None, call_id: Optional[int] = None
    ) -> str:
        url = f"https://api.plivo.com/v1/Account/{self.auth_id}/Call/"
        target_url = callback_url or f"{self.webhook_base}/api/telephony/webhook/incoming"

        if self.auth_id and self.auth_token and not self.auth_id.startswith("mock"):
            try:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(
                        url,
                        json={
                            "to": phone_number,
                            "from": self.from_number,
                            "answer_url": target_url,
                        },
                        auth=(self.auth_id, self.auth_token),
                    )
                    if resp.status_code in (200, 201):
                        return resp.json().get("request_uuid", f"PL{uuid.uuid4().hex[:20]}")
            except Exception as e:
                print(f"Warning: Plivo outbound call error: {e}")

        return f"PL_plivo_{customer_id}_{uuid.uuid4().hex[:12]}"

    def handle_incoming_call(self, external_payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "external_call_id": external_payload.get("CallUUID") or external_payload.get("RequestUUID") or f"PL{uuid.uuid4().hex[:20]}",
            "customer_phone": external_payload.get("From") or "+910000000000",
            "to_phone": external_payload.get("To") or self.from_number,
            "digits": external_payload.get("Digits"),
            "speech_text": external_payload.get("SpeechResult"),
            "recording_url": external_payload.get("RecordUrl"),
            "provider": "plivo",
        }

    def end_call(self, external_call_id: str) -> bool:
        if self.auth_id and self.auth_token and external_call_id.startswith("PL"):
            url = f"https://api.plivo.com/v1/Account/{self.auth_id}/Call/{external_call_id}/"
            try:
                with httpx.Client(timeout=5.0) as client:
                    resp = client.delete(url, auth=(self.auth_id, self.auth_token))
                    return resp.status_code in (200, 204)
            except Exception as e:
                print(f"Warning: Plivo end call error: {e}")
        return True

    def get_recording(self, external_call_id: str) -> Optional[str]:
        return f"https://api.plivo.com/v1/Account/{self.auth_id}/Recording/{external_call_id}/"

    def generate_carrier_response(
        self,
        action_type: str,
        text: Optional[str] = None,
        audio_url: Optional[str] = None,
        action_url: Optional[str] = None,
    ) -> str:
        if action_type == "ivr_prompt":
            act = action_url or f"{self.webhook_base}/api/telephony/webhook/dtmf"
            prompt_text = text or "Press 1 for English. Press 2 for Hindi. Press 3 for Marathi."
            return (
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<Response>\n'
                f'  <GetDigits action="{act}" method="POST" numDigits="1">\n'
                f'    <Speak>{prompt_text}</Speak>\n'
                '  </GetDigits>\n'
                '</Response>'
            )
        elif action_type == "play_and_hangup":
            play_tag = f"  <Play>{audio_url}</Play>\n" if audio_url else ""
            speak_tag = f"  <Speak>{text}</Speak>\n" if text else ""
            return (
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<Response>\n'
                f'{play_tag}'
                f'{speak_tag}'
                '  <Hangup/>\n'
                '</Response>'
            )
        return '<?xml version="1.0" encoding="UTF-8"?>\n<Response>\n  <Hangup/>\n</Response>'


def get_telephony_adapter(provider_name: Optional[str] = None) -> TelephonyAdapter:
    """
    Factory function to retrieve the configured TelephonyAdapter instance.
    Defaults to TELEPHONY_PROVIDER env variable or 'mock'.
    """
    p_name = (provider_name or os.environ.get("TELEPHONY_PROVIDER", "mock")).lower().strip()
    if p_name == "twilio":
        return TwilioTelephonyAdapter()
    elif p_name == "exotel":
        return ExotelTelephonyAdapter()
    elif p_name == "plivo":
        return PlivoTelephonyAdapter()
    return MockTelephonyAdapter()
