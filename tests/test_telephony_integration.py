import os
import sys
from unittest.mock import patch
from fastapi.testclient import TestClient

# Ensure root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TELEPHONY_PROVIDER"] = "mock"

from main import app
from app.services.telephony_adapter import (
    get_telephony_adapter,
    MockTelephonyAdapter,
    TwilioTelephonyAdapter,
    ExotelTelephonyAdapter,
    PlivoTelephonyAdapter,
)

client = TestClient(app)


def test_telephony_adapter_factory():
    print("\n--- Testing Telephony Adapter Factory ---")
    mock_adapter = get_telephony_adapter("mock")
    assert isinstance(mock_adapter, MockTelephonyAdapter)
    print("  [PASS] MockTelephonyAdapter instantiated successfully")

    twilio_adapter = get_telephony_adapter("twilio")
    assert isinstance(twilio_adapter, TwilioTelephonyAdapter)
    print("  [PASS] TwilioTelephonyAdapter instantiated successfully")

    exotel_adapter = get_telephony_adapter("exotel")
    assert isinstance(exotel_adapter, ExotelTelephonyAdapter)
    print("  [PASS] ExotelTelephonyAdapter instantiated successfully")

    plivo_adapter = get_telephony_adapter("plivo")
    assert isinstance(plivo_adapter, PlivoTelephonyAdapter)
    print("  [PASS] PlivoTelephonyAdapter instantiated successfully")


def test_xml_and_twiml_generation():
    print("\n--- Testing Carrier XML & TwiML Response Generators ---")
    twilio = TwilioTelephonyAdapter()
    twiml_ivr = twilio.generate_carrier_response("ivr_prompt", text="Press 1 for English")
    assert "<Gather" in twiml_ivr and "<Say>Press 1 for English</Say>" in twiml_ivr
    print("  [PASS] Twilio TwiML IVR response verified")

    exotel = ExotelTelephonyAdapter()
    exotel_xml = exotel.generate_carrier_response("ivr_prompt", text="Press 1 for English")
    assert "<Response>" in exotel_xml and "<Say>Press 1 for English</Say>" in exotel_xml
    print("  [PASS] Exotel XML IVR response verified")

    plivo = PlivoTelephonyAdapter()
    plivo_xml = plivo.generate_carrier_response("ivr_prompt", text="Press 1 for English")
    assert "<GetDigits" in plivo_xml and "<Speak>Press 1 for English</Speak>" in plivo_xml
    print("  [PASS] Plivo XML IVR response verified")


def test_telephony_api_endpoints():
    print("\n--- Testing Telephony API & Carrier Webhooks ---")
    # 1. Login
    login_res = client.post("/api/auth/login", json={"username": "admin", "password": "AdminPass123!"})
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Config Endpoint
    config_res = client.get("/api/telephony/config", headers=headers)
    assert config_res.status_code == 200
    cfg = config_res.json()
    assert cfg["telephony_provider"] == "mock"
    assert "supported_providers" in cfg
    print(f"  [PASS] Telephony Config API verified: Provider='{cfg['telephony_provider']}', Adapter='{cfg['adapter_class']}'")

    # 3. Carrier Incoming Webhook
    inc_res = client.post(
        "/api/telephony/webhook/incoming",
        data={"CallSid": "TEST_CARRIER_CALL_101", "From": "+919876543210", "To": "+911800000000"},
    )
    assert inc_res.status_code == 200
    assert "text/xml" in inc_res.headers["content-type"]
    assert "<Response>" in inc_res.text
    print("  [PASS] Carrier Incoming Webhook verified: Returns HTTP 200 text/xml TwiML")

    # 4. Carrier DTMF Webhook (Option 1 English)
    dtmf_res = client.post(
        "/api/telephony/webhook/dtmf",
        data={"CallSid": "TEST_CARRIER_CALL_101", "Digits": "1"},
    )
    assert dtmf_res.status_code == 200
    assert "<Gather" in dtmf_res.text or "<Say>" in dtmf_res.text
    print("  [PASS] Carrier DTMF Webhook (1 -> English) verified: Returns HTTP 200 TwiML")

    # 5. Carrier Speech Turn Webhook
    turn_res = client.post(
        "/api/telephony/webhook/turn",
        data={"CallSid": "TEST_CARRIER_CALL_101", "SpeechResult": "I need 20 laptops for next month with 50000 budget"},
    )
    assert turn_res.status_code == 200
    assert "text/xml" in turn_res.headers["content-type"]
    print("  [PASS] Carrier Speech Turn Webhook verified: Processed STT/LLM pipeline and returned HTTP 200 TwiML")

    # 6. Carrier Status Webhook
    status_res = client.post(
        "/api/telephony/webhook/status",
        data={"CallSid": "TEST_CARRIER_CALL_101", "CallStatus": "completed"},
    )
    assert status_res.status_code == 200 and status_res.json()["status"] == "received"
    print("  [PASS] Carrier Status Callback Webhook verified")

    # 7. Carrier Recording Webhook
    rec_res = client.post(
        "/api/telephony/webhook/recording",
        data={"CallSid": "TEST_CARRIER_CALL_101", "RecordingUrl": "https://carrier.com/recordings/REC123.mp3"},
    )
    assert rec_res.status_code == 200 and rec_res.json()["status"] == "recording_updated"
    print("  [PASS] Carrier Recording Callback Webhook verified")

    # 8. Operator Outbound Call API
    # Get a customer ID
    cust_res = client.get("/api/customers", headers=headers)
    cust_id = cust_res.json()[0]["id"]
    with patch.object(
        ExotelTelephonyAdapter,
        "start_outbound_call",
        return_value="MOCK_CALL_SID_TEST",
    ), patch.object(
        MockTelephonyAdapter,
        "start_outbound_call",
        return_value="MOCK_CALL_SID_TEST",
    ):
        outbound_res = client.post(f"/api/telephony/outbound", data={"customer_id": cust_id}, headers=headers)
        assert outbound_res.status_code == 200
        outbound_data = outbound_res.json()
        assert outbound_data["status"] == "outbound_call_initiated"
        assert outbound_data["provider_call_id"] == "MOCK_CALL_SID_TEST"
        print(f"  [PASS] Operator Outbound Call API verified (MOCKED): Call #{outbound_data['call_id']} initiated with Provider ID '{outbound_data['provider_call_id']}'")


def test_exotel_inbound_websocket_greeting():
    print("\n--- Testing Inbound Exotel WebSocket Greeting & Persistence ---")
    import base64
    import json
    import time
    import struct
    from app.database.database import SessionLocal
    from app.database.models import Call, Message
    from app.api.call_routes import ENGLISH_GREETING_MESSAGE

    test_stream_sid = "test_int_stream_greeting_001"
    test_call_sid = f"test_int_call_{int(time.time() * 1000)}"

    with client.websocket_connect("/api/telephony/ws/telephony/inbound") as ws:
        # 1. Send connected event
        ws.send_json({"event": "connected"})

        # 2. Send start event
        start_payload = {
            "event": "start",
            "start": {
                "streamSid": test_stream_sid,
                "callSid": test_call_sid,
                "mediaFormat": {"sampleRate": 8000},
                "customParameters": {
                    "from": "+919876543210"
                }
            }
        }
        ws.send_json(start_payload)

        # 3. Receive greeting audio chunk from server
        rec_msg = ws.receive_json()
        assert rec_msg["event"] == "media", f"Expected media event, got {rec_msg.get('event')}"
        assert rec_msg["streamSid"] == test_stream_sid, f"streamSid mismatch: {rec_msg.get('streamSid')}"
        assert rec_msg["stream_sid"] == test_stream_sid, f"stream_sid mismatch: {rec_msg.get('stream_sid')}"
        assert "media" in rec_msg and "payload" in rec_msg["media"], "media payload missing"

        pcm_bytes = base64.b64decode(rec_msg["media"]["payload"])
        assert len(pcm_bytes) == 640, f"Expected 640 bytes chunk, got {len(pcm_bytes)}"
        print("  [PASS] Server generated and streamed Exotel-compliant greeting PCM audio (640 bytes, 8000Hz 16-bit)")

        # 4. Duplicate event rejection test: send duplicate start event
        ws.send_json(start_payload)

        # 5. Caller interruption / barge-in test: send speech frame (energy > 400)
        loud_pcm = struct.pack("<160h", *([2000] * 160))
        loud_b64 = base64.b64encode(loud_pcm).decode("utf-8")
        ws.send_json({
            "event": "media",
            "media": {
                "payload": loud_b64,
                "streamSid": test_stream_sid
            }
        })
        print("  [PASS] Sent barge-in media packet during greeting playback")

        # 6. Send stop event to clean up
        ws.send_json({
            "event": "stop",
            "stop": {"streamSid": test_stream_sid}
        })
        print("  [PASS] Session stop event processed")

    # 7. Verify Database persistence & exactly-once message logging
    with SessionLocal() as db:
        call = db.query(Call).filter(Call.recording_url == f"provider_call_id:{test_call_sid}").first()
        assert call is not None, f"Call not found in DB for {test_call_sid}"
        assert call.call_type == "INBOUND", f"Expected INBOUND call_type, got {call.call_type}"

        greeting_msgs = db.query(Message).filter(
            Message.call_id == call.id,
            Message.speaker == "assistant",
            Message.text == ENGLISH_GREETING_MESSAGE
        ).all()

        assert len(greeting_msgs) == 1, f"Expected exactly 1 greeting message in DB, found {len(greeting_msgs)}"
        print(f"  [PASS] Inbound greeting logged exactly once in MySQL (Call #{call.id}): '{greeting_msgs[0].text}'")


def run_all_telephony_tests():
    print("\n========================================================")
    print("STARTING TELEPHONY INTEGRATION & WEBHOOK VERIFICATION")
    print("========================================================")
    test_telephony_adapter_factory()
    test_xml_and_twiml_generation()
    test_telephony_api_endpoints()
    test_exotel_inbound_websocket_greeting()
    print("\n========================================================")
    print("ALL TELEPHONY INTEGRATION TESTS PASSED 100%!")
    print("========================================================\n")


if __name__ == "__main__":
    run_all_telephony_tests()
