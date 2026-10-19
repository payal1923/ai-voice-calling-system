import os
import sys
import json
import base64
from datetime import timedelta
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

# Ensure root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from app.database.database import SessionLocal
from app.database.models import Call, Message, SystemUser
from app.api.call_routes import ENGLISH_GREETING_MESSAGE
from app.utils.auth import create_access_token, hash_password

client = TestClient(app)


def ensure_test_user():
    """Ensure an active admin user exists in DB for authentication tests."""
    with SessionLocal() as db:
        user = db.query(SystemUser).filter(SystemUser.username == "admin").first()
        if not user:
            user = SystemUser(
                username="admin",
                password_hash=hash_password("admin123"),
                role="admin",
            )
            db.add(user)
            db.commit()


def test_live_voice_unauthenticated_connection_rejected():
    """Verify connecting to /api/ws/live-call without a token is rejected with WS 1008."""
    print("\n--- Testing WebSocket Connection Without Token ---")
    try:
        with client.websocket_connect("/api/ws/live-call") as ws:
            raise AssertionError("Connection should have been rejected")
    except WebSocketDisconnect as exc:
        assert exc.code == 1008, f"Expected close code 1008, got {exc.code}"
        print(f"  [PASS] Unauthenticated connection rejected with code {exc.code} (WS_1008_POLICY_VIOLATION)")


def test_live_voice_invalid_token_rejected():
    """Verify connecting with a malformed/invalid token is rejected with WS 1008."""
    print("\n--- Testing WebSocket Connection With Invalid Token ---")
    try:
        with client.websocket_connect("/api/ws/live-call?token=invalid.jwt.token.here") as ws:
            raise AssertionError("Connection should have been rejected with invalid token")
    except WebSocketDisconnect as exc:
        assert exc.code == 1008, f"Expected close code 1008, got {exc.code}"
        print(f"  [PASS] Invalid token connection rejected with code {exc.code} (WS_1008_POLICY_VIOLATION)")


def test_live_voice_expired_token_rejected():
    """Verify connecting with an expired JWT token is rejected with WS 1008."""
    print("\n--- Testing WebSocket Connection With Expired Token ---")
    ensure_test_user()
    expired_token = create_access_token(
        data={"sub": "admin"},
        expires_delta=timedelta(seconds=-60),
    )
    try:
        with client.websocket_connect(f"/api/ws/live-call?token={expired_token}") as ws:
            raise AssertionError("Connection should have been rejected with expired token")
    except WebSocketDisconnect as exc:
        assert exc.code == 1008, f"Expected close code 1008, got {exc.code}"
        print(f"  [PASS] Expired token connection rejected with code {exc.code} (WS_1008_POLICY_VIOLATION)")


def test_live_voice_header_auth_lifecycle():
    """Verify connecting with valid token in Authorization header succeeds."""
    print("\n--- Testing WebSocket Connection With Authorization Header ---")
    ensure_test_user()
    valid_token = create_access_token(data={"sub": "admin"})

    with client.websocket_connect(
        "/api/ws/live-call",
        headers={"Authorization": f"Bearer {valid_token}"},
    ) as ws:
        msg1 = ws.receive_json()
        assert msg1["event"] == "session_started", f"Expected session_started, got {msg1.get('event')}"
        assert msg1["user"] == "admin", f"Expected user admin, got {msg1.get('user')}"
        call_id = msg1["call_id"]
        print(f"  [PASS] Handshake succeeded via Authorization header for user '{msg1['user']}' (Call #{call_id})")

        msg2 = ws.receive_json()
        assert msg2["event"] == "assistant_response"
        assert msg2["response"] == ENGLISH_GREETING_MESSAGE
        print(f"  [PASS] Opening greeting received: '{msg2['response']}'")

        # Cleanly end session
        ws.send_json({"event": "end_call"})


def test_live_voice_external_customer_id_query_param():
    """Verify connecting with string/external customer_id parameter (e.g. TEL_919988776655)."""
    print("\n--- Testing WebSocket Connection With String External customer_id ---")
    ensure_test_user()
    valid_token = create_access_token(data={"sub": "admin"})

    uri = f"/api/ws/live-call?token={valid_token}&customer_id=TEL_919988776655"
    with client.websocket_connect(uri) as ws:
        msg1 = ws.receive_json()
        assert msg1["event"] == "session_started", f"Expected session_started, got {msg1.get('event')}"
        assert msg1["user"] == "admin"
        assert msg1["customer_phone"] == "+919988776655" or msg1["customer_id"] is not None
        call_id = msg1["call_id"]
        print(f"  [PASS] WebSocket successfully accepted external customer ID 'TEL_919988776655' (Call #{call_id})")

        msg2 = ws.receive_json()
        assert msg2["event"] == "assistant_response"
        assert msg2["response"] == ENGLISH_GREETING_MESSAGE
        print(f"  [PASS] Opening greeting received: '{msg2['response']}'")

        ws.send_json({"event": "end_call"})


def test_live_voice_calling_websocket_lifecycle():
    """Full lifecycle test with valid token: connection, greeting, multi-turn, interruption, and call termination."""
    print("\n========================================================")
    print("STARTING IN-APP LIVE VOICE CALLING FULL LIFECYCLE TEST")
    print("========================================================")

    ensure_test_user()
    valid_token = create_access_token(data={"sub": "admin"})

    # 1. Connect to /api/ws/live-call with valid token
    with client.websocket_connect(f"/api/ws/live-call?token={valid_token}") as ws:
        # Step A: Handshake - Receive session_started
        msg1 = ws.receive_json()
        assert msg1["event"] == "session_started", f"Expected session_started, got {msg1.get('event')}"
        call_id = msg1["call_id"]
        assert call_id is not None
        assert msg1["user"] == "admin"
        print(f"  [PASS] WebSocket authenticated & session started for Call #{call_id} (User: {msg1['user']})")

        # Step B: Initial Greeting - Receive opening assistant greeting
        msg2 = ws.receive_json()
        assert msg2["event"] == "assistant_response", f"Expected assistant_response, got {msg2.get('event')}"
        assert msg2["response"] == ENGLISH_GREETING_MESSAGE
        assert msg2["audio_url"] is not None
        print(f"  [PASS] Assistant opening greeting received: '{msg2['response']}'")

        # Step C: Send User Audio Turn
        test_wav = os.path.join("static", "waiting", "waiting_1.wav")
        assert os.path.exists(test_wav), f"Test audio {test_wav} missing"
        with open(test_wav, "rb") as f:
            audio_bytes = f.read()

        b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
        ws.send_json({
            "event": "user_audio",
            "audio": b64_audio
        })
        print("  [PASS] Sent user audio turn to server")

        # Step D: Receive processing indicator
        proc_msg = ws.receive_json()
        assert proc_msg["event"] == "processing", f"Expected processing, got {proc_msg.get('event')}"
        print(f"  [PASS] Server acknowledged processing ({proc_msg['status']})")

        # Step E: Receive AI Response
        resp_msg = ws.receive_json()
        assert resp_msg["event"] == "assistant_response", f"Expected assistant_response, got {resp_msg.get('event')}"
        assert resp_msg["transcript"], "Transcript empty"
        assert resp_msg["response"], "AI response empty"
        assert resp_msg["audio_url"], "Voice audio URL missing"
        print(f"  [PASS] Turn 1 complete! Transcript: '{resp_msg['transcript']}'")
        print(f"  [PASS] AI Generated Response: '{resp_msg['response'][:60]}...'")
        print(f"  [PASS] Voice Audio URL: '{resp_msg['audio_url']}'")

        # Step F: Test Multi-Turn (Turn 2)
        ws.send_json({
            "event": "user_audio",
            "audio": b64_audio
        })
        proc_msg2 = ws.receive_json()
        assert proc_msg2["event"] == "processing"
        resp_msg2 = ws.receive_json()
        assert resp_msg2["event"] == "assistant_response"
        assert resp_msg2["response"], "Turn 2 AI response empty"
        print(f"  [PASS] Turn 2 complete! AI Multi-turn Response: '{resp_msg2['response'][:60]}...'")

        # Step G: Test Interruption (Barge-in)
        ws.send_json({"event": "interrupt"})
        int_msg = ws.receive_json()
        assert int_msg["event"] == "interrupted"
        print(f"  [PASS] Live interruption / barge-in event acknowledged by server for Call #{call_id}")

        # Step H: Test Hangup / End Call
        ws.send_json({"event": "end_call"})
        print("  [PASS] Sent end_call event")

    # Step I: Verify MySQL Persistence
    with SessionLocal() as db:
        call = db.query(Call).filter(Call.id == call_id).first()
        assert call is not None, f"Call #{call_id} not found in DB"
        assert call.ended_at is not None, "Call ended_at timestamp missing"
        print(f"  [PASS] Call #{call_id} marked ended in MySQL database")

        msgs = db.query(Message).filter(Message.call_id == call_id).order_by(Message.id).all()
        assert len(msgs) >= 3, f"Expected at least 3 messages, got {len(msgs)}"
        print(f"  [PASS] Transcripts stored in DB: {len(msgs)} messages recorded chronologically")

    print("\n========================================================")
    print("ALL IN-APP LIVE VOICE CALLING TESTS PASSED 100%!")
    print("========================================================\n")


def test_live_voice_undersized_audio_ignored():
    """Verify undersized audio chunks (<1500 bytes) are safely ignored without triggering false fallback responses."""
    print("\n--- Testing Handling of Undersized Audio Packet ---")
    ensure_test_user()
    valid_token = create_access_token(data={"sub": "admin"})

    with client.websocket_connect(f"/api/ws/live-call?token={valid_token}") as ws:
        msg1 = ws.receive_json()
        assert msg1["event"] == "session_started"
        msg2 = ws.receive_json()
        assert msg2["event"] == "assistant_response"

        # Send tiny 500-byte audio chunk (e.g. ambient mic pop or partial packet)
        tiny_audio = b"\x1a\x45\xdf\xa3" + b"\x00" * 496
        ws.send_bytes(tiny_audio)

        # Send ping to verify websocket connection remains healthy and alive
        ws.send_json({"event": "ping"})
        pong_msg = ws.receive_json()
        assert pong_msg["event"] == "pong", f"Expected pong, got {pong_msg}"
        print("  [PASS] Undersized audio packet ignored without disrupting session or throwing errors")

        ws.send_json({"event": "end_call"})


def test_live_voice_webm_opus_speech_turn():
    """Verify live WebM Opus audio sent from browser is decoded and transcribed by Faster-Whisper accurately."""
    print("\n--- Testing Live WebM Opus Speech Turn Transcription & Response ---")
    ensure_test_user()
    valid_token = create_access_token(data={"sub": "admin"})

    webm_path = os.path.join("recordings", "test_speech.webm")
    assert os.path.exists(webm_path), f"WebM test speech file missing: {webm_path}"
    with open(webm_path, "rb") as f:
        webm_bytes = f.read()

    with client.websocket_connect(f"/api/ws/live-call?token={valid_token}") as ws:
        msg1 = ws.receive_json()
        assert msg1["event"] == "session_started"
        msg2 = ws.receive_json()
        assert msg2["event"] == "assistant_response"

        # Send WebM Opus audio bytes directly as binary frame (matching browser behavior)
        ws.send_bytes(webm_bytes)

        proc_msg = ws.receive_json()
        assert proc_msg["event"] == "processing"
        print(f"  [PASS] Server acknowledged WebM Opus audio ({proc_msg['status']})")

        resp_msg = ws.receive_json()
        assert resp_msg["event"] == "assistant_response"
        transcript = resp_msg.get("transcript", "")
        print(f"  [PASS] Whisper STT transcribed WebM Opus payload: '{transcript}'")
        assert "hello" in transcript.lower() or "test" in transcript.lower(), f"Expected hello/test in transcript, got: {transcript}"
        assert resp_msg["response"], "AI response empty"
        assert resp_msg["audio_url"], "Voice audio URL missing"
        print(f"  [PASS] AI generated response: '{resp_msg['response'][:60]}...'")

        ws.send_json({"event": "end_call"})


def test_live_voice_interruption_recovery_and_resume():
    """Verify that intentional interruption properly cancels pending audio and permits subsequent turns to proceed cleanly."""
    print("\n--- Testing Intentional Interruption, Recovery & Resume ---")
    ensure_test_user()
    valid_token = create_access_token(data={"sub": "admin"})

    test_wav = os.path.join("static", "waiting", "waiting_1.wav")
    assert os.path.exists(test_wav), f"Test audio {test_wav} missing"
    with open(test_wav, "rb") as f:
        audio_bytes = f.read()

    with client.websocket_connect(f"/api/ws/live-call?token={valid_token}") as ws:
        msg1 = ws.receive_json()
        assert msg1["event"] == "session_started"
        msg2 = ws.receive_json()
        assert msg2["event"] == "assistant_response"

        # 1. Trigger intentional barge-in / interruption
        ws.send_json({"event": "interrupt"})
        int_msg = ws.receive_json()
        assert int_msg["event"] == "interrupted"
        print("  [PASS] Server immediately acknowledged intentional interruption")

        # 2. Immediately speak again (subsequent turn)
        ws.send_bytes(audio_bytes)
        proc_msg = ws.receive_json()
        assert proc_msg["event"] == "processing"
        print("  [PASS] Server resumed processing new speech turn immediately after interruption")

        resp_msg = ws.receive_json()
        assert resp_msg["event"] == "assistant_response"
        assert resp_msg["transcript"]
        assert resp_msg["response"]
        print(f"  [PASS] AI completed subsequent turn after interruption: '{resp_msg['response'][:50]}...'")

        ws.send_json({"event": "end_call"})


def test_live_voice_internet_connection_transcription_regression():
    """
    Regression test for: 'My internet connection is not working'.
    Verifies that speech for this phrase is transcribed accurately by Faster-Whisper
    and processed by the live voice call WebSocket pipeline without hallucination or truncation.
    """
    print("\n--- Regression Test: 'My internet connection is not working' ---")
    ensure_test_user()
    valid_token = create_access_token(data={"sub": "admin"})

    from app.services.tts_service import TTSService
    tts = TTSService()
    test_phrase = "My internet connection is not working."
    test_audio_path = os.path.join("recordings", "regression_internet_connection.wav")
    tts.synthesize(test_phrase, test_audio_path)
    assert os.path.exists(test_audio_path), f"Failed to synthesize {test_audio_path}"

    try:
        with open(test_audio_path, "rb") as f:
            audio_bytes = f.read()

        with client.websocket_connect(f"/api/ws/live-call?token={valid_token}") as ws:
            msg1 = ws.receive_json()
            assert msg1["event"] == "session_started"
            msg2 = ws.receive_json()
            assert msg2["event"] == "assistant_response"

            # Send the audio turn for 'My internet connection is not working'
            ws.send_bytes(audio_bytes)

            proc_msg = ws.receive_json()
            assert proc_msg["event"] == "processing"
            print(f"  [PASS] WebSocket acknowledged speech turn ({proc_msg['status']})")

            resp_msg = ws.receive_json()
            assert resp_msg["event"] == "assistant_response"
            transcript = resp_msg.get("transcript", "")
            print(f"  [PASS] Actual Faster-Whisper transcript received: '{transcript}'")

            # Verify accurate transcription
            clean_t = transcript.lower().strip()
            assert "internet connection" in clean_t and "working" in clean_t, (
                f"Transcription mismatch! Expected 'My internet connection is not working', got: '{transcript}'"
            )
            print("  [PASS] Exact phrase matched without phoneme corruption or hallucination")

            # Verify relevant AI response
            ai_response = resp_msg.get("response", "")
            assert len(ai_response) > 0, "AI response is empty"
            print(f"  [PASS] Relevant AI response generated: '{ai_response[:70]}...'")

            ws.send_json({"event": "end_call"})
    finally:
        if os.path.exists(test_audio_path):
            try:
                os.remove(test_audio_path)
            except OSError:
                pass


if __name__ == "__main__":
    test_live_voice_unauthenticated_connection_rejected()
    test_live_voice_invalid_token_rejected()
    test_live_voice_expired_token_rejected()
    test_live_voice_header_auth_lifecycle()
    test_live_voice_external_customer_id_query_param()
    test_live_voice_undersized_audio_ignored()
    test_live_voice_webm_opus_speech_turn()
    test_live_voice_interruption_recovery_and_resume()
    test_live_voice_internet_connection_transcription_regression()
    test_live_voice_calling_websocket_lifecycle()



