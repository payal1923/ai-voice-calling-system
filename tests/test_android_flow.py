import os
import sys
import json
import base64
import unittest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from app.utils.auth import decode_access_token
from app.database.database import SessionLocal
from app.database.models import Customer, Call, SystemUser

client = TestClient(app)


class TestPublicCallerAndAndroidFlow(unittest.TestCase):

    def test_01_public_call_page_available(self):
        """Verify GET /call serves the original Voice Operations Hub with integrated live voice calling."""
        resp = client.get("/call")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers["content-type"])
        body = resp.text
        self.assertIn('id="view-live-call"', body)
        self.assertIn('id="live-voice-orb"', body)
        self.assertIn('id="btn-sip-call"', body)
        self.assertIn('id="live-transcript-box"', body)
        self.assertIn('id="webrtc-remote-audio"', body)

    def test_02_caller_session_token_issuance(self):
        """Verify POST /api/auth/caller-session generates valid, scoped caller JWT."""
        payload = {
            "client_type": "android",
            "caller_name": "Mobile Caller",
            "device_id": "test_pixel_device"
        }
        resp = client.post("/api/auth/caller-session", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("access_token", data)
        self.assertEqual(data["token_type"], "bearer")
        self.assertEqual(data["role"], "caller")
        self.assertEqual(data["client_type"], "android")
        self.assertTrue(data["caller_id"].startswith("caller_"))

        # Decode token and verify claims
        decoded = decode_access_token(data["access_token"])
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded["role"], "caller")
        self.assertEqual(decoded["sub"], data["caller_id"])
        self.assertEqual(decoded["client_type"], "android")

    def test_03_caller_cannot_access_private_customer_data(self):
        """Verify callers CANNOT access private customer data or admin endpoints (Requirement 8 & 9)."""
        # Obtain caller token
        resp = client.post("/api/auth/caller-session", json={"client_type": "web"})
        self.assertEqual(resp.status_code, 200)
        caller_token = resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {caller_token}"}

        # Attempt to access Customers endpoint
        cust_resp = client.get("/api/customers", headers=headers)
        self.assertEqual(cust_resp.status_code, 401, "Caller must not have access to customer records")

        # Attempt to access Users list
        user_resp = client.get("/api/auth/users", headers=headers)
        self.assertEqual(user_resp.status_code, 401, "Caller must not have access to system users")

        # Attempt to access Automation
        auto_resp = client.get("/api/automation/status", headers=headers)
        self.assertEqual(auto_resp.status_code, 401, "Caller must not have access to automation operations")

    def test_04_caller_session_rate_limiting(self):
        """Verify IP-based rate limiting on /api/auth/caller-session protects against abuse."""
        # Rapidly make requests until rate limit triggers
        hit_rate_limit = False
        for i in range(25):
            res = client.post(
                "/api/auth/caller-session",
                json={"client_type": "android"},
                headers={"X-Forwarded-For": "198.51.100.99"}  # Fixed test IP
            )
            if res.status_code == 429:
                hit_rate_limit = True
                self.assertIn("Too many caller session requests", res.json().get("detail", ""))
                break

        self.assertTrue(hit_rate_limit, "Rate limiter should trigger HTTP 429 after exceeding limit")

    def test_05_caller_websocket_session_isolation(self):
        """Verify caller can connect to WebSocket and is strictly isolated into their own customer record."""
        # 1. Obtain caller token
        res = client.post(
            "/api/auth/caller-session",
            json={"client_type": "android", "caller_name": "Test Android User"},
            headers={"X-Forwarded-For": "203.0.113.50"}
        )
        self.assertEqual(res.status_code, 200)
        token = res.json()["access_token"]
        caller_id = res.json()["caller_id"]

        # 2. Connect to WebSocket
        with client.websocket_connect(f"/api/ws/live-call?token={token}") as ws:
            # First event: session_started
            session_msg = ws.receive_json()
            self.assertEqual(session_msg.get("event"), "session_started")
            call_id = session_msg.get("call_id")
            self.assertIsNotNone(call_id)
            self.assertEqual(session_msg.get("user"), caller_id)

            # Second event: initial assistant greeting
            greeting_msg = ws.receive_json()
            self.assertEqual(greeting_msg.get("event"), "assistant_response")
            self.assertIn("AI voice assistant", greeting_msg.get("response", ""))

            # 3. Verify in database: Call was assigned to dedicated isolated customer record
            with SessionLocal() as db:
                call_record = db.query(Call).filter(Call.id == call_id).first()
                self.assertIsNotNone(call_record)
                customer = call_record.customer
                self.assertIsNotNone(customer)
                self.assertEqual(customer.external_customer_id, f"CALLER_{caller_id}")
                self.assertIn("Test Android User", customer.name)

            # 4. Test Interruption
            ws.send_json({"event": "interrupt"})
            intr_msg = ws.receive_json()
            self.assertEqual(intr_msg.get("event"), "interrupted")

            # 5. Clean Hangup
            ws.send_json({"event": "end_call"})

    def test_06_android_client_audio_delivery_and_response(self):
        """Verify sending audio bytes from Android client receives spoken response."""
        res = client.post(
            "/api/auth/caller-session",
            json={"client_type": "android", "caller_name": "Pixel 8 Tester"},
            headers={"X-Forwarded-For": "203.0.113.51"}
        )
        self.assertEqual(res.status_code, 200)
        token = res.json()["access_token"]

        with client.websocket_connect(f"/api/ws/live-call?token={token}") as ws:
            # Drain handshake
            ws.receive_json() # session_started
            ws.receive_json() # greeting

            # Send a synthesized audio turn (using waiting_1.wav)
            sample_wav = os.path.join("static", "waiting", "waiting_1.wav")
            if os.path.exists(sample_wav):
                with open(sample_wav, "rb") as f:
                    audio_bytes = f.read()

                ws.send_bytes(audio_bytes)

                # Expect thinking event
                processing_msg = ws.receive_json()
                self.assertEqual(processing_msg.get("event"), "processing")

                # Expect assistant response with audio URL
                resp_msg = ws.receive_json()
                self.assertEqual(resp_msg.get("event"), "assistant_response")
                self.assertTrue(len(resp_msg.get("response", "")) > 0)
                self.assertTrue(resp_msg.get("audio_url", "").startswith("/audio/"))

            ws.send_json({"event": "end_call"})


if __name__ == "__main__":
    unittest.main()
