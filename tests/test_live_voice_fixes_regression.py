import os
import sys
import asyncio

sys.path.insert(0, os.path.abspath("."))

from app.services.stt_service import STTService
from app.services.response_service import ResponseService
from app.services.conversation_context_service import ConversationContextService
from app.services.knowledge_service import KnowledgeService
from app.database.models import Call, Message


def run_all_tests():
    stt_service = STTService()
    knowledge_service = KnowledgeService()
    response_service = ResponseService()
    context_service = ConversationContextService()

    passed = 0
    failed = 0
    total = 9

    print("======================================================================")
    print("RUNNING LIVE VOICE TARGETED FIXES REGRESSION TEST SUITE")
    print("======================================================================")

    # Test 1: Duplicated question recording live_in_587_c05e66de.webm
    try:
        audio_path = "recordings/live_in_587_c05e66de.webm"
        if os.path.exists(audio_path):
            prompt = (
                "Pullman New Delhi Aerocity front desk inquiries about hotel check-in and check-out times, "
                "room reservations, dining, amenities. What is the check-in time? What is the check-out time?"
            )
            transcript, duration = stt_service.transcribe_with_info(audio_path, initial_prompt=prompt)
            assert transcript != "", "Transcript should not be empty"
            assert "What is the check-in time? What is the check-in time?" not in transcript, "Question duplicated"
            assert transcript.strip() == "What is the check-in time?", f"Unexpected transcript: {transcript}"
            print("[PASS] Test 1: Duplicated check-in question suppressed ->", repr(transcript))
            passed += 1
        else:
            print("[SKIP] Test 1: Recording file not found")
            total -= 1
    except Exception as e:
        print("[FAIL] Test 1:", e)
        failed += 1

    # Test 2: Clean check-in question
    try:
        company_id = "pullman_aerocity"
        system_instruction = knowledge_service.get_system_instruction(company_id)
        facts_in = knowledge_service.get_relevant_knowledge(company_id, "What is the check-in time?")
        resp_in = asyncio.run(response_service.generate_response(
            transcript="What is the check-in time?",
            conversation_context="",
            system_instruction=system_instruction,
            knowledge_context=facts_in,
        ))
        assert "3:00" in resp_in or "15:00" in resp_in, f"Missing 3:00 PM in response: {resp_in}"
        assert not ("thank you for calling" in resp_in.lower() and "how may i assist" in resp_in.lower()), f"Greeting repeated: {resp_in}"
        print("[PASS] Test 2: Clean check-in question response ->", repr(resp_in))
        passed += 1
    except Exception as e:
        print("[FAIL] Test 2:", e)
        failed += 1

    # Test 3: Clean check-out question
    try:
        company_id = "pullman_aerocity"
        system_instruction = knowledge_service.get_system_instruction(company_id)
        facts_out = knowledge_service.get_relevant_knowledge(company_id, "What is the check-out time?")
        resp_out = asyncio.run(response_service.generate_response(
            transcript="What is the check-out time?",
            conversation_context="",
            system_instruction=system_instruction,
            knowledge_context=facts_out,
        ))
        assert "12:00" in resp_out, f"Missing 12:00 PM in response: {resp_out}"
        assert not ("thank you for calling" in resp_out.lower() and "how may i assist" in resp_out.lower()), f"Greeting repeated: {resp_out}"
        print("[PASS] Test 3: Clean check-out question response ->", repr(resp_out))
        passed += 1
    except Exception as e:
        print("[FAIL] Test 3:", e)
        failed += 1

    # Test 4: Empty and whitespace-only customer messages in context
    try:
        call = Call(id=999)
        msg1 = Message(id=1, call_id=999, speaker="assistant", text="Hello! How can I help you?", timestamp=None)
        msg2 = Message(id=2, call_id=999, speaker="customer", text="", timestamp=None)
        msg3 = Message(id=3, call_id=999, speaker="customer", text="   ", timestamp=None)
        msg4 = Message(id=4, call_id=999, speaker="customer", text=None, timestamp=None)
        msg5 = Message(id=5, call_id=999, speaker="assistant", text="I didn't catch that.", timestamp=None)
        call.messages = [msg1, msg2, msg3, msg4, msg5]

        ctx = context_service.build_context([call])
        assert "Customer: " not in ctx, f"Empty customer line found in context: {ctx}"
        assert "Customer: None" not in ctx, f"None customer line found in context: {ctx}"
        assert "Assistant: Hello! How can I help you?" in ctx
        assert "Assistant: I didn't catch that." in ctx
        print("[PASS] Test 4: Empty and whitespace customer messages excluded from context")
        passed += 1
    except Exception as e:
        print("[FAIL] Test 4:", e)
        failed += 1

    # Test 5: Preservation of valid conversation history
    try:
        call = Call(id=1000)
        m1 = Message(id=1, call_id=1000, speaker="customer", text="Is breakfast included?", timestamp=None)
        m2 = Message(id=2, call_id=1000, speaker="assistant", text="Yes, Pluck restaurant serves breakfast.", timestamp=None)
        call.messages = [m1, m2]
        ctx = context_service.build_context([call])
        assert "Customer: Is breakfast included?" in ctx
        assert "Assistant: Yes, Pluck restaurant serves breakfast." in ctx
        print("[PASS] Test 5: Valid conversation history correctly formatted and preserved")
        passed += 1
    except Exception as e:
        print("[FAIL] Test 5:", e)
        failed += 1

    # Test 6: Repeated fallback messages followed by valid hotel question
    try:
        company_id = "pullman_aerocity"
        system_instruction = knowledge_service.get_system_instruction(company_id)
        facts_out = knowledge_service.get_relevant_knowledge(company_id, "What is the check-out time?")

        history_with_fallbacks = (
            "--- Call 1 ---\n"
            "Customer: I don't want to. I don't want to. You are affected.\n"
            "Assistant: I'm sorry, but I'm not able to provide the requested information due to the lack of specific details about the hotel. Could you please provide me with the check-in and check-out times or any additional details about the rooms?\n"
            "Customer: I think it's way of long, so then it's going to be out of check-out.\n"
            "Assistant: I'm sorry, but I don't have the requested information about the hotel. Could you please provide me with the check-in and check-out times or any additional details about the rooms?"
        )

        resp = asyncio.run(response_service.generate_response(
            transcript="What is the check-out time?",
            conversation_context=history_with_fallbacks,
            system_instruction=system_instruction,
            knowledge_context=facts_out,
        ))
        assert "12:00" in resp, f"Failed to provide 12:00 PM: {resp}"
        assert "not able to provide" not in resp.lower() and "don't have the requested" not in resp.lower(), f"Fallback repeated: {resp}"
        print("[PASS] Test 6: Fallback cascade broken by valid question ->", repr(resp))
        passed += 1
    except Exception as e:
        print("[FAIL] Test 6:", e)
        failed += 1

    # Test 7: Unsupported room-count question does not fabricate number
    try:
        company_id = "pullman_aerocity"
        system_instruction = knowledge_service.get_system_instruction(company_id)
        facts_room = knowledge_service.get_relevant_knowledge(company_id, "How many rooms are in the hotel?")
        assert facts_room == "", "Facts should be empty for unverified room count"

        resp = asyncio.run(response_service.generate_response(
            transcript="How many rooms are in the hotel?",
            conversation_context="",
            system_instruction=system_instruction,
            knowledge_context=facts_room,
        ))
        for invented_num in [" 200 ", " 300 ", " 400 ", " 500 ", " 600 ", " 700 ", " 800 ", " 1000 "]:
            assert invented_num not in resp, f"Invented number {invented_num} in response: {resp}"
        print("[PASS] Test 7: Unsupported question safely handled without fabrication ->", repr(resp))
        passed += 1
    except Exception as e:
        print("[FAIL] Test 7:", e)
        failed += 1

    # Test 8: Empty audio and noise-only handling
    try:
        t1, _ = stt_service.transcribe_with_info("recordings/live_in_587_dc7ae198.webm")
        t2, _ = stt_service.transcribe_with_info("recordings/live_in_587_90128af1.webm")
        assert t1 == "", f"Expected empty transcript for dc7ae198, got: {t1}"
        assert t2 == "", f"Expected empty transcript for 90128af1, got: {t2}"
        print("[PASS] Test 8: Empty audio and ambient noise returns empty string without hallucination")
        passed += 1
    except Exception as e:
        print("[FAIL] Test 8:", e)
        failed += 1

    # Test 9: Existing default-company behavior preserved
    try:
        resp = asyncio.run(response_service.generate_response(
            transcript="My WiFi internet connection is slow and not working.",
            conversation_context="",
            system_instruction=None,
            knowledge_context=None,
        ))
        assert len(resp) > 0, "Response should not be empty"
        resp_lower = resp.lower()
        assert any(term in resp_lower for term in ["internet", "wifi", "troubleshoot", "connection", "line", "help", "assist"]), f"Unexpected default response: {resp}"
        print("[PASS] Test 9: Default telecom company workflow preserved ->", repr(resp))
        passed += 1
    except Exception as e:
        print("[FAIL] Test 9:", e)
        failed += 1

    print("======================================================================")
    print(f"REGRESSION SUITE COMPLETED: {passed} PASSED, {failed} FAILED (TOTAL: {total})")
    print("======================================================================")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
