import os
import sys
import asyncio

sys.path.insert(0, os.path.abspath("."))

from app.services.stt_service import STTService
from app.services.knowledge_service import KnowledgeService
from app.services.response_service import ResponseService


def run_tests():
    stt_service = STTService()
    knowledge_service = KnowledgeService()
    response_service = ResponseService()

    company_id = "pullman_aerocity"
    system_instruction = knowledge_service.get_system_instruction(company_id)

    passed = 0
    failed = 0
    total = 5

    print("======================================================================")
    print("RUNNING WATER REQUEST & ERROR HANDLING REGRESSION TEST SUITE")
    print("======================================================================")

    # Test 1: "I need water in my room." (non-streaming and streaming)
    try:
        query_water = "I need water in my room."
        facts = knowledge_service.get_relevant_knowledge(company_id, query_water)

        # Non-streaming
        resp_non_stream = asyncio.run(response_service.generate_response(
            transcript=query_water,
            conversation_context="",
            system_instruction=system_instruction,
            knowledge_context=facts,
        ))
        assert "front desk" in resp_non_stream.lower() or "housekeeping" in resp_non_stream.lower(), f"Unexpected non-stream water response: {resp_non_stream}"
        assert "didn't catch" not in resp_non_stream.lower(), f"False unclear speech fallback returned: {resp_non_stream}"

        # Streaming
        async def _test_stream():
            sentences = []
            async for s in response_service.generate_response_stream_sentences(
                transcript=query_water,
                conversation_context="",
                system_instruction=system_instruction,
                knowledge_context=facts,
            ):
                sentences.append(s)
            return " ".join(sentences)

        resp_stream = asyncio.run(_test_stream())
        assert "front desk" in resp_stream.lower() or "housekeeping" in resp_stream.lower(), f"Unexpected stream water response: {resp_stream}"
        assert "didn't catch" not in resp_stream.lower(), f"False unclear speech fallback in stream: {resp_stream}"

        print(f"[PASS] Test 1: Water request properly handled without policy invention -> {repr(resp_stream)}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 1: {e}")
        failed += 1

    # Test 2: Existing Wi-Fi requests preserved
    try:
        query_wifi = "I need Wi-Fi in my room."
        facts_wifi = knowledge_service.get_relevant_knowledge(company_id, query_wifi)

        resp_wifi = asyncio.run(response_service.generate_response(
            transcript=query_wifi,
            conversation_context="",
            system_instruction=system_instruction,
            knowledge_context=facts_wifi,
        ))
        assert "complimentary" in resp_wifi.lower() and "wi-fi" in resp_wifi.lower(), f"Unexpected Wi-Fi response: {resp_wifi}"

        async def _test_wifi_stream():
            sentences = []
            async for s in response_service.generate_response_stream_sentences(
                transcript=query_wifi,
                conversation_context="",
                system_instruction=system_instruction,
                knowledge_context=facts_wifi,
            ):
                sentences.append(s)
            return " ".join(sentences)

        stream_wifi = asyncio.run(_test_wifi_stream())
        assert "complimentary" in stream_wifi.lower() and "wi-fi" in stream_wifi.lower(), f"Unexpected stream Wi-Fi response: {stream_wifi}"

        print(f"[PASS] Test 2: Existing Wi-Fi request handling preserved -> {repr(stream_wifi)}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 2: {e}")
        failed += 1

    # Test 3: Genuine unclear speech returns unclear speech fallback
    try:
        unclear_phrases = ["booyo helps off", "traffic block", "asdf qwerty"]
        for phrase in unclear_phrases:
            resp_unclear = asyncio.run(response_service.generate_response(
                transcript=phrase,
                conversation_context="",
                system_instruction=system_instruction,
                knowledge_context="",
            ))
            assert "catch that" in resp_unclear.lower() or "repeat" in resp_unclear.lower(), f"Unclear speech failed to trigger clarification: {resp_unclear} for '{phrase}'"

            async def _test_unclear_stream(p):
                sentences = []
                async for s in response_service.generate_response_stream_sentences(
                    transcript=p,
                    conversation_context="",
                    system_instruction=system_instruction,
                    knowledge_context="",
                ):
                    sentences.append(s)
                return " ".join(sentences)

            stream_unclear = asyncio.run(_test_unclear_stream(phrase))
            assert "catch that" in stream_unclear.lower() or "repeat" in stream_unclear.lower(), f"Unclear speech stream failed clarification: {stream_unclear} for '{phrase}'"

        print(f"[PASS] Test 3: Genuine unclear speech triggers clarification fallback -> {repr(resp_unclear)}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 3: {e}")
        failed += 1

    # Test 4: LLM connection failure with valid customer request
    try:
        from unittest.mock import patch
        import httpx

        valid_unmatched_query = "Do you have an on-call physician or medical doctor available tonight?"
        # Speech is clear and not garbled
        assert not response_service._is_unclear_or_garbled_transcript(valid_unmatched_query, "", "")

        # When LLM connection fails, it must return front desk assistance referral, NOT unclear speech fallback
        async def _mock_stream_connect_error(*args, **kwargs):
            raise httpx.ConnectError("All connection attempts failed")
            if False:
                yield ""

        with patch("app.services.llm_service.generate_response_stream", side_effect=_mock_stream_connect_error):
            async def _test_llm_failure_stream():
                sentences = []
                async for s in response_service.generate_response_stream_sentences(
                    transcript=valid_unmatched_query,
                    conversation_context="",
                    system_instruction=system_instruction,
                    knowledge_context="",
                ):
                    sentences.append(s)
                return " ".join(sentences)

            stream_failure_resp = asyncio.run(_test_llm_failure_stream())
            assert "front desk" in stream_failure_resp.lower(), f"LLM failure should refer to front desk, got: {stream_failure_resp}"
            assert "didn't catch that clearly" not in stream_failure_resp.lower(), f"LLM outage misclassified as unclear speech: {stream_failure_resp}"

        async def _mock_call_llm_error(*args, **kwargs):
            raise httpx.ConnectError("All connection attempts failed")

        with patch.object(response_service, "_call_llm", side_effect=_mock_call_llm_error):
            resp_non_stream_fail = asyncio.run(response_service.generate_response(
                transcript=valid_unmatched_query,
                conversation_context="",
                system_instruction=system_instruction,
                knowledge_context="",
            ))
            assert "front desk" in resp_non_stream_fail.lower(), f"Non-stream failure should refer to front desk, got: {resp_non_stream_fail}"
            assert "didn't catch that clearly" not in resp_non_stream_fail.lower(), f"Non-stream outage misclassified as unclear speech: {resp_non_stream_fail}"

        print(f"[PASS] Test 4: LLM connection failure handled separately from unclear speech -> {repr(stream_failure_resp)}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 4: {e}")
        failed += 1

    # Test 5: Replay with actual recorded audio from Call 1165 Turn 1
    try:
        wav_path = "recordings/live_in_1165_57e92122_16k.wav"
        if os.path.exists(wav_path):
            stt_prompt = knowledge_service.get_stt_prompt(company_id)
            transcript, dur = stt_service.transcribe_with_info(wav_path, initial_prompt=stt_prompt)
            assert "water" in transcript.lower(), f"Audio transcription missing water: {transcript}"

            async def _test_audio_stream(t):
                sentences = []
                async for s in response_service.generate_response_stream_sentences(
                    transcript=t,
                    conversation_context="",
                    system_instruction=system_instruction,
                    knowledge_context="",
                ):
                    sentences.append(s)
                return " ".join(sentences)

            end_to_end_resp = asyncio.run(_test_audio_stream(transcript))
            assert "front desk" in end_to_end_resp.lower() or "housekeeping" in end_to_end_resp.lower(), f"End-to-end replay failed: {end_to_end_resp}"
            assert "didn't catch" not in end_to_end_resp.lower(), f"End-to-end replay returned unclear fallback: {end_to_end_resp}"
            print(f"[PASS] Test 5: End-to-end recorded audio replay passed -> {repr(end_to_end_resp)}")
        else:
            print("[SKIP] Test 5: Recording file not found")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 5: {e}")
        failed += 1

    print("======================================================================")
    print(f"REGRESSION SUITE COMPLETED: {passed} PASSED, {failed} FAILED (TOTAL: {total})")
    print("======================================================================")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
