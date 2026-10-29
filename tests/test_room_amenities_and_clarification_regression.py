import os
import sys
import asyncio

sys.path.insert(0, os.path.abspath("."))

from app.services.knowledge_service import KnowledgeService
from app.services.response_service import ResponseService


def run_tests():
    knowledge_service = KnowledgeService()
    response_service = ResponseService()

    company_id = "pullman_aerocity"
    system_instruction = knowledge_service.get_system_instruction(company_id)

    passed = 0
    failed = 0
    total = 9

    print("======================================================================")
    print("RUNNING ROOM AMENITIES & CLARIFICATION REGRESSION TEST SUITE")
    print("======================================================================")

    # Test 1: Fresh towels request
    try:
        query_towels = "I need fresh towels in my room."
        facts = knowledge_service.get_relevant_knowledge(company_id, query_towels)

        resp_non_stream = asyncio.run(response_service.generate_response(
            transcript=query_towels,
            conversation_context="",
            system_instruction=system_instruction,
            knowledge_context=facts,
        ))
        assert "front desk" in resp_non_stream.lower() and "housekeeping" in resp_non_stream.lower() and "towels" in resp_non_stream.lower(), f"Unexpected towels response: {resp_non_stream}"

        async def _test_stream_towels():
            sentences = []
            async for s in response_service.generate_response_stream_sentences(
                transcript=query_towels,
                conversation_context="",
                system_instruction=system_instruction,
                knowledge_context=facts,
            ):
                sentences.append(s)
            return " ".join(sentences)

        resp_stream = asyncio.run(_test_stream_towels())
        assert "front desk" in resp_stream.lower() and "housekeeping" in resp_stream.lower() and "towels" in resp_stream.lower(), f"Unexpected stream towels response: {resp_stream}"

        print(f"[PASS] Test 1: Fresh towels request handled correctly -> {repr(resp_stream)}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 1: {e}")
        failed += 1

    # Test 2: Extra pillow request
    try:
        query_pillow = "Can I get an extra pillow?"
        facts = knowledge_service.get_relevant_knowledge(company_id, query_pillow)

        resp_pillow = asyncio.run(response_service.generate_response(
            transcript=query_pillow,
            conversation_context="",
            system_instruction=system_instruction,
            knowledge_context=facts,
        ))
        assert "front desk" in resp_pillow.lower() and "housekeeping" in resp_pillow.lower() and "pillow" in resp_pillow.lower(), f"Unexpected pillow response: {resp_pillow}"

        print(f"[PASS] Test 2: Extra pillow request handled correctly -> {repr(resp_pillow)}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 2: {e}")
        failed += 1

    # Test 3: Blanket request
    try:
        query_blanket = "I need a blanket in my room."
        facts = knowledge_service.get_relevant_knowledge(company_id, query_blanket)

        resp_blanket = asyncio.run(response_service.generate_response(
            transcript=query_blanket,
            conversation_context="",
            system_instruction=system_instruction,
            knowledge_context=facts,
        ))
        assert "front desk" in resp_blanket.lower() and "housekeeping" in resp_blanket.lower() and "blanket" in resp_blanket.lower(), f"Unexpected blanket response: {resp_blanket}"

        print(f"[PASS] Test 3: Blanket request handled correctly -> {repr(resp_blanket)}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 3: {e}")
        failed += 1

    # Test 4: Housekeeping dispatch request
    try:
        query_hk = "Can you send housekeeping to my room?"
        facts = knowledge_service.get_relevant_knowledge(company_id, query_hk)

        resp_hk = asyncio.run(response_service.generate_response(
            transcript=query_hk,
            conversation_context="",
            system_instruction=system_instruction,
            knowledge_context=facts,
        ))
        assert "front desk" in resp_hk.lower() and "housekeeping" in resp_hk.lower(), f"Unexpected housekeeping response: {resp_hk}"

        print(f"[PASS] Test 4: Housekeeping request handled correctly -> {repr(resp_hk)}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 4: {e}")
        failed += 1

    # Test 5: Clarification requests ("Sorry, what did you say?", "Could you repeat that?", "I didn't hear you.")
    try:
        fitness_ctx = (
            "Customer: Tell me about the fitness center.\n"
            "Assistant: The Fit Lounge is a 24/7 fitness center fully equipped with modern cardiovascular and strength-training machines."
        )
        clarification_queries = [
            "Sorry, what did you say?",
            "Could you repeat that?",
            "I didn't hear you."
        ]
        for q in clarification_queries:
            resp_clarif = asyncio.run(response_service.generate_response(
                transcript=q,
                conversation_context=fitness_ctx,
                system_instruction=system_instruction,
                knowledge_context="",
            ))
            assert "which part would you like me to repeat" in resp_clarif.lower(), f"Clarification request failed for '{q}': {resp_clarif}"

            # Streaming verification
            async def _test_stream_clarif(query):
                sentences = []
                async for s in response_service.generate_response_stream_sentences(
                    transcript=query,
                    conversation_context=fitness_ctx,
                    system_instruction=system_instruction,
                    knowledge_context="",
                ):
                    sentences.append(s)
                return " ".join(sentences)

            stream_clarif = asyncio.run(_test_stream_clarif(q))
            assert "which part would you like me to repeat" in stream_clarif.lower(), f"Clarification stream failed for '{q}': {stream_clarif}"

        print("[PASS] Test 5: All clarification requests receive natural clarification prompt ('Of course. Which part would you like me to repeat?')")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 5: {e}")
        failed += 1

    # Test 6: Substantive questions with 'repeat' are answered with factual content, not hijacked
    try:
        query_rep_checkin = "Could you repeat the check-in time?"
        facts_checkin = knowledge_service.get_relevant_knowledge(company_id, query_rep_checkin)
        resp_checkin = asyncio.run(response_service.generate_response(
            transcript=query_rep_checkin,
            conversation_context="",
            system_instruction=system_instruction,
            knowledge_context=facts_checkin,
        ))
        assert "3:00" in resp_checkin, f"Substantive repetition check-in failed: {resp_checkin}"
        assert "which part" not in resp_checkin.lower(), f"Substantive question was falsely hijacked as generic clarification: {resp_checkin}"

        print(f"[PASS] Test 6: Substantive topic repetition is answered factually -> {repr(resp_checkin)}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 6: {e}")
        failed += 1

    # Test 7: Genuine unclear speech returns unclear speech fallback
    try:
        unclear_phrases = ["booyo helps off", "traffic block", "asdf qwerty"]
        for phrase in unclear_phrases:
            resp_unclear = asyncio.run(response_service.generate_response(
                transcript=phrase,
                conversation_context="",
                system_instruction=system_instruction,
                knowledge_context="",
            ))
            assert "catch that" in resp_unclear.lower() or "repeat" in resp_unclear.lower(), f"Unclear speech failed to trigger fallback: {resp_unclear} for '{phrase}'"
            assert "which part would you like me to repeat" not in resp_unclear.lower(), f"Unclear speech was falsely treated as clarification request: {resp_unclear}"

        print(f"[PASS] Test 7: Genuine unclear speech triggers clarification fallback -> {repr(resp_unclear)}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 7: {e}")
        failed += 1

    # Test 8: Existing Water request handling preserved
    try:
        query_water = "I need water in my room."
        facts_water = knowledge_service.get_relevant_knowledge(company_id, query_water)
        resp_water = asyncio.run(response_service.generate_response(
            transcript=query_water,
            conversation_context="",
            system_instruction=system_instruction,
            knowledge_context=facts_water,
        ))
        assert "front desk" in resp_water.lower() and "water" in resp_water.lower(), f"Water request regression: {resp_water}"

        print(f"[PASS] Test 8: Existing water request handling preserved -> {repr(resp_water)}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 8: {e}")
        failed += 1

    # Test 9: Existing Wi-Fi request handling preserved
    try:
        wifi_queries = [
            "Is Wi-Fi free for guests?",
            "How can I connect to the Wi-Fi?",
            "I need Wi-Fi in my room."
        ]
        for wq in wifi_queries:
            facts_wifi = knowledge_service.get_relevant_knowledge(company_id, wq)
            resp_wifi = asyncio.run(response_service.generate_response(
                transcript=wq,
                conversation_context="",
                system_instruction=system_instruction,
                knowledge_context=facts_wifi,
            ))
            assert "complimentary" in resp_wifi.lower() and "wi-fi" in resp_wifi.lower(), f"Wi-Fi regression for '{wq}': {resp_wifi}"

        print("[PASS] Test 9: Existing Wi-Fi request handling preserved across all queries")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 9: {e}")
        failed += 1

    print("======================================================================")
    print(f"REGRESSION SUITE COMPLETED: {passed} PASSED, {failed} FAILED (TOTAL: {total})")
    print("======================================================================")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
