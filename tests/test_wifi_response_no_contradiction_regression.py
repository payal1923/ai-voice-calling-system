import os
import sys
import unittest
import asyncio

sys.path.insert(0, os.path.abspath("."))

from app.services.knowledge_service import KnowledgeService
from app.services.response_service import ResponseService


class TestWifiResponseNoContradictionRegression(unittest.TestCase):
    """
    Phase 2N Regression Suite:
    Guarantees that Wi-Fi inquiries (both hyphenated 'Wi-Fi' and unhyphenated 'wifi')
    receive factual, complimentary access answers with ZERO unlisted-facility disclaimers
    or apologies, while preserving genuine multi-intent compound functionality.
    """

    def setUp(self):
        self.ks = KnowledgeService()
        self.rs = ResponseService()
        self.company_id = "pullman_aerocity"

    def test_01_target_wifi_queries_have_no_contradiction_direct(self):
        """All 5 target Wi-Fi inquiries must confirm complimentary Wi-Fi without 'not confirmed' or apologies."""
        target_queries = [
            "Do you have Wi-Fi?",
            "Do you have wifi?",
            "Is there Wi-Fi?",
            "Do you offer Wi-Fi?",
            "Can I get Wi-Fi?",
        ]

        for q in target_queries:
            kb = self.ks.get_relevant_knowledge(self.company_id, q)
            self.assertIsNotNone(kb, f"Knowledge should be found for '{q}'")
            self.assertIn("Wi-Fi", kb, f"Knowledge should contain Wi-Fi policy for '{q}'")

            # 1. Compound aggregator must NOT misfire on single-intent query
            compound_resp = self.rs._get_compound_knowledge_response(q, kb)
            self.assertIsNone(
                compound_resp,
                f"Compound aggregator misfired on single-intent query: '{q}' -> {compound_resp}"
            )

            # 2. Direct response must contain complimentary policy and zero disclaimers
            resp = asyncio.run(self.rs.generate_response(q, knowledge_context=kb))
            resp_lower = resp.lower()

            self.assertTrue(
                "complimentary" in resp_lower or "high-speed" in resp_lower,
                f"Missing complimentary Wi-Fi policy in response for '{q}': {resp}"
            )
            self.assertTrue(
                "wi-fi" in resp_lower or "wifi" in resp_lower,
                f"Missing Wi-Fi mention in response for '{q}': {resp}"
            )
            self.assertNotIn(
                "not confirmed",
                resp_lower,
                f"Contradictory 'not confirmed' disclaimer found in response for '{q}': {resp}"
            )
            self.assertNotIn(
                "apologize",
                resp_lower,
                f"Unnecessary apology found in response for '{q}': {resp}"
            )

    def test_02_target_wifi_queries_have_no_contradiction_streaming(self):
        """Streaming generator must stream factual Wi-Fi chunks without any disclaimer chunks."""
        target_queries = [
            "Do you have Wi-Fi?",
            "Do you have wifi?",
            "Is there Wi-Fi?",
            "Do you offer Wi-Fi?",
            "Can I get Wi-Fi?",
        ]

        for q in target_queries:
            kb = self.ks.get_relevant_knowledge(self.company_id, q)

            async def _collect_stream():
                chunks = []
                async for sentence in self.rs.generate_response_stream_sentences(q, knowledge_context=kb):
                    chunks.append(sentence)
                return chunks

            chunks = asyncio.run(_collect_stream())
            full_stream_resp = " ".join(chunks).lower()

            self.assertTrue(
                "complimentary" in full_stream_resp or "high-speed" in full_stream_resp,
                f"Stream missing complimentary policy for '{q}': {full_stream_resp}"
            )
            self.assertNotIn(
                "not confirmed",
                full_stream_resp,
                f"Contradictory disclaimer found in stream for '{q}': {full_stream_resp}"
            )
            self.assertNotIn(
                "apologize",
                full_stream_resp,
                f"Apology found in stream for '{q}': {full_stream_resp}"
            )

    def test_03_genuine_compound_with_unlisted_facility_preserved(self):
        """Multi-intent question pairing a listed facility with an unlisted one answers the listed and declines unlisted."""
        q_fwd = "What time is breakfast and do you have a casino?"
        kb_fwd = self.ks.get_relevant_knowledge(self.company_id, q_fwd)
        resp_fwd = asyncio.run(self.rs.generate_response(q_fwd, knowledge_context=kb_fwd))

        self.assertIn("Pluck", resp_fwd)
        self.assertTrue(
            "do not offer" in resp_fwd.lower() or "not confirmed" in resp_fwd.lower(),
            f"Expected casino refusal in '{resp_fwd}'"
        )

        q_rev = "Do you have a casino and what time is breakfast?"
        kb_rev = self.ks.get_relevant_knowledge(self.company_id, q_rev)
        resp_rev = asyncio.run(self.rs.generate_response(q_rev, knowledge_context=kb_rev))

        self.assertIn("Pluck", resp_rev)
        self.assertTrue(
            "do not offer" in resp_rev.lower() or "not confirmed" in resp_rev.lower(),
            f"Expected casino refusal in '{resp_rev}'"
        )

    def test_04_genuine_compound_wifi_and_checkin_preserved(self):
        """Multi-intent question pairing Wi-Fi with check-in time answers both correctly."""
        q = "How do I connect to Wi-Fi and what time is check-in?"
        kb = self.ks.get_relevant_knowledge(self.company_id, q)
        resp = asyncio.run(self.rs.generate_response(q, knowledge_context=kb))

        self.assertIn("Wi-Fi", resp)
        self.assertIn("3:00 PM", resp)
        self.assertNotIn("not confirmed", resp.lower())
        self.assertNotIn("apologize", resp.lower())


if __name__ == "__main__":
    unittest.main()

