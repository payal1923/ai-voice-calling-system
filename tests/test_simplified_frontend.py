import unittest
import os
import re

class TestSimplifiedFrontend(unittest.TestCase):
    def setUp(self):
        self.html_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend', 'index.html')
        self.assertTrue(os.path.exists(self.html_path), "frontend/index.html does not exist")
        with open(self.html_path, 'r', encoding='utf-8') as f:
            self.content = f.read()

    def test_view_live_call_section_exists(self):
        """Requirement 1: Single primary conversation workflow view exists."""
        self.assertIn('id="view-live-call"', self.content)
        self.assertIn('id="live-pre-call-card"', self.content)
        self.assertIn('id="live-call-active-panel"', self.content)

    def test_pre_call_has_only_start_conversation_button(self):
        """Requirement 2: Keep only a Start Conversation button before call begins."""
        # Check start button exists with Start Conversation text
        self.assertIn('id="btn-sip-call"', self.content)
        match = re.search(r'<button[^>]*id="btn-sip-call"[^>]*>([\s\S]*?)</button>', self.content)
        self.assertIsNotNone(match)
        btn_text = match.group(1).strip()
        self.assertIn('Start Conversation', btn_text)

        # Check customer dropdown is present for profile association
        self.assertIn('id="live-customer-dropdown"', self.content)

    def test_in_call_elements_present(self):
        """Requirement 3: Show listening/speaking/processing status, transcript, and End Conversation button."""
        # End Conversation button
        self.assertIn('id="btn-sip-hangup"', self.content)
        match_end = re.search(r'<button[^>]*id="btn-sip-hangup"[^>]*>([\s\S]*?)</button>', self.content)
        self.assertIsNotNone(match_end)
        self.assertTrue('End Conversation' in match_end.group(1) or 'End Call' in match_end.group(1))

        # Status indicators
        self.assertIn('id="live-mic-indicator"', self.content)
        self.assertIn('id="webrtc-call-status-pill"', self.content)
        self.assertIn('id="sip-reg-status"', self.content)

        # Conversation transcript box and count
        self.assertIn('id="live-transcript-box"', self.content)
        self.assertIn('id="live-transcript-count"', self.content)

    def test_staged_record_and_send_audio_workflow_removed(self):
        """Requirement 4: Remove the old Record Audio and Send Audio workflow from main Live Call screen."""
        self.assertNotIn('id="live-btn-start-record"', self.content)
        self.assertNotIn('id="live-btn-stop-record"', self.content)
        self.assertNotIn('id="live-btn-send-audio"', self.content)
        self.assertNotIn('id="live-recorded-preview"', self.content)
        self.assertNotIn('id="live-audio-ready-section"', self.content)
        self.assertNotIn('id="live-manual-audio-file"', self.content)

    def test_exotel_and_separate_browser_call_removed(self):
        """Requirement 5: Remove Exotel Call and separate Browser Call controls from primary screen."""
        self.assertNotIn('id="btn-start-exotel-call"', self.content)
        self.assertNotIn('id="btn-start-browser-call"', self.content)

    def test_language_selection_controls_removed(self):
        """Requirement 6: Remove redundant language-selection controls from primary workflow."""
        self.assertNotIn('id="live-language-choice"', self.content)
        self.assertNotIn('id="live-language-ivr-banner"', self.content)
        self.assertNotIn('id="btn-ivr-lang-1"', self.content)
        self.assertNotIn('id="btn-ivr-lang-2"', self.content)
        self.assertNotIn('id="btn-ivr-lang-3"', self.content)

    def test_push_to_talk_removed_from_primary_interface(self):
        """Requirement 7: Remove Push-to-Talk button from primary interface."""
        self.assertNotIn('id="btn-live-ptt"', self.content)
        self.assertNotIn('Push to Talk', self.content)

    def test_four_distinct_voice_states_present(self):
        """Requirement 8: Clearly display Listening, Processing, AI Speaking, and Microphone Error states."""
        self.assertIn('Listening', self.content)
        self.assertIn('Processing', self.content)
        self.assertIn('AI Speaking', self.content)
        self.assertIn('Microphone Error', self.content)

    def test_unrelated_pages_preserved(self):
        """Requirement 9: Do not redesign unrelated pages."""
        self.assertIn('id="view-customers"', self.content)
        self.assertIn('id="view-history"', self.content)
        self.assertIn('id="view-automation"', self.content)
        self.assertIn('id="view-escalations"', self.content)

    def test_audio_element_preserved(self):
        """Requirement 8: Real-time audio element preserved."""
        self.assertIn('id="webrtc-remote-audio"', self.content)

    def test_section_c_not_nested_in_history_and_panels_initial_state(self):
        """Regression test: Section C is not swallowed by Section B and pre-call card is visible."""
        # Find index of view-history and view-live-call
        history_idx = self.content.find('id="view-history"')
        live_call_idx = self.content.find('id="view-live-call"')
        self.assertGreater(history_idx, -1)
        self.assertGreater(live_call_idx, -1)
        self.assertGreater(live_call_idx, history_idx)

        # Ensure </section> occurs between history_idx and live_call_idx
        between = self.content[history_idx:live_call_idx]
        self.assertIn('</section>', between, "Section B must be closed before Section C begins")

        # Verify live-pre-call-card does not have hidden class by default
        match_pre = re.search(r'<div[^>]*id="live-pre-call-card"[^>]*>', self.content)
        self.assertIsNotNone(match_pre)
        self.assertNotIn('hidden', match_pre.group(0))

        # Verify live-call-active-panel has hidden class by default
        match_active = re.search(r'<div[^>]*id="live-call-active-panel"[^>]*>', self.content)
        self.assertIsNotNone(match_active)
        self.assertIn('hidden', match_active.group(0))

if __name__ == '__main__':
    unittest.main()
