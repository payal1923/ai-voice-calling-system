import re
from datetime import datetime
from typing import List, Optional
from app.database.models import Call


class ConversationContextService:
    @staticmethod
    def _clean_history_message(speaker: str, text: str) -> Optional[str]:
        """
        Cleans and normalizes historical messages:
        - Drops empty, whitespace-only, or ephemeral repeat prompts.
        - Drops cut-off customer speech fragments ending in hyphens/dashes.
        - Neutralizes stale trailing questions from prior assistant turns so they do not
          contaminate subsequent turns with repetitive clarifications.
        """
        if not text:
            return None
        t = text.strip()
        if not t:
            return None

        if speaker.lower() == "customer":
            # Cut-off fragments ending abruptly in a dash (e.g. 'Do you have-')
            if t.endswith("-") or t.endswith("—"):
                return None
            return t

        if speaker.lower() == "assistant":
            # For assistant clarification prompts, strip polite apology prefixes so that
            # the substantive question (e.g. 'Were you inquiring about dining options...?'
            # or 'Are you asking whether Pluck has vegetarian options?') is preserved in history.
            # This enables tracking consecutive clarifications and understanding follow-up responses.
            apology_patterns = [
                r"^i'm\s+sorry,?\s+(?:i\s+)?didn't\s+catch\s+that\s+clearly[\.\!\,\s]*",
                r"^i'm\s+sorry,?\s+(?:i\s+)?didn't\s+quite\s+catch\s+that[\.\!\,\s]*",
                r"^i'm\s+sorry,?\s+(?:i\s+)?didn't\s+catch\s+that[\.\!\,\s]*",
                r"^i\s+apologize,?\s+but\s+i\s+am\s+(?:still\s+)?having\s+trouble\s+hearing\s+you\s+clearly[\.\!\,\s]*",
                r"^i\s+apologize,?\s+but\s+i\s+am\s+(?:still\s+)?having\s+trouble[\.\!\,\s]*",
                r"^i'm\s+sorry,?\s+could\s+you\s+please\s+clarify\s+",
                r"^i'm\s+sorry,?\s+",
            ]
            cleaned_assistant = t
            for pat in apology_patterns:
                cleaned_assistant = re.sub(pat, "", cleaned_assistant, flags=re.IGNORECASE).strip()

            ca_lower = cleaned_assistant.lower()

            # Preserve focused clarifications and repeat requests
            if any(p in ca_lower for p in [
                "were you inquiring about",
                "are you asking whether",
                "would you like to know more about",
                "could you please repeat your question",
                "could you please clarify what timing"
            ]):
                if cleaned_assistant:
                    return cleaned_assistant[0].upper() + cleaned_assistant[1:]
                return None

            # Ephemeral pure repeat fallback prompts without substantive question are dropped
            if any(p in ca_lower for p in ["please repeat your request", "repeat your request"]):
                return None

            # If an older assistant turn asked for dates/times (e.g. 'Could you please specify the dates...'),
            # or contained procedural closing commands (e.g. 'Please proceed with your check-out...'),
            # trim that trailing text when building historical context to prevent the LLM from
            # looping on that text on new topics.
            sentences = re.split(r"(?<=[.!?])\s+", cleaned_assistant)
            substantive = []
            for s in sentences:
                s_lower = s.lower()
                # If it's a clarification asking for dates/times:
                if any(p in s_lower for p in ["specify the dates", "preferred times", "what dates", "which dates"]):
                    continue
                # If it's a fallback phrase that contains a stale question:
                if "i don't have the information you're looking for" in s_lower:
                    continue
                # Procedural boilerplate commands that cause looping:
                if any(p in s_lower for p in ["please proceed with your", "proceed with your check-out", "proceed with your check-in"]):
                    continue
                if any(p in s_lower for p in ["could you please clarify", "could you please provide", "can you please provide"]):
                    continue
                substantive.append(s)
            if substantive:
                return " ".join(substantive).strip()
            # If the entire message was a clarification question, retain it
            if cleaned_assistant.endswith("?"):
                return cleaned_assistant
            first_clause = cleaned_assistant.split("?")[0].strip()
            if first_clause and not any(p in first_clause.lower() for p in ["specify the dates", "preferred times", "proceed with"]):
                return first_clause + "."
            return None

        return t

    def build_context(self, calls: List[Call], exclude_transcript: Optional[str] = None) -> str:
        """
        Formats a list of Call objects (and their associated messages) into a clean,
        chronological textual context string suitable for LLM prompting.
        Preserves genuine conversation context while preventing stale clarification questions,
        the current in-flight question, or audio artifacts from polluting subsequent turns.
        """
        if not calls:
            return ""

        context_parts = []
        exclude_norm = exclude_transcript.strip().lower() if exclude_transcript else None

        for index, call in enumerate(calls, start=1):
            if not call.messages:
                continue

            # Ensure messages within each call are sorted chronologically
            sorted_messages = sorted(
                call.messages,
                key=lambda m: (m.timestamp if m.timestamp is not None else datetime.min, m.id or 0),
            )

            # If exclude_transcript is provided, ignore the trailing message if it matches the current transcript
            messages_to_process = list(sorted_messages)
            if exclude_norm and messages_to_process:
                last_msg = messages_to_process[-1]
                if (last_msg.speaker or "").lower() == "customer":
                    last_text_norm = (last_msg.text or "").strip().lower()
                    if last_text_norm == exclude_norm or last_text_norm.strip("?!.") == exclude_norm.strip("?!."):
                        messages_to_process.pop()

            valid_lines = []
            for msg in messages_to_process:
                speaker = (msg.speaker or "").lower()
                cleaned_text = self._clean_history_message(speaker, msg.text)
                if cleaned_text:
                    label = "Customer" if speaker == "customer" else "Assistant"
                    valid_lines.append(f"{label}: {cleaned_text}")

            if not valid_lines:
                continue

            # Limit history context to the most recent 6 valid conversational exchanges
            recent_lines = valid_lines[-6:]
            call_lines = [f"--- Call {index} ---"] + recent_lines
            context_parts.append("\n".join(call_lines))

        return "\n\n".join(context_parts)
