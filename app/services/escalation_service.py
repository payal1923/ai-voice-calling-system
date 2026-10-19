import re
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy.orm import Session, joinedload
from app.database.models import HumanAssistanceRequest, Call, Customer, Message

ESCALATION_KEYWORDS = [
    r"\b(human|agent|operator|representative|person|executive|officer)\b",
    r"\b(talk to (someone|a human|an agent|a person|a representative))\b",
    r"\b(speak with (someone|a human|an agent|a person|a representative))\b",
    r"\b(connect me|transfer me|customer care|helpdesk)\b",
    r"\b(frustrated|angry|terrible|horrible|useless|stupid ai|waste of time)\b",
    r"\b(you don't understand|you do not understand|not helpful|can't help)\b",
]


class EscalationService:
    @staticmethod
    def should_escalate(transcript: str) -> tuple[bool, Optional[str]]:
        """
        Analyzes customer statement to check if human escalation is required.
        Returns (is_escalated, reason).
        """
        text = transcript.lower().strip()
        for pattern in ESCALATION_KEYWORDS:
            if re.search(pattern, text):
                if any(w in text for w in ["frustrated", "angry", "terrible", "horrible", "useless", "waste of time", "don't understand"]):
                    return True, f"Customer expressed dissatisfaction or frustration: \"{transcript[:100]}\""
                return True, f"Customer requested a human representative: \"{transcript[:100]}\""
        return False, None

    @staticmethod
    def create_escalation(
        db: Session,
        call_id: int,
        reason: str,
        language_selected: str = "en",
        notes: Optional[str] = None,
        status: str = "waiting",
    ) -> HumanAssistanceRequest:
        """Creates and persists an escalation in human_assistance_requests table."""
        call = db.query(Call).options(joinedload(Call.customer)).filter(Call.id == call_id).first()
        customer = call.customer if call else None

        # Clean language name
        clean_lang = language_selected
        if language_selected in ["hi", "2"]:
            clean_lang = "Hindi"
        elif language_selected in ["mr", "3"]:
            clean_lang = "Marathi"
        elif language_selected in ["en", "1"]:
            clean_lang = "English"

        now = datetime.now(timezone.utc)
        req = HumanAssistanceRequest(
            call_id=call_id,
            customer_id=customer.id if customer else 0,
            customer_name=customer.name if customer else "Unknown Customer",
            customer_phone=customer.phone if customer else "N/A",
            language_selected=clean_lang,
            escalation_reason=reason,
            status=status.lower(),
            notes=notes,
            created_at=now,
            updated_at=now,
        )
        db.add(req)
        if call:
            call.status = "ESCALATED"
            call.ended_at = now
        db.commit()
        db.refresh(req)
        return req

    @staticmethod
    def get_escalations(db: Session, status_filter: Optional[str] = None) -> List[dict]:
        """Retrieves escalations with call, transcript, and callback details."""
        query = db.query(HumanAssistanceRequest).order_by(HumanAssistanceRequest.created_at.desc())
        if status_filter:
            s = status_filter.strip().lower()
            if s in ["pending", "waiting"]:
                query = query.filter(HumanAssistanceRequest.status.in_(["waiting", "PENDING"]))
            elif s in ["resolved"]:
                query = query.filter(HumanAssistanceRequest.status.in_(["resolved", "RESOLVED"]))
            elif s in ["contacted", "completed", "callback_completed"]:
                query = query.filter(HumanAssistanceRequest.status.in_(["contacted", "CONTACTED", "callback_completed"]))
            elif s in ["callback_initiated", "initiated"]:
                query = query.filter(HumanAssistanceRequest.status.in_(["callback_initiated"]))
            else:
                query = query.filter(HumanAssistanceRequest.status.ilike(f"%{s}%"))
        items = query.all()

        results = []
        for item in items:
            call = db.query(Call).options(joinedload(Call.messages)).filter(Call.id == item.call_id).first()
            transcript_msgs = []
            if call and call.messages:
                sorted_msgs = sorted(call.messages, key=lambda m: (m.timestamp or datetime.min, m.id or 0))
                transcript_msgs = [
                    {
                        "speaker": m.speaker,
                        "text": m.text,
                        "timestamp": m.timestamp.isoformat() if m.timestamp else None,
                    }
                    for m in sorted_msgs
                ]

            st = (item.status or "").lower()
            if st in ["waiting", "pending"]:
                cb_status_display = "Waiting for Callback"
            elif st == "callback_initiated":
                cb_status_display = "Callback Initiated"
            elif st in ["contacted", "callback_completed"]:
                cb_status_display = "Callback Completed"
            elif st == "resolved":
                cb_status_display = "Resolved"
            else:
                cb_status_display = item.status

            results.append({
                "id": item.id,
                "call_id": item.call_id,
                "customer_id": item.customer_id,
                "customer_name": item.customer_name or (call.customer.name if call and call.customer else "N/A"),
                "customer_phone": item.customer_phone or (call.customer.phone if call and call.customer else "N/A"),
                "language": item.language_selected,
                "language_selected": item.language_selected,
                "reason": item.escalation_reason,
                "escalation_reason": item.escalation_reason,
                "status": item.status,
                "callback_status": cb_status_display,
                "notes": item.notes,
                "created_at": item.created_at.isoformat() if item.created_at else None,
                "updated_at": item.updated_at.isoformat() if item.updated_at else None,
                "transcript": transcript_msgs,
            })
        return results

    @staticmethod
    def mark_contacted(db: Session, escalation_id: int, notes: str = "") -> dict:
        """Marks an escalation as contacted."""
        item = db.query(HumanAssistanceRequest).filter(HumanAssistanceRequest.id == escalation_id).first()
        if not item:
            raise ValueError(f"Escalation request {escalation_id} not found")

        item.status = "contacted"
        if notes:
            item.notes = f"{item.notes}\n[Contacted]: {notes}".strip() if item.notes else f"[Contacted]: {notes}"
        item.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(item)
        return {"status": "contacted", "id": item.id, "notes": item.notes}

    @staticmethod
    def resolve_escalation(db: Session, escalation_id: int, notes: str = "") -> dict:
        """Marks an escalation as resolved with resolution notes."""
        item = db.query(HumanAssistanceRequest).filter(HumanAssistanceRequest.id == escalation_id).first()
        if not item:
            raise ValueError(f"Escalation request {escalation_id} not found")

        item.status = "resolved"
        if notes:
            item.notes = f"{item.notes}\n[Resolved]: {notes}".strip() if item.notes else f"[Resolved]: {notes}"
        item.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(item)
        return {"status": "resolved", "id": item.id, "notes": item.notes}

    @staticmethod
    def initiate_callback(db: Session, escalation_id: int, notes: str = "") -> dict:
        """
        Initiates a callback for a specific customer.
        Reuses the existing telephony abstraction.
        Clearly distinguishes between callback registration vs actual carrier phone call connected.
        """
        import os
        item = db.query(HumanAssistanceRequest).filter(HumanAssistanceRequest.id == escalation_id).first()
        if not item:
            raise ValueError(f"Escalation request {escalation_id} not found")

        # Telephony check: distinguish actual telephony connection from action registration
        telephony_provider = os.environ.get("TELEPHONY_PROVIDER")
        telephony_configured = bool(telephony_provider and telephony_provider.lower() != "mock")

        now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        if telephony_configured:
            from app.services.telephony_adapter import get_telephony_adapter
            adapter = get_telephony_adapter(telephony_provider)
            provider_call_id = adapter.start_outbound_call(item.customer_id, item.customer_phone or "")
            log = f"[Callback Initiated - {now_str}]: Outbound carrier call initiated via telephony provider ({provider_call_id})."
            msg = f"Live outbound phone call placed to {item.customer_name} ({item.customer_phone})."
        else:
            provider_call_id = None
            log = f"[Callback Initiated - {now_str}]: Operator initiated callback to customer at {item.customer_phone}. Integration status: Ready (Carrier PBX integration pending)."
            msg = f"Callback action registered for {item.customer_name} ({item.customer_phone}). Telephony gateway pending configuration."

        if notes:
            log += f" Operator Notes: {notes}"

        item.notes = f"{item.notes}\n{log}".strip() if item.notes else log
        item.status = "callback_initiated"
        item.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(item)

        return {
            "status": "callback_initiated",
            "callback_status": "Callback Initiated",
            "id": item.id,
            "customer_name": item.customer_name,
            "customer_phone": item.customer_phone,
            "telephony_active": telephony_configured,
            "provider_call_id": provider_call_id,
            "message": msg,
            "notes": item.notes,
        }

    @staticmethod
    def complete_callback(db: Session, escalation_id: int, notes: str = "") -> dict:
        """Marks a callback as completed by the operator."""
        item = db.query(HumanAssistanceRequest).filter(HumanAssistanceRequest.id == escalation_id).first()
        if not item:
            raise ValueError(f"Escalation request {escalation_id} not found")

        now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        log = f"[Callback Completed - {now_str}]: Operator completed callback with customer."
        if notes:
            log += f" Resolution notes: {notes}"

        item.notes = f"{item.notes}\n{log}".strip() if item.notes else log
        item.status = "callback_completed"
        item.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(item)

        return {
            "status": "callback_completed",
            "callback_status": "Callback Completed",
            "id": item.id,
            "notes": item.notes,
        }

    @staticmethod
    def schedule_callback(db: Session, escalation_id: int, notes: str = "") -> dict:
        """Maintains backwards compatibility for schedule_callback."""
        return EscalationService.initiate_callback(db, escalation_id, notes)
