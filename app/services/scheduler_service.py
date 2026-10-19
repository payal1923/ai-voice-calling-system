import asyncio
from typing import List, Optional

from app.database.database import SessionLocal
from app.database.models import Customer
from app.services.conversation_service import ConversationService
from app.services.call_manager_service import CallManagerService


class AutomationScheduler:
    """
    Background scheduler that drives the automated outbound/inbound calling
    queue. It is inert by default and only becomes active when explicitly
    started by an operator through the automation API.
    """

    def __init__(self):
        self.conversation_service = ConversationService()
        self.call_manager = CallManagerService()

        self.is_running = False
        self.simulation_mode = True

        self.outbound_queue: List[int] = []
        self.inbound_queue: List[int] = []

        self.completed_history: List[dict] = []
        self.failed_history: List[dict] = []
        self.completed_calls = 0

        self.current_call_id: Optional[int] = None
        self.current_customer_name: Optional[str] = None
        self.current_call_type: Optional[str] = None

        self._task: Optional[asyncio.Task] = None
        self._active_cancel = False

    # ------------------------------------------------------------------
    # Lifecycle controls
    # ------------------------------------------------------------------
    def start(self, customer_ids: Optional[List[int]] = None) -> None:
        """
        Starts the automated calling queue for the given customer IDs. When no
        IDs are supplied, every customer in the database is queued.
        """
        if self.is_running:
            return

        self.is_running = True
        self._active_cancel = False
        self.completed_history = []
        self.failed_history = []
        self.completed_calls = 0

        ids = self._resolve_customer_ids(customer_ids)
        self.inbound_queue = []
        self.outbound_queue = list(ids)

        try:
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self._run_loop())
        except RuntimeError:
            # No running event loop (e.g. synchronous context) - scheduler stays
            # flagged as running so status reflects the operator's intent.
            self._task = None

    async def stop(self) -> None:
        """Safely stops the scheduler and clears any pending work."""
        self.is_running = False
        self.outbound_queue = []
        self.inbound_queue = []

        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        self.current_call_id = None
        self.current_customer_name = None
        self.current_call_type = None

    async def cancel_active_call(self) -> None:
        """Forcefully cancels the currently active simulated call."""
        self._active_cancel = True
        call_id = self.current_call_id
        if call_id is not None:
            try:
                with SessionLocal() as db:
                    call = self.conversation_service.get_call(db, call_id)
                    if call.ended_at is None:
                        self.conversation_service.end_call(db, call_id, status="FAILED")
            except Exception as e:
                print(f"Warning: failed to cancel call {call_id}: {e}")

        self.current_call_id = None
        self.current_customer_name = None
        self.current_call_type = None

    def trigger_incoming(self, customer_id: int) -> None:
        """Queues an incoming priority call so it is served before outbound work."""
        self.inbound_queue.append(int(customer_id))

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def get_status(self) -> dict:
        """Returns a real-time snapshot of the automation queue."""
        return {
            "is_running": self.is_running,
            "simulation_mode": self.simulation_mode,
            "automation_status": "RUNNING" if self.is_running else "IDLE",
            "outbound_queue": list(self.outbound_queue),
            "inbound_queue": list(self.inbound_queue),
            "total": len(self.outbound_queue) + len(self.inbound_queue),
            "completed_calls": self.completed_calls,
            "completed_history": list(self.completed_history),
            "failed_history": list(self.failed_history),
            "current_call_id": self.current_call_id,
            "current_customer_name": self.current_customer_name,
            "current_call_type": self.current_call_type,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _resolve_customer_ids(self, customer_ids: Optional[List[int]]) -> List[int]:
        """Resolves the queue targets, falling back to all customers."""
        if customer_ids:
            resolved = []
            for cid in customer_ids:
                try:
                    resolved.append(int(cid))
                except (TypeError, ValueError):
                    continue
            return resolved

        try:
            with SessionLocal() as db:
                return [c.id for c in db.query(Customer).order_by(Customer.id.asc()).all()]
        except Exception as e:
            print(f"Warning: failed to load customers for automation: {e}")
            return []

    async def _run_loop(self) -> None:
        """Continuously processes the queue until the scheduler is stopped."""
        try:
            while self.is_running:
                customer_id = None
                call_type = None

                if self.inbound_queue:
                    customer_id = self.inbound_queue.pop(0)
                    call_type = "INBOUND"
                elif self.outbound_queue:
                    customer_id = self.outbound_queue.pop(0)
                    call_type = "OUTBOUND"
                else:
                    # Idle - wait for new work or an explicit stop.
                    await asyncio.sleep(1.0)
                    continue

                if not self.is_running:
                    break

                await self._process_customer(customer_id, call_type)
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"Scheduler loop error: {e}")
        finally:
            self.is_running = False
            self.current_call_id = None
            self.current_customer_name = None
            self.current_call_type = None

    async def _process_customer(self, customer_id: int, call_type: str) -> None:
        """Creates a call for a customer and runs one simulated conversation."""
        call_id = None
        customer_name = None
        try:
            with SessionLocal() as db:
                customer = db.query(Customer).filter(Customer.id == customer_id).first()
                if not customer:
                    self.failed_history.append(
                        {"customer_id": customer_id, "reason": "customer_not_found"}
                    )
                    return

                customer_name = customer.name or customer.external_customer_id
                call = self.conversation_service.create_call(db, customer.id)
                call.call_type = call_type
                call.language = "en"
                db.commit()
                call_id = call.id

            self.current_call_id = call_id
            self.current_customer_name = customer_name
            self.current_call_type = call_type
            self._active_cancel = False

            await self.call_manager.run_call_simulation(call_id)

            status = None
            with SessionLocal() as db:
                call = self.conversation_service.get_call(db, call_id)
                status = call.status

            if self._active_cancel:
                self.failed_history.append(
                    {
                        "customer_id": customer_id,
                        "call_id": call_id,
                        "customer_name": customer_name,
                        "reason": "cancelled",
                    }
                )
            elif status in ("RESOLVED", "ESCALATED"):
                self.completed_calls += 1
                self.completed_history.append(
                    {
                        "customer_id": customer_id,
                        "call_id": call_id,
                        "customer_name": customer_name,
                        "status": status,
                    }
                )
            else:
                self.failed_history.append(
                    {
                        "customer_id": customer_id,
                        "call_id": call_id,
                        "customer_name": customer_name,
                        "reason": status or "unresolved",
                    }
                )
        except Exception as e:
            self.failed_history.append(
                {
                    "customer_id": customer_id,
                    "call_id": call_id,
                    "customer_name": customer_name,
                    "reason": str(e),
                }
            )
        finally:
            self.current_call_id = None
            self.current_customer_name = None
            self.current_call_type = None
