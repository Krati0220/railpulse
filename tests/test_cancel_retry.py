"""A payment link left live on an invoice that has already been collected.

The link is a bearer URL. Anyone holding it can pay it, and once the invoice
is settled nothing on the merchant's side is expecting a second payment. So
revoking it after a success is not housekeeping, it is the thing standing
between a customer and being charged twice.

RailPulse records that revocation under an idempotency key, which is right for
the reason the key exists: a redelivered webhook must not create a second
link. But the key was taken by an attempt that *failed*, so a cancel that
timed out could never be re-issued -- not by a later webhook, and not by
anything else, because for a collected invoice no later webhook is coming.
The case was flagged for a human and the link stayed payable until somebody
read the queue.

These tests hold the fix to three promises: a failed cancel is retried, a
succeeded one is never re-issued, and the retrying is bounded so a link the
provider will never cancel ends up in front of a person instead of spinning.
"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from app.bank_health import BankHealthMonitor
from app.config import Settings
from app.gateway import FakeRazorpayGateway
from app.models import EventType, PaymentEvent, PaymentRail
from app.service import RecoveryService
from app.store import RecoveryStore

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
CANCEL = "payment_link.cancel"


class BrokenCancelGateway(FakeRazorpayGateway):
    """Cancels fail until ``heal_after`` calls have been made."""

    def __init__(self, heal_after: int = 10**6) -> None:
        super().__init__()
        self.heal_after = heal_after
        self.cancel_calls = 0

    def cancel_payment_link(self, link_id: str) -> str:
        self.cancel_calls += 1
        if self.cancel_calls <= self.heal_after:
            raise TimeoutError("gateway timed out cancelling the link")
        return super().cancel_payment_link(link_id)


class CancelRetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = RecoveryStore()
        self.addCleanup(self.store.close)
        self.settings = Settings()

    def _service(self, gateway: FakeRazorpayGateway) -> RecoveryService:
        return RecoveryService(
            self.store, BankHealthMonitor(min_samples=5), gateway, settings=self.settings
        )

    def _open_case_with_link(self, service: RecoveryService) -> None:
        """Drive a real failure through to a live payment link."""
        service.ingest(
            PaymentEvent(
                event_id="evt_fail",
                event_type=EventType.PAYMENT_FAILED,
                logical_key="inv_1",
                occurred_at=NOW,
                amount_paise=49900,
                payment_id="pay_1",
                invoice_id="inv_1",
                customer_id="cust_1",
                issuer="hdfc",
                rail=PaymentRail.CARD,
                failure_code="CARD_EXPIRED",
            )
        )
        service.dispatch_due_actions(NOW)
        case = self.store.get_case("inv_1")
        self.assertIsNotNone(case.payment_link_id, "test setup never produced a link")

    def _collect(self, service: RecoveryService, at: datetime) -> None:
        service.ingest(
            PaymentEvent(
                event_id=f"evt_ok_{at.isoformat()}",
                event_type=EventType.PAYMENT_CAPTURED,
                logical_key="inv_1",
                occurred_at=at,
                amount_paise=49900,
                payment_id="pay_1",
                invoice_id="inv_1",
                customer_id="cust_1",
                issuer="hdfc",
                rail=PaymentRail.CARD,
            )
        )

    def test_a_failed_cancel_leaves_the_link_live_and_flags_a_human(self) -> None:
        """The starting condition, asserted so the fix has something to fix."""
        gateway = BrokenCancelGateway()
        service = self._service(gateway)
        self._open_case_with_link(service)
        self._collect(service, NOW + timedelta(hours=1))

        case = self.store.get_case("inv_1")
        self.assertTrue(case.requires_manual_reconciliation)
        self.assertEqual(case.stop_reason, "payment_link_cancel_failed")
        self.assertNotEqual(case.payment_link_status, "cancelled")

    def test_the_dispatch_tick_retries_it(self) -> None:
        """The headline. No further webhook arrives -- and none would, because
        the invoice is paid -- so the tick has to be what re-issues the call."""
        gateway = BrokenCancelGateway(heal_after=1)
        service = self._service(gateway)
        self._open_case_with_link(service)
        self._collect(service, NOW + timedelta(hours=1))
        self.assertEqual(gateway.cancel_calls, 1)

        service.dispatch_due_actions(NOW + timedelta(hours=2))

        self.assertEqual(gateway.cancel_calls, 2)
        case = self.store.get_case("inv_1")
        self.assertEqual(case.payment_link_status, "cancelled")

    def test_a_healed_case_stops_asking_for_a_human(self) -> None:
        """A reconciliation queue that fills with items which already resolved
        themselves is one nobody reads."""
        gateway = BrokenCancelGateway(heal_after=1)
        service = self._service(gateway)
        self._open_case_with_link(service)
        self._collect(service, NOW + timedelta(hours=1))
        service.dispatch_due_actions(NOW + timedelta(hours=2))

        case = self.store.get_case("inv_1")
        self.assertFalse(case.requires_manual_reconciliation)
        self.assertIsNone(case.stop_reason)

    def test_a_succeeded_cancel_is_never_re_issued(self) -> None:
        """The guarantee the key exists for, which the fix must not spend."""
        gateway = BrokenCancelGateway(heal_after=0)  # always succeeds
        service = self._service(gateway)
        self._open_case_with_link(service)
        self._collect(service, NOW + timedelta(hours=1))
        self.assertEqual(gateway.cancel_calls, 1)

        for hour in range(2, 8):
            service.dispatch_due_actions(NOW + timedelta(hours=hour))
        self.assertEqual(gateway.cancel_calls, 1)

    def test_a_redelivered_success_does_not_cancel_twice(self) -> None:
        gateway = BrokenCancelGateway(heal_after=0)
        service = self._service(gateway)
        self._open_case_with_link(service)
        for _ in range(3):
            service.ingest(
                PaymentEvent(
                    event_id="evt_ok_same",  # same event id: a redelivery
                    event_type=EventType.PAYMENT_CAPTURED,
                    logical_key="inv_1",
                    occurred_at=NOW + timedelta(hours=1),
                    amount_paise=49900,
                    payment_id="pay_1",
                    invoice_id="inv_1",
                    customer_id="cust_1",
                    issuer="hdfc",
                    rail=PaymentRail.CARD,
                )
            )
        self.assertEqual(gateway.cancel_calls, 1)

    def test_retrying_is_bounded(self) -> None:
        """A link the provider will never cancel must become a person's
        problem, not an infinite loop against someone else's API."""
        gateway = BrokenCancelGateway()  # never heals
        service = self._service(gateway)
        self._open_case_with_link(service)
        self._collect(service, NOW + timedelta(hours=1))

        for hour in range(2, 20):
            service.dispatch_due_actions(NOW + timedelta(hours=hour))

        self.assertEqual(gateway.cancel_calls, self.settings.max_link_cancel_attempts)
        case = self.store.get_case("inv_1")
        self.assertTrue(case.requires_manual_reconciliation)
        action = next(
            a for a in self.store.list_actions(case.id) if a.action_type == CANCEL
        )
        self.assertEqual(action.status, "failed")
        self.assertEqual(action.attempts, self.settings.max_link_cancel_attempts)

    def test_a_cancel_someone_else_completed_is_closed_not_re_issued(self) -> None:
        """If the link is already revoked, the sweep must retire the row rather
        than keep calling a gateway that has nothing left to do."""
        gateway = BrokenCancelGateway()
        service = self._service(gateway)
        self._open_case_with_link(service)
        self._collect(service, NOW + timedelta(hours=1))
        calls_after_failure = gateway.cancel_calls

        case = self.store.get_case("inv_1")
        with self.store.transaction():
            case.payment_link_status = "cancelled"
            self.store.save_case(case, case.version)

        service.dispatch_due_actions(NOW + timedelta(hours=2))
        self.assertEqual(gateway.cancel_calls, calls_after_failure)
        action = next(
            a for a in self.store.list_actions(case.id) if a.action_type == CANCEL
        )
        self.assertEqual(action.status, "succeeded")

    def test_the_sweep_is_bounded_per_tick(self) -> None:
        """One tick must not block behind an unbounded run of gateway calls."""
        self.settings = Settings(link_cancel_retry_batch=2)
        gateway = BrokenCancelGateway()
        service = self._service(gateway)
        for index in range(5):
            service.ingest(
                PaymentEvent(
                    event_id=f"evt_f{index}",
                    event_type=EventType.PAYMENT_FAILED,
                    logical_key=f"inv_{index}",
                    occurred_at=NOW,
                    amount_paise=49900,
                    payment_id=f"pay_{index}",
                    invoice_id=f"inv_{index}",
                    customer_id=f"cust_{index}",
                    issuer="hdfc",
                    rail=PaymentRail.CARD,
                    failure_code="CARD_EXPIRED",
                )
            )
            service.dispatch_due_actions(NOW)
            self._collect_index(service, index, NOW + timedelta(hours=1))

        before = gateway.cancel_calls
        service.dispatch_due_actions(NOW + timedelta(hours=2))
        self.assertEqual(gateway.cancel_calls - before, 2)

    def _collect_index(self, service: RecoveryService, index: int, at: datetime) -> None:
        service.ingest(
            PaymentEvent(
                event_id=f"evt_ok{index}",
                event_type=EventType.PAYMENT_CAPTURED,
                logical_key=f"inv_{index}",
                occurred_at=at,
                amount_paise=49900,
                payment_id=f"pay_{index}",
                invoice_id=f"inv_{index}",
                customer_id=f"cust_{index}",
                issuer="hdfc",
                rail=PaymentRail.CARD,
            )
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
