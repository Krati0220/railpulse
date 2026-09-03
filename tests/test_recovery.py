from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from app.bank_health import BankHealthMonitor
from app.config import Settings
from app.gateway import FakeRazorpayGateway, PermanentGatewayError, TransientGatewayError
from app.models import EventType, PaymentEvent, PaymentRail, RecoveryCaseState
from app.service import RecoveryService
from app.store import RecoveryStore

NOW = datetime(2026, 8, 21, tzinfo=UTC)


def event(
    event_id: str,
    event_type: EventType,
    *,
    logical_key: str = "inv_001",
    failure_code: str | None = None,
    payment_link_id: str | None = None,
    captured: bool = False,
    occurred_at: datetime = NOW,
) -> PaymentEvent:
    return PaymentEvent(
        event_id=event_id,
        event_type=event_type,
        logical_key=logical_key,
        occurred_at=occurred_at,
        amount_paise=49900,
        payment_id="pay_original",
        invoice_id=logical_key,
        issuer="hdfc",
        rail=PaymentRail.CARD,
        failure_code=failure_code,
        payment_link_id=payment_link_id,
        captured=captured,
    )


class RecoveryServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = RecoveryStore()
        self.addCleanup(self.store.close)
        self.gateway = FakeRazorpayGateway()
        self.health = BankHealthMonitor(min_samples=3)
        self.settings = Settings()
        self.service = RecoveryService(self.store, self.health, self.gateway, settings=self.settings)

    def test_duplicate_event_has_no_second_action(self) -> None:
        failed = event("evt_1", EventType.PAYMENT_FAILED, failure_code="CARD_EXPIRED")
        case, processed = self.service.ingest(failed)
        self.assertTrue(processed)
        self.assertEqual(case.state, RecoveryCaseState.CONSENT_REQUIRED)
        self.service.dispatch_due_actions(NOW)
        _, processed_again = self.service.ingest(failed)
        self.assertFalse(processed_again)
        self.assertEqual(self.store.action_count(case.id, "payment_link.create"), 1)

    def test_late_authorization_cancels_eligible_link_then_waits_for_capture(self) -> None:
        case, _ = self.service.ingest(event("evt_2", EventType.PAYMENT_FAILED, failure_code="CARD_EXPIRED"))
        self.service.dispatch_due_actions(NOW)
        active = self.store.get_case_by_id(case.id)
        self.assertEqual(active.state, RecoveryCaseState.LINK_SENT)
        self.assertEqual(active.payment_link_status, "issued")

        pending, _ = self.service.ingest(event("evt_3", EventType.PAYMENT_AUTHORIZED))
        self.assertEqual(pending.state, RecoveryCaseState.AUTHORIZED_PENDING_CAPTURE)
        self.assertEqual(pending.payment_link_status, "cancelled")

        recovered, _ = self.service.ingest(event("evt_4", EventType.PAYMENT_CAPTURED))
        self.assertEqual(recovered.state, RecoveryCaseState.RECOVERED_NATURAL)

    def test_observed_outage_enters_cooldown(self) -> None:
        for _ in range(3):
            self.health.observe("sbi", PaymentRail.UPI_AUTOPAY, False, NOW)
        outage = PaymentEvent(
            event_id="evt_5",
            event_type=EventType.PAYMENT_FAILED,
            logical_key="inv_outage",
            occurred_at=NOW,
            amount_paise=20000,
            issuer="sbi",
            rail=PaymentRail.UPI_AUTOPAY,
            failure_code="BANK_DOWN",
        )
        case, _ = self.service.ingest(outage)
        self.assertEqual(case.state, RecoveryCaseState.COOLDOWN)
        self.assertEqual(case.next_action_at, NOW + timedelta(minutes=90))

    def test_opt_out_stops_recovery(self) -> None:
        case, _ = self.service.ingest(
            event("evt_6", EventType.PAYMENT_FAILED, logical_key="inv_optout", failure_code="CARD_EXPIRED")
        )
        case.attention.opted_out = True
        with self.store.transaction():
            self.store.save_case(case, case.version)
        self.service.dispatch_due_actions(NOW)
        stopped = self.store.get_case_by_id(case.id)
        self.assertEqual(stopped.state, RecoveryCaseState.STOPPED)
        self.assertEqual(stopped.stop_reason, "customer_opted_out")

    def test_final_case_cannot_reopen_after_out_of_order_failure(self) -> None:
        self.service.ingest(
            event("evt_7", EventType.PAYMENT_FAILED, logical_key="inv_final", failure_code="CARD_EXPIRED")
        )
        self.service.dispatch_due_actions(NOW)
        self.service.ingest(event("evt_8", EventType.PAYMENT_CAPTURED, logical_key="inv_final"))
        final, _ = self.service.ingest(
            event("evt_9", EventType.PAYMENT_FAILED, logical_key="inv_final", failure_code="GATEWAY_ERROR")
        )
        self.assertEqual(final.state, RecoveryCaseState.RECOVERED_NATURAL)

    def test_outreach_preview_is_revoked_when_late_authorisation_cancels_link(self) -> None:
        case, _ = self.service.ingest(
            event("evt_10", EventType.PAYMENT_FAILED, logical_key="inv_preview", failure_code="CARD_EXPIRED")
        )
        self.service.dispatch_due_actions(NOW)
        preview = self.service.create_outreach_preview(case.id, now=NOW)
        self.assertTrue(preview.approved)
        self.assertIsNotNone(self.store.get_case_by_id(case.id).outreach_preview)

        resolved, _ = self.service.ingest(
            event("evt_11", EventType.PAYMENT_AUTHORIZED, logical_key="inv_preview")
        )
        self.assertEqual(resolved.state, RecoveryCaseState.AUTHORIZED_PENDING_CAPTURE)
        self.assertIsNone(resolved.outreach_preview)
        actions = self.store.list_actions(case.id)
        self.assertTrue(all(action.status == "succeeded" for action in actions))


class RetryEscalationTests(unittest.TestCase):
    """A transient failure used to reach retry_scheduled and stop forever."""

    def setUp(self) -> None:
        self.store = RecoveryStore()
        self.addCleanup(self.store.close)
        self.gateway = FakeRazorpayGateway()
        self.health = BankHealthMonitor(min_samples=3)
        self.settings = Settings()
        self.service = RecoveryService(self.store, self.health, self.gateway, settings=self.settings)

    def test_transient_failure_schedules_a_deadline(self) -> None:
        case, _ = self.service.ingest(
            event("evt_t1", EventType.PAYMENT_FAILED, logical_key="inv_t", failure_code="GATEWAY_ERROR")
        )
        self.assertEqual(case.state, RecoveryCaseState.RETRY_SCHEDULED)
        self.assertEqual(case.next_action_at, NOW + self.settings.retry_escalation_after)

    def test_retry_window_elapsing_escalates_to_a_consented_link(self) -> None:
        case, _ = self.service.ingest(
            event("evt_t2", EventType.PAYMENT_FAILED, logical_key="inv_t2", failure_code="GATEWAY_ERROR")
        )
        # Nothing is due yet.
        self.assertEqual(self.service.dispatch_due_actions(NOW), [])
        self.assertEqual(self.store.get_case_by_id(case.id).state, RecoveryCaseState.RETRY_SCHEDULED)

        # One dispatch pass escalates the timer and then issues the consented
        # link, so a scheduler tick never leaves a case half-advanced.
        later = NOW + self.settings.retry_escalation_after + timedelta(minutes=1)
        self.service.dispatch_due_actions(later)
        settled = self.store.get_case_by_id(case.id)
        self.assertEqual(settled.state, RecoveryCaseState.LINK_SENT)
        self.assertEqual(self.store.action_count(case.id, "payment_link.create"), 1)

        # A second tick must not produce a second link.
        self.service.dispatch_due_actions(later + timedelta(minutes=1))
        self.assertEqual(self.store.action_count(case.id, "payment_link.create"), 1)

    def test_cooldown_extends_while_the_issuer_is_still_degraded(self) -> None:
        for _ in range(5):
            self.health.observe("sbi", PaymentRail.UPI_AUTOPAY, False, NOW)
        case, _ = self.service.ingest(
            PaymentEvent(
                event_id="evt_t3",
                event_type=EventType.PAYMENT_FAILED,
                logical_key="inv_t3",
                occurred_at=NOW,
                amount_paise=20000,
                issuer="sbi",
                rail=PaymentRail.UPI_AUTOPAY,
                failure_code="BANK_DOWN",
            )
        )
        self.assertEqual(case.state, RecoveryCaseState.COOLDOWN)

        due = NOW + self.settings.cooldown + timedelta(minutes=1)
        self.service.dispatch_due_actions(due)
        still_down = self.store.get_case_by_id(case.id)
        self.assertEqual(still_down.state, RecoveryCaseState.COOLDOWN)
        self.assertEqual(still_down.next_action_at, due + self.settings.cooldown)

        # The issuer recovers; the case is released back into a retry window.
        for _ in range(50):
            self.health.observe("sbi", PaymentRail.UPI_AUTOPAY, True, due)
        released_at = due + self.settings.cooldown + timedelta(minutes=1)
        self.service.dispatch_due_actions(released_at)
        self.assertEqual(
            self.store.get_case_by_id(case.id).state, RecoveryCaseState.RETRY_SCHEDULED
        )


class GatewayFailureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = RecoveryStore()
        self.addCleanup(self.store.close)
        self.health = BankHealthMonitor(min_samples=3)

    def _service(self, gateway) -> RecoveryService:
        return RecoveryService(self.store, self.health, gateway, settings=Settings())

    def test_transient_create_failure_is_recorded_as_unavailable(self) -> None:
        class BrokenGateway(FakeRazorpayGateway):
            def create_payment_link(self, **kwargs):
                raise TransientGatewayError("provider unreachable")

        service = self._service(BrokenGateway())
        case, _ = service.ingest(
            event("evt_g1", EventType.PAYMENT_FAILED, logical_key="inv_g1", failure_code="CARD_EXPIRED")
        )
        service.dispatch_due_actions(NOW)
        failed = self.store.get_case_by_id(case.id)
        self.assertEqual(failed.state, RecoveryCaseState.MANUAL_REVIEW)
        self.assertEqual(failed.stop_reason, "payment_link_create_unavailable")
        action = self.store.list_actions(case.id)[0]
        self.assertEqual(action.status, "failed")
        self.assertTrue(action.metadata["retryable"])

    def test_permanent_create_failure_is_distinguished(self) -> None:
        class RejectingGateway(FakeRazorpayGateway):
            def create_payment_link(self, **kwargs):
                raise PermanentGatewayError("amount not allowed")

        service = self._service(RejectingGateway())
        case, _ = service.ingest(
            event("evt_g2", EventType.PAYMENT_FAILED, logical_key="inv_g2", failure_code="CARD_EXPIRED")
        )
        service.dispatch_due_actions(NOW)
        failed = self.store.get_case_by_id(case.id)
        self.assertEqual(failed.stop_reason, "payment_link_create_rejected")
        self.assertFalse(self.store.list_actions(case.id)[0].metadata["retryable"])

    def test_cancel_failure_flags_reconciliation_without_losing_recovery(self) -> None:
        class UncancellableGateway(FakeRazorpayGateway):
            def cancel_payment_link(self, link_id):
                raise TransientGatewayError("cancel unavailable")

        service = self._service(UncancellableGateway())
        case, _ = service.ingest(
            event("evt_g3", EventType.PAYMENT_FAILED, logical_key="inv_g3", failure_code="CARD_EXPIRED")
        )
        service.dispatch_due_actions(NOW)
        resolved, _ = service.ingest(event("evt_g4", EventType.PAYMENT_CAPTURED, logical_key="inv_g3"))
        self.assertEqual(resolved.state, RecoveryCaseState.RECOVERED_NATURAL)
        self.assertTrue(resolved.requires_manual_reconciliation)
        cancel_action = next(
            action for action in self.store.list_actions(case.id) if action.action_type == "payment_link.cancel"
        )
        self.assertEqual(cancel_action.status, "failed")


class LinkEventOrderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = RecoveryStore()
        self.addCleanup(self.store.close)
        self.service = RecoveryService(
            self.store, BankHealthMonitor(min_samples=3), FakeRazorpayGateway(), settings=Settings()
        )

    def test_link_event_on_an_unactioned_case_does_not_crash_the_webhook(self) -> None:
        """This raised InvalidStateTransition and 500'd the webhook before."""
        self.service.ingest(
            event("evt_l1", EventType.PAYMENT_FAILED, logical_key="inv_l1", failure_code="GATEWAY_ERROR")
        )
        resolved, processed = self.service.ingest(
            event("evt_l2", EventType.PAYMENT_LINK_EXPIRED, logical_key="inv_l1")
        )
        self.assertTrue(processed)
        self.assertEqual(resolved.state, RecoveryCaseState.MANUAL_REVIEW)

    def test_expected_cancel_echo_does_not_disturb_a_held_payment(self) -> None:
        case, _ = self.service.ingest(
            event("evt_l3", EventType.PAYMENT_FAILED, logical_key="inv_l3", failure_code="CARD_EXPIRED")
        )
        self.service.dispatch_due_actions(NOW)
        link_id = self.store.get_case_by_id(case.id).payment_link_id
        self.service.ingest(event("evt_l4", EventType.PAYMENT_AUTHORIZED, logical_key="inv_l3"))
        echoed, _ = self.service.ingest(
            event(
                "evt_l5",
                EventType.PAYMENT_LINK_CANCELLED,
                logical_key="inv_l3",
                payment_link_id=link_id,
            )
        )
        self.assertEqual(echoed.state, RecoveryCaseState.AUTHORIZED_PENDING_CAPTURE)

    def test_link_paid_after_late_authorisation_is_reconciled(self) -> None:
        case, _ = self.service.ingest(
            event("evt_l6", EventType.PAYMENT_FAILED, logical_key="inv_l6", failure_code="CARD_EXPIRED")
        )
        self.service.dispatch_due_actions(NOW)
        link_id = self.store.get_case_by_id(case.id).payment_link_id
        self.service.ingest(event("evt_l7", EventType.PAYMENT_AUTHORIZED, logical_key="inv_l6"))
        collided, _ = self.service.ingest(
            event(
                "evt_l8",
                EventType.PAYMENT_LINK_PAID,
                logical_key="inv_l6",
                payment_link_id=link_id,
            )
        )
        self.assertEqual(collided.state, RecoveryCaseState.MANUAL_REVIEW)
        self.assertEqual(collided.stop_reason, "duplicate_collection_requires_reconciliation")


class AttentionBudgetTests(unittest.TestCase):
    def test_contact_budget_expires_after_the_rolling_window(self) -> None:
        from app.models import CustomerAttentionBudget

        budget = CustomerAttentionBudget()
        budget.record_contact(NOW)
        budget.record_contact(NOW + timedelta(hours=25))
        allowed, reason = budget.can_contact(NOW + timedelta(hours=50))
        self.assertFalse(allowed)
        self.assertEqual(reason, "contact_budget_exhausted")

        # Eight days after the last contact the window has rolled over.
        allowed, reason = budget.can_contact(NOW + timedelta(days=9))
        self.assertTrue(allowed)
        self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
