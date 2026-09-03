"""Refunds and disputes: money that was recovered and then taken back.

Before this, neither event existed in ``EventType``. ``normalize_webhook``
raised, the API answered 400, and Razorpay reads 400 as a permanent rejection
and drops the delivery. So a dispute raised against a payment RailPulse had
just claimed as recovered was silently discarded: the case stayed
RECOVERED_BY_LINK and the merchant's recovery figure kept counting money they
no longer had. The test suite even used ``payment.dispute.created`` as its
example of an unsupported event -- it documented the hole and called it correct.
"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from app.bank_health import BankHealthMonitor
from app.config import Settings
from app.gateway import FakeRazorpayGateway
from app.models import EventType, PaymentEvent, PaymentRail, RecoveryCaseState
from app.service import RecoveryService
from app.store import RecoveryStore

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
LATER = NOW + timedelta(days=3)


def failure(event_id: str, key: str, code: str = "CARD_EXPIRED") -> PaymentEvent:
    return PaymentEvent(
        event_id=event_id,
        event_type=EventType.PAYMENT_FAILED,
        logical_key=key,
        occurred_at=NOW,
        amount_paise=49900,
        payment_id=f"pay_{event_id}",
        invoice_id=key,
        issuer="hdfc",
        rail=PaymentRail.CARD,
        failure_code=code,
    )


def reversal(
    event_id: str, key: str, kind: EventType = EventType.PAYMENT_DISPUTE_CREATED
) -> PaymentEvent:
    return PaymentEvent(
        event_id=event_id,
        event_type=kind,
        logical_key=key,
        occurred_at=LATER,
        amount_paise=49900,
        payment_id=f"pay_{key}",
        reversal_id=f"disp_{event_id}",
        issuer="hdfc",
        rail=PaymentRail.CARD,
    )


class ReversalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = RecoveryStore()
        self.addCleanup(self.store.close)
        self.gateway = FakeRazorpayGateway()
        self.service = RecoveryService(
            self.store, BankHealthMonitor(min_samples=5), self.gateway, settings=Settings()
        )

    def _recover_by_link(self, key: str) -> str:
        self.service.ingest(failure(f"f_{key}", key))
        self.service.dispatch_due_actions(NOW)
        case = self.store.get_case(key)
        self.service.ingest(
            PaymentEvent(
                event_id=f"paid_{key}",
                event_type=EventType.PAYMENT_LINK_PAID,
                logical_key=key,
                occurred_at=NOW + timedelta(hours=1),
                amount_paise=49900,
                payment_id=f"pay_{key}",
                payment_link_id=case.payment_link_id,
                issuer="hdfc",
                rail=PaymentRail.CARD,
                captured=True,
            )
        )
        return case.id

    def test_a_dispute_reverses_a_recovered_case(self) -> None:
        """The headline case. RECOVERED_BY_LINK had no outgoing edge, so a
        state machine that could not represent this simply dropped the event."""
        self._recover_by_link("inv_disputed")
        self.assertEqual(
            self.store.get_case("inv_disputed").state, RecoveryCaseState.RECOVERED_BY_LINK
        )
        self.assertEqual(self.store.metrics()["recovered_amount_paise"], 49900)

        case, processed = self.service.ingest(reversal("d1", "inv_disputed"))
        self.assertTrue(processed)
        self.assertEqual(case.state, RecoveryCaseState.RECOVERY_REVERSED)
        self.assertEqual(case.reversal_reason, "payment.dispute.created")
        self.assertEqual(case.reversed_at, LATER)
        self.assertTrue(case.requires_manual_reconciliation)

    def test_reversed_money_stops_counting_as_recovered(self) -> None:
        """The reason this matters: the revenue figure was overstating."""
        self._recover_by_link("inv_a")
        self._recover_by_link("inv_b")
        self.assertEqual(self.store.metrics()["recovered_amount_paise"], 99800)

        self.service.ingest(reversal("d2", "inv_a"))
        metrics = self.store.metrics()
        self.assertEqual(metrics["recovered_cases"], 1)
        self.assertEqual(metrics["recovered_amount_paise"], 49900)
        # Reported, not silently vanished.
        self.assertEqual(metrics["reversed_cases"], 1)
        self.assertEqual(metrics["reversed_amount_paise"], 49900)

    def test_a_dispute_silences_outreach_permanently(self) -> None:
        """Chasing someone who has just disputed a charge is how a dispute
        turns into a regulatory complaint. The contact budget is the wrong
        instrument -- this is 'never', not 'sparingly'."""
        self._recover_by_link("inv_quiet")
        case, _ = self.service.ingest(reversal("d3", "inv_quiet"))
        allowed, reason = case.attention.can_contact(LATER, cooldown=timedelta(hours=24))
        self.assertFalse(allowed)
        self.assertEqual(reason, "customer_opted_out")

    def test_a_dispute_cancels_a_live_recovery_link(self) -> None:
        """A customer must not be able to pay a link for an invoice that has
        already been refunded."""
        self.service.ingest(failure("f_live", "inv_live"))
        self.service.dispatch_due_actions(NOW)
        case = self.store.get_case("inv_live")
        self.assertEqual(case.state, RecoveryCaseState.LINK_SENT)
        self.assertEqual(self.gateway.links[case.payment_link_id], "issued")

        self.service.ingest(reversal("d4", "inv_live", EventType.REFUND_CREATED))
        self.assertEqual(self.gateway.links[case.payment_link_id], "cancelled")
        self.assertEqual(
            self.store.get_case("inv_live").state, RecoveryCaseState.RECOVERY_REVERSED
        )

    def test_a_reversal_stops_an_in_flight_recovery(self) -> None:
        self.service.ingest(failure("f_flight", "inv_flight", code="BANK_DOWN"))
        self.assertIn(
            self.store.get_case("inv_flight").state,
            {RecoveryCaseState.COOLDOWN, RecoveryCaseState.RETRY_SCHEDULED},
        )
        case, _ = self.service.ingest(reversal("d5", "inv_flight"))
        self.assertEqual(case.state, RecoveryCaseState.RECOVERY_REVERSED)
        self.assertTrue(case.is_final)
        # Dispatch must not resurrect it.
        self.service.dispatch_due_actions(LATER + timedelta(days=1))
        self.assertEqual(
            self.store.get_case("inv_flight").state, RecoveryCaseState.RECOVERY_REVERSED
        )

    def test_a_redelivered_dispute_is_a_no_op(self) -> None:
        self._recover_by_link("inv_dup")
        first, processed_first = self.service.ingest(reversal("d6", "inv_dup"))
        _, processed_again = self.service.ingest(reversal("d6", "inv_dup"))
        self.assertTrue(processed_first)
        self.assertFalse(processed_again)
        self.assertEqual(first.state, RecoveryCaseState.RECOVERY_REVERSED)
        self.assertEqual(self.store.metrics()["reversed_cases"], 1)

    def test_a_second_distinct_reversal_does_not_double_count(self) -> None:
        """A dispute followed by a refund for the same payment is normal."""
        self._recover_by_link("inv_two")
        self.service.ingest(reversal("d7", "inv_two"))
        case, processed = self.service.ingest(
            reversal("d8", "inv_two", EventType.REFUND_PROCESSED)
        )
        self.assertTrue(processed, "a distinct event id is still a new delivery")
        self.assertEqual(case.state, RecoveryCaseState.RECOVERY_REVERSED)
        self.assertEqual(self.store.metrics()["reversed_cases"], 1)
        # The first reversal's reason stands; the second must not overwrite it.
        self.assertEqual(case.reversal_reason, "payment.dispute.created")

    def test_a_reversal_for_an_unknown_case_is_survivable(self) -> None:
        case, processed = self.service.ingest(reversal("d9", "inv_never_seen"))
        self.assertIsNone(case)
        self.assertTrue(processed, "the delivery was handled; there was just nothing to reverse")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
