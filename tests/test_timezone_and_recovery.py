"""Two gaps a review turned up: timestamp normalisation, and the fact that the
product's own headline path had no end-to-end test."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta, timezone

from app.bank_health import BankHealthMonitor
from app.config import Settings
from app.gateway import FakeRazorpayGateway
from app.models import EventType, PaymentEvent, PaymentRail, RecoveryCaseState
from app.service import RecoveryService
from app.store import RecoveryStore

IST = timezone(timedelta(hours=5, minutes=30))
NOW = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)


def failure(event_id: str, key: str = "inv_tz", at: datetime = NOW, code: str = "BANK_DOWN"):
    return PaymentEvent(
        event_id=event_id,
        event_type=EventType.PAYMENT_FAILED,
        logical_key=key,
        occurred_at=at,
        amount_paise=49900,
        payment_id=f"pay_{event_id}",
        invoice_id=key,
        issuer="hdfc",
        rail=PaymentRail.CARD,
        failure_code=code,
    )


class TimezoneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = RecoveryStore()
        self.addCleanup(self.store.close)
        self.service = RecoveryService(
            self.store, BankHealthMonitor(min_samples=5), FakeRazorpayGateway(), settings=Settings()
        )

    def test_string_order_now_matches_chronological_order(self) -> None:
        """The bug in one line.

        Timestamps are compared as TEXT, so the comparison is lexicographic.
        A deadline stamped +05:30 sorted *after* the same instant stamped
        +00:00, so due_cases skipped it and the case was stranded forever.
        """
        from app.store import _stamp

        same_instant_ist = NOW.astimezone(IST)
        self.assertNotEqual(same_instant_ist.isoformat(), NOW.isoformat())
        self.assertFalse(same_instant_ist.isoformat() <= NOW.isoformat(), "the original bug")
        self.assertEqual(_stamp(same_instant_ist), _stamp(NOW), "normalisation collapses them")

    def test_a_deadline_stamped_in_ist_still_becomes_due(self) -> None:
        case, _ = self.service.ingest(failure("evt_tz_1"))
        # Re-stamp the deadline in IST: the same instant, a different offset.
        case.next_action_at = (NOW - timedelta(minutes=1)).astimezone(IST)
        case.state = RecoveryCaseState.COOLDOWN
        with self.store.transaction():
            self.store.save_case(case, case.version)

        due = [c.id for c in self.store.due_cases([RecoveryCaseState.COOLDOWN], NOW)]
        self.assertIn(case.id, due, "an IST-stamped deadline was invisible to dispatch")

    def test_a_naive_timestamp_does_not_break_comparison(self) -> None:
        """A naive datetime sorted early and then raised TypeError comparing
        naive-to-aware inside dispatch, 500-ing the whole tick."""
        case, _ = self.service.ingest(failure("evt_tz_2", key="inv_naive"))
        case.next_action_at = (NOW - timedelta(minutes=1)).replace(tzinfo=None)
        case.state = RecoveryCaseState.COOLDOWN
        with self.store.transaction():
            self.store.save_case(case, case.version)

        reloaded = self.store.get_case_by_id(case.id)
        self.assertIsNotNone(reloaded.next_action_at.tzinfo, "stored timestamps must be aware")
        self.service.dispatch_due_actions(NOW)  # must not raise


class RecoveredByLinkTests(unittest.TestCase):
    """The product's headline path. It had no end-to-end test at all --
    RECOVERED_BY_LINK appeared only as a hand-built fixture in store tests."""

    def setUp(self) -> None:
        self.store = RecoveryStore()
        self.addCleanup(self.store.close)
        self.gateway = FakeRazorpayGateway()
        self.service = RecoveryService(
            self.store, BankHealthMonitor(min_samples=5), self.gateway, settings=Settings()
        )

    def test_a_customer_paying_the_link_recovers_the_case(self) -> None:
        case, _ = self.service.ingest(failure("evt_link_1", key="inv_link", code="CARD_EXPIRED"))
        self.assertEqual(case.state, RecoveryCaseState.CONSENT_REQUIRED)

        self.service.dispatch_due_actions(NOW)
        case = self.store.get_case("inv_link")
        self.assertEqual(case.state, RecoveryCaseState.LINK_SENT)
        self.assertTrue(case.payment_link_id)
        self.assertEqual(self.gateway.created, 1)

        recovered, processed = self.service.ingest(
            PaymentEvent(
                event_id="evt_link_paid",
                event_type=EventType.PAYMENT_LINK_PAID,
                logical_key="inv_link",
                occurred_at=NOW + timedelta(hours=2),
                amount_paise=49900,
                payment_id="pay_link_1",
                payment_link_id=case.payment_link_id,
                issuer="hdfc",
                rail=PaymentRail.CARD,
                captured=True,
            )
        )
        self.assertTrue(processed)
        self.assertEqual(recovered.state, RecoveryCaseState.RECOVERED_BY_LINK)
        self.assertTrue(recovered.is_final)

        metrics = self.store.metrics()
        self.assertEqual(metrics["recovered_cases"], 1)
        self.assertEqual(metrics["recovered_amount_paise"], 49900)

    def test_the_link_amount_matches_the_failed_payment(self) -> None:
        """Guards the money bug where a case created from subscription.pending
        carried amount 0 and the engine asked for a zero-value link."""
        self.service.ingest(
            PaymentEvent(
                event_id="evt_amt_1",
                event_type=EventType.SUBSCRIPTION_PENDING,
                logical_key="inv_amt",
                occurred_at=NOW,
                amount_paise=0,
                issuer="hdfc",
                rail=PaymentRail.CARD,
                failure_code="CARD_EXPIRED",
            )
        )
        self.service.ingest(failure("evt_amt_2", key="inv_amt", code="CARD_EXPIRED"))
        self.assertEqual(self.store.get_case("inv_amt").amount_paise, 49900)

        self.service.dispatch_due_actions(NOW)
        case = self.store.get_case("inv_amt")
        self.assertEqual(case.state, RecoveryCaseState.LINK_SENT)
        self.assertEqual(case.amount_paise, 49900)

    def test_a_redelivered_link_paid_event_recovers_once(self) -> None:
        self.service.ingest(failure("evt_dup_1", key="inv_dup_link", code="CARD_EXPIRED"))
        self.service.dispatch_due_actions(NOW)
        link_id = self.store.get_case("inv_dup_link").payment_link_id

        paid = PaymentEvent(
            event_id="evt_dup_paid",
            event_type=EventType.PAYMENT_LINK_PAID,
            logical_key="inv_dup_link",
            occurred_at=NOW + timedelta(hours=1),
            amount_paise=49900,
            payment_id="pay_dup",
            payment_link_id=link_id,
            issuer="hdfc",
            rail=PaymentRail.CARD,
            captured=True,
        )
        first, processed_first = self.service.ingest(paid)
        _, processed_again = self.service.ingest(paid)
        self.assertTrue(processed_first)
        self.assertFalse(processed_again, "a redelivery must be a no-op")
        self.assertEqual(first.state, RecoveryCaseState.RECOVERED_BY_LINK)
        self.assertEqual(self.store.metrics()["recovered_cases"], 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
