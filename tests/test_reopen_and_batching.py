"""Two ways the engine used to be unable to make progress.

MANUAL_REVIEW was a one-way door. It is reached for a transient reason -- a
classifier outage makes every code unclassifiable -- as readily as for a
permanent one, so ten bad minutes at the provider stranded every case that
arrived during them, forever, with no operator action that could bring any of
them back.

And the dispatch tick read every due row without a bound, deserialising each
one into a full aggregate. Cost tracked the size of the backlog rather than
the work that was due, and the backlog is largest exactly when an issuer
outage has parked thousands of cases at once -- which is when the tick most
needs to finish.

The reopen tests spend most of their attention on what must NOT be reopenable.
An escape hatch from manual review is a convenience; an escape hatch that can
resume contacting a customer who opted out or raised a chargeback is a
regulatory incident, and the only thing keeping them apart is that the second
one does not exist.
"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from app.bank_health import BankHealthMonitor
from app.config import Settings
from app.gateway import FakeRazorpayGateway
from app.models import (
    EventType,
    FailureClass,
    PaymentEvent,
    PaymentRail,
    RecoveryCaseState,
)
from app.service import RecoveryService
from app.store import RecoveryStore

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)


def failure(index: int, code: str | None = "CARD_EXPIRED", customer: str | None = None):
    return PaymentEvent(
        event_id=f"evt_{index}",
        event_type=EventType.PAYMENT_FAILED,
        logical_key=f"inv_{index}",
        occurred_at=NOW,
        amount_paise=49900,
        payment_id=f"pay_{index}",
        invoice_id=f"inv_{index}",
        customer_id=customer or f"cust_{index}",
        issuer="hdfc",
        rail=PaymentRail.CARD,
        failure_code=code,
    )


class _Base(unittest.TestCase):
    settings = Settings()

    def setUp(self) -> None:
        self.store = RecoveryStore()
        self.addCleanup(self.store.close)
        self.service = RecoveryService(
            self.store,
            BankHealthMonitor(min_samples=5),
            FakeRazorpayGateway(),
            settings=self.settings,
        )


class ReopenTests(_Base):
    def _stranded(self) -> str:
        """An unclassifiable failure: the case the classifier could not read."""
        self.service.ingest(failure(0, code=None))
        case = self.store.get_case("inv_0")
        self.assertEqual(case.state, RecoveryCaseState.MANUAL_REVIEW)
        self.assertEqual(case.stop_reason, "unclassified_failure")
        return case.id

    def test_a_stranded_case_can_be_returned_to_the_engine(self) -> None:
        case_id = self._stranded()
        case = self.service.reopen(
            case_id,
            NOW + timedelta(hours=1),
            note="issuer confirmed the card was replaced",
            failure_class=FailureClass.CUSTOMER_ACTION,
        )
        self.assertEqual(case.state, RecoveryCaseState.CONSENT_REQUIRED)
        self.assertEqual(case.attempt_count, 0)

    def test_reopening_without_saying_anything_new_goes_straight_back(self) -> None:
        """Honest behaviour, not a bug: if the operator supplies no
        classification, nothing about the case has changed and the engine
        reaches the same conclusion. A reopen is not a way to make the engine
        guess at something it already declined to guess at."""
        case_id = self._stranded()
        case = self.service.reopen(case_id, NOW + timedelta(hours=1), note="had a look")
        self.assertEqual(case.state, RecoveryCaseState.MANUAL_REVIEW)
        self.assertEqual(case.stop_reason, "unclassified_failure")

    def test_an_exhausted_case_gets_a_fresh_budget(self) -> None:
        self.service.ingest(failure(1))
        case = self.store.get_case("inv_1")
        with self.store.transaction():
            case.attempt_count = self.settings.max_recovery_attempts
            case.state = RecoveryCaseState.MANUAL_REVIEW
            case.stop_reason = "retry_attempts_exhausted"
            self.store.save_case(case, case.version)

        reopened = self.service.reopen(
            case.id, NOW + timedelta(hours=1), note="customer says the bank cleared it"
        )
        self.assertEqual(reopened.attempt_count, 0)
        self.assertNotEqual(reopened.state, RecoveryCaseState.MANUAL_REVIEW)

    def test_the_reopen_is_in_the_audit_trail(self) -> None:
        case_id = self._stranded()
        self.service.reopen(
            case_id, NOW + timedelta(hours=1), note="classifier was down at 14:05"
        )
        actions = [
            a for a in self.store.list_actions(case_id) if a.action_type == "case.reopen"
        ]
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].metadata["note"], "classifier was down at 14:05")
        self.assertEqual(actions[0].metadata["previous_stop_reason"], "unclassified_failure")

    def test_an_opted_out_customer_can_never_be_reopened(self) -> None:
        """The one that matters. A dispute opts the customer out; if reopen
        could undo that, the engine would resume chasing someone who raised a
        chargeback -- and it would do it through an authenticated endpoint
        that looks like ordinary operations."""
        self.service.ingest(failure(2, customer="cust_shared"))
        self.service.ingest(
            PaymentEvent(
                event_id="evt_dispute",
                event_type=EventType.PAYMENT_DISPUTE_CREATED,
                logical_key="inv_2",
                occurred_at=NOW + timedelta(hours=1),
                amount_paise=49900,
                payment_id="pay_2",
                customer_id="cust_shared",
                issuer="hdfc",
                rail=PaymentRail.CARD,
            )
        )
        case = self.store.get_case("inv_2")
        self.assertTrue(case.attention.opted_out)
        with self.assertRaises(ValueError):
            self.service.reopen(case.id, NOW + timedelta(hours=2), note="please resume")
        self.assertTrue(self.store.get_case("inv_2").attention.opted_out)

    def test_a_risk_stop_can_never_be_reopened(self) -> None:
        self.service.ingest(failure(3, code="SUSPECTED_FRAUD"))
        case = self.store.get_case("inv_3")
        self.assertEqual(case.state, RecoveryCaseState.STOPPED)
        with self.assertRaises(ValueError):
            self.service.reopen(case.id, NOW + timedelta(hours=1), note="override")

    def test_a_recovered_case_cannot_be_reopened(self) -> None:
        self.service.ingest(failure(4))
        self.service.ingest(
            PaymentEvent(
                event_id="evt_ok",
                event_type=EventType.PAYMENT_CAPTURED,
                logical_key="inv_4",
                occurred_at=NOW + timedelta(hours=1),
                amount_paise=49900,
                payment_id="pay_4",
                customer_id="cust_4",
                issuer="hdfc",
                rail=PaymentRail.CARD,
            )
        )
        case = self.store.get_case("inv_4")
        with self.assertRaises(ValueError):
            self.service.reopen(case.id, NOW + timedelta(hours=2), note="again")

    def test_an_unknown_case_is_a_lookup_error(self) -> None:
        with self.assertRaises(LookupError):
            self.service.reopen("case_nope", NOW, note="ghost")


class DispatchBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = RecoveryStore()
        self.addCleanup(self.store.close)
        self.settings = Settings(dispatch_batch_size=2)
        self.service = RecoveryService(
            self.store,
            BankHealthMonitor(min_samples=5),
            FakeRazorpayGateway(),
            settings=self.settings,
        )

    def test_one_tick_takes_at_most_a_batch(self) -> None:
        for index in range(7):
            self.service.ingest(failure(index))
        self.assertEqual(len(self.service.dispatch_due_actions(NOW)), 2)

    def test_successive_ticks_drain_the_backlog(self) -> None:
        """Bounding the tick must not mean work is dropped, and no case may be
        starved by nothing more than where its uuid sorts."""
        for index in range(7):
            self.service.ingest(failure(index))
        for _ in range(6):
            self.service.dispatch_due_actions(NOW + timedelta(minutes=1))
        linked = sum(
            1 for index in range(7) if self.store.get_case(f"inv_{index}").payment_link_id
        )
        self.assertEqual(linked, 7, "a bounded tick left cases permanently unserved")

    def test_an_unbounded_store_read_is_still_available(self) -> None:
        for index in range(7):
            self.service.ingest(failure(index))
        due = self.store.due_cases([RecoveryCaseState.CONSENT_REQUIRED], NOW)
        self.assertEqual(len(due), 7)
        self.assertEqual(
            len(self.store.due_cases([RecoveryCaseState.CONSENT_REQUIRED], NOW, limit=3)), 3
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
