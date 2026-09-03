"""The contact budget belongs to a person, not an invoice.

`CustomerAttentionBudget` lives on `RecoveryCase`, and `RecoveryCase` had no
customer identifier at all. So "two contacts per 7 days" was enforced per
invoice: five failed subscriptions for one person produced five messages in the
same second, each case correctly believing it had spent one of its two. And a
dispute set `opted_out` on the disputed case only, so a customer who had just
raised a chargeback kept being chased about their other invoices -- the exact
scenario the reversal handler's own docstring calls a regulatory risk.

The README calls the opt-out column "the argument". These tests are what make
that argument true rather than an artefact of one-invoice-per-customer test
data.
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
CUSTOMER = "cust_shared"


def failure(index: int, customer: str | None = CUSTOMER, code: str = "CARD_EXPIRED"):
    return PaymentEvent(
        event_id=f"evt_{index}",
        event_type=EventType.PAYMENT_FAILED,
        logical_key=f"inv_{index}",
        occurred_at=NOW,
        amount_paise=49900,
        payment_id=f"pay_{index}",
        invoice_id=f"inv_{index}",
        customer_id=customer,
        issuer="hdfc",
        rail=PaymentRail.CARD,
        failure_code=code,
    )


class CustomerBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = RecoveryStore()
        self.addCleanup(self.store.close)
        self.settings = Settings()
        self.service = RecoveryService(
            self.store, BankHealthMonitor(min_samples=5), FakeRazorpayGateway(),
            settings=self.settings,
        )

    def _links_sent(self, count: int) -> int:
        return sum(
            1
            for i in range(count)
            if self.store.get_case(f"inv_{i}")
            and self.store.get_case(f"inv_{i}").payment_link_id
        )

    def test_one_customer_five_invoices_is_not_five_messages(self) -> None:
        """The headline regression. This produced 5/5 before the fix."""
        for index in range(5):
            self.service.ingest(failure(index))
        self.service.dispatch_due_actions(NOW)
        sent = self._links_sent(5)
        self.assertLessEqual(
            sent,
            self.settings.max_contacts_7d,
            f"contacted one customer {sent} times in a single tick",
        )
        self.assertGreater(sent, 0, "the budget must not block everything either")

    def test_the_cooldown_applies_across_invoices(self) -> None:
        for index in range(3):
            self.service.ingest(failure(index))
        self.service.dispatch_due_actions(NOW)
        before = self._links_sent(3)
        # Well inside the 24h cooldown.
        self.service.dispatch_due_actions(NOW + timedelta(hours=2))
        self.assertEqual(
            self._links_sent(3), before, "a second tick inside the cooldown contacted again"
        )

    def test_separate_customers_keep_separate_budgets(self) -> None:
        """The fix must not collapse everyone into one global budget."""
        for index in range(4):
            self.service.ingest(failure(index, customer=f"cust_{index}"))
        self.service.dispatch_due_actions(NOW)
        self.assertEqual(self._links_sent(4), 4, "distinct customers were wrongly throttled")

    def test_a_dispute_silences_every_invoice_for_that_customer(self) -> None:
        """A chargeback is about the human, not the invoice they disputed."""
        for index in range(3):
            self.service.ingest(failure(index))
        self.service.ingest(
            PaymentEvent(
                event_id="evt_dispute",
                event_type=EventType.PAYMENT_DISPUTE_CREATED,
                logical_key="inv_0",
                occurred_at=NOW + timedelta(hours=1),
                amount_paise=49900,
                payment_id="pay_0",
                customer_id=CUSTOMER,
                issuer="hdfc",
                rail=PaymentRail.CARD,
            )
        )
        for index in range(1, 3):
            case = self.store.get_case(f"inv_{index}")
            self.assertTrue(
                case.attention.opted_out,
                f"inv_{index} is still contactable after this customer disputed",
            )

    def test_a_disputed_customer_is_never_contacted_again(self) -> None:
        self.service.ingest(failure(0))
        self.service.ingest(
            PaymentEvent(
                event_id="evt_d",
                event_type=EventType.PAYMENT_DISPUTE_CREATED,
                logical_key="inv_0",
                occurred_at=NOW + timedelta(minutes=1),
                amount_paise=49900,
                customer_id=CUSTOMER,
                issuer="hdfc",
                rail=PaymentRail.CARD,
            )
        )
        # A new invoice for the same customer arrives afterwards.
        self.service.ingest(failure(1))
        self.service.dispatch_due_actions(NOW + timedelta(hours=2))
        case = self.store.get_case("inv_1")
        self.assertIsNone(case.payment_link_id, "contacted a customer who had disputed")
        self.assertEqual(case.state, RecoveryCaseState.STOPPED)

    def test_events_without_a_customer_still_work(self) -> None:
        """Not every payload carries a customer. Falling back to per-case
        behaviour is correct; failing open or blocking everything is not."""
        for index in range(3):
            self.service.ingest(failure(index, customer=None))
        self.service.dispatch_due_actions(NOW)
        self.assertEqual(self._links_sent(3), 3)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
