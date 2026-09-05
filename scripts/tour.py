"""Walk one failed payment through RailPulse and narrate every step."""
from datetime import UTC, datetime

from app.bank_health import BankHealthMonitor
from app.config import Settings
from app.gateway import FakeRazorpayGateway
from app.models import EventType, PaymentEvent, PaymentRail
from app.service import RecoveryService
from app.store import RecoveryStore

NOW = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
store = RecoveryStore()
svc = RecoveryService(store, BankHealthMonitor(min_samples=3), FakeRazorpayGateway(), settings=Settings())


def webhook(eid, etype, code=None, captured=False):
    return PaymentEvent(
        event_id=eid, event_type=etype, logical_key="inv_2026_03_001",
        occurred_at=NOW, amount_paise=129900, payment_id="pay_9fK2",
        invoice_id="inv_2026_03_001", issuer="hdfc", rail=PaymentRail.CARD,
        failure_code=code, customer_id="cust_arjun", captured=captured,
    )


def show(step, case):
    print(f"\n  {step}")
    print(f"    state          {case.state.value}")
    print(f"    failure_class  {case.failure_class.value}")
    print(f"    attempts       {case.attempt_count}   contacts {case.attention.contact_count_7d}")
    if case.payment_link_url:
        print(f"    link           {case.payment_link_url}  [{case.payment_link_status}]")


print("=" * 78)
print("ONE PAYMENT, END TO END   inv_2026_03_001 · Rs 1,299.00 · card/hdfc")
print("=" * 78)

case, fresh = svc.ingest(webhook("evt_1", EventType.PAYMENT_FAILED, code="CARD_EXPIRED"))
show("1. payment.failed arrives, issuer says CARD_EXPIRED", case)
print("       -> lookup maps CARD_EXPIRED to failure_class=customer_action.")
print("          No model call: a known code never reaches one.")
print("       -> policy: retrying a dead card fails again; ask for consent instead.")

svc.dispatch_due_actions(NOW)
case = store.get_case_by_id(case.id)
show("2. dispatch tick: contact budget checked, then a link is created", case)
print("       -> _may_contact passed (0 of 2 contacts used this week).")
print("          Intent was claimed in the ledger BEFORE the network call.")

case, again = svc.ingest(webhook("evt_1", EventType.PAYMENT_FAILED, code="CARD_EXPIRED"))
print("\n  3. the same webhook is redelivered")
print(f"    newly_processed        {again}")
print(f"    payment_link.create x  {store.action_count(case.id, 'payment_link.create')}")

case, _ = svc.ingest(webhook("evt_2", EventType.PAYMENT_AUTHORIZED))
show("4. the customer instead pays on the original mandate", case)
print("       -> the live link is a bearer URL on an invoice about to settle,")
print("          so it is revoked rather than left payable.")

case, _ = svc.ingest(webhook("evt_3", EventType.PAYMENT_CAPTURED, captured=True))
show("5. payment.captured", case)

print("\n" + "-" * 78)
print("  AUDIT TRAIL — everything above, as the ledger recorded it")
print("-" * 78)
for a in store.list_actions(case.id):
    print(f"    {a.action_type:24} {a.status:12} key={a.action_key}")
store.close()
