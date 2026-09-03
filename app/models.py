"""Domain models for RailPulse.

The recovery case deliberately has its own state machine. It must not be
confused with the Razorpay subscription lifecycle (active/pending/halted).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any
from uuid import uuid4

DEFAULT_MAX_CONTACTS_7D = 2
DEFAULT_CONTACT_COOLDOWN = timedelta(hours=24)
CONTACT_WINDOW = timedelta(days=7)


def utc_now() -> datetime:
    return datetime.now(UTC)


class RecoveryCaseState(StrEnum):
    OPEN = "open"
    CLASSIFIED = "classified"
    COOLDOWN = "cooldown"
    RETRY_SCHEDULED = "retry_scheduled"
    CONSENT_REQUIRED = "consent_required"
    LINK_SENT = "link_sent"
    AUTHORIZED_PENDING_CAPTURE = "authorized_pending_capture"
    RECOVERED_NATURAL = "recovered_natural"
    RECOVERED_BY_LINK = "recovered_by_link"
    STOPPED = "stopped"
    MANUAL_REVIEW = "manual_review"
    #: Money that was recovered and then taken back -- a refund the merchant
    #: issued, or a dispute the customer raised. Distinct from STOPPED, which
    #: means recovery was abandoned; here it succeeded and was then reversed,
    #: and the recovery metrics must stop counting it.
    RECOVERY_REVERSED = "recovery_reversed"


class FailureClass(StrEnum):
    TRANSIENT = "transient"
    CUSTOMER_ACTION = "customer_action"
    RISK = "risk"
    UNKNOWN = "unknown"


class PaymentRail(StrEnum):
    CARD = "card"
    UPI_AUTOPAY = "upi_autopay"
    EMANDATE = "emandate"
    UNKNOWN = "unknown"


class EventType(StrEnum):
    PAYMENT_FAILED = "payment.failed"
    PAYMENT_AUTHORIZED = "payment.authorized"
    PAYMENT_CAPTURED = "payment.captured"
    ORDER_PAID = "order.paid"
    SUBSCRIPTION_PENDING = "subscription.pending"
    SUBSCRIPTION_HALTED = "subscription.halted"
    PAYMENT_LINK_PAID = "payment_link.paid"
    PAYMENT_LINK_PARTIALLY_PAID = "payment_link.partially_paid"
    PAYMENT_LINK_CANCELLED = "payment_link.cancelled"
    PAYMENT_LINK_EXPIRED = "payment_link.expired"
    PAYMENT_DISPUTE_CREATED = "payment.dispute.created"
    REFUND_CREATED = "refund.created"
    REFUND_PROCESSED = "refund.processed"


FAILURE_EVENTS = {
    EventType.PAYMENT_FAILED,
    EventType.SUBSCRIPTION_PENDING,
    EventType.SUBSCRIPTION_HALTED,
}
SUCCESS_EVENTS = {
    EventType.PAYMENT_AUTHORIZED,
    EventType.PAYMENT_CAPTURED,
    EventType.ORDER_PAID,
}
LINK_EVENTS = {
    EventType.PAYMENT_LINK_PAID,
    EventType.PAYMENT_LINK_PARTIALLY_PAID,
    EventType.PAYMENT_LINK_CANCELLED,
    EventType.PAYMENT_LINK_EXPIRED,
}

#: A dispute or refund reverses money that was already collected. Neither is a
#: reason to try again -- both are hard stops that must also silence outreach,
#: because contacting someone who has just disputed a charge is how a dispute
#: becomes a regulatory complaint.
REVERSAL_EVENTS = {
    EventType.PAYMENT_DISPUTE_CREATED,
    EventType.REFUND_CREATED,
    EventType.REFUND_PROCESSED,
}

FINAL_STATES = {
    RecoveryCaseState.RECOVERED_NATURAL,
    RecoveryCaseState.RECOVERED_BY_LINK,
    RecoveryCaseState.STOPPED,
    RecoveryCaseState.RECOVERY_REVERSED,
}

#: States a live payment link can legitimately be cancelled from.
CANCELLABLE_LINK_STATUSES = {"issued", "created"}


ALLOWED_TRANSITIONS: dict[RecoveryCaseState, set[RecoveryCaseState]] = {
    RecoveryCaseState.OPEN: {
        RecoveryCaseState.RECOVERY_REVERSED,
        RecoveryCaseState.CLASSIFIED,
        RecoveryCaseState.COOLDOWN,
        RecoveryCaseState.RETRY_SCHEDULED,
        RecoveryCaseState.CONSENT_REQUIRED,
        RecoveryCaseState.AUTHORIZED_PENDING_CAPTURE,
        RecoveryCaseState.RECOVERED_NATURAL,
        RecoveryCaseState.STOPPED,
        RecoveryCaseState.MANUAL_REVIEW,
    },
    RecoveryCaseState.CLASSIFIED: {
        RecoveryCaseState.RECOVERY_REVERSED,
        RecoveryCaseState.COOLDOWN,
        RecoveryCaseState.RETRY_SCHEDULED,
        RecoveryCaseState.CONSENT_REQUIRED,
        RecoveryCaseState.STOPPED,
        # An out-of-order link or authorisation event can reach a case that is
        # still only classified. Without these edges the webhook raised and the
        # provider retried the same payload forever.
        RecoveryCaseState.AUTHORIZED_PENDING_CAPTURE,
        RecoveryCaseState.RECOVERED_NATURAL,
        RecoveryCaseState.MANUAL_REVIEW,
    },
    RecoveryCaseState.COOLDOWN: {
        RecoveryCaseState.RECOVERY_REVERSED,
        RecoveryCaseState.RETRY_SCHEDULED,
        RecoveryCaseState.CONSENT_REQUIRED,
        RecoveryCaseState.AUTHORIZED_PENDING_CAPTURE,
        RecoveryCaseState.RECOVERED_NATURAL,
        RecoveryCaseState.STOPPED,
        RecoveryCaseState.MANUAL_REVIEW,
    },
    RecoveryCaseState.RETRY_SCHEDULED: {
        RecoveryCaseState.RECOVERY_REVERSED,
        RecoveryCaseState.COOLDOWN,
        RecoveryCaseState.CONSENT_REQUIRED,
        RecoveryCaseState.AUTHORIZED_PENDING_CAPTURE,
        RecoveryCaseState.RECOVERED_NATURAL,
        RecoveryCaseState.STOPPED,
        RecoveryCaseState.MANUAL_REVIEW,
    },
    RecoveryCaseState.CONSENT_REQUIRED: {
        RecoveryCaseState.RECOVERY_REVERSED,
        RecoveryCaseState.LINK_SENT,
        RecoveryCaseState.AUTHORIZED_PENDING_CAPTURE,
        RecoveryCaseState.RECOVERED_NATURAL,
        RecoveryCaseState.STOPPED,
        RecoveryCaseState.MANUAL_REVIEW,
    },
    RecoveryCaseState.LINK_SENT: {
        RecoveryCaseState.RECOVERY_REVERSED,
        RecoveryCaseState.AUTHORIZED_PENDING_CAPTURE,
        RecoveryCaseState.RECOVERED_NATURAL,
        RecoveryCaseState.RECOVERED_BY_LINK,
        RecoveryCaseState.MANUAL_REVIEW,
        RecoveryCaseState.STOPPED,
    },
    RecoveryCaseState.AUTHORIZED_PENDING_CAPTURE: {
        RecoveryCaseState.RECOVERY_REVERSED,
        RecoveryCaseState.RECOVERED_NATURAL,
        RecoveryCaseState.MANUAL_REVIEW,
        RecoveryCaseState.STOPPED,
    },
    RecoveryCaseState.MANUAL_REVIEW: {
        RecoveryCaseState.RECOVERY_REVERSED,
        RecoveryCaseState.AUTHORIZED_PENDING_CAPTURE,
        RecoveryCaseState.RECOVERED_NATURAL,
        RecoveryCaseState.RECOVERED_BY_LINK,
        RecoveryCaseState.STOPPED,
        # Reachable only through RecoveryService.reopen -- a person deciding
        # the case should go back to the engine. Manual review is entered for
        # transient reasons (a classifier outage makes every code
        # unclassifiable) as readily as for permanent ones, and with no edges
        # out of it, a bad ten minutes at the provider stranded every case
        # that arrived during them for good.
        #
        # Note what is deliberately still absent: STOPPED has no edge back to
        # anything. Opting out, a risk decision and a dispute all land there,
        # and nothing an operator can do in this system undoes them.
        RecoveryCaseState.COOLDOWN,
        RecoveryCaseState.RETRY_SCHEDULED,
        RecoveryCaseState.CONSENT_REQUIRED,
    },
    # Recovery is not permanent. A refund or a dispute can arrive days after a
    # case was closed as recovered, and a state machine with no edge out of
    # 'recovered' cannot represent that -- so the event was dropped and the
    # metrics kept counting money the merchant no longer had.
    RecoveryCaseState.RECOVERED_NATURAL: {RecoveryCaseState.RECOVERY_REVERSED},
    RecoveryCaseState.RECOVERED_BY_LINK: {RecoveryCaseState.RECOVERY_REVERSED},
    RecoveryCaseState.STOPPED: set(),
    RecoveryCaseState.RECOVERY_REVERSED: set(),
}


class InvalidStateTransition(ValueError):
    pass


@dataclass(frozen=True)
class PaymentEvent:
    """Normalized event received from a Razorpay webhook or test fixture."""

    event_id: str
    event_type: EventType
    logical_key: str
    occurred_at: datetime
    amount_paise: int = 0
    payment_id: str | None = None
    order_id: str | None = None
    subscription_id: str | None = None
    invoice_id: str | None = None
    payment_link_id: str | None = None
    #: Dispute or refund id, when the event is a reversal.
    reversal_id: str | None = None
    #: The person behind the payment. Without it every budget in this system is
    #: per-invoice, which is not what "two contacts per 7 days" means to the
    #: human receiving them.
    customer_id: str | None = None
    issuer: str = "unknown"
    rail: PaymentRail = PaymentRail.UNKNOWN
    failure_code: str | None = None
    captured: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def category(self) -> str:
        if self.event_type in FAILURE_EVENTS:
            return "failure"
        if self.event_type in SUCCESS_EVENTS:
            return "success"
        if self.event_type in LINK_EVENTS:
            return "link"
        return "unknown"


@dataclass
class CustomerAttentionBudget:
    contact_count_7d: int = 0
    max_contacts_7d: int = DEFAULT_MAX_CONTACTS_7D
    opted_out: bool = False
    last_contact_at: datetime | None = None

    def can_contact(
        self,
        now: datetime,
        *,
        cooldown: timedelta = DEFAULT_CONTACT_COOLDOWN,
    ) -> tuple[bool, str | None]:
        if self.opted_out:
            return False, "customer_opted_out"
        if self.effective_contacts(now) >= self.max_contacts_7d:
            return False, "contact_budget_exhausted"
        if self.last_contact_at and now - self.last_contact_at < cooldown:
            return False, "contact_cooldown_active"
        return True, None

    def effective_contacts(self, now: datetime) -> int:
        """Contacts that still count against the rolling 7-day budget.

        The counter previously only ever increased, so a customer contacted
        twice in January stayed permanently unreachable. The window now expires.
        """
        if self.last_contact_at and now - self.last_contact_at >= CONTACT_WINDOW:
            return 0
        return self.contact_count_7d

    def record_contact(self, now: datetime) -> None:
        self.contact_count_7d = self.effective_contacts(now) + 1
        self.last_contact_at = now


@dataclass
class RecoveryCase:
    id: str
    logical_key: str
    amount_paise: int
    state: RecoveryCaseState = RecoveryCaseState.OPEN
    failure_class: FailureClass | None = None
    failure_code: str | None = None
    issuer: str = "unknown"
    rail: PaymentRail = PaymentRail.UNKNOWN
    subscription_id: str | None = None
    invoice_id: str | None = None
    customer_id: str | None = None
    original_payment_id: str | None = None
    payment_link_id: str | None = None
    payment_link_url: str | None = None
    payment_link_status: str | None = None
    # A preview is deliberately non-delivery state. It lets a merchant inspect
    # a guarded draft, but does not imply that an SMS or WhatsApp message sent.
    outreach_preview: dict[str, Any] | None = None
    next_action_at: datetime | None = None
    stop_reason: str | None = None
    requires_manual_reconciliation: bool = False
    #: Set when a refund or dispute took back money this case had recorded as
    #: recovered. Kept separate from stop_reason so the two are never confused:
    #: stopping means we chose not to pursue, reversing means we collected and
    #: then lost it, and only the second one has to be subtracted from revenue.
    reversal_reason: str | None = None
    reversed_at: datetime | None = None
    attempt_count: int = 0
    version: int = 0
    attention: CustomerAttentionBudget = field(default_factory=CustomerAttentionBudget)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    @classmethod
    def from_failed_event(cls, event: PaymentEvent) -> RecoveryCase:
        return cls(
            id=f"case_{uuid4().hex[:16]}",
            logical_key=event.logical_key,
            amount_paise=event.amount_paise,
            issuer=event.issuer.lower(),
            rail=event.rail,
            subscription_id=event.subscription_id,
            invoice_id=event.invoice_id,
            customer_id=event.customer_id,
            original_payment_id=event.payment_id,
            created_at=event.occurred_at,
            updated_at=event.occurred_at,
        )

    @property
    def is_final(self) -> bool:
        return self.state in FINAL_STATES

    @property
    def has_cancellable_link(self) -> bool:
        return bool(self.payment_link_id) and self.payment_link_status in CANCELLABLE_LINK_STATUSES

    def can_transition_to(self, target: RecoveryCaseState) -> bool:
        return target == self.state or target in ALLOWED_TRANSITIONS[self.state]

    def transition_to(self, target: RecoveryCaseState) -> None:
        if target == self.state:
            return
        if target not in ALLOWED_TRANSITIONS[self.state]:
            raise InvalidStateTransition(f"cannot transition {self.state} -> {target}")
        self.state = target


@dataclass(frozen=True)
class PolicyDecision:
    target_state: RecoveryCaseState
    reason: str
    next_action_at: datetime | None = None


@dataclass(frozen=True)
class ActionRecord:
    id: str
    case_id: str
    action_type: str
    action_key: str
    status: str
    metadata: dict[str, Any]
    created_at: datetime
    #: How many times this action has been claimed. An action_key guarantees
    #: the action is *performed* once, not that one attempt at it is all we
    #: are ever allowed -- see RecoveryStore.claim_action.
    attempts: int = 1
