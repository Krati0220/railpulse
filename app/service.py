"""Deterministic recovery orchestration.

The model boundary is deliberately absent from money movement. A future model
may provide a structured explanation, but every state transition below is code
and every action is constrained by a stored policy decision.

Transaction discipline
----------------------
No call to the payment provider happens while a SQLite write lock is held.
Each side effect runs as three phases — record the intent inside a
transaction, perform the network call with no lock, then persist the outcome
in a second transaction. A slow or hanging provider therefore delays one case
rather than blocking every other webhook behind the write lock.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import uuid4

from app.ai_copilot import Language, OutreachCopilot, OutreachPreview
from app.bank_health import BankHealthMonitor
from app.config import Settings
from app.gateway import GatewayError, PaymentGateway
from app.models import (
    FAILURE_EVENTS,
    LINK_EVENTS,
    REVERSAL_EVENTS,
    SUCCESS_EVENTS,
    ActionRecord,
    EventType,
    FailureClass,
    PaymentEvent,
    PolicyDecision,
    RecoveryCase,
    RecoveryCaseState,
    utc_now,
)
from app.store import ConcurrentCaseUpdate, RecoveryStore

logger = logging.getLogger(__name__)

HARD_FAILURES = {
    "CARD_EXPIRED",
    "MANDATE_CANCELLED",
    "ACCOUNT_CLOSED",
    "INVALID_VPA",
    # A generic issuer refusal is sticky to the instrument rather than to the
    # customer: retrying the same card mostly fails again, while the same
    # customer paying by another method usually clears. Classifying it as
    # CUSTOMER_ACTION routes it to a consented link -- which is how this
    # system performs a method switch at all, since a mandate is bound to the
    # instrument it was authorised on and cannot be moved silently.
    "DO_NOT_HONOUR",
}
RISK_FAILURES = {"SUSPECTED_FRAUD", "RISK_REJECTED", "CHARGEBACK_OPEN"}
TRANSIENT_FAILURES = {
    "GATEWAY_ERROR",
    "BANK_DOWN",
    "NETWORK_TIMEOUT",
    "ISSUER_UNAVAILABLE",
    "INSUFFICIENT_FUNDS",
}

#: States that dispatch promotes on a timer before any outreach is considered.
_TIMED_STATES = (RecoveryCaseState.COOLDOWN, RecoveryCaseState.RETRY_SCHEDULED)

#: States where the engine has stopped deciding. Both of them owe the reader a
#: reason: "stopped" without one is unexplainable to a customer, and "needs a
#: human" without one is unactionable by the human it is addressed to.
_PARKED_STATES = {RecoveryCaseState.STOPPED, RecoveryCaseState.MANUAL_REVIEW}

#: States where a decision has already been taken and a further decline on the
#: old instrument must not re-open the case.
_ALREADY_ACTIONED = {
    RecoveryCaseState.CONSENT_REQUIRED,
    RecoveryCaseState.LINK_SENT,
    RecoveryCaseState.AUTHORIZED_PENDING_CAPTURE,
    RecoveryCaseState.MANUAL_REVIEW,
}


@dataclass(frozen=True)
class _PendingCancel:
    """A recorded intent to cancel a link, executed outside the transaction."""

    case_id: str
    action_key: str
    payment_link_id: str


class RecoveryService:
    def __init__(
        self,
        store: RecoveryStore,
        health: BankHealthMonitor,
        gateway: PaymentGateway,
        copilot: OutreachCopilot | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.store = store
        self.health = health
        self.gateway = gateway
        self.copilot = copilot or OutreachCopilot()
        self.settings = settings or Settings()

    # --------------------------------------------------------------- ingest

    def ingest(self, event: PaymentEvent) -> tuple[RecoveryCase | None, bool]:
        """Process one normalized event; returns (case, newly_processed)."""
        pending: _PendingCancel | None = None
        with self.store.transaction():
            if not self.store.claim_event(event.event_id, event.occurred_at):
                return self.store.get_case(event.logical_key), False

            case = self.store.get_case(event.logical_key)
            if event.event_type in FAILURE_EVENTS:
                case = self._handle_failure_event(case, event)
            elif event.event_type in SUCCESS_EVENTS:
                case, pending = self._handle_success_event(case, event)
            elif event.event_type in LINK_EVENTS:
                case = self._handle_link_event(case, event)
            elif event.event_type in REVERSAL_EVENTS:
                case, pending = self._handle_reversal_event(case, event)

        if pending is not None:
            case = self._execute_cancel(pending, event.occurred_at) or case
        return case, True

    # ------------------------------------------------------------- dispatch

    def dispatch_due_actions(self, now: datetime | None = None) -> list[RecoveryCase]:
        """Advance every case whose deterministic policy says it is due.

        Runs in two passes: timers first (cooldown expiry and retry
        escalation), then consented link creation. Cases are selected by index
        rather than by loading and deserialising the whole table.
        """
        now = now or utc_now()
        updated: list[RecoveryCase] = []
        seen: set[str] = set()
        # Bounds how many cooled cases one issuer/rail may release on a single
        # tick, so a recovered outage drains rather than bursts.
        budget: dict[tuple[str, str], int] = {}

        batch = self.settings.dispatch_batch_size
        for snapshot in self.store.due_cases(_TIMED_STATES, now, limit=batch):
            promoted = self._promote_timed_case(snapshot, now, budget)
            if promoted is not None:
                updated.append(promoted)
                seen.add(promoted.id)

        for snapshot in self.store.due_cases(
            [RecoveryCaseState.CONSENT_REQUIRED], now, limit=batch
        ):
            result = self._create_recovery_link(snapshot, now)
            if result is not None:
                if result.id in seen:
                    updated = [case for case in updated if case.id != result.id]
                updated.append(result)
                seen.add(result.id)

        # A link left live on a collected invoice is a second charge waiting to
        # happen, and no further webhook is coming for that case. The tick is
        # the only thing still running, so the sweep rides on it.
        for retried in self.retry_failed_link_cancels(now):
            if retried.id in seen:
                updated = [case for case in updated if case.id != retried.id]
            updated.append(retried)
            seen.add(retried.id)
        return updated

    def _promote_timed_case(
        self,
        snapshot: RecoveryCase,
        now: datetime,
        budget: dict[tuple[str, str], int] | None = None,
    ) -> RecoveryCase | None:
        """Cooldown expiry and retry escalation, both bounded and idempotent."""
        if snapshot.next_action_at is None or snapshot.next_action_at > now:
            return None

        if snapshot.state == RecoveryCaseState.COOLDOWN:
            health = self.health.snapshot(snapshot.issuer, snapshot.rail, now)
            if health.degraded:
                # The rail has not recovered. Waiting longer is the safe action;
                # previously the case was released back into retry regardless.
                target, reason = RecoveryCaseState.COOLDOWN, "issuer_still_degraded"
                next_action_at = now + self.settings.cooldown
            elif not self._claim_release_slot(snapshot, budget):
                # The rail is healthy but this issuer's release quota for the
                # tick is spent. Hold the case and let the next tick take it,
                # so a long outage drains steadily instead of all at once.
                target, reason = RecoveryCaseState.COOLDOWN, "release_rate_limited"
                next_action_at = now + self.settings.release_jitter
            else:
                target, reason = RecoveryCaseState.RETRY_SCHEDULED, "issuer_recovered"
                # Spread the survivors across the jitter window. Keyed on the
                # logical key -- the invoice/subscription identity -- not the
                # case id, which is a uuid4 and would give a different schedule
                # on every replay. The same invoice always lands in the same
                # slot, so dispatch stays reproducible across restarts.
                next_action_at = (
                    now
                    + self.settings.retry_escalation_after
                    + self._release_offset(snapshot.logical_key)
                )
        else:
            # RETRY_SCHEDULED: the provider's own retry window elapsed without a
            # success event. Previously nothing ever picked these up and the case
            # sat in retry_scheduled forever. Escalate to a consented link.
            if snapshot.attempt_count >= self.settings.max_recovery_attempts:
                target, reason = RecoveryCaseState.MANUAL_REVIEW, "retry_attempts_exhausted"
                next_action_at = None
            else:
                allowed, blocked_reason = self._may_contact(snapshot, now)
                if not allowed:
                    target, reason = RecoveryCaseState.STOPPED, blocked_reason or "contact_not_permitted"
                    next_action_at = None
                else:
                    target, reason = RecoveryCaseState.CONSENT_REQUIRED, "retry_window_elapsed"
                    next_action_at = None

        return self._commit_state(snapshot, target, reason, next_action_at, now)

    def _may_contact(self, case: RecoveryCase, now: datetime) -> tuple[bool, str | None]:
        """Budget and opt-out across the whole customer, not one invoice.

        ``CustomerAttentionBudget`` lives on the case, so the limits it names
        were enforced per invoice: five failed subscriptions for one person
        sent five messages in the same second, each case correctly believing it
        had spent one of its two. And a dispute silenced only the disputed
        invoice, so the customer kept being chased about the others.

        The case's own budget is still consulted first -- it is the cheap check
        and it catches the common single-invoice path without a query.
        """
        allowed, reason = case.attention.can_contact(
            now, cooldown=self.settings.contact_cooldown
        )
        if not allowed:
            return False, reason
        if not case.customer_id:
            # No customer on the event. Fall back to per-case behaviour rather
            # than failing open or blocking everything.
            return True, None

        contacts, last_contact, opted_out = self.store.customer_contact_state(
            case.customer_id, exclude_case_id=case.id
        )
        if opted_out:
            return False, "customer_opted_out"
        if case.attention.effective_contacts(now) + contacts >= case.attention.max_contacts_7d:
            return False, "contact_budget_exhausted"
        if last_contact and now - last_contact < self.settings.contact_cooldown:
            return False, "contact_cooldown_active"
        return True, None

    def _claim_release_slot(
        self, snapshot: RecoveryCase, budget: dict[tuple[str, str], int] | None
    ) -> bool:
        """Take one of this issuer/rail's release slots for the current tick."""
        if budget is None:
            return True
        key = (snapshot.issuer, str(snapshot.rail))
        used = budget.get(key, 0)
        if used >= self.settings.max_releases_per_tick:
            return False
        budget[key] = used + 1
        return True

    def _release_offset(self, logical_key: str) -> timedelta:
        """A stable per-invoice offset inside the jitter window.

        Deliberately not ``random``: dispatch must be reproducible, and
        ``hash()`` is salted per process so it would give a different schedule
        on every restart. blake2b is stable across processes and machines.
        """
        window = int(self.settings.release_jitter.total_seconds())
        if window <= 0:
            return timedelta(0)
        digest = hashlib.blake2b(logical_key.encode("utf-8"), digest_size=8).digest()
        return timedelta(seconds=int.from_bytes(digest, "big") % window)

    def _commit_state(
        self,
        snapshot: RecoveryCase,
        target: RecoveryCaseState,
        reason: str,
        next_action_at: datetime | None,
        now: datetime,
    ) -> RecoveryCase | None:
        try:
            with self.store.transaction():
                latest = self.store.get_case_by_id(snapshot.id)
                if latest is None or latest.version != snapshot.version or latest.is_final:
                    return None
                if not latest.can_transition_to(target):
                    return None
                latest.transition_to(target)
                latest.next_action_at = next_action_at
                latest.stop_reason = reason if target == RecoveryCaseState.STOPPED else latest.stop_reason
                latest.updated_at = now
                self.store.save_case(latest, latest.version)
                return latest
        except ConcurrentCaseUpdate:
            logger.info("case %s changed during dispatch; skipping this pass", snapshot.id)
            return None

    def _create_recovery_link(self, snapshot: RecoveryCase, now: datetime) -> RecoveryCase | None:
        allowed, reason = self._may_contact(snapshot, now)
        if not allowed:
            return self._commit_state(
                snapshot, RecoveryCaseState.STOPPED, reason or "contact_not_permitted", None, now
            )

        # The unique action key is the app-level protection against a second
        # recovery link for the same invoice and policy revision.
        action_key = f"{snapshot.logical_key}:payment_link:v1"
        action = ActionRecord(
            id=f"act_{uuid4().hex[:16]}",
            case_id=snapshot.id,
            action_type="payment_link.create",
            action_key=action_key,
            status="started",
            metadata={"amount_paise": snapshot.amount_paise},
            created_at=now,
        )
        with self.store.transaction():
            if not self.store.record_action(action):
                return None

        # --- network call, deliberately outside any transaction --------------
        try:
            link = self.gateway.create_payment_link(
                amount_paise=snapshot.amount_paise,
                reference_id=snapshot.logical_key,
                description=f"Recovery for {snapshot.logical_key}",
            )
        except Exception as exc:
            retryable = isinstance(exc, GatewayError) and exc.retryable
            logger.warning("payment link creation failed for %s: %s", snapshot.id, exc)
            with self.store.transaction():
                self.store.complete_action(
                    action_key,
                    status="failed",
                    metadata={"error_type": type(exc).__name__, "retryable": retryable},
                    completed_at=now,
                )
                latest = self.store.get_case_by_id(snapshot.id)
                if latest is None or latest.state != RecoveryCaseState.CONSENT_REQUIRED:
                    return None
                latest.transition_to(RecoveryCaseState.MANUAL_REVIEW)
                latest.stop_reason = (
                    "payment_link_create_unavailable" if retryable else "payment_link_create_rejected"
                )
                latest.updated_at = now
                self.store.save_case(latest, latest.version)
                return latest

        with self.store.transaction():
            latest = self.store.get_case_by_id(snapshot.id)
            if latest is None or latest.is_final or latest.state != RecoveryCaseState.CONSENT_REQUIRED:
                # A concurrent late authorisation won. Cancel the freshly created
                # link rather than risk a second collection.
                self._cancel_quietly(link.id)
                self.store.complete_action(
                    action_key,
                    status="superseded",
                    metadata={"payment_link_id": link.id},
                    completed_at=now,
                )
                return None
            latest.payment_link_id = link.id
            latest.payment_link_url = link.short_url
            latest.payment_link_status = link.status
            latest.attention.record_contact(now)
            latest.transition_to(RecoveryCaseState.LINK_SENT)
            latest.next_action_at = None
            latest.updated_at = now
            self.store.save_case(latest, latest.version)
            self.store.complete_action(
                action_key,
                status="succeeded",
                metadata={"payment_link_id": link.id, "payment_link_status": link.status},
                completed_at=now,
            )
            return latest

    def _cancel_quietly(self, link_id: str) -> None:
        try:
            self.gateway.cancel_payment_link(link_id)
        except Exception as exc:  # pragma: no cover - best effort compensation
            logger.error("failed to cancel superseded payment link %s: %s", link_id, exc)

    # ------------------------------------------------------------- outreach

    def create_outreach_preview(
        self,
        case_id: str,
        *,
        language: Language = "hinglish",
        now: datetime | None = None,
        copilot: OutreachCopilot | None = None,
    ) -> OutreachPreview:
        """Generate a merchant-reviewable draft; this method never sends it.

        ``copilot`` lets a caller substitute a different drafting provider --
        used by the demo to show the validator rejecting a non-compliant draft.
        The policy checks themselves are not overridable, which is the point:
        swapping the provider changes what is *proposed*, never what is
        *allowed*.
        """
        now = now or utc_now()
        case = self.store.get_case_by_id(case_id)
        if case is None:
            raise LookupError("recovery case not found")
        if case.state != RecoveryCaseState.LINK_SENT or not case.has_cancellable_link:
            raise ValueError("an active unpaid recovery link is required for outreach preview")
        if not case.payment_link_url:
            raise ValueError("recovery link URL is unavailable")

        drafting = copilot or self.copilot
        preview = drafting.generate_preview(case, case.payment_link_url, language=language)
        with self.store.transaction():
            latest = self.store.get_case_by_id(case_id)
            if latest is None:
                raise LookupError("recovery case not found")
            if latest.state != RecoveryCaseState.LINK_SENT or latest.payment_link_id != case.payment_link_id:
                raise ValueError("recovery case changed while preview was generated")
            latest.outreach_preview = preview.model_dump(mode="json")
            latest.updated_at = now
            self.store.save_case(latest, latest.version)
        return preview

    # -------------------------------------------------------- event handlers

    def _handle_failure_event(self, existing: RecoveryCase | None, event: PaymentEvent) -> RecoveryCase:
        self.health.observe(event.issuer, event.rail, False, event.occurred_at)
        case = existing or RecoveryCase.from_failed_event(event)
        if existing is None:
            self.store.insert_case(case)
        if case.is_final:
            return case
        if case.state in _ALREADY_ACTIONED:
            # A later failed/subscription-state event cannot reopen an already
            # actioned or payment-held case. Preserve the safer current state.
            #
            # CONSENT_REQUIRED belongs here for the same reason and was missing:
            # once the decision to ask the customer for a fresh authorisation
            # has been taken, another decline on the old instrument is expected
            # and changes nothing. Leaving it out was actively harmful --
            # decide() would return RETRY_SCHEDULED, for which no edge exists
            # out of CONSENT_REQUIRED, so transition_to raised, the whole ingest
            # transaction rolled back *including the claim_event row*, and the
            # API returned 409. Deterministic for a given body, so every
            # Razorpay redelivery reproduced it and the event was never applied.
            return self._record_attempt_without_transition(case, event)

        case.failure_code = event.failure_code or case.failure_code
        case.failure_class = self.classify_failure(case.failure_code)
        case.issuer = event.issuer.lower()
        case.rail = event.rail
        # The amount can arrive late: a subscription.pending carries no payment
        # entity, so the case is created with 0 and the real figure only shows
        # up on the following payment.failed. Without this the engine later
        # asks the gateway for a zero-value link, is permanently rejected, and
        # parks a recoverable case in MANUAL_REVIEW.
        if event.amount_paise > 0:
            case.amount_paise = event.amount_paise
        case.attempt_count += 1
        case.updated_at = max(case.updated_at, event.occurred_at)
        decision = self.decide(case, event.occurred_at)
        if not case.can_transition_to(decision.target_state):
            # Defence in depth. A policy decision the transition table does not
            # allow must never raise out of ingest: the rollback would discard
            # the idempotency claim and the provider would redeliver the same
            # body forever. Keep the safer current state and say so.
            logger.warning(
                "case %s: policy proposed %s from %s, which is not permitted; holding state",
                case.id,
                decision.target_state.value,
                case.state.value,
            )
            self.store.save_case(case, case.version)
            return case
        case.transition_to(decision.target_state)
        case.next_action_at = decision.next_action_at
        # MANUAL_REVIEW used to clear this, so a case sat in front of a human
        # announcing that it needed attention and not saying what for. The
        # dashboard's "needs a human" tile had nothing to show, and neither did
        # an operator deciding whether to reopen it.
        case.stop_reason = (
            decision.reason if decision.target_state in _PARKED_STATES else None
        )
        self.store.save_case(case, case.version)
        return case

    def _handle_reversal_event(
        self, case: RecoveryCase | None, event: PaymentEvent
    ) -> tuple[RecoveryCase | None, _PendingCancel | None]:
        """A refund or dispute took the money back.

        Previously these events had no EventType at all, so normalize_webhook
        raised and the API answered 400. Razorpay reads 400 as a permanent
        rejection and drops the delivery -- meaning a dispute raised against a
        payment this system had just claimed as recovered was silently
        discarded, the case stayed RECOVERED_BY_LINK, and the merchant's
        recovery figure kept counting money they no longer had.

        Three things have to happen, in this order of importance:

        1. Stop. Never retry a disputed payment, and never contact the
           customer again -- chasing someone who has just disputed a charge is
           how a dispute becomes a regulatory complaint.
        2. Cancel any live recovery link, so the customer cannot pay a link for
           an invoice that has already been refunded.
        3. Subtract it from recovered revenue.
        """
        if case is None:
            # A reversal for something never seen. Nothing to correct, but it
            # is worth a log line rather than silence.
            logger.warning(
                "reversal %s for unknown case %s; nothing to reverse",
                event.event_type.value,
                event.logical_key,
            )
            return None, None

        if case.state is RecoveryCaseState.RECOVERY_REVERSED:
            return case, None

        pending: _PendingCancel | None = None
        if case.has_cancellable_link and case.payment_link_id:
            action_key = f"{case.logical_key}:payment_link:cancel:reversal"
            action = ActionRecord(
                id=f"act_{uuid4().hex[:16]}",
                case_id=case.id,
                action_type="payment_link.cancel",
                action_key=action_key,
                status="started",
                metadata={
                    "trigger": event.event_type.value,
                    # Recorded so the retry sweep can re-issue this cancel
                    # without depending on the case still naming the link.
                    "payment_link_id": case.payment_link_id,
                },
                created_at=event.occurred_at,
            )
            if self.store.claim_action(
                action, max_attempts=self.settings.max_link_cancel_attempts
            ):
                pending = _PendingCancel(case.id, action_key, case.payment_link_id)

        # Silence outreach permanently, across every invoice belonging to this
        # person. The contact budget is the wrong tool here -- this is not
        # "spend carefully", it is "do not contact at all" -- and doing it on
        # one case would leave the customer being chased about their others.
        case.attention.opted_out = True
        if case.customer_id:
            silenced = self.store.opt_out_customer(case.customer_id, exclude_case_id=case.id)
            if silenced:
                logger.info(
                    "reversal on %s silenced %s other case(s) for customer %s",
                    case.id,
                    silenced,
                    case.customer_id,
                )
        case.reversal_reason = event.event_type.value
        case.reversed_at = event.occurred_at
        case.requires_manual_reconciliation = True
        case.updated_at = max(case.updated_at, event.occurred_at)
        if case.can_transition_to(RecoveryCaseState.RECOVERY_REVERSED):
            case.transition_to(RecoveryCaseState.RECOVERY_REVERSED)
        self.store.save_case(case, case.version)
        logger.info(
            "case %s reversed by %s; outreach disabled", case.id, event.event_type.value
        )
        return case, pending

    def _record_attempt_without_transition(
        self, case: RecoveryCase, event: PaymentEvent
    ) -> RecoveryCase:
        """Count a failure against a case whose state must not change.

        The attempt still happened: it feeds the attempt cap and the issuer
        health signal. Only the state is preserved.
        """
        case.attempt_count += 1
        case.updated_at = max(case.updated_at, event.occurred_at)
        if event.amount_paise > 0 and case.amount_paise <= 0:
            case.amount_paise = event.amount_paise
        self.store.save_case(case, case.version)
        return case

    def _handle_success_event(
        self, case: RecoveryCase | None, event: PaymentEvent
    ) -> tuple[RecoveryCase | None, _PendingCancel | None]:
        self.health.observe(event.issuer, event.rail, True, event.occurred_at)
        if case is None:
            return None, None
        if case.is_final:
            return case, None

        case.updated_at = max(case.updated_at, event.occurred_at)
        captured = event.captured or event.event_type in {EventType.PAYMENT_CAPTURED, EventType.ORDER_PAID}
        if captured:
            case.transition_to(RecoveryCaseState.RECOVERED_NATURAL)
            case.stop_reason = "original_payment_captured"
        else:
            # Do not report money recovered before capture/order-paid. Still
            # suppress a new recovery action while the original payment is held.
            case.transition_to(RecoveryCaseState.AUTHORIZED_PENDING_CAPTURE)
            case.stop_reason = "late_authorisation_pending_capture"
        case.next_action_at = None
        pending = self._schedule_link_cancel(case, event.occurred_at)
        self.store.save_case(case, case.version)
        return case, pending

    def _handle_link_event(self, case: RecoveryCase | None, event: PaymentEvent) -> RecoveryCase | None:
        if case is None:
            return None
        if event.payment_link_id and case.payment_link_id and event.payment_link_id != case.payment_link_id:
            # A link belongs to a different recovery case; do not mutate this one.
            return case
        case.updated_at = max(case.updated_at, event.occurred_at)
        if case.is_final:
            # Do not reopen a final case. A late payment against a cancelled or
            # already-settled link is retained as a reconciliation signal.
            case.requires_manual_reconciliation = True
            case.stop_reason = "late_link_event_requires_reconciliation"
            self.store.save_case(case, case.version)
            return case

        case.outreach_preview = None
        awaiting_capture = case.state == RecoveryCaseState.AUTHORIZED_PENDING_CAPTURE
        target: RecoveryCaseState | None
        reason = ""

        if event.event_type == EventType.PAYMENT_LINK_PAID:
            case.payment_link_status = "paid"
            if awaiting_capture:
                target = RecoveryCaseState.MANUAL_REVIEW
                reason = "duplicate_collection_requires_reconciliation"
            else:
                target, reason = RecoveryCaseState.RECOVERED_BY_LINK, "payment_link_captured"
        elif event.event_type == EventType.PAYMENT_LINK_PARTIALLY_PAID:
            case.payment_link_status = "partially_paid"
            target, reason = RecoveryCaseState.MANUAL_REVIEW, "partial_recovery_requires_reconciliation"
        elif event.event_type == EventType.PAYMENT_LINK_CANCELLED:
            case.payment_link_status = "cancelled"
            # While awaiting capture this is the echo of RailPulse's own
            # cancellation, which is the expected outcome rather than a problem.
            target = None if awaiting_capture else RecoveryCaseState.MANUAL_REVIEW
            reason = "recovery_link_cancelled"
        else:
            case.payment_link_status = "expired"
            target = None if awaiting_capture else RecoveryCaseState.MANUAL_REVIEW
            reason = "recovery_link_expired"

        if target is not None:
            if case.can_transition_to(target):
                case.transition_to(target)
                case.stop_reason = reason
                case.next_action_at = None
            else:
                # An unexpected ordering should be reconciled by a human, never
                # crash the webhook and trigger an infinite provider retry.
                case.requires_manual_reconciliation = True
                case.stop_reason = f"unexpected_link_event_from_{case.state.value}"
        self.store.save_case(case, case.version)
        return case

    # ------------------------------------------------------- side effects

    def _schedule_link_cancel(self, case: RecoveryCase, now: datetime) -> _PendingCancel | None:
        """Record the intent to cancel; the network call happens after commit."""
        if not case.has_cancellable_link or case.payment_link_id is None:
            return None
        action_key = f"{case.logical_key}:payment_link.cancel:{case.payment_link_id}"
        action = ActionRecord(
            id=f"act_{uuid4().hex[:16]}",
            case_id=case.id,
            action_type="payment_link.cancel",
            action_key=action_key,
            status="started",
            metadata={"payment_link_id": case.payment_link_id},
            created_at=now,
        )
        if not self.store.claim_action(
            action, max_attempts=self.settings.max_link_cancel_attempts
        ):
            return None
        # The preview points at a link that is about to be revoked.
        case.outreach_preview = None
        return _PendingCancel(case.id, action_key, case.payment_link_id)

    def retry_failed_link_cancels(self, now: datetime) -> list[RecoveryCase]:
        """Re-attempt cancellations that failed, without waiting for a webhook.

        A payment link is a bearer URL: whoever holds it can pay it. When an
        invoice is collected the link must be revoked, and if that gateway call
        times out the link stays live and payable on an invoice that is already
        settled. Before this, the only path back was another webhook arriving
        for the same case -- which, for a collected invoice, is precisely the
        traffic that has stopped -- and even that was refused, because the
        action key had been burned by the attempt that failed.

        So the sweep runs on the dispatch tick, which keeps ticking whether or
        not the merchant sends anything else. Cancelling is idempotent at the
        provider, so a retry costs nothing when the first attempt actually
        landed and the response was what got lost.
        """
        updated: list[RecoveryCase] = []
        stale = self.store.failed_actions(
            "payment_link.cancel",
            max_attempts=self.settings.max_link_cancel_attempts,
            limit=self.settings.link_cancel_retry_batch,
        )
        for action in stale:
            case = self.store.get_case_by_id(action.case_id)
            if case is None:
                continue
            link_id = action.metadata.get("payment_link_id") or case.payment_link_id
            if not link_id:
                continue
            if case.payment_link_status == "cancelled":
                # Something else revoked it. Close the action rather than
                # calling the gateway again, so the row stops being swept.
                with self.store.transaction():
                    self.store.complete_action(
                        action.action_key,
                        status="succeeded",
                        metadata={"resolved_by": "already_cancelled"},
                        completed_at=now,
                    )
                continue
            # A fresh id: the row already exists, and ON CONFLICT targets
            # action_key, so reusing the old id risks tripping the primary key
            # instead of taking the update branch.
            claim = ActionRecord(
                id=f"act_{uuid4().hex[:16]}",
                case_id=action.case_id,
                action_type=action.action_type,
                action_key=action.action_key,
                status="started",
                metadata=action.metadata,
                created_at=action.created_at,
            )
            if not self.store.claim_action(
                claim, max_attempts=self.settings.max_link_cancel_attempts
            ):
                continue
            result = self._execute_cancel(
                _PendingCancel(case.id, action.action_key, str(link_id)), now
            )
            if result is not None:
                updated.append(result)
        return updated

    def _execute_cancel(self, pending: _PendingCancel, now: datetime) -> RecoveryCase | None:
        status: str | None = None
        error: Exception | None = None
        try:
            status = self.gateway.cancel_payment_link(pending.payment_link_id)
        except Exception as exc:
            error = exc
            logger.warning("payment link cancel failed for %s: %s", pending.case_id, exc)

        with self.store.transaction():
            latest = self.store.get_case_by_id(pending.case_id)
            if error is not None:
                self.store.complete_action(
                    pending.action_key,
                    status="failed",
                    metadata={
                        "payment_link_id": pending.payment_link_id,
                        "error_type": type(error).__name__,
                    },
                    completed_at=now,
                )
                if latest is not None:
                    # The original payment is still held or recovered, so no new
                    # outreach can be created. Flag it for reconciliation.
                    latest.requires_manual_reconciliation = True
                    latest.stop_reason = "payment_link_cancel_failed"
                    latest.updated_at = now
                    self.store.save_case(latest, latest.version)
                return latest

            self.store.complete_action(
                pending.action_key,
                status="succeeded",
                metadata={"payment_link_id": pending.payment_link_id, "payment_link_status": status},
                completed_at=now,
            )
            if latest is None:
                return None
            latest.payment_link_status = status
            latest.outreach_preview = None
            if status != "cancelled":
                latest.requires_manual_reconciliation = True
                latest.stop_reason = "link_cancel_returned_non_cancellable_status"
            elif latest.stop_reason == "payment_link_cancel_failed":
                # An earlier attempt failed and put this case in front of a
                # human. The link is now revoked, so the flag is stale -- and a
                # reconciliation queue that fills with items which have already
                # resolved themselves is one nobody reads.
                latest.requires_manual_reconciliation = False
                latest.stop_reason = None
            latest.updated_at = now
            self.store.save_case(latest, latest.version)
            return latest

    # ------------------------------------------------------ operator actions

    def reopen(
        self,
        case_id: str,
        now: datetime,
        *,
        note: str,
        failure_class: FailureClass | None = None,
    ) -> RecoveryCase:
        """Put a case that was parked for a human back into the machine.

        MANUAL_REVIEW was a one-way door. That is wrong for both of the reasons
        it is reached: a provider outage that made every code unclassifiable is
        transient, and "four attempts is enough for an automated system to
        decide on its own" is not the same claim as "this invoice is
        unrecoverable". With no way back, a bad ten minutes at the classifier
        permanently stranded every case that arrived during it.

        The door opens only from the outside, and only for MANUAL_REVIEW.
        STOPPED is never reopenable here, whatever the caller passes: it is
        reached by a customer opting out, by a risk decision, and by a dispute,
        and an endpoint that could reverse any of those would be a way to
        resume contacting someone who told the merchant to stop. That guarantee
        is worth more than the convenience of an override, so the override does
        not exist -- a genuinely mistaken opt-out is a deliberate data
        correction, not a button on a dashboard.

        Reopening grants a fresh attempt budget, because that is the decision
        the human is actually making, and records who did it and why in the
        same audit trail as everything else.
        """
        with self.store.transaction():
            case = self.store.get_case_by_id(case_id)
            if case is None:
                raise LookupError(f"no such case: {case_id}")
            if case.state is not RecoveryCaseState.MANUAL_REVIEW:
                raise ValueError(
                    "only a case in manual_review can be reopened; "
                    f"this one is {case.state.value}"
                )

            self.store.record_action(
                ActionRecord(
                    id=f"act_{uuid4().hex[:16]}",
                    case_id=case.id,
                    action_type="case.reopen",
                    action_key=f"{case.logical_key}:reopen:{now.isoformat()}",
                    status="succeeded",
                    metadata={
                        "note": note[:500],
                        "previous_stop_reason": case.stop_reason,
                        "previous_attempt_count": case.attempt_count,
                        "failure_class_set": failure_class.value if failure_class else None,
                    },
                    created_at=now,
                )
            )

            if failure_class is not None:
                case.failure_class = failure_class
            case.attempt_count = 0
            case.stop_reason = None
            case.next_action_at = None

            # Re-decide rather than picking a target state: the opt-out and
            # risk guards live in decide(), so routing through it means a
            # reopen can never step around them.
            decision = self.decide(case, now)
            if case.can_transition_to(decision.target_state):
                case.transition_to(decision.target_state)
            case.stop_reason = (
                decision.reason if decision.target_state in _PARKED_STATES else None
            )
            case.next_action_at = decision.next_action_at
            case.updated_at = now
            self.store.save_case(case, case.version)
            return case

    # ----------------------------------------------------------------- policy

    def decide(self, case: RecoveryCase, now: datetime) -> PolicyDecision:
        if case.attention.opted_out:
            return PolicyDecision(RecoveryCaseState.STOPPED, "customer_opted_out")
        if case.failure_class == FailureClass.RISK:
            return PolicyDecision(RecoveryCaseState.STOPPED, "risk_policy_stop")
        if case.attempt_count >= self.settings.max_recovery_attempts:
            # >= not >. The dispatch path at _promote_timed_case already used
            # >=, so the two paths disagreed by one and a case could take five
            # attempts through ingest but only four through dispatch.
            return PolicyDecision(RecoveryCaseState.MANUAL_REVIEW, "retry_attempts_exhausted")
        if case.failure_class == FailureClass.UNKNOWN:
            # The classifier could not say what happened -- an absent code, an
            # answer below the confidence floor, or a provider outage.
            #
            # This branch did not exist, so an unclassified case fell through
            # to the health check and out the bottom as RETRY_SCHEDULED. That
            # is precisely backwards: the model saying "this looks like a dead
            # account but I am only 30% sure" was converted into a retry
            # against a dead account, which is the harm the confidence floor
            # exists to prevent. Not knowing why a payment failed is a reason
            # to ask a human, not a reason to try again.
            return PolicyDecision(RecoveryCaseState.MANUAL_REVIEW, "unclassified_failure")
        if case.failure_class == FailureClass.CUSTOMER_ACTION:
            return PolicyDecision(RecoveryCaseState.CONSENT_REQUIRED, "customer_action_required")

        health = self.health.snapshot(case.issuer, case.rail, now)
        if health.degraded:
            return PolicyDecision(
                RecoveryCaseState.COOLDOWN,
                f"observed_issuer_degradation:{health.success_rate:.0%}/{health.attempts}",
                now + self.settings.cooldown,
            )
        # A retry belongs to the provider's own schedule. RailPulse simply sets
        # the deadline after which an unrecovered case escalates to consent.
        return PolicyDecision(
            RecoveryCaseState.RETRY_SCHEDULED,
            "safe_retry_window",
            now + self.settings.retry_escalation_after,
        )

    @staticmethod
    def classify_failure(failure_code: str | None) -> FailureClass:
        normalized = (failure_code or "").upper()
        if normalized in HARD_FAILURES:
            return FailureClass.CUSTOMER_ACTION
        if normalized in RISK_FAILURES:
            return FailureClass.RISK
        if normalized in TRANSIENT_FAILURES:
            return FailureClass.TRANSIENT
        return FailureClass.UNKNOWN

