"""Drives the real RecoveryService through the simulated world.

This deliberately does not reimplement the policy. It instantiates the same
``RecoveryService``, ``RecoveryStore`` and ``BankHealthMonitor`` that serve the
API, feeds it genuine ``PaymentEvent`` webhooks, and translates whatever the
state machine decides into a world action. If the engine has a bug, the
scorecard inherits it -- which is the only arrangement under which the number
says anything about the shipped system.

Method switching
----------------
The action space includes switching the customer to a different method, but
never silently: an eMandate or UPI Autopay mandate is bound to the instrument
it was authorised on, so a switch is expressed as a consented payment link the
customer completes on a method of their choosing. The world scores that path
using its do-not-honour "other method" dynamics, so choosing to escalate a
sticky instrument decline to a link is rewarded exactly as much as it deserves
to be, and no more.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Protocol
from uuid import uuid4

from app.bank_health import BankHealthMonitor
from app.config import Settings
from app.gateway import FakeRazorpayGateway
from app.models import EventType, PaymentEvent, PaymentRail, RecoveryCaseState
from app.service import RecoveryService
from app.sim.runner import Policy
from app.sim.world import ActionKind, Decision, Method, Observation, Outcome, OutcomeKind
from app.store import RecoveryStore

#: How often to look in on a case the engine has no timer for (link live,
#: awaiting the customer). Six hours keeps polling cheap without letting a
#: 24-hour link quietly expire unobserved.
IDLE_POLL_MINUTES = 6 * 60

RAIL_FOR_METHOD = {
    Method.CARD: PaymentRail.CARD,
    Method.UPI_AUTOPAY: PaymentRail.UPI_AUTOPAY,
    Method.EMANDATE: PaymentRail.EMANDATE,
}

#: States from which no further recovery work is possible.
_TERMINAL = {
    RecoveryCaseState.STOPPED,
    RecoveryCaseState.MANUAL_REVIEW,
    RecoveryCaseState.RECOVERED_NATURAL,
    RecoveryCaseState.RECOVERED_BY_LINK,
}


class CodeNormaliser(Protocol):
    """Turns a possibly-absent code plus free-text issuer prose into a code
    the deterministic classifier understands. The seam an LLM plugs into."""

    def normalise(self, failure_code: str | None, issuer_message: str) -> str | None: ...


class PassthroughNormaliser:
    """No normalisation: whatever the acquirer sent is what the engine sees.

    This is the honest pre-LLM baseline. Roughly a third of the world's
    payments arrive with ``failure_code=None``, and every one of those
    classifies as UNKNOWN, so the gap between this and a real classifier is
    the value the model actually adds.
    """

    def normalise(self, failure_code: str | None, issuer_message: str) -> str | None:
        return failure_code


class RailPulsePolicy(Policy):
    name = "railpulse"

    def __init__(
        self,
        settings: Settings | None = None,
        normaliser: CodeNormaliser | None = None,
        label: str | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.normaliser = normaliser or PassthroughNormaliser()
        if label:
            self.name = label
        self.store = RecoveryStore()
        self.health = BankHealthMonitor(
            window=self.settings.health_window,
            # The production default of 30 samples is tuned for a live merchant's
            # traffic. A 500-case batch spread over six issuers never reaches it,
            # so outage detection would be dead code in the benchmark. Lowering
            # it here measures the mechanism; it does not change the engine.
            min_samples=6,
            degraded_success_rate=self.settings.health_degraded_success_rate,
            max_tracked_keys=self.settings.health_max_tracked_keys,
        )
        self.gateway = FakeRazorpayGateway()
        self.service = RecoveryService(
            self.store, self.health, self.gateway, settings=self.settings
        )
        self._ingested: set[str] = set()
        self._consent_sent: set[str] = set()
        self._last: dict[str, Observation] = {}
        #: Counted for the writeup: how often the model/lookup left the engine
        #: with nothing to go on.
        self.unknown_codes = 0

    # --------------------------------------------------------- reconciliation

    def engine_ledger(self) -> dict[str, int]:
        """What the engine's own store believes, independent of the world.

        The module docstring claims the scorecard inherits the engine's bugs.
        It did not: ``_score`` reads only ``World.ledger()``, so an engine that
        silently failed to record a recovery scored identically to one that
        worked. That is exactly what happened -- the adapter sent a fabricated
        payment_link_id, every link recovery was declined as belonging to
        another case, and a hundred-plus cases sat in LINK_SENT while the
        scorecard reported them as recovered.

        Comparing the two ledgers is what turns the claim into a fact, so
        ``tests/test_sim_railpulse.py`` asserts they agree.
        """
        metrics = self.store.metrics()
        return {
            "recovered_cases": metrics["recovered_cases"],
            "recovered_paise": metrics["recovered_amount_paise"],
        }

    # ------------------------------------------------------------- webhooks

    def _event(
        self,
        observation: Observation,
        event_type: EventType,
        failure_code: str | None = None,
    ) -> PaymentEvent:
        return PaymentEvent(
            event_id=f"evt_{uuid4().hex[:16]}",
            event_type=event_type,
            logical_key=observation.payment_id,
            occurred_at=observation.now,
            amount_paise=observation.amount_paise,
            payment_id=observation.payment_id,
            issuer=observation.issuer,
            rail=RAIL_FOR_METHOD[observation.method],
            failure_code=failure_code,
            raw={"issuer_message": observation.issuer_message},
        )

    def _feed_failure(self, observation: Observation) -> None:
        code = self.normaliser.normalise(observation.failure_code, observation.issuer_message)
        if code is None:
            self.unknown_codes += 1
        self.service.ingest(self._event(observation, EventType.PAYMENT_FAILED, code))

    # -------------------------------------------------------------- deciding

    def decide(self, observation: Observation) -> Decision:
        self._last[observation.payment_id] = observation

        if observation.payment_id not in self._ingested:
            self._ingested.add(observation.payment_id)
            self._feed_failure(observation)

        # A due RETRY_SCHEDULED must be executed BEFORE dispatch runs. In
        # production the provider performs that retry and RailPulse only sees
        # the result; dispatch's job is to escalate a retry window that
        # elapsed *without* a success. Calling dispatch first inverts that --
        # it escalates the case to a consented link and the retry never
        # happens, which is why the first run of this adapter recorded zero
        # attempts against every baseline's several thousand.
        case = self.store.get_case(observation.payment_id)
        if (
            case is not None
            and case.state is RecoveryCaseState.RETRY_SCHEDULED
            and case.next_action_at is not None
            and case.next_action_at <= observation.now
        ):
            return Decision(
                ActionKind.RETRY,
                method=observation.method,
                wake_after_minutes=self._wake_minutes(case.next_action_at, observation.now),
            )

        # Otherwise let the engine advance its own timers against sim time.
        self.service.dispatch_due_actions(observation.now)

        case = self.store.get_case(observation.payment_id)
        if case is None:
            return Decision(ActionKind.WAIT, wake_after_minutes=IDLE_POLL_MINUTES)

        wake = self._wake_minutes(case.next_action_at, observation.now)

        if case.state in _TERMINAL:
            return Decision(ActionKind.ABANDON)

        if case.state is RecoveryCaseState.RETRY_SCHEDULED:
            return Decision(ActionKind.WAIT, wake_after_minutes=wake)

        if case.state is RecoveryCaseState.LINK_SENT:
            # dispatch_due_actions has created the link. Present it once; after
            # that the customer's own timing decides, so we just look in.
            if observation.payment_id not in self._consent_sent:
                self._consent_sent.add(observation.payment_id)
                return Decision(
                    ActionKind.REQUEST_CONSENT, wake_after_minutes=IDLE_POLL_MINUTES
                )
            return Decision(ActionKind.WAIT, wake_after_minutes=IDLE_POLL_MINUTES)

        # OPEN, CLASSIFIED, COOLDOWN, CONSENT_REQUIRED, AUTHORIZED_PENDING_CAPTURE:
        # all mean "the engine is deliberately not acting yet".
        return Decision(ActionKind.WAIT, wake_after_minutes=wake)

    def _wake_minutes(self, next_action_at: datetime | None, now: datetime) -> int:
        if next_action_at is None:
            return IDLE_POLL_MINUTES
        delta = (next_action_at - now).total_seconds() / 60.0
        return max(1, min(IDLE_POLL_MINUTES, int(delta) + 1))

    # -------------------------------------------------------------- feedback

    def on_outcome(self, outcome: Outcome) -> None:
        observation = self._last.get(outcome.payment_id)
        if observation is None:
            return

        if outcome.kind is OutcomeKind.RECOVERED:
            via_link = outcome.payment_id in self._consent_sent
            event_type = (
                EventType.PAYMENT_LINK_PAID if via_link else EventType.PAYMENT_CAPTURED
            )
            event = self._event(observation, event_type)
            if via_link:
                # The link id has to be the one the engine actually issued.
                # This used to send the literal "plink_sim", so _handle_link_event
                # compared it against the real plink_demo_* id, concluded the
                # event belonged to some other case, and declined to transition.
                # Result: zero RECOVERED_BY_LINK across an entire 500-case run,
                # with a hundred-plus cases stranded in LINK_SENT -- and the
                # scorecard never noticed, because it scores from the world's
                # ledger rather than the engine's.
                case = self.store.get_case(outcome.payment_id)
                link_id = case.payment_link_id if case else None
                event = PaymentEvent(**{**event.__dict__, "payment_link_id": link_id})
            self.service.ingest(event)
            self.health.observe(
                observation.issuer,
                RAIL_FOR_METHOD[observation.method],
                True,
                observation.now,
            )
            return

        if outcome.kind is OutcomeKind.ATTEMPT_FAILED:
            # A real failed retry is a real webhook: it bumps attempt_count,
            # feeds issuer health, and lets the engine re-decide.
            self._feed_failure(
                Observation(**{**observation.__dict__, "now": observation.now + timedelta(seconds=1)})
            )
            return

        if outcome.kind is OutcomeKind.OPTED_OUT:
            case = self.store.get_case(outcome.payment_id)
            if case is not None and not case.is_final:
                with self.store.transaction():
                    latest = self.store.get_case_by_id(case.id)
                    if latest is not None and latest.can_transition_to(RecoveryCaseState.STOPPED):
                        latest.transition_to(RecoveryCaseState.STOPPED)
                        latest.stop_reason = "customer_opted_out"
                        latest.next_action_at = None
                        self.store.save_case(latest, latest.version)
