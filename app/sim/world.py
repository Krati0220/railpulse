"""A simulated recovery world whose ground truth no policy can read.

Why this file exists
--------------------
The previous benchmark drew each policy's outcome from a probability that was
assigned to that policy in advance. RailPulse "won" because ``KIND_PROFILE``
said it would. No decision the engine made could change the result, so the
headline rupee figure was a restatement of the constants, not a measurement.

This module replaces that with a world. Every payment carries a latent
recovery function -- when the customer becomes solvent, when the issuer's
outage clears, whether a different instrument would have worked -- and the
only way to discover any of it is to act and observe the consequence. Two
policies given the same seed face a byte-identical world, so the comparison
between them is earned.

The contract, stated so it can be checked
-----------------------------------------
* A policy receives an :class:`Observation` and returns a :class:`Decision`.
* :class:`Observation` carries only what a real recovery system would know
  from a webhook plus its own stored history.
* Latent state lives in ``World._latents`` and is never handed out. A test in
  ``tests/test_sim_world.py`` asserts that no policy module references it.

If that contract is ever broken, every number downstream becomes theatre.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum

# --------------------------------------------------------------------- money

#: Gateway charges for a retry attempt whether or not it succeeds.
RETRY_COST_PAISE = 35
#: Cost of one outbound customer contact (SMS/WhatsApp template).
CONTACT_COST_PAISE = 50

EPOCH = datetime(2026, 8, 1, 0, 0, tzinfo=UTC)


class Method(StrEnum):
    CARD = "card"
    UPI_AUTOPAY = "upi_autopay"
    EMANDATE = "emandate"


class Cause(StrEnum):
    """Ground-truth cause. The policy must infer this; it is never told."""

    INSUFFICIENT_FUNDS = "insufficient_funds"
    ISSUER_OUTAGE = "issuer_outage"
    DO_NOT_HONOUR = "do_not_honour"
    TECHNICAL_DECLINE = "technical_decline"
    INSTRUMENT_DEAD = "instrument_dead"
    MANDATE_CANCELLED = "mandate_cancelled"
    RISK_BLOCKED = "risk_blocked"


class ActionKind(StrEnum):
    WAIT = "wait"
    RETRY = "retry"
    REQUEST_CONSENT = "request_consent"
    ABANDON = "abandon"


class OutcomeKind(StrEnum):
    NOTHING = "nothing"
    RECOVERED = "recovered"
    ATTEMPT_FAILED = "attempt_failed"
    CONSENT_IGNORED = "consent_ignored"
    OPTED_OUT = "opted_out"
    ABANDONED = "abandoned"


# ------------------------------------------------------------ issuer messages

#: Deliberately inconsistent free text, the way real acquirer feeds arrive.
#: The LLM classifier's whole job is normalising this; a lookup table on
#: ``failure_code`` alone cannot, because some codes are absent or reused.
ISSUER_MESSAGES: dict[Cause, tuple[str, ...]] = {
    Cause.INSUFFICIENT_FUNDS: (
        "INSUFFICIENT_FUNDS",
        "Insufficient balance in account",
        "NOT SUFFICIENT FUNDS",
        "acct bal low - declined",
        "51 - insufficient funds",
        "Txn declined: low balance",
    ),
    Cause.ISSUER_OUTAGE: (
        "ISSUER_UNAVAILABLE",
        "Issuer down, please retry",
        "91 - issuer or switch inoperative",
        "bank server not responding",
        "Upstream timeout at issuer",
        "NPCI unavailable - try later",
    ),
    Cause.DO_NOT_HONOUR: (
        "DO_NOT_HONOUR",
        "05 - Do not honor",
        "Transaction not permitted by issuer",
        "declined by bank (no reason given)",
        "GENERIC_DECLINE",
    ),
    Cause.TECHNICAL_DECLINE: (
        "GATEWAY_ERROR",
        "Temporary processing error",
        "network timeout at acquirer",
        "SYSTEM_MALFUNCTION",
        "96 - system malfunction",
    ),
    Cause.INSTRUMENT_DEAD: (
        "CARD_EXPIRED",
        "54 - expired card",
        "Card no longer valid",
        "account closed",
        "INVALID_VPA",
    ),
    Cause.MANDATE_CANCELLED: (
        "MANDATE_CANCELLED",
        "e-mandate revoked by customer",
        "UPI autopay mandate not active",
        "SI cancelled at bank",
    ),
    Cause.RISK_BLOCKED: (
        "SUSPECTED_FRAUD",
        "Risk rejected by issuer",
        "blocked - fraud rules",
        "CHARGEBACK_OPEN",
    ),
}

#: Codes an acquirer actually populates. ``None`` models the very common case
#: where only free text arrives, which is what forces real classification.
CODE_FOR_CAUSE: dict[Cause, tuple[str | None, ...]] = {
    Cause.INSUFFICIENT_FUNDS: ("INSUFFICIENT_FUNDS", None, "51"),
    Cause.ISSUER_OUTAGE: ("ISSUER_UNAVAILABLE", "BANK_DOWN", None, "91"),
    # Note: DO_NOT_HONOUR and RISK_BLOCKED share "05"/GENERIC_DECLINE in the
    # wild. A code lookup cannot separate them; the message sometimes can.
    Cause.DO_NOT_HONOUR: ("GENERIC_DECLINE", None, "05"),
    Cause.TECHNICAL_DECLINE: ("GATEWAY_ERROR", None, "96"),
    Cause.INSTRUMENT_DEAD: ("CARD_EXPIRED", "INVALID_VPA", None),
    Cause.MANDATE_CANCELLED: ("MANDATE_CANCELLED", None),
    Cause.RISK_BLOCKED: ("SUSPECTED_FRAUD", "GENERIC_DECLINE", None, "05"),
}


# --------------------------------------------------------------------- config


@dataclass(frozen=True)
class WorldConfig:
    """Every knob of the hidden recovery function.

    ``shifted()`` returns a materially different world. A policy tuned on the
    default config and evaluated only there has not been shown to generalise;
    the harness reports both so overfitting is visible rather than assumed
    away.
    """

    cause_weights: dict[Cause, float] = field(
        default_factory=lambda: {
            Cause.INSUFFICIENT_FUNDS: 0.30,
            Cause.ISSUER_OUTAGE: 0.18,
            Cause.DO_NOT_HONOUR: 0.16,
            Cause.TECHNICAL_DECLINE: 0.14,
            Cause.INSTRUMENT_DEAD: 0.12,
            Cause.MANDATE_CANCELLED: 0.06,
            Cause.RISK_BLOCKED: 0.04,
        }
    )
    # Insufficient funds: solvency arrives near payday.
    payday_days: tuple[int, ...] = (1, 7, 15, 25)
    solvency_jitter_hours: float = 14.0
    solvency_never_prob: float = 0.18

    # Issuer outage: clears in 30-120 minutes.
    outage_min_minutes: int = 30
    outage_max_minutes: int = 120

    # Do-not-honour: sticky to the instrument, not the customer.
    dnh_same_instrument_prob: float = 0.08
    dnh_other_method_prob: float = 0.62

    # Technical decline: clears almost immediately.
    technical_clear_minutes: int = 3

    # Dead instrument / cancelled mandate: only a new consented instrument.
    dead_consent_success_prob: float = 0.55
    mandate_consent_success_prob: float = 0.42

    # Customer patience. Contacts past tolerance decay response and can
    # trigger a permanent opt-out, which is why contacts are a real cost.
    contact_tolerance: int = 2
    contact_decay: float = 0.55
    opt_out_prob_per_excess_contact: float = 0.22
    base_consent_response_prob: float = 0.58

    #: Some customers notice a failed renewal and pay unprompted once whatever
    #: blocked them has cleared. Without this the do-nothing baseline scores a
    #: trivial zero and tells you nothing about how much recovery is simply
    #: time passing.
    self_serve_daily_prob: float = 0.07

    horizon_days: int = 14
    amount_min_paise: int = 19_900
    amount_max_paise: int = 499_900

    def shifted(self) -> WorldConfig:
        """A second parameterisation: harsher, slower, less forgiving.

        Not a held-out test set -- it was written by the same hand at the same
        time as the defaults, so it can show that a result is not an artefact
        of one set of constants, and cannot show that it is not an artefact of
        the author's assumptions.
        """
        return replace(
            self,
            cause_weights={
                Cause.INSUFFICIENT_FUNDS: 0.22,
                Cause.ISSUER_OUTAGE: 0.12,
                Cause.DO_NOT_HONOUR: 0.22,
                Cause.TECHNICAL_DECLINE: 0.10,
                Cause.INSTRUMENT_DEAD: 0.18,
                Cause.MANDATE_CANCELLED: 0.10,
                Cause.RISK_BLOCKED: 0.06,
            },
            solvency_never_prob=0.28,
            solvency_jitter_hours=22.0,
            outage_max_minutes=240,
            dnh_other_method_prob=0.48,
            dead_consent_success_prob=0.40,
            contact_tolerance=1,
            base_consent_response_prob=0.44,
        )


# --------------------------------------------------------------------- latent


@dataclass
class _Latent:
    """Ground truth for one payment. Never leaves the world."""

    cause: Cause
    method: Method
    issuer: str
    amount_paise: int
    failed_at: datetime

    solvent_from: datetime | None
    outage_until: datetime | None
    technical_clear_at: datetime | None
    dnh_same_ok: bool
    dnh_other_ok: bool
    consent_would_succeed: bool

    failure_code: str | None
    issuer_message: str

    # mutable interaction state
    contacts: int = 0
    attempts: int = 0
    opted_out: bool = False
    resolved: bool = False
    recovered_at: datetime | None = None
    consent_pending_until: datetime | None = None
    cost_paise: int = 0
    last_checked_at: datetime | None = None
    recovered_naturally: bool = False


# ---------------------------------------------------------------- observation


@dataclass(frozen=True)
class Observation:
    """Exactly what a real recovery system would know."""

    payment_id: str
    amount_paise: int
    method: Method
    issuer: str
    failure_code: str | None
    issuer_message: str
    failed_at: datetime
    now: datetime
    attempts_made: int
    contacts_made: int
    consent_outstanding: bool
    #: Observable issuer health: recent failure rate across the batch. This is
    #: inference from the policy's own traffic, not a peek at the outage flag.
    issuer_recent_failure_rate: float


@dataclass(frozen=True)
class Decision:
    action: ActionKind
    #: For RETRY: the method to attempt on. Switching to a method the customer
    #: has not consented to is rejected by the world -- see ``_apply_retry``.
    method: Method | None = None
    #: Minutes until the policy wants to be consulted again.
    wake_after_minutes: int = 60


@dataclass(frozen=True)
class Outcome:
    kind: OutcomeKind
    payment_id: str
    at: datetime
    detail: str = ""


# ---------------------------------------------------------------------- world


class World:
    """Holds latent truth and adjudicates actions against it."""

    def __init__(self, config: WorldConfig | None = None, seed: int = 0, cases: int = 500) -> None:
        self.config = config or WorldConfig()
        self.seed = seed
        self._rng = random.Random(seed)
        self._latents: dict[str, _Latent] = {}
        self._issuers = ("HDFC", "ICICI", "SBI", "AXIS", "KOTAK", "PNB")
        self._build(cases)

    # ----------------------------------------------------------- construction

    def _build(self, cases: int) -> None:
        cfg = self.config
        causes = list(cfg.cause_weights)
        weights = [cfg.cause_weights[c] for c in causes]
        # One outage window per issuer, shared by every payment on it: that is
        # what makes an outage detectable from the policy's own traffic.
        outages: dict[str, tuple[datetime, datetime]] = {}
        for issuer in self._issuers:
            start = EPOCH + timedelta(minutes=self._rng.randrange(0, 60 * 24 * cfg.horizon_days))
            length = self._rng.randint(cfg.outage_min_minutes, cfg.outage_max_minutes)
            outages[issuer] = (start, start + timedelta(minutes=length))

        for index in range(cases):
            cause = self._rng.choices(causes, weights=weights, k=1)[0]
            method = self._rng.choice(list(Method))
            issuer = self._rng.choice(self._issuers)
            amount = self._rng.randrange(cfg.amount_min_paise, cfg.amount_max_paise)

            if cause is Cause.ISSUER_OUTAGE:
                window = outages[issuer]
                failed_at = window[0] + timedelta(
                    minutes=self._rng.randrange(0, max(1, int((window[1] - window[0]).total_seconds() // 60)))
                )
            else:
                failed_at = EPOCH + timedelta(
                    minutes=self._rng.randrange(0, 60 * 24 * (cfg.horizon_days - 3))
                )

            self._latents[f"pay_{index:05d}"] = _Latent(
                cause=cause,
                method=method,
                issuer=issuer,
                amount_paise=amount,
                failed_at=failed_at,
                solvent_from=self._solvency(cause, failed_at),
                outage_until=outages[issuer][1] if cause is Cause.ISSUER_OUTAGE else None,
                technical_clear_at=(
                    failed_at + timedelta(minutes=cfg.technical_clear_minutes)
                    if cause is Cause.TECHNICAL_DECLINE
                    else None
                ),
                dnh_same_ok=(
                    cause is Cause.DO_NOT_HONOUR and self._rng.random() < cfg.dnh_same_instrument_prob
                ),
                dnh_other_ok=(
                    cause is Cause.DO_NOT_HONOUR and self._rng.random() < cfg.dnh_other_method_prob
                ),
                consent_would_succeed=self._consent_truth(cause),
                failure_code=self._rng.choice(CODE_FOR_CAUSE[cause]),
                issuer_message=self._rng.choice(ISSUER_MESSAGES[cause]),
            )

    def _solvency(self, cause: Cause, failed_at: datetime) -> datetime | None:
        if cause is not Cause.INSUFFICIENT_FUNDS:
            return None
        cfg = self.config
        if self._rng.random() < cfg.solvency_never_prob:
            return None
        # Solvency probability rises toward the next payday.
        day = failed_at.day
        next_payday = min((d for d in cfg.payday_days if d > day), default=cfg.payday_days[0] + 31)
        days_out = next_payday - day
        jitter = self._rng.gauss(0, cfg.solvency_jitter_hours)
        return failed_at + timedelta(days=days_out, hours=jitter)

    def _consent_truth(self, cause: Cause) -> bool:
        cfg = self.config
        if cause is Cause.INSTRUMENT_DEAD:
            return self._rng.random() < cfg.dead_consent_success_prob
        if cause is Cause.MANDATE_CANCELLED:
            return self._rng.random() < cfg.mandate_consent_success_prob
        if cause is Cause.RISK_BLOCKED:
            return False
        if cause is Cause.DO_NOT_HONOUR:
            return self._rng.random() < cfg.dnh_other_method_prob
        # Funds/outage/technical resolve on their own; a consented link works
        # only once the underlying block has actually cleared.
        return True

    def settle_to_horizon(self) -> int:
        """Roll the remaining tail for every case still open at the horizon.

        Without this the run is still cadence-dependent, just more subtly: the
        scheduler stops waking a case once its next wake would fall past the
        horizon, so a weekly poller silently forfeits its final partial week of
        hazard while a daily poller does not. That is an artefact of polling
        frequency, not of policy quality, and the do-nothing anchor every
        uplift figure is quoted against sat right on top of it.

        Called once at the end of a run, so every policy is credited with the
        same elapsed time regardless of how often it looked.
        """
        horizon = EPOCH + timedelta(days=self.config.horizon_days)
        settled = 0
        for payment_id, latent in self._latents.items():
            if latent.resolved:
                continue
            start = latent.last_checked_at or latent.failed_at
            if self._natural_recovery_over(latent, payment_id, start, horizon):
                settled += 1
        return settled

    # ------------------------------------------------------------- inspection

    @property
    def payment_ids(self) -> list[str]:
        return list(self._latents)

    def is_resolved(self, payment_id: str) -> bool:
        return self._latents[payment_id].resolved

    def failed_at(self, payment_id: str) -> datetime:
        return self._latents[payment_id].failed_at

    def ledger(self, payment_id: str) -> dict[str, object]:
        """Post-hoc truth, for scoring only. Never call this from a policy."""
        latent = self._latents[payment_id]
        return {
            "cause": latent.cause.value,
            "amount_paise": latent.amount_paise,
            "recovered": latent.recovered_at is not None,
            "attempts": latent.attempts,
            "contacts": latent.contacts,
            "opted_out": latent.opted_out,
            "cost_paise": latent.cost_paise,
            "recovered_naturally": latent.recovered_naturally,
        }

    def observe(self, payment_id: str, now: datetime, issuer_failure_rate: float = 0.0) -> Observation:
        latent = self._latents[payment_id]
        return Observation(
            payment_id=payment_id,
            amount_paise=latent.amount_paise,
            method=latent.method,
            issuer=latent.issuer,
            failure_code=latent.failure_code,
            issuer_message=latent.issuer_message,
            failed_at=latent.failed_at,
            now=now,
            attempts_made=latent.attempts,
            contacts_made=latent.contacts,
            consent_outstanding=(
                latent.consent_pending_until is not None and latent.consent_pending_until > now
            ),
            issuer_recent_failure_rate=issuer_failure_rate,
        )

    # ----------------------------------------------------------------- acting

    def apply(self, payment_id: str, decision: Decision, now: datetime) -> Outcome:
        latent = self._latents[payment_id]
        if latent.resolved:
            return Outcome(OutcomeKind.NOTHING, payment_id, now, "already resolved")

        if decision.action is ActionKind.WAIT:
            natural = self._natural_recovery(latent, payment_id, now)
            if natural is not None:
                return natural
            return self._settle_pending_consent(latent, payment_id, now)
        if decision.action is ActionKind.ABANDON:
            # Giving up means the merchant stops working the case -- not that
            # the customer is forbidden from ever paying. The tail is rolled
            # across the remaining horizon so a policy that abandons early is
            # not punished for it.
            #
            # But this has to be evaluated in *steps*, not as one draw at the
            # horizon. A single draw with now=horizon evaluated _block_cleared
            # at a point where almost everything has cleared, handing the
            # abandoning policy the full elapsed hazard at the most favourable
            # moment. Abandon-on-sight scored 29.8% -- higher than any polling
            # policy, with zero actions and zero cost -- and the do-nothing
            # anchor every uplift figure is quoted against swung from 15.2% to
            # 26.0% purely on how often the policy woke up.
            horizon = EPOCH + timedelta(days=self.config.horizon_days)
            tail = self._natural_recovery_over(latent, payment_id, now, horizon)
            latent.resolved = True
            if tail is not None:
                return tail
            return Outcome(OutcomeKind.ABANDONED, payment_id, now)
        if decision.action is ActionKind.RETRY:
            return self._apply_retry(latent, payment_id, decision, now)
        return self._apply_consent(latent, payment_id, now)

    def _apply_retry(
        self, latent: _Latent, payment_id: str, decision: Decision, now: datetime
    ) -> Outcome:
        method = decision.method or latent.method
        # Consent gate: an eMandate/UPI-Autopay mandate is bound to the
        # instrument it was created on. Charging a different method without a
        # fresh consented authorisation is not available in India, so the
        # world refuses it rather than quietly rewarding an illegal action.
        if method is not latent.method:
            return Outcome(
                OutcomeKind.ATTEMPT_FAILED,
                payment_id,
                now,
                "method switch requires consent; use REQUEST_CONSENT",
            )

        latent.attempts += 1
        latent.cost_paise += RETRY_COST_PAISE
        if self._would_succeed(latent, now):
            latent.resolved = True
            latent.recovered_at = now
            return Outcome(OutcomeKind.RECOVERED, payment_id, now, "retry succeeded")
        return Outcome(OutcomeKind.ATTEMPT_FAILED, payment_id, now, latent.issuer_message)

    def _apply_consent(self, latent: _Latent, payment_id: str, now: datetime) -> Outcome:
        cfg = self.config
        if latent.opted_out:
            return Outcome(OutcomeKind.OPTED_OUT, payment_id, now, "customer opted out")

        latent.contacts += 1
        latent.cost_paise += CONTACT_COST_PAISE

        excess = max(0, latent.contacts - cfg.contact_tolerance)
        if excess and self._rng.random() < cfg.opt_out_prob_per_excess_contact * excess:
            latent.opted_out = True
            latent.resolved = True
            return Outcome(OutcomeKind.OPTED_OUT, payment_id, now, "contacted past tolerance")

        # Response probability decays with every contact past tolerance.
        response = cfg.base_consent_response_prob * (cfg.contact_decay**excess)
        if self._rng.random() >= response:
            return Outcome(OutcomeKind.CONSENT_IGNORED, payment_id, now, "no response")

        # The customer engaged. Whether the money actually moves still depends
        # on the underlying cause having cleared.
        latent.consent_pending_until = now + timedelta(hours=24)
        return self._settle_pending_consent(latent, payment_id, now, forced=True)

    def _settle_pending_consent(
        self, latent: _Latent, payment_id: str, now: datetime, forced: bool = False
    ) -> Outcome:
        if latent.consent_pending_until is None:
            return Outcome(OutcomeKind.NOTHING, payment_id, now)
        if not forced and latent.consent_pending_until < now:
            latent.consent_pending_until = None
            return Outcome(OutcomeKind.CONSENT_IGNORED, payment_id, now, "link expired")
        if latent.consent_would_succeed and self._block_cleared(latent, now):
            latent.resolved = True
            latent.recovered_at = now
            latent.consent_pending_until = None
            return Outcome(OutcomeKind.RECOVERED, payment_id, now, "recovered via consented link")
        return Outcome(OutcomeKind.NOTHING, payment_id, now, "consent open, block not cleared")

    def _natural_recovery_over(
        self, latent: _Latent, payment_id: str, start: datetime, end: datetime
    ) -> Outcome | None:
        """Roll the tail in daily steps so the block is re-checked as it goes.

        Stepping is what makes the hazard cadence-independent: whether a policy
        polls hourly, daily, or abandons and never looks again, the same
        elapsed time yields the same expected recoveries.
        """
        moment = start
        while moment < end:
            moment = min(moment + timedelta(days=1), end)
            outcome = self._natural_recovery(latent, payment_id, moment)
            if outcome is not None:
                return outcome
        return None

    def _natural_recovery(self, latent: _Latent, payment_id: str, now: datetime) -> Outcome | None:
        """The customer pays unprompted. Costs the merchant nothing.

        Hazard is scaled by elapsed time so a policy cannot manufacture
        recoveries by polling more often.
        """
        last = latent.last_checked_at or latent.failed_at
        if latent.opted_out or not self._block_cleared(latent, now):
            # Deliberately do NOT advance last_checked_at here. Time spent
            # blocked is not time in which the customer could have paid, and
            # consuming it meant a policy that polled through an outage threw
            # away the hazard it should have accrued once the block lifted.
            return None
        hours = max(0.0, (now - last).total_seconds() / 3600.0)
        latent.last_checked_at = now
        if latent.cause in (Cause.INSTRUMENT_DEAD, Cause.MANDATE_CANCELLED, Cause.RISK_BLOCKED):
            return None
        daily = self.config.self_serve_daily_prob
        probability = 1.0 - (1.0 - daily) ** (hours / 24.0)
        if self._rng.random() < probability:
            latent.resolved = True
            latent.recovered_at = now
            latent.recovered_naturally = True
            return Outcome(OutcomeKind.RECOVERED, payment_id, now, "customer paid unprompted")
        return None

    def _block_cleared(self, latent: _Latent, now: datetime) -> bool:
        """A consented link still cannot beat an unresolved underlying block."""
        if latent.cause is Cause.INSUFFICIENT_FUNDS:
            return latent.solvent_from is not None and now >= latent.solvent_from
        if latent.cause is Cause.ISSUER_OUTAGE:
            return latent.outage_until is not None and now >= latent.outage_until
        return True

    def _would_succeed(self, latent: _Latent, now: datetime) -> bool:
        """The hidden recovery function. This is the thing under test."""
        cause = latent.cause
        if cause is Cause.INSUFFICIENT_FUNDS:
            return latent.solvent_from is not None and now >= latent.solvent_from
        if cause is Cause.ISSUER_OUTAGE:
            return latent.outage_until is not None and now >= latent.outage_until
        if cause is Cause.TECHNICAL_DECLINE:
            return latent.technical_clear_at is not None and now >= latent.technical_clear_at
        if cause is Cause.DO_NOT_HONOUR:
            return latent.dnh_same_ok
        # Dead instrument, cancelled mandate and risk blocks never clear on a
        # retry of the same instrument, however many times it is attempted.
        return False
