"""Baseline policies.

These exist so the agent has something honest to beat. Each is deliberately
simple and each is what some real merchant actually does today:

* ``do-nothing``          -- no dunning at all. Isolates how much recovery is
                             just time passing, which is the number most
                             uplift claims quietly bundle into their own.
* ``retry-3x-immediate``  -- three retries as fast as the gateway allows.
* ``retry-3x-backoff``    -- three retries on a fixed 24h ladder, the most
                             common configuration in the wild.

None of them can read the world's recovery function, and neither can the
agent. That symmetry is the only reason the comparison means anything.
"""

from __future__ import annotations

from app.sim.runner import Policy
from app.sim.world import ActionKind, Decision, Observation

#: A day, in minutes. Baselines poll rather than schedule precisely.
DAY = 24 * 60


class DoNothingPolicy(Policy):
    name = "do-nothing"

    def decide(self, observation: Observation) -> Decision:
        # Wake daily purely so the world can advance and natural self-service
        # recovery has a chance to land. No action is ever taken.
        return Decision(ActionKind.WAIT, wake_after_minutes=DAY)


class RetryThriceImmediatePolicy(Policy):
    name = "retry-3x-immediate"

    def decide(self, observation: Observation) -> Decision:
        if observation.attempts_made >= 3:
            return Decision(ActionKind.ABANDON)
        return Decision(ActionKind.RETRY, method=observation.method, wake_after_minutes=5)


class RetryThriceBackoffPolicy(Policy):
    name = "retry-3x-backoff"

    def decide(self, observation: Observation) -> Decision:
        if observation.attempts_made >= 3:
            return Decision(ActionKind.ABANDON)
        return Decision(ActionKind.RETRY, method=observation.method, wake_after_minutes=DAY)


class StaticDunningPolicy(Policy):
    """Retry ladder plus a contact after every failure -- the aggressive
    configuration RailPulse's contact budget is meant to improve on."""

    name = "static-dunning"

    def decide(self, observation: Observation) -> Decision:
        if observation.attempts_made >= 3:
            return Decision(ActionKind.ABANDON)
        if observation.attempts_made and not observation.consent_outstanding:
            return Decision(ActionKind.REQUEST_CONSENT, wake_after_minutes=DAY)
        return Decision(ActionKind.RETRY, method=observation.method, wake_after_minutes=DAY)


BASELINES: tuple[type[Policy], ...] = (
    DoNothingPolicy,
    RetryThriceImmediatePolicy,
    RetryThriceBackoffPolicy,
    StaticDunningPolicy,
)
