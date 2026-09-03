"""Drives a policy through a world and scores what actually happened.

Every policy sees the same world for a given seed, is woken only when it asks
to be, and is scored on the same five metrics. Nothing here knows which policy
it is running, which is the point: the harness cannot flatter one of them.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

from app.sim.world import (
    EPOCH,
    ActionKind,
    Decision,
    Observation,
    Outcome,
    OutcomeKind,
    World,
)

#: Window over which a policy can infer issuer health from its own traffic.
HEALTH_WINDOW_MINUTES = 60
HEALTH_MIN_SAMPLES = 4


class Policy:
    """Interface every policy implements. Receives observations, nothing more."""

    name = "unnamed"

    def decide(self, observation: Observation) -> Decision:  # pragma: no cover - interface
        raise NotImplementedError

    def on_outcome(self, outcome: Outcome) -> None:
        """Optional hook so a policy can learn from what it just did."""


@dataclass(frozen=True)
class Scorecard:
    policy: str
    cases: int
    recovered_cases: int
    recovered_paise: int
    cost_paise: int
    attempts: int
    contacts: int
    opt_outs: int

    @property
    def net_paise(self) -> int:
        return self.recovered_paise - self.cost_paise

    @property
    def recovery_rate(self) -> float:
        return self.recovered_cases / self.cases if self.cases else 0.0

    @property
    def attempts_per_recovery(self) -> float:
        return self.attempts / self.recovered_cases if self.recovered_cases else float("inf")

    @property
    def contacts_per_recovery(self) -> float:
        return self.contacts / self.recovered_cases if self.recovered_cases else float("inf")

    @property
    def cost_per_rupee_recovered(self) -> float:
        return self.cost_paise / self.recovered_paise if self.recovered_paise else float("inf")

    def as_row(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "net_recovered": self.net_paise,
            "recovery_rate": self.recovery_rate,
            "attempts_per_recovery": self.attempts_per_recovery,
            "contacts_per_recovery": self.contacts_per_recovery,
            "cost_per_rupee": self.cost_per_rupee_recovered,
            "opt_outs": self.opt_outs,
        }


class _IssuerHealth:
    """Observable health, derived only from attempts the policy itself made."""

    def __init__(self) -> None:
        self._events: dict[str, deque[tuple[datetime, bool]]] = {}

    def record(self, issuer: str, at: datetime, failed: bool) -> None:
        self._events.setdefault(issuer, deque()).append((at, failed))

    def failure_rate(self, issuer: str, now: datetime) -> float:
        events = self._events.get(issuer)
        if not events:
            return 0.0
        cutoff = now - timedelta(minutes=HEALTH_WINDOW_MINUTES)
        while events and events[0][0] < cutoff:
            events.popleft()
        if len(events) < HEALTH_MIN_SAMPLES:
            return 0.0
        return sum(1 for _, failed in events if failed) / len(events)


def run(world: World, policy: Policy) -> Scorecard:
    """Run one policy to the horizon and score it."""
    horizon = EPOCH + timedelta(days=world.config.horizon_days)
    health = _IssuerHealth()

    queue: list[tuple[datetime, str]] = [
        (world.failed_at(pid), pid) for pid in world.payment_ids
    ]
    heapq.heapify(queue)

    while queue:
        now, payment_id = heapq.heappop(queue)
        if now > horizon:
            continue
        if world.is_resolved(payment_id):
            continue

        observation = world.observe(
            payment_id,
            now,
            issuer_failure_rate=health.failure_rate(
                world.observe(payment_id, now).issuer, now
            ),
        )
        decision = policy.decide(observation)
        outcome = world.apply(payment_id, decision, now)
        policy.on_outcome(outcome)

        if decision.action is ActionKind.RETRY:
            health.record(
                observation.issuer, now, failed=outcome.kind is not OutcomeKind.RECOVERED
            )

        if decision.action is ActionKind.ABANDON or world.is_resolved(payment_id):
            continue
        next_wake = now + timedelta(minutes=max(1, decision.wake_after_minutes))
        if next_wake <= horizon:
            heapq.heappush(queue, (next_wake, payment_id))

    # Credit every policy with the same elapsed time. The scheduler stops
    # waking a case whose next wake falls past the horizon, so without this a
    # policy that polls rarely forfeits its final partial interval and looks
    # worse purely for having a slower cadence.
    world.settle_to_horizon()
    return _score(world, policy.name)


def _score(world: World, policy_name: str) -> Scorecard:
    recovered_cases = recovered_paise = cost_paise = attempts = contacts = opt_outs = 0
    ids = world.payment_ids
    for payment_id in ids:
        entry = world.ledger(payment_id)
        attempts += int(entry["attempts"])
        contacts += int(entry["contacts"])
        cost_paise += int(entry["cost_paise"])
        if entry["opted_out"]:
            opt_outs += 1
        if entry["recovered"]:
            recovered_cases += 1
            recovered_paise += int(entry["amount_paise"])
    return Scorecard(
        policy=policy_name,
        cases=len(ids),
        recovered_cases=recovered_cases,
        recovered_paise=recovered_paise,
        cost_paise=cost_paise,
        attempts=attempts,
        contacts=contacts,
        opt_outs=opt_outs,
    )


def rupees(paise: float) -> str:
    return f"₹{paise / 100:,.0f}"


def render(scorecards: list[Scorecard]) -> str:
    """A table a judge can read in five seconds."""
    header = (
        f"{'policy':<26}{'net recovered':>16}{'rec. rate':>11}"
        f"{'att/rec':>10}{'contacts/rec':>14}{'opt-outs':>10}"
    )
    lines = [header, "-" * len(header)]
    for card in scorecards:
        lines.append(
            f"{card.policy:<26}{rupees(card.net_paise):>16}{card.recovery_rate:>10.1%}"
            f"{card.attempts_per_recovery:>10.2f}{card.contacts_per_recovery:>14.2f}"
            f"{card.opt_outs:>10}"
        )
    return "\n".join(lines)
