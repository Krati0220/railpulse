"""SUPERSEDED by ``app/sim/``. Run ``python -m app.sim.report`` instead.

Kept for one honest reason: ``measure_duplicate_actions`` replays real webhook
redeliveries and a late authorisation through the real engine and counts what
the store actually recorded. That number was always earned.

The rupee figures were not. ``KIND_PROFILE`` assigns each policy its own
recovery probability in advance -- RailPulse 0.76 on an issuer outage against
static dunning's 0.35 -- so "incremental net recovered" restates those
constants rather than measuring any decision the engine made. No change in
behaviour could have moved it. ``app/sim/`` replaces this with a world holding
a latent recovery function no policy can read, where outcomes follow from what
a policy actually did.

This file is left in the tree so that substitution is visible rather than
quietly erased from history.

--- original docstring ---

Reproducible paired replay benchmark.

This is intentionally not described as a real-world A/B result. Both policies
receive the same seeded synthetic cases, so the metric is an engineering
comparison, not a production-revenue claim.

Two things changed from the first version, both about honesty of the number:

* A single seed produced a single point estimate that looked more precise than
  it was. The headline now runs many independent seeds and reports a mean with
  a 95% confidence interval, so the spread is visible.
* ``duplicate_recovery_actions`` used to be the hardcoded literal ``0``. It is
  now produced by replaying events — including deliberate webhook redeliveries
  and a late authorisation — through the real engine and counting what the
  store actually recorded.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from statistics import mean, stdev

from app.bank_health import BankHealthMonitor
from app.config import Settings
from app.gateway import FakeRazorpayGateway
from app.models import EventType, PaymentEvent, PaymentRail
from app.service import RecoveryService
from app.store import RecoveryStore

CONTACT_COST_PAISE = 50
DEFAULT_SEED = 20260821
DEFAULT_TRIALS = 40


@dataclass(frozen=True)
class SyntheticCase:
    amount_paise: int
    kind: str
    railpulse_recovers: bool
    baseline_recovers: bool
    railpulse_contacts: int
    baseline_contacts: int


@dataclass(frozen=True)
class Interval:
    """Mean of a repeated measurement with a 95% confidence interval."""

    mean: float
    low: float
    high: float
    trials: int

    @classmethod
    def from_samples(cls, samples: list[float]) -> Interval:
        if not samples:
            return cls(0.0, 0.0, 0.0, 0)
        average = mean(samples)
        if len(samples) < 2:
            return cls(average, average, average, len(samples))
        margin = 1.96 * (stdev(samples) / math.sqrt(len(samples)))
        return cls(average, average - margin, average + margin, len(samples))


@dataclass(frozen=True)
class BenchmarkReport:
    cases: int
    railpulse_net_recovered_paise: int
    baseline_net_recovered_paise: int
    incremental_net_recovered_paise: int
    railpulse_contacts: int
    baseline_contacts: int
    contact_reduction_rate: float
    duplicate_recovery_actions: int
    incremental_interval: Interval | None = None
    contact_reduction_interval: Interval | None = None

    def as_text(self) -> str:
        def rupees(paise: float) -> str:
            return f"₹{paise / 100:,.2f}"

        lines = [
            "RailPulse paired replay benchmark (synthetic, seeded)",
            f"Cases per trial: {self.cases}",
            f"RailPulse net recovered: {rupees(self.railpulse_net_recovered_paise)}",
            f"Static-dunning net recovered: {rupees(self.baseline_net_recovered_paise)}",
            f"Incremental net recovered: {rupees(self.incremental_net_recovered_paise)}",
            f"RailPulse contacts: {self.railpulse_contacts}",
            f"Static-dunning contacts: {self.baseline_contacts}",
            f"Contact reduction: {self.contact_reduction_rate:.1%}",
            f"Duplicate recovery actions observed: {self.duplicate_recovery_actions}",
        ]
        if self.incremental_interval and self.incremental_interval.trials > 1:
            interval = self.incremental_interval
            lines += [
                "",
                f"Across {interval.trials} independent seeds:",
                f"  Incremental net recovered: {rupees(interval.mean)} "
                f"(95% CI {rupees(interval.low)} – {rupees(interval.high)})",
            ]
            if self.contact_reduction_interval:
                reduction = self.contact_reduction_interval
                lines.append(
                    f"  Contact reduction: {reduction.mean:.1%} "
                    f"(95% CI {reduction.low:.1%} – {reduction.high:.1%})"
                )
            lines.append(
                "  Synthetic recovery probabilities encode the project hypothesis; "
                "the interval shows sampling spread only, not real-merchant uplift."
            )
        return "\n".join(lines)


# --------------------------------------------------------------- generation

#: (railpulse_p, baseline_p, railpulse_contacts, baseline_contacts)
KIND_PROFILE = {
    "issuer_outage": (0.76, 0.35, 0, 3),
    "insufficient_funds": (0.52, 0.46, 0, 3),
    "expired_card": (0.64, 0.12, 1, 3),
    "mandate_cancelled": (0.50, 0.08, 1, 3),
    "risk": (0.00, 0.00, 0, 2),
}
KINDS = list(KIND_PROFILE)
KIND_WEIGHTS = [25, 30, 20, 15, 10]


def generate_cases(count: int = 500, seed: int = DEFAULT_SEED) -> list[SyntheticCase]:
    rng = random.Random(seed)
    cases: list[SyntheticCase] = []
    for _ in range(count):
        kind = rng.choices(KINDS, weights=KIND_WEIGHTS, k=1)[0]
        amount = rng.randrange(19900, 499900)
        # The simulator deliberately encodes the project hypothesis: waiting out
        # an outage, and a consented link for expired credentials, outperform a
        # static retry. Risk cases are stopped, avoiding unnecessary contact.
        railpulse_p, baseline_p, rp_contacts, baseline_contacts = KIND_PROFILE[kind]
        cases.append(
            SyntheticCase(
                amount_paise=amount,
                kind=kind,
                railpulse_recovers=rng.random() < railpulse_p,
                baseline_recovers=rng.random() < baseline_p,
                railpulse_contacts=rp_contacts,
                baseline_contacts=baseline_contacts,
            )
        )
    return cases


@dataclass(frozen=True)
class _Trial:
    incremental_net: int
    contact_reduction: float
    railpulse_net: int
    baseline_net: int
    railpulse_contacts: int
    baseline_contacts: int


def _score(cases: list[SyntheticCase]) -> _Trial:
    railpulse_gross = sum(case.amount_paise for case in cases if case.railpulse_recovers)
    baseline_gross = sum(case.amount_paise for case in cases if case.baseline_recovers)
    railpulse_contacts = sum(case.railpulse_contacts for case in cases)
    baseline_contacts = sum(case.baseline_contacts for case in cases)
    railpulse_net = railpulse_gross - railpulse_contacts * CONTACT_COST_PAISE
    baseline_net = baseline_gross - baseline_contacts * CONTACT_COST_PAISE
    return _Trial(
        incremental_net=railpulse_net - baseline_net,
        contact_reduction=1 - (railpulse_contacts / baseline_contacts) if baseline_contacts else 0.0,
        railpulse_net=railpulse_net,
        baseline_net=baseline_net,
        railpulse_contacts=railpulse_contacts,
        baseline_contacts=baseline_contacts,
    )


# ------------------------------------------------------ measured duplicates


def measure_duplicate_actions(*, cases: int = 50, redeliveries: int = 3) -> int:
    """Replay real events through the real engine and count duplicate actions.

    Each case is delivered, redelivered several times (as a flaky webhook
    producer would), dispatched twice, and then hit with a late authorisation.
    The returned number is read back out of the store, not asserted.
    """
    now = datetime(2026, 8, 21, tzinfo=UTC)
    store = RecoveryStore()
    service = RecoveryService(
        store,
        BankHealthMonitor(min_samples=10_000),
        FakeRazorpayGateway(),
        settings=Settings(),
    )
    for index in range(cases):
        logical_key = f"bench_inv_{index}"
        failed = PaymentEvent(
            event_id=f"bench_evt_{index}",
            event_type=EventType.PAYMENT_FAILED,
            logical_key=logical_key,
            occurred_at=now,
            amount_paise=49900,
            payment_id=f"bench_pay_{index}",
            invoice_id=logical_key,
            issuer="hdfc",
            rail=PaymentRail.CARD,
            failure_code="CARD_EXPIRED",
        )
        for _ in range(redeliveries):
            service.ingest(failed)
        service.dispatch_due_actions(now)
        service.dispatch_due_actions(now + timedelta(seconds=1))
        service.ingest(
            PaymentEvent(
                event_id=f"bench_auth_{index}",
                event_type=EventType.PAYMENT_AUTHORIZED,
                logical_key=logical_key,
                occurred_at=now + timedelta(minutes=1),
                payment_id=f"bench_pay_{index}",
                invoice_id=logical_key,
                issuer="hdfc",
                rail=PaymentRail.CARD,
            )
        )
    duplicates = store.duplicate_action_count()
    store.close()
    return duplicates


# ------------------------------------------------------------------- runner


def run_benchmark(
    cases: list[SyntheticCase] | None = None,
    *,
    trials: int = DEFAULT_TRIALS,
    count: int = 500,
    seed: int = DEFAULT_SEED,
    measure_duplicates: bool = True,
) -> BenchmarkReport:
    headline_cases = cases if cases is not None else generate_cases(count, seed)
    headline = _score(headline_cases)

    incremental_samples: list[float] = []
    reduction_samples: list[float] = []
    if cases is None and trials > 1:
        for offset in range(trials):
            trial = _score(generate_cases(count, seed + offset))
            incremental_samples.append(float(trial.incremental_net))
            reduction_samples.append(trial.contact_reduction)

    return BenchmarkReport(
        cases=len(headline_cases),
        railpulse_net_recovered_paise=headline.railpulse_net,
        baseline_net_recovered_paise=headline.baseline_net,
        incremental_net_recovered_paise=headline.incremental_net,
        railpulse_contacts=headline.railpulse_contacts,
        baseline_contacts=headline.baseline_contacts,
        contact_reduction_rate=headline.contact_reduction,
        duplicate_recovery_actions=measure_duplicate_actions() if measure_duplicates else 0,
        incremental_interval=Interval.from_samples(incremental_samples) if incremental_samples else None,
        contact_reduction_interval=Interval.from_samples(reduction_samples) if reduction_samples else None,
    )


if __name__ == "__main__":
    print(run_benchmark().as_text())
