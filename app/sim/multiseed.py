"""Repeat the whole comparison across seeds, and report only what survives.

A single seed gives a point estimate. Quoted to the rupee it reads like a
measurement, but it is one draw from a stochastic world. The benchmark this
simulator replaced knew that -- its docstring warned that a single seed "looks
more precise than it is" and it ran forty of them -- and the replacement
quietly lost the property. This module puts it back.

Two decisions here are worth stating, because they change the numbers.

**The comparison is paired.** For a given seed every policy faces a
byte-identical world: the same customers, causes, amounts and outage windows,
in the same order. So the statistic reported is the per-seed *difference*, not
the gap between two independently-computed averages.

It is worth being precise about how much that buys, because the textbook claim
for paired designs is much larger than what happens here. Measured on this
world, pairing narrows the interval by roughly 15-30% (the baseline arm's
standard deviation runs 1.2-1.4x the standard deviation of the paired
difference). The reduction is modest because the policies do not respond to
world variation in the same direction: a seed that draws many insolvent
customers hurts static dunning far more than it hurts RailPulse, so a good
chunk of the difference's variance is real interaction rather than shared
world noise. Pairing is still the right design -- but the honest reason is the
next paragraph, not a variance argument.

**A win count sits next to the interval,** and it is only meaningful because
the comparison is paired: "won 12 of 12 worlds" is a sentence you can write
only when both policies faced the same twelve worlds. A mean difference of
+₹66k with 12/12 says something quite different from the same mean with 7/12,
and the reader should not have to reverse-engineer which one they are looking
at from the interval alone.

Nothing here decides what counts as a good result. It reports the interval and
the win count; if the interval spans zero, the renderer says so in words.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import sqrt

from app.sim.runner import Policy, Scorecard, run, rupees
from app.sim.world import World, WorldConfig

#: Twelve seeds: enough for an interval to carry information, few enough that
#: the whole report still finishes in seconds.
SEEDS: tuple[int, ...] = tuple(range(1, 13))

#: Two-sided 95% critical values of Student's t. The sample is small and its
#: standard deviation is estimated from that same sample, so the normal 1.96
#: understates every interval printed here -- at twelve seeds by about 12%.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def _t95(df: int) -> float:
    if df < 1:
        return float("inf")
    if df <= 30:
        return _T95[df]
    for limit, value in ((40, 2.021), (50, 2.009), (60, 2.000), (80, 1.990), (100, 1.984)):
        if df <= limit:
            return value
    return 1.960


@dataclass(frozen=True)
class Interval:
    """A mean and the half-width of its 95% confidence interval."""

    mean: float
    half_width: float
    n: int

    @property
    def low(self) -> float:
        return self.mean - self.half_width

    @property
    def high(self) -> float:
        return self.mean + self.half_width

    @property
    def excludes_zero(self) -> bool:
        """True when the sign of the effect is resolved at this sample size."""
        return self.n > 1 and (self.low > 0 or self.high < 0)


def interval(values: Sequence[float]) -> Interval:
    """Mean and 95% CI half-width. One observation has no interval, and says so."""
    n = len(values)
    if n == 0:
        return Interval(0.0, float("inf"), 0)
    mean = sum(values) / n
    if n == 1:
        return Interval(mean, float("inf"), 1)
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    return Interval(mean, _t95(n - 1) * sqrt(variance / n), n)


@dataclass(frozen=True)
class Series:
    """One policy's scorecards across the seed sweep, in seed order."""

    policy: str
    seeds: tuple[int, ...]
    cards: tuple[Scorecard, ...]

    def net(self) -> list[float]:
        return [float(card.net_paise) for card in self.cards]

    def net_interval(self) -> Interval:
        return interval(self.net())

    def rate_interval(self) -> Interval:
        return interval([card.recovery_rate for card in self.cards])

    def opt_out_interval(self) -> Interval:
        return interval([float(card.opt_outs) for card in self.cards])

    def attempts_per_recovery(self) -> Interval:
        return interval([card.attempts_per_recovery for card in self.cards])

    def contacts_per_recovery(self) -> Interval:
        return interval([card.contacts_per_recovery for card in self.cards])


@dataclass(frozen=True)
class Paired:
    """Per-seed differences between two policies on identical worlds."""

    challenger: str
    baseline: str
    deltas: tuple[float, ...]

    @property
    def interval(self) -> Interval:
        return interval(self.deltas)

    @property
    def wins(self) -> int:
        return sum(1 for delta in self.deltas if delta > 0)

    @property
    def seeds(self) -> int:
        return len(self.deltas)


def sweep(
    config: WorldConfig,
    factories: Sequence[Callable[[], Policy]],
    *,
    seeds: Sequence[int] = SEEDS,
    cases: int = 500,
) -> list[Series]:
    """Run every policy on every seed. Factories, not instances: a policy may
    carry state across a run, and reusing one across seeds would leak."""
    series: list[Series] = []
    for factory in factories:
        cards = tuple(
            run(World(config, seed=seed, cases=cases), factory()) for seed in seeds
        )
        series.append(Series(cards[0].policy, tuple(seeds), cards))
    return series


def paired(challenger: Series, baseline: Series) -> Paired:
    if challenger.seeds != baseline.seeds:
        raise ValueError("a paired comparison requires the same seeds on both sides")
    return Paired(
        challenger.policy,
        baseline.policy,
        tuple(
            a - b for a, b in zip(challenger.net(), baseline.net(), strict=True)
        ),
    )


def best_baseline(series: Sequence[Series]) -> Series:
    """The baseline to beat is the strongest one, not the most convenient."""
    return max(series, key=lambda s: s.net_interval().mean)


def _signed(paise: float) -> str:
    return f"{'+' if paise >= 0 else '-'}{rupees(abs(paise))}"


def render_intervals(series: Sequence[Series]) -> str:
    """Per-policy means with intervals. Read the paired table below it for the
    comparison -- these intervals overlap heavily and that is expected."""
    header = (
        f"{'policy':<26}{'mean net':>14}{'95% CI':>28}{'rec. rate':>16}"
        f"{'att/rec':>10}{'cont/rec':>10}{'opt-outs':>10}"
    )
    lines = [header, "-" * len(header)]
    for item in series:
        net = item.net_interval()
        rate = item.rate_interval()
        band = f"{rupees(net.low)} to {rupees(net.high)}"
        lines.append(
            f"{item.policy:<26}{rupees(net.mean):>14}{band:>28}"
            f"{f'{rate.mean:.1%} ± {rate.half_width:.1%}':>16}"
            f"{item.attempts_per_recovery().mean:>10.2f}"
            f"{item.contacts_per_recovery().mean:>10.2f}"
            f"{item.opt_out_interval().mean:>10.1f}"
        )
    return "\n".join(lines)


def render_paired(comparisons: Sequence[Paired]) -> str:
    """The table that actually supports the claim."""
    header = (
        f"{'paired difference in net recovered':<52}{'mean Δ':>13}"
        f"{'95% CI':>30}{'seeds won':>12}"
    )
    lines = [header, "-" * len(header)]
    unresolved: list[Paired] = []
    for item in comparisons:
        band = item.interval
        label = f"{item.challenger} vs {item.baseline}"
        span = f"{_signed(band.low)} to {_signed(band.high)}"
        lines.append(
            f"{label:<52}{_signed(band.mean):>13}{span:>30}"
            f"{f'{item.wins}/{item.seeds}':>12}"
        )
        if not band.excludes_zero:
            unresolved.append(item)
    if unresolved:
        lines.append("")
        for item in unresolved:
            lines.append(
                f"NOT RESOLVED: the interval for {item.challenger} vs {item.baseline} "
                "spans zero. At this sample size the sign of that difference is "
                "not established; do not quote it as a win."
            )
    return "\n".join(lines)
