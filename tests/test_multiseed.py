"""The statistics the report prints, and the claim the README makes.

Two different jobs in one file. The first half checks the interval arithmetic
against values worked out by hand, because a confidence interval that is
quietly wrong is worse than no interval at all -- it launders noise into
authority. The second half runs the actual sweep and asserts the ordering the
README claims, so that if a future change makes RailPulse stop beating the
baselines, CI says so instead of the README continuing to say otherwise.
"""

from __future__ import annotations

import unittest

from app.classifier import KeywordProvider, RootCauseClassifier
from app.sim.multiseed import (
    SEEDS,
    Interval,
    Paired,
    Series,
    interval,
    paired,
    render_intervals,
    render_paired,
    sweep,
)
from app.sim.policies import (
    DoNothingPolicy,
    RetryThriceBackoffPolicy,
    RetryThriceImmediatePolicy,
    StaticDunningPolicy,
)
from app.sim.railpulse_policy import RailPulsePolicy
from app.sim.runner import Policy
from app.sim.world import ActionKind, Decision, Observation, WorldConfig

#: Smaller than the report's 500 so the suite stays quick. The effect being
#: asserted survives it; if it ever stops surviving it, that is a finding, not
#: a reason to raise the number until the test passes again.
CASES = 250


class IntervalTests(unittest.TestCase):
    def test_matches_a_hand_computed_interval(self) -> None:
        """[1,2,3,4,5]: mean 3, sd 1.5811, se 0.7071, t(4)=2.776."""
        result = interval([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(result.mean, 3.0)
        self.assertAlmostEqual(result.half_width, 1.9630, places=3)
        self.assertAlmostEqual(result.low, 1.0370, places=3)
        self.assertAlmostEqual(result.high, 4.9630, places=3)

    def test_uses_t_not_the_normal_approximation(self) -> None:
        """At twelve samples 1.96 is about 10% too narrow. Using it would make
        every interval in the report look tighter than the data supports."""
        values = [float(i) for i in range(12)]
        widened = interval(values)
        normal = 1.96 * (sum((v - 5.5) ** 2 for v in values) / 11) ** 0.5 / 12**0.5
        self.assertGreater(widened.half_width, normal * 1.05)

    def test_zero_variance_gives_a_zero_width_interval(self) -> None:
        result = interval([4.0] * 8)
        self.assertEqual(result.mean, 4.0)
        self.assertEqual(result.half_width, 0.0)

    def test_a_single_observation_has_no_interval(self) -> None:
        """One seed cannot bound itself. It must not silently report ±0."""
        result = interval([7.0])
        self.assertEqual(result.mean, 7.0)
        self.assertEqual(result.half_width, float("inf"))
        self.assertFalse(result.excludes_zero)

    def test_empty_input_does_not_claim_a_result(self) -> None:
        self.assertFalse(interval([]).excludes_zero)

    def test_excludes_zero_is_symmetric(self) -> None:
        self.assertTrue(Interval(100.0, 10.0, 12).excludes_zero)
        self.assertTrue(Interval(-100.0, 10.0, 12).excludes_zero)
        self.assertFalse(Interval(5.0, 10.0, 12).excludes_zero)


class PairedTests(unittest.TestCase):
    def test_refuses_to_pair_different_seed_sets(self) -> None:
        left = Series("a", (1, 2), ())
        right = Series("b", (1, 3), ())
        with self.assertRaises(ValueError):
            paired(left, right)

    def test_wins_counts_strict_improvements_only(self) -> None:
        item = Paired("a", "b", (5.0, 0.0, -1.0, 3.0))
        self.assertEqual(item.wins, 2)
        self.assertEqual(item.seeds, 4)

    def test_render_flags_an_interval_that_spans_zero(self) -> None:
        """The renderer must not let an unresolved comparison read as a win."""
        spans = Paired("a", "b", (10.0, -12.0, 8.0, -9.0, 11.0, -7.0))
        self.assertFalse(spans.interval.excludes_zero)
        self.assertIn("NOT RESOLVED", render_paired([spans]))

    def test_render_stays_quiet_when_the_sign_is_resolved(self) -> None:
        clear = Paired("a", "b", (100.0, 105.0, 98.0, 102.0, 101.0, 99.0))
        self.assertTrue(clear.interval.excludes_zero)
        self.assertNotIn("NOT RESOLVED", render_paired([clear]))


class SweepTests(unittest.TestCase):
    def test_each_seed_gets_a_fresh_policy(self) -> None:
        """A policy carries state. Reusing one instance across seeds would let
        seed 1's memory decide seed 2's actions, and the sweep would measure
        the order the seeds happened to run in."""
        built: list[object] = []

        class Counting(Policy):
            name = "counting"

            def __init__(self) -> None:
                built.append(self)

            def decide(self, observation: Observation) -> Decision:
                return Decision(ActionKind.ABANDON)

        sweep(WorldConfig(), [Counting], seeds=(1, 2, 3), cases=20)
        self.assertEqual(len(built), 3)
        self.assertEqual(len({id(item) for item in built}), 3)

    def test_the_sweep_is_deterministic(self) -> None:
        left = sweep(WorldConfig(), [DoNothingPolicy], seeds=(1, 2, 3), cases=60)
        right = sweep(WorldConfig(), [DoNothingPolicy], seeds=(1, 2, 3), cases=60)
        self.assertEqual(left[0].net(), right[0].net())

    def test_seeds_actually_produce_different_worlds(self) -> None:
        """If every seed gave the same world the interval would be zero-width
        and the whole exercise would be theatre."""
        series = sweep(WorldConfig(), [DoNothingPolicy], seeds=SEEDS, cases=CASES)
        self.assertGreater(len(set(series[0].net())), 1)

    def test_render_intervals_names_every_policy(self) -> None:
        series = sweep(
            WorldConfig(), [DoNothingPolicy, RetryThriceBackoffPolicy], seeds=(1, 2), cases=40
        )
        text = render_intervals(series)
        self.assertIn("do-nothing", text)
        self.assertIn("retry-3x-backoff", text)


class ClaimTests(unittest.TestCase):
    """What the README is allowed to say.

    These are slow by the standards of the rest of the suite (a few seconds)
    and that is the correct trade: they are the only thing standing between a
    regression and a headline number that has quietly stopped being true.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cache: dict = {}
        factories = [
            DoNothingPolicy,
            RetryThriceImmediatePolicy,
            RetryThriceBackoffPolicy,
            StaticDunningPolicy,
            lambda: RailPulsePolicy(label="railpulse (no classifier)"),
            lambda: RailPulsePolicy(
                label="railpulse + classifier",
                # The keyword provider, never a live model: a test that makes
                # HTTP calls is not a test.
                normaliser=RootCauseClassifier(provider=KeywordProvider(), cache=cache),
            ),
        ]
        cls.default = sweep(WorldConfig(), factories, seeds=SEEDS, cases=CASES)
        cls.held_out = sweep(WorldConfig().shifted(), factories, seeds=SEEDS, cases=CASES)

    def _comparisons(self, series: list[Series]) -> list[Paired]:
        return [paired(series[-1], other) for other in series[:-1]]

    def test_railpulse_beats_every_baseline_on_the_default_world(self) -> None:
        for item in self._comparisons(self.default):
            with self.subTest(baseline=item.baseline):
                band = item.interval
                self.assertTrue(
                    band.excludes_zero and band.low > 0,
                    f"vs {item.baseline}: interval {band.low:,.0f} to {band.high:,.0f} "
                    "does not establish a win",
                )

    def test_railpulse_beats_every_baseline_on_the_held_out_world(self) -> None:
        """If the ordering only holds on the default world, the policy is
        fitted to that world's constants. The second parameterisation is a
        robustness check, not an independent test set -- but a result that
        survives it has at least cleared the lower bar."""
        for item in self._comparisons(self.held_out):
            with self.subTest(baseline=item.baseline):
                band = item.interval
                self.assertTrue(
                    band.excludes_zero and band.low > 0,
                    f"vs {item.baseline}: interval {band.low:,.0f} to {band.high:,.0f} "
                    "does not establish a win",
                )

    def test_the_classifier_is_what_earns_the_difference(self) -> None:
        """The ablation the whole Track-03 argument rests on. Without a usable
        failure code the engine refuses to guess and routes to a human, which
        is correct and also worth a great deal less money."""
        for label, series in (("default", self.default), ("second", self.held_out)):
            with self.subTest(world=label):
                ablation = paired(series[-1], series[-2])
                self.assertEqual(ablation.wins, ablation.seeds)
                self.assertGreater(ablation.interval.low, 0)

    def test_railpulse_never_burns_a_customer(self) -> None:
        """Static dunning buys its rupees with opt-outs. The guardrails mean
        RailPulse cannot, and net recovered does not price that in -- so this
        column is the argument the rupee column cannot make."""
        for label, series in (("default", self.default), ("second", self.held_out)):
            with self.subTest(world=label):
                by_name = {item.policy: item for item in series}
                self.assertEqual(by_name["railpulse + classifier"].opt_out_interval().mean, 0.0)
                self.assertGreater(by_name["static-dunning"].opt_out_interval().mean, 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
