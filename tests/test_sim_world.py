"""Tests that protect the one property everything else rests on: no policy
can see, or manufacture, the world's recovery function."""

from __future__ import annotations

import unittest
from pathlib import Path

from app.sim.policies import (
    BASELINES,
    DoNothingPolicy,
    RetryThriceBackoffPolicy,
    StaticDunningPolicy,
)
from app.sim.runner import Policy, run
from app.sim.world import (
    ActionKind,
    Cause,
    Decision,
    Observation,
    OutcomeKind,
    World,
    WorldConfig,
)

SEED = 7
CASES = 300


class ContractTests(unittest.TestCase):
    def test_observation_exposes_no_ground_truth(self) -> None:
        """The observable surface must not leak latent fields."""
        leaked = {
            "cause",
            "solvent_from",
            "outage_until",
            "technical_clear_at",
            "dnh_same_ok",
            "dnh_other_ok",
            "consent_would_succeed",
        }
        fields = set(Observation.__dataclass_fields__)
        self.assertEqual(fields & leaked, set(), "Observation leaks ground truth")

    def test_policy_modules_never_touch_latent_state(self) -> None:
        """Structural check, not a convention: policies cannot read internals."""
        source = Path("app/sim/policies.py").read_text(encoding="utf-8")
        for forbidden in ("_latent", "_Latent", "_would_succeed", "ledger(", "Cause."):
            self.assertNotIn(forbidden, source, f"policy module references {forbidden}")

    def test_same_seed_produces_identical_world(self) -> None:
        """Two policies must face a byte-identical world."""
        left = World(WorldConfig(), seed=SEED, cases=CASES)
        right = World(WorldConfig(), seed=SEED, cases=CASES)
        self.assertEqual(left.payment_ids, right.payment_ids)
        for pid in left.payment_ids:
            self.assertEqual(left.ledger(pid)["cause"], right.ledger(pid)["cause"])
            self.assertEqual(left.ledger(pid)["amount_paise"], right.ledger(pid)["amount_paise"])

    def test_acting_beats_doing_nothing(self) -> None:
        """A sane world rewards a sane policy. If this fails, the world is broken."""
        idle = run(World(WorldConfig(), seed=SEED, cases=CASES), DoNothingPolicy())
        acting = run(World(WorldConfig(), seed=SEED, cases=CASES), RetryThriceBackoffPolicy())
        self.assertGreater(acting.recovered_cases, idle.recovered_cases)

    def test_recovery_does_not_depend_on_polling_cadence(self) -> None:
        """Zero-action policies must score the same however often they look.

        The old version of this test compared only daily against hourly, with
        a 25% band, and passed while the real leak sat elsewhere: ABANDON took
        a single hazard draw at the horizon with the block evaluated at its
        most favourable moment, and the scheduler silently forfeited a slow
        poller's final partial interval. Across cadences the "do nothing"
        anchor ranged 15.2% to 29.8% -- and abandon-on-sight, which takes no
        action and spends nothing, beat every polling policy. Every uplift
        figure in the README is quoted against that anchor.

        The spread that remains is RNG draw-order noise: different cadences
        consume the shared generator in a different order. A systematic edge
        would be much larger.
        """

        class Poller(DoNothingPolicy):
            def __init__(self, minutes: int) -> None:
                self.minutes = minutes
                self.name = f"poll-{minutes}m"

            def decide(self, observation: Observation) -> Decision:
                return Decision(ActionKind.WAIT, wake_after_minutes=self.minutes)

        class Abandon(Policy):
            name = "abandon-on-sight"

            def decide(self, observation: Observation) -> Decision:
                return Decision(ActionKind.ABANDON)

        rates = []
        for policy in (Poller(10080), Poller(1440), Poller(360), Poller(60), Abandon()):
            rates.append(run(World(WorldConfig(), seed=SEED, cases=CASES), policy).recovery_rate)

        spread = max(rates) - min(rates)
        self.assertLess(
            spread,
            0.08,
            f"doing nothing scored {min(rates):.1%}-{max(rates):.1%} depending only on "
            "how often the policy woke up",
        )


class MechanicsTests(unittest.TestCase):
    def test_method_switch_without_consent_is_refused(self) -> None:
        world = World(WorldConfig(), seed=SEED, cases=CASES)
        pid = world.payment_ids[0]
        observation = world.observe(pid, world.failed_at(pid))
        other = next(m for m in type(observation.method) if m is not observation.method)
        outcome = world.apply(
            pid,
            Decision(ActionKind.RETRY, method=other),
            world.failed_at(pid),
        )
        self.assertIs(outcome.kind, OutcomeKind.ATTEMPT_FAILED)
        self.assertIn("consent", outcome.detail)

    def test_dead_instrument_never_recovers_on_retry(self) -> None:
        """However many times you retry a closed account, it stays closed."""
        world = World(WorldConfig(), seed=SEED, cases=800)
        dead = [p for p in world.payment_ids if world.ledger(p)["cause"] == Cause.INSTRUMENT_DEAD.value]
        self.assertTrue(dead, "expected some dead-instrument cases")
        pid = dead[0]
        now = world.failed_at(pid)
        for _ in range(6):
            outcome = world.apply(pid, Decision(ActionKind.RETRY), now)
            self.assertIsNot(outcome.kind, OutcomeKind.RECOVERED)

    def test_over_contacting_causes_opt_out(self) -> None:
        world = World(WorldConfig(contact_tolerance=0), seed=SEED, cases=CASES)
        card = run(world, StaticDunningPolicy())
        self.assertGreater(card.opt_outs, 0, "contacting past tolerance should cost customers")

    def test_all_baselines_run_on_the_held_out_world(self) -> None:
        for cls in BASELINES:
            card = run(World(WorldConfig().shifted(), seed=SEED, cases=120), cls())
            self.assertEqual(card.cases, 120)
            self.assertGreaterEqual(card.recovered_cases, 0)


class ObservableHealthTests(unittest.TestCase):
    """``issuer_recent_failure_rate`` is the substitute for the outage flag.

    The world knows exactly when an issuer is down. ``Observation`` must not
    say so -- ``test_observation_exposes_no_ground_truth`` enforces that -- but
    a policy able to learn nothing at all about issuer health would be an
    unfair strawman, since the real system runs a health monitor over its own
    traffic. This field is that inference, and these tests are what keep it
    from being decoration: it must be zero before a policy has attempted
    anything, and it must move only in response to what that policy itself
    observed.
    """

    def test_it_is_zero_before_a_policy_has_attempted_anything(self) -> None:
        world = World(WorldConfig(), seed=SEED, cases=CASES)
        pid = world.payment_ids[0]
        self.assertEqual(
            world.observe(pid, world.failed_at(pid)).issuer_recent_failure_rate, 0.0
        )

    def test_it_reflects_the_policys_own_attempts_and_nothing_else(self) -> None:
        """Two policies on the same world see different rates, because the rate
        is a function of what each of them did -- not of the world."""
        seen: dict[str, list[float]] = {"retrying": [], "idle": []}

        class Recording(Policy):
            def __init__(self, key: str, retry: bool) -> None:
                self.name = key
                self.key = key
                self.retry = retry

            def decide(self, observation: Observation) -> Decision:
                seen[self.key].append(observation.issuer_recent_failure_rate)
                if self.retry:
                    return Decision(ActionKind.RETRY, wake_after_minutes=60)
                return Decision(ActionKind.WAIT, wake_after_minutes=60)

        run(World(WorldConfig(), seed=SEED, cases=CASES), Recording("retrying", True))
        run(World(WorldConfig(), seed=SEED, cases=CASES), Recording("idle", False))

        self.assertGreater(
            max(seen["retrying"]), 0.0, "a retrying policy never observed a failure rate"
        )
        self.assertEqual(
            max(seen["idle"]), 0.0, "a policy that attempted nothing was told about failures"
        )


class PolicyInterfaceTests(unittest.TestCase):
    def test_custom_policy_only_receives_observations(self) -> None:
        seen: list[object] = []

        class Recording(Policy):
            name = "recording"

            def decide(self, observation: Observation) -> Decision:
                seen.append(observation)
                return Decision(ActionKind.ABANDON)

        run(World(WorldConfig(), seed=SEED, cases=50), Recording())
        self.assertTrue(seen)
        self.assertTrue(all(isinstance(item, Observation) for item in seen))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
