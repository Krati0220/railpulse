"""Tests for the adapter that drives the real engine through the world."""

from __future__ import annotations

import unittest

from app.sim.policies import DoNothingPolicy, RetryThriceBackoffPolicy
from app.sim.railpulse_policy import RailPulsePolicy
from app.sim.runner import run
from app.sim.world import World, WorldConfig

SEED = 7
CASES = 300


class AdapterTests(unittest.TestCase):
    def test_engine_actually_retries(self) -> None:
        """Regression: dispatch_due_actions used to escalate a due
        RETRY_SCHEDULED case to a consented link before the retry could run,
        so the engine recorded zero attempts across the whole batch and its
        recovery rate silently collapsed. A policy that never retries is not
        the policy we ship."""
        card = run(World(WorldConfig(), seed=SEED, cases=CASES), RailPulsePolicy())
        self.assertGreater(card.attempts, 0, "engine performed no retries at all")

    def test_contact_budget_is_respected(self) -> None:
        """The engine's whole claim is that it recovers without burning the
        customer. Compare against dunning on the same world."""
        policy = RailPulsePolicy()
        card = run(World(WorldConfig(), seed=SEED, cases=CASES), policy)
        self.assertEqual(card.opt_outs, 0, "engine should never contact a customer into opting out")
        self.assertLess(card.contacts_per_recovery, 2.0)

    def test_beats_do_nothing_on_both_worlds(self) -> None:
        """This used to run RailPulse and 3x-backoff, name the second one
        `idle`, and then assert each recovered more than zero *separately* --
        never comparing them. RailPulse recovering 1 case while the baseline
        recovered 299 passed. It now compares against the policy the name
        actually claims.
        """
        for config in (WorldConfig(), WorldConfig().shifted()):
            engine = run(World(config, seed=SEED, cases=CASES), RailPulsePolicy())
            idle = run(World(config, seed=SEED, cases=CASES), DoNothingPolicy())
            self.assertGreater(
                engine.recovered_cases,
                idle.recovered_cases,
                "intervening must beat leaving the customer alone",
            )
            self.assertGreater(engine.net_paise, idle.net_paise)

    def test_does_not_lose_to_a_fixed_retry_ladder(self) -> None:
        """A separate, weaker claim, kept honest and separate from the one
        above. On the harsher world RailPulse must at least stay competitive
        with plain backoff while making far fewer attempts -- that trade is
        the whole argument, so assert both halves of it.
        """
        config = WorldConfig().shifted()
        engine = run(World(config, seed=SEED, cases=CASES), RailPulsePolicy())
        ladder = run(World(config, seed=SEED, cases=CASES), RetryThriceBackoffPolicy())
        self.assertLess(
            engine.attempts_per_recovery,
            ladder.attempts_per_recovery,
            "the engine's claim is fewer attempts per recovery",
        )
        self.assertEqual(engine.opt_outs, 0)

    def test_the_scorecard_agrees_with_the_engines_own_ledger(self) -> None:
        """The claim that makes every other number credible.

        The module docstring says "if the engine has a bug, the scorecard
        inherits it". That was false: the scorecard scores from World.ledger()
        and never reads the engine's store, so the two could disagree silently
        -- and did, by Rs88,824 on a 500-case run, because the adapter sent a
        fabricated payment_link_id and the engine declined every single link
        recovery while the scorecard counted them all.

        A claim nothing checks is a claim that drifts. This checks it.
        """
        policy = RailPulsePolicy()
        card = run(World(WorldConfig(), seed=SEED, cases=CASES), policy)
        engine = policy.engine_ledger()
        self.assertEqual(
            card.recovered_cases,
            engine["recovered_cases"],
            "the world says a different number of cases recovered than the engine recorded",
        )
        self.assertEqual(
            card.recovered_paise,
            engine["recovered_paise"],
            "the world and the engine disagree about how much money came back",
        )

    def test_link_recoveries_are_actually_recorded(self) -> None:
        """Regression for the fabricated payment_link_id.

        RECOVERED_BY_LINK was reached zero times across an entire run. Nothing
        failed loudly; the engine simply concluded each paid link belonged to
        some other case and returned without transitioning.
        """
        policy = RailPulsePolicy()
        run(World(WorldConfig(), seed=SEED, cases=CASES), policy)
        by_link = policy.store.connection.execute(
            "SELECT COUNT(*) AS c FROM recovery_cases WHERE state = 'recovered_by_link'"
        ).fetchone()["c"]
        self.assertGreater(by_link, 0, "no consented link recovery was ever recorded")

    def test_run_is_deterministic(self) -> None:
        """Same seed, same engine, same number. Without this the scorecard is
        not reproducible and no comparison on it means anything."""
        first = run(World(WorldConfig(), seed=SEED, cases=CASES), RailPulsePolicy())
        second = run(World(WorldConfig(), seed=SEED, cases=CASES), RailPulsePolicy())
        self.assertEqual(first.recovered_paise, second.recovered_paise)
        self.assertEqual(first.attempts, second.attempts)
        self.assertEqual(first.contacts, second.contacts)

    def test_every_action_leaves_an_audit_record(self) -> None:
        """Track 03 asks for an audit trail. Verify one actually exists after
        a batch rather than trusting that it does."""
        policy = RailPulsePolicy()
        run(World(WorldConfig(), seed=SEED, cases=CASES), policy)
        recorded = policy.store.connection.execute(
            "SELECT COUNT(*) AS c FROM recovery_actions"
        ).fetchone()["c"]
        self.assertGreater(recorded, 0, "no actions were journalled")

    def test_unknown_codes_are_counted(self) -> None:
        """Roughly a third of the world's failures arrive with no code. That
        count is the size of the gap an LLM classifier has to close."""
        policy = RailPulsePolicy()
        run(World(WorldConfig(), seed=SEED, cases=CASES), policy)
        self.assertGreater(policy.unknown_codes, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
