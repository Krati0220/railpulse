"""The command the README tells a reader to run has to actually run.

Every number in the README is defended with "`python -m app.sim.report`
reproduces this". That promise is the whole basis on which a reader is asked
to believe any of it, and until now nothing tested it: `app/sim/report.py` sat
at 0% coverage, so a rename or a signature change anywhere it touches would
have broken the one command the project stakes its credibility on, silently,
until someone typed it.

The report is slow at full size (twelve seeds, two worlds, six policies), so
these tests shrink the sweep rather than skip it. What is being checked is
that the wiring holds end to end and that the output says what it claims --
not the figures themselves, which `tests/test_multiseed.py` owns.

Also covers config's environment parsing, because a malformed
RAILPULSE_MAX_RECOVERY_ATTEMPTS silently becoming something other than the
documented default is the kind of bug that only shows up as a policy behaving
oddly three layers away.
"""

from __future__ import annotations

import io
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from app.config import Settings
from app.sim import report


class ReportSmokeTests(unittest.TestCase):
    """Small enough to run in the suite, real enough to catch a break."""

    def _run(self) -> str:
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(report, "SEEDS", (1, 2)), mock.patch.object(
            report, "CASES", 40
        ), redirect_stdout(out), redirect_stderr(err):
            code = report.main([])
        self.assertEqual(code, 0)
        return out.getvalue()

    def setUp(self) -> None:
        # No key, so the keyword floor is used and nothing here makes an HTTP
        # call. A test that reaches the network is not a test.
        patcher = mock.patch.dict(
            os.environ, {"ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": ""}, clear=False
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.text = self._run()

    def test_it_runs_at_all(self) -> None:
        self.assertIn("RailPulse", self.text)

    def test_it_reports_both_worlds(self) -> None:
        self.assertIn("DEFAULT WORLD", self.text)
        self.assertIn("SECOND WORLD", self.text)

    def test_it_reports_every_policy_including_the_ablation_arm(self) -> None:
        for policy in (
            "do-nothing",
            "retry-3x-immediate",
            "retry-3x-backoff",
            "static-dunning",
            "railpulse (no classifier)",
            "railpulse + classifier",
        ):
            self.assertIn(policy, self.text, f"{policy} missing from the report")

    def test_it_prints_the_paired_comparison(self) -> None:
        self.assertIn("paired difference in net recovered", self.text)
        self.assertIn("seeds won", self.text)

    def test_it_says_loudly_when_there_is_no_model(self) -> None:
        """The failure this guards against is a demo quietly passing off a
        lookup table as AI. Without a key the report must say so in the header,
        not bury it."""
        self.assertIn("NO API KEY SET", self.text)
        self.assertIn("rule-based floor", self.text)

    def test_it_scores_the_classifier_on_unseen_phrasings(self) -> None:
        self.assertIn("phrasings absent from the simulator", self.text)

    def test_it_keeps_the_honesty_note(self) -> None:
        """The note is not decoration: it is the scope limit on every figure
        above it, and a report that dropped it would overclaim by omission."""
        self.assertIn("synthetic worlds, not merchant data", self.text)
        self.assertIn("not priced into net recovered", self.text)

    def test_it_is_deterministic(self) -> None:
        self.assertEqual(self.text, self._run())


class SettingsFromEnvTests(unittest.TestCase):
    """A malformed value must fall back to the documented default, not crash
    and not silently become zero."""

    def _settings(self, **env: str) -> Settings:
        with mock.patch.dict(os.environ, env, clear=True):
            return Settings.from_env()

    def test_defaults_apply_with_a_bare_environment(self) -> None:
        settings = self._settings()
        self.assertEqual(settings.max_recovery_attempts, 4)
        self.assertEqual(settings.max_contacts_7d, 2)
        self.assertEqual(settings.dispatch_batch_size, 500)
        self.assertEqual(settings.max_link_cancel_attempts, 4)

    def test_a_valid_override_is_taken(self) -> None:
        settings = self._settings(RAILPULSE_MAX_RECOVERY_ATTEMPTS="7")
        self.assertEqual(settings.max_recovery_attempts, 7)

    def test_garbage_falls_back_rather_than_crashing(self) -> None:
        """Booting with a typo'd env var must not take the process down, and
        must not quietly reinterpret the value as something else."""
        for bad in ("", "   ", "four", "4.5", "-"):
            with self.subTest(value=bad):
                settings = self._settings(RAILPULSE_MAX_RECOVERY_ATTEMPTS=bad)
                self.assertEqual(settings.max_recovery_attempts, 4)

    def test_a_bad_float_falls_back_too(self) -> None:
        settings = self._settings(RAILPULSE_HEALTH_DEGRADED_RATE="not-a-rate")
        self.assertEqual(
            settings.health_degraded_success_rate,
            Settings().health_degraded_success_rate,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
