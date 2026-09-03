from __future__ import annotations

import unittest
from datetime import UTC, datetime

from app.benchmark import generate_cases, measure_duplicate_actions, run_benchmark
from app.models import ActionRecord, RecoveryCase, RecoveryCaseState
from app.store import RecoveryStore


class BenchmarkTests(unittest.TestCase):
    def test_generation_is_deterministic_for_a_seed(self) -> None:
        self.assertEqual(generate_cases(50, seed=7), generate_cases(50, seed=7))
        self.assertNotEqual(generate_cases(50, seed=7), generate_cases(50, seed=8))

    def test_report_is_paired_and_reproducible(self) -> None:
        report = run_benchmark(generate_cases(200, seed=11), measure_duplicates=False)
        self.assertEqual(report.cases, 200)
        self.assertEqual(
            report.incremental_net_recovered_paise,
            report.railpulse_net_recovered_paise - report.baseline_net_recovered_paise,
        )
        self.assertGreater(report.baseline_contacts, report.railpulse_contacts)

    def test_confidence_interval_brackets_the_mean(self) -> None:
        report = run_benchmark(trials=8, count=200, measure_duplicates=False)
        interval = report.incremental_interval
        self.assertIsNotNone(interval)
        self.assertEqual(interval.trials, 8)
        self.assertLessEqual(interval.low, interval.mean)
        self.assertLessEqual(interval.mean, interval.high)
        self.assertIn("95% CI", report.as_text())

    def test_replaying_redeliveries_produces_no_duplicate_actions(self) -> None:
        """The engine's guarantee, measured by replaying real events."""
        self.assertEqual(measure_duplicate_actions(cases=10, redeliveries=4), 0)

    def test_the_duplicate_counter_can_actually_see_a_duplicate(self) -> None:
        """Proves the zero above is a measurement, not a constant.

        Asserting ``== 0`` alone cannot distinguish a real count from
        ``return 0`` -- the same mistake the function exists to rule out. So
        write two succeeded create-intents for one case directly and require
        the query to report the extra one.

        Worth noting what this exercise turned up: duplicate suppression is
        defended twice over, independently. Bypassing the UNIQUE action key
        alone still yields zero, because dispatch only ever selects
        CONSENT_REQUIRED cases and the case has already moved to LINK_SENT by
        then. Either layer alone would hold.
        """
        store = RecoveryStore()
        self.addCleanup(store.close)
        now = datetime(2026, 8, 21, tzinfo=UTC)
        case = RecoveryCase(
            id="case_dup",
            logical_key="inv_dup",
            state=RecoveryCaseState.CONSENT_REQUIRED,
            amount_paise=49900,
            updated_at=now,
        )
        with store.transaction():
            store.insert_case(case)
            for index in range(2):
                store.record_action(
                    ActionRecord(
                        id=f"act_{index}",
                        case_id=case.id,
                        action_type="payment_link.create",
                        action_key=f"unique_{index}",
                        status="succeeded",
                        metadata={},
                        created_at=now,
                    )
                )
        self.assertEqual(
            store.duplicate_action_count(),
            1,
            "the counter is blind if two create-intents on one case report zero",
        )


if __name__ == "__main__":
    unittest.main()
