from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from app.bank_health import BankHealthMonitor
from app.models import PaymentRail
from app.store import RecoveryStore

NOW = datetime(2026, 8, 21, tzinfo=UTC)


class BankHealthTests(unittest.TestCase):
    def test_snapshot_does_not_allocate_for_unknown_issuers(self) -> None:
        """Reading health used to create a window per issuer asked about."""
        monitor = BankHealthMonitor()
        for index in range(500):
            monitor.snapshot(f"bank{index}", PaymentRail.CARD, NOW)
        self.assertEqual(len(monitor._observations), 0)

    def test_tracked_keys_are_capped(self) -> None:
        monitor = BankHealthMonitor(max_tracked_keys=25)
        for index in range(200):
            monitor.observe(f"bank{index}", PaymentRail.CARD, False, NOW)
        self.assertLessEqual(len(monitor._observations), 25)

    def test_degraded_requires_the_minimum_sample_size(self) -> None:
        monitor = BankHealthMonitor(min_samples=10, degraded_success_rate=0.65)
        for _ in range(9):
            monitor.observe("hdfc", PaymentRail.CARD, False, NOW)
        self.assertFalse(monitor.snapshot("hdfc", PaymentRail.CARD, NOW).degraded)
        monitor.observe("hdfc", PaymentRail.CARD, False, NOW)
        snapshot = monitor.snapshot("hdfc", PaymentRail.CARD, NOW)
        self.assertTrue(snapshot.degraded)
        self.assertEqual(snapshot.attempts, 10)

    def test_observations_outside_the_window_expire(self) -> None:
        monitor = BankHealthMonitor(window=timedelta(hours=1), min_samples=2)
        monitor.observe("hdfc", PaymentRail.CARD, False, NOW - timedelta(hours=3))
        monitor.observe("hdfc", PaymentRail.CARD, False, NOW - timedelta(hours=2))
        snapshot = monitor.snapshot("hdfc", PaymentRail.CARD, NOW)
        self.assertEqual(snapshot.attempts, 0)
        self.assertFalse(snapshot.degraded)

    def test_health_survives_a_restart(self) -> None:
        """A redeploy mid-outage used to reset every issuer to healthy."""
        store = RecoveryStore()
        self.addCleanup(store.close)
        monitor = BankHealthMonitor(min_samples=5, sink=store)
        with store.transaction():
            for _ in range(6):
                monitor.observe("sbi", PaymentRail.UPI_AUTOPAY, False, NOW)
        self.assertTrue(monitor.snapshot("sbi", PaymentRail.UPI_AUTOPAY, NOW).degraded)

        # A fresh process reads the same durable observations back.
        restarted = BankHealthMonitor(min_samples=5, sink=store)
        self.assertFalse(restarted.snapshot("sbi", PaymentRail.UPI_AUTOPAY, NOW).degraded)
        restored = restarted.restore(NOW)
        self.assertEqual(restored, 6)
        self.assertTrue(restarted.snapshot("sbi", PaymentRail.UPI_AUTOPAY, NOW).degraded)

    def test_degraded_snapshots_lists_only_unhealthy_rails(self) -> None:
        monitor = BankHealthMonitor(min_samples=3)
        for _ in range(4):
            monitor.observe("sbi", PaymentRail.UPI_AUTOPAY, False, NOW)
        for _ in range(4):
            monitor.observe("hdfc", PaymentRail.CARD, True, NOW)
        degraded = monitor.degraded_snapshots(NOW)
        self.assertEqual([snapshot.issuer for snapshot in degraded], ["sbi"])
        self.assertIn("success_rate", degraded[0].as_dict())


if __name__ == "__main__":
    unittest.main()
