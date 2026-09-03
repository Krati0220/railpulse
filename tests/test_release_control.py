"""The thundering herd on outage recovery.

An outage parks every case for an issuer in COOLDOWN together. RailPulse
already handled the obvious edge -- a case whose cooldown expires while the
rail is *still* degraded is re-cooled rather than released, and that state is
persisted so a restart mid-outage does not dump a wave.

The herd simply moved next door. When the rail genuinely recovered, every
parked case for that issuer became due on the same dispatch tick and was given
the same ``next_action_at``, so they all retried in one synchronised burst --
into an issuer that had just come back up. Defending one edge and leaving the
adjacent one open is the more interesting half of this bug.
"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from app.bank_health import BankHealthMonitor
from app.config import Settings
from app.gateway import FakeRazorpayGateway
from app.models import EventType, PaymentEvent, PaymentRail, RecoveryCaseState
from app.service import RecoveryService
from app.store import RecoveryStore

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
ISSUER = "hdfc"
PARKED = 120


def _failure(index: int, at: datetime) -> PaymentEvent:
    return PaymentEvent(
        event_id=f"evt_{index}",
        event_type=EventType.PAYMENT_FAILED,
        logical_key=f"inv_{index}",
        occurred_at=at,
        amount_paise=49_900,
        payment_id=f"pay_{index}",
        issuer=ISSUER,
        rail=PaymentRail.CARD,
        failure_code="ISSUER_UNAVAILABLE",
    )


class ReleaseControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings()
        self.store = RecoveryStore()
        self.addCleanup(self.store.close)
        self.health = BankHealthMonitor(min_samples=5)
        self.service = RecoveryService(
            self.store, self.health, FakeRazorpayGateway(), settings=self.settings
        )
        # An outage: every case fails onto the same issuer and parks in cooldown.
        for index in range(PARKED):
            self.service.ingest(_failure(index, NOW))

    def _cooled(self) -> list:
        return [
            case
            for case in self.store.due_cases([RecoveryCaseState.COOLDOWN], NOW + timedelta(days=1))
            if case.issuer == ISSUER
        ]

    def _recover_the_rail(self, at: datetime) -> None:
        # The outage put PARKED failures into a six-hour health window, so a
        # handful of successes is not enough to clear it -- the rail has to
        # actually process healthy traffic again before it reads as recovered.
        for _ in range(PARKED * 4):
            self.health.observe(ISSUER, PaymentRail.CARD, True, at)

    def test_cases_park_in_cooldown_during_the_outage(self) -> None:
        self.assertGreater(len(self._cooled()), 0, "an outage should park cases in cooldown")

    def test_recovered_rail_releases_at_a_bounded_rate(self) -> None:
        """The core assertion: one tick must not release the whole backlog."""
        due = NOW + self.settings.cooldown + timedelta(minutes=1)
        self._recover_the_rail(due)
        parked_before = len(self._cooled())
        self.assertGreater(parked_before, self.settings.max_releases_per_tick)

        released = [
            case
            for case in self.service.dispatch_due_actions(due)
            if case.state is RecoveryCaseState.RETRY_SCHEDULED
        ]
        self.assertLessEqual(
            len(released),
            self.settings.max_releases_per_tick,
            "a recovered rail dumped its whole backlog on one tick",
        )
        self.assertGreater(len(released), 0, "a healthy rail should release something")

    def test_released_cases_do_not_share_one_retry_instant(self) -> None:
        """Rate limiting alone is not enough: whatever is released still has
        to be spread, or each tick fires its own smaller synchronised burst."""
        due = NOW + self.settings.cooldown + timedelta(minutes=1)
        self._recover_the_rail(due)
        released = [
            case
            for case in self.service.dispatch_due_actions(due)
            if case.state is RecoveryCaseState.RETRY_SCHEDULED
        ]
        instants = {case.next_action_at for case in released}
        self.assertGreater(
            len(instants), 1, "every released case was scheduled for the same instant"
        )
        span = max(instants) - min(instants)  # type: ignore[type-var]
        self.assertGreater(span, timedelta(0))
        self.assertLessEqual(span, self.settings.release_jitter)

    def test_backlog_drains_over_successive_ticks(self) -> None:
        """Held cases must not be stranded -- a later tick has to take them."""
        due = NOW + self.settings.cooldown + timedelta(minutes=1)
        self._recover_the_rail(due)
        total_released = 0
        moment = due
        for _ in range(8):
            self._recover_the_rail(moment)
            total_released += sum(
                1
                for case in self.service.dispatch_due_actions(moment)
                if case.state is RecoveryCaseState.RETRY_SCHEDULED
            )
            moment += self.settings.release_jitter + timedelta(minutes=1)
        self.assertGreater(
            total_released,
            self.settings.max_releases_per_tick,
            "rate limiting stranded the backlog instead of draining it",
        )

    def test_jitter_is_stable_across_restarts(self) -> None:
        """Two services with separate stores must schedule the same invoice
        into the same slot, or a restart reshuffles every pending retry."""
        offsets = []
        for _ in range(2):
            store = RecoveryStore()
            self.addCleanup(store.close)
            service = RecoveryService(
                store, BankHealthMonitor(min_samples=5), FakeRazorpayGateway(), settings=self.settings
            )
            offsets.append(service._release_offset("inv_42"))
        self.assertEqual(offsets[0], offsets[1])
        self.assertLess(offsets[0], self.settings.release_jitter)

    def test_still_degraded_rail_is_never_released(self) -> None:
        """The original guard must survive the new one."""
        due = NOW + self.settings.cooldown + timedelta(minutes=1)
        for _ in range(40):
            self.health.observe(ISSUER, PaymentRail.CARD, False, due)
        released = [
            case
            for case in self.service.dispatch_due_actions(due)
            if case.state is RecoveryCaseState.RETRY_SCHEDULED
        ]
        self.assertEqual(released, [], "a degraded rail must not release anything")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
