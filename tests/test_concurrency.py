"""Regression tests for the concurrent-webhook failure.

The original store shared one SQLite connection across every request thread.
Concurrent ``BEGIN IMMEDIATE`` statements collided with "cannot start a
transaction within a transaction" and roughly two thirds of the events were
dropped on the floor. These tests pin the fixed behaviour.
"""

from __future__ import annotations

import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from app.bank_health import BankHealthMonitor
from app.config import Settings
from app.gateway import FakeRazorpayGateway
from app.models import EventType, PaymentEvent, PaymentRail, RecoveryCaseState
from app.service import RecoveryService
from app.store import RecoveryStore

NOW = datetime(2026, 8, 21, tzinfo=UTC)
WORKERS = 16
EVENTS = 48


def failure_event(index: int) -> PaymentEvent:
    return PaymentEvent(
        event_id=f"evt_concurrent_{index}",
        event_type=EventType.PAYMENT_FAILED,
        logical_key=f"inv_concurrent_{index}",
        occurred_at=NOW,
        amount_paise=49900,
        payment_id=f"pay_{index}",
        invoice_id=f"inv_concurrent_{index}",
        issuer="hdfc",
        rail=PaymentRail.CARD,
        failure_code="CARD_EXPIRED",
    )


class ConcurrentIngestTests(unittest.TestCase):
    def _service(self, store: RecoveryStore) -> RecoveryService:
        return RecoveryService(
            store, BankHealthMonitor(min_samples=10_000), FakeRazorpayGateway(), settings=Settings()
        )

    def test_every_concurrent_event_is_persisted_in_memory(self) -> None:
        store = RecoveryStore()
        self.addCleanup(store.close)
        service = self._service(store)
        errors: list[BaseException] = []
        # Distinct logical keys means no contention on a single case, so the
        # only property under test is that threads can write at all. The
        # barrier at least guarantees they are genuinely writing at the same
        # time rather than being scheduled one after another.
        barrier = threading.Barrier(WORKERS)

        def ingest(index: int) -> None:
            try:
                barrier.wait(timeout=10)
                service.ingest(failure_event(index))
            except BaseException as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(ingest, range(WORKERS)))

        self.assertEqual(errors, [], f"concurrent ingest raised: {errors[:3]}")
        self.assertEqual(store.count_cases(), WORKERS)

    def test_every_concurrent_event_is_persisted_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = RecoveryStore(Path(directory) / "concurrent.db")
            self.addCleanup(store.close)
            service = self._service(store)
            errors: list[BaseException] = []
            barrier = threading.Barrier(WORKERS)

            def ingest(index: int) -> None:
                try:
                    barrier.wait(timeout=10)
                    service.ingest(failure_event(index))
                except BaseException as exc:
                    errors.append(exc)

            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                list(pool.map(ingest, range(WORKERS)))

            self.assertEqual(errors, [], f"concurrent ingest raised: {errors[:3]}")
            self.assertEqual(store.count_cases(), WORKERS)

    def test_redelivery_storm_creates_exactly_one_case_and_one_link(self) -> None:
        """The same event delivered from many threads must act exactly once.

        The barriers are load-bearing. Without them this passed with
        ``max_workers=1``: ThreadPoolExecutor is free to run fast tasks
        nose-to-tail, and ``sum(processed_flags) == 1`` holds trivially under
        serial execution, so the test proved the code correct *sequentially* --
        not what its name claims. Every worker now blocks until all have
        arrived, forcing real contention on ``claim_event``'s primary key and
        on the unique action key.
        """
        store = RecoveryStore()
        self.addCleanup(store.close)
        service = self._service(store)
        event = failure_event(0)
        processed_flags: list[bool] = []
        lock = threading.Lock()
        ingest_barrier = threading.Barrier(WORKERS)

        def ingest(_: int) -> None:
            ingest_barrier.wait(timeout=10)
            _, processed = service.ingest(event)
            with lock:
                processed_flags.append(processed)

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(ingest, range(WORKERS)))

        self.assertEqual(sum(processed_flags), 1, "exactly one delivery should be processed")
        self.assertEqual(store.count_cases(), 1)

        case = store.get_case(event.logical_key)
        dispatch_barrier = threading.Barrier(WORKERS)

        def dispatch(_: int) -> None:
            dispatch_barrier.wait(timeout=10)
            service.dispatch_due_actions(NOW)

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(dispatch, range(WORKERS)))

        self.assertEqual(store.action_count(case.id, "payment_link.create"), 1)
        self.assertEqual(store.duplicate_action_count(), 0)
        self.assertEqual(store.get_case_by_id(case.id).state, RecoveryCaseState.LINK_SENT)

    def test_slow_link_cancel_does_not_block_unrelated_webhooks(self) -> None:
        """The late-authorisation cancel used to run inside the write lock.

        ``ingest`` held ``BEGIN IMMEDIATE`` across ``cancel_payment_link``, so a
        provider that took its full timeout stalled every other webhook behind
        it. The cancel now runs between two transactions instead of inside one.
        """
        release = threading.Event()
        entered = threading.Event()

        class HangingCancelGateway(FakeRazorpayGateway):
            def cancel_payment_link(self, link_id):
                entered.set()
                release.wait(timeout=10)
                return super().cancel_payment_link(link_id)

        with tempfile.TemporaryDirectory() as directory:
            store = RecoveryStore(Path(directory) / "latency.db")
            self.addCleanup(store.close)
            service = RecoveryService(
                store, BankHealthMonitor(min_samples=10_000), HangingCancelGateway(), settings=Settings()
            )
            service.ingest(failure_event(1))
            service.dispatch_due_actions(NOW)
            self.assertEqual(store.get_case("inv_concurrent_1").state, RecoveryCaseState.LINK_SENT)

            authorized = PaymentEvent(
                event_id="evt_late_auth",
                event_type=EventType.PAYMENT_AUTHORIZED,
                logical_key="inv_concurrent_1",
                occurred_at=NOW,
                payment_id="pay_1",
                issuer="hdfc",
                rail=PaymentRail.CARD,
            )
            worker = threading.Thread(target=service.ingest, args=(authorized,))
            worker.start()
            self.addCleanup(worker.join)
            self.assertTrue(entered.wait(timeout=5), "cancel call never started")

            # While the provider call is in flight an unrelated webhook must
            # still commit. Under the old code this waited on the write lock
            # until the provider returned or the busy timeout expired.
            try:
                service.ingest(failure_event(2))
                self.assertEqual(
                    store.get_case("inv_concurrent_2").state, RecoveryCaseState.CONSENT_REQUIRED
                )
            finally:
                release.set()
            worker.join(timeout=10)
            self.assertEqual(store.get_case("inv_concurrent_1").payment_link_status, "cancelled")


if __name__ == "__main__":
    unittest.main()
