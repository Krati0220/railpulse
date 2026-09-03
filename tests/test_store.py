from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.models import ActionRecord, PaymentRail, RecoveryCase, RecoveryCaseState
from app.store import ConcurrentCaseUpdate, RecoveryStore

NOW = datetime(2026, 8, 21, tzinfo=UTC)

LEGACY_SCHEMA = """
CREATE TABLE processed_events (event_id TEXT PRIMARY KEY, processed_at TEXT NOT NULL);
CREATE TABLE recovery_cases (
  id TEXT PRIMARY KEY,
  logical_key TEXT UNIQUE NOT NULL,
  payload TEXT NOT NULL,
  version INTEGER NOT NULL
);
CREATE TABLE recovery_actions (
  id TEXT PRIMARY KEY, case_id TEXT NOT NULL, action_type TEXT NOT NULL,
  action_key TEXT UNIQUE NOT NULL, status TEXT NOT NULL, metadata TEXT NOT NULL,
  created_at TEXT NOT NULL
);
"""


def make_case(**overrides) -> RecoveryCase:
    defaults = {
        "id": "case_1",
        "logical_key": "inv_1",
        "amount_paise": 49900,
        "state": RecoveryCaseState.CONSENT_REQUIRED,
        "rail": PaymentRail.CARD,
        "created_at": NOW,
        "updated_at": NOW,
    }
    defaults.update(overrides)
    return RecoveryCase(**defaults)


class StoreProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = RecoveryStore()
        self.addCleanup(self.store.close)

    def test_due_cases_filters_by_state_and_deadline(self) -> None:
        with self.store.transaction():
            self.store.insert_case(make_case(id="c1", logical_key="k1"))
            self.store.insert_case(
                make_case(
                    id="c2",
                    logical_key="k2",
                    state=RecoveryCaseState.COOLDOWN,
                    next_action_at=NOW + timedelta(hours=2),
                )
            )
            self.store.insert_case(
                make_case(
                    id="c3",
                    logical_key="k3",
                    state=RecoveryCaseState.COOLDOWN,
                    next_action_at=NOW - timedelta(minutes=5),
                )
            )
            self.store.insert_case(
                make_case(id="c4", logical_key="k4", state=RecoveryCaseState.RECOVERED_BY_LINK)
            )

        due = self.store.due_cases([RecoveryCaseState.COOLDOWN], NOW)
        self.assertEqual([case.id for case in due], ["c3"])

        consent = self.store.due_cases([RecoveryCaseState.CONSENT_REQUIRED], NOW)
        self.assertEqual([case.id for case in consent], ["c1"])

    def test_metrics_match_a_full_scan(self) -> None:
        with self.store.transaction():
            self.store.insert_case(
                make_case(id="m1", logical_key="mk1", state=RecoveryCaseState.RECOVERED_NATURAL, amount_paise=1000)
            )
            self.store.insert_case(
                make_case(id="m2", logical_key="mk2", state=RecoveryCaseState.RECOVERED_BY_LINK, amount_paise=2500)
            )
            self.store.insert_case(
                make_case(id="m3", logical_key="mk3", state=RecoveryCaseState.MANUAL_REVIEW, amount_paise=700)
            )
        metrics = self.store.metrics()
        self.assertEqual(metrics["total_cases"], 3)
        self.assertEqual(metrics["recovered_cases"], 2)
        self.assertEqual(metrics["recovered_amount_paise"], 3500)
        self.assertEqual(metrics["manual_review"], 1)

        recovered = {RecoveryCaseState.RECOVERED_NATURAL, RecoveryCaseState.RECOVERED_BY_LINK}
        scanned = [case for case in self.store.list_cases() if case.state in recovered]
        self.assertEqual(metrics["recovered_amount_paise"], sum(case.amount_paise for case in scanned))

    def test_optimistic_version_conflict_raises_a_typed_error(self) -> None:
        case = make_case()
        with self.store.transaction():
            self.store.insert_case(case)
        with self.store.transaction():
            self.store.save_case(case, case.version)
        stale = make_case(version=0)
        with self.assertRaises(ConcurrentCaseUpdate), self.store.transaction():
            self.store.save_case(stale, 0)

    def test_nested_transactions_do_not_raise(self) -> None:
        with self.store.transaction():
            self.store.insert_case(make_case())
            with self.store.transaction():
                self.store.record_action(
                    ActionRecord(
                        id="act_1",
                        case_id="case_1",
                        action_type="payment_link.create",
                        action_key="k",
                        status="started",
                        metadata={},
                        created_at=NOW,
                    )
                )
        self.assertEqual(self.store.action_count("case_1"), 1)

    def test_list_cases_pagination(self) -> None:
        with self.store.transaction():
            for index in range(5):
                self.store.insert_case(
                    make_case(
                        id=f"p{index}",
                        logical_key=f"pk{index}",
                        updated_at=NOW + timedelta(minutes=index),
                    )
                )
        page = self.store.list_cases(limit=2)
        self.assertEqual(len(page), 2)
        self.assertEqual(page[0].id, "p4", "most recently updated first")
        self.assertEqual(self.store.count_cases(), 5)


class LegacyMigrationTests(unittest.TestCase):
    def test_an_existing_database_is_migrated_and_backfilled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.db"
            connection = sqlite3.connect(path)
            connection.executescript(LEGACY_SCHEMA)
            payload = {
                "id": "legacy_1",
                "logical_key": "legacy_key",
                "amount_paise": 12345,
                "state": "cooldown",
                "failure_class": "transient",
                "failure_code": "BANK_DOWN",
                "issuer": "sbi",
                "rail": "upi_autopay",
                "subscription_id": None,
                "invoice_id": "legacy_key",
                "original_payment_id": None,
                "payment_link_id": None,
                "payment_link_url": None,
                "payment_link_status": None,
                "outreach_preview": None,
                "next_action_at": (NOW - timedelta(minutes=1)).isoformat(),
                "stop_reason": None,
                "requires_manual_reconciliation": False,
                "version": 3,
                "attention": {
                    "contact_count_7d": 0,
                    "max_contacts_7d": 2,
                    "opted_out": False,
                    "last_contact_at": None,
                },
                "created_at": NOW.isoformat(),
                "updated_at": NOW.isoformat(),
            }
            connection.execute(
                "INSERT INTO recovery_cases(id, logical_key, payload, version) VALUES (?, ?, ?, ?)",
                ("legacy_1", "legacy_key", json.dumps(payload), 3),
            )
            connection.commit()
            connection.close()

            store = RecoveryStore(path)
            self.addCleanup(store.close)

            # The old row is readable, and the new indexed columns were filled
            # in so that dispatch can find it without a full scan.
            case = store.get_case_by_id("legacy_1")
            self.assertIsNotNone(case)
            self.assertEqual(case.state, RecoveryCaseState.COOLDOWN)
            self.assertEqual(case.attempt_count, 0, "new fields fall back to their defaults")

            due = store.due_cases([RecoveryCaseState.COOLDOWN], NOW)
            self.assertEqual([found.id for found in due], ["legacy_1"])
            self.assertEqual(store.metrics()["total_cases"], 1)


class HealthPersistenceTests(unittest.TestCase):
    def test_observations_round_trip_and_prune(self) -> None:
        store = RecoveryStore()
        self.addCleanup(store.close)
        with store.transaction():
            store.record_observation("hdfc", "card", False, NOW)
            store.record_observation("hdfc", "card", True, NOW - timedelta(days=2))
        recent = store.load_observations(NOW - timedelta(hours=6))
        self.assertEqual(len(recent), 1)
        with store.transaction():
            removed = store.prune_observations(NOW - timedelta(hours=6))
        self.assertEqual(removed, 1)
        self.assertEqual(len(store.load_observations(NOW - timedelta(days=30))), 1)


if __name__ == "__main__":
    unittest.main()
