"""SQLite persistence with uniqueness constraints for webhook idempotency.

Concurrency notes
-----------------
FastAPI runs synchronous endpoints in a worker thread pool, so several webhooks
are processed at genuinely the same time.

For a file-backed database each thread owns its own connection, WAL mode lets
readers run while a writer holds the lock, and ``busy_timeout`` makes writers
queue instead of failing.

An in-memory database cannot work that way: it is private to the connection
that created it, so every thread must share one. That sharing was previously
guarded only around writes, in ``transaction()``. Reads ran unguarded on the
same connection, and two threads stepping statements at once crossed each
other's error codes -- a UNIQUE violation came back as a plain ``DatabaseError``
rather than ``IntegrityError``, escaped the handler in ``record_action`` and
killed the worker under a redelivery storm. Roughly one run in three failed.

Two changes close it. ``_SerialisedConnection`` puts every statement, read or
write, under the lock and materialises rows before releasing it. And
``record_action`` no longer infers "duplicate" from an exception class at all;
it uses ``ON CONFLICT DO NOTHING`` and checks a row count, because a dedupe
guarantee that depends on which exception subclass arrives is not a guarantee.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.models import (
    ActionRecord,
    CustomerAttentionBudget,
    FailureClass,
    PaymentRail,
    RecoveryCase,
    RecoveryCaseState,
)

BUSY_TIMEOUT_MS = 5_000


def _stamp(moment: datetime | None) -> str | None:
    """Serialise a datetime as a UTC ISO-8601 string.

    Every timestamp column is compared as TEXT, so the comparison is
    lexicographic rather than chronological. That is fine only if every value
    shares one offset. It did not: a deadline stamped +05:30 sorts *after* the
    same instant stamped +00:00 --

        '2026-08-21T17:30:00+05:30' <= '2026-08-21T12:00:00+00:00'  ->  False

    so ``due_cases`` skipped it and the case was stranded forever, while a
    naive datetime sorted early and then raised TypeError comparing
    naive-to-aware in dispatch. Normalising on write makes the string order and
    the chronological order the same thing.
    """
    if moment is None:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat()


class ConcurrentCaseUpdate(RuntimeError):
    """Raised when an optimistic-versioned case write loses a race."""


class _Result:
    """Rows materialised while the connection lock is still held.

    Returning a live cursor would defeat the point: the caller would step the
    statement after the lock was released, which is the race this exists to
    close.
    """

    __slots__ = ("_rows", "lastrowid", "rowcount")

    def __init__(self, rows: list[sqlite3.Row], rowcount: int, lastrowid: int | None) -> None:
        self._rows = rows
        self.rowcount = rowcount
        self.lastrowid = lastrowid

    def fetchall(self) -> list[sqlite3.Row]:
        return self._rows

    def fetchone(self) -> sqlite3.Row | None:
        return self._rows[0] if self._rows else None

    def __iter__(self) -> Iterator[sqlite3.Row]:
        return iter(self._rows)


class _SerialisedConnection:
    """Runs each statement to completion under a lock, then hands back rows."""

    __slots__ = ("_connection", "_lock")

    def __init__(self, connection: sqlite3.Connection, lock: threading.RLock) -> None:
        self._connection = connection
        self._lock = lock

    def execute(self, sql: str, parameters: tuple = ()) -> _Result:
        with self._lock:
            cursor = self._connection.execute(sql, parameters)
            try:
                rows = cursor.fetchall() if cursor.description else []
                return _Result(rows, cursor.rowcount, cursor.lastrowid)
            finally:
                cursor.close()

    def executescript(self, script: str) -> None:
        with self._lock:
            self._connection.executescript(script)

    def __getattr__(self, name: str) -> object:
        return getattr(self._connection, name)


class RecoveryStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._is_memory = self.path == ":memory:" or "mode=memory" in self.path
        self._local = threading.local()
        # An in-memory database is private to its connection, so a shared one is
        # kept alive for the lifetime of the store and guarded by a lock.
        self._shared: sqlite3.Connection | None = None
        self._memory_lock = threading.RLock() if self._is_memory else None
        if self._is_memory:
            self._shared = self._new_connection()
        self._initialize()

    # ------------------------------------------------------------------ setup

    def _new_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            isolation_level=None,
            check_same_thread=False,
            timeout=BUSY_TIMEOUT_MS / 1000,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        connection.execute("PRAGMA foreign_keys = ON")
        if not self._is_memory:
            # WAL lets dashboard reads proceed while a webhook write is in flight.
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    @property
    def connection(self) -> sqlite3.Connection:
        if self._shared is not None:
            # The in-memory database is one connection shared by every thread.
            # sqlite3 tolerates that only if calls are serialised: two threads
            # stepping statements on one connection cross each other's error
            # codes, which surfaced as a UNIQUE violation arriving as a plain
            # DatabaseError and escaping the IntegrityError handler in
            # record_action. transaction() locked writes but reads ran
            # unguarded, so the wrapper below covers every statement.
            return _SerialisedConnection(self._shared, self._memory_lock)  # type: ignore[arg-type]
        existing = getattr(self._local, "connection", None)
        if existing is None:
            existing = self._new_connection()
            self._local.connection = existing
        return existing

    def close(self) -> None:
        for connection in (self._shared, getattr(self._local, "connection", None)):
            if connection is not None:
                connection.close()
        self._shared = None
        self._local = threading.local()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS processed_events (
              event_id TEXT PRIMARY KEY,
              processed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS recovery_cases (
              id TEXT PRIMARY KEY,
              logical_key TEXT UNIQUE NOT NULL,
              payload TEXT NOT NULL,
              version INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS recovery_actions (
              id TEXT PRIMARY KEY,
              case_id TEXT NOT NULL,
              action_type TEXT NOT NULL,
              action_key TEXT UNIQUE NOT NULL,
              status TEXT NOT NULL,
              metadata TEXT NOT NULL,
              created_at TEXT NOT NULL,
              attempts INTEGER NOT NULL DEFAULT 1,
              FOREIGN KEY(case_id) REFERENCES recovery_cases(id)
            );
            CREATE TABLE IF NOT EXISTS health_observations (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              issuer TEXT NOT NULL,
              rail TEXT NOT NULL,
              succeeded INTEGER NOT NULL,
              observed_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_actions_case ON recovery_actions(case_id);
            CREATE INDEX IF NOT EXISTS idx_health_observed_at ON health_observations(observed_at);
            """
        )
        self._migrate_case_columns()
        self._migrate_action_columns()

    def _migrate_action_columns(self) -> None:
        existing = {row["name"] for row in self.connection.execute("PRAGMA table_info(recovery_actions)")}
        if "attempts" not in existing:
            self.connection.execute(
                "ALTER TABLE recovery_actions ADD COLUMN attempts INTEGER NOT NULL DEFAULT 1"
            )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_actions_sweep "
            "ON recovery_actions(action_type, status)"
        )

    def _migrate_case_columns(self) -> None:
        """Promote hot query fields out of the JSON blob into indexed columns.

        The payload stays the source of truth for the full aggregate; these
        columns exist purely so dispatch and metrics stop scanning and
        deserialising every row on every call.
        """
        existing = {row["name"] for row in self.connection.execute("PRAGMA table_info(recovery_cases)")}
        added = False
        for column, ddl in (
            ("state", "ALTER TABLE recovery_cases ADD COLUMN state TEXT"),
            ("next_action_at", "ALTER TABLE recovery_cases ADD COLUMN next_action_at TEXT"),
            ("customer_id", "ALTER TABLE recovery_cases ADD COLUMN customer_id TEXT"),
            ("amount_paise", "ALTER TABLE recovery_cases ADD COLUMN amount_paise INTEGER NOT NULL DEFAULT 0"),
            ("updated_at", "ALTER TABLE recovery_cases ADD COLUMN updated_at TEXT"),
        ):
            if column not in existing:
                self.connection.execute(ddl)
                added = True
        self.connection.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_cases_state ON recovery_cases(state);
            CREATE INDEX IF NOT EXISTS idx_cases_due ON recovery_cases(state, next_action_at);
            CREATE INDEX IF NOT EXISTS idx_cases_customer ON recovery_cases(customer_id);
            CREATE INDEX IF NOT EXISTS idx_cases_updated ON recovery_cases(updated_at);
            """
        )
        if added:
            self._backfill_projection()

    def _backfill_projection(self) -> None:
        """Populate the new columns for databases written by the old schema."""
        rows = self.connection.execute(
            "SELECT id, payload FROM recovery_cases WHERE state IS NULL"
        ).fetchall()
        for row in rows:
            self._write_projection(self._case_from_payload(row["payload"]))

    def _write_projection(self, case: RecoveryCase) -> None:
        self.connection.execute(
            """UPDATE recovery_cases
               SET state = ?, next_action_at = ?, amount_paise = ?, updated_at = ?
               WHERE id = ?""",
            (
                case.state.value,
                _stamp(case.next_action_at),
                case.amount_paise,
                _stamp(case.updated_at),
                case.id,
            ),
        )

    # ------------------------------------------------------------ transactions

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Re-entrant write transaction.

        Nesting is tolerated so a caller can compose store operations without
        having to know whether an outer transaction is already open.
        """
        depth = getattr(self._local, "depth", 0)
        if depth:
            self._local.depth = depth + 1
            try:
                yield
            finally:
                self._local.depth -= 1
            return

        lock = self._memory_lock
        if lock is not None:
            lock.acquire()
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            self._local.depth = 1
            try:
                yield
            except Exception:
                self.connection.execute("ROLLBACK")
                raise
            else:
                self.connection.execute("COMMIT")
            finally:
                self._local.depth = 0
        finally:
            if lock is not None:
                lock.release()

    # ------------------------------------------------------------ idempotency

    def claim_event(self, event_id: str, processed_at: datetime) -> bool:
        """Claim the right to process this delivery exactly once.

        Same reasoning as record_action: inferring "already seen" from an
        exception subclass is not a guarantee, because sqlite does not always
        classify a constraint violation as IntegrityError. ON CONFLICT moves
        the decision into SQL, where the answer is a row count.
        """
        result = self.connection.execute(
            """INSERT INTO processed_events(event_id, processed_at) VALUES (?, ?)
               ON CONFLICT(event_id) DO NOTHING""",
            (event_id, _stamp(processed_at)),
        )
        return result.rowcount == 1


    # ------------------------------------------------------------------ cases

    def get_case(self, logical_key: str) -> RecoveryCase | None:
        row = self.connection.execute(
            "SELECT payload FROM recovery_cases WHERE logical_key = ?", (logical_key,)
        ).fetchone()
        return self._case_from_payload(row["payload"]) if row else None

    def get_case_by_id(self, case_id: str) -> RecoveryCase | None:
        row = self.connection.execute(
            "SELECT payload FROM recovery_cases WHERE id = ?", (case_id,)
        ).fetchone()
        return self._case_from_payload(row["payload"]) if row else None

    def list_cases(self, *, limit: int | None = None, offset: int = 0) -> list[RecoveryCase]:
        sql = "SELECT payload FROM recovery_cases ORDER BY updated_at DESC, id"
        params: list[Any] = []
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params += [limit, offset]
        rows = self.connection.execute(sql, params).fetchall()
        return [self._case_from_payload(row["payload"]) for row in rows]

    def count_cases(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) AS c FROM recovery_cases").fetchone()["c"])

    def due_cases(
        self,
        states: Iterable[RecoveryCaseState],
        now: datetime,
        *,
        limit: int | None = None,
    ) -> list[RecoveryCase]:
        """Cases eligible for dispatch, selected by index rather than by scan.

        Bounded on purpose. The row filter was already indexed, but the result
        set was not capped and every row returned is JSON-decoded into a full
        aggregate, so one tick's cost grew with the size of the backlog rather
        than with the work due -- and the backlog is largest exactly when an
        issuer outage has parked thousands of cases at once, which is when the
        tick most needs to finish. A tick now takes a batch and the next tick
        takes the next one.

        The ordering has to be fair for that to be safe. Sorting by ``id`` (a
        uuid) after the due-time key meant a bounded read would serve the same
        arbitrary head every tick, so a case the contact budget kept declining
        could be starved indefinitely by nothing more than its uuid. Ordering
        by ``updated_at`` makes a skipped case age toward the front instead.
        """
        state_values = [state.value for state in states]
        if not state_values:
            return []
        placeholders = ",".join("?" for _ in state_values)
        clause = "" if limit is None else " LIMIT ?"
        parameters: list[object] = [*state_values, _stamp(now)]
        if limit is not None:
            parameters.append(limit)
        rows = self.connection.execute(
            f"""SELECT payload FROM recovery_cases
                WHERE state IN ({placeholders})
                  AND (next_action_at IS NULL OR next_action_at <= ?)
                ORDER BY next_action_at IS NULL DESC, next_action_at, updated_at, id{clause}""",
            parameters,
        ).fetchall()
        return [self._case_from_payload(row["payload"]) for row in rows]

    def insert_case(self, case: RecoveryCase) -> RecoveryCase:
        self.connection.execute(
            """INSERT INTO recovery_cases
               (id, logical_key, payload, version, state, next_action_at, amount_paise, updated_at,
                customer_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                case.id,
                case.logical_key,
                self._case_payload(case),
                case.version,
                case.state.value,
                _stamp(case.next_action_at),
                case.amount_paise,
                _stamp(case.updated_at),
                case.customer_id,
            ),
        )
        return case

    def save_case(self, case: RecoveryCase, expected_version: int) -> RecoveryCase:
        case.version = expected_version + 1
        result = self.connection.execute(
            """UPDATE recovery_cases
               SET payload = ?, version = ?, state = ?, next_action_at = ?, amount_paise = ?,
                   updated_at = ?, customer_id = ?
               WHERE id = ? AND version = ?""",
            (
                self._case_payload(case),
                case.version,
                case.state.value,
                _stamp(case.next_action_at),
                case.amount_paise,
                _stamp(case.updated_at),
                case.customer_id,
                case.id,
                expected_version,
            ),
        )
        if result.rowcount != 1:
            case.version = expected_version
            raise ConcurrentCaseUpdate(
                f"recovery case {case.id} changed underneath version {expected_version}"
            )
        return case

    def customer_contact_state(
        self, customer_id: str, exclude_case_id: str | None = None
    ) -> tuple[int, datetime | None, bool]:
        """Contacts, last contact and opt-out across ALL of one person's cases.

        The attention budget lives on RecoveryCase, so "two contacts per seven
        days" was enforced per invoice. Five failed subscriptions for one
        person produced five messages in the same second, each case correctly
        believing it had spent one of its two. Worse, a dispute set opted_out
        on that case alone, so someone who had just disputed a charge kept
        being chased about their other invoices -- the exact scenario the
        reversal handler's docstring describes as a regulatory risk.

        Returns (contacts, most_recent_contact, opted_out_anywhere).
        """
        rows = self.connection.execute(
            "SELECT payload FROM recovery_cases WHERE customer_id = ? AND id != ?",
            (customer_id, exclude_case_id or ""),
        ).fetchall()
        contacts = 0
        latest: datetime | None = None
        opted_out = False
        for row in rows:
            attention = self._case_from_payload(row["payload"]).attention
            opted_out = opted_out or attention.opted_out
            contacts += attention.contact_count_7d
            if attention.last_contact_at and (latest is None or attention.last_contact_at > latest):
                latest = attention.last_contact_at
        return contacts, latest, opted_out

    def opt_out_customer(self, customer_id: str, exclude_case_id: str | None = None) -> int:
        """Silence every OTHER case belonging to one person.

        A dispute is about the human, not the invoice they disputed.

        ``exclude_case_id`` matters: the caller is mid-flight on the disputed
        case and will save it itself. Bumping its version here makes the
        caller's own save lose the optimistic-version check against a write it
        performed, which surfaces as a spurious ConcurrentCaseUpdate.
        """
        rows = self.connection.execute(
            "SELECT id, payload, version FROM recovery_cases "
            "WHERE customer_id = ? AND id != ?",
            (customer_id, exclude_case_id or ""),
        ).fetchall()
        changed = 0
        for row in rows:
            case = self._case_from_payload(row["payload"])
            if case.attention.opted_out:
                continue
            case.attention.opted_out = True
            case.version = row["version"]
            self.save_case(case, case.version)
            changed += 1
        return changed

    def metrics(self) -> dict[str, int]:
        """Aggregate in SQL instead of loading and deserialising every case."""
        rows = self.connection.execute(
            "SELECT state, COUNT(*) AS count, COALESCE(SUM(amount_paise), 0) AS amount "
            "FROM recovery_cases GROUP BY state"
        ).fetchall()
        by_state = {row["state"]: (int(row["count"]), int(row["amount"])) for row in rows}

        def count(*states: RecoveryCaseState) -> int:
            return sum(by_state.get(state.value, (0, 0))[0] for state in states)

        def amount(*states: RecoveryCaseState) -> int:
            return sum(by_state.get(state.value, (0, 0))[1] for state in states)

        recovered = (RecoveryCaseState.RECOVERED_NATURAL, RecoveryCaseState.RECOVERED_BY_LINK)
        return {
            "total_cases": sum(value[0] for value in by_state.values()),
            "recovered_cases": count(*recovered),
            "recovered_amount_paise": amount(*recovered),
            "pending_consent": count(RecoveryCaseState.CONSENT_REQUIRED),
            "links_sent": count(RecoveryCaseState.LINK_SENT),
            "manual_review": count(RecoveryCaseState.MANUAL_REVIEW),
            "cooldown": count(RecoveryCaseState.COOLDOWN),
            "retry_scheduled": count(RecoveryCaseState.RETRY_SCHEDULED),
            "stopped": count(RecoveryCaseState.STOPPED),
            "awaiting_capture": count(RecoveryCaseState.AUTHORIZED_PENDING_CAPTURE),
            # Reversed cases leave the recovered states, so recovered_amount
            # already excludes them. They are reported anyway: a recovery
            # figure that quietly shrinks is worse than one that shows what
            # was clawed back.
            "reversed_cases": count(RecoveryCaseState.RECOVERY_REVERSED),
            "reversed_amount_paise": amount(RecoveryCaseState.RECOVERY_REVERSED),
        }

    # ---------------------------------------------------------------- actions

    def record_action(self, action: ActionRecord) -> bool:
        """Claim the right to perform this action exactly once.

        Deliberately not written as ``try/except IntegrityError``. sqlite does
        not always classify a UNIQUE violation as IntegrityError under
        concurrent access, and a dedupe guarantee that depends on which
        exception subclass arrives is not a guarantee. ON CONFLICT DO NOTHING
        moves the decision into SQL, where the answer is a row count.
        """
        result = self.connection.execute(
            """INSERT INTO recovery_actions
            (id, case_id, action_type, action_key, status, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(action_key) DO NOTHING""",
            (
                action.id,
                action.case_id,
                action.action_type,
                action.action_key,
                action.status,
                json.dumps(action.metadata, sort_keys=True),
                _stamp(action.created_at),
            ),
        )
        return result.rowcount == 1

    def claim_action(self, action: ActionRecord, *, max_attempts: int) -> bool:
        """Like ``record_action``, but a *failed* attempt can be claimed again.

        ``record_action`` conflates two different guarantees. One is real: a
        duplicate webhook must not cause a second payment link to be created.
        The other is an accident: because the key is also taken by an attempt
        that *failed*, the action could never be retried at all.

        For payment-link cancellation that accident has teeth. The link is a
        bearer URL. If the invoice is collected and the cancel call then times
        out, the link stays live and payable by anyone holding it, the case is
        flagged for a human, and every later attempt to cancel is refused by
        the key it already burned. The customer can be charged twice and the
        only thing standing between them and that is somebody reading a
        reconciliation queue.

        So the key guards *success*, not *attempts*: a row in ``failed`` is
        claimable until ``max_attempts``, and a row in ``started`` or
        ``succeeded`` is never claimable. The bound matters -- a cancel that
        fails because the link no longer exists would otherwise be retried
        forever -- and when it is reached the row stays failed and the case
        stays flagged, which is the honest end state.

        Only safe for actions that are idempotent at the provider: cancelling
        an already-cancelled link is a no-op, creating a second payment link
        is not. Link creation deliberately still uses ``record_action``.
        """
        result = self.connection.execute(
            """INSERT INTO recovery_actions
            (id, case_id, action_type, action_key, status, metadata, created_at, attempts)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(action_key) DO UPDATE SET
                status = excluded.status,
                attempts = recovery_actions.attempts + 1
            WHERE recovery_actions.status = 'failed'
              AND recovery_actions.attempts < ?""",
            (
                action.id,
                action.case_id,
                action.action_type,
                action.action_key,
                action.status,
                json.dumps(action.metadata, sort_keys=True),
                _stamp(action.created_at),
                max_attempts,
            ),
        )
        return result.rowcount == 1

    def failed_actions(
        self, action_type: str, *, max_attempts: int, limit: int = 100
    ) -> list[ActionRecord]:
        """Actions that failed and have retries left, oldest first.

        Without this the retry above only happens if some later webhook
        happens to arrive for the same case, which for a collected invoice is
        exactly the traffic that has stopped.
        """
        rows = self.connection.execute(
            """SELECT id, case_id, action_type, action_key, status, metadata, created_at, attempts
            FROM recovery_actions
            WHERE action_type = ? AND status = 'failed' AND attempts < ?
            ORDER BY created_at, id
            LIMIT ?""",
            (action_type, max_attempts, limit),
        ).fetchall()
        return [self._action_from_row(row) for row in rows]

    def get_action(self, action_key: str) -> ActionRecord | None:
        row = self.connection.execute(
            """SELECT id, case_id, action_type, action_key, status, metadata, created_at, attempts
            FROM recovery_actions WHERE action_key = ?""",
            (action_key,),
        ).fetchone()
        return self._action_from_row(row) if row else None

    def complete_action(
        self,
        action_key: str,
        *,
        status: str,
        metadata: dict[str, object],
        completed_at: datetime,
    ) -> bool:
        """Persist the outcome of an external side effect for auditability."""
        existing = self.get_action(action_key)
        if existing is None:
            return False
        merged_metadata = {
            **existing.metadata,
            **metadata,
            "completed_at": _stamp(completed_at),
        }
        result = self.connection.execute(
            "UPDATE recovery_actions SET status = ?, metadata = ? WHERE action_key = ?",
            (status, json.dumps(merged_metadata, sort_keys=True), action_key),
        )
        return result.rowcount == 1

    def list_actions(self, case_id: str) -> list[ActionRecord]:
        rows = self.connection.execute(
            """SELECT id, case_id, action_type, action_key, status, metadata, created_at, attempts
            FROM recovery_actions WHERE case_id = ? ORDER BY created_at, id""",
            (case_id,),
        ).fetchall()
        return [self._action_from_row(row) for row in rows]

    def action_count(self, case_id: str, action_type: str | None = None) -> int:
        if action_type:
            row = self.connection.execute(
                "SELECT COUNT(*) AS count FROM recovery_actions WHERE case_id = ? AND action_type = ?",
                (case_id, action_type),
            ).fetchone()
        else:
            row = self.connection.execute(
                "SELECT COUNT(*) AS count FROM recovery_actions WHERE case_id = ?", (case_id,)
            ).fetchone()
        return int(row["count"])

    def duplicate_action_count(self) -> int:
        """Successful create-intents beyond the first for any single case.

        The UNIQUE action-key constraint makes this structurally zero; the query
        exists so the benchmark can *measure* the guarantee rather than assert it.
        """
        row = self.connection.execute(
            """SELECT COALESCE(SUM(extra), 0) AS total FROM (
                 SELECT COUNT(*) - 1 AS extra FROM recovery_actions
                 WHERE action_type = 'payment_link.create' AND status = 'succeeded'
                 GROUP BY case_id
               ) WHERE extra > 0"""
        ).fetchone()
        return int(row["total"])

    # ----------------------------------------------------------------- health

    def record_observation(self, issuer: str, rail: str, succeeded: bool, at: datetime) -> None:
        self.connection.execute(
            "INSERT INTO health_observations(issuer, rail, succeeded, observed_at) VALUES (?, ?, ?, ?)",
            (issuer.lower(), rail, 1 if succeeded else 0, _stamp(at)),
        )

    def load_observations(self, since: datetime) -> list[tuple[str, str, bool, datetime]]:
        rows = self.connection.execute(
            "SELECT issuer, rail, succeeded, observed_at FROM health_observations "
            "WHERE observed_at >= ? ORDER BY observed_at",
            (_stamp(since),),
        ).fetchall()
        return [
            (row["issuer"], row["rail"], bool(row["succeeded"]), datetime.fromisoformat(row["observed_at"]))
            for row in rows
        ]

    def prune_processed_events(self, before: datetime) -> int:
        """Age out idempotency claims.

        Nothing pruned this table, so it grew without bound for the life of the
        deployment -- one row per webhook, forever. The retention window only
        needs to outlast the provider's redelivery schedule; past that, a
        replayed event is not a redelivery, it is a replay attack, and letting
        the claim expire is the wrong protection for that anyway.
        """
        result = self.connection.execute(
            "DELETE FROM processed_events WHERE processed_at < ?", (_stamp(before),)
        )
        return result.rowcount

    def prune_observations(self, before: datetime) -> int:
        result = self.connection.execute(
            "DELETE FROM health_observations WHERE observed_at < ?", (_stamp(before),)
        )
        return result.rowcount

    # --------------------------------------------------------- serialisation

    @staticmethod
    def _action_from_row(row: sqlite3.Row) -> ActionRecord:
        return ActionRecord(
            id=row["id"],
            case_id=row["case_id"],
            action_type=row["action_type"],
            action_key=row["action_key"],
            status=row["status"],
            metadata=json.loads(row["metadata"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            attempts=row["attempts"],
        )

    @staticmethod
    def _case_payload(case: RecoveryCase) -> str:
        payload = asdict(case)
        payload["state"] = case.state.value
        payload["failure_class"] = case.failure_class.value if case.failure_class else None
        payload["rail"] = case.rail.value
        payload["created_at"] = _stamp(case.created_at)
        payload["updated_at"] = _stamp(case.updated_at)
        payload["next_action_at"] = _stamp(case.next_action_at)
        payload["reversed_at"] = _stamp(case.reversed_at)
        if case.attention.last_contact_at:
            payload["attention"]["last_contact_at"] = _stamp(case.attention.last_contact_at)
        return json.dumps(payload, sort_keys=True)

    @staticmethod
    def _case_from_payload(raw: str) -> RecoveryCase:
        payload = json.loads(raw)
        attention_payload = payload.pop("attention")
        last_contact_at = attention_payload.get("last_contact_at")
        attention = CustomerAttentionBudget(
            contact_count_7d=attention_payload["contact_count_7d"],
            max_contacts_7d=attention_payload["max_contacts_7d"],
            opted_out=attention_payload["opted_out"],
            last_contact_at=datetime.fromisoformat(last_contact_at) if last_contact_at else None,
        )
        payload["attention"] = attention
        payload["state"] = RecoveryCaseState(payload["state"])
        raw_class = payload["failure_class"]
        payload["failure_class"] = FailureClass(raw_class) if raw_class else None
        payload["rail"] = PaymentRail(payload["rail"])
        payload["created_at"] = datetime.fromisoformat(payload["created_at"])
        payload["updated_at"] = datetime.fromisoformat(payload["updated_at"])
        if payload["next_action_at"]:
            payload["next_action_at"] = datetime.fromisoformat(payload["next_action_at"])
        if payload.get("reversed_at"):
            payload["reversed_at"] = datetime.fromisoformat(payload["reversed_at"])
        return RecoveryCase(**payload)
