"""Central policy and runtime settings.

Every number that a merchant would plausibly want to tune lives here rather
than being buried in the orchestration code. Defaults are the demo values
documented in the README; production deployments override them by environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration resolved once at startup."""

    database_path: str = "railpulse.db"

    # --- issuer/rail health -------------------------------------------------
    health_window: timedelta = timedelta(hours=6)
    health_min_samples: int = 30
    health_degraded_success_rate: float = 0.65
    health_max_tracked_keys: int = 2_000

    # --- recovery policy ----------------------------------------------------
    cooldown: timedelta = timedelta(minutes=90)
    # How long a transient failure is allowed to sit in RETRY_SCHEDULED waiting
    # for the provider's own retry to succeed before RailPulse escalates to a
    # consented recovery link. Without this, transient cases never progressed.
    retry_escalation_after: timedelta = timedelta(hours=6)
    max_contacts_7d: int = 2
    contact_cooldown: timedelta = timedelta(hours=24)
    # Failed attempts on one case before RailPulse stops deciding and asks a
    # human. Without a cap, a flapping rail could cycle a case indefinitely.
    max_recovery_attempts: int = 4
    # How many times a failed payment-link cancellation may be re-attempted.
    # Cancelling is idempotent at the provider, so retrying is safe; the bound
    # exists so a link the provider will never cancel stops being retried and
    # starts being a human's problem.
    max_link_cancel_attempts: int = 4
    # Ceiling on how many failed cancels one dispatch tick will re-attempt, so
    # a backlog drains steadily instead of blocking the tick behind a few
    # hundred gateway calls.
    link_cancel_retry_batch: int = 25
    # Cases one dispatch tick will consider, per pass. An outage parks every
    # case for an issuer at once, so the backlog is largest exactly when the
    # tick most needs to finish; without a bound its cost tracks the backlog
    # rather than the work that is due.
    dispatch_batch_size: int = 500

    # --- release control ----------------------------------------------------
    # An outage parks every case for that issuer in COOLDOWN together. When the
    # rail recovers they all became due on the same dispatch tick and retried
    # in one synchronised wave -- into an issuer that had just come back up,
    # which is a good way to push it over again. Cases are now spread across a
    # window and released at a bounded rate per issuer and rail.
    release_jitter: timedelta = timedelta(minutes=15)
    max_releases_per_tick: int = 25

    # --- retention ----------------------------------------------------------
    # How long an idempotency claim is kept. It only has to outlast the
    # provider's redelivery schedule; nothing pruned this table before, so it
    # grew by one row per webhook for the life of the deployment.
    processed_event_retention: timedelta = timedelta(days=30)

    # --- transport ----------------------------------------------------------
    gateway_timeout_seconds: float = 10.0
    gateway_max_attempts: int = 3
    gateway_backoff_seconds: float = 0.5

    # --- api ----------------------------------------------------------------
    max_webhook_bytes: int = 1_048_576
    default_case_page_size: int = 100
    max_case_page_size: int = 500

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            database_path=os.getenv("RAILPULSE_DB", "railpulse.db"),
            health_window=timedelta(minutes=_int("RAILPULSE_HEALTH_WINDOW_MINUTES", 360)),
            health_min_samples=_int("RAILPULSE_HEALTH_MIN_SAMPLES", 30),
            health_degraded_success_rate=_float("RAILPULSE_HEALTH_DEGRADED_RATE", 0.65),
            health_max_tracked_keys=_int("RAILPULSE_HEALTH_MAX_KEYS", 2_000),
            cooldown=timedelta(minutes=_int("RAILPULSE_COOLDOWN_MINUTES", 90)),
            retry_escalation_after=timedelta(minutes=_int("RAILPULSE_RETRY_ESCALATION_MINUTES", 360)),
            max_contacts_7d=_int("RAILPULSE_MAX_CONTACTS_7D", 2),
            contact_cooldown=timedelta(hours=_int("RAILPULSE_CONTACT_COOLDOWN_HOURS", 24)),
            max_recovery_attempts=_int("RAILPULSE_MAX_RECOVERY_ATTEMPTS", 4),
            max_link_cancel_attempts=_int("RAILPULSE_MAX_LINK_CANCEL_ATTEMPTS", 4),
            link_cancel_retry_batch=_int("RAILPULSE_LINK_CANCEL_RETRY_BATCH", 25),
            dispatch_batch_size=_int("RAILPULSE_DISPATCH_BATCH_SIZE", 500),
            processed_event_retention=timedelta(
                days=_int("RAILPULSE_EVENT_RETENTION_DAYS", 30)
            ),
            release_jitter=timedelta(minutes=_int("RAILPULSE_RELEASE_JITTER_MINUTES", 15)),
            max_releases_per_tick=_int("RAILPULSE_MAX_RELEASES_PER_TICK", 25),
            gateway_timeout_seconds=_float("RAILPULSE_GATEWAY_TIMEOUT_SECONDS", 10.0),
            gateway_max_attempts=_int("RAILPULSE_GATEWAY_MAX_ATTEMPTS", 3),
            gateway_backoff_seconds=_float("RAILPULSE_GATEWAY_BACKOFF_SECONDS", 0.5),
            max_webhook_bytes=_int("RAILPULSE_MAX_WEBHOOK_BYTES", 1_048_576),
        )
