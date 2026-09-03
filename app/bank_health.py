"""Small issuer/rail health monitor.

This is intentionally an observed-signal cache, not a claim to consume live
bank outage feeds. Production deployments would replace this with a bounded,
privacy-reviewed telemetry pipeline.

Two properties matter for correctness:

* ``snapshot`` is a read. It used to allocate a window for every issuer/rail
  pair it was asked about, so simply *querying* unknown issuers grew the
  process without bound.
* Outage detection survives a restart. Holding the window purely in memory
  meant a redeploy mid-outage reset every issuer to "healthy" and released a
  wave of retries into a degraded rail.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from app.models import PaymentRail

_EMPTY: deque[tuple[datetime, bool]] = deque()


@dataclass(frozen=True)
class HealthSnapshot:
    issuer: str
    rail: PaymentRail
    attempts: int
    successes: int
    success_rate: float
    degraded: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "issuer": self.issuer,
            "rail": self.rail.value,
            "attempts": self.attempts,
            "successes": self.successes,
            "success_rate": round(self.success_rate, 4),
            "degraded": self.degraded,
        }


class ObservationSink(Protocol):
    """The slice of the store the monitor needs, kept narrow on purpose."""

    def record_observation(self, issuer: str, rail: str, succeeded: bool, at: datetime) -> None: ...

    def load_observations(self, since: datetime) -> list[tuple[str, str, bool, datetime]]: ...

    def prune_observations(self, before: datetime) -> int: ...


class BankHealthMonitor:
    def __init__(
        self,
        *,
        window: timedelta = timedelta(hours=6),
        min_samples: int = 30,
        degraded_success_rate: float = 0.65,
        max_tracked_keys: int = 2_000,
        sink: ObservationSink | None = None,
    ) -> None:
        self.window = window
        self.min_samples = min_samples
        self.degraded_success_rate = degraded_success_rate
        self.max_tracked_keys = max_tracked_keys
        self.sink = sink
        # LRU-ordered so a long tail of rare issuers cannot grow without bound.
        self._observations: OrderedDict[tuple[str, PaymentRail], deque[tuple[datetime, bool]]] = OrderedDict()

    # ------------------------------------------------------------- lifecycle

    def restore(self, now: datetime) -> int:
        """Rehydrate the rolling window from durable storage after a restart."""
        if self.sink is None:
            return 0
        cutoff = now - self.window
        restored = 0
        for issuer, rail, succeeded, at in self.sink.load_observations(cutoff):
            try:
                parsed_rail = PaymentRail(rail)
            except ValueError:
                parsed_rail = PaymentRail.UNKNOWN
            self._window_for(issuer, parsed_rail, create=True).append((at, succeeded))
            restored += 1
        return restored

    def prune(self, now: datetime) -> int:
        if self.sink is None:
            return 0
        return self.sink.prune_observations(now - self.window)

    # ------------------------------------------------------------------ core

    def observe(self, issuer: str, rail: PaymentRail, succeeded: bool, at: datetime) -> None:
        observations = self._window_for(issuer, rail, create=True)
        observations.append((at, succeeded))
        self._trim(observations, at)
        if self.sink is not None:
            self.sink.record_observation(issuer, rail.value, succeeded, at)

    def snapshot(self, issuer: str, rail: PaymentRail, now: datetime) -> HealthSnapshot:
        observations = self._window_for(issuer, rail, create=False)
        self._trim(observations, now)
        attempts = len(observations)
        successes = sum(1 for _, succeeded in observations if succeeded)
        success_rate = successes / attempts if attempts else 1.0
        degraded = attempts >= self.min_samples and success_rate < self.degraded_success_rate
        return HealthSnapshot(issuer.lower(), rail, attempts, successes, success_rate, degraded)

    def degraded_snapshots(self, now: datetime) -> list[HealthSnapshot]:
        results = []
        for issuer, rail in list(self._observations):
            snapshot = self.snapshot(issuer, rail, now)
            if snapshot.degraded:
                results.append(snapshot)
        return results

    # --------------------------------------------------------------- helpers

    def _window_for(
        self, issuer: str, rail: PaymentRail, *, create: bool
    ) -> deque[tuple[datetime, bool]]:
        key = (issuer.lower(), rail)
        observations = self._observations.get(key)
        if observations is None:
            if not create:
                # A read must never allocate. Returning a shared empty window
                # keeps `snapshot` free of side effects.
                return _EMPTY
            observations = deque()
            self._observations[key] = observations
            self._evict_if_needed()
        if key in self._observations:
            self._observations.move_to_end(key)
        return observations

    def _evict_if_needed(self) -> None:
        while len(self._observations) > self.max_tracked_keys:
            self._observations.popitem(last=False)

    def _trim(self, observations: deque[tuple[datetime, bool]], now: datetime) -> None:
        if not observations:
            return
        cutoff = now - self.window
        while observations and observations[0][0] < cutoff:
            observations.popleft()
