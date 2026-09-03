"""Razorpay edge adapter.

The fake client is the default for tests and demos. The real client requires
explicit credentials and is never selected automatically.

Failures are split into two kinds so the orchestrator can respond differently:
a transient transport error is worth another attempt, while a 4xx rejection is
a permanent decision that should go to manual review instead of being retried.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
import urllib.error
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.request import Request, urlopen
from uuid import uuid4

logger = logging.getLogger(__name__)

#: 409 is deliberately absent for creates: a conflict on a non-idempotent
#: create usually means the resource already exists, and retrying compounds it.
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}

#: Largest provider response we will read into memory.
MAX_RESPONSE_BYTES = 1_048_576


class GatewayError(RuntimeError):
    """Base class for payment-link adapter failures."""

    retryable = False


class TransientGatewayError(GatewayError):
    """Network or 5xx failure; another attempt may succeed."""

    retryable = True


class PermanentGatewayError(GatewayError):
    """The provider rejected the request; retrying will not help."""

    retryable = False


@dataclass(frozen=True)
class CreatedPaymentLink:
    id: str
    short_url: str
    status: str


class PaymentGateway(Protocol):
    def create_payment_link(
        self, *, amount_paise: int, reference_id: str, description: str
    ) -> CreatedPaymentLink: ...

    def cancel_payment_link(self, link_id: str) -> str: ...


class FakeRazorpayGateway:
    """In-memory stand-in used by tests, the benchmark and the dashboard demos."""

    def __init__(self) -> None:
        self.links: dict[str, str] = {}
        self.created = 0
        self.cancelled = 0

    def create_payment_link(
        self, *, amount_paise: int, reference_id: str, description: str
    ) -> CreatedPaymentLink:
        if amount_paise <= 0:
            raise PermanentGatewayError("payment link amount must be positive")
        link_id = f"plink_demo_{uuid4().hex[:12]}"
        self.links[link_id] = "issued"
        self.created += 1
        return CreatedPaymentLink(link_id, f"https://rzp.io/i/{link_id}", "issued")

    def cancel_payment_link(self, link_id: str) -> str:
        status = self.links.get(link_id)
        if status is None:
            raise PermanentGatewayError(f"unknown payment link: {link_id}")
        if status != "issued":
            return status
        self.links[link_id] = "cancelled"
        self.cancelled += 1
        return "cancelled"


class RazorpayPaymentLinkGateway:
    """Minimal real API client for explicit test/live configuration only."""

    api_base = "https://api.razorpay.com/v1"

    def __init__(
        self,
        key_id: str,
        key_secret: str,
        *,
        timeout_seconds: float = 10.0,
        max_attempts: int = 3,
        backoff_seconds: float = 0.5,
    ) -> None:
        self._auth = base64.b64encode(f"{key_id}:{key_secret}".encode()).decode()
        self._timeout = timeout_seconds
        self._max_attempts = max(1, max_attempts)
        self._backoff = backoff_seconds

    def create_payment_link(
        self, *, amount_paise: int, reference_id: str, description: str
    ) -> CreatedPaymentLink:
        body = json.dumps(
            {
                "amount": amount_paise,
                "currency": "INR",
                "reference_id": reference_id[:40],
                "description": description[:2048],
                "reminder_enable": False,
            }
        ).encode()
        # Creating a payment link is not idempotent, and _request retries on
        # 5xx and read timeouts. Without a stable key, a create that succeeded
        # at the provider but whose response was lost produces a SECOND live
        # link for the same invoice -- and only the second id comes back, so
        # the first is orphaned, uncancellable, and payable by the customer
        # after the original payment has already recovered. The key is derived
        # from the reference id so a retry of the same logical action reuses
        # it, while a genuinely new recovery for the same invoice does not.
        idempotency_key = hashlib.blake2b(
            f"payment_link:v1:{reference_id}:{amount_paise}".encode(), digest_size=16
        ).hexdigest()
        payload = self._request(
            "POST", "/payment_links", body, idempotency_key=idempotency_key
        )
        try:
            return CreatedPaymentLink(payload["id"], payload["short_url"], payload["status"])
        except KeyError as exc:  # pragma: no cover - provider contract drift
            raise PermanentGatewayError(f"payment link response missing {exc}") from exc
        except TypeError as exc:  # pragma: no cover - provider returned a non-object
            raise PermanentGatewayError("payment link response was not an object") from exc

    def cancel_payment_link(self, link_id: str) -> str:
        # Cancel is idempotent at the provider, so it needs no key.
        payload = self._request("POST", f"/payment_links/{link_id}/cancel", b"{}")
        status = payload.get("status")
        if not isinstance(status, str):  # pragma: no cover - provider contract drift
            raise PermanentGatewayError("cancel response did not include a status")
        return status

    def _request(
        self, method: str, path: str, body: bytes, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return self._attempt(method, path, body, idempotency_key)
            except TransientGatewayError as exc:
                last_error = exc
                if attempt == self._max_attempts:
                    break
                delay = self._backoff * (2 ** (attempt - 1))
                logger.warning(
                    "razorpay %s %s failed (attempt %s/%s), retrying in %.2fs: %s",
                    method,
                    path,
                    attempt,
                    self._max_attempts,
                    delay,
                    exc,
                )
                time.sleep(delay)
        raise TransientGatewayError(
            f"razorpay {method} {path} failed after {self._max_attempts} attempts: {last_error}"
        ) from last_error

    def _attempt(
        self, method: str, path: str, body: bytes, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Basic {self._auth}",
            "Content-Type": "application/json",
            "User-Agent": "RailPulse/0.2",
        }
        if idempotency_key:
            headers["X-Razorpay-Idempotency-Key"] = idempotency_key
        request = Request(f"{self.api_base}{path}", data=body, method=method, headers=headers)
        try:
            with urlopen(request, timeout=self._timeout) as response:  # nosec B310: fixed API base
                # Cap the read: an unbounded response.read() lets a misbehaving
                # or hostile endpoint exhaust memory.
                payload = json.loads(response.read(MAX_RESPONSE_BYTES))
            if not isinstance(payload, dict):
                raise PermanentGatewayError("provider returned a non-object body")
            return payload
        except urllib.error.HTTPError as exc:
            detail = self._error_detail(exc)
            if exc.code in RETRYABLE_STATUS:
                raise TransientGatewayError(f"HTTP {exc.code}: {detail}") from exc
            raise PermanentGatewayError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise TransientGatewayError(f"transport error: {exc.reason}") from exc
        except TimeoutError as exc:
            raise TransientGatewayError("request timed out") from exc
        except json.JSONDecodeError as exc:
            raise PermanentGatewayError("provider returned a non-JSON body") from exc

    @staticmethod
    def _error_detail(exc: urllib.error.HTTPError) -> str:
        try:
            payload = json.loads(exc.read())
        except Exception:  # pragma: no cover - best-effort diagnostics only
            return exc.reason or "unknown error"
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            return str(error.get("description") or error.get("code") or payload)
        return str(payload)
