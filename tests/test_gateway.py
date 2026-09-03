"""Tests for the live Razorpay adapter.

This was the least-covered file in the repo at 52%: every branch describing
what happens when the provider misbehaves was unexercised. It is also the only
module that can lose real money, so it is the last place that should be
running on hope. `urlopen` is monkeypatched, so nothing here touches a network.
"""

from __future__ import annotations

import io
import json
import unittest
import urllib.error
from unittest import mock

from app import gateway as mod
from app.gateway import (
    FakeRazorpayGateway,
    PermanentGatewayError,
    RazorpayPaymentLinkGateway,
    TransientGatewayError,
)


class _Response(io.BytesIO):
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _ok(payload: object) -> _Response:
    return _Response(json.dumps(payload).encode())


def _http_error(code: int, body: object = None) -> urllib.error.HTTPError:
    raw = json.dumps(body or {"error": {"description": "boom"}}).encode()
    return urllib.error.HTTPError("u", code, "err", {}, io.BytesIO(raw))


LINK = {"id": "plink_1", "short_url": "https://rzp.io/i/plink_1", "status": "created"}


class CreateLinkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gw = RazorpayPaymentLinkGateway("key", "secret", backoff_seconds=0)

    def test_happy_path_sends_auth_and_parses_the_link(self) -> None:
        calls = []

        def fake(request, timeout):
            calls.append(request)
            return _ok(LINK)

        with mock.patch.object(mod, "urlopen", fake):
            link = self.gw.create_payment_link(
                amount_paise=49900, reference_id="inv_1", description="Recovery"
            )
        self.assertEqual(link.id, "plink_1")
        self.assertEqual(link.short_url, "https://rzp.io/i/plink_1")
        request = calls[0]
        self.assertTrue(request.get_header("Authorization").startswith("Basic "))
        self.assertEqual(json.loads(request.data)["amount"], 49900)

    def test_create_carries_a_stable_idempotency_key(self) -> None:
        """The money-critical one.

        _request retries on 5xx and read timeouts. Without a stable key, a
        create that succeeded at the provider but whose response was lost
        produces a second live payment link for the same invoice, and only the
        second id is returned -- so the first is orphaned, uncancellable, and
        still payable by the customer.
        """
        seen: list[str | None] = []

        def fake(request, timeout):
            seen.append(request.get_header("X-razorpay-idempotency-key"))
            if len(seen) == 1:
                raise TimeoutError
            return _ok(LINK)

        with mock.patch.object(mod, "urlopen", fake):
            self.gw.create_payment_link(
                amount_paise=49900, reference_id="inv_1", description="d"
            )
        self.assertEqual(len(seen), 2, "expected the timed-out attempt to be retried")
        self.assertIsNotNone(seen[0])
        self.assertEqual(seen[0], seen[1], "retry must reuse the key, or it duplicates the link")

    def test_the_key_differs_per_invoice(self) -> None:
        seen: list[str | None] = []

        def fake(request, timeout):
            seen.append(request.get_header("X-razorpay-idempotency-key"))
            return _ok(LINK)

        with mock.patch.object(mod, "urlopen", fake):
            self.gw.create_payment_link(amount_paise=1, reference_id="inv_1", description="d")
            self.gw.create_payment_link(amount_paise=1, reference_id="inv_2", description="d")
        self.assertNotEqual(seen[0], seen[1])

    def test_cancel_carries_no_key(self) -> None:
        seen: list[str | None] = []

        def fake(request, timeout):
            seen.append(request.get_header("X-razorpay-idempotency-key"))
            return _ok({"status": "cancelled"})

        with mock.patch.object(mod, "urlopen", fake):
            self.assertEqual(self.gw.cancel_payment_link("plink_1"), "cancelled")
        self.assertIsNone(seen[0], "cancel is already idempotent at the provider")


class ErrorClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gw = RazorpayPaymentLinkGateway("key", "secret", max_attempts=3, backoff_seconds=0)

    def _create(self):
        return self.gw.create_payment_link(
            amount_paise=100, reference_id="inv_1", description="d"
        )

    def test_5xx_is_transient_and_retried_to_the_cap(self) -> None:
        attempts = []

        def fake(request, timeout):
            attempts.append(1)
            raise _http_error(503)

        with mock.patch.object(mod, "urlopen", fake), self.assertRaises(TransientGatewayError):
            self._create()
        self.assertEqual(len(attempts), 3)

    def test_4xx_is_permanent_and_not_retried(self) -> None:
        attempts = []

        def fake(request, timeout):
            attempts.append(1)
            raise _http_error(400)

        with mock.patch.object(mod, "urlopen", fake), self.assertRaises(PermanentGatewayError):
            self._create()
        self.assertEqual(len(attempts), 1, "a rejected request must not be retried")

    def test_409_is_permanent_for_a_create(self) -> None:
        """A conflict on a non-idempotent create usually means it already
        exists. Retrying compounds the problem rather than resolving it."""
        attempts = []

        def fake(request, timeout):
            attempts.append(1)
            raise _http_error(409)

        with mock.patch.object(mod, "urlopen", fake), self.assertRaises(PermanentGatewayError):
            self._create()
        self.assertEqual(len(attempts), 1)

    def test_transport_error_is_transient(self) -> None:
        def fake(request, timeout):
            raise urllib.error.URLError("connection refused")

        with mock.patch.object(mod, "urlopen", fake), self.assertRaises(TransientGatewayError):
            self._create()

    def test_recovers_after_a_transient_failure(self) -> None:
        state = {"n": 0}

        def fake(request, timeout):
            state["n"] += 1
            if state["n"] == 1:
                raise _http_error(502)
            return _ok(LINK)

        with mock.patch.object(mod, "urlopen", fake):
            self.assertEqual(self._create().id, "plink_1")

    def test_non_json_body_is_permanent(self) -> None:
        with mock.patch.object(mod, "urlopen", lambda r, timeout: _Response(b"<html>502</html>")), self.assertRaises(
            PermanentGatewayError
        ):
                self._create()

    def test_non_object_body_is_permanent_not_a_typeerror(self) -> None:
        """A JSON array used to reach payload["id"] and raise TypeError, which
        escaped the adapter's own error taxonomy entirely."""
        with mock.patch.object(mod, "urlopen", lambda r, timeout: _ok([1, 2, 3])), self.assertRaises(
            PermanentGatewayError
        ):
                self._create()

    def test_missing_fields_are_permanent(self) -> None:
        with mock.patch.object(mod, "urlopen", lambda r, timeout: _ok({"id": "plink_1"})), self.assertRaises(
            PermanentGatewayError
        ):
                self._create()

    def test_error_detail_survives_an_unparseable_error_body(self) -> None:
        def fake(request, timeout):
            raise urllib.error.HTTPError("u", 400, "Bad Request", {}, io.BytesIO(b"not json"))

        with mock.patch.object(mod, "urlopen", fake), self.assertRaises(
            PermanentGatewayError
        ) as caught:
            self._create()
        self.assertIn("400", str(caught.exception))


class FakeGatewayTests(unittest.TestCase):
    def test_zero_amount_is_rejected(self) -> None:
        """Guards the money bug where a case created from subscription.pending
        carried amount 0 and asked for a zero-value link."""
        with self.assertRaises(PermanentGatewayError):
            FakeRazorpayGateway().create_payment_link(
                amount_paise=0, reference_id="inv_1", description="d"
            )

    def test_cancelling_twice_is_idempotent(self) -> None:
        gw = FakeRazorpayGateway()
        link = gw.create_payment_link(amount_paise=100, reference_id="inv_1", description="d")
        self.assertEqual(gw.cancel_payment_link(link.id), "cancelled")
        self.assertEqual(gw.cancel_payment_link(link.id), "cancelled")
        self.assertEqual(gw.cancelled, 1, "a second cancel must not count as new work")

    def test_unknown_link_is_permanent(self) -> None:
        with self.assertRaises(PermanentGatewayError):
            FakeRazorpayGateway().cancel_payment_link("plink_nope")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
